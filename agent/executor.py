"""Executes a goal in one of two modes.

plan-execute (default, cheap):
    1 LLM call to produce a plan (planner.plan), then every step's tool is invoked
    deterministically (no LLM call per step), then 1 LLM call to synthesize the
    final answer from all observations. Total LLM calls: 2, regardless of plan length.

react (adaptive, more expensive):
    The classic ReAct loop: one LLM call per step, each time re-sending the full
    scratchpad (every prior Thought/Action/Observation) so the model can react to
    what it just learned and decide the next action -- or stop with a Final Answer.
    Total LLM calls: up to max_steps, and each call's prompt is bigger than the last
    because the scratchpad keeps growing.

Both modes produce the same kind of trace (Step objects with thought/action/
action_input/observation) so the CLI and Streamlit UI don't need to know which mode
produced them. Both are implemented as generators so callers can render each step as
it happens instead of waiting for the whole run to finish.
"""

from __future__ import annotations

import re
from typing import Generator, Optional

from agent.llm import TrackedLLM
from agent.memory import MemoryStore
from agent.planner import plan, resolve_placeholders
from agent.tools import Tool, tools_doc
from agent.trace import Step, TraceLogger

MAX_REACT_STEPS = 6


# ---------------------------------------------------------------------------
# plan-execute mode
# ---------------------------------------------------------------------------

def build_synthesis_prompt(goal: str, observations: dict[int, str]) -> str:
    obs_block = "\n".join(f"Observation {i}: {obs}" for i, obs in sorted(observations.items()))
    return f"""TASK_TYPE: SYNTHESIZE
GOAL: {goal}

Observations gathered while working on the goal:
{obs_block}

Write a concise, well-organized final answer to the goal using only the observations
above. Do not invent facts that aren't supported by them.
"""


def run_plan_execute(
    goal: str, llm: TrackedLLM, tools: dict[str, Tool], memory: MemoryStore, trace: TraceLogger
) -> Generator[Step, None, None]:
    steps = plan(goal, llm, tools)

    plan_summary = "\n".join(f"{s.index}. {s.description} [tool={s.tool}]" for s in steps)
    yield trace.log_step(
        thought=f"Planned {len(steps)} step(s) for the goal (1 LLM call).",
        observation=plan_summary,
    )

    step_outputs: dict[int, str] = {}
    for planned_step in steps:
        resolved_input = resolve_placeholders(planned_step.tool_input, step_outputs)
        tool = tools.get(planned_step.tool)
        if tool is None:
            observation = f"[EXECUTOR] Unknown tool '{planned_step.tool}', skipped this step."
        else:
            observation = tool.func(resolved_input)
        step_outputs[planned_step.index] = observation

        yield trace.log_step(
            thought=planned_step.description,
            action=planned_step.tool,
            action_input=resolved_input,
            observation=observation,
        )

    synth_prompt = build_synthesis_prompt(goal, step_outputs)
    final_answer = llm.generate(synth_prompt, task_type="synthesize")

    memory.record_run(goal, "plan-execute", final_answer)
    trace.finish(final_answer)


# ---------------------------------------------------------------------------
# react mode
# ---------------------------------------------------------------------------

_REACT_PATTERN = re.compile(
    r"Thought:\s*(?P<thought>.*?)\s*\n\s*"
    r"(?:Action:\s*(?P<action>\S+)\s*\n\s*Action Input:\s*(?P<action_input>.*?)"
    r"|Final Answer:\s*(?P<final_answer>.*))\s*\Z",
    re.DOTALL,
)


def build_react_prompt(goal: str, tools: dict[str, Tool], scratchpad: str) -> str:
    return f"""TASK_TYPE: REACT_STEP
GOAL: {goal}

Available tools:
{tools_doc(tools)}

You are solving the goal step by step. Progress so far:
{scratchpad or "(nothing yet -- this is the first step)"}

Respond with EITHER of these two formats, and nothing else:

Thought: <your reasoning about what to do next>
Action: <one tool name>
Action Input: <the input to give that tool>

Thought: <your reasoning that you now have enough information>
Final Answer: <the final answer to the goal>
"""


def parse_react_output(raw: str) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Returns (thought, action, action_input, final_answer). Exactly one of
    (action, final_answer) will be set when parsing succeeds."""
    match = _REACT_PATTERN.search(raw.strip())
    if not match:
        # Model didn't follow the format -- treat the whole thing as a final answer
        # rather than looping forever on unparseable output.
        return ("Could not parse a structured next step; treating output as final.", None, None, raw.strip())

    groups = match.groupdict()
    return (groups["thought"].strip(), groups.get("action"),
            groups.get("action_input").strip() if groups.get("action_input") else None,
            groups.get("final_answer").strip() if groups.get("final_answer") else None)


def run_react(
    goal: str,
    llm: TrackedLLM,
    tools: dict[str, Tool],
    memory: MemoryStore,
    trace: TraceLogger,
    max_steps: int = MAX_REACT_STEPS,
) -> Generator[Step, None, None]:
    scratchpad = ""

    for i in range(1, max_steps + 1):
        prompt = build_react_prompt(goal, tools, scratchpad)
        raw = llm.generate(prompt, task_type="react_step")
        thought, action, action_input, final_answer = parse_react_output(raw)

        if final_answer is not None:
            yield trace.log_step(thought=thought)
            memory.record_run(goal, "react", final_answer)
            trace.finish(final_answer)
            return

        tool = tools.get(action or "")
        if tool is None:
            observation = f"[EXECUTOR] Unknown tool '{action}'."
        else:
            observation = tool.func(action_input or "")

        yield trace.log_step(thought=thought, action=action, action_input=action_input, observation=observation)

        # Growing the scratchpad every iteration is exactly what makes react mode
        # more expensive in tokens as the run gets longer -- see README cost comparison.
        scratchpad += (
            f"\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\n"
            f"Observation {i}: {observation}\n"
        )

    fallback = (
        f"Reached the {max_steps}-step limit without a Final Answer. "
        "Best-effort summary of the scratchpad:\n" + scratchpad
    )
    memory.record_run(goal, "react", fallback)
    trace.finish(fallback)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def run(
    goal: str, mode: str, llm: TrackedLLM, tools: dict[str, Tool], memory: MemoryStore, trace: TraceLogger
) -> Generator[Step, None, None]:
    if mode == "react":
        yield from run_react(goal, llm, tools, memory, trace)
    elif mode == "plan-execute":
        yield from run_plan_execute(goal, llm, tools, memory, trace)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'plan-execute' or 'react')")
