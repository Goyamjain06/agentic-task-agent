"""Planning step: turn a high-level goal into an ordered list of sub-steps.

This is the one LLM call that "plan-execute" mode spends on deciding *what* to do.
Every subsequent step is executed deterministically by executor.py without calling
the LLM again -- that's the core cost-saving idea explained in the README.

The plan step can also reference the output of an earlier step with the placeholder
``<STEP<i>_OUTPUT>``, e.g. a "match" step referencing the resume and job-description
text read by steps 1 and 2. The executor substitutes these before calling the tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.llm import TrackedLLM
from agent.tools import Tool, tools_doc

PLACEHOLDER_RE = re.compile(r"<STEP(\d+)_OUTPUT>")


@dataclass
class PlanStep:
    index: int
    description: str
    tool: str
    tool_input: str


def build_planning_prompt(goal: str, tools: dict[str, Tool]) -> str:
    return f"""TASK_TYPE: PLAN
GOAL: {goal}

You are an autonomous agent's planning module. Break the goal above into a short,
ordered list of concrete sub-steps. Each step must use exactly one of these tools:

{tools_doc(tools)}

You may reference the output of an earlier step with the placeholder <STEPi_OUTPUT>
(e.g. <STEP1_OUTPUT>) inside a later step's INPUT.

Respond ONLY in this exact format, one block per step, no extra commentary:

STEP 1: <short description of what this step accomplishes>
TOOL: <tool name>
INPUT: <the exact input to pass to the tool>
STEP 2: ...
"""


def parse_plan(raw_output: str) -> list[PlanStep]:
    """Parses the STEP/TOOL/INPUT block format produced by build_planning_prompt."""
    pattern = re.compile(
        r"STEP\s+(\d+):\s*(.+?)\s*\n\s*TOOL:\s*(\S+)\s*\n\s*INPUT:\s*(.+?)(?=\n\s*STEP\s+\d+:|\Z)",
        re.DOTALL,
    )
    steps = []
    for match in pattern.finditer(raw_output):
        idx, desc, tool, tool_input = match.groups()
        steps.append(PlanStep(index=int(idx), description=desc.strip(), tool=tool.strip(),
                               tool_input=tool_input.strip()))
    return steps


def plan(goal: str, llm: TrackedLLM, tools: dict[str, Tool]) -> list[PlanStep]:
    """Calls the LLM once to decompose `goal`, with a safe fallback if parsing fails."""
    prompt = build_planning_prompt(goal, tools)
    raw = llm.generate(prompt, task_type="plan")
    steps = parse_plan(raw)

    if not steps:
        # Parsing the model's output failed (e.g. it didn't follow the format).
        # Fall back to a single, safe web_search step rather than crashing the run.
        steps = [PlanStep(index=1, description=f"Search for information on: {goal}",
                           tool="web_search", tool_input=goal)]
    return steps


def resolve_placeholders(text: str, step_outputs: dict[int, str]) -> str:
    """Replaces <STEPi_OUTPUT> in `text` with the recorded observation from step i."""

    def _sub(match: re.Match) -> str:
        i = int(match.group(1))
        return step_outputs.get(i, match.group(0))

    return PLACEHOLDER_RE.sub(_sub, text)
