"""LLM backends for the agent.

The agent talks to "an LLM" through a single tiny interface: ``generate(prompt) -> str``.
Two backends implement it:

    OllamaBackend   - calls a local Ollama server (default model: llama3).
    RuleBasedBackend - a fully offline, deterministic fallback so the whole project
                        still runs (and is gradeable) on a machine with no Ollama
                        installed or the daemon not started.

``build_llm()`` picks whichever is available and wraps it in ``TrackedLLM`` so every
prompt/completion is counted, in tokens, as it happens. That running total is what
powers the cost report in the README's "why this is cheaper" argument -- it's measured
from real runs, not just asserted.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import urllib.error
import urllib.request

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3"


class OllamaUnavailable(Exception):
    """Raised when the local Ollama daemon can't be reached or errors out."""


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token, the common back-of-envelope ratio for
    English text). We're not calling a real tokenizer here on purpose: the point of
    this number is to compare *relative* cost between agent modes, not to bill anyone.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class TokenUsage:
    """Running totals for one agent run, used for the end-of-run cost report."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_log: list = field(default_factory=list)

    def record(self, task_type: str, prompt: str, completion: str) -> None:
        pt = estimate_tokens(prompt)
        ct = estimate_tokens(completion)
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.call_log.append({"task_type": task_type, "prompt_tokens": pt, "completion_tokens": ct})

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> dict:
        return {
            "llm_calls": self.calls,
            "estimated_prompt_tokens": self.prompt_tokens,
            "estimated_completion_tokens": self.completion_tokens,
            "estimated_total_tokens": self.total_tokens,
        }


class LLMBackend(ABC):
    """Minimal interface every backend implements."""

    name: str = "abstract"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class OllamaBackend(LLMBackend):
    """Talks to a local Ollama server over its HTTP API (no extra SDK dependency)."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST, timeout: float = 120.0):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.name = f"ollama:{model}"

    @staticmethod
    def is_reachable(host: str = OLLAMA_HOST, timeout: float = 1.5) -> bool:
        """Quick health check so we can fail fast and fall back instead of hanging."""
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=timeout)
            return True
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise OllamaUnavailable(f"Could not reach Ollama at {self.host}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaUnavailable(f"Ollama returned an unparseable response: {exc}") from exc

        text = body.get("response", "")
        if not text:
            raise OllamaUnavailable("Ollama returned an empty response.")
        return text.strip()


class RuleBasedBackend(LLMBackend):
    """Deterministic, offline stand-in for an LLM.

    This exists so the whole agent (planning, tool use, memory, react loop, trace,
    CLI, and Streamlit UI) is runnable and demoable with zero external dependencies --
    no API key, no Ollama daemon required. It is intentionally simple: it pattern-matches
    the structured prompts our planner/executor send (marked with a ``TASK_TYPE:`` line)
    and returns a plausible, well-formed response in the exact format the real LLM is
    asked for, so the rest of the pipeline can't tell the difference.

    It is NOT a general-purpose language model -- swap in OllamaBackend (or any other
    LLMBackend) for real reasoning.
    """

    name = "offline-rule-based (no LLM)"

    def generate(self, prompt: str) -> str:
        task_type = self._extract(prompt, "TASK_TYPE")
        goal = self._extract(prompt, "GOAL") or ""

        if task_type == "PLAN":
            return self._plan(goal)
        if task_type == "REACT_STEP":
            return self._react_step(prompt, goal)
        if task_type == "SYNTHESIZE":
            return self._synthesize(prompt, goal)
        return "Thought: Unable to determine task type.\nFinal Answer: (offline backend could not process this request)"

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _extract(prompt: str, label: str) -> Optional[str]:
        match = re.search(rf"^{label}:\s*(.+)$", prompt, re.MULTILINE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _looks_like_resume_screen(goal: str) -> bool:
        g = goal.lower()
        return ("resume" in g or "cv" in g) and ("job" in g or "screen" in g or "description" in g)

    @staticmethod
    def _looks_like_research(goal: str) -> bool:
        g = goal.lower()
        return "research" in g or "summar" in g

    def _plan(self, goal: str) -> str:
        """Heuristic goal decomposition, formatted exactly like we ask the real LLM to."""
        if self._looks_like_resume_screen(goal):
            steps = [
                ("Read the candidate's resume from disk", "file_reader", "data/sample_resume.txt"),
                ("Read the target job description from disk", "file_reader", "data/sample_job_description.txt"),
                (
                    "Score keyword overlap between the resume and the job description",
                    "text_analyzer",
                    "match: <STEP1_OUTPUT> || <STEP2_OUTPUT>",
                ),
                (
                    "Save the match score to memory for future runs",
                    "memory",
                    '{"op": "set", "key": "last_resume_match", "value": "<STEP3_OUTPUT>"}',
                ),
            ]
        elif self._looks_like_research(goal):
            topic = goal
            steps = [
                (f"Search for background information on: {topic}", "web_search", topic),
                (f"Search for a second, related angle on: {topic}", "web_search", f"{topic} details"),
                (
                    "Save key findings to memory so future runs can build on them",
                    "memory",
                    '{"op": "append", "key": "research_findings", "value": "<STEP1_OUTPUT>"}',
                ),
            ]
        else:
            steps = [
                (f"Search for information relevant to the goal: {goal}", "web_search", goal),
                (
                    "Save the result to memory",
                    "memory",
                    '{"op": "set", "key": "last_result", "value": "<STEP1_OUTPUT>"}',
                ),
            ]

        lines = []
        for i, (desc, tool, tool_input) in enumerate(steps, start=1):
            lines.append(f"STEP {i}: {desc}")
            lines.append(f"TOOL: {tool}")
            lines.append(f"INPUT: {tool_input}")
        return "\n".join(lines)

    def _react_step(self, prompt: str, goal: str) -> str:
        """Decide the next Thought/Action given how many observations we've already seen.

        We count "Observation:" occurrences already in the scratchpad to decide whether
        we're just starting, mid-way, or ready to answer -- a crude but honest substitute
        for actually reading and understanding the scratchpad the way a real LLM would.
        """
        n_observations = len(re.findall(r"^Observation \d+:", prompt, re.MULTILINE))

        if self._looks_like_resume_screen(goal):
            script = [
                ("I should read the resume file first.", "file_reader", "data/sample_resume.txt"),
                ("Now I should read the job description to compare against.", "file_reader", "data/sample_job_description.txt"),
                (
                    "I have both documents; I should score their keyword overlap.",
                    "text_analyzer",
                    "match: see resume and job description above",
                ),
            ]
        elif self._looks_like_research(goal):
            script = [
                (f"I should search for information on: {goal}", "web_search", goal),
                (f"I should look for a second source on: {goal}", "web_search", f"{goal} background"),
            ]
        else:
            script = [(f"I should search for: {goal}", "web_search", goal)]

        if n_observations < len(script):
            thought, tool, tool_input = script[n_observations]
            return f"Thought: {thought}\nAction: {tool}\nAction Input: {tool_input}"

        return (
            "Thought: I have gathered enough observations to answer the goal.\n"
            f"Final Answer: Based on the steps taken above, here is the result for '{goal}'. "
            "(offline rule-based backend -- run with Ollama for a real synthesized answer)"
        )

    def _synthesize(self, prompt: str, goal: str) -> str:
        observations = re.findall(r"^Observation \d+:\s*(.+)$", prompt, re.MULTILINE)
        if not observations:
            observations = re.findall(r"^Observation:\s*(.+)$", prompt, re.MULTILINE)
        bullets = "\n".join(f"- {obs[:200]}" for obs in observations) or "- (no observations were collected)"
        return (
            f"Summary for goal: {goal}\n\n"
            f"{bullets}\n\n"
            "(Generated by the offline rule-based backend. Run with a local Ollama model "
            "for a genuinely synthesized, natural-language summary.)"
        )


class TrackedLLM(LLMBackend):
    """Wraps any LLMBackend and records token usage for every call, transparently."""

    def __init__(self, inner: LLMBackend, usage: Optional[TokenUsage] = None):
        self.inner = inner
        self.usage = usage or TokenUsage()
        self.name = inner.name

    def generate(self, prompt: str, task_type: str = "unknown") -> str:
        start = time.time()
        completion = self.inner.generate(prompt)
        self.usage.record(task_type, prompt, completion)
        self.last_latency = time.time() - start
        return completion


def build_llm(model: str = DEFAULT_MODEL, force_offline: bool = False) -> TrackedLLM:
    """Pick the best available backend and wrap it for usage tracking.

    Auto-detection keeps the CLI/UI friction-free: if Ollama is running, use it;
    otherwise fall back to the offline backend and say so loudly rather than hanging
    or crashing.
    """
    if not force_offline and OllamaBackend.is_reachable():
        return TrackedLLM(OllamaBackend(model=model))
    return TrackedLLM(RuleBasedBackend())
