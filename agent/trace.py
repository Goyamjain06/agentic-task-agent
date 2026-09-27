"""Reasoning-trace logging.

The brief asks for the agent's full thought -> action -> observation -> next thought
process to be visible and auditable. `TraceLogger` is the single place that happens:
executor.py yields step dicts, TraceLogger keeps them in order, can print them (CLI),
hand them to Streamlit one at a time, and dump the whole run to a JSON file in logs/
for later inspection -- e.g. to compare a plan-execute run against a react run.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


@dataclass
class Step:
    index: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }

    def render(self) -> str:
        lines = [f"[Step {self.index}] Thought: {self.thought}"]
        if self.action:
            lines.append(f"           Action: {self.action}")
            lines.append(f"           Action Input: {self.action_input}")
        if self.observation is not None:
            lines.append(f"           Observation: {self.observation}")
        return "\n".join(lines)


class TraceLogger:
    """Collects Steps for one run and can render/save them."""

    def __init__(self, goal: str, mode: str, backend_name: str):
        self.goal = goal
        self.mode = mode
        self.backend_name = backend_name
        self.steps: list[Step] = []
        self.final_answer: Optional[str] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None

    def log_step(self, thought: str, action: str | None = None, action_input: str | None = None,
                 observation: str | None = None) -> Step:
        step = Step(index=len(self.steps) + 1, thought=thought, action=action,
                    action_input=action_input, observation=observation)
        self.steps.append(step)
        return step

    def finish(self, final_answer: str) -> None:
        self.final_answer = final_answer
        self.finished_at = time.time()

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.started_at, 3)

    def to_dict(self, usage_summary: dict | None = None) -> dict:
        return {
            "goal": self.goal,
            "mode": self.mode,
            "backend": self.backend_name,
            "elapsed_seconds": self.elapsed_seconds,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "token_usage": usage_summary,
        }

    def save(self, usage_summary: dict | None = None) -> str:
        os.makedirs(LOGS_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(LOGS_DIR, f"trace_{self.mode}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(usage_summary), f, indent=2, ensure_ascii=False)
        return path
