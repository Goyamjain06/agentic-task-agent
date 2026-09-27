"""Persistent JSON-backed memory.

This simulates the "long-term memory" piece of an agent architecture: a place the
agent can write facts to and read them back from, both within a single run (across
plan steps) and across separate runs of the program (the file survives on disk).

A real system would likely swap this for a vector store or a database, but the
interface (get/set/append) would stay the same -- that's the point of keeping it
behind a small class rather than scattering file I/O through the agent.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any


class MemoryStore:
    """A small, thread-safe key -> value store persisted as a single JSON file."""

    def __init__(self, path: str = "memory_store.json"):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                try:
                    self._data = json.load(f)
                except json.JSONDecodeError:
                    self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._save()

    def append(self, key: str, value: Any) -> None:
        """Append to a list stored under `key`, creating it if needed."""
        with self._lock:
            bucket = self._data.setdefault(key, [])
            if not isinstance(bucket, list):
                bucket = [bucket]
            bucket.append(value)
            self._data[key] = bucket
            self._save()

    def record_run(self, goal: str, mode: str, final_answer: str) -> None:
        """Append a summary of a completed run to a `run_history` list, timestamped."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "goal": goal,
            "mode": mode,
            "final_answer": final_answer,
        }
        self.append("run_history", entry)

    def all(self) -> dict:
        with self._lock:
            return dict(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._save()
