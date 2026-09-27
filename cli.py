#!/usr/bin/env python3
"""Command-line entry point for the autonomous task agent.

Example usage:
    python cli.py --goal "Research AI agents and summarize the key ideas"
    python cli.py --goal "Screen data/sample_resume.txt against data/sample_job_description.txt" --mode react
    python cli.py --goal "..." --model llama3 --offline

Prints each reasoning step (Thought -> Action -> Observation) as it happens, then the
final answer and a token-usage cost report. Saves the full trace as JSON under logs/.
"""

from __future__ import annotations

import argparse

from agent.executor import run
from agent.llm import OllamaBackend, build_llm
from agent.memory import MemoryStore
from agent.tools import build_tools
from agent.trace import TraceLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autonomous task agent from the CLI.")
    parser.add_argument("--goal", required=True, help="High-level goal for the agent to accomplish.")
    parser.add_argument(
        "--mode", choices=["plan-execute", "react"], default="plan-execute",
        help="plan-execute (default): 1 planning call + deterministic execution + 1 synthesis call. "
        "react: full step-by-step ReAct loop, one LLM call per step.",
    )
    parser.add_argument("--model", default="llama3", help="Ollama model name (default: llama3).")
    parser.add_argument("--memory-file", default="memory_store.json", help="Path to the persistent memory JSON file.")
    parser.add_argument("--offline", action="store_true", help="Force the offline rule-based backend, skipping Ollama.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    llm = build_llm(model=args.model, force_offline=args.offline)
    if llm.name.startswith("offline"):
        print(
            f"[!] Using OFFLINE backend ({llm.name}) -- Ollama not reachable at "
            f"{OllamaBackend().host}. Start it with `ollama serve` and `ollama pull {args.model}` "
            "for real LLM reasoning.\n"
        )
    else:
        print(f"[i] Using backend: {llm.name}\n")

    memory = MemoryStore(path=args.memory_file)
    tools = build_tools(memory)
    trace = TraceLogger(goal=args.goal, mode=args.mode, backend_name=llm.name)

    print(f"GOAL: {args.goal}")
    print(f"MODE: {args.mode}\n{'-' * 60}")

    for step in run(args.goal, args.mode, llm, tools, memory, trace):
        print(step.render())
        print("-" * 60)

    print("\n=== FINAL ANSWER ===")
    print(trace.final_answer)

    usage = llm.usage.summary()
    print("\n=== COST REPORT ===")
    for key, value in usage.items():
        print(f"{key}: {value}")
    print(f"elapsed_seconds: {trace.elapsed_seconds}")

    log_path = trace.save(usage_summary=usage)
    print(f"\nFull trace saved to: {log_path}")


if __name__ == "__main__":
    main()
