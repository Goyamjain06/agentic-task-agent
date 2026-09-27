"""Autonomous task agent package.

Modules:
    llm       - pluggable LLM backends (local Ollama, or an offline rule-based fallback)
    memory    - JSON-backed persistent memory store (also exposed as a tool)
    tools     - the agent's tool belt: web search, file reader, calculator/text-stats, memory
    planner   - turns a high-level goal into an ordered list of sub-steps
    executor  - runs the plan (two modes: cheap "plan-execute" and adaptive "react")
    trace     - records and prints the full thought -> action -> observation trace
"""
