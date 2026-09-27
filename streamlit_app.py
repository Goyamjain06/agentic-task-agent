"""Streamlit UI for the autonomous task agent.

Run with:
    streamlit run streamlit_app.py

Lets you type a goal, pick a mode (plan-execute vs react), and watch the agent's
reasoning trace (Thought -> Action -> Observation) render step by step as it runs,
followed by the final answer and a token-usage cost report.
"""

from __future__ import annotations

import json

import streamlit as st

from agent.executor import run
from agent.llm import OllamaBackend, build_llm
from agent.memory import MemoryStore
from agent.tools import build_tools
from agent.trace import TraceLogger

st.set_page_config(page_title="Autonomous Task Agent", page_icon="🤖", layout="centered")

EXAMPLE_GOALS = {
    "Research a topic": "Research AI agents and summarize the key ideas",
    "Screen a resume": "Screen data/sample_resume.txt against data/sample_job_description.txt",
}

st.title("🤖 Autonomous Task Agent")
st.caption("Planning loop + tool-calling + persistent memory, with a fully visible reasoning trace.")

# --- Sidebar: backend + mode configuration ----------------------------------

with st.sidebar:
    st.header("Configuration")

    ollama_up = OllamaBackend.is_reachable()
    if ollama_up:
        st.success("Ollama detected at localhost:11434")
    else:
        st.warning("Ollama not reachable -- will use the offline rule-based backend.")

    model = st.text_input("Ollama model", value="llama3")
    force_offline = st.checkbox("Force offline (rule-based) backend", value=not ollama_up)
    mode = st.radio(
        "Execution mode",
        options=["plan-execute", "react"],
        help=(
            "plan-execute: 1 planning call + deterministic tool execution + 1 synthesis call. Cheap.\n\n"
            "react: one LLM call per step, re-sending the whole scratchpad each time. Adaptive, more expensive."
        ),
    )
    memory_file = st.text_input("Memory file", value="memory_store.json")

    st.divider()
    st.subheader("Persistent memory")
    memory_preview = MemoryStore(path=memory_file)
    st.json(memory_preview.all(), expanded=False)
    if st.button("Clear memory"):
        memory_preview.clear()
        st.rerun()

# --- Main: goal input ---------------------------------------------------------

if "goal" not in st.session_state:
    st.session_state.goal = ""

cols = st.columns(len(EXAMPLE_GOALS))
for col, (label, example_goal) in zip(cols, EXAMPLE_GOALS.items()):
    if col.button(label):
        st.session_state.goal = example_goal

goal = st.text_area("Goal", key="goal", height=80)
run_clicked = st.button("Run agent", type="primary", disabled=not goal.strip())

# --- Run and stream the reasoning trace --------------------------------------

if run_clicked:
    llm = build_llm(model=model, force_offline=force_offline)
    if llm.name.startswith("offline"):
        st.info(f"Using offline backend ({llm.name}) -- reasoning is rule-based, not a real LLM.")
    else:
        st.info(f"Using backend: {llm.name}")

    memory = MemoryStore(path=memory_file)
    tools = build_tools(memory)
    trace = TraceLogger(goal=goal, mode=mode, backend_name=llm.name)

    with st.status("Agent is working...", expanded=True) as status:
        for step in run(goal, mode, llm, tools, memory, trace):
            st.markdown(f"**Step {step.index} -- Thought:** {step.thought}")
            if step.action:
                st.markdown(f"> **Action:** `{step.action}`  \n> **Action Input:** {step.action_input}")
            if step.observation is not None:
                st.code(step.observation, language=None)
        status.update(label="Done", state="complete")

    st.subheader("Final answer")
    st.write(trace.final_answer)

    usage = llm.usage.summary()
    st.subheader("Cost report")
    metric_cols = st.columns(4)
    metric_cols[0].metric("LLM calls", usage["llm_calls"])
    metric_cols[1].metric("Prompt tokens (est.)", usage["estimated_prompt_tokens"])
    metric_cols[2].metric("Completion tokens (est.)", usage["estimated_completion_tokens"])
    metric_cols[3].metric("Total tokens (est.)", usage["estimated_total_tokens"])

    log_path = trace.save(usage_summary=usage)
    st.caption(f"Full trace saved to `{log_path}`")
    st.download_button(
        "Download trace JSON",
        data=json.dumps(trace.to_dict(usage), indent=2),
        file_name="trace.json",
        mime="application/json",
    )
