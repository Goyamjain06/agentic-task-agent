# Autonomous Task Agent

A small, dependency-light autonomous agent: give it a goal, it plans sub-steps,
calls tools to gather information, remembers things across steps and across runs,
and shows its entire reasoning trace as it works.

Built for a job application to an agentic-AI team, so the priority is a *clear,
correct, honestly-explained* architecture over feature count.

```
python cli.py --goal "Research AI agents and summarize the key ideas"
```

or

```
streamlit run streamlit_app.py
```

Both run with **zero setup** — see [Setup](#setup) for why.

---

## What it does

1. **Takes a high-level goal** — e.g. *"research a topic and summarize findings"*
   or *"screen this resume against a job description"*.
2. **Plans autonomously** — one call to an LLM decomposes the goal into an ordered
   list of concrete sub-steps, each tagged with the tool that should handle it.
3. **Calls tools** — four of them: `web_search`, `file_reader`, `text_analyzer`
   (calculator + text stats + resume/job-description match scoring), and `memory`
   (persisted to a JSON file, so it survives across runs).
4. **Logs a full reasoning trace** — every step is recorded as
   `Thought -> Action -> Action Input -> Observation` and printed as it happens,
   plus saved to `logs/*.json` for later inspection.
5. **Ships a CLI and a Streamlit UI**, both showing the trace unfold step by step.

---

## Architecture

```
        goal
         |
         v
   +-----------+        1 LLM call        +-------------------------+
   |  planner  |  ------------------->    | ordered list of         |
   +-----------+                          | PlanStep(desc, tool,    |
                                           | tool_input)             |
                                           +-------------------------+
                                                      |
                                                      v
   +-------------------------------------------------------------------+
   |  executor  (deterministic loop, no LLM call per step)              |
   |  for each step: resolve <STEPi_OUTPUT> placeholders -> call tool   |
   |  -> record Thought/Action/Observation in TraceLogger               |
   +-------------------------------------------------------------------+
                                                      |
                                                      v
                                           +-------------------------+
                                           |  memory.json             |  <- tools also
                                           |  (persists across runs)  |     read/write
                                           +-------------------------+     this directly
                                                      |
                                              1 LLM call (synthesis)
                                                      v
                                              final answer + cost report
```

That's **plan-execute** mode, the default. There's a second mode, **react**, which
implements the classic [ReAct](https://arxiv.org/abs/2210.03629) loop instead: the
LLM is called *every* step, sees the full scratchpad of everything so far, and
decides the next `Thought` + `Action` (or a `Final Answer`) itself. It's more
adaptive — it can react to a surprising tool result and change course — but it
costs more, which is the whole point of having both (see below).

Both modes are generators (`executor.run_plan_execute` / `executor.run_react`)
that `yield` a `Step` object every time one happens, so the CLI and the Streamlit
app render the trace live instead of waiting for the whole run to finish, and
neither one needs to know which mode produced the steps it's rendering.

### The pieces

| Module | Responsibility |
|---|---|
| `agent/llm.py` | Pluggable LLM backends behind one `generate(prompt) -> str` method: `OllamaBackend` (local `llama3` via HTTP) and `RuleBasedBackend` (offline, deterministic, zero dependencies). `TrackedLLM` wraps either one and counts tokens on every call. |
| `agent/planner.py` | Builds the planning prompt, parses the model's `STEP/TOOL/INPUT` response into `PlanStep` objects, and substitutes `<STEPi_OUTPUT>` placeholders so later steps can reference earlier results. |
| `agent/tools.py` | The four tools, plus the registry (`build_tools`) and the text block (`tools_doc`) that gets embedded in every prompt so the model knows what's available. |
| `agent/memory.py` | `MemoryStore` — a tiny thread-safe JSON key-value store. Used both as the `memory` tool (so the *agent* can read/write it mid-run) and internally to log every run's goal/mode/answer to `run_history`. |
| `agent/executor.py` | The two execution modes described above, plus the ReAct prompt/parsing and the synthesis prompt. |
| `agent/trace.py` | `TraceLogger` / `Step` — records, renders, and saves the reasoning trace as JSON. |
| `cli.py` / `streamlit_app.py` | Thin front ends. All the logic above is UI-agnostic. |

### Why plan-once instead of decide-every-step?

Because most of what an agent does per step — "call this tool with this input" —
doesn't require creative reasoning once the plan exists. Pushing that decision
into cheap, deterministic Python (`executor.run_plan_execute`) instead of an LLM
call is the single biggest lever on cost in this codebase. `react` mode exists
specifically so you can *measure* that difference yourself rather than take it on
faith (see next section).

### Memory design

`MemoryStore` is a plain dict persisted to `memory_store.json` after every write.
It plays two roles:

- **Within a run**: the `memory` tool lets a plan or a react loop explicitly
  store/recall intermediate findings (e.g. "save the resume match score").
- **Across runs**: every completed run appends `{timestamp, goal, mode,
  final_answer}` to a `run_history` list, so a later run — or a human — can see
  what the agent has already looked into. A real system would likely swap the
  flat JSON file for a vector store or database, but the `get/set/append`
  interface would stay the same, since nothing outside `memory.py` touches the
  file directly.

---

## Why this can be cheaper than always calling a large model

Every LLM call costs tokens in both directions: the prompt you send it, and the
completion it generates. A ReAct-style agent that calls the LLM at every single
step has to resend the **entire scratchpad so far** each time, so token cost
grows with every step, not just linearly in the number of steps but in the total
size of history repeated across calls. A plan-once agent pays for the full
context exactly twice: once to plan, once to synthesize.

This isn't just an assertion — the CLI prints a **Cost Report** after every run
(LLM calls, estimated prompt/completion/total tokens), and every trace saved to
`logs/*.json` includes it too. Two real runs of the *same goal*, offline backend,
measured by this repo's own tracker:

```
python cli.py --goal "Research AI agents and summarize the key ideas" --mode plan-execute
python cli.py --goal "Research AI agents and summarize the key ideas" --mode react
```

| Mode | LLM calls | Est. total tokens |
|---|---|---|
| `plan-execute` | 2 (flat, regardless of plan length) | ~970 |
| `react` | 3 (grows with steps taken) | ~1,650 |

That's already a ~40% reduction for a 2-step task. The gap widens with plan
length: `plan-execute` stays at exactly 2 LLM calls no matter how many tool
steps the plan has (the tool calls themselves are free — ordinary Python), while
`react` adds one full LLM call *per step*, and each of those calls re-sends
every prior Thought/Action/Observation. For a 10-step task, a back-of-envelope
estimate (~150 tokens of scratchpad added per step) puts naive per-step ReAct at
roughly 10x the prompt tokens of a single synthesis call over the same
observations — because it pays for that growing history ten times over, not once.

The trade-off is real, not free: `react` can adapt mid-run (e.g. try a different
search query if the first one comes back empty, or bail out early if it already
has the answer), while `plan-execute` commits to a plan upfront and only
resolves it. This repo implements both so that trade-off is something you can
see and measure, not just take on faith — which is the argument an
agentic-AI team actually cares about.

---

## Setup

### Zero-setup (offline) mode

The agent auto-detects whether a local Ollama server is running. If it isn't,
it falls back to `RuleBasedBackend` — a deterministic, dependency-free stand-in
for the LLM that still exercises the full pipeline (planning format, tool
calls, memory, trace, cost tracking) so the project is gradeable/demoable with
nothing installed beyond Python 3.10+ and (for the UI) Streamlit:

```bash
pip install -r requirements.txt   # only needed for the Streamlit UI
python cli.py --goal "Research AI agents and summarize the key ideas"
```

The CLI prints a clear `[!] Using OFFLINE backend` banner whenever this fallback
is active, so it's never silently pretending to be a real LLM.

### Real LLM mode (Ollama + llama3)

```bash
brew install ollama          # or see https://ollama.com
ollama serve                 # start the local server
ollama pull llama3           # pull the model once
python cli.py --goal "Research AI agents and summarize the key ideas"
```

`build_llm()` in `agent/llm.py` will detect the running server automatically —
no flags needed. Force one backend or the other explicitly with `--offline` or
`--model <name>`.

### Streamlit UI

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Pick a mode in the sidebar, click one of the two example-goal buttons (or type
your own), hit **Run agent**, and watch each Thought/Action/Observation render
as it happens, followed by the final answer and the cost report. Persistent
memory contents are visible (and clearable) in the sidebar.

### Running the examples from the brief

```bash
# "research a topic and summarize findings"
python cli.py --goal "Research AI agents and summarize the key ideas"

# "screen this resume against a job description"
python cli.py --goal "Screen data/sample_resume.txt against data/sample_job_description.txt"
```

Sample data lives in `data/`: a resume, a matching job description, and a small
mock search corpus (`mock_search_corpus.json`) that stands in for a real search
API — swap `web_search()` in `agent/tools.py` for an HTTP call to a real
provider and nothing else in the pipeline needs to change.

---

## Testing & debugging

Every code path was run end-to-end, not just read over: both modes on both
example goals (research + resume screening), the Streamlit UI smoke-tested
headless, and the memory file / trace JSON inspected after each run. That
caught two real bugs before they shipped:

- **Memory tool JSON parsing** — a `memory` step storing raw multi-line tool
  output (e.g. a search result) as a JSON string value failed under strict
  `json.loads`, since literal newlines inside a JSON string aren't valid per
  spec. It silently fell back to a `get` and returned `null` instead of
  storing anything. Fixed with `json.loads(..., strict=False)` in
  `agent/tools.py`, which allows literal control characters in string values.
- **React-mode step counter mismatch** — the offline backend's react loop
  decided how far along it was by counting `Observation:` lines in the
  scratchpad, but the executor actually writes numbered labels
  (`Observation 1:`, `Observation 2:`, ...). The count never advanced, so the
  same search step repeated until the step limit instead of progressing to a
  final answer. Fixed the regex in `agent/llm.py` to match `Observation \d+:`.

Both were caught by actually reading the printed trace and the saved
`memory_store.json` / `logs/*.json`, not by inspection — the auditable trace
this agent produces is what made them visible in the first place.

---

## Project layout

```
agent/
  llm.py        LLM backends (Ollama + offline fallback) and token-usage tracking
  memory.py     Persistent JSON key-value store
  tools.py      web_search, file_reader, text_analyzer, memory tool
  planner.py    Goal -> ordered plan (1 LLM call)
  executor.py   plan-execute and react execution modes
  trace.py      Reasoning-trace recording, rendering, and JSON export
cli.py          Command-line entry point
streamlit_app.py Streamlit UI
data/           Sample resume, job description, mock search corpus
logs/           Saved trace JSON files (one per run)
```
