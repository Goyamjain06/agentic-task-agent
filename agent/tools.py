"""The agent's tool belt.

Every tool has the same shape: a name, a one-line description (shown to the LLM so it
knows what the tool is for and how to call it), and a function that takes a single
string input and returns a single string observation. Keeping the interface this
narrow is what makes the planner/executor code tool-agnostic -- adding a fifth tool
never requires touching planner.py or executor.py.

Tools included (per the brief):
    web_search    - searches a small local mock corpus (swap in a real API key-based
                    client later without changing anything else)
    file_reader   - reads a text file from disk
    text_analyzer - calculator + text-analysis + resume/job-description match scoring
    memory        - get/set/append against the persistent MemoryStore

All tool code is deterministic, ordinary Python -- no LLM call happens inside a tool.
That's a deliberate cost/architecture choice, explained in the README.
"""

from __future__ import annotations

import ast
import json
import operator
import os
import re
from dataclasses import dataclass
from typing import Callable

from agent.memory import MemoryStore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(PROJECT_ROOT, "data", "mock_search_corpus.json")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "be", "as", "at", "by", "this", "that", "it", "we", "you", "our", "will", "your",
    "from", "has", "have", "using", "use", "such", "into", "their", "these", "those",
}


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

def _load_corpus() -> list[dict]:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["documents"]


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def web_search(query: str) -> str:
    """Mock search: ranks a small local corpus by keyword overlap with the query.

    Stands in for a real search API (e.g. SerpAPI, Bing, Tavily). Swapping this
    function's body for an HTTP call to a real provider is the only change needed
    to go from "offline demo" to "real web search" -- the rest of the agent doesn't
    know or care where the observation string came from.
    """
    query_kw = _keywords(query)
    if not query_kw:
        return "[MOCK SEARCH] No usable keywords in query."

    scored = []
    for doc in _load_corpus():
        doc_kw = _keywords(doc["title"] + " " + doc["snippet"])
        overlap = len(query_kw & doc_kw)
        if overlap:
            scored.append((overlap, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:2]

    if not top:
        return f"[MOCK SEARCH] No results found for '{query}'."

    lines = [f"[MOCK SEARCH] Top results for '{query}':"]
    for _, doc in top:
        lines.append(f"- {doc['title']} ({doc['url']}): {doc['snippet']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# file_reader
# ---------------------------------------------------------------------------

MAX_FILE_CHARS = 4000


def file_reader(path: str) -> str:
    """Reads a text file, relative to the project root unless an absolute path is given.

    Truncates long files rather than dumping them whole into the LLM's context --
    a small, concrete example of keeping tool output bounded so token cost stays
    predictable as the agent scales to bigger inputs.
    """
    path = path.strip()
    full_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    full_path = os.path.normpath(full_path)

    if not full_path.startswith(PROJECT_ROOT) and not os.path.isabs(path):
        return f"[FILE READER] Refused: path escapes project root: {path}"

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return f"[FILE READER] File not found: {path}"
    except OSError as exc:
        return f"[FILE READER] Error reading file: {exc}"

    if len(content) > MAX_FILE_CHARS:
        content = content[:MAX_FILE_CHARS] + "\n...[truncated]"
    return content


# ---------------------------------------------------------------------------
# text_analyzer (calculator + text stats + resume/JD match scoring)
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluates a restricted arithmetic AST (numbers + + - * / ** % ()).

    Deliberately not using Python's `eval()` on LLM-produced text: the input crosses
    a trust boundary (it was generated by a model, possibly steered by earlier tool
    output), so only a small, closed set of arithmetic operations is permitted.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Expression contains disallowed syntax.")


def _calculate(expr: str) -> str:
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree)
        return f"[CALCULATOR] {expr.strip()} = {result}"
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as exc:
        return f"[CALCULATOR] Could not evaluate '{expr}': {exc}"


def _text_stats(text: str) -> str:
    words = re.findall(r"[a-zA-Z']+", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    kw = _keywords(text)
    from collections import Counter

    top_keywords = [w for w, _ in Counter([w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w in kw]).most_common(5)]
    return (
        f"[TEXT STATS] words={len(words)}, sentences={len(sentences)}, "
        f"unique_keywords={len(kw)}, top_keywords={top_keywords}"
    )


def _match_score(resume_text: str, jd_text: str) -> str:
    """Keyword-overlap score between a resume and a job description.

    Simple by design (Jaccard-style overlap on non-stopword tokens): good enough to
    demo the "screen this resume against a job description" use case from the brief
    without pulling in embeddings/a vector DB, which would be a reasonable v2.
    """
    resume_kw = _keywords(resume_text)
    jd_kw = _keywords(jd_text)
    if not jd_kw:
        return "[TEXT ANALYZER] Job description had no usable keywords."

    matched = resume_kw & jd_kw
    missing = jd_kw - resume_kw
    score = round(100 * len(matched) / len(jd_kw), 1)
    return (
        f"[TEXT ANALYZER] Resume/JD match score: {score}% "
        f"({len(matched)}/{len(jd_kw)} JD keywords found). "
        f"Matched: {sorted(matched)[:15]}. Missing: {sorted(missing)[:15]}."
    )


def text_analyzer(input_str: str) -> str:
    """One tool, three modes, chosen by a prefix:

        calc: <expression>            -> arithmetic
        textstats: <text>             -> word/sentence/keyword counts
        match: <text a> || <text b>   -> resume-vs-job-description keyword overlap

    With no recognized prefix, falls back to a calculator attempt, then text stats.
    """
    s = input_str.strip()
    lower = s.lower()

    if lower.startswith("calc:"):
        return _calculate(s[len("calc:"):])
    if lower.startswith("textstats:"):
        return _text_stats(s[len("textstats:"):])
    if lower.startswith("match:"):
        body = s[len("match:"):]
        if "||" not in body:
            return "[TEXT ANALYZER] 'match:' input must contain '||' separating the two texts."
        a, b = body.split("||", 1)
        return _match_score(a, b)

    # No prefix: guess. Looks numeric-ish -> try calculator, else text stats.
    if re.fullmatch(r"[\d\s+\-*/().%]+", s):
        return _calculate(s)
    return _text_stats(s)


# ---------------------------------------------------------------------------
# memory tool
# ---------------------------------------------------------------------------

def make_memory_tool(memory: MemoryStore) -> Callable[[str], str]:
    """Binds the memory tool function to a specific MemoryStore instance.

    Input is a small JSON object, e.g.:
        {"op": "set", "key": "foo", "value": "bar"}
        {"op": "get", "key": "foo"}
        {"op": "append", "key": "history", "value": "did a thing"}
        {"op": "all"}
    """

    def memory_tool(input_str: str) -> str:
        try:
            # strict=False: tool inputs often embed raw multi-line tool output (e.g.
            # a search result) as a JSON string value, which contains literal newlines
            # that strict JSON parsing rejects. Plain text and control characters are
            # a normal thing to store in memory, so we relax that constraint here.
            request = json.loads(input_str, strict=False)
        except json.JSONDecodeError:
            # Still not JSON at all: be forgiving and treat it as a "get" on that key.
            request = {"op": "get", "key": input_str.strip()}

        op = request.get("op", "get")
        key = request.get("key")

        if op == "set":
            memory.set(key, request.get("value"))
            return f"[MEMORY] Stored key '{key}'."
        if op == "append":
            memory.append(key, request.get("value"))
            return f"[MEMORY] Appended to key '{key}'."
        if op == "get":
            value = memory.get(key)
            return f"[MEMORY] {key} = {json.dumps(value)}"
        if op == "all":
            return f"[MEMORY] All keys: {json.dumps(memory.all())}"
        return f"[MEMORY] Unknown operation: {op}"

    return memory_tool


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def build_tools(memory: MemoryStore) -> dict[str, Tool]:
    """Assembles the four tools the agent can call, bound to a shared MemoryStore."""
    tools = [
        Tool(
            name="web_search",
            description="Search for information on a topic. Input: a search query string. "
            "(Mock search over a small local corpus -- swap for a real API in production.)",
            func=web_search,
        ),
        Tool(
            name="file_reader",
            description="Read a text file from disk. Input: a file path (relative to the "
            "project root, e.g. 'data/sample_resume.txt').",
            func=file_reader,
        ),
        Tool(
            name="text_analyzer",
            description="Calculator and text analysis. Input: 'calc: <expr>' for arithmetic, "
            "'textstats: <text>' for word/keyword counts, or 'match: <resume> || <job description>' "
            "for a resume/job-description keyword match score.",
            func=text_analyzer,
        ),
        Tool(
            name="memory",
            description='Persist or recall data across steps and runs. Input: JSON like '
            '{"op":"set","key":"k","value":"v"}, {"op":"get","key":"k"}, '
            '{"op":"append","key":"k","value":"v"}, or {"op":"all"}.',
            func=make_memory_tool(memory),
        ),
    ]
    return {t.name: t for t in tools}


def tools_doc(tools: dict[str, Tool]) -> str:
    """Renders the tool registry as text for embedding in LLM prompts."""
    return "\n".join(f"- {t.name}: {t.description}" for t in tools.values())
