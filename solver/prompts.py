"""
Category-aware system prompts.

Each category gets a short, specific prompt rather than one generic prompt,
because strategy genuinely differs per category (TEAM_PLAN.md section 8,
deliverable 2). Every prompt ends with the same final-answer envelope
instruction so answer_parser.py has one stable format to parse regardless
of category.
"""

from __future__ import annotations

CATEGORIES = (
    "Needle in the Haystack",
    "The Dark Web",
    "Ship It",
    "Ancient Scrolls",
    "Cryptic",
    "Heavy Compute",
)

_FINAL_ANSWER_ENVELOPE = """
When you are confident in a final answer, respond with exactly one line in
this form and nothing else on that line:

FINAL_ANSWER: <answer>

Rules:
- If a tool call returned an exact_value, copy it into <answer> character
  for character. Do not retype, reformat, or "clean up" an exact_value.
- Do not include explanation, units, or punctuation the answer format
  forbids.
- If you are not confident, keep working: call another tool or reconsider,
  do not emit FINAL_ANSWER until you believe the answer is correct.
""".strip()

_BASE = """
You are solving one Jeopardy-style tile under a strict turn and token
budget. You have tools available; use them for anything that benefits from
real computation, file access, or HTTP rather than guessing from memory.
Prefer a tool-verified answer over a remembered one whenever a tool can
check it.
""".strip()

_CATEGORY_PROMPTS: dict[str, str] = {
    "Needle in the Haystack": """
Category: Needle in the Haystack. The prompt or attached files contain a
large amount of data with one specific fact buried inside. Do not read the
whole haystack into your own reasoning if a tool can search, filter, or
grep it. Prefer targeted lookups over broad summarization. State exactly
which record/row/line supports your answer as evidence.
""".strip(),
    "The Dark Web": """
Category: The Dark Web. Solving this requires stateful HTTP: login flows,
cookies, redirects, and HTML forms. Do not attempt this from memory or by
guessing URLs — delegate every request to the web tool so cookies and
sessions are handled correctly. Read tool output carefully for rejected
logins, redirect loops, or missing form fields before retrying.
""".strip(),
    "Ship It": """
Category: Ship It. This is a code diagnosis/fix task. Do not just describe
what looks wrong — delegate to the runtime tool to actually run the code
and its tests. Diagnose from real failing output, make the smallest fix
that passes, and re-run tests before answering. An answer without a
passing test run behind it should be treated as unverified.
""".strip(),
    "Ancient Scrolls": """
Category: Ancient Scrolls. The material is a long document (or set of
documents). Do not try to hold the whole text in your head — use the
document tool to chunk, index, and search rather than reading linearly.
Cite the specific passage your answer comes from.
""".strip(),
    "Cryptic": """
Category: Cryptic. The prompt likely involves encoded, obfuscated, or
binary content (ciphers, encodings, archives, unusual file formats).
Identify the encoding/format first using a tool before attempting to
decode anything by eye — do not hand-decode base64/hex/rot13 etc. from
memory when a tool can do it exactly.
""".strip(),
    "Heavy Compute": """
Category: Heavy Compute. This requires real calculation, simulation, or
constraint solving that is unsafe to do by mental arithmetic. Delegate
computation to the runtime tool. After you have a candidate result,
independently re-derive or re-check it against the stated constraints
before answering — do not trust a single computation pass.
""".strip(),
}


def get_system_prompt(category: str) -> str:
    """
    Returns the full system prompt for a category. Unknown categories fall
    back to the base prompt plus the envelope rather than raising — a
    malformed/unexpected category string must degrade gracefully, not
    crash the solve.
    """
    category_block = _CATEGORY_PROMPTS.get(category, "")
    parts = [_BASE]
    if category_block:
        parts.append(category_block)
    parts.append(_FINAL_ANSWER_ENVELOPE)
    return "\n\n".join(parts)
