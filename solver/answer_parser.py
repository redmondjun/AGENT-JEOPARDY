"""
Answer parsing for all server answer formats: exact, exact_ci, numeric,
literal, validator.

Two separate concerns live here on purpose:
  1. extract_final_answer() pulls the raw candidate string out of the
     model's free-form turn text (the FINAL_ANSWER: <answer> envelope).
  2. normalize_answer() reformats that raw string per answer_format
     WITHOUT changing its semantic value — e.g. numeric normalization can
     strip "$", ",", trailing ".0", but must never round or truncate in a
     way that changes the answer.

A ToolResult.exact_value, when present, bypasses both of these — it goes
straight into CandidateAnswer.value untouched. That path lives in
agent_loop.py, not here, because this module has no concept of tool
results, only text.
"""

from __future__ import annotations

import re

from contracts import AnswerFormat

_FINAL_ANSWER_RE = re.compile(r"^FINAL_ANSWER:\s*(.+?)\s*$", re.MULTILINE)


def extract_final_answer(model_text: str) -> str | None:
    """
    Returns the raw answer string from the last FINAL_ANSWER: line in the
    model's turn text, or None if no such line exists. Uses the LAST match
    so that if the model second-guesses itself mid-turn and emits a
    corrected envelope, the corrected one wins.
    """
    matches = _FINAL_ANSWER_RE.findall(model_text)
    if not matches:
        return None
    return matches[-1]


def normalize_answer(raw: str, answer_format: AnswerFormat) -> str:
    """
    Reformat `raw` for the given answer_format. Never changes the
    semantic value of the answer — only strips formatting noise that the
    server would otherwise reject on a technicality.
    """
    if answer_format == "exact":
        return raw.strip()

    if answer_format == "exact_ci":
        # Case-insensitive exact match on the server side; we still return
        # the model's original casing (server does the folding), just
        # trimmed.
        return raw.strip()

    if answer_format == "numeric":
        return _normalize_numeric(raw)

    if answer_format == "literal":
        return _normalize_literal(raw)

    if answer_format == "validator":
        # Validator-checked answers are opaque to us; pass through as-is,
        # trimmed only. Any real checking happens in verification.py against
        # task.metadata's constraint description, not here.
        return raw.strip()

    # Unknown/future format: safest behavior is to pass through unmodified
    # rather than guess at a transformation.
    return raw.strip()


_NUMERIC_STRIP_CHARS = re.compile(r"[,$\s]")


def _normalize_numeric(raw: str) -> str:
    text = raw.strip()
    text = _NUMERIC_STRIP_CHARS.sub("", text)

    # Strip a single trailing "%" — caller-side verification decides
    # whether percent vs fraction matters for a given task; we only strip
    # presentation noise here.
    if text.endswith("%"):
        text = text[:-1]

    # Collapse "-0" style negative zero and trailing ".0"/".00" without
    # touching the actual magnitude.
    try:
        value = float(text)
    except ValueError:
        # Not parseable as a plain number (e.g. "1/2", "3.14e2" edge cases
        # our regex mangled) — return the lightly-cleaned text rather than
        # raising, so verification.py can still attempt to recompute it.
        return text

    if value == int(value):
        return str(int(value))
    return repr(value)


def _normalize_literal(raw: str) -> str:
    # "literal" answers (e.g. a specific word/name/phrase) get whitespace
    # collapsed and outer punctuation trimmed, but internal casing and
    # punctuation are preserved since they may be semantically load-bearing
    # (e.g. a proper noun, a filename, a code identifier).
    text = raw.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("\"'` ")
    return text
