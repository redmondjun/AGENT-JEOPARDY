"""Fail-closed deterministic solvers for unambiguous task signatures."""

from __future__ import annotations

import ast
import base64
import binascii
import codecs
import csv
import operator
import re
import time
from pathlib import Path
from typing import Callable

from contracts import CandidateAnswer, SolveResult, SolveTelemetry, TaskContext
from solver.verification import verify_candidate

FastPath = Callable[[TaskContext], CandidateAnswer | None]

_ARITHMETIC = re.compile(
    r"^\s*(?:evaluate|calculate)\s*[:：]\s*([0-9eE+\-*/().%\s]+)\s*[?.]?\s*$",
    re.IGNORECASE,
)
_CSV_ROWS = re.compile(
    r"^\s*how many (?:data )?rows (?:are there|are in (?:the )?(?:csv|file))\??\s*$",
    re.IGNORECASE,
)
_COUNT_TEXT = re.compile(
    r"^\s*count the case-sensitive occurrences of ['\"](.+?)['\"] "
    r"in (?:the )?(?:document|file)\.?\s*$",
    re.IGNORECASE,
)
_DECODE = re.compile(
    r"^\s*decode (?:the )?(?:contents|message|value) (?:as|using) "
    r"(base64|hex|rot13)\.?\s*$",
    re.IGNORECASE,
)

_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


class CompositeTileSolver:
    """Try deterministic category handlers, then use the general solver."""

    def __init__(self, fallback, *, logger=None) -> None:
        self._fallback = fallback
        self._logger = logger or (lambda _message: None)
        self._handlers: dict[str, FastPath] = {
            "Needle in the Haystack": _solve_csv_row_count,
            "The Dark Web": _solve_declarative_recipe,
            "Ship It": _solve_declarative_recipe,
            "Ancient Scrolls": _solve_text_count,
            "Cryptic": _solve_explicit_decode,
            "Heavy Compute": _solve_arithmetic,
        }

    def solve(self, task: TaskContext) -> SolveResult:
        started = time.monotonic()
        cancel_event = task.metadata.get("cancel_event")
        if (
            cancel_event is not None
            and getattr(cancel_event, "is_set", lambda: False)()
        ):
            return SolveResult(
                candidate=None,
                retryable=False,
                failure_code="TILE_BECAME_STALE",
                telemetry=SolveTelemetry(),
            )
        handler = self._handlers.get(task.category)
        try:
            candidate = handler(task) if handler is not None else None
        except Exception as exc:  # optimization failures fall back, never retry
            self._logger(
                f"event=fast_path task={task.task_id} outcome=fallback "
                f"error_type={type(exc).__name__}"
            )
            candidate = None
        if candidate is not None:
            verified = verify_candidate(candidate, task)
            if verified.passed:
                return SolveResult(
                    candidate=CandidateAnswer(
                        value=candidate.value,
                        confidence=max(candidate.confidence, verified.confidence),
                        evidence=candidate.evidence,
                        strategy=candidate.strategy,
                        exact_value_from_tool=False,
                    ),
                    retryable=False,
                    telemetry=SolveTelemetry(
                        elapsed_ms=int((time.monotonic() - started) * 1000)
                    ),
                )
        return self._fallback.solve(task)


def _candidate(task: TaskContext, value: str, evidence: str) -> CandidateAnswer:
    return CandidateAnswer(
        value=value,
        confidence=0.95,
        evidence=(evidence,),
        strategy=f"{task.category}:deterministic_fast_path",
    )


def _single_file(task: TaskContext, suffixes: set[str] | None = None) -> Path | None:
    files = [
        path
        for path in task.files
        if suffixes is None or path.suffix.lower() in suffixes
    ]
    return files[0] if len(files) == 1 else None


def _solve_csv_row_count(task: TaskContext) -> CandidateAnswer | None:
    if not _CSV_ROWS.fullmatch(task.prompt):
        return None
    path = _single_file(task, {".csv"})
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        count = max(0, sum(1 for _ in csv.reader(handle)) - 1)
    return _candidate(task, str(count), f"counted {count} CSV data rows")


def _solve_text_count(task: TaskContext) -> CandidateAnswer | None:
    match = _COUNT_TEXT.fullmatch(task.prompt)
    path = _single_file(task, {".txt", ".md"})
    if match is None or path is None:
        return None
    needle = match.group(1)
    count = path.read_text(encoding="utf-8", errors="replace").count(needle)
    return _candidate(task, str(count), f"case-sensitive count computed as {count}")


def _solve_explicit_decode(task: TaskContext) -> CandidateAnswer | None:
    match = _DECODE.fullmatch(task.prompt)
    path = _single_file(task, {".txt", ".dat", ""})
    if match is None or path is None:
        return None
    value = path.read_text(encoding="utf-8", errors="strict").strip()
    encoding = match.group(1).lower()
    try:
        if encoding == "base64":
            decoded = base64.b64decode(value, validate=True).decode("utf-8")
        elif encoding == "hex":
            decoded = binascii.unhexlify("".join(value.split())).decode("utf-8")
        else:
            decoded = codecs.decode(value, "rot_13")
    except (ValueError, UnicodeError, binascii.Error):
        return None
    return _candidate(task, decoded, f"decoded {encoding} deterministically")


def _solve_arithmetic(task: TaskContext) -> CandidateAnswer | None:
    match = _ARITHMETIC.fullmatch(task.prompt)
    if match is None or task.files:
        return None
    value = _safe_arithmetic(match.group(1))
    if value is None:
        return None
    rendered = str(int(value)) if value.is_integer() else repr(value)
    return _candidate(task, rendered, f"independently evaluated expression to {rendered}")


def _safe_arithmetic(expression: str) -> float | None:
    try:
        root = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError

    try:
        value = evaluate(root)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return value if abs(value) < 1e100 else None


def _solve_declarative_recipe(task: TaskContext) -> CandidateAnswer | None:
    """Fixture hook for deterministic web/code generators.

    The live API currently provides no stable generator signature for these
    categories. A fixture may supply a pre-verified deterministic answer in
    metadata, but ordinary live tasks always fall back to the tool-use solver.
    """
    recipe = task.metadata.get("deterministic_fast_path")
    if not isinstance(recipe, dict) or recipe.get("verified") is not True:
        return None
    value = recipe.get("answer")
    evidence = recipe.get("evidence")
    if not isinstance(value, str) or not isinstance(evidence, str):
        return None
    return _candidate(task, value, evidence)
