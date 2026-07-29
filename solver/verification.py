"""
Verification: the deliverable that "verification is stronger than ask the
model again" (TEAM_PLAN.md section 8, deliverable 5).

verify_candidate() runs a pipeline of checks and returns a
(passed: bool, confidence: float, reasons: list[str]) tuple. Nandh's
submission gate treats `passed=False` as "do not submit" regardless of the
model's own stated confidence — this module is the thing that can veto the
model, not just rubber-stamp it.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass

from contracts import CandidateAnswer, TaskContext

# Category-specific checks are registered by specialists/*.py via
# register_category_check() so this module doesn't need to import every
# specialist directly (keeps verification.py generic; category logic
# stays owned in specialists/).
_CategoryCheck = "callable(candidate, task) -> tuple[bool, str]"
_category_checks: dict[str, list] = {}


def register_category_check(category: str, check_fn) -> None:
    _category_checks.setdefault(category, []).append(check_fn)


@dataclass(frozen=True)
class VerificationOutcome:
    passed: bool
    confidence: float
    reasons: tuple[str, ...]


def verify_candidate(
    candidate: CandidateAnswer, task: TaskContext
) -> VerificationOutcome:
    reasons: list[str] = []
    passed = True
    confidence = candidate.confidence

    # 1. Exact-token preservation: if the candidate claims to be a tool's
    #    exact_value, its value must not have been touched by the model.
    #    (agent_loop.py is responsible for actually setting this flag only
    #    when it copied a ToolResult.exact_value verbatim; here we just
    #    trust and record it — re-checking requires the tool log, which
    #    agent_loop passes through evidence.)
    if candidate.exact_value_from_tool:
        reasons.append("value sourced verbatim from tool exact_value")

    # 2. Format-specific local checks.
    if task.answer_format == "numeric":
        ok, reason = _verify_numeric(candidate)
        passed &= ok
        reasons.append(reason)
    elif task.answer_format == "literal":
        ok, reason = _verify_literal(candidate)
        passed &= ok
        reasons.append(reason)
    elif task.answer_format in ("exact", "exact_ci"):
        ok, reason = _verify_nonempty(candidate)
        passed &= ok
        reasons.append(reason)
    elif task.answer_format == "validator":
        # No local check possible without the validator spec; category
        # checks below (e.g. Heavy Compute's constraint check) are the
        # real gate for this format.
        reasons.append("validator format: relying on category-specific check")

    # 3. Category-specific checks (Cryptic encodings, Ship It test
    #    evidence, Heavy Compute constraints, etc.), contributed by
    #    specialists/*.py.
    for check_fn in _category_checks.get(task.category, []):
        ok, reason = check_fn(candidate, task)
        passed &= ok
        reasons.append(reason)

    # 4. No evidence at all is itself a red flag — a candidate with a
    #    non-trivial confidence but zero cited evidence did not earn that
    #    confidence.
    if not candidate.evidence and candidate.confidence > 0.5:
        passed = False
        reasons.append("high confidence but no evidence recorded")
        confidence = min(confidence, 0.5)

    if not passed:
        confidence = min(confidence, 0.4)

    return VerificationOutcome(passed=passed, confidence=confidence, reasons=tuple(reasons))


def _verify_nonempty(candidate: CandidateAnswer) -> tuple[bool, str]:
    if candidate.value.strip():
        return True, "exact/exact_ci: non-empty value"
    return False, "exact/exact_ci: empty value rejected"


def _verify_literal(candidate: CandidateAnswer) -> tuple[bool, str]:
    value = candidate.value.strip()
    if not value:
        return False, "literal: empty value rejected"
    if value.upper().startswith("FINAL_ANSWER"):
        return False, "literal: envelope leaked into value"
    return True, "literal: format check passed"


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_arithmetic(expr: str) -> float | None:
    """
    Evaluates a plain arithmetic expression (+ - * / ** parens, numeric
    literals only) without calling Python's eval(). Returns None if the
    expression isn't pure arithmetic — this is a recompute check, not a
    general expression evaluator, and must fail closed on anything odd.
    """
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(n.op)](_eval(n.operand))
        raise ValueError("disallowed expression node")

    try:
        return float(_eval(node))
    except Exception:  # noqa: BLE001 — deliberately fail closed to None
        return None


def _verify_numeric(candidate: CandidateAnswer) -> tuple[bool, str]:
    try:
        claimed = float(candidate.value)
    except ValueError:
        return False, f"numeric: value {candidate.value!r} is not a number"

    # If any evidence line looks like a pure arithmetic expression that
    # was supposedly used to derive the answer, recompute it independently
    # and require it to match within a small tolerance. This is the
    # "recompute numeric answers when possible" deliverable — it does NOT
    # require the model to show its work in a specific format, it just
    # takes advantage of it when present.
    for line in candidate.evidence:
        recomputed = _safe_eval_arithmetic(line.strip())
        if recomputed is not None:
            if abs(recomputed - claimed) > max(1e-6, abs(claimed) * 1e-9):
                return (
                    False,
                    f"numeric: independent recompute of {line!r} = {recomputed}, "
                    f"does not match claimed {claimed}",
                )
            return True, f"numeric: recompute of {line!r} matches claimed value"

    return True, "numeric: parses as a number (no recomputable expression in evidence)"
