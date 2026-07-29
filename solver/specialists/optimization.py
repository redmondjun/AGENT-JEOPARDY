"""
Heavy Compute category: TEAM_PLAN.md section 8 explicitly calls for
"independent objective checking" here, not just running one computation
and trusting it. This module extracts simple numeric constraints stated in
the prompt (">= 10", "at most 5", "== 42") and checks the final candidate
against them — a cheap, deterministic sanity net that catches an answer
that satisfies "the math" but violates a stated constraint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contracts import CandidateAnswer, TaskContext
from solver.verification import register_category_check

_CONSTRAINT_RE = re.compile(
    r"(>=|<=|==|>|<|at least|at most|no more than|no fewer than)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_WORD_OP = {
    "at least": ">=",
    "at most": "<=",
    "no more than": "<=",
    "no fewer than": ">=",
}


@dataclass(frozen=True)
class Constraint:
    op: str
    bound: float

    def satisfied_by(self, value: float) -> bool:
        if self.op == ">=":
            return value >= self.bound
        if self.op == "<=":
            return value <= self.bound
        if self.op == "==":
            return abs(value - self.bound) < 1e-9
        if self.op == ">":
            return value > self.bound
        if self.op == "<":
            return value < self.bound
        return False


def extract_constraints(prompt: str) -> list[Constraint]:
    constraints = []
    for match in _CONSTRAINT_RE.finditer(prompt):
        raw_op, raw_bound = match.groups()
        op = _WORD_OP.get(raw_op.lower(), raw_op)
        constraints.append(Constraint(op=op, bound=float(raw_bound)))
    return constraints


def _check_heavy_compute_constraints(
    candidate: CandidateAnswer, task: TaskContext
) -> tuple[bool, str]:
    constraints = extract_constraints(task.prompt)
    if not constraints:
        return True, "heavy_compute: no extractable numeric constraints to check"

    try:
        value = float(candidate.value)
    except ValueError:
        return True, "heavy_compute: candidate is non-numeric, skipping constraint check"

    violated = [c for c in constraints if not c.satisfied_by(value)]
    if violated:
        details = ", ".join(f"{c.op} {c.bound}" for c in violated)
        return False, f"heavy_compute: candidate {value} violates stated constraint(s): {details}"
    return True, f"heavy_compute: candidate satisfies all {len(constraints)} extracted constraint(s)"


register_category_check("Heavy Compute", _check_heavy_compute_constraints)
