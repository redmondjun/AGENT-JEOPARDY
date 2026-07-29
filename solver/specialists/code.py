"""
Ship It category: the answer is almost always "what's the fix" or "what
does this now output", and both are only trustworthy if they come from
actually running code (via Vidula's runtime tool), not from reading the
code and guessing. This module's verification check enforces that: it
looks for evidence that a test or command was actually executed, and
rejects candidates that only ever "reasoned about" the code.
"""

from __future__ import annotations

import re

from contracts import CandidateAnswer, TaskContext
from solver.verification import register_category_check

_RUN_EVIDENCE_MARKERS = (
    "passed",
    "failed",
    "exit code",
    "returncode",
    "traceback",
    "assert",
    "pytest",
    "stdout",
    "stderr",
)


def looks_like_execution_evidence(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _RUN_EVIDENCE_MARKERS)


def extract_exit_code(text: str) -> int | None:
    match = re.search(r"exit code[:\s]+(-?\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _check_ship_it_has_run_evidence(
    candidate: CandidateAnswer, task: TaskContext
) -> tuple[bool, str]:
    if any(looks_like_execution_evidence(line) for line in candidate.evidence):
        return True, "ship_it: evidence includes what looks like real test/command output"
    return (
        False,
        "ship_it: no evidence of an actual test/command run — answer is unverified reasoning",
    )


register_category_check("Ship It", _check_ship_it_has_run_evidence)
