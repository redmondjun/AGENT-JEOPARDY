"""Agent Jeopardy entrypoint owned by Nandh.

This module assembles the board orchestrator around an injected single-tile
solver. Sara's workstream can add ``solver.build_solver`` without editing this
file. Until then, the organizer's naive one-call baseline is available for
local plumbing checks, but its default confidence is zero so guesses never
reach the submission API.
"""

from __future__ import annotations

import os
from pathlib import Path

import jeopardy as jp

from contracts import CandidateAnswer, SolveResult, TaskContext, TileSolver
from orchestrator import AgentOrchestrator, OrchestratorConfig, SubmissionGate
from orchestrator.submission_gate import SubmissionPolicy

VERBOSE = os.environ.get("VERBOSE") == "1"
TASK_FILTER = tuple(
    task_id.strip()
    for task_id in os.environ.get("TASK_FILTER", "").split(",")
    if task_id.strip()
)
MAX_TILES_SETTING = os.environ.get("MAX_TILES")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
MAX_SOLVE_ATTEMPTS = int(os.environ.get("MAX_SOLVE_ATTEMPTS", "3"))
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "2"))
TASK_TIMEOUT_SECONDS = float(os.environ.get("TASK_TIMEOUT_SECONDS", "90"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.80"))
SUBMISSION_INTERVAL_SECONDS = float(
    os.environ.get("SUBMISSION_INTERVAL_SECONDS", "3.1")
)
BASELINE_CONFIDENCE = float(os.environ.get("BASELINE_CONFIDENCE", "0"))


class NaiveBaselineSolver:
    """Compatibility floor used only until Sara's solver package is present."""

    def __init__(self, client, *, verbose: bool = False) -> None:
        self._client = client
        self._verbose = verbose

    def solve(self, task: TaskContext) -> SolveResult:
        prompt = (
            f"{task.prompt}\n\n"
            f"Files downloaded to {task.workdir}: "
            f"{[path.name for path in task.files] or 'none'}\n"
            f"Answer checking: {task.answer_format}\n\n"
            "Reply with ONLY the final answer — no working, no explanation."
        )
        if self._verbose:
            jp.log(f"{task.task_id} baseline prompt:\n{prompt}\n---")
        response = self._client.messages.create(
            model=jp.MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if self._verbose:
            jp.log(f"{task.task_id} baseline reply:\n{answer}\n---")
        return SolveResult(
            candidate=CandidateAnswer(
                value=answer,
                confidence=BASELINE_CONFIDENCE,
                evidence=(),
                strategy="naive-baseline",
            ),
            retryable=False,
        )


def build_solver() -> tuple[TileSolver, bool]:
    """Load Sara's solver through a stable factory, or use the safe baseline."""
    solver_package = Path(__file__).with_name("solver") / "__init__.py"
    if not solver_package.exists():
        jp.log(
            "solver package not installed; using bounded naive baseline "
            "with submissions disabled by confidence gate"
        )
        return NaiveBaselineSolver(jp.anthropic_client(), verbose=VERBOSE), False
    # Import failures inside Sara's package are deployment failures. Never hide
    # them by silently falling back to a zero-confidence baseline.
    from solver import build_solver as build_team_solver

    return build_team_solver(game=jp, verbose=VERBOSE), True


def build_orchestrator(solver: TileSolver, *, max_tiles: int) -> AgentOrchestrator:
    config = OrchestratorConfig(
        max_workers=MAX_WORKERS,
        poll_interval_seconds=POLL_SECONDS,
        task_timeout_seconds=TASK_TIMEOUT_SECONDS,
        max_tiles=max_tiles,
        max_solve_attempts=MAX_SOLVE_ATTEMPTS,
        task_filter=TASK_FILTER,
    )
    gate = SubmissionGate(
        jp,
        SubmissionPolicy(
            minimum_interval_seconds=SUBMISSION_INTERVAL_SECONDS,
            default_minimum_confidence=MIN_CONFIDENCE,
        ),
    )
    return AgentOrchestrator(
        jp,
        solver,
        gate,
        config=config,
        fatal_error_types=(jp.AuthError,),
    )


def main() -> None:
    solver, team_solver_loaded = build_solver()
    max_tiles = int(MAX_TILES_SETTING) if MAX_TILES_SETTING is not None else (
        0 if team_solver_loaded else 3
    )
    orchestrator = build_orchestrator(solver, max_tiles=max_tiles)
    run_forever_override = os.environ.get("RUN_FOREVER")
    run_forever = (
        team_solver_loaded
        if run_forever_override is None
        else run_forever_override == "1"
    )
    if run_forever:
        orchestrator.run_forever()
        return

    # Safe local baseline mode: one bounded batch, then process its results.
    try:
        first = orchestrator.run_cycle()
        jp.log(
            f"baseline cycle: open={first.open_tiles} dispatched={first.dispatched}"
        )
        orchestrator.drain_workers(timeout=TASK_TIMEOUT_SECONDS + 5)
        final = orchestrator.run_cycle()
        jp.log(
            f"baseline complete: completed={final.completed} "
            f"submitted={final.submitted}"
        )
    finally:
        orchestrator.close()


if __name__ == "__main__":
    try:
        main()
    except jp.AuthError as exc:
        raise SystemExit(f"[auth] {exc}") from exc
