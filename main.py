"""Agent Jeopardy entrypoint owned by Nandh.

This module assembles the board orchestrator around an injected single-tile
solver. Sara's workstream can add ``solver.build_solver`` without editing this
file. Until then, the organizer's naive one-call baseline is available for
local plumbing checks, but its default confidence is zero so guesses never
reach the submission API.
"""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping


def _load_local_environment() -> None:
    """Load private credentials, then fill in tracked deployment defaults."""
    for filename in (".env", "agent.env"):
        env_path = Path(__file__).with_name(filename)
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                parts = shlex.split(line, comments=True)
            except ValueError:
                continue
            if parts and parts[0] == "export":
                parts = parts[1:]
            if len(parts) != 1 or "=" not in parts[0]:
                continue
            key, value = parts[0].split("=", 1)
            if key.isidentifier():
                os.environ.setdefault(key, os.path.expandvars(value))


def _use_local_virtualenv() -> None:
    """Make `python3 main.py` use the prepared local runtime when necessary."""
    local_python = Path(__file__).with_name(".venv") / "bin" / "python"
    if not local_python.is_file():
        return
    if Path(sys.prefix).resolve() == local_python.parent.parent.resolve():
        return
    os.execv(str(local_python), [str(local_python), *sys.argv])


if __name__ == "__main__":
    _load_local_environment()
    _use_local_virtualenv()

import jeopardy as jp

from contracts import CandidateAnswer, SolveResult, TaskContext, TileSolver
from orchestrator import AgentOrchestrator, OrchestratorConfig, SubmissionGate
from orchestrator.submission_gate import SubmissionPolicy

VERBOSE = os.environ.get("VERBOSE") == "1"
MAX_TILES_SETTING = os.environ.get("MAX_TILES")
BASELINE_CONFIDENCE = float(os.environ.get("BASELINE_CONFIDENCE", "0"))

# The six category names are part of the event API contract.  Keep the
# tier-one relaxation scoped to known categories so an unexpected category or
# malformed task cannot silently inherit the more aggressive threshold.
SCORING_CATEGORIES = (
    "Needle in the Haystack",
    "The Dark Web",
    "Ship It",
    "Ancient Scrolls",
    "Cryptic",
    "Heavy Compute",
)
SCORED_BOARDS = frozenset({"qual", "main"})


def build_submission_policy(
    environ: Mapping[str, str] | None = None,
) -> SubmissionPolicy:
    """Build the calibrated, environment-configurable submission policy.

    Practice results showed that verified, non-tool tier-one candidates at
    confidence 0.70 were correct 7/7 times.  A 0.65 threshold therefore lets
    those candidates compete for 100-point scored tiles without weakening the
    0.80 threshold for higher tiers.  Deterministic tool outputs are emitted by
    the solver at confidence 0.95 and remain eligible at every tier.

    ``MIN_CONFIDENCE`` controls the conservative default,
    ``TIER_ONE_MIN_CONFIDENCE`` controls the calibrated tier-one threshold,
    and ``TIER_ONE_POINTS`` can adapt to an event with a different first tier.
    """
    settings = os.environ if environ is None else environ
    default_minimum = float(settings.get("MIN_CONFIDENCE", "0.80"))
    tier_one_minimum = float(
        settings.get("TIER_ONE_MIN_CONFIDENCE", "0.65")
    )
    tier_one_points = int(settings.get("TIER_ONE_POINTS", "100"))
    interval = float(settings.get("SUBMISSION_INTERVAL_SECONDS", "3.1"))

    if tier_one_points < 0:
        raise ValueError("TIER_ONE_POINTS must be non-negative")
    if tier_one_minimum > default_minimum:
        raise ValueError(
            "TIER_ONE_MIN_CONFIDENCE cannot exceed MIN_CONFIDENCE"
        )

    return SubmissionPolicy(
        minimum_interval_seconds=interval,
        default_minimum_confidence=default_minimum,
        confidence_overrides={
            (category, tier_one_points): tier_one_minimum
            for category in SCORING_CATEGORIES
        },
    )


class CompetitionSubmissionGate(SubmissionGate):
    """Apply calibrated overrides only to the two scored boards.

    The API reports practice, qualifier, and game tasks as ``practice``,
    ``qual``, and ``main`` respectively.  Unknown/empty board values use the
    conservative default too; a partial task response must never opt itself
    into the lower competitive threshold.
    """

    def __init__(self, game, policy: SubmissionPolicy) -> None:
        super().__init__(game, policy)
        strict_policy = replace(policy, confidence_overrides={})
        self._strict_validation_gate = SubmissionGate(game, strict_policy)

    def validate(self, record, candidate) -> str | None:
        if record.board in SCORED_BOARDS:
            return super().validate(record, candidate)
        return self._strict_validation_gate.validate(record, candidate)


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


def solver_budget_log_fields(team_solver_loaded: bool) -> str:
    """Render the solver's effective budgets for the startup line.

    Read the defaults from ``solver.agent_loop`` rather than repeating them, so
    the one log line an operator greps at minute 50 cannot claim a turn limit
    the solver is not actually running.  Imported lazily because the naive
    baseline path exists precisely for when the solver package is absent.
    """
    if not team_solver_loaded:
        return (
            "solver_turn_limit=n/a solver_token_limit=n/a "
            "tool_timeout_seconds=n/a"
        )
    from solver.agent_loop import (
        MAX_TOTAL_TOKENS_DEFAULT,
        MAX_TURNS_DEFAULT,
        TOOL_TIMEOUT_SECONDS_DEFAULT,
    )

    turns = os.environ.get("SOLVER_MAX_TURNS", str(MAX_TURNS_DEFAULT))
    tokens = os.environ.get(
        "SOLVER_MAX_TOTAL_TOKENS", str(MAX_TOTAL_TOKENS_DEFAULT)
    )
    tool_timeout = os.environ.get(
        "TOOL_TIMEOUT_SECONDS", str(TOOL_TIMEOUT_SECONDS_DEFAULT)
    )
    return (
        f"solver_turn_limit={turns} solver_token_limit={tokens} "
        f"tool_timeout_seconds={tool_timeout}"
    )


def build_orchestrator_config(
    *,
    max_tiles: int,
    environ: Mapping[str, str] | None = None,
) -> OrchestratorConfig:
    """Build throughput settings calibrated for the hosted qualifier.

    Six workers are the measured safe width for the hosted model proxy.
    Production telemetry showed that retiring claimed tiles early allowed
    their still-running model calls to overlap replacement workers, pushing
    effective concurrency and token usage above the 95k/minute proxy limit.
    Cooperative cancellation now prevents that hidden growth, while six true
    workers preserve useful I/O overlap without creating a throttle queue.

    The 180-second deadline is set deliberately in step with
    ``agent_loop.MAX_TURNS_DEFAULT``: the deadline is checked before every turn,
    so a turn budget the clock cannot reach is dead configuration.  It also
    absorbs the extra proxy latency that heavier parallelism invites — the event
    proxy delays calls past the shared per-minute token limit rather than
    failing them, and a tile killed at its deadline while waiting on a delayed
    call scores nothing.

    Every value remains overridable for live rate-limit or latency tuning.
    """
    settings = os.environ if environ is None else environ
    task_filter = tuple(
        task_id.strip()
        for task_id in settings.get("TASK_FILTER", "").split(",")
        if task_id.strip()
    )
    return OrchestratorConfig(
        max_workers=int(settings.get("MAX_WORKERS", "6")),
        poll_interval_seconds=float(settings.get("POLL_SECONDS", "2")),
        task_timeout_seconds=float(
            settings.get("TASK_TIMEOUT_SECONDS", "180")
        ),
        max_tiles=max_tiles,
        max_solve_attempts=int(settings.get("MAX_SOLVE_ATTEMPTS", "3")),
        task_filter=task_filter,
        heartbeat_interval_seconds=float(
            settings.get("HEARTBEAT_INTERVAL_SECONDS", "30")
        ),
        task_prefetch_enabled=settings.get("TASK_PREFETCH", "1") == "1",
        prefetch_workers=int(settings.get("PREFETCH_WORKERS", "2")),
        prefetch_lookahead=int(settings.get("PREFETCH_LOOKAHEAD", "12")),
        prefetch_timeout_seconds=float(
            settings.get("PREFETCH_TIMEOUT_SECONDS", "10")
        ),
        prefetch_join_seconds=float(
            settings.get("PREFETCH_JOIN_SECONDS", "0.25")
        ),
    )


def build_orchestrator(
    solver: TileSolver,
    *,
    max_tiles: int,
    config: OrchestratorConfig | None = None,
) -> AgentOrchestrator:
    resolved_config = config or build_orchestrator_config(max_tiles=max_tiles)
    gate = CompetitionSubmissionGate(
        jp,
        build_submission_policy(),
    )
    return AgentOrchestrator(
        jp,
        solver,
        gate,
        config=resolved_config,
        fatal_error_types=(jp.AuthError,),
    )


def main() -> None:
    solver, team_solver_loaded = build_solver()
    max_tiles = int(MAX_TILES_SETTING) if MAX_TILES_SETTING is not None else (
        0 if team_solver_loaded else 3
    )
    config = build_orchestrator_config(max_tiles=max_tiles)
    policy = build_submission_policy()
    jp.log(
        f"event=agent_start workers={config.max_workers} "
        f"task_timeout_seconds={config.task_timeout_seconds:g} "
        f"heartbeat_interval_seconds={config.heartbeat_interval_seconds:g} "
        f"max_solve_attempts={config.max_solve_attempts} "
        f"min_confidence={policy.default_minimum_confidence:.3f} "
        f"task_filter_count={len(config.task_filter)} "
        f"{solver_budget_log_fields(team_solver_loaded)} "
        f"task_prefetch={config.task_prefetch_enabled} "
        f"prefetch_workers={config.prefetch_workers} "
        f"prefetch_lookahead={config.prefetch_lookahead}"
    )
    orchestrator = build_orchestrator(
        solver, max_tiles=max_tiles, config=config
    )
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
        orchestrator.drain_workers(timeout=config.task_timeout_seconds + 5)
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
