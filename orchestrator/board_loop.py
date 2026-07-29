"""Long-running board polling, solving, and submission orchestration."""

from __future__ import annotations

import threading
import time
from concurrent.futures import wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from contracts import GameAPI, SolveResult, TaskContext, TileSolver

from .priority import PriorityPolicy
from .scheduler import BoundedWorkerPool
from .state import InvalidTransition, TERMINAL_STATES, TileRecord, TileState, TileTracker
from .submission_gate import SubmissionAction, SubmissionGate


class FatalSubmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrchestratorConfig:
    max_workers: int = 4
    poll_interval_seconds: float = 2.0
    error_backoff_seconds: float = 5.0
    task_timeout_seconds: float = 90.0
    solve_retry_seconds: float = 10.0
    max_tiles: int = 0
    max_solve_attempts: int = 3
    task_filter: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll interval must be positive")
        if self.error_backoff_seconds <= 0:
            raise ValueError("error backoff must be positive")
        if self.task_timeout_seconds <= 0:
            raise ValueError("task timeout must be positive")
        if self.solve_retry_seconds < 0:
            raise ValueError("solve retry must be non-negative")
        if self.max_tiles < 0:
            raise ValueError("max_tiles must be non-negative")
        if self.max_solve_attempts <= 0:
            raise ValueError("max_solve_attempts must be positive")


@dataclass(frozen=True)
class CycleReport:
    phase: str
    open_tiles: int
    discovered: int
    dispatched: int
    completed: int
    submitted: int
    active_workers: int


@dataclass(frozen=True)
class _WorkerResult:
    solve_result: SolveResult
    answer_format: str
    board: str


class AgentOrchestrator:
    def __init__(
        self,
        game: GameAPI,
        solver: TileSolver,
        submission_gate: SubmissionGate,
        *,
        config: OrchestratorConfig | None = None,
        tracker: TileTracker | None = None,
        priority: PriorityPolicy | None = None,
        clock=time.monotonic,
        fatal_error_types: tuple[type[BaseException], ...] = (),
    ) -> None:
        self._game = game
        self._solver = solver
        self._gate = submission_gate
        self._config = config or OrchestratorConfig()
        self._clock = clock
        self._tracker = tracker or TileTracker(clock=clock)
        self._priority = priority or PriorityPolicy()
        self._pool: BoundedWorkerPool[_WorkerResult] = BoundedWorkerPool(
            self._config.max_workers,
            work_timeout_seconds=self._config.task_timeout_seconds,
            clock=clock,
        )
        self._fatal_error_types = fatal_error_types
        self._limited_task_ids: set[str] = set()
        self._phase: str | None = None
        self._closed = False

    @property
    def tracker(self) -> TileTracker:
        return self._tracker

    def run_cycle(self) -> CycleReport:
        self._ensure_open()
        completed = self._collect_completed()
        board = self._game.board()
        phase = str(board.get("phase") or "unknown")
        open_tiles = self._game.open_tiles(board)
        discovered = self._tracker.observe_open_tiles(open_tiles)
        active_ids = {str(tile["id"]) for tile in open_tiles}
        retired = self._pool.retire_except(active_ids)
        if retired:
            self._game.log(
                f"retired {len(retired)} stale workers: {','.join(retired)}"
            )
        if self._phase is not None and phase != self._phase:
            # MAX_TILES is a per-board sampling limit. A practice selection
            # must not prevent qualifier/finale ids from being considered.
            self._limited_task_ids.clear()
            self._game.log(f"phase changed: {self._phase} -> {phase}")
        self._phase = phase
        self._tracker.release_submission_cooldowns(now=self._clock())
        submitted = self._process_one_submission()
        dispatched = self._dispatch_available()
        return CycleReport(
            phase=phase,
            open_tiles=len(open_tiles),
            discovered=discovered,
            dispatched=dispatched,
            completed=completed,
            submitted=submitted,
            active_workers=self._pool.active_count,
        )

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop = stop_event or threading.Event()
        try:
            while not stop.is_set():
                delay = self._config.poll_interval_seconds
                try:
                    report = self.run_cycle()
                    self._game.log(
                        "phase="
                        f"{report.phase} open={report.open_tiles} "
                        f"active={report.active_workers} dispatched={report.dispatched} "
                        f"completed={report.completed} submitted={report.submitted}"
                    )
                    gate_wait = self._gate.seconds_until_available()
                    if self._tracker.ready() and gate_wait > 0:
                        delay = min(delay, gate_wait)
                except self._fatal_error_types:
                    raise
                except FatalSubmissionError:
                    raise
                except Exception as exc:  # noqa: BLE001 - supervisor boundary
                    self._game.log(f"orchestrator cycle failed: {exc!r}")
                    delay = self._config.error_backoff_seconds
                stop.wait(delay)
        finally:
            self.close(wait_for_workers=False)

    def drain_workers(self, timeout: float | None = None) -> int:
        futures = self._pool.futures()
        if futures:
            wait(futures, timeout=timeout)
        return self._collect_completed()

    def close(self, *, wait_for_workers: bool = True) -> None:
        if not self._closed:
            self._closed = True
            self._pool.close(wait=wait_for_workers)

    def _collect_completed(self) -> int:
        count = 0
        for work in self._pool.completed():
            count += 1
            try:
                result = work.future.result()
            except self._fatal_error_types:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate one tile
                record = self._tracker.snapshot(work.task_id)
                if record.state not in TERMINAL_STATES:
                    self._tracker.defer(
                        work.task_id,
                        self._config.solve_retry_seconds,
                        f"WORKER_EXCEPTION:{type(exc).__name__}",
                    )
                self._game.log(f"{work.task_id}: worker failed: {exc!r}")
                continue

            record = self._tracker.snapshot(work.task_id)
            if record.state in TERMINAL_STATES:
                continue
            solve = result.solve_result
            if solve.candidate is None:
                if solve.retryable:
                    self._tracker.defer(
                        work.task_id,
                        solve.retry_after_seconds
                        if solve.retry_after_seconds is not None
                        else self._config.solve_retry_seconds,
                        solve.failure_code or "SOLVE_FAILED",
                    )
                else:
                    self._tracker.fail(work.task_id, solve.failure_code or "SOLVE_FAILED")
                continue

            self._tracker.transition(work.task_id, TileState.VERIFYING)
            self._tracker.mark_ready(
                work.task_id,
                solve.candidate,
                result.answer_format,
                result.board,
            )
        return count

    def _dispatch_available(self) -> int:
        capacity = self._pool.capacity
        if capacity <= 0:
            return 0
        records = self._tracker.available(now=self._clock())
        if self._config.task_filter:
            allowed = set(self._config.task_filter)
            records = [record for record in records if record.task_id in allowed]
        ranked = self._priority.rank(records)
        if self._config.max_tiles:
            if not self._limited_task_ids:
                self._limited_task_ids = {
                    record.task_id for record in ranked[: self._config.max_tiles]
                }
            ranked = [
                record for record in ranked if record.task_id in self._limited_task_ids
            ]
        dispatched = 0
        for record in ranked[:capacity]:
            if record.solve_attempts >= self._config.max_solve_attempts:
                self._tracker.fail(record.task_id, "SOLVE_ATTEMPTS_EXHAUSTED")
                continue
            claimed = self._tracker.try_claim_for_fetch(
                record.task_id, now=self._clock()
            )
            if claimed is None:
                continue
            self._pool.submit(
                record.task_id,
                lambda task_id=record.task_id: self._solve_one(task_id),
            )
            dispatched += 1
        return dispatched

    def _solve_one(self, task_id: str) -> _WorkerResult:
        deadline = self._clock() + self._config.task_timeout_seconds
        detail = self._game.task(task_id)
        workdir = Path(self._game.workdir(task_id))
        names = self._game.fetch_files(task_id, detail, workdir)
        try:
            self._tracker.transition(task_id, TileState.SOLVING)
        except InvalidTransition:
            return _WorkerResult(
                solve_result=SolveResult(
                    candidate=None,
                    retryable=False,
                    failure_code="TILE_BECAME_STALE",
                ),
                answer_format=str(detail.get("answer_format") or "exact"),
                board=str(detail.get("board") or ""),
            )

        answer_format = str(detail.get("answer_format") or "exact")
        context = TaskContext(
            task_id=task_id,
            category=str(detail.get("category") or self._tracker.snapshot(task_id).category),
            points=int(detail.get("points") or self._tracker.snapshot(task_id).points),
            prompt=str(detail.get("prompt") or ""),
            answer_format=answer_format,  # type: ignore[arg-type]
            workdir=workdir,
            files=tuple(workdir / name for name in names),
            deadline_monotonic=deadline,
            metadata={
                **detail,
                "rejected_answers": self._tracker.snapshot(task_id).rejected_answers,
            },
        )
        return _WorkerResult(
            solve_result=self._solver.solve(context),
            answer_format=answer_format,
            board=str(detail.get("board") or ""),
        )

    def _process_one_submission(self) -> int:
        ready = self._priority.rank(self._tracker.ready())
        for record in ready:
            candidate = record.candidate
            if candidate is None:
                continue
            validation_error = self._gate.validate(record, candidate)
            if validation_error:
                self._tracker.fail(record.task_id, validation_error)
                self._game.log(f"{record.task_id}: submission rejected: {validation_error}")
                continue
            if self._gate.seconds_until_available() > 0:
                return 0

            self._tracker.transition(record.task_id, TileState.SUBMITTING)
            current = self._tracker.snapshot(record.task_id)
            try:
                decision = self._gate.attempt(
                    current,
                    candidate,
                    is_open=self._is_open_now,
                )
            except self._fatal_error_types:
                raise
            except Exception as exc:  # noqa: BLE001 - preserve candidate via retry
                self._tracker.defer(
                    record.task_id,
                    self._config.error_backoff_seconds,
                    f"SUBMIT_EXCEPTION:{type(exc).__name__}",
                    preserve_candidate=True,
                )
                self._game.log(f"{record.task_id}: submission failed: {exc!r}")
                return 0

            if decision.action == SubmissionAction.SOLVED:
                self._tracker.transition(record.task_id, TileState.SOLVED)
                self._game.log(f"{record.task_id}: correct")
                return 1
            if decision.action == SubmissionAction.DEAD:
                self._tracker.force_dead(record.task_id, decision.reason)
                return 1
            if decision.action in {SubmissionAction.RETRY, SubmissionAction.DEFERRED}:
                if decision.reason == "incorrect":
                    self._tracker.note_incorrect(record.task_id, candidate.value)
                preserve_candidate = decision.reason in {
                    "rate_limited",
                    "locked_out",
                    "GLOBAL_RATE_LIMIT",
                }
                self._tracker.defer(
                    record.task_id,
                    decision.retry_after_seconds or self._config.solve_retry_seconds,
                    decision.reason,
                    preserve_candidate=preserve_candidate,
                )
                return 1
            if decision.action == SubmissionAction.REJECTED:
                self._tracker.fail(record.task_id, decision.reason)
                return 0
            if decision.action == SubmissionAction.FATAL:
                self._tracker.fail(record.task_id, decision.reason)
                raise FatalSubmissionError(
                    f"{record.task_id}: fatal submission result {decision.reason}"
                )
        return 0

    def _is_open_now(self, task_id: str) -> bool:
        board = self._game.board()
        return any(tile.get("id") == task_id for tile in self._game.open_tiles(board))

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("orchestrator is closed")
