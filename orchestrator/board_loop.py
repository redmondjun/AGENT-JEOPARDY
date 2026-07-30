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
from .prefetch import TaskDetailPrefetcher
from .scheduler import BoundedWorkerPool
from .state import InvalidTransition, TERMINAL_STATES, TileRecord, TileState, TileTracker
from .submission_gate import SubmissionAction, SubmissionDecision, SubmissionGate
from .telemetry import ScoreTracker


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
    heartbeat_interval_seconds: float = 30.0
    task_prefetch_enabled: bool = True
    prefetch_workers: int = 2
    prefetch_lookahead: int = 12
    prefetch_timeout_seconds: float = 10.0
    prefetch_join_seconds: float = 0.25

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
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        if self.prefetch_workers <= 0:
            raise ValueError("prefetch workers must be positive")
        if self.prefetch_lookahead < 0:
            raise ValueError("prefetch lookahead must be non-negative")
        if self.prefetch_timeout_seconds <= 0:
            raise ValueError("prefetch timeout must be positive")
        if self.prefetch_join_seconds < 0:
            raise ValueError("prefetch join must be non-negative")


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
        self._prefetcher = (
            TaskDetailPrefetcher(
                self._game.task,
                max_workers=self._config.prefetch_workers,
                timeout_seconds=self._config.prefetch_timeout_seconds,
                join_seconds=self._config.prefetch_join_seconds,
                logger=self._game.log,
            )
            if self._config.task_prefetch_enabled
            and self._config.prefetch_lookahead > 0
            else None
        )
        self._limited_task_ids: set[str] = set()
        self._open_tile_ids: set[str] | None = None
        self._cancel_events: dict[str, threading.Event] = {}
        self._phase: str | None = None
        self._closed = False
        self._score = ScoreTracker(clock=clock)
        self._last_cycle_fingerprint: tuple[object, ...] | None = None
        self._last_cycle_log_at: float | None = None

    @property
    def tracker(self) -> TileTracker:
        return self._tracker

    def run_cycle(self) -> CycleReport:
        self._ensure_open()
        completed = self._collect_completed()
        board = self._game.board()
        phase = str(board.get("phase") or "unknown")
        open_tiles = self._game.open_tiles(board)
        self._score.observe_phase(phase)
        self._score.set_visible_open_points(
            sum(max(0, int(tile.get("points") or 0)) for tile in open_tiles)
        )
        discovered = self._tracker.observe_open_tiles(open_tiles)
        active_ids = {str(tile["id"]) for tile in open_tiles}
        # Publish this cycle's snapshot for the pre-submit recheck so the
        # latency-critical submission path does not re-fetch a board we just
        # read. See _is_open_now.
        self._open_tile_ids = active_ids
        if self._prefetcher is not None:
            self._prefetcher.retain(active_ids)
        stale_running = [
            task_id
            for task_id, cancel_event in self._cancel_events.items()
            if task_id not in active_ids and not cancel_event.is_set()
        ]
        for task_id in stale_running:
            self._cancel_events[task_id].set()
            self._game.log(
                f"event=cancel task={task_id} reason=no_longer_open"
            )
        phase_changed = self._phase is not None and phase != self._phase
        # Normal board churn must not free capacity while an unkillable model
        # call is still consuming tokens. That hidden over-concurrency pushed
        # live usage above the 95k/minute proxy limit. A true phase transition
        # is the exception: the new scored board gets one fresh generation.
        retired = self._pool.retire_except(active_ids) if phase_changed else ()
        if retired:
            for task_id in retired:
                self._cancel_events.pop(task_id, None)
            self._game.log(
                f"retired {len(retired)} stale workers: {','.join(retired)}"
            )
        if phase_changed:
            # MAX_TILES is a per-board sampling limit. A practice selection
            # must not prevent qualifier/finale ids from being considered.
            self._limited_task_ids.clear()
            self._game.log(f"phase changed: {self._phase} -> {phase}")
        self._phase = phase
        self._tracker.release_submission_cooldowns(now=self._clock())
        submitted = self._process_one_submission()
        dispatched = self._dispatch_available()
        if dispatched == 0 and self._revive_failed_if_idle(active_ids):
            dispatched = self._dispatch_available()
        self._schedule_prefetch()
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
                    self._log_cycle(report)
                    gate_wait = self._gate.seconds_until_available()
                    if self._tracker.ready() and gate_wait > 0:
                        delay = min(delay, gate_wait)
                except self._fatal_error_types:
                    raise
                except FatalSubmissionError:
                    raise
                except Exception as exc:  # noqa: BLE001 - supervisor boundary
                    self._game.log(
                        f"event=orchestrator_error error_type={type(exc).__name__} "
                        f"message_chars={len(str(exc))}"
                    )
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
            for cancel_event in self._cancel_events.values():
                cancel_event.set()
            self._pool.close(wait=wait_for_workers)
            self._cancel_events.clear()
            if self._prefetcher is not None:
                self._prefetcher.close()

    def _collect_completed(self) -> int:
        count = 0
        for work in self._pool.completed():
            count += 1
            self._cancel_events.pop(work.task_id, None)
            try:
                result = work.future.result()
            except self._fatal_error_types:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate one tile
                record = self._tracker.snapshot(work.task_id)
                if record.state in TERMINAL_STATES:
                    self._game.log(
                        f"event=worker_complete task={work.task_id} "
                        f"outcome=stale_exception elapsed_ms={work.elapsed_ms} "
                        f"state={record.state.value} error_type={type(exc).__name__}"
                    )
                    continue
                self._tracker.defer(
                    work.task_id,
                    self._config.solve_retry_seconds,
                    f"WORKER_EXCEPTION:{type(exc).__name__}",
                )
                updated = self._tracker.snapshot(work.task_id)
                self._game.log(
                    f"event=worker_complete task={work.task_id} outcome=exception "
                    f"elapsed_ms={work.elapsed_ms} error_type={type(exc).__name__} "
                    f"reason={updated.last_error or type(exc).__name__} "
                    f"retry_in_seconds={self._config.solve_retry_seconds:g} "
                    f"next_attempt={updated.solve_attempts + 1}"
                )
                continue

            record = self._tracker.snapshot(work.task_id)
            if record.state in TERMINAL_STATES:
                self._game.log(
                    f"event=worker_complete task={work.task_id} outcome=stale "
                    f"elapsed_ms={work.elapsed_ms} state={record.state.value}"
                )
                continue
            solve = result.solve_result
            self._tracker.note_solve_elapsed(work.task_id, work.elapsed_ms)
            if solve.candidate is None:
                if solve.retryable:
                    retry_delay = (
                        solve.retry_after_seconds
                        if solve.retry_after_seconds is not None
                        else self._config.solve_retry_seconds
                    )
                    self._tracker.defer(
                        work.task_id,
                        retry_delay,
                        solve.failure_code or "SOLVE_FAILED",
                    )
                    self._game.log(
                        f"event=worker_complete task={work.task_id} outcome=retry "
                        f"elapsed_ms={work.elapsed_ms} "
                        f"reason={solve.failure_code or 'SOLVE_FAILED'} "
                        f"retry_in_seconds={retry_delay:g} preserve_candidate=False "
                        f"next_attempt={record.solve_attempts + 1}"
                    )
                else:
                    self._tracker.fail(work.task_id, solve.failure_code or "SOLVE_FAILED")
                    self._priority.observe(
                        record,
                        correct=False,
                        elapsed_seconds=max(0.001, work.elapsed_ms / 1000.0),
                    )
                    self._game.log(
                        f"event=worker_complete task={work.task_id} outcome=failed "
                        f"elapsed_ms={work.elapsed_ms} "
                        f"reason={solve.failure_code or 'SOLVE_FAILED'} "
                        f"attempt={record.solve_attempts}"
                    )
                continue

            self._tracker.transition(work.task_id, TileState.VERIFYING)
            self._tracker.mark_ready(
                work.task_id,
                solve.candidate,
                result.answer_format,
                result.board,
            )
            self._game.log(
                f"event=worker_complete task={work.task_id} outcome=ready "
                f"elapsed_ms={work.elapsed_ms} attempt={record.solve_attempts} "
                f"confidence={solve.candidate.confidence:.3f} "
                f"evidence_count={len(solve.candidate.evidence)} "
                f"strategy={solve.candidate.strategy!r} "
                f"model_turns={solve.telemetry.model_turns} "
                f"input_tokens={solve.telemetry.input_tokens} "
                f"output_tokens={solve.telemetry.output_tokens} "
                f"tool_calls={','.join(solve.telemetry.tool_calls) or 'none'}"
            )
        return count

    def _revive_failed_if_idle(self, open_task_ids: set[str]) -> int:
        """Resurrect attempt-exhausted tiles when workers would otherwise idle.

        Idle workers late in a scored round are donated points: a tile that
        burned its solve budget is still worth another shot when nothing else
        is dispatchable, no worker is active, and no verified candidate is
        waiting. Only tiles still open on the live board are revived, at most
        one worker-pool's width per cycle, and only in unrestricted runs so
        MAX_TILES/TASK_FILTER sampling stays deterministic.
        """
        if self._config.task_filter or self._config.max_tiles:
            return 0
        if self._pool.active_count > 0 or self._tracker.ready():
            return 0
        if self._tracker.available(now=self._clock()):
            return 0
        revivable = self._tracker.failed_task_ids() & open_task_ids
        if not revivable:
            return 0
        now = self._clock()
        rested = [
            record
            for record in (
                self._tracker.snapshot(task_id) for task_id in sorted(revivable)
            )
            # A freshly failed tile rests one retry interval before revival so
            # an instantly failing solver cannot hot-loop on the same tile.
            if now - record.updated_at >= self._config.solve_retry_seconds
        ]
        if not rested:
            return 0
        ranked = self._priority.rank(rested)
        revived = 0
        for record in ranked[: self._pool.capacity]:
            if self._tracker.revive_failed(record.task_id) is None:
                continue
            self._game.log(
                f"event=revive task={record.task_id} reason=idle_capacity "
                f"category={record.category!r} points={record.points} "
                f"wrong_attempts={record.wrong_attempts}"
            )
            revived += 1
        return revived

    def _dispatch_available(self) -> int:
        capacity = self._pool.capacity
        if capacity <= 0:
            return 0
        ranked = self._ranked_available()
        dispatched = 0
        for record in ranked[:capacity]:
            if record.solve_attempts >= self._config.max_solve_attempts:
                self._tracker.fail(record.task_id, "SOLVE_ATTEMPTS_EXHAUSTED")
                self._game.log(
                    f"event=dispatch task={record.task_id} outcome=failed "
                    f"reason=SOLVE_ATTEMPTS_EXHAUSTED "
                    f"attempts={record.solve_attempts}/{self._config.max_solve_attempts}"
                )
                continue
            claimed = self._tracker.try_claim_for_fetch(
                record.task_id, now=self._clock()
            )
            if claimed is None:
                continue
            cancel_event = threading.Event()
            self._cancel_events[record.task_id] = cancel_event
            self._pool.submit(
                record.task_id,
                lambda task_id=record.task_id, event=cancel_event: self._solve_one(
                    task_id, event
                ),
            )
            self._game.log(
                f"event=dispatch task={record.task_id} "
                f"attempt={claimed.solve_attempts}/{self._config.max_solve_attempts} "
                f"category={record.category!r} points={record.points} "
                f"worker_timeout_seconds={self._config.task_timeout_seconds:g}"
            )
            dispatched += 1
        return dispatched

    def _ranked_available(self) -> list[TileRecord]:
        """One source of truth for solver and prefetch eligibility."""
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
        return ranked

    def _schedule_prefetch(self) -> int:
        if self._prefetcher is None:
            return 0
        ranked = self._ranked_available()
        return self._prefetcher.schedule(
            [
                record.task_id
                for record in ranked[: self._config.prefetch_lookahead]
            ]
        )

    def _solve_one(
        self, task_id: str, cancel_event: threading.Event
    ) -> _WorkerResult:
        deadline = self._clock() + self._config.task_timeout_seconds
        detail_started = time.monotonic()
        if self._prefetcher is not None:
            detail, detail_source = self._prefetcher.get(task_id)
        else:
            detail = self._game.task(task_id)
            detail_source = "network"
        detail_elapsed_ms = int((time.monotonic() - detail_started) * 1000)
        self._game.log(
            f"event=task_detail task={task_id} source={detail_source} "
            f"elapsed_ms={detail_elapsed_ms}"
        )
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

        record = self._tracker.snapshot(task_id)
        answer_format = str(detail.get("answer_format") or "exact")
        context = TaskContext(
            task_id=task_id,
            category=str(detail.get("category") or record.category),
            points=int(detail.get("points") or record.points),
            prompt=str(detail.get("prompt") or ""),
            answer_format=answer_format,  # type: ignore[arg-type]
            workdir=workdir,
            files=tuple(workdir / name for name in names),
            deadline_monotonic=deadline,
            metadata={
                **detail,
                "rejected_answers": record.rejected_answers,
                "cancel_event": cancel_event,
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
                if (
                    validation_error.startswith("LOW_CONFIDENCE:")
                    and record.solve_attempts < self._config.max_solve_attempts
                ):
                    self._tracker.defer(
                        record.task_id,
                        self._config.solve_retry_seconds,
                        validation_error,
                        preserve_candidate=False,
                    )
                    self._game.log(
                        f"event=submission task={record.task_id} action=retry "
                        f"reason={validation_error} "
                        f"retry_in_seconds={self._config.solve_retry_seconds:g} "
                        f"preserve_candidate=False "
                        f"solve_attempt={record.solve_attempts} "
                        f"next_attempt={record.solve_attempts + 1} "
                        f"confidence={candidate.confidence:.3f} "
                        f"wrong_attempts={record.wrong_attempts}"
                    )
                    continue
                self._tracker.fail(record.task_id, validation_error)
                self._game.log(
                    f"event=submission task={record.task_id} action=rejected "
                    f"reason={validation_error} attempt={record.submission_attempts} "
                    f"solve_attempt={record.solve_attempts} "
                    f"confidence={candidate.confidence:.3f} "
                    f"wrong_attempts={record.wrong_attempts}"
                )
                continue
            if self._gate.seconds_until_available() > 0:
                return 0

            self._tracker.transition(record.task_id, TileState.SUBMITTING)
            current = self._tracker.snapshot(record.task_id)
            self._game.log(
                f"event=submission task={record.task_id} action=attempt "
                f"attempt={current.submission_attempts} "
                f"confidence={candidate.confidence:.3f} "
                f"format={current.answer_format} wrong_attempts={current.wrong_attempts}"
            )
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
                self._game.log(
                    f"event=submission task={record.task_id} action=retry "
                    f"reason=SUBMIT_EXCEPTION:{type(exc).__name__} "
                    f"retry_in_seconds={self._config.error_backoff_seconds:g} "
                    f"preserve_candidate=True attempt={current.submission_attempts}"
                )
                return 0

            if decision.action == SubmissionAction.SOLVED:
                self._priority.observe(
                    current,
                    correct=True,
                    elapsed_seconds=max(
                        0.001, current.last_solve_elapsed_ms / 1000.0
                    ),
                )
                self._score.record_correct(current, decision.raw)
                self._tracker.transition(record.task_id, TileState.SOLVED)
                self._log_submission_decision(current, decision, preserve_candidate=False)
                self._log_score()
                return 1
            if decision.action == SubmissionAction.DEAD:
                self._tracker.force_dead(record.task_id, decision.reason)
                self._log_submission_decision(current, decision, preserve_candidate=False)
                return 1
            if decision.action in {SubmissionAction.RETRY, SubmissionAction.DEFERRED}:
                if decision.reason == "incorrect":
                    self._tracker.note_incorrect(record.task_id, candidate.value)
                    self._priority.observe(
                        current,
                        correct=False,
                        elapsed_seconds=max(
                            0.001, current.last_solve_elapsed_ms / 1000.0
                        ),
                    )
                    self._score.record_incorrect(current)
                preserve_candidate = decision.reason in {
                    "rate_limited",
                    "locked_out",
                    "GLOBAL_RATE_LIMIT",
                }
                retry_delay = (
                    decision.retry_after_seconds or self._config.solve_retry_seconds
                )
                self._tracker.defer(
                    record.task_id,
                    retry_delay,
                    decision.reason,
                    preserve_candidate=preserve_candidate,
                )
                updated = self._tracker.snapshot(record.task_id)
                self._log_submission_decision(
                    updated,
                    decision,
                    preserve_candidate=preserve_candidate,
                    retry_in_seconds=retry_delay,
                )
                if decision.reason == "incorrect":
                    self._log_score()
                return 1
            if decision.action == SubmissionAction.REJECTED:
                self._tracker.fail(record.task_id, decision.reason)
                self._log_submission_decision(current, decision, preserve_candidate=False)
                return 0
            if decision.action == SubmissionAction.FATAL:
                self._tracker.fail(record.task_id, decision.reason)
                self._log_submission_decision(current, decision, preserve_candidate=False)
                raise FatalSubmissionError(
                    f"{record.task_id}: fatal submission result {decision.reason}"
                )
        return 0

    def _log_submission_decision(
        self,
        record: TileRecord,
        decision: SubmissionDecision,
        *,
        preserve_candidate: bool,
        retry_in_seconds: float | None = None,
    ) -> None:
        retry = (
            retry_in_seconds
            if retry_in_seconds is not None
            else decision.retry_after_seconds
        )
        retry_text = f"{retry:g}" if retry is not None else "none"
        self._game.log(
            f"event=submission task={record.task_id} action={decision.action.value} "
            f"reason={decision.reason} attempt={record.submission_attempts} "
            f"retry_in_seconds={retry_text} "
            f"preserve_candidate={preserve_candidate} "
            f"wrong_attempts={record.wrong_attempts}"
        )

    def _log_score(self) -> None:
        score = self._score.snapshot()
        penalty_ratio = (
            score.penalty_points / score.earned_points
            if score.earned_points
            else 0.0
        )
        self._game.log(
            f"event=score correct={score.correct_tiles} "
            f"incorrect={score.incorrect_tiles} earned={score.earned_points} "
            f"penalties={score.penalty_points} net={score.net_points} "
            f"pace_per_minute={score.net_points_per_minute:.1f} "
            f"penalty_ratio={penalty_ratio:.4f} "
            f"visible_open_points={score.visible_open_points}"
        )

    def _log_cycle(self, report: CycleReport) -> None:
        state_counts = {state: 0 for state in TileState}
        for record in self._tracker.snapshots():
            state_counts[record.state] += 1
        ordered_states = (
            TileState.QUEUED,
            TileState.FETCHING,
            TileState.SOLVING,
            TileState.VERIFYING,
            TileState.READY,
            TileState.COOLDOWN,
            TileState.SOLVED,
            TileState.DEAD,
            TileState.FAILED,
        )
        counts = tuple(state_counts[state] for state in ordered_states)
        activity = (
            report.discovered,
            report.dispatched,
            report.completed,
            report.submitted,
        )
        fingerprint = (
            report.phase,
            report.open_tiles,
            report.active_workers,
            counts,
            activity,
        )
        now = self._clock()
        heartbeat_due = (
            self._last_cycle_log_at is None
            or now - self._last_cycle_log_at >= self._config.heartbeat_interval_seconds
        )
        if fingerprint == self._last_cycle_fingerprint and not heartbeat_due:
            return
        kind = "heartbeat" if fingerprint == self._last_cycle_fingerprint else "change"
        state_text = ",".join(
            f"{state.value}:{count}"
            for state, count in zip(ordered_states, counts)
            if count
        ) or "none"
        prefetch_text = "disabled"
        if self._prefetcher is not None:
            stats = self._prefetcher.snapshot()
            prefetch_text = (
                f"scheduled:{stats.scheduled},completed:{stats.completed},"
                f"hits:{stats.cache_hits},joined:{stats.joined_hits},"
                f"misses:{stats.misses},failures:{stats.failures}"
            )
        self._game.log(
            f"event=cycle kind={kind} phase={report.phase} open={report.open_tiles} "
            f"active={report.active_workers} discovered={report.discovered} "
            f"dispatched={report.dispatched} completed={report.completed} "
            f"submitted={report.submitted} states={state_text} "
            f"prefetch={prefetch_text}"
        )
        self._last_cycle_fingerprint = fingerprint
        self._last_cycle_log_at = now

    def _is_open_now(self, task_id: str) -> bool:
        """Last-moment claim check, served from this cycle's board snapshot.

        This runs inside the submission critical path, which is bounded by the
        one-submission-per-3-seconds team limit. Re-fetching /api/board here
        added a serialized round trip (30s timeout) to the one code path where
        latency is literally score — first correct answer takes the tile — even
        though run_cycle read the same board moments earlier.

        The snapshot is at most poll_interval_seconds stale, and a stale
        positive is cheap: submitting to a tile another team just took returns
        ``already_claimed``, which carries no point penalty and is handled as a
        normal game outcome. Spending a submission slot on that is a far better
        trade than delaying every submission we make. Falls back to a live
        fetch if no cycle has published a snapshot yet.
        """
        if self._open_tile_ids is None:
            board = self._game.board()
            return any(
                tile.get("id") == task_id
                for tile in self._game.open_tiles(board)
            )
        return task_id in self._open_tile_ids

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("orchestrator is closed")
