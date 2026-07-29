"""Thread-safe tile lifecycle state for the long-running agent."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping

from contracts import CandidateAnswer


class TileState(str, Enum):
    DISCOVERED = "discovered"
    QUEUED = "queued"
    FETCHING = "fetching"
    SOLVING = "solving"
    VERIFYING = "verifying"
    READY = "ready"
    SUBMITTING = "submitting"
    COOLDOWN = "cooldown"
    SOLVED = "solved"
    DEAD = "dead"
    FAILED = "failed"


TERMINAL_STATES = frozenset({TileState.SOLVED, TileState.DEAD, TileState.FAILED})

_ALLOWED_TRANSITIONS: Mapping[TileState, frozenset[TileState]] = {
    TileState.DISCOVERED: frozenset({TileState.QUEUED, TileState.DEAD}),
    TileState.QUEUED: frozenset({TileState.FETCHING, TileState.COOLDOWN, TileState.DEAD}),
    TileState.FETCHING: frozenset(
        {TileState.SOLVING, TileState.COOLDOWN, TileState.FAILED, TileState.DEAD}
    ),
    TileState.SOLVING: frozenset(
        {TileState.VERIFYING, TileState.COOLDOWN, TileState.FAILED, TileState.DEAD}
    ),
    TileState.VERIFYING: frozenset(
        {TileState.READY, TileState.COOLDOWN, TileState.FAILED, TileState.DEAD}
    ),
    TileState.READY: frozenset(
        {TileState.SUBMITTING, TileState.COOLDOWN, TileState.FAILED, TileState.DEAD}
    ),
    TileState.SUBMITTING: frozenset(
        {TileState.SOLVED, TileState.COOLDOWN, TileState.FAILED, TileState.DEAD}
    ),
    TileState.COOLDOWN: frozenset({TileState.QUEUED, TileState.DEAD, TileState.FAILED}),
    TileState.SOLVED: frozenset(),
    TileState.DEAD: frozenset(),
    TileState.FAILED: frozenset(),
}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class TileRecord:
    task_id: str
    category: str
    points: int
    board: str = ""
    state: TileState = TileState.DISCOVERED
    available_at: float = 0.0
    solve_attempts: int = 0
    submission_attempts: int = 0
    wrong_attempts: int = 0
    candidate: CandidateAnswer | None = None
    answer_format: str = "exact"
    rejected_answers: tuple[str, ...] = ()
    last_error: str | None = None
    updated_at: float = 0.0


class TileTracker:
    """Owns all mutable tile state behind one re-entrant lock."""

    def __init__(self, *, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._records: dict[str, TileRecord] = {}

    def observe_open_tiles(self, tiles: Iterable[dict]) -> int:
        """Upsert open tiles and mark disappeared non-terminal work as dead."""
        now = self._clock()
        open_ids: set[str] = set()
        created = 0
        with self._lock:
            for tile in tiles:
                task_id = str(tile["id"])
                open_ids.add(task_id)
                record = self._records.get(task_id)
                if record is None:
                    self._records[task_id] = TileRecord(
                        task_id=task_id,
                        category=str(tile.get("category") or "Unknown"),
                        points=int(tile.get("points") or 0),
                        board=str(tile.get("board") or ""),
                        updated_at=now,
                    )
                    created += 1
                elif record.state not in TERMINAL_STATES:
                    record.category = str(tile.get("category") or record.category)
                    record.points = int(tile.get("points") or record.points)
                    record.board = str(tile.get("board") or record.board)
                    record.updated_at = now

            for record in self._records.values():
                if record.task_id not in open_ids and record.state not in TERMINAL_STATES:
                    record.state = TileState.DEAD
                    record.candidate = None
                    record.last_error = "NO_LONGER_OPEN"
                    record.updated_at = now
        return created

    def transition(
        self,
        task_id: str,
        target: TileState,
        *,
        error: str | None = None,
    ) -> TileRecord:
        with self._lock:
            record = self._require(task_id)
            if record.state == target:
                raise InvalidTransition(
                    f"{task_id}: duplicate transition to {target.value}"
                )
            if target not in _ALLOWED_TRANSITIONS[record.state]:
                raise InvalidTransition(
                    f"{task_id}: cannot transition {record.state.value} -> {target.value}"
                )
            record.state = target
            record.last_error = error
            record.updated_at = self._clock()
            if target == TileState.FETCHING:
                record.solve_attempts += 1
            if target == TileState.SUBMITTING:
                record.submission_attempts += 1
            if target in TERMINAL_STATES:
                record.candidate = None
            return replace(record)

    def mark_ready(
        self,
        task_id: str,
        candidate: CandidateAnswer,
        answer_format: str,
        board: str,
    ) -> TileRecord:
        with self._lock:
            record = self._require(task_id)
            if record.state != TileState.VERIFYING:
                raise InvalidTransition(
                    f"{task_id}: candidate can only be stored while verifying"
                )
            record.candidate = candidate
            record.answer_format = answer_format
            record.board = board or record.board
            record.state = TileState.READY
            record.last_error = None
            record.updated_at = self._clock()
            return replace(record)

    def defer(
        self,
        task_id: str,
        delay_seconds: float,
        error: str,
        *,
        preserve_candidate: bool = False,
    ) -> TileRecord:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        with self._lock:
            record = self._require(task_id)
            if record.state in TERMINAL_STATES:
                return replace(record)
            if TileState.COOLDOWN not in _ALLOWED_TRANSITIONS[record.state]:
                raise InvalidTransition(
                    f"{task_id}: cannot defer from {record.state.value}"
                )
            record.state = TileState.COOLDOWN
            record.available_at = self._clock() + delay_seconds
            if not preserve_candidate:
                record.candidate = None
            record.last_error = error
            record.updated_at = self._clock()
            return replace(record)

    def release_submission_cooldowns(self, *, now: float | None = None) -> int:
        current = self._clock() if now is None else now
        released = 0
        with self._lock:
            for record in self._records.values():
                if (
                    record.state == TileState.COOLDOWN
                    and record.candidate is not None
                    and record.available_at <= current
                ):
                    record.state = TileState.READY
                    record.last_error = None
                    record.updated_at = self._clock()
                    released += 1
        return released

    def try_claim_for_fetch(
        self, task_id: str, *, now: float | None = None
    ) -> TileRecord | None:
        current = self._clock() if now is None else now
        with self._lock:
            record = self._require(task_id)
            claimable = record.state == TileState.DISCOVERED or (
                record.state == TileState.COOLDOWN
                and record.candidate is None
                and record.available_at <= current
            )
            if not claimable:
                return None
            record.state = TileState.FETCHING
            record.solve_attempts += 1
            record.last_error = None
            record.updated_at = self._clock()
            return replace(record)

    def note_incorrect(self, task_id: str, answer: str) -> TileRecord:
        with self._lock:
            record = self._require(task_id)
            record.wrong_attempts += 1
            if answer not in record.rejected_answers:
                record.rejected_answers = (*record.rejected_answers, answer)
            record.updated_at = self._clock()
            return replace(record)

    def fail(self, task_id: str, error: str) -> TileRecord:
        return self.transition(task_id, TileState.FAILED, error=error)

    def revive_failed(self, task_id: str) -> TileRecord | None:
        """Give a FAILED tile a fresh solve budget.

        FAILED stays terminal for every generic transition; only this explicit
        idle-capacity path may resurrect a tile. Wrong-answer history and
        rejected answers survive so a revived tile can never resubmit an
        already penalized answer.
        """
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.state != TileState.FAILED:
                return None
            record.state = TileState.DISCOVERED
            record.solve_attempts = 0
            record.candidate = None
            record.last_error = "REVIVED_AFTER_IDLE"
            record.updated_at = self._clock()
            return replace(record)

    def failed_task_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(
                record.task_id
                for record in self._records.values()
                if record.state == TileState.FAILED
            )

    def force_dead(self, task_id: str, reason: str) -> TileRecord:
        with self._lock:
            record = self._require(task_id)
            if record.state not in TERMINAL_STATES:
                record.state = TileState.DEAD
                record.candidate = None
                record.last_error = reason
                record.updated_at = self._clock()
            return replace(record)

    def available(self, *, now: float | None = None) -> list[TileRecord]:
        current = self._clock() if now is None else now
        with self._lock:
            return [
                replace(record)
                for record in self._records.values()
                if record.state == TileState.DISCOVERED
                or (
                    record.state == TileState.COOLDOWN
                    and record.candidate is None
                    and record.available_at <= current
                )
            ]

    def ready(self) -> list[TileRecord]:
        with self._lock:
            return [
                replace(record)
                for record in self._records.values()
                if record.state == TileState.READY and record.candidate is not None
            ]

    def snapshot(self, task_id: str) -> TileRecord:
        with self._lock:
            return replace(self._require(task_id))

    def snapshots(self) -> list[TileRecord]:
        with self._lock:
            return [replace(record) for record in self._records.values()]

    def _require(self, task_id: str) -> TileRecord:
        try:
            return self._records[task_id]
        except KeyError:
            raise KeyError(f"unknown tile: {task_id}") from None
