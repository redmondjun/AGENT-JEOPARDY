"""Competition-safe aggregate score and pace telemetry."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Mapping

from .state import TileRecord

_SCORED_BOARDS = frozenset({"qual", "main"})


@dataclass(frozen=True)
class ScoreSnapshot:
    correct_tiles: int = 0
    incorrect_tiles: int = 0
    earned_points: int = 0
    penalty_points: int = 0
    net_points: int = 0
    elapsed_seconds: float = 0.0
    net_points_per_minute: float = 0.0
    visible_open_points: int = 0


class ScoreTracker:
    def __init__(self, *, clock=time.monotonic) -> None:
        self._clock = clock
        self._started_at: float | None = None
        self._snapshot = ScoreSnapshot()

    def observe_phase(self, phase: str) -> None:
        if phase in {"round1", "game"} and self._started_at is None:
            self._started_at = self._clock()

    def set_visible_open_points(self, value: int) -> None:
        self._snapshot = replace(
            self._snapshot, visible_open_points=max(0, value)
        )

    def record_correct(
        self, record: TileRecord, raw: Mapping[str, object]
    ) -> ScoreSnapshot:
        awarded = _awarded_points(raw, record.points)
        scored = record.board in _SCORED_BOARDS
        earned = self._snapshot.earned_points + (awarded if scored else 0)
        self._snapshot = replace(
            self._snapshot,
            correct_tiles=self._snapshot.correct_tiles + 1,
            earned_points=earned,
        )
        return self.snapshot()

    def record_incorrect(self, record: TileRecord) -> ScoreSnapshot:
        scored = record.board in _SCORED_BOARDS
        penalty = round(record.points * 0.25) if scored else 0
        self._snapshot = replace(
            self._snapshot,
            incorrect_tiles=self._snapshot.incorrect_tiles + 1,
            penalty_points=self._snapshot.penalty_points + penalty,
        )
        return self.snapshot()

    def snapshot(self) -> ScoreSnapshot:
        elapsed = (
            0.0
            if self._started_at is None
            else max(0.0, self._clock() - self._started_at)
        )
        net = self._snapshot.earned_points - self._snapshot.penalty_points
        pace = net / (elapsed / 60.0) if elapsed > 0 else 0.0
        return replace(
            self._snapshot,
            net_points=net,
            elapsed_seconds=elapsed,
            net_points_per_minute=pace,
        )


def _awarded_points(raw: Mapping[str, object], fallback: int) -> int:
    for key in ("awarded_points", "points_awarded"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return fallback * 2 if raw.get("daily_double") is True else fallback
