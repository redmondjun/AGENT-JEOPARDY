"""Practice-calibrated tile prioritization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .state import TileRecord


@dataclass(frozen=True)
class Calibration:
    solve_probability: float
    expected_seconds: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.solve_probability <= 1.0:
            raise ValueError("solve_probability must be between 0.0 and 1.0")
        if not math.isfinite(self.expected_seconds) or self.expected_seconds <= 0:
            raise ValueError("expected_seconds must be positive")


_DEFAULT_PROBABILITY = {100: 0.90, 200: 0.75, 300: 0.55, 400: 0.35, 500: 0.20}
_DEFAULT_SECONDS = {100: 20.0, 200: 28.0, 300: 38.0, 400: 52.0, 500: 70.0}


class PriorityPolicy:
    """Ranks by expected points per second with deterministic tie-breaking."""

    def __init__(
        self,
        calibrations: Mapping[tuple[str, int], Calibration] | None = None,
        category_weights: Mapping[str, float] | None = None,
    ) -> None:
        self._calibrations = dict(calibrations or {})
        self._category_weights = dict(category_weights or {})
        if any(
            not math.isfinite(weight) or weight < 0
            for weight in self._category_weights.values()
        ):
            raise ValueError("category weights must be finite and non-negative")

    def calibration_for(self, record: TileRecord) -> Calibration:
        calibration = self._calibrations.get((record.category, record.points))
        if calibration is not None:
            return calibration
        return Calibration(
            solve_probability=_DEFAULT_PROBABILITY.get(record.points, 0.25),
            expected_seconds=_DEFAULT_SECONDS.get(record.points, 60.0),
        )

    def score(self, record: TileRecord) -> float:
        calibration = self.calibration_for(record)
        weight = self._category_weights.get(record.category, 1.0)
        return (
            record.points
            * calibration.solve_probability
            * max(weight, 0.0)
            / calibration.expected_seconds
        )

    def rank(self, records: Sequence[TileRecord]) -> list[TileRecord]:
        return sorted(
            records,
            key=lambda record: (
                -self.score(record),
                self.calibration_for(record).expected_seconds,
                -record.points,
                record.task_id,
            ),
        )
