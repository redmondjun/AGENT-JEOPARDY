"""Practice-calibrated tile prioritization."""

from __future__ import annotations

import math
from dataclasses import dataclass
import threading
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
# Practice cleared tier 1 at 7/7 in seconds, while qualifier tier-2 solves
# took roughly 12-35 seconds. Keep the probability estimate conservative, but
# use the observed tier-1 latency so the default EPS order harvests fast wins.
_DEFAULT_SECONDS = {100: 12.0, 200: 28.0, 300: 38.0, 400: 52.0, 500: 70.0}
_COMPARABLE_SCORE_RATIO = 0.90


class PriorityPolicy:
    """Rank by expected points/second, diversifying only comparable work.

    Tiles within 10% of the best remaining score form a cohort. A cohort is
    interleaved by category, then by category/points cell, so an initial worker
    wave does not duplicate one solution strategy when equally valuable work
    exists. Cohorts never mix, preserving clear expected-value advantages.
    """

    def __init__(
        self,
        calibrations: Mapping[tuple[str, int], Calibration] | None = None,
        category_weights: Mapping[str, float] | None = None,
    ) -> None:
        self._calibrations = dict(calibrations or {})
        self._category_weights = dict(category_weights or {})
        self._lock = threading.RLock()
        self._observations: dict[tuple[str, int], _AdaptiveCalibration] = {}
        if any(
            not math.isfinite(weight) or weight < 0
            for weight in self._category_weights.values()
        ):
            raise ValueError("category weights must be finite and non-negative")

    def calibration_for(self, record: TileRecord) -> Calibration:
        key = (record.category, record.points)
        with self._lock:
            adaptive = self._observations.get(key)
            if adaptive is not None:
                return adaptive.value
            calibration = self._calibrations.get(key)
            if calibration is not None:
                return calibration
            return Calibration(
                solve_probability=_DEFAULT_PROBABILITY.get(record.points, 0.25),
                expected_seconds=_DEFAULT_SECONDS.get(record.points, 60.0),
            )

    def observe(self, record: TileRecord, *, correct: bool, elapsed_seconds: float) -> None:
        """Update category/tier priors without letting one result dominate."""
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            return
        key = (record.category, record.points)
        with self._lock:
            adaptive = self._observations.get(key)
            if adaptive is None:
                prior = self.calibration_for(record)
                adaptive = _AdaptiveCalibration.from_prior(prior)
                self._observations[key] = adaptive
            adaptive.observe(correct=correct, elapsed_seconds=elapsed_seconds)

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
        ranked = sorted(
            records,
            key=self._base_rank_key,
        )
        diversified: list[TileRecord] = []
        start = 0
        while start < len(ranked):
            best_score = self.score(ranked[start])
            end = start + 1
            while end < len(ranked) and self._is_comparable(
                best_score, self.score(ranked[end])
            ):
                end += 1
            diversified.extend(self._diversify_cohort(ranked[start:end]))
            start = end
        return diversified

    def _base_rank_key(
        self, record: TileRecord
    ) -> tuple[float, float, int, int, str]:
        return (
            -self.score(record),
            self.calibration_for(record).expected_seconds,
            -record.points,
            record.discovery_order,
            record.task_id,
        )

    @staticmethod
    def _is_comparable(best_score: float, candidate_score: float) -> bool:
        if best_score <= 0:
            return candidate_score == best_score
        return candidate_score >= best_score * _COMPARABLE_SCORE_RATIO

    @staticmethod
    def _diversify_cohort(records: Sequence[TileRecord]) -> list[TileRecord]:
        """Balance category and cell use while retaining base-rank tie breaks."""
        remaining = list(enumerate(records))
        category_uses: dict[str, int] = {}
        cell_uses: dict[tuple[str, int], int] = {}
        result: list[TileRecord] = []
        while remaining:
            index, record = min(
                remaining,
                key=lambda item: (
                    category_uses.get(item[1].category, 0),
                    cell_uses.get((item[1].category, item[1].points), 0),
                    item[0],
                ),
            )
            remaining.remove((index, record))
            result.append(record)
            category_uses[record.category] = category_uses.get(record.category, 0) + 1
            cell = (record.category, record.points)
            cell_uses[cell] = cell_uses.get(cell, 0) + 1
        return result


@dataclass
class _AdaptiveCalibration:
    alpha: float
    beta: float
    expected_seconds: float

    @classmethod
    def from_prior(cls, prior: Calibration) -> "_AdaptiveCalibration":
        prior_weight = 4.0
        return cls(
            alpha=max(0.01, prior.solve_probability * prior_weight),
            beta=max(0.01, (1.0 - prior.solve_probability) * prior_weight),
            expected_seconds=prior.expected_seconds,
        )

    @property
    def value(self) -> Calibration:
        return Calibration(
            solve_probability=self.alpha / (self.alpha + self.beta),
            expected_seconds=max(0.1, self.expected_seconds),
        )

    def observe(self, *, correct: bool, elapsed_seconds: float) -> None:
        if correct:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        # A conservative EWMA reacts during a round without thrashing after
        # one unusually fast or slow tile.
        self.expected_seconds = 0.75 * self.expected_seconds + 0.25 * elapsed_seconds
