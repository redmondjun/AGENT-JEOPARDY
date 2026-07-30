from __future__ import annotations

import unittest

from orchestrator.state import TileRecord
from orchestrator.telemetry import ScoreTracker


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class ScoreTrackerTests(unittest.TestCase):
    def test_scored_correct_and_penalty_are_aggregated(self) -> None:
        clock = FakeClock()
        tracker = ScoreTracker(clock=clock)
        tracker.observe_phase("round1")
        tile = TileRecord("Q-A2", "A", 200, board="qual")

        tracker.record_correct(tile, {})
        tracker.record_incorrect(tile)
        clock.value += 60
        result = tracker.snapshot()

        self.assertEqual(result.earned_points, 200)
        self.assertEqual(result.penalty_points, 50)
        self.assertEqual(result.net_points, 150)
        self.assertEqual(result.net_points_per_minute, 150)

    def test_practice_does_not_change_points(self) -> None:
        tracker = ScoreTracker()
        tile = TileRecord("PR-A1", "A", 100, board="practice")
        tracker.record_correct(tile, {})
        tracker.record_incorrect(tile)
        result = tracker.snapshot()
        self.assertEqual(result.net_points, 0)
