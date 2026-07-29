from __future__ import annotations

import unittest

from orchestrator.priority import Calibration, PriorityPolicy
from orchestrator.state import TileRecord


class PriorityPolicyTests(unittest.TestCase):
    def test_calibration_controls_expected_points_per_second(self) -> None:
        fast = TileRecord("fast", "Data", 100)
        slow = TileRecord("slow", "Web", 500)
        policy = PriorityPolicy(
            {
                ("Data", 100): Calibration(0.9, 10.0),
                ("Web", 500): Calibration(0.2, 100.0),
            }
        )
        self.assertEqual([r.task_id for r in policy.rank([slow, fast])], ["fast", "slow"])

    def test_ties_are_deterministic(self) -> None:
        policy = PriorityPolicy({("Data", 100): Calibration(1.0, 10.0)})
        records = [TileRecord("b", "Data", 100), TileRecord("a", "Data", 100)]
        self.assertEqual([r.task_id for r in policy.rank(records)], ["a", "b"])

    def test_invalid_calibration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Calibration(1.2, 10.0)

    def test_equal_score_prefers_faster_tile(self) -> None:
        fast = TileRecord("fast", "Fast", 100)
        slow = TileRecord("slow", "Slow", 500)
        policy = PriorityPolicy(
            {
                ("Fast", 100): Calibration(1.0, 10.0),
                ("Slow", 500): Calibration(1.0, 50.0),
            }
        )
        self.assertEqual([r.task_id for r in policy.rank([slow, fast])], ["fast", "slow"])


if __name__ == "__main__":
    unittest.main()
