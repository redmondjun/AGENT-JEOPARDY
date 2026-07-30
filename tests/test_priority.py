from __future__ import annotations

import unittest

from orchestrator.priority import Calibration, PriorityPolicy
from orchestrator.state import TileRecord


class PriorityPolicyTests(unittest.TestCase):
    def test_default_calibration_prioritizes_tier_one_before_tier_two(self) -> None:
        tier_one = TileRecord("tier-one", "Data", 100)
        tier_two = TileRecord("tier-two", "Data", 200)
        policy = PriorityPolicy()

        self.assertGreater(policy.score(tier_one), policy.score(tier_two))
        self.assertEqual(
            [r.task_id for r in policy.rank([tier_two, tier_one])],
            ["tier-one", "tier-two"],
        )

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

    def test_comparable_scores_diversify_initial_categories(self) -> None:
        records = [
            TileRecord("data-1", "Data", 100),
            TileRecord("data-2", "Data", 100),
            TileRecord("data-3", "Data", 100),
            TileRecord("web-1", "Web", 100),
        ]
        policy = PriorityPolicy(
            {
                ("Data", 100): Calibration(1.0, 10.0),
                ("Web", 100): Calibration(0.95, 10.0),
            }
        )

        self.assertEqual(
            [r.task_id for r in policy.rank(records)],
            ["data-1", "web-1", "data-2", "data-3"],
        )

    def test_clear_score_advantage_is_not_sacrificed_for_diversity(self) -> None:
        records = [
            TileRecord("fast-1", "Fast", 100),
            TileRecord("fast-2", "Fast", 100),
            TileRecord("slow", "Slow", 100),
        ]
        policy = PriorityPolicy(
            {
                ("Fast", 100): Calibration(1.0, 10.0),
                ("Slow", 100): Calibration(0.5, 10.0),
            }
        )

        self.assertEqual(
            [r.task_id for r in policy.rank(records)],
            ["fast-1", "fast-2", "slow"],
        )

    def test_comparable_category_round_uses_distinct_cells_first(self) -> None:
        records = [
            TileRecord("a-100-1", "A", 100),
            TileRecord("a-100-2", "A", 100),
            TileRecord("a-200", "A", 200),
        ]
        policy = PriorityPolicy(
            {
                ("A", 100): Calibration(0.5, 5.0),
                ("A", 200): Calibration(0.5, 10.0),
            }
        )

        self.assertEqual(
            [r.task_id for r in policy.rank(records)],
            ["a-100-1", "a-200", "a-100-2"],
        )

    def test_diversified_ranking_is_independent_of_input_order(self) -> None:
        records = [
            TileRecord("a-2", "A", 100),
            TileRecord("b-1", "B", 100),
            TileRecord("a-1", "A", 100),
        ]
        policy = PriorityPolicy()

        forward = [r.task_id for r in policy.rank(records)]
        reverse = [r.task_id for r in policy.rank(list(reversed(records)))]

        self.assertEqual(forward, reverse)
        self.assertEqual(forward, ["a-1", "b-1", "a-2"])

    def test_online_observations_update_category_tier_calibration(self) -> None:
        record = TileRecord("a", "Data", 100)
        policy = PriorityPolicy()
        before = policy.calibration_for(record)

        policy.observe(record, correct=False, elapsed_seconds=40.0)
        after = policy.calibration_for(record)

        self.assertLess(after.solve_probability, before.solve_probability)
        self.assertGreater(after.expected_seconds, before.expected_seconds)

    def test_server_discovery_order_breaks_equal_priority_ties(self) -> None:
        first = TileRecord("z", "Data", 100, discovery_order=1)
        second = TileRecord("a", "Data", 100, discovery_order=2)
        self.assertEqual(
            [item.task_id for item in PriorityPolicy().rank([second, first])],
            ["z", "a"],
        )


if __name__ == "__main__":
    unittest.main()
