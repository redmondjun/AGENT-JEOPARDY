"""Tests for competition configuration assembled by ``main``."""

from __future__ import annotations

import os
import unittest

# ``jeopardy`` validates configuration at import time.  These inert values let
# us exercise local assembly without contacting the event service.
os.environ.setdefault("JEOPARDY_BASE_URL", "https://example.invalid")
os.environ.setdefault("TEAM_API_KEY", "team_test_only")

import main
from contracts import CandidateAnswer
from orchestrator.state import TileRecord


class FakeGame:
    def submit(self, task_id: str, answer: str) -> dict[str, object]:
        raise AssertionError("configuration tests must not submit")


def record(*, category: str, points: int, board: str = "main") -> TileRecord:
    return TileRecord(
        task_id=f"Q-{points}",
        category=category,
        points=points,
        board=board,
        answer_format="exact",
    )


def candidate(
    confidence: float, *, exact_value_from_tool: bool = False
) -> CandidateAnswer:
    return CandidateAnswer(
        value="verified answer",
        confidence=confidence,
        evidence=("independent verification",),
        strategy="test",
        exact_value_from_tool=exact_value_from_tool,
    )


class MainSubmissionPolicyTests(unittest.TestCase):
    def gate(
        self, environ: dict[str, str] | None = None
    ) -> main.CompetitionSubmissionGate:
        return main.CompetitionSubmissionGate(
            FakeGame(), main.build_submission_policy(environ)
        )

    def test_verified_tier_one_candidate_uses_calibrated_threshold(self) -> None:
        gate = self.gate()
        tile = record(category="Ancient Scrolls", points=100)

        self.assertIsNone(gate.validate(tile, candidate(0.70)))
        self.assertEqual(
            gate.validate(tile, candidate(0.64)),
            "LOW_CONFIDENCE:0.640<0.650",
        )

    def test_higher_tiers_retain_conservative_threshold(self) -> None:
        gate = self.gate()

        for points in (200, 300, 400, 500):
            with self.subTest(points=points):
                tile = record(category="Ancient Scrolls", points=points)
                self.assertEqual(
                    gate.validate(tile, candidate(0.70)),
                    "LOW_CONFIDENCE:0.700<0.800",
                )
                self.assertIsNone(gate.validate(tile, candidate(0.80)))

    def test_tier_one_relaxation_is_scored_phase_only(self) -> None:
        gate = self.gate()

        for board in ("practice", "", "unexpected"):
            with self.subTest(board=board):
                tile = record(
                    category="Ancient Scrolls", points=100, board=board
                )
                self.assertEqual(
                    gate.validate(tile, candidate(0.70)),
                    "LOW_CONFIDENCE:0.700<0.800",
                )

    def test_exact_tool_candidate_remains_eligible_at_higher_tiers(self) -> None:
        gate = self.gate()
        tile = record(category="Heavy Compute", points=500)

        # The solver assigns deterministic exact tool values confidence 0.95.
        self.assertIsNone(
            gate.validate(tile, candidate(0.95, exact_value_from_tool=True))
        )

    def test_unknown_category_does_not_inherit_tier_one_relaxation(self) -> None:
        gate = self.gate()
        tile = record(category="Unexpected Category", points=100)

        self.assertEqual(
            gate.validate(tile, candidate(0.70)),
            "LOW_CONFIDENCE:0.700<0.800",
        )

    def test_thresholds_and_first_tier_are_environment_configurable(self) -> None:
        gate = self.gate(
            {
                "MIN_CONFIDENCE": "0.90",
                "TIER_ONE_MIN_CONFIDENCE": "0.60",
                "TIER_ONE_POINTS": "150",
                "SUBMISSION_INTERVAL_SECONDS": "3.5",
            }
        )

        self.assertIsNone(
            gate.validate(
                record(category="Cryptic", points=150), candidate(0.60)
            )
        )
        self.assertEqual(
            gate.validate(
                record(category="Cryptic", points=100), candidate(0.80)
            ),
            "LOW_CONFIDENCE:0.800<0.900",
        )

    def test_tier_one_threshold_cannot_be_less_conservative_than_default(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "TIER_ONE_MIN_CONFIDENCE cannot exceed MIN_CONFIDENCE"
        ):
            main.build_submission_policy(
                {
                    "MIN_CONFIDENCE": "0.70",
                    "TIER_ONE_MIN_CONFIDENCE": "0.80",
                }
            )


class MainOrchestratorConfigTests(unittest.TestCase):
    def test_qualifier_defaults_use_modest_width_and_longer_deadline(self) -> None:
        config = main.build_orchestrator_config(max_tiles=0, environ={})

        self.assertEqual(config.max_workers, 12)
        self.assertEqual(config.task_timeout_seconds, 180.0)
        self.assertEqual(config.poll_interval_seconds, 2.0)
        self.assertEqual(config.max_solve_attempts, 3)
        self.assertEqual(config.max_tiles, 0)
        self.assertEqual(config.task_filter, ())

    def test_every_orchestration_setting_remains_overridable(self) -> None:
        config = main.build_orchestrator_config(
            max_tiles=11,
            environ={
                "MAX_WORKERS": "3",
                "TASK_TIMEOUT_SECONDS": "45.5",
                "POLL_SECONDS": "0.75",
                "MAX_SOLVE_ATTEMPTS": "5",
                "TASK_FILTER": " Q-A1, ,Q-C2,Q-W3 ",
            },
        )

        self.assertEqual(config.max_workers, 3)
        self.assertEqual(config.task_timeout_seconds, 45.5)
        self.assertEqual(config.poll_interval_seconds, 0.75)
        self.assertEqual(config.max_solve_attempts, 5)
        self.assertEqual(config.max_tiles, 11)
        self.assertEqual(config.task_filter, ("Q-A1", "Q-C2", "Q-W3"))

    def test_invalid_resource_overrides_fail_before_agent_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_workers must be positive"):
            main.build_orchestrator_config(
                max_tiles=0, environ={"MAX_WORKERS": "0"}
            )
        with self.assertRaisesRegex(ValueError, "task timeout must be positive"):
            main.build_orchestrator_config(
                max_tiles=0, environ={"TASK_TIMEOUT_SECONDS": "0"}
            )


if __name__ == "__main__":
    unittest.main()
