from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("JEOPARDY_BASE_URL", "https://example.invalid")
os.environ.setdefault("TEAM_API_KEY", "team_test")
os.environ.setdefault("ANTHROPIC_API_KEY", "team_test")

from main import (  # noqa: E402
    CompetitiveAgent,
    Config,
    TaskState,
)
from tools import CandidateAnswer  # noqa: E402


def board(ids: list[str], phase: str = "practice") -> dict:
    return {
        "phase": phase,
        "you": {"solved_ids": []},
        "boards": {
            "practice": [
                {
                    "name": "Cryptic",
                    "tiles": [
                        {
                            "id": ids[0] if ids else "gone",
                            "open_ids": ids,
                            "remaining": len(ids),
                            "total": 2,
                            "points": 500,
                            "locked": False,
                        }
                    ],
                }
            ]
        },
    }


class FakeClock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


class MainTests(unittest.IsolatedAsyncioTestCase):
    def make_agent(self, directory: str, clock: FakeClock) -> CompetitiveAgent:
        config = Config(
            verbose=False,
            task_filter=(),
            max_tiles=0,
            workers=2,
            max_turns=2,
            tile_timeout_seconds=10,
            poll_seconds=0.25,
            submit_interval_seconds=3.2,
            calibration_path=Path(directory) / "calibration.json",
        )
        return CompetitiveAgent(config=config, monotonic=clock)

    async def test_reconcile_admits_every_variant_and_marks_claimed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory, FakeClock())
            agent.reconcile(board(["A", "B"]))
            self.assertEqual(set(agent.records), {"A", "B"})
            self.assertTrue(agent.records["A"].leading_variant)
            self.assertFalse(agent.records["B"].leading_variant)
            agent.reconcile(board(["B"]))
            self.assertEqual(agent.records["A"].state, TaskState.TERMINAL)
            self.assertNotEqual(agent.records["B"].state, TaskState.TERMINAL)

    async def test_inactive_phase_does_not_schedule_stale_board(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory, FakeClock())
            agent.reconcile(board(["A"], phase="ended"))
            self.assertEqual(agent.latest_open_ids, set())
            self.assertEqual(agent.records, {})

    async def test_incorrect_candidate_cools_then_requeues(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory, clock)
            current = board(["A"])
            agent.reconcile(current)
            record = agent.records["A"]
            record.state = TaskState.CANDIDATE
            record.candidate = CandidateAnswer(
                "wrong", 0.9, ("evidence",), "checked"
            )
            with patch("main.jp.board", return_value=current), patch(
                "main.jp.submit",
                return_value={"result": "incorrect", "retry_in": 10},
            ):
                await agent.submit_ready()
            self.assertEqual(record.state, TaskState.COOLDOWN)
            self.assertIn("wrong", record.incorrect_answers)
            clock.value += 10
            agent.reconcile(current)
            self.assertEqual(record.state, TaskState.QUEUED)

    async def test_submission_gate_enforces_interval(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory, clock)
            current = board(["A", "B"])
            agent.reconcile(current)
            for identifier in ("A", "B"):
                record = agent.records[identifier]
                record.state = TaskState.CANDIDATE
                record.candidate = CandidateAnswer(
                    identifier, 0.9, ("evidence",), "checked", True
                )
            with patch("main.jp.board", return_value=current), patch(
                "main.jp.submit", return_value={"result": "correct"}
            ) as submit:
                await agent.submit_ready()
                await agent.submit_ready()
                self.assertEqual(submit.call_count, 1)
                clock.value += 3.2
                await agent.submit_ready()
                self.assertEqual(submit.call_count, 2)

    async def test_priority_penalizes_incorrect_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self.make_agent(directory, FakeClock())
            agent.reconcile(board(["A", "B"]))
            before = agent.priority(agent.records["B"])
            agent.records["B"].incorrect_answers.add("bad")
            self.assertLess(agent.priority(agent.records["B"]), before)


if __name__ == "__main__":
    unittest.main()
