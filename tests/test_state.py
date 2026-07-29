from __future__ import annotations

import threading
import unittest

from contracts import CandidateAnswer
from orchestrator.state import InvalidTransition, TileState, TileTracker


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TileTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.tracker = TileTracker(clock=self.clock)
        self.tracker.observe_open_tiles(
            [{"id": "PR-A1", "category": "Ancient Scrolls", "points": 100}]
        )

    def test_complete_ready_transition(self) -> None:
        self.tracker.transition("PR-A1", TileState.QUEUED)
        fetching = self.tracker.transition("PR-A1", TileState.FETCHING)
        self.assertEqual(fetching.solve_attempts, 1)
        self.tracker.transition("PR-A1", TileState.SOLVING)
        self.tracker.transition("PR-A1", TileState.VERIFYING)
        candidate = CandidateAnswer("42", 0.9, ("computed",), "test")
        ready = self.tracker.mark_ready("PR-A1", candidate, "numeric", "practice")
        self.assertEqual(ready.state, TileState.READY)
        self.assertEqual(ready.candidate, candidate)
        self.assertEqual(ready.answer_format, "numeric")

    def test_invalid_transition_raises(self) -> None:
        with self.assertRaises(InvalidTransition):
            self.tracker.transition("PR-A1", TileState.SUBMITTING)

    def test_disappeared_tile_becomes_dead(self) -> None:
        self.tracker.observe_open_tiles([])
        record = self.tracker.snapshot("PR-A1")
        self.assertEqual(record.state, TileState.DEAD)
        self.assertEqual(record.last_error, "NO_LONGER_OPEN")

    def test_cooldown_becomes_available_at_deadline(self) -> None:
        self.tracker.transition("PR-A1", TileState.QUEUED)
        self.tracker.transition("PR-A1", TileState.FETCHING)
        self.tracker.defer("PR-A1", 10.0, "retry")
        self.assertEqual(self.tracker.available(), [])
        self.clock.advance(10.0)
        self.assertEqual([r.task_id for r in self.tracker.available()], ["PR-A1"])

    def test_incorrect_attempts_are_distinct_from_api_calls(self) -> None:
        self.tracker.transition("PR-A1", TileState.QUEUED)
        self.tracker.transition("PR-A1", TileState.FETCHING)
        self.tracker.transition("PR-A1", TileState.SOLVING)
        self.tracker.transition("PR-A1", TileState.VERIFYING)
        self.tracker.mark_ready(
            "PR-A1",
            CandidateAnswer("x", 0.9, ("e",), "test"),
            "exact",
            "practice",
        )
        submitted = self.tracker.transition("PR-A1", TileState.SUBMITTING)
        self.assertEqual(submitted.submission_attempts, 1)
        self.assertEqual(submitted.wrong_attempts, 0)
        wrong = self.tracker.note_incorrect("PR-A1", "x")
        self.assertEqual(wrong.wrong_attempts, 1)
        self.assertEqual(wrong.rejected_answers, ("x",))

    def test_atomic_claim_allows_only_one_worker(self) -> None:
        barrier = threading.Barrier(20)
        successes: list[object] = []
        lock = threading.Lock()

        def claim() -> None:
            barrier.wait()
            result = self.tracker.try_claim_for_fetch("PR-A1")
            if result is not None:
                with lock:
                    successes.append(result)

        threads = [threading.Thread(target=claim) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(successes), 1)
        self.assertEqual(self.tracker.snapshot("PR-A1").solve_attempts, 1)

    def test_submission_cooldown_preserves_candidate_without_resolve(self) -> None:
        claimed = self.tracker.try_claim_for_fetch("PR-A1")
        self.assertIsNotNone(claimed)
        self.tracker.transition("PR-A1", TileState.SOLVING)
        self.tracker.transition("PR-A1", TileState.VERIFYING)
        candidate = CandidateAnswer("token", 0.9, (), "tool", True)
        self.tracker.mark_ready("PR-A1", candidate, "exact", "practice")
        self.tracker.transition("PR-A1", TileState.SUBMITTING)
        self.tracker.defer(
            "PR-A1", 4.0, "rate_limited", preserve_candidate=True
        )
        self.assertEqual(self.tracker.available(), [])
        self.clock.advance(4.0)
        self.assertEqual(self.tracker.release_submission_cooldowns(), 1)
        ready = self.tracker.ready()[0]
        self.assertEqual(ready.candidate, candidate)
        self.assertEqual(ready.solve_attempts, 1)

    def test_revive_failed_resets_only_solve_budget_and_preserves_penalty_history(
        self,
    ) -> None:
        self.assertIsNotNone(self.tracker.try_claim_for_fetch("PR-A1"))
        self.tracker.transition("PR-A1", TileState.SOLVING)
        self.tracker.transition("PR-A1", TileState.VERIFYING)
        self.tracker.mark_ready(
            "PR-A1",
            CandidateAnswer("penalized", 0.9, ("evidence",), "test"),
            "exact",
            "practice",
        )
        self.tracker.transition("PR-A1", TileState.SUBMITTING)
        self.tracker.note_incorrect("PR-A1", "penalized")
        self.tracker.fail("PR-A1", "SOLVE_ATTEMPTS_EXHAUSTED")

        revived = self.tracker.revive_failed("PR-A1")

        self.assertIsNotNone(revived)
        assert revived is not None
        self.assertEqual(revived.state, TileState.DISCOVERED)
        self.assertEqual(revived.solve_attempts, 0)
        self.assertEqual(revived.submission_attempts, 1)
        self.assertEqual(revived.wrong_attempts, 1)
        self.assertEqual(revived.rejected_answers, ("penalized",))
        self.assertIsNone(revived.candidate)
        self.assertEqual(revived.last_error, "REVIVED_AFTER_IDLE")
        self.assertNotIn("PR-A1", self.tracker.failed_task_ids())
        self.assertIsNone(self.tracker.revive_failed("PR-A1"))


if __name__ == "__main__":
    unittest.main()
