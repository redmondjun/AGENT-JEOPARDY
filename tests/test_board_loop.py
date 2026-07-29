from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from contracts import CandidateAnswer, SolveResult, TaskContext
from orchestrator.board_loop import AgentOrchestrator, CycleReport, OrchestratorConfig
from orchestrator.state import TileState
from orchestrator.submission_gate import SubmissionGate, SubmissionPolicy


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeGame:
    def __init__(self, tiles: list[dict], responses: list[dict] | None = None) -> None:
        self.tiles = tiles
        self.responses = list(responses or [])
        self.submit_calls: list[tuple[str, str]] = []
        self.logs: list[str] = []
        self.phase = "practice"
        self._root = Path(tempfile.mkdtemp(prefix="agent-loop-test-"))

    def board(self) -> dict:
        return {"phase": self.phase}

    def open_tiles(self, board=None) -> list[dict]:
        return [dict(tile) for tile in self.tiles]

    def task(self, task_id: str) -> dict:
        tile = next(tile for tile in self.tiles if tile["id"] == task_id)
        return {
            **tile,
            "prompt": f"solve {task_id}",
            "answer_format": tile.get("answer_format", "exact"),
            "board": tile.get("board", "practice"),
            "files": [],
        }

    def workdir(self, task_id: str) -> Path:
        path = self._root / task_id
        path.mkdir(exist_ok=True)
        return path

    def fetch_files(self, task_id: str, detail: dict, dest=None) -> list[str]:
        return []

    def submit(self, task_id: str, answer: str) -> dict:
        self.submit_calls.append((task_id, answer))
        return self.responses.pop(0)

    def log(self, *values: object) -> None:
        self.logs.append(" ".join(map(str, values)))


class AnswerSolver:
    def __init__(self, *, fail_task: str | None = None, confidence: float = 0.95) -> None:
        self.fail_task = fail_task
        self.confidence = confidence

    def solve(self, task: TaskContext) -> SolveResult:
        if task.task_id == self.fail_task:
            raise RuntimeError("boom")
        return SolveResult(
            CandidateAnswer(
                value=f"answer-{task.task_id}",
                confidence=self.confidence,
                evidence=("computed",),
                strategy="fake",
            ),
            retryable=False,
        )


def make_orchestrator(
    game: FakeGame,
    solver: AnswerSolver,
    clock: FakeClock,
    *,
    max_workers: int = 2,
) -> AgentOrchestrator:
    gate = SubmissionGate(
        game,
        SubmissionPolicy(default_minimum_confidence=0.8),
        clock=clock,
    )
    return AgentOrchestrator(
        game,
        solver,
        gate,
        config=OrchestratorConfig(
            max_workers=max_workers,
            poll_interval_seconds=0.01,
            error_backoff_seconds=1.0,
            task_timeout_seconds=10.0,
            solve_retry_seconds=5.0,
        ),
        clock=clock,
    )


class BoardLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()

    def test_end_to_end_correct_submission(self) -> None:
        game = FakeGame(
            [{"id": "PR-A1", "category": "Ancient Scrolls", "points": 100}],
            [{"result": "correct"}],
        )
        agent = make_orchestrator(game, AnswerSolver(), self.clock)
        try:
            first = agent.run_cycle()
            self.assertEqual(first.dispatched, 1)
            agent.drain_workers(timeout=1)
            second = agent.run_cycle()
            self.assertEqual(second.submitted, 1)
            self.assertEqual(agent.tracker.snapshot("PR-A1").state, TileState.SOLVED)
            self.assertEqual(game.submit_calls, [("PR-A1", "answer-PR-A1")])
            combined = "\n".join(game.logs)
            self.assertIn("event=dispatch task=PR-A1 attempt=1/3", combined)
            self.assertIn("event=worker_complete task=PR-A1 outcome=ready", combined)
            self.assertIn("event=submission task=PR-A1 action=attempt", combined)
            self.assertIn("event=submission task=PR-A1 action=solved", combined)
            self.assertNotIn("answer-PR-A1", combined)
        finally:
            agent.close()

    def test_worker_failure_does_not_poison_other_tile(self) -> None:
        game = FakeGame(
            [
                {"id": "PR-A1", "category": "A", "points": 100},
                {"id": "PR-B1", "category": "B", "points": 100},
            ]
        )
        agent = make_orchestrator(game, AnswerSolver(fail_task="PR-A1"), self.clock)
        try:
            agent.run_cycle()
            agent.drain_workers(timeout=1)
            self.assertEqual(agent.tracker.snapshot("PR-A1").state, TileState.COOLDOWN)
            self.assertEqual(agent.tracker.snapshot("PR-B1").state, TileState.READY)
            failure_log = next(
                log
                for log in game.logs
                if "event=worker_complete task=PR-A1 outcome=exception" in log
            )
            self.assertIn("error_type=RuntimeError", failure_log)
            self.assertIn("next_attempt=2", failure_log)
            self.assertNotIn("boom", failure_log)
        finally:
            agent.close()

    def test_low_confidence_candidate_never_submits(self) -> None:
        game = FakeGame(
            [{"id": "PR-A1", "category": "A", "points": 100}],
            [{"result": "correct"}],
        )
        agent = make_orchestrator(game, AnswerSolver(confidence=0.2), self.clock)
        try:
            agent.run_cycle()
            agent.drain_workers(timeout=1)
            agent.run_cycle()
            self.assertEqual(agent.tracker.snapshot("PR-A1").state, TileState.FAILED)
            self.assertEqual(game.submit_calls, [])
            self.assertTrue(
                any(
                    "event=submission task=PR-A1 action=rejected" in log
                    and "reason=LOW_CONFIDENCE" in log
                    for log in game.logs
                )
            )
        finally:
            agent.close()

    def test_claimed_during_solve_is_discarded(self) -> None:
        game = FakeGame(
            [{"id": "PR-A1", "category": "A", "points": 100}],
            [{"result": "correct"}],
        )
        agent = make_orchestrator(game, AnswerSolver(), self.clock)
        try:
            agent.run_cycle()
            agent.drain_workers(timeout=1)
            game.tiles = []
            agent.run_cycle()
            self.assertEqual(agent.tracker.snapshot("PR-A1").state, TileState.DEAD)
            self.assertEqual(game.submit_calls, [])
        finally:
            agent.close()

    def test_submission_gate_serializes_ready_answers(self) -> None:
        game = FakeGame(
            [
                {"id": "PR-A1", "category": "A", "points": 100},
                {"id": "PR-B1", "category": "B", "points": 100},
            ],
            [{"result": "correct"}, {"result": "correct"}],
        )
        agent = make_orchestrator(game, AnswerSolver(), self.clock)
        try:
            agent.run_cycle()
            agent.drain_workers(timeout=1)
            first_submit = agent.run_cycle()
            self.assertEqual(first_submit.submitted, 1)
            self.assertEqual(len(game.submit_calls), 1)
            blocked = agent.run_cycle()
            self.assertEqual(blocked.submitted, 0)
            self.clock.advance(3.1)
            second_submit = agent.run_cycle()
            self.assertEqual(second_submit.submitted, 1)
            self.assertEqual(len(game.submit_calls), 2)
        finally:
            agent.close()

    def test_max_tiles_is_total_selection_not_per_cycle(self) -> None:
        game = FakeGame(
            [
                {"id": "PR-A1", "category": "A", "points": 100},
                {"id": "PR-B1", "category": "B", "points": 100},
            ]
        )
        gate = SubmissionGate(
            game,
            SubmissionPolicy(default_minimum_confidence=0.8),
            clock=self.clock,
        )
        agent = AgentOrchestrator(
            game,
            AnswerSolver(confidence=0.2),
            gate,
            config=OrchestratorConfig(max_workers=2, max_tiles=1),
            clock=self.clock,
        )
        try:
            agent.run_cycle()
            agent.drain_workers(timeout=1)
            agent.run_cycle()
            states = {record.task_id: record.state for record in agent.tracker.snapshots()}
            self.assertEqual(sum(state == TileState.FAILED for state in states.values()), 1)
            self.assertEqual(sum(state == TileState.DISCOVERED for state in states.values()), 1)
        finally:
            agent.close()

    def test_rate_limited_candidate_is_resubmitted_without_resolve(self) -> None:
        game = FakeGame(
            [{"id": "PR-A1", "category": "A", "points": 100}],
            [{"result": "rate_limited", "retry_in": 2}, {"result": "correct"}],
        )
        solver = CountingSolver()
        agent = make_orchestrator(game, solver, self.clock)
        try:
            agent.run_cycle()
            agent.drain_workers(timeout=1)
            agent.run_cycle()
            self.assertEqual(solver.calls, 1)
            self.clock.advance(3.1)
            agent.run_cycle()
            self.assertEqual(solver.calls, 1)
            self.assertEqual(agent.tracker.snapshot("PR-A1").state, TileState.SOLVED)
            self.assertTrue(
                any(
                    "event=submission task=PR-A1 action=retry" in log
                    and "reason=rate_limited" in log
                    and "preserve_candidate=True" in log
                    for log in game.logs
                )
            )
        finally:
            agent.close()

    def test_cycle_log_reports_changes_and_periodic_heartbeat(self) -> None:
        game = FakeGame([])
        agent = make_orchestrator(game, AnswerSolver(), self.clock)
        report = CycleReport(
            phase="practice",
            open_tiles=0,
            discovered=0,
            dispatched=0,
            completed=0,
            submitted=0,
            active_workers=0,
        )
        try:
            agent._log_cycle(report)
            agent._log_cycle(report)
            self.assertEqual(len(game.logs), 1)
            self.assertIn("event=cycle kind=change", game.logs[0])
            self.assertIn("states=none", game.logs[0])

            self.clock.advance(29.9)
            agent._log_cycle(report)
            self.assertEqual(len(game.logs), 1)

            self.clock.advance(0.1)
            agent._log_cycle(report)
            self.assertEqual(len(game.logs), 2)
            self.assertIn("event=cycle kind=heartbeat", game.logs[1])
        finally:
            agent.close()

    def test_phase_change_retires_blocked_practice_workers_and_dispatches_scored(self) -> None:
        release_practice = threading.Event()
        game = FakeGame(
            [
                {
                    "id": "PR-A1",
                    "category": "A",
                    "points": 100,
                    "board": "practice",
                },
                {
                    "id": "PR-B1",
                    "category": "B",
                    "points": 100,
                    "board": "practice",
                },
            ]
        )
        gate = SubmissionGate(
            game,
            SubmissionPolicy(default_minimum_confidence=0.8),
            clock=self.clock,
        )
        agent = AgentOrchestrator(
            game,
            PhaseBlockingSolver(release_practice),
            gate,
            config=OrchestratorConfig(
                max_workers=2,
                max_tiles=2,
                task_timeout_seconds=30.0,
            ),
            clock=self.clock,
        )
        try:
            practice = agent.run_cycle()
            self.assertEqual(practice.dispatched, 2)
            self.assertEqual(practice.active_workers, 2)

            game.phase = "round1"
            game.tiles = [
                {
                    "id": "Q-A1",
                    "category": "A",
                    "points": 100,
                    "board": "qual",
                },
                {
                    "id": "Q-B1",
                    "category": "B",
                    "points": 100,
                    "board": "qual",
                },
            ]
            qualifier = agent.run_cycle()

            # No clock advance or 30-second worker deadline is needed: scored
            # work gets a fresh generation of bounded capacity immediately.
            self.assertEqual(qualifier.dispatched, 2)
            self.assertEqual(qualifier.active_workers, 2)
            self.assertEqual(
                agent.tracker.snapshot("PR-A1").state, TileState.DEAD
            )
            self.assertTrue(
                any("retired 2 stale workers" in log for log in game.logs)
            )
            agent.drain_workers(timeout=1)
            self.assertEqual(agent.tracker.snapshot("Q-A1").state, TileState.READY)
            self.assertEqual(agent.tracker.snapshot("Q-B1").state, TileState.READY)
        finally:
            release_practice.set()
            agent.close(wait_for_workers=False)


class CountingSolver(AnswerSolver):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def solve(self, task: TaskContext) -> SolveResult:
        self.calls += 1
        return super().solve(task)


class PhaseBlockingSolver(AnswerSolver):
    def __init__(self, release_practice: threading.Event) -> None:
        super().__init__()
        self.release_practice = release_practice

    def solve(self, task: TaskContext) -> SolveResult:
        if task.task_id.startswith("PR-"):
            self.release_practice.wait()
        return super().solve(task)


if __name__ == "__main__":
    unittest.main()
