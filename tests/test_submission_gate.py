from __future__ import annotations

import unittest

from contracts import CandidateAnswer
from orchestrator.state import TileRecord, TileState
from orchestrator.submission_gate import (
    SubmissionAction,
    SubmissionGate,
    SubmissionPolicy,
    SubmissionProtocolError,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeGame:
    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def submit(self, task_id: str, answer: str) -> dict:
        self.calls.append((task_id, answer))
        return self.responses.pop(0)


def record(*, answer_format: str = "exact", board: str = "practice") -> TileRecord:
    return TileRecord(
        task_id="PR-A1" if board == "practice" else "A1",
        category="Ancient Scrolls",
        points=100,
        board=board,
        state=TileState.READY,
        answer_format=answer_format,
    )


def candidate(
    value: str = "answer",
    *,
    confidence: float = 0.9,
    evidence: tuple[str, ...] = ("checked",),
    exact: bool = False,
) -> CandidateAnswer:
    return CandidateAnswer(value, confidence, evidence, "test", exact)


class SubmissionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()

    def gate(self, game: FakeGame, **policy_kwargs) -> SubmissionGate:
        return SubmissionGate(
            game,  # type: ignore[arg-type]
            SubmissionPolicy(**policy_kwargs),
            clock=self.clock,
        )

    def test_low_confidence_is_rejected_without_api_call(self) -> None:
        game = FakeGame({"result": "correct"})
        decision = self.gate(game).attempt(
            record(), candidate(confidence=0.2), is_open=lambda _: True
        )
        self.assertEqual(decision.action, SubmissionAction.REJECTED)
        self.assertEqual(game.calls, [])

    def test_exact_tool_value_can_be_evidence_by_construction(self) -> None:
        game = FakeGame({"result": "correct"})
        decision = self.gate(game).attempt(
            record(), candidate(evidence=(), exact=True), is_open=lambda _: True
        )
        self.assertEqual(decision.action, SubmissionAction.SOLVED)

    def test_numeric_and_literal_formats_are_validated(self) -> None:
        gate = self.gate(FakeGame())
        self.assertEqual(
            gate.validate(record(answer_format="numeric"), candidate("not-a-number")),
            "INVALID_NUMERIC_ANSWER",
        )
        self.assertEqual(
            gate.validate(record(answer_format="literal"), candidate("[broken")),
            "INVALID_LITERAL_ANSWER",
        )
        self.assertIsNone(
            gate.validate(record(answer_format="literal"), candidate("true"))
        )
        self.assertIsNone(
            gate.validate(record(answer_format="literal"), candidate("null"))
        )

    def test_known_wrong_answer_is_rejected_without_api_call(self) -> None:
        game = FakeGame({"result": "correct"})
        rejected = record()
        rejected.rejected_answers = ("answer",)
        decision = self.gate(game).attempt(
            rejected, candidate("answer"), is_open=lambda _: True
        )
        self.assertEqual(decision.action, SubmissionAction.REJECTED)
        self.assertEqual(game.calls, [])

    def test_board_recheck_prevents_dead_submission(self) -> None:
        game = FakeGame({"result": "correct"})
        decision = self.gate(game).attempt(
            record(), candidate(), is_open=lambda _: False
        )
        self.assertEqual(decision.action, SubmissionAction.DEAD)
        self.assertEqual(game.calls, [])

    def test_global_rate_limit_defers_second_answer(self) -> None:
        game = FakeGame({"result": "correct"}, {"result": "correct"})
        gate = self.gate(game)
        first = gate.attempt(record(), candidate("one"), is_open=lambda _: True)
        second = gate.attempt(record(), candidate("two"), is_open=lambda _: True)
        self.assertEqual(first.action, SubmissionAction.SOLVED)
        self.assertEqual(second.action, SubmissionAction.DEFERRED)
        self.assertEqual(len(game.calls), 1)
        self.clock.advance(3.1)
        third = gate.attempt(record(), candidate("two"), is_open=lambda _: True)
        self.assertEqual(third.action, SubmissionAction.SOLVED)

    def test_practice_incorrect_uses_ten_second_cooldown(self) -> None:
        decision = self.gate(FakeGame({"result": "incorrect"})).attempt(
            record(board="practice"), candidate(), is_open=lambda _: True
        )
        self.assertEqual(decision.action, SubmissionAction.RETRY)
        self.assertEqual(decision.retry_after_seconds, 10.0)

    def test_scored_incorrect_cooldown_doubles_by_wrong_attempts(self) -> None:
        scored = record(board="main")
        scored.wrong_attempts = 2
        decision = self.gate(FakeGame({"result": "incorrect"})).attempt(
            scored, candidate(), is_open=lambda _: True
        )
        self.assertEqual(decision.retry_after_seconds, 120.0)

    def test_server_retry_value_overrides_fallback(self) -> None:
        decision = self.gate(
            FakeGame({"result": "locked_out", "retry_in": 17})
        ).attempt(record(), candidate(), is_open=lambda _: True)
        self.assertEqual(decision.retry_after_seconds, 17.0)

    def test_forbidden_is_fatal(self) -> None:
        decision = self.gate(FakeGame({"result": "forbidden"})).attempt(
            record(), candidate(), is_open=lambda _: True
        )
        self.assertEqual(decision.action, SubmissionAction.FATAL)

    def test_missing_result_raises_protocol_error(self) -> None:
        with self.assertRaises(SubmissionProtocolError):
            self.gate(FakeGame({"detail": "bad key"})).attempt(
                record(), candidate(), is_open=lambda _: True
            )


if __name__ == "__main__":
    unittest.main()
