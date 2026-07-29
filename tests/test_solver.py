from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("JEOPARDY_BASE_URL", "https://example.invalid")
os.environ.setdefault("TEAM_API_KEY", "team_test")
os.environ.setdefault("ANTHROPIC_API_KEY", "team_test")

from solver import SolverConfig, TileSolver, validate_answer  # noqa: E402


@dataclass
class Response:
    content: list[dict]
    stop_reason: str = "tool_use"


class CapturingConversation:
    def __init__(self, direct: bool = False):
        self.calls: list[dict] = []
        self.direct = direct

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            return Response(
                [{"type": "text", "text": '{"approve": true, "reason": "supported"}'}],
                "end_turn",
            )
        tool_calls = [call for call in self.calls if "tools" in call]
        if len(tool_calls) == 1:
            if self.direct:
                return Response(
                    [
                        {
                            "type": "tool_use",
                            "id": "final",
                            "name": "finalize_answer",
                            "input": {
                                "answer": "42",
                                "confidence": 0.95,
                                "evidence": ["derived from the supplied equation"],
                                "verification_notes": "substitution returns equality",
                            },
                        }
                    ]
                )
            return Response(
                [
                    {
                        "type": "tool_use",
                        "id": "run",
                        "name": "run_python",
                        "input": {
                            "code": "print(6 * 7)",
                            "capture_answer": True,
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "list",
                        "name": "list_files",
                        "input": {},
                    },
                ]
            )
        prior_results = kwargs["messages"][-1]["content"]
        self.last_tool_results = prior_results
        run_payload = json.loads(prior_results[0]["content"])
        return Response(
            [
                {
                    "type": "tool_use",
                    "id": "final",
                    "name": "finalize_answer",
                    "input": {
                        "answer_ref": run_payload["answer_ref"],
                        "confidence": 0.9,
                        "evidence": ["Python computed the arithmetic"],
                        "verification_notes": "6 * 7 was executed",
                    },
                }
            ]
        )


class FakeClient:
    def __init__(self, direct: bool = False):
        self.messages = CapturingConversation(direct=direct)


class SolverTests(unittest.TestCase):
    def detail(self) -> dict:
        return {
            "id": "PR-H1",
            "category": "Heavy Compute",
            "points": 100,
            "prompt": "Compute six times seven.",
            "answer_format": "numeric",
        }

    def test_answer_formats(self) -> None:
        self.assertEqual(validate_answer("  hello \n world ", "exact"), "hello world")
        self.assertEqual(validate_answer("$1,200.50", "numeric"), "1200.50")
        self.assertEqual(validate_answer('["a", 2]', "literal"), "['a', 2]")
        self.assertEqual(validate_answer(" witness ", "validator"), "witness")
        with self.assertRaises(ValueError):
            validate_answer("not-a-number", "numeric")

    def test_tool_loop_orders_results_and_uses_answer_ref(self) -> None:
        client = FakeClient()
        solver = TileSolver(
            SolverConfig(max_turns=3),
            client_factory=lambda: client,
        )
        with tempfile.TemporaryDirectory() as directory:
            outcome = solver.solve(
                self.detail(), Path(directory), "practice", set()
            )
        self.assertIsNotNone(outcome.candidate)
        self.assertEqual(outcome.candidate.value, "42")
        self.assertTrue(outcome.candidate.exact_value_from_tool)
        results = client.messages.last_tool_results
        self.assertEqual(
            [item["tool_use_id"] for item in results], ["run", "list"]
        )

    def test_scored_direct_answer_requires_and_passes_review(self) -> None:
        client = FakeClient(direct=True)
        solver = TileSolver(
            SolverConfig(max_turns=2),
            client_factory=lambda: client,
        )
        with tempfile.TemporaryDirectory() as directory:
            outcome = solver.solve(
                self.detail(), Path(directory), "game", set()
            )
        self.assertEqual(outcome.candidate.value, "42")
        self.assertTrue(any("tools" not in call for call in client.messages.calls))

    def test_scored_reviewer_rejection_fails_closed(self) -> None:
        client = FakeClient(direct=True)

        def rejecting_create(**kwargs):
            client.messages.calls.append(kwargs)
            if "tools" not in kwargs:
                return Response(
                    [
                        {
                            "type": "text",
                            "text": '{"approve": false, "reason": "insufficient proof"}',
                        }
                    ],
                    "end_turn",
                )
            return Response(
                [
                    {
                        "type": "tool_use",
                        "id": f"final-{len(client.messages.calls)}",
                        "name": "finalize_answer",
                        "input": {
                            "answer": "42",
                            "confidence": 0.95,
                            "evidence": ["unsupported claim"],
                            "verification_notes": "not independently checked",
                        },
                    }
                ]
            )

        client.messages.create = rejecting_create
        solver = TileSolver(
            SolverConfig(max_turns=2),
            client_factory=lambda: client,
        )
        with tempfile.TemporaryDirectory() as directory:
            outcome = solver.solve(
                self.detail(), Path(directory), "game", set()
            )
        self.assertIsNone(outcome.candidate)
        self.assertEqual(outcome.failure_code, "turn_limit")

    def test_turn_limit_without_tool_call(self) -> None:
        class NoToolMessages:
            def create(self, **_kwargs):
                return Response([{"type": "text", "text": "guess"}], "end_turn")

        client = type("Client", (), {"messages": NoToolMessages()})()
        solver = TileSolver(
            SolverConfig(max_turns=2),
            client_factory=lambda: client,
        )
        with tempfile.TemporaryDirectory() as directory:
            outcome = solver.solve(
                self.detail(), Path(directory), "practice", set()
            )
        self.assertIsNone(outcome.candidate)
        self.assertEqual(outcome.failure_code, "turn_limit")


if __name__ == "__main__":
    unittest.main()
