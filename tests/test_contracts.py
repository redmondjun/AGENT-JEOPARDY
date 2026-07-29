from __future__ import annotations

import unittest
import math
from pathlib import Path

from contracts import CandidateAnswer, SolveResult, TaskContext, ToolRequest, ToolResult


class ContractTests(unittest.TestCase):
    def test_task_context_rejects_unknown_answer_format(self) -> None:
        with self.assertRaises(ValueError):
            TaskContext(
                task_id="PR-A1",
                category="Ancient Scrolls",
                points=100,
                prompt="prompt",
                answer_format="xml",  # type: ignore[arg-type]
                workdir=Path("/tmp/task"),
                files=(),
                deadline_monotonic=10.0,
            )

    def test_candidate_confidence_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            CandidateAnswer("answer", 1.1, (), "test")

    def test_failed_tool_result_requires_error_code(self) -> None:
        with self.assertRaises(ValueError):
            ToolResult(ok=False, output="")

    def test_successful_tool_result_cannot_have_error_code(self) -> None:
        with self.assertRaises(ValueError):
            ToolResult(ok=True, output="ok", error_code="BAD")

    def test_tool_timeout_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            ToolRequest(name="python", arguments={}, timeout_seconds=0)

    def test_solve_failure_requires_code(self) -> None:
        with self.assertRaises(ValueError):
            SolveResult(candidate=None, retryable=True)

    def test_contract_mappings_are_defensively_copied(self) -> None:
        metadata = {"phase": "practice"}
        task = TaskContext(
            task_id="PR-A1",
            category="A",
            points=100,
            prompt="p",
            answer_format="exact",
            workdir=Path("/tmp/task"),
            files=(),
            deadline_monotonic=10.0,
            metadata=metadata,
        )
        metadata["phase"] = "game"
        self.assertEqual(task.metadata["phase"], "practice")
        with self.assertRaises(TypeError):
            task.metadata["phase"] = "game"  # type: ignore[index]

    def test_success_cannot_also_request_retry(self) -> None:
        answer = CandidateAnswer("x", 0.9, ("e",), "test")
        with self.assertRaises(ValueError):
            SolveResult(candidate=answer, retryable=True)

    def test_non_finite_timeout_is_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ToolRequest(name="python", arguments={}, timeout_seconds=value)


if __name__ == "__main__":
    unittest.main()
