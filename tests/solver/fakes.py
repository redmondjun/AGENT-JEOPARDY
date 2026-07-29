"""
Fake Tool and ModelClient implementations for solver unit tests. No
network, no real Anthropic SDK — this is what lets Sara's acceptance
tests (TEAM_PLAN.md section 8) run standalone before Jun/Vidula's real
tools or the event proxy exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from contracts import TaskContext, ToolRequest, ToolResult
from solver.agent_loop import ModelResponse, ModelToolCall


def make_task(
    *,
    task_id: str = "t1",
    category: str = "Needle in the Haystack",
    answer_format: str = "exact",
    prompt: str = "What is the answer?",
    deadline_monotonic: float = 1_000_000_000.0,
) -> TaskContext:
    return TaskContext(
        task_id=task_id,
        category=category,
        points=100,
        prompt=prompt,
        answer_format=answer_format,  # type: ignore[arg-type]
        workdir=Path("/tmp"),
        files=(),
        deadline_monotonic=deadline_monotonic,
    )


class FakeTool:
    """A tool whose behavior is scripted per test rather than doing real work."""

    def __init__(self, name: str, result: ToolResult | None = None, raise_exc: Exception | None = None):
        self.name = name
        self._result = result or ToolResult(ok=True, output="fake output")
        self._raise_exc = raise_exc
        self.calls: list[ToolRequest] = []

    def execute(self, request: ToolRequest, task: TaskContext) -> ToolResult:
        self.calls.append(request)
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._result


@dataclass
class ScriptedModelClient:
    """
    Returns one ModelResponse per call to create_turn, taken in order from
    `responses`. Raises AssertionError if called more times than scripted
    — that mismatch usually means the loop isn't terminating when it
    should.
    """

    responses: list[ModelResponse] = field(default_factory=list)
    calls: int = field(default=0, init=False)

    def create_turn(self, *, system, messages, tools, max_tokens) -> ModelResponse:
        if self.calls >= len(self.responses):
            raise AssertionError(
                f"ScriptedModelClient exhausted after {self.calls} calls — "
                "loop did not terminate when the script expected it to."
            )
        response = self.responses[self.calls]
        self.calls += 1
        return response


def text_response(text: str, tokens: int = 10) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=(),
        input_tokens=tokens,
        output_tokens=tokens,
        raw_content=[{"type": "text", "text": text}],
    )


def tool_call_response(name: str, arguments: dict, call_id: str = "call_1", tokens: int = 10) -> ModelResponse:
    return ModelResponse(
        text="",
        tool_calls=(ModelToolCall(id=call_id, name=name, arguments=arguments),),
        input_tokens=tokens,
        output_tokens=tokens,
        raw_content=[{"type": "tool_use", "id": call_id, "name": name, "input": arguments}],
    )
