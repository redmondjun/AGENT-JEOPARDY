import time

from contracts import ToolResult
from solver.agent_loop import SolverEngine, _compact_history
from solver.registry import ToolRegistry, ToolSchema
from tests.solver.fakes import (
    FakeTool,
    ScriptedModelClient,
    make_task,
    text_response,
    tool_call_response,
)


def _empty_registry() -> ToolRegistry:
    return ToolRegistry()


def test_history_compaction_keeps_tool_use_and_result_as_an_atomic_pair():
    initial = {"role": "user", "content": "task"}
    messages = [initial]
    for index in range(7):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": f"call_{index}",
                            "name": "lookup",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"call_{index}",
                            "content": "ok",
                        }
                    ],
                },
            ]
        )

    compacted = _compact_history(messages)

    assert compacted[0] is initial
    assert compacted[1]["role"] == "assistant"
    assert compacted[1]["content"][0]["type"] == "tool_use"
    assert compacted[2]["role"] == "user"
    assert compacted[2]["content"][0]["type"] == "tool_result"


def test_history_compaction_preserves_every_multi_tool_exchange():
    messages = [{"role": "user", "content": "task"}]
    for index in range(8):
        call_ids = (f"call_{index}_a", f"call_{index}_b")
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": call_id, "name": "lookup"}
                        for call_id in call_ids
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": "ok",
                        }
                        for call_id in call_ids
                    ],
                },
            ]
        )

    compacted = _compact_history(messages)

    assert compacted[0] is messages[0]
    assert len(compacted) <= 12
    for assistant, user in zip(compacted[1::2], compacted[2::2]):
        tool_use_ids = {
            block["id"]
            for block in assistant["content"]
            if block["type"] == "tool_use"
        }
        tool_result_ids = {
            block["tool_use_id"]
            for block in user["content"]
            if block["type"] == "tool_result"
        }
        assert tool_use_ids == tool_result_ids


def test_model_can_request_multiple_tools_over_multiple_turns():
    tool = FakeTool("lookup", result=ToolResult(ok=True, output="row 7: banana"))
    registry = ToolRegistry()
    registry.register(tool, ToolSchema("lookup", "look something up", {"type": "object"}))

    client = ScriptedModelClient(
        responses=[
            tool_call_response("lookup", {"query": "fruit"}),
            tool_call_response("lookup", {"query": "fruit row 2"}),
            text_response("Based on the rows.\nFINAL_ANSWER: banana"),
        ]
    )

    engine = SolverEngine(client, registry, max_turns=5)
    result = engine.solve(make_task(answer_format="literal"))

    assert client.calls == 3
    assert len(tool.calls) == 2
    assert result.candidate is not None
    assert result.candidate.value == "banana"
    assert result.retryable is False


def test_unknown_tool_name_does_not_crash_the_loop():
    registry = ToolRegistry()  # no tools registered at all
    client = ScriptedModelClient(
        responses=[
            tool_call_response("nonexistent_tool", {}),
            text_response("FINAL_ANSWER: fallback"),
        ]
    )
    engine = SolverEngine(client, registry, max_turns=5)

    result = engine.solve(make_task(answer_format="literal"))  # must not raise

    assert result.candidate is not None
    assert result.candidate.value == "fallback"


def test_turn_budget_exhaustion_is_typed_and_retryable():
    registry = _empty_registry()
    # Model never emits FINAL_ANSWER and never calls a tool -> loop should
    # bail after the very first turn via NO_ACTIONABLE_OUTPUT, so to
    # actually exercise TURN_BUDGET_EXHAUSTED we make it call an (unknown)
    # tool every turn, which keeps the loop going without ever answering.
    client = ScriptedModelClient(
        responses=[tool_call_response("noop", {}) for _ in range(3)]
    )
    engine = SolverEngine(client, registry, max_turns=3)

    result = engine.solve(make_task())

    assert result.candidate is None
    assert result.retryable is True
    assert result.failure_code == "TURN_BUDGET_EXHAUSTED"


def test_no_actionable_output_is_typed_and_retryable():
    registry = _empty_registry()
    client = ScriptedModelClient(responses=[text_response("I am thinking out loud.")])
    engine = SolverEngine(client, registry, max_turns=5)

    result = engine.solve(make_task())

    assert result.candidate is None
    assert result.retryable is True
    assert result.failure_code == "NO_ACTIONABLE_OUTPUT"


def test_token_budget_exhaustion_is_typed_and_retryable():
    registry = _empty_registry()
    client = ScriptedModelClient(
        responses=[
            text_response("still going", tokens=10_000),
            text_response("FINAL_ANSWER: too_late", tokens=10_000),
        ]
    )
    engine = SolverEngine(client, registry, max_turns=5, max_total_tokens=15_000)

    result = engine.solve(make_task())

    assert result.candidate is None
    assert result.retryable is True
    assert result.failure_code == "TOKEN_BUDGET_EXHAUSTED"


def test_deadline_exceeded_before_first_turn_is_typed_and_retryable():
    registry = _empty_registry()
    client = ScriptedModelClient(responses=[])
    engine = SolverEngine(client, registry)

    past_deadline_task = make_task(deadline_monotonic=time.monotonic() - 1.0)
    result = engine.solve(past_deadline_task)

    assert result.candidate is None
    assert result.retryable is True
    assert result.failure_code == "DEADLINE_EXCEEDED"
    assert client.calls == 0  # must not even attempt a model call past deadline


def test_blocked_model_call_returns_at_deadline_as_typed_retryable_failure():
    class BlockingModelClient:
        def create_turn(self, **_kwargs):
            time.sleep(1.0)
            return text_response("FINAL_ANSWER: too late")

    logs: list[str] = []
    engine = SolverEngine(BlockingModelClient(), _empty_registry(), logger=logs.append)
    started = time.monotonic()

    result = engine.solve(
        make_task(deadline_monotonic=time.monotonic() + 0.05)
    )

    assert time.monotonic() - started < 0.5
    assert result.candidate is None
    assert result.retryable is True
    assert result.failure_code == "MODEL_CALL_TIMEOUT"
    assert any("failed=MODEL_CALL_TIMEOUT" in message for message in logs)


def test_model_api_exception_is_typed_retryable_and_logged_without_message():
    secret_marker = "SENSITIVE_MARKER_DO_NOT_LOG"

    class ExplodingModelClient:
        def create_turn(self, **_kwargs):
            raise RuntimeError(f"Authorization: Bearer {secret_marker}")

    logs: list[str] = []
    engine = SolverEngine(ExplodingModelClient(), _empty_registry(), logger=logs.append)

    result = engine.solve(make_task())

    assert result.candidate is None
    assert result.retryable is True
    assert result.failure_code == "MODEL_API_ERROR"
    joined_logs = "\n".join(logs)
    assert "exception=RuntimeError" in joined_logs
    assert secret_marker not in joined_logs
    assert "Authorization" not in joined_logs


def test_tool_timeout_is_clamped_to_remaining_task_deadline():
    tool = FakeTool("lookup", result=ToolResult(ok=True, output="found"))
    registry = ToolRegistry()
    registry.register(tool, ToolSchema("lookup", "lookup", {"type": "object"}))
    client = ScriptedModelClient(
        responses=[
            tool_call_response("lookup", {}),
            text_response("FINAL_ANSWER: found"),
        ]
    )
    engine = SolverEngine(client, registry, tool_timeout_seconds=20.0)

    result = engine.solve(
        make_task(deadline_monotonic=time.monotonic() + 2.0)
    )

    assert result.candidate is not None
    assert len(tool.calls) == 1
    assert 0 < tool.calls[0].timeout_seconds <= 2.0


def test_blocked_tool_call_returns_at_deadline_as_typed_retryable_failure():
    class BlockingTool:
        name = "blocked"

        def execute(self, _request, _task):
            time.sleep(1.0)
            return ToolResult(ok=True, output="too late")

    registry = ToolRegistry()
    registry.register(
        BlockingTool(),
        ToolSchema("blocked", "block forever", {"type": "object"}),
    )
    client = ScriptedModelClient(
        responses=[tool_call_response("blocked", {})]
    )
    logs: list[str] = []
    engine = SolverEngine(client, registry, logger=logs.append)
    started = time.monotonic()

    result = engine.solve(
        make_task(deadline_monotonic=time.monotonic() + 0.05)
    )

    assert time.monotonic() - started < 0.5
    assert result.candidate is None
    assert result.retryable is True
    assert result.failure_code == "TOOL_CALL_TIMEOUT"
    assert any("error=TOOL_CALL_TIMEOUT" in message for message in logs)


def test_exact_value_from_tool_bypasses_model_retyping():
    tool = FakeTool(
        "calc",
        result=ToolResult(ok=True, output="computed 3.14159265", exact_value="3.14159265"),
    )
    registry = ToolRegistry()
    registry.register(tool, ToolSchema("calc", "compute something", {"type": "object"}))

    client = ScriptedModelClient(
        responses=[
            tool_call_response("calc", {}),
            # Model mistypes the value in its own final-answer text —
            # exact_value pass-through must win anyway.
            text_response("FINAL_ANSWER: 3.14159"),
        ]
    )

    engine = SolverEngine(client, registry, max_turns=5)
    result = engine.solve(make_task(answer_format="numeric"))

    assert result.candidate is not None
    assert result.candidate.value == "3.14159265"
    assert result.candidate.exact_value_from_tool is True
