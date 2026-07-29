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
