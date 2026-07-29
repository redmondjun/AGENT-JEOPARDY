import time
from dataclasses import replace
from pathlib import Path

from contracts import ToolResult
from solver.agent_loop import SolverEngine, _compact_history, _initial_user_content
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


def test_initial_prompt_uses_task_relative_file_paths_for_runtime_tools():
    workdir = Path("/tmp/jeopardy_task")
    task = replace(
        make_task(),
        workdir=workdir,
        files=(workdir / "inputs" / "records.csv",),
    )

    prompt = _initial_user_content(task)

    assert "Attached files:\n- inputs/records.csv" in prompt
    assert f"- {workdir}/inputs/records.csv" not in prompt


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


def test_model_and_budget_logs_include_cumulative_diagnostics():
    logs: list[str] = []
    client = ScriptedModelClient(
        responses=[text_response("do-not-log-me", tokens=8_000)]
    )
    engine = SolverEngine(
        client,
        _empty_registry(),
        max_total_tokens=15_000,
        logger=logs.append,
    )

    result = engine.solve(make_task())

    assert result.failure_code == "TOKEN_BUDGET_EXHAUSTED"
    turn_log = next(log for log in logs if "event=model_turn" in log)
    assert "input_tokens=8000" in turn_log
    assert "output_tokens=8000" in turn_log
    assert "turn_tokens=16000" in turn_log
    assert "total_tokens=16000" in turn_log
    assert "token_limit=15000" in turn_log
    assert "elapsed_ms=" in turn_log
    stop_log = next(log for log in logs if "event=solver_stop" in log)
    assert "reason=TOKEN_BUDGET_EXHAUSTED" in stop_log
    assert "turn=1" in stop_log
    assert "deadline_remaining_ms=" in stop_log
    assert "do-not-log-me" not in "\n".join(logs)


def test_tool_logs_include_safe_structure_but_not_values_or_output():
    logs: list[str] = []
    secret = "super-secret-value"
    output = f"failure output containing {secret}"
    tool = FakeTool(
        "read_file",
        result=ToolResult(ok=False, output=output, error_code="PATH_BLOCKED"),
    )
    registry = ToolRegistry()
    registry.register(
        tool,
        ToolSchema("read_file", "read something", {"type": "object"}),
    )
    client = ScriptedModelClient(
        responses=[
            tool_call_response(
                "read_file",
                {"path": "/private/data.txt", "password": secret},
            ),
            text_response("FINAL_ANSWER: hidden-answer"),
        ]
    )
    engine = SolverEngine(client, registry, logger=logs.append)

    result = engine.solve(make_task())

    assert result.candidate is not None
    tool_log = next(log for log in logs if "event=tool_result" in log)
    assert "error=PATH_BLOCKED" in tool_log
    assert f"output_chars={len(output)}" in tool_log
    assert "arg_keys=password,path" in tool_log
    assert "path_kind=absolute" in tool_log
    combined = "\n".join(logs)
    assert secret not in combined
    assert output not in combined
    assert "/private/data.txt" not in combined
    assert "hidden-answer" not in combined


def test_over_budget_same_response_final_answer_is_preserved():
    client = ScriptedModelClient(
        responses=[text_response("FINAL_ANSWER: recovered", tokens=10_000)]
    )
    engine = SolverEngine(
        client,
        _empty_registry(),
        max_total_tokens=15_000,
    )

    result = engine.solve(make_task(answer_format="literal"))

    assert result.candidate is not None
    assert result.candidate.value == "recovered"
    assert result.candidate.confidence == 0.70
    assert result.retryable is False


def test_over_budget_prose_preserves_prior_exact_value():
    tool = FakeTool(
        "calc",
        result=ToolResult(ok=True, output="done", exact_value="12345"),
    )
    registry = ToolRegistry()
    registry.register(tool, ToolSchema("calc", "compute", {"type": "object"}))
    client = ScriptedModelClient(
        responses=[
            tool_call_response("calc", {}),
            text_response("The exact result is above.", tokens=10_000),
        ]
    )
    engine = SolverEngine(client, registry, max_total_tokens=15_000)

    result = engine.solve(make_task(answer_format="numeric"))

    assert result.candidate is not None
    assert result.candidate.value == "12345"
    assert result.candidate.confidence == 0.95
    assert result.candidate.exact_value_from_tool is True


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


def test_exact_value_auto_finalizes_when_model_omits_answer_envelope():
    tool = FakeTool(
        "calc",
        result=ToolResult(
            ok=True,
            output="verified computation complete",
            exact_value="8675309",
        ),
    )
    registry = ToolRegistry()
    registry.register(tool, ToolSchema("calc", "compute", {"type": "object"}))
    client = ScriptedModelClient(
        responses=[
            tool_call_response("calc", {}),
            text_response("The tool produced the verified result above."),
        ]
    )
    logs: list[str] = []
    engine = SolverEngine(client, registry, max_turns=5, logger=logs.append)

    result = engine.solve(make_task(answer_format="numeric"))

    assert result.retryable is False
    assert result.failure_code is None
    assert result.candidate is not None
    assert result.candidate.value == "8675309"
    assert result.candidate.exact_value_from_tool is True
    assert result.candidate.confidence == 0.95
    assert "[calc] verified computation complete" in result.candidate.evidence
    assert "The tool produced the verified result above." in result.candidate.evidence
    assert any("auto-finalizing tool exact_value" in message for message in logs)


def test_successful_web_output_grounds_normalized_final_answer():
    tool = FakeTool(
        "web",
        result=ToolResult(ok=True, output='{"answer":"WINTER   DAWN"}'),
    )
    registry = ToolRegistry()
    registry.register(tool, ToolSchema("web", "browse", {"type": "object"}))
    client = ScriptedModelClient(
        responses=[
            tool_call_response("web", {}),
            text_response("FINAL_ANSWER: Winter Dawn"),
        ]
    )
    engine = SolverEngine(client, registry)

    result = engine.solve(make_task(answer_format="literal"))

    assert result.candidate is not None
    assert result.candidate.value == "Winter Dawn"
    assert result.candidate.confidence == 0.82
    assert result.candidate.exact_value_from_tool is False


def test_grounding_requires_successful_supported_tool_and_nontrivial_answer():
    cases = (
        ("web", ToolResult(ok=False, output="winter dawn", error_code="FAILED"), "winter dawn"),
        ("lookup", ToolResult(ok=True, output="winter dawn"), "winter dawn"),
        ("read_file", ToolResult(ok=True, output="the answer is yes"), "yes"),
        ("read_file", ToolResult(ok=True, output="wintergreen"), "winter"),
    )

    for tool_name, tool_result, answer in cases:
        tool = FakeTool(tool_name, result=tool_result)
        registry = ToolRegistry()
        registry.register(
            tool,
            ToolSchema(tool_name, "test", {"type": "object"}),
        )
        client = ScriptedModelClient(
            responses=[
                tool_call_response(tool_name, {}),
                text_response(f"FINAL_ANSWER: {answer}"),
            ]
        )

        result = SolverEngine(client, registry).solve(
            make_task(answer_format="literal")
        )

        assert result.candidate is not None
        assert result.candidate.confidence == 0.70
