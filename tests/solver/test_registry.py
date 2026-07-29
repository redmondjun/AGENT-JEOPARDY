from pathlib import Path

from contracts import ToolResult
from solver.registry import ToolRegistry, ToolSchema
from tests.solver.fakes import FakeTool, make_task


def _registry_with(tool: FakeTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool, ToolSchema(name=tool.name, description="fake", input_schema={"type": "object"}))
    return registry


def test_unknown_tool_returns_typed_error_not_exception():
    registry = ToolRegistry()
    task = make_task()
    result = registry.dispatch("does_not_exist", {}, 5.0, task)
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error_code == "UNKNOWN_TOOL"


def test_malformed_arguments_returns_typed_error():
    tool = FakeTool("echo")
    registry = _registry_with(tool)
    task = make_task()
    result = registry.dispatch("echo", "not a mapping", 5.0, task)  # type: ignore[arg-type]
    assert result.ok is False
    assert result.error_code == "INVALID_ARGUMENT"
    assert tool.calls == []  # tool.execute must not have been reached


def test_tool_exception_is_converted_not_raised():
    tool = FakeTool("boom", raise_exc=RuntimeError("kaboom"))
    registry = _registry_with(tool)
    task = make_task()

    result = registry.dispatch("boom", {}, 5.0, task)  # must not raise

    assert result.ok is False
    assert result.error_code == "TOOL_EXCEPTION"
    assert "kaboom" in result.output


def test_successful_dispatch_returns_tool_result():
    tool = FakeTool("echo", result=ToolResult(ok=True, output="hi", exact_value="42"))
    registry = _registry_with(tool)
    task = make_task()

    result = registry.dispatch("echo", {"x": 1}, 5.0, task)

    assert result.ok is True
    assert result.exact_value == "42"
    assert tool.calls[0].arguments == {"x": 1}
