"""
Tool registry for the solver loop.

Solver never imports tools.web or tools.runtime directly (see TEAM_PLAN.md
"Files Sara must not edit"). Instead, main.py wires Jun's and Vidula's real
Tool implementations into a ToolRegistry and hands the registry to the
solver. This keeps solver/ testable with fake tools and keeps the
directory-ownership boundary intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from contracts import Tool, ToolRequest, ToolResult, TaskContext


class UnknownToolError(Exception):
    """Raised internally, never propagated past the registry boundary."""


@dataclass(frozen=True)
class ToolSchema:
    """Anthropic tool-use schema for one registered tool."""

    name: str
    description: str
    input_schema: Mapping[str, Any]

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """
    Holds tool implementations and their schemas. Dispatch is defensive:
    unknown tool names and malformed arguments always come back as a typed
    ToolResult(ok=False, error_code=...), never a raised exception — the
    agent loop must be able to keep running after any single bad tool call.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._schemas: dict[str, ToolSchema] = {}

    def register(self, tool: Tool, schema: ToolSchema) -> None:
        if schema.name != tool.name:
            raise ValueError(
                f"schema name {schema.name!r} does not match tool.name {tool.name!r}"
            )
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        self._schemas[tool.name] = schema

    def schemas_for_api(self) -> list[dict[str, Any]]:
        return [schema.to_api_dict() for schema in self._schemas.values()]

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        timeout_seconds: float,
        task: TaskContext,
    ) -> ToolResult:
        """
        Never raises. Any failure — unknown tool, bad arguments, or an
        exception inside the tool itself — is converted to a structured
        ToolResult so a single misbehaving tool call cannot crash the
        worker running this tile.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                output=f"no such tool: {name!r}",
                error_code="UNKNOWN_TOOL",
            )

        if not isinstance(arguments, Mapping):
            return ToolResult(
                ok=False,
                output="tool arguments must be a JSON object",
                error_code="INVALID_ARGUMENT",
            )

        request = ToolRequest(
            name=name,
            arguments=arguments,
            timeout_seconds=timeout_seconds,
        )

        try:
            return tool.execute(request, task)
        except Exception as exc:  # noqa: BLE001 — intentional: tools are untrusted
            return ToolResult(
                ok=False,
                output=f"tool {name!r} raised {type(exc).__name__}: {exc}",
                error_code="TOOL_EXCEPTION",
            )
