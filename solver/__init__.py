"""Production assembly for Sara's category-aware solver."""

from __future__ import annotations

import os
from typing import Any

from solver import specialists  # noqa: F401  (register verification checks)
from solver.agent_loop import (
    MAX_TOTAL_TOKENS_DEFAULT,
    MAX_TURNS_DEFAULT,
    TOOL_TIMEOUT_SECONDS_DEFAULT,
    AnthropicModelClient,
    SolverEngine,
)
from solver.registry import ToolRegistry, ToolSchema
from tools.runtime.tool import get_tools as get_runtime_tools
from tools.web.tool import WebTool


_WEB_SCHEMA = ToolSchema(
    name="web",
    description=(
        "Make stateful HTTP requests within the task's allowed origin, follow "
        "redirects, retain cookies, inspect semantic HTML, and submit forms."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["request", "submit_form", "reset"],
                "default": "request",
            },
            "method": {"type": "string", "default": "GET"},
            "url": {"type": "string"},
            "params": {"type": "object"},
            "data": {"type": "object"},
            "json": {"type": "object"},
            "headers": {"type": "object"},
            "expect_html": {"type": "boolean", "default": True},
            "form_ref": {"type": "string"},
            "fields": {"type": "object"},
        },
    },
)


def build_solver(*, game: Any, verbose: bool = False) -> SolverEngine:
    """Build the real model/tool solver expected by ``main.build_solver``."""
    registry = ToolRegistry()
    for tool in get_runtime_tools():
        registry.register(
            tool,
            ToolSchema(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            ),
        )
    registry.register(WebTool(), _WEB_SCHEMA)

    if verbose:
        game.log(
            "team solver enabled with tools: "
            + ", ".join(schema["name"] for schema in registry.schemas_for_api())
        )

    model_client = AnthropicModelClient(
        game.anthropic_client(),
        model=getattr(game, "MODEL", "claude-haiku-4-5"),
    )
    return SolverEngine(
        model_client,
        registry,
        # Defer to agent_loop's constants so the tuned budgets cannot drift
        # away from the values documented next to them.
        max_turns=int(
            os.environ.get("SOLVER_MAX_TURNS", str(MAX_TURNS_DEFAULT))
        ),
        max_total_tokens=int(
            os.environ.get(
                "SOLVER_MAX_TOTAL_TOKENS", str(MAX_TOTAL_TOKENS_DEFAULT)
            )
        ),
        tool_timeout_seconds=float(
            os.environ.get(
                "TOOL_TIMEOUT_SECONDS", str(TOOL_TIMEOUT_SECONDS_DEFAULT)
            )
        ),
        logger=game.log,
    )


__all__ = ["build_solver"]
