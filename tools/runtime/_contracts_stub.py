"""TEMPORARY mirror of the frozen dataclasses defined in root-level
`contracts.py` (TEAM_PLAN.md #5, owned by Nandh).

That file does not exist in this branch yet — Nandh's Gate 1 PR hasn't
landed — so tools/runtime/ imports from here instead of blocking on it.
Every type below is copied verbatim from the plan; nothing here should ever
diverge from the real thing on purpose.

DELETE THIS FILE once `contracts.py` exists at the repo root. The only
consumer is `tool.py`'s import fallback:

    try:
        from contracts import TaskContext, ToolRequest, ToolResult, Tool
    except ImportError:
        from tools.runtime._contracts_stub import (
            TaskContext, ToolRequest, ToolResult, Tool)

so removing this file plus that fallback is a one-line change per module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

AnswerFormat = Literal["exact", "exact_ci", "numeric", "literal", "validator"]


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    category: str
    points: int
    prompt: str
    answer_format: AnswerFormat
    workdir: Path
    files: tuple[Path, ...]
    deadline_monotonic: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: Mapping[str, Any]
    timeout_seconds: float


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    error_code: str | None = None
    elapsed_ms: int = 0
    exact_value: str | None = None


class Tool(Protocol):
    name: str

    def execute(self, request: ToolRequest, task: TaskContext) -> ToolResult: ...
