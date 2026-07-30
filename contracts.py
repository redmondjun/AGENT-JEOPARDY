"""Stable interfaces shared by the Agent Jeopardy workstreams.

Keep this module dependency-free. Specialist modules may import these types,
but the contracts must never import specialist implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

AnswerFormat = Literal["exact", "exact_ci", "numeric", "literal", "validator"]
SUPPORTED_ANSWER_FORMATS: frozenset[str] = frozenset(
    {"exact", "exact_ci", "numeric", "literal", "validator"}
)


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

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if self.points < 0:
            raise ValueError("points must be non-negative")
        if self.answer_format not in SUPPORTED_ANSWER_FORMATS:
            raise ValueError(f"unsupported answer format: {self.answer_format!r}")
        if not math.isfinite(self.deadline_monotonic):
            raise ValueError("deadline_monotonic must be finite")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: Mapping[str, Any]
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("tool timeout must be positive")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    error_code: str | None = None
    elapsed_ms: int = 0
    exact_value: str | None = None

    def __post_init__(self) -> None:
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")
        if self.ok and self.error_code is not None:
            raise ValueError("successful tool result cannot carry an error_code")
        if not self.ok and not self.error_code:
            raise ValueError("failed tool result requires an error_code")


@dataclass(frozen=True)
class CandidateAnswer:
    value: str
    confidence: float
    evidence: tuple[str, ...]
    strategy: str
    exact_value_from_tool: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not self.strategy.strip():
            raise ValueError("strategy must not be empty")


@dataclass(frozen=True)
class SolveTelemetry:
    """Non-secret measurements produced while solving one tile."""

    elapsed_ms: int = 0
    model_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(
            self.elapsed_ms,
            self.model_turns,
            self.input_tokens,
            self.output_tokens,
        ) < 0:
            raise ValueError("solve telemetry counters must be non-negative")


@dataclass(frozen=True)
class SolveResult:
    candidate: CandidateAnswer | None
    retryable: bool
    failure_code: str | None = None
    retry_after_seconds: float | None = None
    telemetry: SolveTelemetry = field(default_factory=SolveTelemetry)

    def __post_init__(self) -> None:
        if self.candidate is not None and self.failure_code is not None:
            raise ValueError("successful solve result cannot carry a failure_code")
        if self.candidate is not None and self.retryable:
            raise ValueError("successful solve result cannot be retryable")
        if self.candidate is not None and self.retry_after_seconds is not None:
            raise ValueError("successful solve result cannot have a retry delay")
        if self.candidate is None and not self.failure_code:
            raise ValueError("failed solve result requires a failure_code")
        if not self.retryable and self.retry_after_seconds is not None:
            raise ValueError("terminal solve failure cannot have a retry delay")
        if self.retry_after_seconds is not None and self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must be non-negative")


@runtime_checkable
class Tool(Protocol):
    name: str

    def execute(self, request: ToolRequest, task: TaskContext) -> ToolResult: ...


@runtime_checkable
class TileSolver(Protocol):
    def solve(self, task: TaskContext) -> SolveResult: ...


@runtime_checkable
class GameAPI(Protocol):
    """The small surface the orchestrator needs from jeopardy.py."""

    def board(self) -> dict[str, Any]: ...

    def open_tiles(self, board: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

    def task(self, task_id: str) -> dict[str, Any]: ...

    def workdir(self, task_id: str) -> Path: ...

    def fetch_files(
        self,
        task_id: str,
        detail: dict[str, Any],
        dest: str | Path | None = None,
    ) -> Sequence[str]: ...

    def submit(self, task_id: str, answer: str) -> dict[str, Any]: ...

    def log(self, *values: object) -> None: ...
