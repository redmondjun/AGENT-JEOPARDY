"""
The solver's tool-use loop: one TaskContext in, one SolveResult out.
Never calls jp.submit() — that is Nandh's submission gate, not ours
(TEAM_PLAN.md section 4, "Only the submission gate calls jp.submit()").

Design decisions worth flagging to the team:

1. Model I/O is behind a small ModelClient protocol, not a direct
   `anthropic.Anthropic()` call. This is what lets tests run with zero
   network access and a fake client — see tests/solver/conftest.py. Wire
   the real client at the bottom of this file (AnthropicModelClient) once
   we know the event proxy's base_url/auth from the organizer README.

2. Exact-value pass-through: whenever a tool call's ToolResult carries a
   non-null exact_value, we remember it as `last_exact_value`. If the
   model later emits FINAL_ANSWER while an exact_value is still the most
   recent one on record, we use THAT value verbatim as
   CandidateAnswer.value (and set exact_value_from_tool=True) instead of
   whatever string the model typed after "FINAL_ANSWER:". This is what
   satisfies the contract invariant "exact_value bypasses model retyping"
   — it has to happen here, at construction time, not later in
   verification, because verification only ever sees the candidate we
   already built.

3. History compaction: once the conversation exceeds MAX_HISTORY_MESSAGES,
   we drop the oldest tool_use/tool_result turn pairs (never the original
   system/user turn) rather than truncating text mid-message. This keeps
   token usage bounded without corrupting message structure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from contracts import CandidateAnswer, SolveResult, TaskContext
from solver.answer_parser import extract_final_answer, normalize_answer
from solver.prompts import get_system_prompt
from solver.registry import ToolRegistry
from solver.verification import verify_candidate

MAX_TURNS_DEFAULT = 8
MAX_TOTAL_TOKENS_DEFAULT = 20_000
MAX_TOKENS_PER_CALL = 4096  # hard cap per starter guide, section 1
MAX_HISTORY_MESSAGES = 12
TOOL_TIMEOUT_SECONDS_DEFAULT = 20.0
EVIDENCE_TRUNCATE_CHARS = 300


@dataclass(frozen=True)
class ModelResponse:
    """Normalized shape agent_loop needs, regardless of which SDK produced it."""

    text: str
    tool_calls: tuple["ModelToolCall", ...]
    input_tokens: int
    output_tokens: int
    raw_content: Any  # passed back into the next message's assistant turn


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


class ModelClient(Protocol):
    def create_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse: ...


class SolverEngine:
    """Implements contracts.TileSolver."""

    def __init__(
        self,
        model_client: ModelClient,
        registry: ToolRegistry,
        *,
        max_turns: int = MAX_TURNS_DEFAULT,
        max_total_tokens: int = MAX_TOTAL_TOKENS_DEFAULT,
        tool_timeout_seconds: float = TOOL_TIMEOUT_SECONDS_DEFAULT,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self._model_client = model_client
        self._registry = registry
        self._max_turns = max_turns
        self._max_total_tokens = max_total_tokens
        self._tool_timeout_seconds = tool_timeout_seconds
        self._logger = logger or (lambda _message: None)

    def solve(self, task: TaskContext) -> SolveResult:
        self._logger(
            f"{task.task_id}: solve start category={task.category!r} "
            f"points={task.points} files={len(task.files)}"
        )
        system = get_system_prompt(task.category)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": _initial_user_content(task)}
        ]
        tools_schema = self._registry.schemas_for_api()

        evidence: list[str] = []
        last_exact_value: str | None = None
        total_tokens = 0

        for turn_index in range(self._max_turns):
            if time.monotonic() >= task.deadline_monotonic:
                self._logger(f"{task.task_id}: solver stopped DEADLINE_EXCEEDED")
                return SolveResult(
                    candidate=None,
                    retryable=True,
                    failure_code="DEADLINE_EXCEEDED",
                )

            messages = _compact_history(messages)

            response = self._model_client.create_turn(
                system=system,
                messages=messages,
                tools=tools_schema,
                max_tokens=MAX_TOKENS_PER_CALL,
            )
            total_tokens += response.input_tokens + response.output_tokens
            self._logger(
                f"{task.task_id}: model turn={turn_index + 1} "
                f"tools={','.join(call.name for call in response.tool_calls) or 'none'} "
                f"tokens={response.input_tokens + response.output_tokens}"
            )
            if total_tokens > self._max_total_tokens:
                self._logger(f"{task.task_id}: solver stopped TOKEN_BUDGET_EXHAUSTED")
                return SolveResult(
                    candidate=None,
                    retryable=True,
                    failure_code="TOKEN_BUDGET_EXHAUSTED",
                )

            messages.append({"role": "assistant", "content": response.raw_content})

            if response.tool_calls:
                tool_result_blocks = []
                for call in response.tool_calls:
                    result = self._registry.dispatch(
                        call.name,
                        call.arguments,
                        self._tool_timeout_seconds,
                        task,
                    )
                    self._logger(
                        f"{task.task_id}: tool={call.name} ok={result.ok} "
                        f"elapsed_ms={result.elapsed_ms} "
                        f"error={result.error_code or 'none'}"
                    )
                    evidence.append(_truncate(f"[{call.name}] {result.output}"))
                    if result.ok and result.exact_value is not None:
                        last_exact_value = result.exact_value

                    tool_result_blocks.append(
                        _tool_result_block(call.id, result.output, is_error=not result.ok)
                    )

                messages.append({"role": "user", "content": tool_result_blocks})
                continue

            # No tool call this turn — look for the final-answer envelope.
            raw_answer = extract_final_answer(response.text)
            if raw_answer is None:
                # Model produced neither a tool call nor a final answer.
                # Rather than loop forever hoping the next turn is
                # different, treat this as a retryable failure — the
                # orchestrator can requeue the tile with a fresh worker.
                self._logger(f"{task.task_id}: solver stopped NO_ACTIONABLE_OUTPUT")
                return SolveResult(
                    candidate=None,
                    retryable=True,
                    failure_code="NO_ACTIONABLE_OUTPUT",
                )

            evidence.append(_truncate(response.text))
            candidate = self._build_candidate(
                task=task,
                raw_answer=raw_answer,
                last_exact_value=last_exact_value,
                evidence=tuple(evidence),
            )
            outcome = verify_candidate(candidate, task)
            if not outcome.passed:
                # Verification vetoed the model's own confidence. This is
                # not automatically retryable at the solver level — the
                # candidate is real, just unverified — so we hand it back
                # with confidence clamped and let Nandh's submission gate
                # apply the category/tier threshold and decide.
                candidate = CandidateAnswer(
                    value=candidate.value,
                    confidence=outcome.confidence,
                    evidence=candidate.evidence + outcome.reasons,
                    strategy=candidate.strategy,
                    exact_value_from_tool=candidate.exact_value_from_tool,
                )
            else:
                candidate = CandidateAnswer(
                    value=candidate.value,
                    confidence=outcome.confidence,
                    evidence=candidate.evidence,
                    strategy=candidate.strategy,
                    exact_value_from_tool=candidate.exact_value_from_tool,
                )

            self._logger(
                f"{task.task_id}: candidate strategy={candidate.strategy!r} "
                f"confidence={candidate.confidence:.3f} "
                f"verified={outcome.passed} exact_tool={candidate.exact_value_from_tool}"
            )
            return SolveResult(candidate=candidate, retryable=False)

        self._logger(f"{task.task_id}: solver stopped TURN_BUDGET_EXHAUSTED")
        return SolveResult(
            candidate=None,
            retryable=True,
            failure_code="TURN_BUDGET_EXHAUSTED",
        )

    def _build_candidate(
        self,
        *,
        task: TaskContext,
        raw_answer: str,
        last_exact_value: str | None,
        evidence: tuple[str, ...],
    ) -> CandidateAnswer:
        if last_exact_value is not None:
            return CandidateAnswer(
                value=last_exact_value,
                confidence=0.95,
                evidence=evidence,
                strategy=f"{task.category}:tool_exact_value",
                exact_value_from_tool=True,
            )

        value = normalize_answer(raw_answer, task.answer_format)
        return CandidateAnswer(
            value=value,
            confidence=0.7,
            evidence=evidence,
            strategy=task.category,
            exact_value_from_tool=False,
        )


def _initial_user_content(task: TaskContext) -> str:
    file_lines = "\n".join(f"- {path}" for path in task.files) or "(none)"
    return (
        f"Task ID: {task.task_id}\n"
        f"Category: {task.category}\n"
        f"Points: {task.points}\n"
        f"Answer format: {task.answer_format}\n"
        f"Working directory: {task.workdir}\n"
        f"Attached files:\n{file_lines}\n\n"
        f"Prompt:\n{task.prompt}"
    )


def _tool_result_block(tool_use_id: str, output: str, *, is_error: bool) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": output,
        "is_error": is_error,
    }


def _truncate(text: str) -> str:
    if len(text) <= EVIDENCE_TRUNCATE_CHARS:
        return text
    return text[:EVIDENCE_TRUNCATE_CHARS] + "…[truncated]"


def _compact_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keeps the original user turn (index 0) plus the most recent
    MAX_HISTORY_MESSAGES - 1 messages. Drops from just after index 0 so we
    never lose the task statement, only aging tool_use/tool_result pairs.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    head = messages[:1]
    tail_start = len(messages) - (MAX_HISTORY_MESSAGES - 1)

    # Anthropic requires every user tool_result block to immediately follow
    # the assistant message containing its matching tool_use block. A blind
    # slice can begin on a tool_result, after which the SDK merges it with the
    # preserved initial user prompt and the API rejects the whole request.
    # Keep the preceding assistant turn as an atomic pair, even if that makes
    # the compacted history one message larger than the soft limit.
    if tail_start > 1 and _is_tool_result_message(messages[tail_start]):
        tail_start -= 1
    return head + messages[tail_start:]


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


# ---------------------------------------------------------------------------
# Real Anthropic-backed client. Not exercised by tests (no network in this
# sandbox) — wire in the event proxy's base_url once the organizer README
# specifics are confirmed, per TEAM_PLAN.md "Fixed model: Claude Haiku 4.5
# through the event proxy".
# ---------------------------------------------------------------------------


class AnthropicModelClient:
    """Thin adapter from the anthropic SDK's response shape to ModelResponse."""

    def __init__(self, client: Any, model: str = "claude-haiku-4-5") -> None:
        self._client = client
        self._model = model

    def create_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse:
        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )

        text_parts: list[str] = []
        tool_calls: list[ModelToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ModelToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return ModelResponse(
            text="\n".join(text_parts),
            tool_calls=tuple(tool_calls),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_content=response.content,
        )
