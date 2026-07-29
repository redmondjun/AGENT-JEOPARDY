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
from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable, Protocol, TypeVar

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
_CallResult = TypeVar("_CallResult")


class _CallDeadlineExceeded(Exception):
    """Internal signal used when a blocking dependency outlives the tile."""


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

            try:
                response = _call_before_deadline(
                    lambda: self._model_client.create_turn(
                        system=system,
                        messages=messages,
                        tools=tools_schema,
                        max_tokens=MAX_TOKENS_PER_CALL,
                    ),
                    task.deadline_monotonic,
                )
            except _CallDeadlineExceeded:
                self._logger(
                    f"{task.task_id}: model turn={turn_index + 1} "
                    "failed=MODEL_CALL_TIMEOUT"
                )
                return SolveResult(
                    candidate=None,
                    retryable=True,
                    failure_code="MODEL_CALL_TIMEOUT",
                )
            except Exception as exc:  # noqa: BLE001 - SDK/network boundary
                # Exception messages from HTTP clients can contain request URLs,
                # headers, or response bodies. Log only the exception class.
                self._logger(
                    f"{task.task_id}: model turn={turn_index + 1} "
                    f"failed=MODEL_API_ERROR exception={type(exc).__name__}"
                )
                return SolveResult(
                    candidate=None,
                    retryable=True,
                    failure_code="MODEL_API_ERROR",
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
                    remaining_seconds = task.deadline_monotonic - time.monotonic()
                    if remaining_seconds <= 0:
                        self._logger(
                            f"{task.task_id}: solver stopped DEADLINE_EXCEEDED "
                            f"before_tool={call.name}"
                        )
                        return SolveResult(
                            candidate=None,
                            retryable=True,
                            failure_code="DEADLINE_EXCEEDED",
                        )

                    # A tool must never receive a timeout that extends beyond
                    # the tile's global deadline. Well-behaved runtime/web
                    # tools use this value as their own hard I/O boundary.
                    tool_timeout_seconds = min(
                        self._tool_timeout_seconds,
                        remaining_seconds,
                    )
                    try:
                        result = _call_before_deadline(
                            lambda: self._registry.dispatch(
                                call.name,
                                call.arguments,
                                tool_timeout_seconds,
                                task,
                            ),
                            task.deadline_monotonic,
                        )
                    except _CallDeadlineExceeded:
                        self._logger(
                            f"{task.task_id}: tool={call.name} ok=False "
                            "error=TOOL_CALL_TIMEOUT"
                        )
                        return SolveResult(
                            candidate=None,
                            retryable=True,
                            failure_code="TOOL_CALL_TIMEOUT",
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
                if last_exact_value is None:
                    # Model produced neither a tool call nor a final answer.
                    # Rather than loop forever hoping the next turn is
                    # different, treat this as a retryable failure — the
                    # orchestrator can requeue the tile with a fresh worker.
                    self._logger(
                        f"{task.task_id}: solver stopped NO_ACTIONABLE_OUTPUT"
                    )
                    return SolveResult(
                        candidate=None,
                        retryable=True,
                        failure_code="NO_ACTIONABLE_OUTPUT",
                    )

                # A successful runtime tool's exact_value is already the
                # authoritative, no-retyping answer channel. Live qualifier
                # runs showed the model sometimes acknowledged that result in
                # prose but omitted the FINAL_ANSWER envelope on its next
                # no-tool turn. Do not discard a proven answer for a formatting
                # lapse; build and verify the normal exact-tool candidate.
                raw_answer = last_exact_value
                self._logger(
                    f"{task.task_id}: auto-finalizing tool exact_value "
                    "after missing FINAL_ANSWER envelope"
                )

            if response.text:
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
    relative_files: list[str] = []
    for path in task.files:
        try:
            relative_files.append(str(path.relative_to(task.workdir)))
        except ValueError:
            # Tool paths are intentionally task-relative. Never encourage the
            # model to pass an absolute path that the runtime sandbox rejects.
            relative_files.append(path.name)
    file_lines = "\n".join(f"- {path}" for path in relative_files) or "(none)"
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


def _call_before_deadline(
    operation: Callable[[], _CallResult],
    deadline_monotonic: float,
) -> _CallResult:
    """Run a blocking call without allowing it to pin a worker forever.

    Python cannot safely kill a blocked thread, so a timed-out SDK call is
    detached as a daemon. The tile worker still returns at its deadline and
    the process remains able to exit. Network clients should additionally
    retain their own transport timeouts.
    """
    remaining_seconds = deadline_monotonic - time.monotonic()
    if remaining_seconds <= 0:
        raise _CallDeadlineExceeded

    outcome: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def invoke() -> None:
        try:
            value: Any = operation()
            item = (True, value)
        except Exception as exc:  # noqa: BLE001 - transfer to solver thread
            item = (False, exc)
        outcome.put(item)

    worker = Thread(target=invoke, name="solver-model-call", daemon=True)
    worker.start()
    try:
        succeeded, value = outcome.get(timeout=remaining_seconds)
    except Empty as exc:
        raise _CallDeadlineExceeded from exc

    if succeeded:
        return value
    raise value


def _compact_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keeps the original user turn (index 0) plus the most recent
    MAX_HISTORY_MESSAGES - 1 messages. Drops from just after index 0 so we
    never lose the task statement, only aging tool_use/tool_result pairs.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages

    # Anthropic requires every user tool_result block to immediately follow
    # the assistant message containing its matching tool_use block. Remove
    # complete oldest exchanges until the history fits rather than choosing a
    # slice boundary and trying to repair it afterward. This also preserves
    # assistant turns containing multiple tool_use blocks as one unit.
    compacted = list(messages)
    while len(compacted) > MAX_HISTORY_MESSAGES:
        if len(compacted) >= 3 and _is_tool_use_message(compacted[1]):
            del compacted[1:3]
        else:
            del compacted[1]
    return compacted


def _is_tool_use_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    return isinstance(content, list) and any(
        _block_value(block, "type") == "tool_use" for block in content
    )


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    return isinstance(content, list) and any(
        _block_value(block, "type") == "tool_result" for block in content
    )


def _block_value(block: Any, field: str) -> Any:
    """Read both test dictionaries and Anthropic SDK content block objects."""
    if isinstance(block, dict):
        return block.get(field)
    return getattr(block, field, None)


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
