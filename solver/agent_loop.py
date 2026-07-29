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
import unicodedata
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
GROUNDED_CONFIDENCE = 0.82
UNGROUNDED_CONFIDENCE = 0.70
MIN_GROUNDED_ANSWER_ALNUM_CHARS = 5
GROUNDING_TOOL_NAMES = frozenset({"web", "read_file"})
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
            f"event=solve_start task={task.task_id} category={task.category!r} "
            f"points={task.points} files={len(task.files)} "
            f"turn_limit={self._max_turns} token_limit={self._max_total_tokens} "
            f"deadline_remaining_ms={_remaining_ms(task.deadline_monotonic)}"
        )
        system = get_system_prompt(task.category)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": _initial_user_content(task)}
        ]
        tools_schema = self._registry.schemas_for_api()

        evidence: list[str] = []
        grounding_outputs: list[str] = []
        last_exact_value: str | None = None
        total_tokens = 0

        for turn_index in range(self._max_turns):
            if time.monotonic() >= task.deadline_monotonic:
                self._log_stop(
                    task,
                    reason="DEADLINE_EXCEEDED",
                    turn=turn_index,
                    total_tokens=total_tokens,
                )
                return SolveResult(
                    candidate=None,
                    retryable=True,
                    failure_code="DEADLINE_EXCEEDED",
                )

            messages = _compact_history(messages)

            model_started = time.monotonic()
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
                    f"event=model_error task={task.task_id} turn={turn_index + 1} "
                    f"reason=MODEL_CALL_TIMEOUT failed=MODEL_CALL_TIMEOUT "
                    f"elapsed_ms={int((time.monotonic() - model_started) * 1000)} "
                    f"deadline_remaining_ms={_remaining_ms(task.deadline_monotonic)}"
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
                    f"event=model_error task={task.task_id} turn={turn_index + 1} "
                    f"reason=MODEL_API_ERROR exception={type(exc).__name__} "
                    f"elapsed_ms={int((time.monotonic() - model_started) * 1000)} "
                    f"deadline_remaining_ms={_remaining_ms(task.deadline_monotonic)}"
                )
                return SolveResult(
                    candidate=None,
                    retryable=True,
                    failure_code="MODEL_API_ERROR",
                )
            model_elapsed_ms = int((time.monotonic() - model_started) * 1000)
            turn_tokens = response.input_tokens + response.output_tokens
            total_tokens += turn_tokens
            self._logger(
                f"event=model_turn task={task.task_id} turn={turn_index + 1} "
                f"tools={','.join(call.name for call in response.tool_calls) or 'none'} "
                f"input_tokens={response.input_tokens} "
                f"output_tokens={response.output_tokens} "
                f"turn_tokens={turn_tokens} total_tokens={total_tokens} "
                f"token_limit={self._max_total_tokens} elapsed_ms={model_elapsed_ms} "
                f"deadline_remaining_ms={_remaining_ms(task.deadline_monotonic)}"
            )
            over_token_budget = total_tokens > self._max_total_tokens
            response_final_answer = (
                None
                if response.tool_calls
                else extract_final_answer(response.text)
            )
            if over_token_budget and (
                response.tool_calls
                or (response_final_answer is None and last_exact_value is None)
            ):
                self._log_stop(
                    task,
                    reason="TOKEN_BUDGET_EXHAUSTED",
                    turn=turn_index + 1,
                    total_tokens=total_tokens,
                )
                return SolveResult(
                    candidate=None,
                    retryable=True,
                    failure_code="TOKEN_BUDGET_EXHAUSTED",
                )

            messages.append({"role": "assistant", "content": response.raw_content})

            if response.tool_calls:
                tool_result_blocks = []
                for call_index, call in enumerate(response.tool_calls, start=1):
                    remaining_seconds = task.deadline_monotonic - time.monotonic()
                    if remaining_seconds <= 0:
                        self._log_stop(
                            task,
                            reason="DEADLINE_EXCEEDED",
                            turn=turn_index + 1,
                            total_tokens=total_tokens,
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
                    tool_started = time.monotonic()
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
                            f"event=tool_result task={task.task_id} "
                            f"turn={turn_index + 1} "
                            f"call={call_index}/{len(response.tool_calls)} "
                            f"tool={call.name} ok=False "
                            f"elapsed_ms={int((time.monotonic() - tool_started) * 1000)} "
                            f"error=TOOL_CALL_TIMEOUT output_chars=0 "
                            f"exact_value=False "
                            f"arg_keys={_argument_keys(call.arguments)} "
                            f"path_kind={_path_kind(call.arguments)}"
                        )
                        return SolveResult(
                            candidate=None,
                            retryable=True,
                            failure_code="TOOL_CALL_TIMEOUT",
                        )
                    self._logger(
                        f"event=tool_result task={task.task_id} turn={turn_index + 1} "
                        f"call={call_index}/{len(response.tool_calls)} tool={call.name} "
                        f"ok={result.ok} "
                        f"elapsed_ms={result.elapsed_ms} "
                        f"error={result.error_code or 'none'} "
                        f"output_chars={len(result.output)} "
                        f"exact_value={result.exact_value is not None} "
                        f"arg_keys={_argument_keys(call.arguments)} "
                        f"path_kind={_path_kind(call.arguments)}"
                    )
                    evidence.append(_truncate(f"[{call.name}] {result.output}"))
                    if result.ok and call.name in GROUNDING_TOOL_NAMES:
                        grounding_outputs.append(result.output)
                    if result.ok and result.exact_value is not None:
                        last_exact_value = result.exact_value

                    tool_result_blocks.append(
                        _tool_result_block(call.id, result.output, is_error=not result.ok)
                    )

                messages.append({"role": "user", "content": tool_result_blocks})
                continue

            # No tool call this turn — look for the final-answer envelope.
            raw_answer = response_final_answer
            if raw_answer is None:
                if last_exact_value is None:
                    # Model produced neither a tool call nor a final answer.
                    # Rather than loop forever hoping the next turn is
                    # different, treat this as a retryable failure — the
                    # orchestrator can requeue the tile with a fresh worker.
                    self._log_stop(
                        task,
                        reason="NO_ACTIONABLE_OUTPUT",
                        turn=turn_index + 1,
                        total_tokens=total_tokens,
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
                    f"event=auto_finalize task={task.task_id} "
                    f"reason=TOOL_EXACT_VALUE turn={turn_index + 1} "
                    f"detail='auto-finalizing tool exact_value'"
                )

            if response.text:
                evidence.append(_truncate(response.text))
            candidate = self._build_candidate(
                task=task,
                raw_answer=raw_answer,
                last_exact_value=last_exact_value,
                evidence=tuple(evidence),
                grounding_outputs=tuple(grounding_outputs),
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
                f"event=candidate task={task.task_id} strategy={candidate.strategy!r} "
                f"confidence={candidate.confidence:.3f} "
                f"verified={outcome.passed} exact_tool={candidate.exact_value_from_tool} "
                f"evidence_count={len(candidate.evidence)} "
                f"verification_reason_count={len(outcome.reasons)} "
                f"turn={turn_index + 1} total_tokens={total_tokens}"
            )
            return SolveResult(candidate=candidate, retryable=False)

        self._log_stop(
            task,
            reason="TURN_BUDGET_EXHAUSTED",
            turn=self._max_turns,
            total_tokens=total_tokens,
        )
        return SolveResult(
            candidate=None,
            retryable=True,
            failure_code="TURN_BUDGET_EXHAUSTED",
        )

    def _log_stop(
        self,
        task: TaskContext,
        *,
        reason: str,
        turn: int,
        total_tokens: int,
    ) -> None:
        self._logger(
            f"event=solver_stop task={task.task_id} reason={reason} turn={turn} "
            f"turn_limit={self._max_turns} total_tokens={total_tokens} "
            f"token_limit={self._max_total_tokens} "
            f"deadline_remaining_ms={_remaining_ms(task.deadline_monotonic)}"
        )

    def _build_candidate(
        self,
        *,
        task: TaskContext,
        raw_answer: str,
        last_exact_value: str | None,
        evidence: tuple[str, ...],
        grounding_outputs: tuple[str, ...],
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
        grounded = _answer_is_grounded(value, grounding_outputs)
        return CandidateAnswer(
            value=value,
            confidence=(
                GROUNDED_CONFIDENCE if grounded else UNGROUNDED_CONFIDENCE
            ),
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


def _remaining_ms(deadline_monotonic: float) -> int:
    return max(0, int((deadline_monotonic - time.monotonic()) * 1000))


def _argument_keys(arguments: dict[str, Any]) -> str:
    return ",".join(sorted(str(key) for key in arguments)) or "none"


def _path_kind(arguments: dict[str, Any]) -> str:
    path = arguments.get("path")
    if not isinstance(path, str):
        return "none"
    return "absolute" if path.startswith(("/", "\\")) else "relative"


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


def _answer_is_grounded(value: str, outputs: tuple[str, ...]) -> bool:
    """Require a non-trivial normalized answer as a whole output token/phrase.

    Case, Unicode presentation, and whitespace differences are harmless, but
    punctuation is deliberately preserved. Alphanumeric boundaries prevent a
    candidate such as ``cat`` from being credited merely because an output
    contains ``category``. Short generic values remain at the conservative
    ungrounded confidence even if they occur coincidentally in a page.
    """
    answer = _normalize_grounding_text(value)
    if sum(character.isalnum() for character in answer) < (
        MIN_GROUNDED_ANSWER_ALNUM_CHARS
    ):
        return False

    for output in outputs:
        haystack = _normalize_grounding_text(output)
        start = 0
        while True:
            match = haystack.find(answer, start)
            if match < 0:
                break
            before = haystack[match - 1] if match > 0 else ""
            end = match + len(answer)
            after = haystack[end] if end < len(haystack) else ""
            if (not before or not before.isalnum()) and (
                not after or not after.isalnum()
            ):
                return True
            start = match + 1
    return False


def _normalize_grounding_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


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
