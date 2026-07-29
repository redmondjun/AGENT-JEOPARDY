"""Validation, board recheck, and global submission throttling."""

from __future__ import annotations

import ast
import json
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Callable, Mapping

from contracts import CandidateAnswer, GameAPI, SUPPORTED_ANSWER_FORMATS

from .state import TileRecord


class SubmissionAction(str, Enum):
    DEFERRED = "deferred"
    REJECTED = "rejected"
    SOLVED = "solved"
    RETRY = "retry"
    DEAD = "dead"
    FATAL = "fatal"


@dataclass(frozen=True)
class SubmissionDecision:
    action: SubmissionAction
    reason: str
    retry_after_seconds: float | None = None
    raw: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SubmissionPolicy:
    minimum_interval_seconds: float = 3.1
    default_minimum_confidence: float = 0.80
    confidence_overrides: Mapping[tuple[str, int], float] = field(default_factory=dict)
    require_evidence: bool = True

    def __post_init__(self) -> None:
        if self.minimum_interval_seconds < 3.0:
            raise ValueError("submission interval must respect the 3-second team limit")
        if not 0.0 <= self.default_minimum_confidence <= 1.0:
            raise ValueError("default confidence must be between 0.0 and 1.0")
        for threshold in self.confidence_overrides.values():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("confidence override must be between 0.0 and 1.0")

    def minimum_confidence(self, category: str, points: int) -> float:
        return self.confidence_overrides.get(
            (category, points), self.default_minimum_confidence
        )


class SubmissionProtocolError(RuntimeError):
    pass


class SubmissionGate:
    def __init__(
        self,
        game: GameAPI,
        policy: SubmissionPolicy | None = None,
        *,
        clock=time.monotonic,
    ) -> None:
        self._game = game
        self._policy = policy or SubmissionPolicy()
        self._clock = clock
        self._last_submission_at: float | None = None

    def seconds_until_available(self) -> float:
        if self._last_submission_at is None:
            return 0.0
        elapsed = self._clock() - self._last_submission_at
        remaining = self._policy.minimum_interval_seconds - elapsed
        # Monotonic clocks are floats; an exact boundary such as 3.1 seconds
        # can otherwise leave a ~1e-15 remainder and defer a legal submission.
        return 0.0 if remaining <= 1e-9 else remaining

    def validate(
        self,
        record: TileRecord,
        candidate: CandidateAnswer,
    ) -> str | None:
        if not candidate.value.strip():
            return "EMPTY_ANSWER"
        if candidate.value in record.rejected_answers:
            return "REPEATED_INCORRECT_ANSWER"
        minimum = self._policy.minimum_confidence(record.category, record.points)
        if candidate.confidence < minimum:
            return f"LOW_CONFIDENCE:{candidate.confidence:.3f}<{minimum:.3f}"
        if (
            self._policy.require_evidence
            and not candidate.evidence
            and not candidate.exact_value_from_tool
        ):
            return "MISSING_EVIDENCE"
        if record.answer_format not in SUPPORTED_ANSWER_FORMATS:
            return f"UNSUPPORTED_ANSWER_FORMAT:{record.answer_format}"

        value = candidate.value.strip()
        if record.answer_format == "numeric":
            try:
                number = Decimal(value.replace(",", "").replace("$", ""))
            except InvalidOperation:
                return "INVALID_NUMERIC_ANSWER"
            if not number.is_finite():
                return "NON_FINITE_NUMERIC_ANSWER"
        elif record.answer_format == "literal":
            try:
                json.loads(value)
            except (json.JSONDecodeError, TypeError):
                try:
                    ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return "INVALID_LITERAL_ANSWER"
        return None

    def attempt(
        self,
        record: TileRecord,
        candidate: CandidateAnswer,
        *,
        is_open: Callable[[str], bool],
    ) -> SubmissionDecision:
        validation_error = self.validate(record, candidate)
        if validation_error:
            return SubmissionDecision(SubmissionAction.REJECTED, validation_error)

        wait = self.seconds_until_available()
        if wait > 0:
            return SubmissionDecision(
                SubmissionAction.DEFERRED,
                "GLOBAL_RATE_LIMIT",
                retry_after_seconds=wait,
            )
        if not is_open(record.task_id):
            return SubmissionDecision(SubmissionAction.DEAD, "NO_LONGER_OPEN")

        # Reserve the global slot before the network call. A timeout can happen
        # after the server accepted the request; delaying the next attempt is
        # safer than accidentally double-submitting inside three seconds.
        self._last_submission_at = self._clock()
        response = self._game.submit(record.task_id, candidate.value)
        result = response.get("result")
        if not isinstance(result, str):
            raise SubmissionProtocolError(
                f"{record.task_id}: submit response has no string result: {response!r}"
            )

        if result == "correct":
            return SubmissionDecision(SubmissionAction.SOLVED, result, raw=response)
        if result in {"already_claimed", "voided", "wrong_phase", "unknown_task"}:
            return SubmissionDecision(SubmissionAction.DEAD, result, raw=response)
        if result == "forbidden":
            return SubmissionDecision(SubmissionAction.FATAL, result, raw=response)
        if result == "rate_limited":
            return SubmissionDecision(
                SubmissionAction.RETRY,
                result,
                retry_after_seconds=_retry_seconds(
                    response, self._policy.minimum_interval_seconds
                ),
                raw=response,
            )
        if result == "locked_out":
            return SubmissionDecision(
                SubmissionAction.RETRY,
                result,
                retry_after_seconds=_retry_seconds(response, 30.0),
                raw=response,
            )
        if result == "incorrect":
            fallback = _incorrect_cooldown(record)
            return SubmissionDecision(
                SubmissionAction.RETRY,
                result,
                retry_after_seconds=_retry_seconds(response, fallback),
                raw=response,
            )
        return SubmissionDecision(
            SubmissionAction.RETRY,
            f"UNRECOGNIZED_RESULT:{result}",
            retry_after_seconds=5.0,
            raw=response,
        )


def _retry_seconds(response: Mapping[str, object], fallback: float) -> float:
    value = response.get("retry_in", fallback)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed) or parsed < 0:
        return fallback
    return parsed


def _incorrect_cooldown(record: TileRecord) -> float:
    if record.board == "practice" or record.task_id.startswith("PR-"):
        return 10.0
    exponent = max(0, record.wrong_attempts)
    return min(30.0 * (2**exponent), 480.0)
