"""Tool-using tile solver and adaptive answer verification."""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import ast
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import jeopardy as jp
from tools import CandidateAnswer, TOOL_SCHEMAS, TaskTools


CATEGORY_GUIDANCE = {
    "Needle in the Haystack": (
        "Inspect schemas and samples first. Use Python/pandas or streaming "
        "parsers for exact computation; never estimate from a preview."
    ),
    "The Dark Web": (
        "Use http_request so cookies persist. Inspect forms, redirects, hidden "
        "fields, CSRF tokens, and response bodies before taking the next step."
    ),
    "Ship It": (
        "Read the code, reproduce the failure, edit only the task workspace, "
        "and run the program or tests to prove the answer."
    ),
    "Ancient Scrolls": (
        "Search documents programmatically, follow cross-references and "
        "amendments, and cite the exact evidence used."
    ),
    "Cryptic": (
        "Inspect raw bytes and file signatures, recurse through archives and "
        "encodings, and capture exact decoded tokens programmatically."
    ),
    "Heavy Compute": (
        "Implement the search or optimization in Python, validate constraints, "
        "and independently check the returned objective or witness."
    ),
}


@dataclass(frozen=True)
class SolverConfig:
    max_turns: int = 8
    max_tokens: int = 4096
    practice_confidence: float = 0.55
    scored_exact_confidence: float = 0.75
    scored_review_confidence: float = 0.85
    verbose: bool = False


@dataclass(frozen=True)
class SolveOutcome:
    candidate: CandidateAnswer | None
    elapsed_seconds: float
    failure_code: str | None = None
    detail: str = ""


def validate_answer(value: str, answer_format: str) -> str:
    """Validate and canonicalize an answer without changing its meaning."""
    if not isinstance(value, str):
        value = str(value)
    answer_format = answer_format or "exact"
    if answer_format in {"exact", "exact_ci"}:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("answer is empty")
        return normalized
    if answer_format == "numeric":
        cleaned = value.strip().replace(",", "").replace("$", "")
        try:
            number = Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError("answer is not numeric") from exc
        if not number.is_finite():
            raise ValueError("numeric answer must be finite")
        return format(number, "f")
    if answer_format == "literal":
        try:
            parsed = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError) as exc:
            raise ValueError("answer is not a Python/JSON literal") from exc
        return repr(parsed)
    if answer_format == "validator":
        normalized = value.strip()
        if not normalized:
            raise ValueError("validator answer is empty")
        return normalized
    raise ValueError(f"unknown answer format {answer_format!r}")


def _get(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _serialize_block(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    result = {"type": _get(block, "type")}
    for name in ("id", "name", "input", "text"):
        value = _get(block, name)
        if value is not None:
            result[name] = value
    return result


def _text(response: Any) -> str:
    return "".join(
        str(_get(block, "text", ""))
        for block in _get(response, "content", [])
        if _get(block, "type") == "text"
    ).strip()


class TileSolver:
    def __init__(
        self,
        config: SolverConfig | None = None,
        client_factory: Callable[[], Any] = jp.anthropic_client,
    ):
        self.config = config or SolverConfig()
        self.client_factory = client_factory

    def solve(
        self,
        detail: dict[str, Any],
        workdir: Path,
        phase: str,
        rejected_answers: set[str] | None = None,
    ) -> SolveOutcome:
        started = time.monotonic()
        task_id = str(detail["id"])
        answer_format = str(detail.get("answer_format", "exact"))
        tools = TaskTools(task_id, workdir, jp.BASE)
        rejected = rejected_answers or set()
        category = str(detail.get("category", ""))
        guidance = CATEGORY_GUIDANCE.get(
            category,
            "Use the available tools to compute and verify the answer.",
        )
        prompt = (
            f"Task {task_id} ({category}, {detail.get('points', 0)} points)\n"
            f"{detail.get('prompt', '')}\n\n"
            f"Files are in {workdir}. Answer format: {answer_format}.\n"
            f"Category guidance: {guidance}\n\n"
            "Work through the task with tools. Do not guess. Before finishing, "
            "verify the result. Use finalize_answer exactly once when ready. "
            "When an exact value is produced by code, finalize with answer_ref "
            "instead of retyping it."
        )
        if rejected:
            prompt += f"\nNever submit these previously rejected answers: {sorted(rejected)!r}"
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        client = self.client_factory()

        for turn in range(self.config.max_turns):
            try:
                response = client.messages.create(
                    model=jp.MODEL,
                    max_tokens=self.config.max_tokens,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                )
            except Exception as exc:  # noqa: BLE001 - isolate model failures
                return SolveOutcome(
                    None,
                    time.monotonic() - started,
                    "model_error",
                    repr(exc),
                )
            blocks = [_serialize_block(block) for block in _get(response, "content", [])]
            messages.append({"role": "assistant", "content": blocks})
            tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
            if not tool_uses:
                instruction = (
                    "Your response did not call a tool. Continue the analysis and "
                    "call finalize_answer only after verification."
                )
                if _get(response, "stop_reason") == "max_tokens":
                    instruction = (
                        "The response hit max_tokens. Continue concisely, use tools "
                        "as needed, then call finalize_answer."
                    )
                messages.append({"role": "user", "content": instruction})
                continue

            tool_results: list[dict[str, Any]] = []
            accepted: CandidateAnswer | None = None
            for block in tool_uses:
                tool_id = str(block.get("id", "missing_tool_id"))
                name = str(block.get("name", ""))
                raw_input = block.get("input", {})
                if not isinstance(raw_input, dict):
                    execution_content = "Tool input must be a JSON object."
                    is_error = True
                    candidate = None
                else:
                    execution = tools.execute(name, raw_input)
                    execution_content = execution.content
                    is_error = execution.is_error
                    candidate = execution.candidate
                if candidate is not None:
                    try:
                        canonical = validate_answer(candidate.value, answer_format)
                        candidate = replace(candidate, value=canonical)
                        if canonical in rejected:
                            raise ValueError("candidate was previously rejected")
                        approved, reason = self._approve(
                            client, detail, phase, candidate
                        )
                        if approved:
                            accepted = candidate
                            execution_content = "Candidate accepted by policy."
                            is_error = False
                        else:
                            execution_content = (
                                f"Candidate rejected by policy: {reason}. "
                                "Continue solving and produce stronger evidence."
                            )
                            is_error = True
                    except ValueError as exc:
                        execution_content = (
                            f"Candidate validation failed: {exc}. Continue solving."
                        )
                        is_error = True
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": execution_content,
                        "is_error": is_error,
                    }
                )

            if accepted is not None:
                return SolveOutcome(accepted, time.monotonic() - started)
            messages.append({"role": "user", "content": tool_results})

        return SolveOutcome(
            None,
            time.monotonic() - started,
            "turn_limit",
            f"no accepted candidate after {self.config.max_turns} turns",
        )

    def _approve(
        self,
        client: Any,
        detail: dict[str, Any],
        phase: str,
        candidate: CandidateAnswer,
    ) -> tuple[bool, str]:
        if phase == "practice":
            if candidate.confidence >= self.config.practice_confidence:
                return True, "practice threshold met"
            return False, "confidence below practice threshold"
        if (
            candidate.exact_value_from_tool
            and candidate.confidence >= self.config.scored_exact_confidence
            and detail.get("answer_format", "exact")
            in {"exact", "exact_ci", "numeric", "literal"}
        ):
            return True, "programmatic scored threshold met"
        if candidate.confidence < self.config.scored_review_confidence:
            return False, "confidence below scored review threshold"
        return self._review(client, detail, candidate)

    def _review(
        self,
        client: Any,
        detail: dict[str, Any],
        candidate: CandidateAnswer,
    ) -> tuple[bool, str]:
        review_prompt = (
            "Act as an independent answer verifier. Reject guesses, unsupported "
            "claims, formatting mistakes, and evidence that does not prove the "
            "candidate. Return only JSON: "
            '{"approve": true_or_false, "reason": "brief reason"}.\n\n'
            f"Task: {detail.get('prompt', '')}\n"
            f"Answer format: {detail.get('answer_format', 'exact')}\n"
            f"Candidate: {candidate.value!r}\n"
            f"Evidence: {list(candidate.evidence)!r}\n"
            f"Verification notes: {candidate.verification_notes}"
        )
        try:
            response = client.messages.create(
                model=jp.MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": review_prompt}],
            )
            raw = _text(response)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(match.group(0) if match else raw)
            approved = parsed.get("approve") is True
            return approved, str(parsed.get("reason", "reviewer gave no reason"))
        except Exception as exc:  # noqa: BLE001 - failure must fail closed
            return False, f"review failed closed: {exc!r}"
