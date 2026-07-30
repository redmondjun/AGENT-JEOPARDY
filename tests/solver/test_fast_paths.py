from __future__ import annotations

import time
from pathlib import Path

from contracts import CandidateAnswer, SolveResult, TaskContext
from solver.fast_paths import CompositeTileSolver
from solver.preprocessing import _cached_summary, preprocess_task


class Fallback:
    def __init__(self) -> None:
        self.calls = 0

    def solve(self, task: TaskContext) -> SolveResult:
        self.calls += 1
        return SolveResult(
            CandidateAnswer("fallback", 0.8, ("tool output",), "fallback"),
            retryable=False,
        )


def context(tmp_path: Path, *, category: str, prompt: str, files=()) -> TaskContext:
    return TaskContext(
        task_id="PR-X1",
        category=category,
        points=100,
        prompt=prompt,
        answer_format="numeric" if category == "Heavy Compute" else "exact",
        workdir=tmp_path,
        files=tuple(files),
        deadline_monotonic=time.monotonic() + 30,
    )


def test_unambiguous_arithmetic_uses_fast_path(tmp_path: Path) -> None:
    fallback = Fallback()
    result = CompositeTileSolver(fallback).solve(
        context(tmp_path, category="Heavy Compute", prompt="Evaluate: (7 + 5) * 3")
    )
    assert result.candidate is not None
    assert result.candidate.value == "36"
    assert "deterministic_fast_path" in result.candidate.strategy
    assert result.telemetry.model_turns == 0
    assert fallback.calls == 0


def test_uncertain_prompt_falls_back(tmp_path: Path) -> None:
    fallback = Fallback()
    result = CompositeTileSolver(fallback).solve(
        context(tmp_path, category="Heavy Compute", prompt="Optimize this graph")
    )
    assert result.candidate is not None
    assert result.candidate.value == "fallback"
    assert fallback.calls == 1


def test_fast_path_exception_falls_back_without_consuming_retry(tmp_path: Path) -> None:
    fallback = Fallback()
    solver = CompositeTileSolver(fallback)

    def broken_handler(_task):
        raise OSError("fixture disappeared")

    solver._handlers["Heavy Compute"] = broken_handler
    result = solver.solve(
        context(tmp_path, category="Heavy Compute", prompt="Evaluate: 1 + 1")
    )

    assert result.candidate is not None
    assert result.candidate.value == "fallback"
    assert fallback.calls == 1


def test_csv_row_count_is_programmatic(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    path.write_text("id,name\n1,a\n2,b\n")
    result = CompositeTileSolver(Fallback()).solve(
        context(
            tmp_path,
            category="Needle in the Haystack",
            prompt="How many data rows are in the CSV?",
            files=(path,),
        )
    )
    assert result.candidate is not None
    assert result.candidate.value == "2"


def test_preprocessing_adds_bounded_manifest(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    path.write_text("id,name\n1,a\n")
    enriched = preprocess_task(
        context(
            tmp_path,
            category="Needle in the Haystack",
            prompt="Find a record",
            files=(path,),
        )
    )
    summary = enriched.metadata["preprocessing"]
    assert "Preflight manifest" in summary
    assert "columns: id, name" in summary
    assert len(summary) <= 4_050


def test_preprocessing_is_cached_across_unchanged_retries(tmp_path: Path) -> None:
    _cached_summary.cache_clear()
    path = tmp_path / "records.csv"
    path.write_text("id,name\n1,a\n")
    task = context(
        tmp_path,
        category="Needle in the Haystack",
        prompt="Find a record",
        files=(path,),
    )

    preprocess_task(task)
    preprocess_task(task)

    assert _cached_summary.cache_info().hits == 1
