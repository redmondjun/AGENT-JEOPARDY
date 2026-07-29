"""Competitive Agent Jeopardy v1.

The process polls the live board indefinitely, schedules every open variant,
solves tiles concurrently, and funnels verified candidates through one
rate-limited submission gate.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import jeopardy as jp
from solver import CandidateAnswer, SolverConfig, SolveOutcome, TileSolver


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANDIDATE = "candidate"
    COOLDOWN = "cooldown"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class Config:
    verbose: bool = os.environ.get("VERBOSE") == "1"
    task_filter: tuple[str, ...] = tuple(
        value.strip()
        for value in os.environ.get("TASK_FILTER", "").split(",")
        if value.strip()
    )
    max_tiles: int = int(os.environ.get("MAX_TILES", "0"))
    workers: int = max(1, int(os.environ.get("WORKERS", "6")))
    max_turns: int = max(1, int(os.environ.get("MAX_TURNS", "8")))
    tile_timeout_seconds: float = max(
        10.0, float(os.environ.get("TILE_TIMEOUT_SECONDS", "120"))
    )
    poll_seconds: float = max(0.25, float(os.environ.get("POLL_SECONDS", "2")))
    submit_interval_seconds: float = max(
        3.0, float(os.environ.get("SUBMIT_INTERVAL_SECONDS", "3.2"))
    )
    calibration_path: Path = Path(
        os.environ.get(
            "CALIBRATION_PATH",
            str(Path(__file__).with_name("calibration.json")),
        )
    )


@dataclass
class TileRecord:
    task_id: str
    category: str
    points: int
    cell_key: tuple[str, int]
    leading_variant: bool
    state: TaskState = TaskState.QUEUED
    attempts: int = 0
    incorrect_answers: set[str] = field(default_factory=set)
    next_eligible: float = 0.0
    candidate: CandidateAnswer | None = None
    last_error: str = ""
    last_duration: float = 0.0


class Calibration:
    """Anonymous category/tier solve-rate and latency aggregates."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict[str, Any]] = {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data = loaded
        except (FileNotFoundError, ValueError, OSError):
            self.data = {}

    @staticmethod
    def key(category: str, points: int) -> str:
        return f"{category}|{points}"

    def estimate(self, category: str, points: int) -> tuple[float, float]:
        row = self.data.get(self.key(category, points), {})
        attempts = max(0, int(row.get("attempts", 0)))
        correct = max(0, int(row.get("correct", 0)))
        # Beta(3,2) prior: optimistic enough to explore uncalibrated cells.
        success = (correct + 3) / (attempts + 5)
        durations = [
            float(value)
            for value in row.get("durations", [])
            if isinstance(value, (int, float)) and value > 0
        ]
        latency = statistics.median(durations) if durations else 45.0
        return success, max(5.0, latency)

    def record(
        self, category: str, points: int, correct: bool, duration: float
    ) -> None:
        key = self.key(category, points)
        row = self.data.setdefault(
            key, {"attempts": 0, "correct": 0, "durations": []}
        )
        row["attempts"] = int(row.get("attempts", 0)) + 1
        row["correct"] = int(row.get("correct", 0)) + int(correct)
        durations = list(row.get("durations", []))
        if duration > 0:
            durations.append(round(float(duration), 3))
        row["durations"] = durations[-50:]
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


class CompetitiveAgent:
    def __init__(
        self,
        config: Config | None = None,
        solver: TileSolver | None = None,
        monotonic: Any = time.monotonic,
    ):
        self.config = config or Config()
        self.solver = solver or TileSolver(
            SolverConfig(
                max_turns=self.config.max_turns,
                verbose=self.config.verbose,
            )
        )
        self.monotonic = monotonic
        self.records: dict[str, TileRecord] = {}
        self.active: dict[str, asyncio.Task[SolveOutcome]] = {}
        self.admitted: set[str] = set()
        self.latest_board: dict[str, Any] | None = None
        self.latest_open_ids: set[str] = set()
        self.phase = "setup"
        self.last_submit_at = float("-inf")
        self.calibration = Calibration(self.config.calibration_path)
        self._board_failures = 0

    async def run(self) -> None:
        jp.log(
            "competitive agent starting:",
            f"workers={self.config.workers}",
            f"max_turns={self.config.max_turns}",
            f"tile_timeout={self.config.tile_timeout_seconds}s",
        )
        while True:
            loop_started = self.monotonic()
            try:
                board = await asyncio.to_thread(jp.board)
                self._board_failures = 0
                self.reconcile(board)
                await self.harvest()
                await self.submit_ready()
                self.dispatch()
            except jp.AuthError:
                raise
            except Exception as exc:  # noqa: BLE001 - unattended outer loop
                self._board_failures += 1
                delay = min(30.0, 2.0 ** min(self._board_failures, 5))
                jp.log(f"board loop error: {exc!r}; retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            elapsed = self.monotonic() - loop_started
            await asyncio.sleep(max(0.0, self.config.poll_seconds - elapsed))

    def reconcile(self, board: dict[str, Any]) -> None:
        self.latest_board = board
        self.phase = str(board.get("phase", "setup"))
        open_tiles = jp.open_tiles(board)
        allowed = set(self.config.task_filter)
        if allowed:
            open_tiles = [tile for tile in open_tiles if tile["id"] in allowed]
        current_ids = {str(tile["id"]) for tile in open_tiles}
        self.latest_open_ids = current_ids
        now = self.monotonic()

        for tile in open_tiles:
            task_id = str(tile["id"])
            if task_id not in self.records:
                if (
                    not allowed
                    and self.config.max_tiles > 0
                    and len(self.admitted) >= self.config.max_tiles
                ):
                    continue
                open_ids = [str(value) for value in tile.get("open_ids", [])]
                self.records[task_id] = TileRecord(
                    task_id=task_id,
                    category=str(tile.get("category", "")),
                    points=int(tile.get("points", 0)),
                    cell_key=(
                        str(tile.get("category", "")),
                        int(tile.get("points", 0)),
                    ),
                    leading_variant=not open_ids or open_ids[0] == task_id,
                )
                self.admitted.add(task_id)
            record = self.records[task_id]
            if (
                record.state == TaskState.COOLDOWN
                and record.next_eligible <= now
            ):
                record.state = TaskState.QUEUED

        for task_id, record in self.records.items():
            if task_id in current_ids or record.state == TaskState.TERMINAL:
                continue
            record.state = TaskState.TERMINAL
            record.candidate = None
            record.last_error = "tile no longer open"

        if self.config.verbose:
            counts = {
                state.value: sum(r.state == state for r in self.records.values())
                for state in TaskState
            }
            jp.log(
                f"phase={self.phase} open={len(current_ids)} "
                f"admitted={len(self.admitted)} states={counts}"
            )

    def priority(self, record: TileRecord) -> float:
        success, latency = self.calibration.estimate(
            record.category, record.points
        )
        score = record.points * success / latency
        if record.leading_variant:
            score *= 1.15
        active_in_cell = any(
            other.task_id != record.task_id
            and other.cell_key == record.cell_key
            and other.state in {TaskState.RUNNING, TaskState.CANDIDATE}
            for other in self.records.values()
        )
        if not active_in_cell:
            score *= 1.25
        score *= 0.5 ** len(record.incorrect_answers)
        return score

    def dispatch(self) -> None:
        capacity = self.config.workers - len(self.active)
        if capacity <= 0:
            return
        now = self.monotonic()
        eligible = [
            record
            for record in self.records.values()
            if record.state == TaskState.QUEUED
            and record.task_id in self.latest_open_ids
            and record.next_eligible <= now
        ]
        eligible.sort(
            key=lambda record: (self.priority(record), record.points),
            reverse=True,
        )
        for record in eligible[:capacity]:
            record.state = TaskState.RUNNING
            record.attempts += 1
            self.active[record.task_id] = asyncio.create_task(
                self._solve(record), name=f"solve-{record.task_id}"
            )

    async def _solve(self, record: TileRecord) -> SolveOutcome:
        try:
            detail = await asyncio.to_thread(jp.task, record.task_id)
            workdir = jp.workdir(record.task_id)
            await asyncio.to_thread(jp.fetch_files, record.task_id, detail)
            outcome = await asyncio.wait_for(
                asyncio.to_thread(
                    self.solver.solve,
                    detail,
                    workdir,
                    self.phase,
                    set(record.incorrect_answers),
                ),
                timeout=self.config.tile_timeout_seconds,
            )
            return outcome
        except asyncio.TimeoutError:
            return SolveOutcome(
                None,
                self.config.tile_timeout_seconds,
                "tile_timeout",
                "tile exceeded wall-clock budget",
            )
        except jp.AuthError:
            raise
        except jp.TileUnavailable as exc:
            return SolveOutcome(None, 0.0, "tile_unavailable", str(exc))
        except Exception as exc:  # noqa: BLE001 - isolate a tile
            return SolveOutcome(None, 0.0, "tile_error", repr(exc))

    async def harvest(self) -> None:
        for task_id, task in list(self.active.items()):
            if not task.done():
                continue
            del self.active[task_id]
            record = self.records[task_id]
            try:
                outcome = task.result()
            except jp.AuthError:
                raise
            except Exception as exc:  # noqa: BLE001
                outcome = SolveOutcome(None, 0.0, "worker_error", repr(exc))
            if record.state == TaskState.TERMINAL:
                continue
            record.last_duration = outcome.elapsed_seconds
            if outcome.candidate is not None:
                record.candidate = outcome.candidate
                record.state = TaskState.CANDIDATE
                record.next_eligible = self.monotonic()
                jp.log(
                    f"{task_id}: verified candidate ready "
                    f"(confidence={outcome.candidate.confidence:.2f})"
                )
            else:
                record.last_error = (
                    f"{outcome.failure_code or 'solve_failed'}: {outcome.detail}"
                )
                backoff = min(60.0, 5.0 * (2 ** min(record.attempts - 1, 4)))
                record.next_eligible = self.monotonic() + backoff
                record.state = TaskState.COOLDOWN
                jp.log(f"{task_id}: {record.last_error}; retry in {backoff:.1f}s")

    async def submit_ready(self) -> None:
        now = self.monotonic()
        if now - self.last_submit_at < self.config.submit_interval_seconds:
            return
        ready = [
            record
            for record in self.records.values()
            if record.state == TaskState.CANDIDATE
            and record.candidate is not None
            and record.next_eligible <= now
        ]
        if not ready:
            return
        ready.sort(key=self.priority, reverse=True)
        record = ready[0]

        fresh = await asyncio.to_thread(jp.board)
        self.reconcile(fresh)
        if (
            record.state == TaskState.TERMINAL
            or record.task_id not in self.latest_open_ids
        ):
            return
        candidate = record.candidate
        if candidate is None:
            return
        self.last_submit_at = self.monotonic()
        result = await asyncio.to_thread(
            jp.submit, record.task_id, candidate.value
        )
        outcome = str(result.get("result", ""))
        jp.log(
            f"{record.task_id}: submitted {candidate.value[:60]!r} -> {outcome}"
        )

        if outcome == "correct":
            record.state = TaskState.TERMINAL
            record.candidate = None
            self.calibration.record(
                record.category, record.points, True, record.last_duration
            )
            return
        if outcome == "incorrect":
            record.incorrect_answers.add(candidate.value)
            record.candidate = None
            misses = len(record.incorrect_answers)
            default = 10.0 if self.phase == "practice" else min(
                480.0, 30.0 * (2 ** (misses - 1))
            )
            record.next_eligible = self.monotonic() + self._retry_in(
                result, default
            )
            record.state = TaskState.COOLDOWN
            self.calibration.record(
                record.category, record.points, False, record.last_duration
            )
            return
        if outcome in {"rate_limited", "locked", "locked_out"}:
            record.next_eligible = self.monotonic() + self._retry_in(result, 4.0)
            record.state = TaskState.CANDIDATE
            return
        if outcome in {
            "already_claimed",
            "voided",
            "wrong_phase",
            "forbidden",
            "unknown_task",
        }:
            record.state = TaskState.TERMINAL
            record.candidate = None
            return
        record.last_error = f"unexpected submit result: {result!r}"
        record.next_eligible = self.monotonic() + 5.0
        record.state = TaskState.CANDIDATE

    @staticmethod
    def _retry_in(result: dict[str, Any], default: float) -> float:
        try:
            return max(0.1, float(result.get("retry_in", default)))
        except (TypeError, ValueError):
            return default


def main() -> None:
    try:
        asyncio.run(CompetitiveAgent().run())
    except KeyboardInterrupt:
        jp.log("stopped by operator")
    except jp.AuthError as exc:
        raise SystemExit(f"[auth] {exc}") from exc


if __name__ == "__main__":
    main()
