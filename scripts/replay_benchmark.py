#!/usr/bin/env python3
"""Score structured agent logs against the 80K readiness gates."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

FIELD_RE = re.compile(r"(\w+)=('(?:[^'\\]|\\.)*'|[^\s]+)")
TIME_RE = re.compile(r"\[(\d\d):(\d\d):(\d\d)\]")
TARGET_MINUTES = 70.0


@dataclass
class Tile:
    category: str = "Unknown"
    points: int = 0
    elapsed_ms: int = 0
    tokens: int = 0
    first_submission: str | None = None
    solved: bool = False


def parse_log(path: Path) -> tuple[dict[str, Tile], float]:
    tiles: dict[str, Tile] = {}
    first_second: int | None = None
    last_second: int | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        time_match = TIME_RE.search(raw_line)
        if time_match:
            hour, minute, second = map(int, time_match.groups())
            value = hour * 3600 + minute * 60 + second
            if first_second is None:
                first_second = value
            if last_second is not None and value < last_second - 43_200:
                value += 86_400
            last_second = value
        fields = {
            key: _unquote(value) for key, value in FIELD_RE.findall(raw_line)
        }
        task_id = fields.get("task")
        if not task_id:
            continue
        tile = tiles.setdefault(task_id, Tile())
        event = fields.get("event")
        if event == "dispatch":
            tile.category = fields.get("category", tile.category)
            tile.points = _integer(fields.get("points"), tile.points)
        elif event == "worker_complete":
            tile.elapsed_ms = max(
                tile.elapsed_ms, _integer(fields.get("elapsed_ms"), 0)
            )
            tile.tokens = max(
                tile.tokens,
                _integer(fields.get("input_tokens"), 0)
                + _integer(fields.get("output_tokens"), 0),
            )
        elif event == "submission":
            action = fields.get("action")
            reason = fields.get("reason")
            outcome = "correct" if action == "solved" else (
                "incorrect" if reason == "incorrect" else None
            )
            if outcome is not None and tile.first_submission is None:
                tile.first_submission = outcome
            if outcome == "correct":
                tile.solved = True
    elapsed = (
        0.0
        if first_second is None or last_second is None
        else max(1.0, float(last_second - first_second))
    )
    return tiles, elapsed


def calculate(
    tiles: dict[str, Tile],
    elapsed_seconds: float,
    *,
    available_points: int,
) -> dict[str, object]:
    submitted = [tile for tile in tiles.values() if tile.first_submission]
    correct = [tile for tile in tiles.values() if tile.solved]
    earned = sum(tile.points for tile in correct)
    penalties = sum(
        round(tile.points * 0.25)
        for tile in submitted
        if tile.first_submission == "incorrect"
    )
    net = earned - penalties
    attempted_value = sum(tile.points for tile in submitted)
    first_correct_value = sum(
        tile.points for tile in submitted if tile.first_submission == "correct"
    )
    weighted_correctness = (
        first_correct_value / attempted_value if attempted_value else 0.0
    )
    minutes = elapsed_seconds / 60.0 if elapsed_seconds else 0.0
    correct_per_minute = len(correct) / minutes if minutes else 0.0
    net_per_minute = net / minutes if minutes else 0.0
    solve_seconds = [tile.elapsed_ms / 1000 for tile in tiles.values() if tile.elapsed_ms]
    token_counts = [tile.tokens for tile in correct if tile.tokens]
    projected = min(
        available_points * weighted_correctness,
        net_per_minute * TARGET_MINUTES,
    )
    penalty_ratio = penalties / earned if earned else 0.0
    report: dict[str, object] = {
        "tiles_seen": len(tiles),
        "submitted": len(submitted),
        "correct": len(correct),
        "coverage_cells": len(
            {(tile.category, tile.points) for tile in tiles.values() if tile.points}
        ),
        "point_weighted_first_submission_correctness": weighted_correctness,
        "correct_tiles_per_minute": correct_per_minute,
        "earned_points": earned,
        "penalty_points": penalties,
        "penalty_ratio": penalty_ratio,
        "net_points": net,
        "net_points_per_minute": net_per_minute,
        "median_solve_seconds": _percentile(solve_seconds, 0.50),
        "p90_solve_seconds": _percentile(solve_seconds, 0.90),
        "median_tokens_per_correct_tile": (
            statistics.median(token_counts) if token_counts else 0
        ),
        "projected_combined_score": round(projected),
    }
    report["gates"] = {
        "weighted_correctness_90pct": weighted_correctness >= 0.90,
        "four_correct_per_minute": correct_per_minute >= 4.0,
        "net_1200_points_per_minute": net_per_minute >= 1_200,
        "median_solve_at_most_30s": report["median_solve_seconds"] <= 30,
        "p90_solve_at_most_90s": report["p90_solve_seconds"] <= 90,
        "median_tokens_below_6000": report["median_tokens_per_correct_tile"] < 6_000,
        "penalties_below_2pct": penalty_ratio < 0.02,
        "all_30_cells_covered": report["coverage_cells"] >= 30,
        "projected_score_80000": report["projected_combined_score"] >= 80_000,
    }
    report["ready"] = all(report["gates"].values())
    return report


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return round(ordered[index], 3)


def _integer(value: str | None, fallback: int) -> int:
    try:
        return int(value) if value is not None else fallback
    except ValueError:
        return fallback


def _unquote(value: str) -> str:
    return value[1:-1].replace("\\'", "'") if value.startswith("'") else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--available-points", type=int, default=100_000)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    tiles, elapsed = parse_log(args.log)
    report = calculate(tiles, elapsed, available_points=args.available_points)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.enforce and not report["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
