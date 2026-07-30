#!/usr/bin/env python3
"""Choose eight workers only when it clears the agreed benchmark gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select(six: dict, eight: dict) -> tuple[int, list[str]]:
    reasons: list[str] = []
    six_rate = float(six.get("net_points_per_minute", 0))
    eight_rate = float(eight.get("net_points_per_minute", 0))
    improvement = (eight_rate / six_rate - 1.0) if six_rate > 0 else 0.0
    if improvement < 0.10:
        reasons.append(f"points/min improvement {improvement:.1%} is below 10%")
    if float(eight.get("token_rate_per_minute", 0)) >= 80_000:
        reasons.append("eight-worker token rate reaches 80k/min")
    six_p95 = float(six.get("p95_model_latency_seconds", 0))
    eight_p95 = float(eight.get("p95_model_latency_seconds", 0))
    if six_p95 > 0 and eight_p95 > six_p95 * 1.20:
        reasons.append("eight-worker p95 model latency regresses by more than 20%")
    if float(eight.get("timeout_rate", 0)) > float(six.get("timeout_rate", 0)):
        reasons.append("eight-worker timeout rate is higher")
    if float(eight.get("error_rate", 0)) > float(six.get("error_rate", 0)):
        reasons.append("eight-worker error rate is higher")
    return (6 if reasons else 8), reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("six_worker_report", type=Path)
    parser.add_argument("eight_worker_report", type=Path)
    args = parser.parse_args()
    six = json.loads(args.six_worker_report.read_text())
    eight = json.loads(args.eight_worker_report.read_text())
    workers, reasons = select(six, eight)
    print(json.dumps({"selected_workers": workers, "reasons": reasons}, indent=2))


if __name__ == "__main__":
    main()
