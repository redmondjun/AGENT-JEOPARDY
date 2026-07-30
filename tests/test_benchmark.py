from __future__ import annotations

from scripts.replay_benchmark import calculate, parse_log
from scripts.select_worker_count import select


def test_replay_benchmark_calculates_correctness_and_penalty(tmp_path) -> None:
    log = tmp_path / "practice.log"
    log.write_text(
        "\n".join(
            [
                "[10:00:00] event=dispatch task=Q-A1 category='Cryptic' points=100",
                "[10:00:10] event=worker_complete task=Q-A1 outcome=ready "
                "elapsed_ms=10000 input_tokens=1000 output_tokens=100",
                "[10:00:12] event=submission task=Q-A1 action=solved reason=correct",
                "[10:00:15] event=dispatch task=Q-A2 category='Cryptic' points=200",
                "[10:00:25] event=worker_complete task=Q-A2 outcome=ready "
                "elapsed_ms=10000 input_tokens=1200 output_tokens=100",
                "[10:00:30] event=submission task=Q-A2 action=retry reason=incorrect",
                "[10:01:00] event=cycle kind=heartbeat",
            ]
        )
    )
    tiles, elapsed = parse_log(log)
    report = calculate(tiles, elapsed, available_points=100_000)

    assert report["correct"] == 1
    assert report["earned_points"] == 100
    assert report["penalty_points"] == 50
    assert report["point_weighted_first_submission_correctness"] == 1 / 3
    assert report["median_tokens_per_correct_tile"] == 1100


def test_worker_selector_requires_full_eight_worker_gate() -> None:
    six = {
        "net_points_per_minute": 1000,
        "p95_model_latency_seconds": 10,
        "timeout_rate": 0.01,
        "error_rate": 0.01,
    }
    eight = {
        "net_points_per_minute": 1150,
        "token_rate_per_minute": 70_000,
        "p95_model_latency_seconds": 11,
        "timeout_rate": 0.01,
        "error_rate": 0.01,
    }
    assert select(six, eight) == (8, [])

    eight["token_rate_per_minute"] = 80_000
    workers, reasons = select(six, eight)
    assert workers == 6
    assert reasons
