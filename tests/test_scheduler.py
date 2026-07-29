from __future__ import annotations

import threading
import unittest

from orchestrator.scheduler import BoundedWorkerPool, WorkerTimedOut


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SchedulerTests(unittest.TestCase):
    def test_hung_daemon_worker_expires_and_releases_capacity(self) -> None:
        clock = FakeClock()
        never = threading.Event()
        pool: BoundedWorkerPool[str] = BoundedWorkerPool(
            1, work_timeout_seconds=5.0, clock=clock
        )
        pool.submit("PR-A1", lambda: never.wait() or "impossible")
        self.assertEqual(pool.capacity, 0)
        clock.now = 5.0
        completed = pool.completed()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].elapsed_ms, 5000)
        with self.assertRaises(WorkerTimedOut):
            completed[0].future.result()
        self.assertEqual(pool.capacity, 1)
        pool.close(wait=False)

    def test_retired_phase_worker_frees_capacity_with_bounded_quarantine(self) -> None:
        first = threading.Event()
        second = threading.Event()
        pool: BoundedWorkerPool[str] = BoundedWorkerPool(
            1, work_timeout_seconds=30.0
        )
        try:
            pool.submit("PR-A1", lambda: first.wait() or "practice")
            self.assertEqual(pool.retire_except({"Q-A1"}), ("PR-A1",))
            self.assertEqual(pool.retired_count, 1)
            self.assertEqual(pool.capacity, 1)

            pool.submit("Q-A1", lambda: second.wait() or "qualifier")
            self.assertEqual(pool.retire_except(set()), ("Q-A1",))

            # Two generations (one active + one stale) are the hard ceiling.
            self.assertEqual(pool.retired_count, 2)
            self.assertEqual(pool.capacity, 0)
            with self.assertRaisesRegex(RuntimeError, "worker pool is full"):
                pool.submit("F-A1", lambda: "finale")

            first.set()
            first.wait(0.1)
            for _ in range(100):
                if pool.capacity:
                    break
                threading.Event().wait(0.001)
            self.assertEqual(pool.capacity, 1)
        finally:
            first.set()
            second.set()
            pool.close(wait=False)


if __name__ == "__main__":
    unittest.main()
