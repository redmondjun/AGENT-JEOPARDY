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
        with self.assertRaises(WorkerTimedOut):
            completed[0].future.result()
        self.assertEqual(pool.capacity, 1)
        pool.close(wait=False)


if __name__ == "__main__":
    unittest.main()
