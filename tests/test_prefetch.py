from __future__ import annotations

import threading
import time
import unittest

from orchestrator.prefetch import TaskDetailPrefetcher


class TaskDetailPrefetcherTests(unittest.TestCase):
    def test_completed_prefetch_is_used_without_second_fetch(self) -> None:
        calls: list[str] = []

        def fetch(task_id: str) -> dict:
            calls.append(task_id)
            return {"id": task_id, "prompt": "safe"}

        prefetcher = TaskDetailPrefetcher(fetch, max_workers=1)
        try:
            prefetcher.retain({"Q-A1"})
            self.assertEqual(prefetcher.schedule(["Q-A1"]), 1)
            for _ in range(100):
                if prefetcher.snapshot().completed:
                    break
                time.sleep(0.001)
            detail, source = prefetcher.get("Q-A1")
            self.assertEqual(detail["id"], "Q-A1")
            self.assertEqual(source, "cache")
            self.assertEqual(calls, ["Q-A1"])
        finally:
            prefetcher.close()

    def test_inflight_fetch_is_joined(self) -> None:
        release = threading.Event()

        def fetch(task_id: str) -> dict:
            release.wait(1)
            return {"id": task_id}

        prefetcher = TaskDetailPrefetcher(
            fetch, max_workers=1, join_seconds=0.5
        )
        try:
            prefetcher.retain({"Q-A1"})
            prefetcher.schedule(["Q-A1"])
            timer = threading.Timer(0.01, release.set)
            timer.start()
            detail, source = prefetcher.get("Q-A1")
            timer.join()
            self.assertEqual(detail["id"], "Q-A1")
            self.assertEqual(source, "joined")
        finally:
            release.set()
            prefetcher.close()

    def test_failed_prefetch_does_not_poison_direct_retry(self) -> None:
        calls = 0

        def fetch(task_id: str) -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary")
            return {"id": task_id}

        prefetcher = TaskDetailPrefetcher(fetch, max_workers=1)
        try:
            prefetcher.retain({"Q-A1"})
            prefetcher.schedule(["Q-A1"])
            for _ in range(100):
                prefetcher.collect()
                if calls:
                    break
                time.sleep(0.001)
            detail, source = prefetcher.get("Q-A1")
            self.assertEqual(detail["id"], "Q-A1")
            self.assertEqual(source, "network")
            self.assertEqual(calls, 2)
        finally:
            prefetcher.close()

    def test_retain_prunes_stale_cache(self) -> None:
        prefetcher = TaskDetailPrefetcher(
            lambda task_id: {"id": task_id}, max_workers=1
        )
        try:
            prefetcher.retain({"PR-A1"})
            prefetcher.schedule(["PR-A1"])
            for _ in range(100):
                if prefetcher.snapshot().completed:
                    break
                time.sleep(0.001)
            prefetcher.retain({"Q-A1"})
            _, source = prefetcher.get("PR-A1")
            self.assertEqual(source, "network")
        finally:
            prefetcher.close()

    def test_stale_inflight_fetch_does_not_open_hidden_capacity(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def fetch(task_id: str) -> dict:
            if task_id == "PR-A1":
                started.set()
                release.wait(1)
            return {"id": task_id}

        prefetcher = TaskDetailPrefetcher(fetch, max_workers=1)
        try:
            prefetcher.retain({"PR-A1"})
            self.assertEqual(prefetcher.schedule(["PR-A1"]), 1)
            self.assertTrue(started.wait(1))

            prefetcher.retain({"PR-B1"})
            self.assertEqual(prefetcher.schedule(["PR-B1"]), 0)

            release.set()
            for _ in range(100):
                prefetcher.collect()
                if prefetcher.schedule(["PR-B1"]) == 1:
                    break
                time.sleep(0.001)
            else:
                self.fail("prefetch capacity did not recover")
        finally:
            release.set()
            prefetcher.close()


if __name__ == "__main__":
    unittest.main()
