"""Bounded, model-free task-detail prefetching."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from .scheduler import BoundedWorkerPool


@dataclass(frozen=True)
class PrefetchStats:
    scheduled: int = 0
    completed: int = 0
    cache_hits: int = 0
    joined_hits: int = 0
    misses: int = 0
    failures: int = 0


class TaskDetailPrefetcher:
    """Warm task JSON without occupying any tile-solver worker."""

    def __init__(
        self,
        fetch: Callable[[str], dict[str, Any]],
        *,
        max_workers: int = 2,
        timeout_seconds: float = 10.0,
        join_seconds: float = 0.25,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        if join_seconds < 0:
            raise ValueError("prefetch join_seconds must be non-negative")
        self._fetch = fetch
        self._join_seconds = join_seconds
        self._logger = logger or (lambda _message: None)
        self._pool: BoundedWorkerPool[dict[str, Any]] = BoundedWorkerPool(
            max_workers,
            work_timeout_seconds=timeout_seconds,
            thread_name_prefix="prefetch",
        )
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._events: dict[str, threading.Event] = {}
        self._allowed: set[str] = set()
        self._stats = PrefetchStats()

    def retain(self, task_ids: set[str]) -> None:
        with self._lock:
            self._allowed = set(task_ids)
            self._cache = {
                task_id: detail
                for task_id, detail in self._cache.items()
                if task_id in task_ids
            }
        # Do not retire in-flight fetches: Python cannot stop their threads,
        # and freeing those slots early would violate the two-worker bound.
        # Obsolete results finish normally but _allowed prevents caching them.
        self.collect()

    def schedule(self, task_ids: list[str]) -> int:
        self.collect()
        scheduled = 0
        for task_id in task_ids:
            with self._lock:
                if task_id in self._cache or task_id in self._events:
                    continue
                if self._pool.capacity <= 0:
                    break
                self._allowed.add(task_id)
                self._events[task_id] = threading.Event()
                self._stats = replace(
                    self._stats, scheduled=self._stats.scheduled + 1
                )
            try:
                self._pool.submit(
                    task_id,
                    lambda selected=task_id: self._fetch_and_store(selected),
                )
            except Exception:
                with self._lock:
                    event = self._events.pop(task_id, None)
                    if event is not None:
                        event.set()
                raise
            scheduled += 1
        return scheduled

    def get(self, task_id: str) -> tuple[dict[str, Any], str]:
        with self._lock:
            cached = self._cache.get(task_id)
            event = self._events.get(task_id)
            if cached is not None:
                self._stats = replace(
                    self._stats, cache_hits=self._stats.cache_hits + 1
                )
                return dict(cached), "cache"

        if event is not None and event.wait(self._join_seconds):
            with self._lock:
                cached = self._cache.get(task_id)
                if cached is not None:
                    self._stats = replace(
                        self._stats, joined_hits=self._stats.joined_hits + 1
                    )
                    return dict(cached), "joined"

        try:
            detail = dict(self._fetch(task_id))
        except Exception:
            with self._lock:
                self._stats = replace(
                    self._stats, failures=self._stats.failures + 1
                )
            raise
        with self._lock:
            if task_id in self._allowed:
                self._cache[task_id] = detail
            self._stats = replace(self._stats, misses=self._stats.misses + 1)
        return dict(detail), "network"

    def collect(self) -> int:
        completed = 0
        for work in self._pool.completed():
            completed += 1
            try:
                work.future.result()
            except Exception as exc:  # optimization failures never affect tiles
                with self._lock:
                    self._stats = replace(
                        self._stats, failures=self._stats.failures + 1
                    )
                self._logger(
                    f"event=prefetch task={work.task_id} outcome=failed "
                    f"error_type={type(exc).__name__} elapsed_ms={work.elapsed_ms}"
                )
        return completed

    def snapshot(self) -> PrefetchStats:
        with self._lock:
            return replace(self._stats)

    def close(self) -> None:
        self._pool.close(wait=False)
        with self._lock:
            for event in self._events.values():
                event.set()
            self._events.clear()
            self._cache.clear()
            self._allowed.clear()

    def _fetch_and_store(self, task_id: str) -> dict[str, Any]:
        started = time.monotonic()
        try:
            detail = dict(self._fetch(task_id))
            with self._lock:
                if task_id in self._allowed:
                    self._cache[task_id] = detail
                self._stats = replace(
                    self._stats, completed=self._stats.completed + 1
                )
            self._logger(
                f"event=prefetch task={task_id} outcome=ready "
                f"elapsed_ms={int((time.monotonic() - started) * 1000)}"
            )
            return detail
        finally:
            with self._lock:
                event = self._events.pop(task_id, None)
                if event is not None:
                    event.set()
