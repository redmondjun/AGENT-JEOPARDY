"""Bounded daemon worker scheduling with hard orchestration deadlines."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class WorkerTimedOut(TimeoutError):
    pass


class WorkerRetired(RuntimeError):
    """The tile left the active board while its daemon worker was running."""


@dataclass(frozen=True)
class ScheduledWork(Generic[T]):
    task_id: str
    future: Future[T]


@dataclass
class _ActiveWork(Generic[T]):
    task_id: str
    future: Future[T]
    thread: threading.Thread
    started_at: float


class BoundedWorkerPool(Generic[T]):
    """Runs work in daemon threads so a hung dependency cannot pin process exit.

    Python cannot safely kill a running thread. At the configured deadline we
    detach that daemon worker, surface WorkerTimedOut to the orchestrator, and
    free one scheduling slot. Stale-board workers can be retired the same way
    so a phase transition does not block the new board.

    Detached threads are quarantined and still counted against a hard
    ``2 * max_workers`` live-thread ceiling. This gives a new phase one full
    generation of capacity, but repeated transitions or timeouts cannot create
    unbounded daemon threads. Specialist tools must still enforce their own
    network/process timeouts; this is the final liveness boundary.
    """

    def __init__(
        self,
        max_workers: int,
        *,
        work_timeout_seconds: float = 90.0,
        clock=time.monotonic,
        thread_name_prefix: str = "tile",
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if work_timeout_seconds <= 0:
            raise ValueError("work_timeout_seconds must be positive")
        self._max_workers = max_workers
        self._work_timeout_seconds = work_timeout_seconds
        self._clock = clock
        self._thread_name_prefix = thread_name_prefix
        self._counter = 0
        self._lock = threading.RLock()
        self._work: dict[Future[T], _ActiveWork[T]] = {}
        self._retired: list[_ActiveWork[T]] = []

    @property
    def capacity(self) -> int:
        with self._lock:
            self._reap_retired_locked()
            active_capacity = self._max_workers - len(self._work)
            live_capacity = (
                2 * self._max_workers - len(self._work) - len(self._retired)
            )
            return max(0, min(active_capacity, live_capacity))

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._work)

    @property
    def retired_count(self) -> int:
        with self._lock:
            self._reap_retired_locked()
            return len(self._retired)

    def submit(self, task_id: str, function: Callable[[], T]) -> None:
        with self._lock:
            if self.capacity <= 0:
                raise RuntimeError("worker pool is full")
            future: Future[T] = Future()
            self._counter += 1

            def run() -> None:
                if not future.set_running_or_notify_cancel():
                    return
                try:
                    result = function()
                except BaseException as exc:  # Future must preserve exact exception
                    try:
                        future.set_exception(exc)
                    except InvalidStateError:
                        pass  # Deadline monitor won the race.
                else:
                    try:
                        future.set_result(result)
                    except InvalidStateError:
                        pass  # Deadline monitor won the race.

            thread = threading.Thread(
                target=run,
                name=f"{self._thread_name_prefix}-{self._counter}-{task_id}",
                daemon=True,
            )
            self._work[future] = _ActiveWork(
                task_id=task_id,
                future=future,
                thread=thread,
                started_at=self._clock(),
            )
            thread.start()

    def retire_except(self, active_task_ids: set[str]) -> tuple[str, ...]:
        """Quarantine running work whose tile is no longer on the live board."""
        retired: list[str] = []
        with self._lock:
            self._reap_retired_locked()
            for future, active in list(self._work.items()):
                if active.task_id in active_task_ids:
                    continue
                if not future.cancel():
                    try:
                        future.set_exception(
                            WorkerRetired(
                                f"{active.task_id}: tile left the active board"
                            )
                        )
                    except InvalidStateError:
                        pass  # Worker completed between the checks.
                del self._work[future]
                if active.thread.is_alive():
                    self._retired.append(active)
                retired.append(active.task_id)
        return tuple(retired)

    def completed(self) -> list[ScheduledWork[T]]:
        now = self._clock()
        done: list[ScheduledWork[T]] = []
        with self._lock:
            self._reap_retired_locked()
            for future, active in list(self._work.items()):
                if not future.done() and now - active.started_at >= self._work_timeout_seconds:
                    try:
                        future.set_exception(
                            WorkerTimedOut(
                                f"{active.task_id}: worker exceeded "
                                f"{self._work_timeout_seconds:.1f}s"
                            )
                        )
                    except InvalidStateError:
                        pass  # Worker completed between the checks.
                if future.done():
                    del self._work[future]
                    if active.thread.is_alive():
                        self._retired.append(active)
                    done.append(ScheduledWork(task_id=active.task_id, future=future))
        return done

    def futures(self) -> tuple[Future[T], ...]:
        with self._lock:
            return tuple(self._work)

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            active = list(self._work.values()) + list(self._retired)
            self._work.clear()
            self._retired.clear()
        if wait:
            deadline = time.monotonic() + self._work_timeout_seconds
            for work in active:
                work.thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def _reap_retired_locked(self) -> None:
        self._retired = [work for work in self._retired if work.thread.is_alive()]
