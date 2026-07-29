"""Bounded process execution (TEAM_PLAN #10.2): finite timeout, capped
stdout/stderr, process-group termination on timeout, workdir locked to the
tile, no shell interpolation, and a semaphore capping CPU-heavy jobs at 2
concurrent (2 CPUs per README's hosted-container spec).

`argv` is always `list[str]` — never a shell string — so there is no
injection surface to worry about; `subprocess.Popen` is called without
`shell=True` anywhere in this module.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tools.runtime.errors import (
    INVALID_ARGUMENT, NOT_FOUND, PROCESS_FAILED, TIMEOUT, RuntimeToolError,
)
from tools.runtime.limits import truncation_marker

DEFAULT_MAX_OUTPUT_BYTES = 200_000

# 2 CPUs per the hosted container (README "Your agent's environment") — this
# bounds concurrent CPU-heavy subprocess launches across the whole process,
# not per-tile, since two tiles racing two heavy jobs each would oversubscribe
# the same two cores regardless of which tile asked.
CPU_HEAVY_CONCURRENCY = 2
_cpu_semaphore = threading.Semaphore(CPU_HEAVY_CONCURRENCY)


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    exit_code: int | None  # None if killed before it could exit
    stdout: str
    stdout_truncated: bool
    stderr: str
    stderr_truncated: bool
    timed_out: bool
    elapsed_ms: int


def run_process(workdir: Path, argv: list[str], *, timeout_seconds: float,
                env: dict[str, str] | None = None,
                max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
                cpu_heavy: bool = True) -> ProcessResult:
    """Run `argv` with cwd locked to `workdir`. Raises RuntimeToolError for
    argument problems and unlaunchable commands; a timeout is NOT an
    exception — it comes back as `ProcessResult(timed_out=True, ...)` so the
    caller sees whatever partial output was captured before the kill.
    """
    _validate_argv(argv)
    if timeout_seconds <= 0:
        raise RuntimeToolError(INVALID_ARGUMENT, "timeout_seconds must be positive")
    if max_output_bytes <= 0:
        raise RuntimeToolError(INVALID_ARGUMENT, "max_output_bytes must be positive")

    workdir = workdir.resolve(strict=True)

    acquired = False
    if cpu_heavy:
        acquired = _cpu_semaphore.acquire(timeout=timeout_seconds)
        if not acquired:
            raise RuntimeToolError(
                TIMEOUT,
                f"{CPU_HEAVY_CONCURRENCY} CPU-heavy jobs already running; "
                f"no slot freed within {timeout_seconds}s")
    try:
        return _run(workdir, argv, timeout_seconds, env, max_output_bytes)
    finally:
        if acquired:
            _cpu_semaphore.release()


def _validate_argv(argv: list[str]) -> None:
    if isinstance(argv, (str, bytes)) or not argv:
        raise RuntimeToolError(
            INVALID_ARGUMENT,
            "argv must be a non-empty list[str] — no shell strings")
    if not all(isinstance(a, str) for a in argv):
        raise RuntimeToolError(INVALID_ARGUMENT, "every argv element must be a str")


def _run(workdir: Path, argv: list[str], timeout_seconds: float,
         env: dict[str, str] | None, max_output_bytes: int) -> ProcessResult:
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv, cwd=str(workdir), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,  # own process group -> killable as a tree
        )
    except FileNotFoundError as e:
        raise RuntimeToolError(NOT_FOUND, f"executable not found: {argv[0]!r}") from e
    except OSError as e:
        raise RuntimeToolError(PROCESS_FAILED, f"failed to start {argv[0]!r}: {e}") from e

    out_box: dict = {}
    err_box: dict = {}
    t_out = threading.Thread(target=_drain, args=(proc.stdout, max_output_bytes, out_box), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, max_output_bytes, err_box), daemon=True)
    t_out.start()
    t_err.start()

    timed_out = False
    exit_code: int | None
    try:
        exit_code = proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        try:
            exit_code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            exit_code = None  # still wouldn't die; nothing more we can do

    t_out.join(timeout=5)
    t_err.join(timeout=5)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    stdout_text, stdout_trunc = _decode(out_box)
    stderr_text, stderr_trunc = _decode(err_box)

    return ProcessResult(
        argv=tuple(argv), exit_code=exit_code,
        stdout=stdout_text, stdout_truncated=stdout_trunc,
        stderr=stderr_text, stderr_truncated=stderr_trunc,
        timed_out=timed_out, elapsed_ms=elapsed_ms,
    )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the whole process group `start_new_session=True` created, not
    just the direct child — a child that forks (a shell script, a compiler
    driver) leaves orphaned descendants behind otherwise.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _drain(stream, max_bytes: int, box: dict) -> None:
    """Read a pipe to EOF in a background thread, keeping only the first
    `max_bytes` but continuing to read (and discard) past that so the child
    never blocks writing into a full pipe buffer after we've stopped caring.
    """
    chunks: list[bytes] = []
    kept = 0
    total = 0
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if kept < max_bytes:
                take = chunk[: max_bytes - kept]
                chunks.append(take)
                kept += len(take)
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass
    box["data"] = b"".join(chunks)
    box["total_bytes"] = total


def _decode(box: dict) -> tuple[str, bool]:
    data: bytes = box.get("data", b"")
    total: int = box.get("total_bytes", len(data))
    truncated = total > len(data)
    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += truncation_marker(total, len(data))
    return text, truncated
