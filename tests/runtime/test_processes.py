from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from tools.runtime.errors import INVALID_ARGUMENT, NOT_FOUND, RuntimeToolError
from tools.runtime.processes import CPU_HEAVY_CONCURRENCY, run_process


def test_argv_must_be_a_list_not_a_string(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        run_process(workdir, "echo hi", timeout_seconds=5)  # type: ignore[arg-type]
    assert exc.value.code == INVALID_ARGUMENT


def test_empty_argv_rejected(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        run_process(workdir, [], timeout_seconds=5)
    assert exc.value.code == INVALID_ARGUMENT


def test_missing_executable(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        run_process(workdir, ["/no/such/binary-xyz"], timeout_seconds=5)
    assert exc.value.code == NOT_FOUND


def test_no_shell_interpolation(workdir: Path) -> None:
    """Shell metacharacters in an argv element must be passed through
    literally, never interpreted, because we never call with shell=True.
    """
    payload = "hello; touch /tmp/should_not_exist_$$"
    result = run_process(workdir, [sys.executable, "-c",
                                   "import sys; print(sys.argv[1])", payload],
                         timeout_seconds=10)
    assert result.exit_code == 0
    assert payload in result.stdout


def test_captures_stdout_and_exit_code(workdir: Path) -> None:
    result = run_process(
        workdir, [sys.executable, "-c", "print('hi'); import sys; sys.exit(3)"],
        timeout_seconds=10)
    assert "hi" in result.stdout
    assert result.exit_code == 3
    assert not result.timed_out


def test_cwd_is_locked_to_workdir(workdir: Path) -> None:
    result = run_process(
        workdir, [sys.executable, "-c", "import os; print(os.getcwd())"],
        timeout_seconds=10)
    assert result.stdout.strip() == str(workdir.resolve())


def test_stdout_truncated_keeps_both_ends_with_marker(workdir: Path) -> None:
    result = run_process(
        workdir,
        [sys.executable, "-c", "import sys; sys.stdout.write('x' * 100000)"],
        timeout_seconds=10, max_output_bytes=100)
    assert result.stdout_truncated
    assert "truncated: kept 100 of 100000 bytes" in result.stdout

    # Head and tail share the single byte budget, so both ends of the stream
    # survive and the marker names the gap sitting between them. Asserted as a
    # property rather than a fixed split so the head:tail ratio can be retuned.
    head, separator, remainder = result.stdout.partition("\n...[truncated:")
    assert separator, "expected an in-band truncation marker"
    _, _, tail = remainder.partition("]\n")
    assert head and tail, "both ends of a truncated stream should survive"
    assert set(head) == {"x"} and set(tail) == {"x"}
    assert len(head) + len(tail) == 100


def test_trailing_output_survives_truncation(workdir: Path) -> None:
    """The `ANSWER:` channel lives on the last line, so the tail must survive.

    Head-only retention silently dropped it for any program that printed more
    than the byte budget first, losing the one mechanism that carries an exact
    answer without the model retyping it.
    """
    result = run_process(
        workdir,
        [sys.executable, "-c",
         "import sys; sys.stdout.write('x' * 100000 + '\\nANSWER: keep-me\\n')"],
        timeout_seconds=10, max_output_bytes=100)

    assert result.stdout_truncated
    assert result.stdout.rstrip().endswith("ANSWER: keep-me")
    # It has to come from the retained tail, not from the head, so assert it
    # appears after the marker naming the elided gap.
    assert result.stdout.index("ANSWER: keep-me") > result.stdout.index("truncated:")


def test_timeout_kills_and_reports(workdir: Path) -> None:
    start = time.monotonic()
    result = run_process(
        workdir, [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_seconds=0.5)
    elapsed = time.monotonic() - start
    assert result.timed_out
    assert elapsed < 10  # killed promptly, not left to run the full 60s


def test_timeout_kills_descendant_process(workdir: Path, tmp_path: Path) -> None:
    """A timed-out process's CHILD must die too, not just the direct child we
    launched — acceptance test: hung child processes and descendants are
    terminated.
    """
    pidfile = tmp_path / "child.pid"
    child_code = (
        "import os, pathlib, time\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
        "while True:\n"
        "    time.sleep(0.05)\n"
    )
    parent_code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "time.sleep(60)\n"
    )
    result = run_process(workdir, [sys.executable, "-c", parent_code], timeout_seconds=1.5)
    assert result.timed_out

    deadline = time.monotonic() + 5
    child_pid = None
    while time.monotonic() < deadline:
        if pidfile.exists() and pidfile.read_text().strip():
            child_pid = int(pidfile.read_text().strip())
            break
        time.sleep(0.05)
    assert child_pid is not None, "child never started"

    deadline = time.monotonic() + 5
    dead = False
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            dead = True
            break
        # Minimal containers may not run an init process that promptly reaps
        # orphaned children. A killed descendant can therefore remain in the
        # Linux process table as a zombie even though it is no longer running.
        proc_stat = Path(f"/proc/{child_pid}/stat")
        try:
            if proc_stat.read_text().split()[2] == "Z":
                dead = True
                break
        except (FileNotFoundError, IndexError):
            pass
        time.sleep(0.1)
    assert dead, "descendant process survived the timeout kill"


def test_cpu_heavy_semaphore_caps_concurrency(workdir: Path, tmp_path: Path) -> None:
    """Acceptance: concurrent CPU jobs never exceed CPU_HEAVY_CONCURRENCY.

    Each subprocess reports its own start/end time from INSIDE the process,
    so this measures real execution overlap rather than Python thread
    scheduling around the `run_process` call (which would race ahead of the
    semaphore and show false concurrency).
    """
    logfile = tmp_path / "concurrency.log"
    script = (
        "import time\n"
        f"with open({str(logfile)!r}, 'a') as f:\n"
        "    f.write(f'{time.time()} start\\n')\n"
        "time.sleep(0.4)\n"
        f"with open({str(logfile)!r}, 'a') as f:\n"
        "    f.write(f'{time.time()} end\\n')\n"
    )

    def worker() -> None:
        run_process(workdir, [sys.executable, "-c", script],
                   timeout_seconds=10, cpu_heavy=True)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    events: list[tuple[float, int]] = []
    for line in logfile.read_text().splitlines():
        ts_str, kind = line.split()
        events.append((float(ts_str), 1 if kind == "start" else -1))
    events.sort()

    current = 0
    observed_max = 0
    for _, delta in events:
        current += delta
        observed_max = max(observed_max, current)

    assert len(events) == 12  # 6 workers x (start, end)
    assert observed_max <= CPU_HEAVY_CONCURRENCY
