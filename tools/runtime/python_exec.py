"""Python execution, specialized on top of `processes.py`: the model hands
over a snippet of code, we run it as a real subprocess (never `exec()` in
this process — that would share memory/timeout/GIL with the agent itself)
using the same interpreter running the agent, cwd locked to the task workdir.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

from tools.runtime import files
from tools.runtime.errors import NOT_FOUND, RuntimeToolError
from tools.runtime.paths import resolve_in_workdir
from tools.runtime.processes import DEFAULT_MAX_OUTPUT_BYTES, ProcessResult, run_process

_SCRATCH_DIR = "_scratch_py"


def run_python_code(workdir: Path, code: str, *, timeout_seconds: float,
                    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
                    args: list[str] | None = None) -> ProcessResult:
    """Write `code` to a scratch file inside the workdir and run it with the
    current interpreter. The file is left behind under `_scratch_py/` (task
    scratch space is reused across a session, not wiped per call) so a later
    tool call in the same tile can read its output files back.
    """
    rel_path = f"{_SCRATCH_DIR}/{uuid.uuid4().hex}.py"
    files.write_scratch(workdir, rel_path, code)
    return run_python_file(workdir, rel_path, args=args,
                           timeout_seconds=timeout_seconds,
                           max_output_bytes=max_output_bytes)


def run_python_file(workdir: Path, rel_path: str, *, timeout_seconds: float,
                    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
                    args: list[str] | None = None) -> ProcessResult:
    """Run an existing `.py` file already inside the workdir (e.g. one the
    model wrote earlier, or extracted from a task archive).
    """
    script = resolve_in_workdir(workdir, rel_path)
    if not script.is_file():
        raise RuntimeToolError(NOT_FOUND, f"no such script: {rel_path!r}")
    argv = [sys.executable, str(script), *(args or [])]
    return run_process(workdir, argv, timeout_seconds=timeout_seconds,
                       max_output_bytes=max_output_bytes, cpu_heavy=True)
