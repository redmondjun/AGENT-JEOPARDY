from __future__ import annotations

from pathlib import Path

import pytest

from tools.runtime import python_exec
from tools.runtime.errors import NOT_FOUND, RuntimeToolError


def test_run_python_code_executes_and_captures_stdout(workdir: Path) -> None:
    result = python_exec.run_python_code(workdir, "print('hello from tile')",
                                         timeout_seconds=10)
    assert result.exit_code == 0
    assert "hello from tile" in result.stdout


def test_run_python_code_leaves_scratch_file_behind(workdir: Path) -> None:
    python_exec.run_python_code(workdir, "print(1)", timeout_seconds=10)
    scratch = list((workdir / "_scratch_py").glob("*.py"))
    assert len(scratch) == 1


def test_run_python_code_passes_args(workdir: Path) -> None:
    result = python_exec.run_python_code(
        workdir, "import sys; print(sys.argv[1:])",
        timeout_seconds=10, args=["a", "b"])
    assert "['a', 'b']" in result.stdout


def test_run_python_code_timeout(workdir: Path) -> None:
    result = python_exec.run_python_code(
        workdir, "import time; time.sleep(60)", timeout_seconds=0.5)
    assert result.timed_out


def test_run_python_file_missing(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        python_exec.run_python_file(workdir, "nope.py", timeout_seconds=5)
    assert exc.value.code == NOT_FOUND


def test_run_python_file_runs_existing_script(workdir: Path) -> None:
    (workdir / "script.py").write_text("print('from file')")
    result = python_exec.run_python_file(workdir, "script.py", timeout_seconds=10)
    assert "from file" in result.stdout
