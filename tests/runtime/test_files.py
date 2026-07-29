from __future__ import annotations

import base64
from pathlib import Path

import pytest

from tools.runtime import files
from tools.runtime.errors import (
    INVALID_ARGUMENT, NOT_FOUND, OUTPUT_TOO_LARGE, PATH_BLOCKED, RuntimeToolError,
)


def test_list_dir_metadata(workdir: Path) -> None:
    (workdir / "a.txt").write_text("hi")
    (workdir / "sub").mkdir()
    entries = files.list_dir(workdir)
    names = {e.name for e in entries}
    assert names == {"a.txt", "sub"}
    a = next(e for e in entries if e.name == "a.txt")
    assert a.size == 2 and not a.is_dir


def test_list_dir_not_found(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        files.list_dir(workdir, "nope")
    assert exc.value.code == NOT_FOUND


def test_list_dir_rejects_traversal(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        files.list_dir(workdir, "../../secret")
    assert exc.value.code == PATH_BLOCKED


def test_read_text_whole_file(workdir: Path) -> None:
    (workdir / "a.txt").write_text("hello world")
    text, truncated = files.read_text(workdir, "a.txt")
    assert text == "hello world"
    assert not truncated


def test_read_text_truncates_with_marker(workdir: Path) -> None:
    (workdir / "big.txt").write_text("x" * 1000)
    text, truncated = files.read_text(workdir, "big.txt", max_bytes=100)
    assert truncated
    assert "truncated" in text
    assert text.startswith("x" * 100)


def test_read_text_missing_file(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        files.read_text(workdir, "missing.txt")
    assert exc.value.code == NOT_FOUND


def test_read_bytes_roundtrip(workdir: Path) -> None:
    payload = bytes(range(256))
    (workdir / "b.bin").write_bytes(payload)
    b64, truncated = files.read_bytes(workdir, "b.bin")
    assert not truncated
    assert base64.b64decode(b64) == payload


def test_read_line_range(workdir: Path) -> None:
    (workdir / "lines.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    text, truncated = files.read_line_range(workdir, "lines.txt", 3, 5)
    assert text == "line3\nline4\nline5\n"
    assert not truncated


def test_read_line_range_invalid(workdir: Path) -> None:
    (workdir / "lines.txt").write_text("a\nb\n")
    with pytest.raises(RuntimeToolError) as exc:
        files.read_line_range(workdir, "lines.txt", 5, 2)
    assert exc.value.code == INVALID_ARGUMENT


def test_read_byte_range(workdir: Path) -> None:
    (workdir / "b.bin").write_bytes(b"0123456789")
    b64, truncated = files.read_byte_range(workdir, "b.bin", 2, 4)
    assert base64.b64decode(b64) == b"2345"
    assert not truncated


def test_write_scratch_creates_parents(workdir: Path) -> None:
    n = files.write_scratch(workdir, "out/sub/result.txt", "computed answer")
    assert n == len(b"computed answer")
    assert (workdir / "out" / "sub" / "result.txt").read_text() == "computed answer"


def test_write_scratch_rejects_oversized(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        files.write_scratch(workdir, "big.txt", "x" * 1000, max_bytes=10)
    assert exc.value.code == OUTPUT_TOO_LARGE


def test_write_scratch_rejects_traversal(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        files.write_scratch(workdir, "../../escape.txt", "data")
    assert exc.value.code == PATH_BLOCKED


def test_scratch_reused_across_calls(workdir: Path) -> None:
    """Acceptance: same-session tool calls reuse task scratch data safely."""
    files.write_scratch(workdir, "_scratch/state.txt", "step1")
    text, _ = files.read_text(workdir, "_scratch/state.txt")
    assert text == "step1"
    files.write_scratch(workdir, "_scratch/state.txt", "step2")
    text, _ = files.read_text(workdir, "_scratch/state.txt")
    assert text == "step2"
