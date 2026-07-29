"""File tools scoped to `TaskContext.workdir` (TEAM_PLAN #10.1).

Every function takes `workdir` and a path *relative to it*; every path goes
through `paths.resolve_in_workdir` before touching disk, so traversal and
absolute paths fail the same way no matter which function is called.
Functions return plain Python values (list/str/bytes/dict) — never
`ToolResult` — so they stay unit-testable without importing contracts.py.
`tool.py` is the only place that wraps these into `ToolResult`.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from tools.runtime.errors import (
    INVALID_ARGUMENT, NOT_FOUND, OUTPUT_TOO_LARGE, RuntimeToolError,
)
from tools.runtime.limits import truncate, truncation_marker
from tools.runtime.paths import resolve_in_workdir

DEFAULT_MAX_READ_BYTES = 200_000
DEFAULT_MAX_LIST_ENTRIES = 2_000
DEFAULT_MAX_WRITE_BYTES = 10_000_000


@dataclass(frozen=True)
class Entry:
    name: str
    path: str  # relative to workdir, forward-slash separated
    is_dir: bool
    size: int
    mtime: float


def list_dir(workdir: Path, sub_path: str = ".", *,
             max_entries: int = DEFAULT_MAX_LIST_ENTRIES) -> list[Entry]:
    """List one directory's immediate metadata (name, size, mtime, is_dir).
    Not recursive — the model asks again for a subdirectory it cares about,
    which keeps one call bounded regardless of tree size.
    """
    target = resolve_in_workdir(workdir, sub_path)
    if not target.exists():
        raise RuntimeToolError(NOT_FOUND, f"no such path: {sub_path!r}")
    if not target.is_dir():
        raise RuntimeToolError(INVALID_ARGUMENT, f"not a directory: {sub_path!r}")

    entries: list[Entry] = []
    for i, child in enumerate(sorted(target.iterdir(), key=lambda p: p.name)):
        if i >= max_entries:
            break
        stat = child.stat()
        rel = child.relative_to(workdir.resolve(strict=True))
        entries.append(Entry(
            name=child.name,
            path=rel.as_posix(),
            is_dir=child.is_dir(),
            size=stat.st_size,
            mtime=stat.st_mtime,
        ))
    return entries


def read_text(workdir: Path, path: str, *,
              max_bytes: int = DEFAULT_MAX_READ_BYTES,
              encoding: str = "utf-8") -> tuple[str, bool]:
    """Read a file as text, capped at `max_bytes`. Returns (text, truncated).
    Decoding errors are replaced, not raised — a partial read of a binary
    file should not crash the tool, `read_bytes` is the right call for that.
    """
    data, truncated = _read_capped(workdir, path, max_bytes)
    text = data.decode(encoding, errors="replace")
    if truncated:
        text += truncation_marker(_size_of(workdir, path), len(data))
    return text, truncated


def read_bytes(workdir: Path, path: str, *,
               max_bytes: int = DEFAULT_MAX_READ_BYTES) -> tuple[str, bool]:
    """Read a file as base64, capped at `max_bytes` of *raw* file content
    (i.e. before base64 expansion). Returns (base64_text, truncated).
    """
    data, truncated = _read_capped(workdir, path, max_bytes)
    return base64.b64encode(data).decode("ascii"), truncated


def read_line_range(workdir: Path, path: str, start_line: int, end_line: int,
                    *, max_bytes: int = DEFAULT_MAX_READ_BYTES) -> tuple[str, bool]:
    """1-indexed, inclusive line range. For the "which lines matter" case in
    a file too large to read whole.
    """
    if start_line < 1 or end_line < start_line:
        raise RuntimeToolError(
            INVALID_ARGUMENT,
            f"invalid line range: start={start_line} end={end_line}")

    target = _existing_file(workdir, path)
    out_lines: list[str] = []
    truncated = False
    kept_bytes = 0
    with target.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            if lineno < start_line:
                continue
            if lineno > end_line:
                break
            kept_bytes += len(line.encode("utf-8"))
            if kept_bytes > max_bytes:
                truncated = True
                break
            out_lines.append(line)
    text = "".join(out_lines)
    if truncated:
        text += truncation_marker(kept_bytes, max_bytes)
    return text, truncated


def read_byte_range(workdir: Path, path: str, offset: int, length: int, *,
                    max_bytes: int = DEFAULT_MAX_READ_BYTES) -> tuple[str, bool]:
    """Base64 of `length` bytes starting at `offset`, capped at `max_bytes`."""
    if offset < 0 or length < 0:
        raise RuntimeToolError(
            INVALID_ARGUMENT, f"invalid byte range: offset={offset} length={length}")
    target = _existing_file(workdir, path)
    read_len = min(length, max_bytes)
    truncated = read_len < length
    with target.open("rb") as fh:
        fh.seek(offset)
        data = fh.read(read_len)
    return base64.b64encode(data).decode("ascii"), truncated


def write_scratch(workdir: Path, path: str, content: str, *,
                  max_bytes: int = DEFAULT_MAX_WRITE_BYTES) -> int:
    """Write a text scratch file inside the workdir. Returns bytes written.
    Parent directories are created as needed (still inside the workdir, since
    the path already passed containment). Refuses to write through a symlink
    that would land outside the workdir — resolve_in_workdir already blocks
    that — and refuses payloads over `max_bytes` rather than truncating a
    write, since a silently-shortened scratch file is worse than a loud
    rejection.
    """
    data = content.encode("utf-8")
    if len(data) > max_bytes:
        raise RuntimeToolError(
            OUTPUT_TOO_LARGE,
            f"write of {len(data)} bytes exceeds the {max_bytes}-byte limit")
    target = resolve_in_workdir(workdir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data)


# ---------------------------------------------------------------- internals

def _existing_file(workdir: Path, path: str) -> Path:
    target = resolve_in_workdir(workdir, path)
    if not target.exists():
        raise RuntimeToolError(NOT_FOUND, f"no such file: {path!r}")
    if not target.is_file():
        raise RuntimeToolError(INVALID_ARGUMENT, f"not a file: {path!r}")
    return target


def _size_of(workdir: Path, path: str) -> int:
    return resolve_in_workdir(workdir, path).stat().st_size


def _read_capped(workdir: Path, path: str, max_bytes: int) -> tuple[bytes, bool]:
    target = _existing_file(workdir, path)
    with target.open("rb") as fh:
        data = fh.read(max_bytes + 1)
    return truncate(data, max_bytes)
