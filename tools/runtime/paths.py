"""Path containment: every runtime tool that touches the filesystem resolves
its path through here first. This is the single security boundary acceptance
tests #1 and #2 (README/TEAM_PLAN #10) hold accountable, so it is stricter
than "resolve then compare" — traversal is rejected on the raw input too,
before symlinks get a chance to be clever.
"""
from __future__ import annotations

import os
from pathlib import Path

from tools.runtime.errors import PATH_BLOCKED, RuntimeToolError


def resolve_in_workdir(workdir: Path, user_path: str) -> Path:
    """Return an absolute Path guaranteed to be `workdir` or a descendant of
    it. Raises RuntimeToolError(PATH_BLOCKED) otherwise.

    Two layers, deliberately redundant:
      1. Reject absolute paths and `..` components in the *raw* input — this
         catches the obvious case with a clear message before any syscall.
      2. Resolve symlinks and re-check containment against the resolved
         workdir — this catches a symlink planted *inside* the workdir that
         points outside it, which layer 1 cannot see.
    """
    if not user_path or not user_path.strip():
        raise RuntimeToolError(PATH_BLOCKED, "empty path")

    candidate_raw = Path(user_path)
    if candidate_raw.is_absolute():
        raise RuntimeToolError(
            PATH_BLOCKED, f"absolute paths are not allowed: {user_path!r}")
    if ".." in candidate_raw.parts:
        raise RuntimeToolError(
            PATH_BLOCKED, f"path traversal ('..') is not allowed: {user_path!r}")

    workdir_resolved = workdir.resolve(strict=True)
    candidate = (workdir_resolved / candidate_raw).resolve(strict=False)

    try:
        common = os.path.commonpath([str(workdir_resolved), str(candidate)])
    except ValueError:
        # Different drives on Windows, or otherwise incomparable.
        raise RuntimeToolError(
            PATH_BLOCKED, f"path escapes the task workdir: {user_path!r}") from None

    if common != str(workdir_resolved):
        raise RuntimeToolError(
            PATH_BLOCKED, f"path escapes the task workdir: {user_path!r}")

    return candidate
