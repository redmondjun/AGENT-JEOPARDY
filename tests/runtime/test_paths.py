from __future__ import annotations

from pathlib import Path

import pytest

from tools.runtime.errors import PATH_BLOCKED, RuntimeToolError
from tools.runtime.paths import resolve_in_workdir


def test_plain_relative_path_resolves_inside_workdir(workdir: Path) -> None:
    (workdir / "a.txt").write_text("hi")
    resolved = resolve_in_workdir(workdir, "a.txt")
    assert resolved == (workdir / "a.txt").resolve()


def test_nested_relative_path_ok(workdir: Path) -> None:
    (workdir / "sub").mkdir()
    resolved = resolve_in_workdir(workdir, "sub/b.txt")
    assert resolved.parent == (workdir / "sub").resolve()


def test_dotdot_traversal_rejected(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        resolve_in_workdir(workdir, "../../secret")
    assert exc.value.code == PATH_BLOCKED


def test_dotdot_in_middle_rejected(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        resolve_in_workdir(workdir, "sub/../../escape")
    assert exc.value.code == PATH_BLOCKED


def test_absolute_path_rejected(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        resolve_in_workdir(workdir, "/etc/passwd")
    assert exc.value.code == PATH_BLOCKED


def test_empty_path_rejected(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        resolve_in_workdir(workdir, "")
    assert exc.value.code == PATH_BLOCKED


def test_symlink_escape_rejected(workdir: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope")
    link = workdir / "escape_link"
    link.symlink_to(outside)
    with pytest.raises(RuntimeToolError) as exc:
        resolve_in_workdir(workdir, "escape_link/secret.txt")
    assert exc.value.code == PATH_BLOCKED
