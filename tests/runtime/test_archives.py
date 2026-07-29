from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.runtime import archives
from tools.runtime.errors import (
    NOT_FOUND, OUTPUT_TOO_LARGE, PATH_BLOCKED, UNSUPPORTED_FORMAT, RuntimeToolError,
)


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _make_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------- inspect

def test_inspect_zip(workdir: Path) -> None:
    _make_zip(workdir / "a.zip", {"one.txt": b"hello", "dir/two.txt": b"world!!"})
    info = archives.inspect(workdir, "a.zip")
    assert info.format == "zip"
    assert info.member_count == 2
    assert info.total_uncompressed_size == len(b"hello") + len(b"world!!")


def test_inspect_tar(workdir: Path) -> None:
    _make_tar(workdir / "a.tar", {"one.txt": b"hello"})
    info = archives.inspect(workdir, "a.tar")
    assert info.format == "tar"
    assert info.member_count == 1


def test_inspect_missing_archive(workdir: Path) -> None:
    with pytest.raises(RuntimeToolError) as exc:
        archives.inspect(workdir, "nope.zip")
    assert exc.value.code == NOT_FOUND


def test_inspect_not_an_archive(workdir: Path) -> None:
    (workdir / "plain.txt").write_text("just text")
    with pytest.raises(RuntimeToolError) as exc:
        archives.inspect(workdir, "plain.txt")
    assert exc.value.code == UNSUPPORTED_FORMAT


def test_inspect_rejects_absolute_member_path(workdir: Path) -> None:
    _make_zip(workdir / "evil.zip", {"/etc/passwd": b"nope"})
    with pytest.raises(RuntimeToolError) as exc:
        archives.inspect(workdir, "evil.zip")
    assert exc.value.code == PATH_BLOCKED


def test_inspect_rejects_dotdot_member_path(workdir: Path) -> None:
    _make_zip(workdir / "evil.zip", {"../../escape.txt": b"nope"})
    with pytest.raises(RuntimeToolError) as exc:
        archives.inspect(workdir, "evil.zip")
    assert exc.value.code == PATH_BLOCKED


def test_inspect_member_count_limit(workdir: Path) -> None:
    _make_zip(workdir / "many.zip", {f"f{i}.txt": b"x" for i in range(10)})
    with pytest.raises(RuntimeToolError) as exc:
        archives.inspect(workdir, "many.zip", max_members=5)
    assert exc.value.code == OUTPUT_TOO_LARGE


def test_inspect_total_size_limit(workdir: Path) -> None:
    _make_zip(workdir / "heavy.zip", {"big.bin": b"x" * 10_000})
    with pytest.raises(RuntimeToolError) as exc:
        archives.inspect(workdir, "heavy.zip", max_total_uncompressed=1_000)
    assert exc.value.code == OUTPUT_TOO_LARGE


# ---------------------------------------------------------------- extract

def test_extract_writes_files_inside_dest(workdir: Path) -> None:
    _make_zip(workdir / "a.zip", {"one.txt": b"hello", "dir/two.txt": b"world"})
    result = archives.extract(workdir, "a.zip", "out")
    assert result.extracted_count == 2
    assert (workdir / "out" / "one.txt").read_bytes() == b"hello"
    assert (workdir / "out" / "dir" / "two.txt").read_bytes() == b"world"


def test_extract_malicious_zip_cannot_write_outside_workdir(workdir: Path, tmp_path: Path) -> None:
    """Acceptance: archive traversal cannot write outside workdir."""
    _make_zip(workdir / "evil.zip", {"../../../outside.txt": b"pwned"})
    with pytest.raises(RuntimeToolError) as exc:
        archives.extract(workdir, "evil.zip", "out")
    assert exc.value.code == PATH_BLOCKED
    # Nothing escaped: no file written above tmp_path's own workdir tree.
    assert not (tmp_path / "outside.txt").exists()


def test_extract_tar(workdir: Path) -> None:
    _make_tar(workdir / "a.tar", {"one.txt": b"hi"})
    result = archives.extract(workdir, "a.tar", "out")
    assert result.extracted_count == 1
    assert (workdir / "out" / "one.txt").read_bytes() == b"hi"


def test_extract_reports_nested_archive_without_descending(workdir: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("deep.txt", b"deep")
    _make_zip(workdir / "outer.zip", {"inner.zip": inner.getvalue()})

    result = archives.extract(workdir, "outer.zip", "out")
    assert result.extracted_count == 1
    assert result.nested_archives == ("out/inner.zip",)
    # Not descended into — only the outer extract ran.
    assert not (workdir / "out" / "inner.zip__extracted").exists()


def test_extract_recursive_descends_into_nested_archive(workdir: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("deep.txt", b"deep")
    _make_zip(workdir / "outer.zip", {"inner.zip": inner.getvalue()})

    result = archives.extract_recursive(workdir, "outer.zip", "out", max_depth=3)
    assert result.extracted_count == 2  # inner.zip itself + deep.txt
    nested_dest = "out/inner.zip__extracted"
    assert (workdir / nested_dest / "deep.txt").read_bytes() == b"deep"


def test_extract_recursive_depth_limit_exceeded(workdir: Path) -> None:
    level = io.BytesIO()
    with zipfile.ZipFile(level, "w") as zf:
        zf.writestr("bottom.txt", b"x")
    for _ in range(4):  # build a chain nested deeper than max_depth below
        wrapper = io.BytesIO()
        with zipfile.ZipFile(wrapper, "w") as zf:
            zf.writestr("nested.zip", level.getvalue())
        level = wrapper
    (workdir / "chain.zip").write_bytes(level.getvalue())

    with pytest.raises(RuntimeToolError) as exc:
        archives.extract_recursive(workdir, "chain.zip", "out", max_depth=2)
    assert exc.value.code == UNSUPPORTED_FORMAT
