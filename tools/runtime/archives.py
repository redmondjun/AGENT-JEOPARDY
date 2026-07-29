"""Safe archive inspection/extraction (TEAM_PLAN #10.3): ZIP/TAR detection,
member-count and expanded-size limits, absolute-path and `..` rejection, and
a nested-archive depth limit so an archive-in-an-archive-in-an-archive can't
turn one small download into an unbounded write.

Two entry points: `inspect` reads member metadata only (no bytes touch disk)
so a tool call can ask "what's in this?" cheaply; `extract` actually writes
files, member-by-member through `paths.resolve_in_workdir`, and only
`extract_recursive` ever looks inside a member that is itself an archive.
"""
from __future__ import annotations

import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tools.runtime.errors import (
    NOT_FOUND, OUTPUT_TOO_LARGE, PATH_BLOCKED, UNSUPPORTED_FORMAT,
    RuntimeToolError,
)
from tools.runtime.paths import resolve_in_workdir

DEFAULT_MAX_MEMBERS = 10_000
DEFAULT_MAX_TOTAL_UNCOMPRESSED = 100_000_000  # 100 MB
DEFAULT_MAX_DEPTH = 3

_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


@dataclass(frozen=True)
class MemberInfo:
    name: str
    size: int
    is_dir: bool


@dataclass(frozen=True)
class ArchiveInfo:
    format: str  # "zip" | "tar"
    member_count: int
    total_uncompressed_size: int
    members: tuple[MemberInfo, ...]


@dataclass(frozen=True)
class ExtractResult:
    dest: str  # relative to workdir
    extracted_count: int
    extracted_bytes: int
    nested_archives: tuple[str, ...]  # extracted members that are themselves archives


def detect_format(path: Path) -> str:
    if zipfile.is_zipfile(path):
        return "zip"
    if tarfile.is_tarfile(path):
        return "tar"
    raise RuntimeToolError(UNSUPPORTED_FORMAT, f"not a zip or tar archive: {path.name!r}")


def inspect(workdir: Path, archive_path: str, *,
           max_members: int = DEFAULT_MAX_MEMBERS,
           max_total_uncompressed: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED) -> ArchiveInfo:
    """Read member metadata without extracting anything."""
    target = _existing_archive(workdir, archive_path)
    fmt = detect_format(target)
    members: list[MemberInfo] = []
    total = 0

    if fmt == "zip":
        with zipfile.ZipFile(target) as zf:
            raw = zf.infolist()
            _check_member_count(len(raw), max_members)
            for info in raw:
                _validate_member_name(info.filename)
                total += info.file_size
                members.append(MemberInfo(info.filename, info.file_size, info.is_dir()))
    else:
        with tarfile.open(target, mode="r:*") as tf:
            raw = tf.getmembers()
            _check_member_count(len(raw), max_members)
            for info in raw:
                _validate_member_name(info.name)
                size = max(info.size, 0)
                total += size
                members.append(MemberInfo(info.name, size, info.isdir()))

    if total > max_total_uncompressed:
        raise RuntimeToolError(
            OUTPUT_TOO_LARGE,
            f"archive expands to {total} bytes, over the {max_total_uncompressed}-byte limit")

    return ArchiveInfo(fmt, len(members), total, tuple(members))


def extract(workdir: Path, archive_path: str, dest_dir: str, *,
           max_members: int = DEFAULT_MAX_MEMBERS,
           max_total_uncompressed: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED) -> ExtractResult:
    """Extract one archive (not its nested archives) into `dest_dir`, both
    paths relative to `workdir`. Every member path is re-validated against
    the workdir individually — `inspect`'s pass already checked names, but
    this checks the actual join+resolve for each write, which is the layer
    that matters.
    """
    info = inspect(workdir, archive_path,
                   max_members=max_members, max_total_uncompressed=max_total_uncompressed)
    target = _existing_archive(workdir, archive_path)
    dest_root = resolve_in_workdir(workdir, dest_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    extracted_bytes = 0
    nested: list[str] = []

    if info.format == "zip":
        with zipfile.ZipFile(target) as zf:
            for zinfo in zf.infolist():
                rel = f"{dest_dir.rstrip('/')}/{zinfo.filename}"
                member_dest = resolve_in_workdir(workdir, rel)
                if zinfo.is_dir():
                    member_dest.mkdir(parents=True, exist_ok=True)
                    continue
                member_dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(zinfo) as src:
                    data = src.read()
                member_dest.write_bytes(data)
                extracted_count += 1
                extracted_bytes += len(data)
                if _looks_like_archive(zinfo.filename):
                    nested.append(_relative(workdir, member_dest))
    else:
        with tarfile.open(target, mode="r:*") as tf:
            for tinfo in tf.getmembers():
                rel = f"{dest_dir.rstrip('/')}/{tinfo.name}"
                member_dest = resolve_in_workdir(workdir, rel)
                if tinfo.isdir():
                    member_dest.mkdir(parents=True, exist_ok=True)
                    continue
                if not tinfo.isfile():
                    continue  # skip devices/symlinks/etc — not a regular file to write
                member_dest.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(tinfo)
                data = src.read() if src is not None else b""
                member_dest.write_bytes(data)
                extracted_count += 1
                extracted_bytes += len(data)
                if _looks_like_archive(tinfo.name):
                    nested.append(_relative(workdir, member_dest))

    return ExtractResult(dest_dir, extracted_count, extracted_bytes, tuple(nested))


def extract_recursive(workdir: Path, archive_path: str, dest_dir: str, *,
                      max_members: int = DEFAULT_MAX_MEMBERS,
                      max_total_uncompressed: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED,
                      max_depth: int = DEFAULT_MAX_DEPTH) -> ExtractResult:
    """Extract `archive_path`, then descend into any extracted member that is
    itself an archive, up to `max_depth` levels. Member counts and expanded
    bytes are a CUMULATIVE budget across every level — a bomb built from many
    small nested archives, each individually under the limit, still trips
    this because the budget is shared, not reset per level.
    """
    remaining_members = max_members
    remaining_bytes = max_total_uncompressed
    total_count = 0
    total_bytes = 0
    all_nested: list[str] = []

    def _go(rel_archive: str, rel_dest: str, depth: int) -> None:
        nonlocal remaining_members, remaining_bytes, total_count, total_bytes
        if depth > max_depth:
            raise RuntimeToolError(
                UNSUPPORTED_FORMAT,
                f"nested archive depth exceeds the limit of {max_depth}: {rel_archive!r}")
        result = extract(workdir, rel_archive, rel_dest,
                         max_members=remaining_members, max_total_uncompressed=remaining_bytes)
        remaining_members -= result.extracted_count
        remaining_bytes -= result.extracted_bytes
        total_count += result.extracted_count
        total_bytes += result.extracted_bytes
        all_nested.extend(result.nested_archives)
        for nested_archive in result.nested_archives:
            nested_dest = f"{nested_archive}__extracted"
            _go(nested_archive, nested_dest, depth + 1)

    _go(archive_path, dest_dir, depth=1)
    return ExtractResult(dest_dir, total_count, total_bytes, tuple(all_nested))


# ---------------------------------------------------------------- internals

def _existing_archive(workdir: Path, archive_path: str) -> Path:
    target = resolve_in_workdir(workdir, archive_path)
    if not target.exists():
        raise RuntimeToolError(NOT_FOUND, f"no such archive: {archive_path!r}")
    if not target.is_file():
        raise RuntimeToolError(UNSUPPORTED_FORMAT, f"not a file: {archive_path!r}")
    return target


def _check_member_count(count: int, max_members: int) -> None:
    if count > max_members:
        raise RuntimeToolError(
            OUTPUT_TOO_LARGE, f"archive has {count} members, over the {max_members} limit")


def _validate_member_name(name: str) -> None:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        raise RuntimeToolError(PATH_BLOCKED, f"archive member has an absolute path: {name!r}")
    if len(normalized) > 1 and normalized[1] == ":":  # e.g. "C:/..."
        raise RuntimeToolError(PATH_BLOCKED, f"archive member has a drive path: {name!r}")
    if ".." in PurePosixPath(normalized).parts:
        raise RuntimeToolError(PATH_BLOCKED, f"archive member escapes via '..': {name!r}")


def _looks_like_archive(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)


def _relative(workdir: Path, path: Path) -> str:
    return path.relative_to(workdir.resolve(strict=True)).as_posix()
