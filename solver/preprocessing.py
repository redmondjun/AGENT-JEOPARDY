"""Cheap, bounded task context prepared before the first model turn."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from contracts import TaskContext
from solver.specialists.cryptic import identify_binary_format, identify_text_encoding
from solver.specialists.data import profile_file
from solver.specialists.documents import chunk_file, search_chunks
from solver.specialists.optimization import extract_constraints

MAX_PREPROCESS_CHARS = 4_000
MAX_PROFILE_BYTES = 8_000_000
MAX_ARCHIVE_MEMBERS = 50


def preprocess_task(task: TaskContext) -> TaskContext:
    """Return an enriched context without changing task files or answer state."""
    signatures = tuple(_file_signature(path) for path in task.files)
    summary = _cached_summary(task.category, task.prompt, signatures)
    metadata = dict(task.metadata)
    metadata["preprocessing"] = summary
    return replace(task, metadata=metadata)


@lru_cache(maxsize=256)
def _cached_summary(
    category: str,
    prompt: str,
    signatures: tuple[tuple[str, int, int], ...],
) -> str:
    """Cache bounded preprocessing across retries of immutable task files."""
    paths = tuple(Path(path) for path, _size, _mtime_ns in signatures)
    lines = [_file_manifest(paths)]
    if category == "Needle in the Haystack":
        lines.append(_data_hints(paths))
    elif category == "Ancient Scrolls":
        lines.append(_document_hints(paths, prompt))
    elif category == "Cryptic":
        lines.append(_cryptic_hints(paths))
    elif category == "Ship It":
        lines.append(_code_hints(paths))
    elif category == "Heavy Compute":
        constraints = extract_constraints(prompt)
        if constraints:
            lines.append(
                "Extracted numeric constraints: "
                + ", ".join(f"{item.op} {item.bound:g}" for item in constraints)
            )
    elif category == "The Dark Web":
        urls = tuple(dict.fromkeys(re.findall(r"https?://[^\s<>'\"]+", prompt)))
        if urls:
            lines.append("Prompt URLs: " + ", ".join(urls[:5]))

    summary = "\n".join(line for line in lines if line).strip()
    if len(summary) > MAX_PREPROCESS_CHARS:
        summary = summary[:MAX_PREPROCESS_CHARS] + "…[preprocessing truncated]"
    return summary


def _file_signature(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns
    except OSError:
        return str(path), -1, -1


def _file_manifest(paths: tuple[Path, ...]) -> str:
    if not paths:
        return "Preflight manifest: no attached files."
    rows = ["Preflight manifest (do not call list_files unless paths change):"]
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        rows.append(f"- {path.name}: {size} bytes, suffix={path.suffix.lower() or '(none)'}")
    return "\n".join(rows)


def _data_hints(paths: tuple[Path, ...]) -> str:
    profiles = []
    for path in paths[:3]:
        try:
            if 0 <= path.stat().st_size <= MAX_PROFILE_BYTES:
                profiles.append(profile_file(path))
        except Exception:  # preprocessing must never consume a solve attempt
            continue
    return "\n".join(profiles)


def _document_hints(paths: tuple[Path, ...], prompt: str) -> str:
    hints = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".html", ".xml", ".json"}:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            hints.append(f"{path.name}: searchable text document, {size} bytes")
            if suffix in {".txt", ".md"} and size <= MAX_PROFILE_BYTES:
                try:
                    relevant = search_chunks(
                        chunk_file(path), prompt, top_k=2
                    )
                    for chunk in relevant:
                        excerpt = chunk.text[:1_200].replace("\x00", "")
                        hints.append(
                            f"{path.name} relevant excerpt at char "
                            f"{chunk.start_char}:\n{excerpt}"
                        )
                except Exception:  # bounded hinting must fail open
                    pass
        elif suffix == ".pdf":
            hints.append(f"{path.name}: PDF; extract/search it before targeted reading")
    return "\n".join(hints)


def _cryptic_hints(paths: tuple[Path, ...]) -> str:
    hints = []
    for path in paths[:3]:
        try:
            data = path.read_bytes()[:4096]
        except OSError:
            continue
        binary_kind = identify_binary_format(data)
        if binary_kind != "unknown_binary":
            hints.append(f"{path.name}: magic bytes identify {binary_kind}")
        else:
            text = data.decode("utf-8", errors="ignore")
            hints.append(f"{path.name}: looks like {identify_text_encoding(text)}")
        if zipfile.is_zipfile(path):
            try:
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()[:MAX_ARCHIVE_MEMBERS]
                hints.append(
                    f"{path.name}: zip members ({len(names)} shown): "
                    + ", ".join(names)
                )
            except (OSError, zipfile.BadZipFile):
                pass
    return "\n".join(hints)


def _code_hints(paths: tuple[Path, ...]) -> str:
    tests = [path.name for path in paths if "test" in path.name.lower()]
    sources = [
        path.name
        for path in paths
        if path.suffix.lower() in {".py", ".js", ".ts", ".go", ".rs", ".java"}
    ]
    return json.dumps(
        {"likely_tests": tests[:20], "likely_sources": sources[:20]},
        separators=(",", ":"),
    )
