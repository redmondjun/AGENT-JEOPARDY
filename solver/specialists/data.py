"""
Needle in the Haystack preprocessing.

The model should never receive a raw dump of a large CSV/JSON file. This
module produces a compact schema/profile summary — column names, dtypes,
row count, and a small sample — so the model can decide which targeted
tool call (grep/filter/query) to make next, instead of trying to eyeball
the whole haystack.

Deliberately dependency-light: uses csv/json from stdlib rather than
pandas, since the hosted image (python:3.12-slim, 2 CPU/2GB) may not have
pandas installed and Vidula owns requirements.txt (TEAM_PLAN.md section
10, deliverable 5 — "add a package only when a measured task requires
it").
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

MAX_SAMPLE_ROWS = 5
MAX_FIELD_PREVIEW_CHARS = 80


def profile_csv(path: Path) -> str:
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return f"{path.name}: empty file"

        sample_rows = []
        row_count = 0
        for row in reader:
            row_count += 1
            if len(sample_rows) < MAX_SAMPLE_ROWS:
                sample_rows.append(row)

    lines = [f"{path.name}: {len(header)} columns, {row_count} data rows"]
    lines.append("columns: " + ", ".join(header))
    for row in sample_rows:
        preview = ", ".join(_preview(cell) for cell in row)
        lines.append(f"  sample row: {preview}")
    return "\n".join(lines)


def profile_json(path: Path) -> str:
    with path.open(encoding="utf-8", errors="replace") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            return f"{path.name}: invalid JSON ({exc})"

    if isinstance(data, list):
        keys = sorted({k for item in data[:50] if isinstance(item, dict) for k in item})
        return (
            f"{path.name}: JSON array, {len(data)} items, "
            f"observed keys (first 50 items): {', '.join(keys) or '(none)'}"
        )
    if isinstance(data, dict):
        return f"{path.name}: JSON object with top-level keys: {', '.join(sorted(data))}"
    return f"{path.name}: JSON scalar of type {type(data).__name__}"


def _preview(value: str) -> str:
    if len(value) <= MAX_FIELD_PREVIEW_CHARS:
        return value
    return value[:MAX_FIELD_PREVIEW_CHARS] + "…"


def profile_file(path: Path) -> str:
    """Dispatches on extension; falls back to a byte-size note for unknown types."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return profile_csv(path)
    if suffix == ".json":
        return profile_json(path)
    try:
        size = path.stat().st_size
    except OSError:
        return f"{path.name}: unreadable"
    return f"{path.name}: {size} bytes, unrecognized format for profiling"
