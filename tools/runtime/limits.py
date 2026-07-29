"""Shared truncation helper. `files.py` (oversized reads) and `processes.py`
(oversized stdout/stderr) both need to cap output at a byte budget and leave
an unambiguous marker behind — acceptance test: "Output is capped with an
explicit truncation marker." One implementation so the marker text can't
drift between the two call sites.
"""
from __future__ import annotations


def truncate(data: bytes, max_bytes: int) -> tuple[bytes, bool]:
    """Return (possibly-shortened data, was_truncated)."""
    if len(data) <= max_bytes:
        return data, False
    return data[:max_bytes], True


def truncation_marker(total_bytes: int, kept_bytes: int) -> str:
    omitted = total_bytes - kept_bytes
    return f"\n...[truncated: kept {kept_bytes} of {total_bytes} bytes, {omitted} omitted]"
