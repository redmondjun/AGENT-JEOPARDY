"""
Cryptic category: identify encodings/formats deterministically rather than
having the model hand-decode base64/hex/rot13 etc. from memory, which is
exactly the kind of "model retypes an exact token incorrectly" failure
mode TEAM_PLAN.md section 16 calls out. This module only *identifies* —
actual decoding should go through a runtime-tool call so the decoded bytes
become a ToolResult.exact_value and bypass model transcription entirely
(see agent_loop.py's exact-value pass-through).
"""

from __future__ import annotations

import base64
import binascii
import re

from contracts import CandidateAnswer, TaskContext
from solver.verification import register_category_check

_HEX_RE = re.compile(r"^[0-9A-Fa-f\s]+$")
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_MAGIC_BYTES = {
    b"PK\x03\x04": "zip",
    b"\x1f\x8b": "gzip",
    b"BZh": "bzip2",
    b"\x89PNG": "png",
    b"%PDF": "pdf",
}


def identify_text_encoding(text: str) -> str:
    """Best-effort, deterministic guess at what encoding a text blob is in."""
    stripped = text.strip()
    if not stripped:
        return "empty"

    if _HEX_RE.match(stripped) and len(stripped.replace(" ", "").replace("\n", "")) % 2 == 0:
        try:
            binascii.unhexlify(stripped.replace(" ", "").replace("\n", ""))
            return "hex"
        except binascii.Error:
            pass

    if _BASE64_RE.match(stripped) and len(stripped) % 4 == 0:
        try:
            base64.b64decode(stripped, validate=True)
            return "base64"
        except (binascii.Error, ValueError):
            pass

    if re.fullmatch(r"[01\s]+", stripped) and len(stripped.replace(" ", "")) % 8 == 0:
        return "binary"

    if all(c.isalpha() or c.isspace() for c in stripped):
        return "possible_classical_cipher (rot/caesar/vigenere — verify with tool, don't hand-decode)"

    return "unknown"


def identify_binary_format(data: bytes) -> str:
    for magic, label in _MAGIC_BYTES.items():
        if data.startswith(magic):
            return label
    return "unknown_binary"


def _looks_still_encoded(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if _HEX_RE.match(stripped) and len(stripped) > 8:
        return True
    if _BASE64_RE.match(stripped) and len(stripped) >= 16 and "=" in stripped:
        return True
    return False


def _check_cryptic_not_still_encoded(
    candidate: CandidateAnswer, task: TaskContext
) -> tuple[bool, str]:
    if candidate.exact_value_from_tool:
        # Trust the tool's decoded output over this heuristic.
        return True, "cryptic: exact_value from tool, skipping encoded-heuristic check"
    if _looks_still_encoded(candidate.value):
        return False, "cryptic: candidate value still looks encoded (hex/base64), likely undecoded"
    return True, "cryptic: candidate does not look like raw encoded text"


register_category_check("Cryptic", _check_cryptic_not_still_encoded)
