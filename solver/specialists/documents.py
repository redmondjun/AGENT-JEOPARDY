"""
Ancient Scrolls preprocessing: long documents get chunked and lightly
indexed so the model can search for relevant passages instead of reading
the whole thing linearly (which burns turn/token budget fast, see
TEAM_PLAN.md section 1, item 4 — strict per-tile budgets).

This is deliberately a simple word-overlap search, not embeddings — no
extra dependency, no network call, works fully offline inside the hosted
sandbox, and is "good enough" to point the model at the right chunk before
it uses a tool for the precise read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    index: int
    start_char: int
    text: str


def chunk_document(text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    index = 0
    length = len(text)
    if length == 0:
        return chunks
    while start < length:
        end = min(start + CHUNK_SIZE_CHARS, length)
        chunks.append(Chunk(index=index, start_char=start, text=text[start:end]))
        if end == length:
            break
        start = end - CHUNK_OVERLAP_CHARS
        index += 1
    return chunks


def chunk_file(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return chunk_document(text)


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def search_chunks(chunks: list[Chunk], query: str, top_k: int = 3) -> list[Chunk]:
    """
    Ranks chunks by word-overlap with the query. Not semantic search — a
    cheap, dependency-free relevance signal good enough to narrow a long
    document down to a handful of candidate chunks before a tool call
    reads the precise passage.
    """
    query_words = _words(query)
    if not query_words:
        return chunks[:top_k]

    scored = []
    for chunk in chunks:
        overlap = len(query_words & _words(chunk.text))
        if overlap > 0:
            scored.append((overlap, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]] or chunks[:top_k]


def summarize_index(chunks: list[Chunk]) -> str:
    return (
        f"{len(chunks)} chunks, ~{CHUNK_SIZE_CHARS} chars each "
        f"(overlap {CHUNK_OVERLAP_CHARS}); use search_chunks(query) to find "
        f"the passage most relevant to a specific sub-question."
    )
