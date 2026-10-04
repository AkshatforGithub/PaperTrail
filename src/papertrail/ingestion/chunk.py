from __future__ import annotations

import re
from dataclasses import dataclass

from papertrail.config import get_settings
from papertrail.ingestion.parse import Section

MIN_CHARS = 60  # drop fragments too short to be useful
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextChunk:
    section: str | None
    content: str


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into chunks of at most `size` characters, on sentence boundaries,
    with roughly `overlap` characters carried over between neighbours."""
    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        while len(sentence) > size:  # a single oversized "sentence"
            pieces.append(sentence[:size])
            sentence = sentence[size:]
        if sentence:
            pieces.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in pieces:
        if current and length + len(piece) + 1 > size:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_len = 0
            for p in reversed(current):  # carry the last few sentences forward
                if tail_len + len(p) + 1 > overlap:
                    break
                tail.insert(0, p)
                tail_len += len(p) + 1
            if tail_len + len(piece) + 1 > size:
                tail, tail_len = [], 0
            current, length = tail, tail_len
        current.append(piece)
        length += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_sections(
    sections: list[Section], size: int | None = None, overlap: int | None = None
) -> list[TextChunk]:
    """Chunk each section separately so a chunk never crosses a section boundary."""
    settings = get_settings()
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    out: list[TextChunk] = []
    for section in sections:
        for piece in split_text(section.text, size, overlap):
            if len(piece) >= MIN_CHARS:
                out.append(TextChunk(section.title, piece))
    return out