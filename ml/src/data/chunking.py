"""Sentence/bullet-level chunking.

Used in Phase 1 to build training triples, and reused unchanged at inference
time in Phase 4 to build the chunk-level similarity heatmap for
explainability - keep this module free of training-only assumptions.
"""
import re
from typing import List

from src.utils.text_cleaning import normalize_whitespace

_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•●‣⁃]|\d+[.)])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_bullets(text: str) -> List[str]:
    """Split on newlines, stripping common bullet markers (-, *, •, 1.)."""
    bullets = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = _BULLET_PREFIX_RE.sub("", line)
        bullets.append(line)
    return bullets


def split_sentences(text: str) -> List[str]:
    """Regex sentence splitter - avoids an nltk/spaCy runtime dependency
    (and punkt downloads) for what is otherwise a simple heuristic split."""
    text = text.replace("\n", " ")
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _split_long_chunk(chunk: str, max_chars: int) -> List[str]:
    if len(chunk) <= max_chars:
        return [chunk]
    sentences = split_sentences(chunk)
    if len(sentences) <= 1:
        return [chunk[i : i + max_chars] for i in range(0, len(chunk), max_chars)]

    pieces, current = [], ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > max_chars:
            pieces.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}".strip()
    if current:
        pieces.append(current.strip())
    return pieces


def chunk_text(
    text: str,
    mode: str = "sentence",
    min_chunk_chars: int = 15,
    max_chunk_chars: int = 400,
) -> List[str]:
    """Split a resume/JD document into sentence- or bullet-level segments.

    mode="bullet" falls back to sentence splitting for documents with no
    line-level bullet structure (e.g. a JD written as prose paragraphs).
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    if mode == "bullet":
        raw_chunks = split_bullets(text)
        if len(raw_chunks) <= 1:
            raw_chunks = split_sentences(text)
    elif mode == "sentence":
        raw_chunks = split_sentences(text)
    else:
        raise ValueError(f"Unknown chunking mode: {mode!r} (expected 'sentence' or 'bullet')")

    chunks: List[str] = []
    for raw in raw_chunks:
        chunks.extend(_split_long_chunk(raw, max_chunk_chars))

    return [c for c in chunks if len(c) >= min_chunk_chars]
