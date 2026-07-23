"""Chunk-level TF-IDF similarity for explainable job matching (Phase 4).

Given a resume and a JD, breaks both into chunks (src/data/chunking.py,
unchanged from Phase 1) and scores every resume-chunk/JD-chunk pair with the
same TF-IDF vectorizer used for whole-document matching - Phase 3b's hybrid
investigation (docs/PHASE3_FINDINGS.md) concluded TF-IDF alone is the best
matcher on this dataset, so it is what Phase 4's explanations are built on
instead of the fine-tuned SBERT model docs/PROJECT_SPEC.md originally
envisioned for this role.
"""
import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.chunking import chunk_text
from src.explainability.schema import ChunkMatch

DEFAULT_TOP_K = 5

_DEDUP_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_dedup(text: str) -> str:
    return _DEDUP_WHITESPACE_RE.sub(" ", text.strip().lower())


def chunk_document(text: str, chunking_cfg: dict) -> List[str]:
    return chunk_text(
        text,
        mode=chunking_cfg.get("mode", "sentence"),
        min_chunk_chars=chunking_cfg.get("min_chunk_chars", 15),
        max_chunk_chars=chunking_cfg.get("max_chunk_chars", 400),
    )


def chunk_similarity_matrix(
    resume_chunks: List[str], jd_chunks: List[str], vectorizer: TfidfVectorizer
) -> np.ndarray:
    """`vectorizer` must already be fit (e.g. TfidfMatcher.fit()'s
    vectorizer, fit on the full resume+JD corpus) - reusing that fitted
    vocabulary/IDF keeps chunk-level scores consistent with the
    document-level score being explained, instead of re-fitting on just
    these two documents' small, unstable vocabulary."""
    resume_vecs = vectorizer.transform(resume_chunks)
    jd_vecs = vectorizer.transform(jd_chunks)
    return cosine_similarity(resume_vecs, jd_vecs)


def count_exact_duplicate_pairs(resume_chunks: List[str], jd_chunks: List[str]) -> int:
    """Count resume-chunk/JD-chunk pairs that are byte-identical once
    whitespace/case-normalized - on this dataset these come almost entirely
    from the `responsibilities`/`responsibilities.1` field, a byte-identical,
    category-keyed boilerplate template shared by every resume and JD in a
    category (see docs/PHASE3_FINDINGS.md's "Ground truth contamination"
    section), not genuine chunk-level matching signal."""
    jd_normalized = {_normalize_for_dedup(j) for j in jd_chunks}
    return sum(1 for r in resume_chunks if _normalize_for_dedup(r) in jd_normalized)


def top_k_chunk_matches(
    resume_chunks: List[str],
    jd_chunks: List[str],
    sim_matrix: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    dedupe_resume_chunks: bool = True,
    exclude_exact_duplicates: bool = False,
) -> List[ChunkMatch]:
    """Rank every resume-chunk/JD-chunk pair by cosine similarity and return
    the top `top_k`.

    `dedupe_resume_chunks=True` (default) keeps only each resume chunk's
    single best-matching JD chunk before ranking, so one strong resume
    bullet can't fill every explanation slot - each slot highlights a
    different part of the resume.

    `exclude_exact_duplicates=False` (default, unchanged behavior): set True
    to drop pairs whose text is byte-identical (whitespace/case-normalized)
    before ranking - see `count_exact_duplicate_pairs` docstring for why
    that matters on this dataset. Off by default because a short exact
    overlap (e.g. a single shared skill token) can also be a perfectly
    legitimate explanation; callers who know their corpus has a
    boilerplate-duplication issue should opt in explicitly.
    """
    if sim_matrix.size == 0:
        return []

    if dedupe_resume_chunks:
        best_jd_idx = sim_matrix.argmax(axis=1)
        candidates = [
            (sim_matrix[r, best_jd_idx[r]], r, int(best_jd_idx[r]))
            for r in range(sim_matrix.shape[0])
        ]
    else:
        candidates = [
            (sim_matrix[r, j], r, j)
            for r in range(sim_matrix.shape[0])
            for j in range(sim_matrix.shape[1])
        ]

    if exclude_exact_duplicates:
        candidates = [
            c for c in candidates
            if _normalize_for_dedup(resume_chunks[c[1]]) != _normalize_for_dedup(jd_chunks[c[2]])
        ]

    candidates.sort(key=lambda c: -c[0])

    return [
        ChunkMatch(
            resume_chunk=resume_chunks[r],
            jd_chunk=jd_chunks[j],
            score=float(score),
            resume_chunk_idx=r,
            jd_chunk_idx=j,
        )
        for score, r, j in candidates[:top_k]
    ]


def explain_match(
    resume_text: str,
    jd_text: str,
    vectorizer: TfidfVectorizer,
    chunking_cfg: dict,
    top_k: int = DEFAULT_TOP_K,
    dedupe_resume_chunks: bool = True,
    exclude_exact_duplicates: bool = False,
) -> List[ChunkMatch]:
    """End-to-end: chunk both documents, score every pair, return the
    top_k matches - the function Phase 5's backend calls per (resume,
    ranked JD) pair to build the explanation shown alongside a
    recommendation. See `top_k_chunk_matches` for `exclude_exact_duplicates`."""
    resume_chunks = chunk_document(resume_text, chunking_cfg)
    jd_chunks = chunk_document(jd_text, chunking_cfg)
    if not resume_chunks or not jd_chunks:
        return []

    sim_matrix = chunk_similarity_matrix(resume_chunks, jd_chunks, vectorizer)
    return top_k_chunk_matches(
        resume_chunks,
        jd_chunks,
        sim_matrix,
        top_k=top_k,
        dedupe_resume_chunks=dedupe_resume_chunks,
        exclude_exact_duplicates=exclude_exact_duplicates,
    )


def format_explanation(match: ChunkMatch) -> str:
    """Human-readable sentence for one ChunkMatch, e.g. docs/PROJECT_SPEC.md's
    Phase 4 example: "Your experience with 'Python, pandas' matched the
    requirement 'data analysis in Python'"."""
    return f"Your experience with '{match.resume_chunk}' matched the requirement '{match.jd_chunk}'"
