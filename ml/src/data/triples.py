"""Build (resume_chunk, positive_jd_chunk, negative_jd_chunk) training
triples - the direct input to Phase 3's MultipleNegativesRankingLoss /
TripletLoss fine-tuning of SBERT.

Positive JD chunks are sampled from JDs in the SAME (normalized) category as
the resume; negative JD chunks from a DIFFERENT (normalized) category - weak
(category-based) supervision, matching pairing.py. There is no per-resume
matched_score anymore (see docs/PHASE3_FINDINGS.md): resumes and postings
are independently sourced, so category co-membership is the only relevance
signal available.
"""
import random
from typing import Dict, List, Tuple

import pandas as pd

from src.data.chunking import chunk_text
from src.data.schema import Triple
from src.utils.text_cleaning import normalize_category


def _chunk_documents(df: pd.DataFrame, chunking_cfg: dict) -> Dict[str, List[str]]:
    chunks_by_doc = {}
    for _, row in df.iterrows():
        chunks = chunk_text(
            row["text"],
            mode=chunking_cfg.get("mode", "sentence"),
            min_chunk_chars=chunking_cfg.get("min_chunk_chars", 15),
            max_chunk_chars=chunking_cfg.get("max_chunk_chars", 400),
        )
        if chunks:
            chunks_by_doc[row["doc_id"]] = chunks
    return chunks_by_doc


def _filter_informative(
    chunks_by_doc: Dict[str, List[str]], min_informative_chars: int
) -> Dict[str, List[str]]:
    """Drop chunks shorter than `min_informative_chars` before they're
    eligible for pairing, so triples aren't built from boilerplate fragments
    (e.g. "Full-time", "Bachelor's degree") that chunking.py's own
    min_chunk_chars=15 lets through. If a document has no chunk that clears
    the bar, keep its single longest chunk rather than losing the document
    entirely.
    """
    filtered = {}
    for doc_id, chunks in chunks_by_doc.items():
        kept = [c for c in chunks if len(c) >= min_informative_chars]
        filtered[doc_id] = kept if kept else [max(chunks, key=len)]
    return filtered


def _weighted_choice(rng: random.Random, chunks: List[str]) -> str:
    """Pick one chunk with probability proportional to its length, so longer
    (more informative) chunks are favored over uniform random choice."""
    return rng.choices(chunks, weights=[len(c) for c in chunks], k=1)[0]


def _weighted_sample_without_replacement(rng: random.Random, chunks: List[str], k: int) -> List[str]:
    pool = list(chunks)
    picks = []
    for _ in range(min(k, len(pool))):
        pick = _weighted_choice(rng, pool)
        picks.append(pick)
        pool.remove(pick)
    return picks


def build_training_triples(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    chunking_cfg: dict,
    triples_cfg: dict,
    split_by_id: Dict[str, str],
    normalize_categories: bool = True,
) -> Tuple[List[Triple], Dict[str, int]]:
    """Returns (triples, stats) where `stats` counts why resumes were
    skipped, for reporting."""
    rng = random.Random(triples_cfg.get("random_seed", 42))
    chunks_per_resume = triples_cfg.get("chunks_per_resume", 3)

    resumes = resumes.copy()
    jobs = jobs.copy()
    resumes["norm_category"] = (
        resumes["category"].map(normalize_category) if normalize_categories else resumes["category"]
    )
    jobs["norm_category"] = (
        jobs["category"].map(normalize_category) if normalize_categories else jobs["category"]
    )

    min_informative_chars = triples_cfg.get("min_informative_chars", 40)
    resume_chunks = _filter_informative(_chunk_documents(resumes, chunking_cfg), min_informative_chars)
    job_chunks = _filter_informative(_chunk_documents(jobs, chunking_cfg), min_informative_chars)

    jobs_by_category = {cat: grp for cat, grp in jobs.groupby("norm_category")}
    all_categories = list(jobs_by_category.keys())

    stats = {
        "resumes_total": len(resumes),
        "skipped_no_resume_chunks": 0,
        "skipped_no_positive_category": 0,
        "skipped_no_negative_category": 0,
    }

    triples: List[Triple] = []
    for _, resume in resumes.iterrows():
        r_chunks = resume_chunks.get(resume["doc_id"])
        if not r_chunks:
            stats["skipped_no_resume_chunks"] += 1
            continue

        pos_candidates = jobs_by_category.get(resume["norm_category"])
        if pos_candidates is None or len(pos_candidates) == 0:
            stats["skipped_no_positive_category"] += 1
            continue

        neg_categories = [c for c in all_categories if c != resume["norm_category"]]
        if not neg_categories:
            stats["skipped_no_negative_category"] += 1
            continue

        sampled_r_chunks = _weighted_sample_without_replacement(rng, r_chunks, chunks_per_resume)

        for r_chunk in sampled_r_chunks:
            pos_job = pos_candidates.sample(n=1, random_state=rng.randint(0, 2**31)).iloc[0]
            pos_chunks = job_chunks.get(pos_job["doc_id"])
            if not pos_chunks:
                continue

            neg_category = rng.choice(neg_categories)
            neg_job = jobs_by_category[neg_category].sample(n=1, random_state=rng.randint(0, 2**31)).iloc[0]
            neg_chunks = job_chunks.get(neg_job["doc_id"])
            if not neg_chunks:
                continue

            triples.append(
                Triple(
                    resume_chunk=r_chunk,
                    positive_jd_chunk=_weighted_choice(rng, pos_chunks),
                    negative_jd_chunk=_weighted_choice(rng, neg_chunks),
                    category=resume["category"],
                    resume_id=resume["doc_id"],
                    positive_jd_id=pos_job["doc_id"],
                    negative_jd_id=neg_job["doc_id"],
                    split=split_by_id[resume["doc_id"]],
                )
            )

    return triples, stats
