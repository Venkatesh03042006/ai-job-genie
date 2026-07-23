"""Evaluate a matcher (src/models/base.py's BaseMatcher interface) on a
held-out split: rank a candidate pool of JDs against every query resume, and
score Precision@k / MRR using category co-membership as the relevance signal.

Resumes and JDs are independently sourced (see src/data/loader.py) - there
is no per-pair ground-truth label, so (as in src/data/pairing.py /
src/data/triples.py) a JD is treated as relevant to a resume if it shares
the resume's (normalized) category. A resume can have many relevant JDs in
the pool, not just one, so Precision@k here is genuinely "fraction of the
top k that are on-category", not a disguised hit@k.

`jobs` is never split by src/data/split.py (only resumes are - see its
docstring), so callers should typically pass the *full* target-category job
pool as `jobs` regardless of which split `resumes` is restricted to: ranking
a held-out resume against the whole live job market, not an artificially
shrunk slice of it, is both the realistic retrieval setting and the only
one available here.
"""
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation.metrics import aggregate_metrics, precision_at_k, reciprocal_rank
from src.models.base import BaseMatcher
from src.utils.text_cleaning import normalize_category

DEFAULT_KS = (1, 3, 5)


def evaluate_matcher(
    matcher: BaseMatcher,
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    ks: Sequence[int] = DEFAULT_KS,
    normalize_categories: bool = True,
) -> Dict:
    """Every row of `resumes` is used as a query, ranked against every row
    of `jobs` (the candidate pool). A JD is relevant to a query resume if it
    shares the resume's (normalized) category - resumes whose category has
    zero candidate JDs in `jobs` are skipped (reflected in `n_queries` being
    smaller than len(resumes)).
    """
    if resumes.empty:
        raise ValueError("No resumes given - cannot evaluate.")
    if jobs.empty:
        raise ValueError("No candidate JDs given - cannot evaluate.")

    resumes = resumes.reset_index(drop=True)
    jobs = jobs.reset_index(drop=True)

    job_categories = jobs["category"].map(normalize_category) if normalize_categories else jobs["category"]
    job_categories = job_categories.to_numpy()
    resume_categories = (
        resumes["category"].map(normalize_category) if normalize_categories else resumes["category"]
    )

    matcher.fit(resumes["text"].tolist(), jobs["text"].tolist())
    if hasattr(matcher, "similarity_matrix"):
        # Matchers like HybridMatcher blend two independently-computed
        # similarity matrices and can't be reduced to one encode() call.
        sims = matcher.similarity_matrix(resumes["text"].tolist(), jobs["text"].tolist())
    else:
        resume_vecs = matcher.encode(resumes["text"].tolist())
        job_vecs = matcher.encode(jobs["text"].tolist())
        sims = cosine_similarity(resume_vecs, job_vecs)  # (n_queries, n_jobs)

    metric_keys = [f"precision_at_{k}" for k in ks] + ["reciprocal_rank"]
    per_query_records = []

    for i, resume_row in resumes.iterrows():
        relevant_mask = job_categories == resume_categories.iloc[i]
        if not relevant_mask.any():
            continue  # no candidate JDs of this resume's category in the pool

        ranked_idx = np.argsort(-sims[i])
        ranked_relevant_flags = relevant_mask[ranked_idx].tolist()

        record = {f"precision_at_{k}": precision_at_k(ranked_relevant_flags, k) for k in ks}
        record["reciprocal_rank"] = reciprocal_rank(ranked_relevant_flags)
        record["rank_of_first_relevant"] = (
            ranked_relevant_flags.index(True) + 1 if True in ranked_relevant_flags else None
        )
        record["n_relevant"] = int(relevant_mask.sum())
        record["resume_id"] = resume_row["doc_id"]
        record["category"] = resume_row["category"]
        per_query_records.append(record)

    overall = aggregate_metrics(per_query_records, metric_keys)
    overall["n_queries"] = len(per_query_records)
    overall["n_candidates"] = len(jobs)

    by_category = {}
    if per_query_records:
        for category, group in pd.DataFrame(per_query_records).groupby("category"):
            group_records = group.to_dict("records")
            cat_metrics = aggregate_metrics(group_records, metric_keys)
            cat_metrics["n_queries"] = len(group_records)
            by_category[category] = cat_metrics

    return {
        "matcher": matcher.name,
        "overall": overall,
        "by_category": by_category,
    }
