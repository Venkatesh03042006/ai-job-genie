"""Content-based category assignment for src/data/loader.py's deduplicated
resume profiles - see docs/PHASE3_FINDINGS.md's "Category label mismatch"
section. resume_data.csv's `job_position_name` is a cross-product label
stamped onto every one of a profile's ~14-28 category-copies regardless of
the profile's actual content, so (once deduplicated) it carries no signal
at all - there's only one profile left, not one per category. This module
derives a real label instead: for each profile, the target category whose
job-postings pool it's most textually similar to.

A profile is assigned to the category maximizing a CONSENSUS of several
matchers (typically TF-IDF + pretrained SBERT checkpoints - see
scripts/build_resume_labels_cache.py) rather than any single one, to avoid
favoring one baseline's own later Phase 2 evaluation: if TF-IDF alone were
used to relabel, TF-IDF's Phase 2 P@k/MRR would be inflated near-
tautologically (it would tend to rediscover the very label it assigned),
while SBERT wouldn't share that advantage, making the two incomparable.
Consensus doesn't eliminate this asymmetry - all matchers used here still
get partial credit for having helped define the labels, which anything
evaluated later without being part of the relabeling (e.g. Phase 3's
fine-tuned model) won't get - it only distributes the advantage evenly
across the Phase 2 baselines being compared to each other. Documented as an
explicit limitation in docs/PHASE3_FINDINGS.md.
"""
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseMatcher


def _category_mean_similarity(sims: np.ndarray, jobs: pd.DataFrame, target_categories: List[str]) -> np.ndarray:
    """sims: (n_resumes, n_jobs) -> (n_resumes, n_categories), each entry
    the resume's mean similarity to that category's job pool."""
    job_categories = jobs["category"].to_numpy()
    out = np.zeros((sims.shape[0], len(target_categories)))
    for j, cat in enumerate(target_categories):
        mask = job_categories == cat
        out[:, j] = sims[:, mask].mean(axis=1) if mask.any() else 0.0
    return out


def _row_min_max_normalize(mat: np.ndarray) -> np.ndarray:
    """Per-resume (row) min-max scale each matcher's category-mean-
    similarity vector to [0, 1] before averaging across matchers - matchers'
    raw similarity scales aren't comparable (see src/models/hybrid_baseline.py
    for the same issue/fix in the TF-IDF/SBERT score-blending context)."""
    row_min = mat.min(axis=1, keepdims=True)
    row_max = mat.max(axis=1, keepdims=True)
    span = row_max - row_min
    span[span == 0] = 1.0  # constant row - leave at 0 rather than divide by 0
    return (mat - row_min) / span


def assign_content_categories(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    target_categories: List[str],
    matchers: List[BaseMatcher],
) -> pd.DataFrame:
    """Returns a copy of `resumes` with a new `category` column (the
    content-derived, consensus label) and `category_score` (the winning
    category's consensus score, for inspection), plus one
    `<matcher.name>_category` column per matcher recording that matcher's
    own individual pick - so agreement/disagreement between matchers can be
    audited (see scripts/build_resume_labels_cache.py).
    """
    resumes = resumes.copy()
    resume_texts = resumes["text"].tolist()
    job_texts = jobs["text"].tolist()

    normalized_mats = []
    for matcher in matchers:
        matcher.fit(resume_texts, job_texts)
        resume_vecs = matcher.encode(resume_texts)
        job_vecs = matcher.encode(job_texts)
        sims = cosine_similarity(resume_vecs, job_vecs)

        cat_means = _category_mean_similarity(sims, jobs, target_categories)
        resumes[f"{matcher.name}_category"] = [target_categories[i] for i in cat_means.argmax(axis=1)]
        normalized_mats.append(_row_min_max_normalize(cat_means))

    consensus = np.mean(normalized_mats, axis=0)
    best_idx = consensus.argmax(axis=1)
    resumes["category"] = [target_categories[i] for i in best_idx]
    resumes["category_score"] = consensus[np.arange(len(resumes)), best_idx]
    return resumes
