"""Build positive and negative resume-JD pairs via weak (category-based)
supervision.

Positive pair: a resume and a JD sampled from the SAME (normalized)
category - assumed relevant, since both sides are independently sourced and
there is no real per-pair relevance label available (see
docs/PHASE3_FINDINGS.md - the dataset's `matched_score` was a formulaic
label from a synthetic cross-product, not a real signal).
Negative pair: a resume and a JD sampled from a DIFFERENT (normalized)
category - assumed irrelevant.
"""
import random
from typing import Dict, List

import pandas as pd

from src.data.schema import Pair
from src.utils.text_cleaning import normalize_category


def _with_norm_category(df: pd.DataFrame, normalize: bool) -> pd.DataFrame:
    df = df.copy()
    df["norm_category"] = df["category"].map(normalize_category) if normalize else df["category"]
    return df


def build_positive_pairs(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    split_by_id: Dict[str, str],
    positives_per_resume: int = 3,
    normalize_categories: bool = True,
    random_seed: int = 42,
) -> List[Pair]:
    """For each resume, sample `positives_per_resume` JDs from the same
    (normalized) category as positives."""
    rng = random.Random(random_seed)
    resumes = _with_norm_category(resumes, normalize_categories)
    jobs = _with_norm_category(jobs, normalize_categories)
    jobs_by_category = {cat: grp for cat, grp in jobs.groupby("norm_category")}

    pairs = []
    for _, resume in resumes.iterrows():
        candidates = jobs_by_category.get(resume["norm_category"])
        if candidates is None or len(candidates) == 0:
            continue
        n = min(positives_per_resume, len(candidates))
        sampled = candidates.sample(n=n, random_state=rng.randint(0, 2**31))
        for _, job in sampled.iterrows():
            pairs.append(
                Pair(
                    resume_id=resume["doc_id"],
                    jd_id=job["doc_id"],
                    category=resume["category"],
                    label=1,
                    split=split_by_id[resume["doc_id"]],
                )
            )
    return pairs


def build_negative_pairs(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    split_by_id: Dict[str, str],
    negatives_per_resume: int = 3,
    normalize_categories: bool = True,
    random_seed: int = 42,
) -> List[Pair]:
    rng = random.Random(random_seed)
    resumes = _with_norm_category(resumes, normalize_categories)
    jobs = _with_norm_category(jobs, normalize_categories)
    jobs_by_category = {cat: grp for cat, grp in jobs.groupby("norm_category")}
    all_categories = list(jobs_by_category.keys())

    pairs = []
    for _, resume in resumes.iterrows():
        other_categories = [c for c in all_categories if c != resume["norm_category"]]
        if not other_categories:
            continue
        sampled_categories = rng.sample(
            other_categories, k=min(negatives_per_resume, len(other_categories))
        )
        for category in sampled_categories:
            job = jobs_by_category[category].sample(n=1, random_state=rng.randint(0, 2**31)).iloc[0]
            pairs.append(
                Pair(
                    resume_id=resume["doc_id"],
                    jd_id=job["doc_id"],
                    category=resume["category"],
                    label=0,
                    split=split_by_id[resume["doc_id"]],
                )
            )
    return pairs


def build_pairs(
    resumes: pd.DataFrame, jobs: pd.DataFrame, pairing_cfg: dict, split_by_id: Dict[str, str]
) -> List[Pair]:
    """Convenience wrapper: builds positive + negative pairs from config.yaml's `pairing:` block.

    `split_by_id` (see src/data/split.py) assigns each pair's split from its
    anchor resume, so a resume's examples never straddle train/val/test.
    """
    positives = build_positive_pairs(
        resumes,
        jobs,
        split_by_id,
        positives_per_resume=pairing_cfg.get("positives_per_resume", 3),
        normalize_categories=pairing_cfg.get("normalize_categories", True),
        random_seed=pairing_cfg.get("random_seed", 42),
    )
    negatives = build_negative_pairs(
        resumes,
        jobs,
        split_by_id,
        negatives_per_resume=pairing_cfg.get("negatives_per_resume", 3),
        normalize_categories=pairing_cfg.get("normalize_categories", True),
        random_seed=pairing_cfg.get("random_seed", 42),
    )
    return positives + negatives
