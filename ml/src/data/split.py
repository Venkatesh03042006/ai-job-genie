"""Stratified train/val/test split, applied at the resume level so that a
given resume's pairs/triples all land in the same split - a plain global
random split would risk leaking near-duplicate chunks from one resume
across train and test, and could easily starve small categories (e.g. the
41 qualifying "Business Development Executive" resumes) of test coverage.
"""
import random
from typing import Dict

import pandas as pd


def stratified_split_ids(
    df: pd.DataFrame,
    category_col: str = "category",
    id_col: str = "doc_id",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, str]:
    """Assign every row's id to "train" / "val" / "test", splitting each
    category group independently in the given proportions - so every
    category (not just the high-volume ones) is represented in val/test.

    Returns {doc_id: split_name}.
    """
    fracs_sum = train_frac + val_frac + test_frac
    if abs(fracs_sum - 1.0) > 1e-6:
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0, got {fracs_sum}")

    rng = random.Random(random_seed)
    split_by_id: Dict[str, str] = {}

    for _, group in df.groupby(category_col):
        ids = group[id_col].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_train = min(round(n * train_frac), n)
        n_val = min(round(n * val_frac), n - n_train)

        for doc_id in ids[:n_train]:
            split_by_id[doc_id] = "train"
        for doc_id in ids[n_train : n_train + n_val]:
            split_by_id[doc_id] = "val"
        for doc_id in ids[n_train + n_val :]:
            split_by_id[doc_id] = "test"

    return split_by_id
