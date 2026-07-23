"""Ranking metrics for resume->JD retrieval: Precision@k and reciprocal
rank, computed per query (resume) over a ranked candidate pool of JDs with
binary relevance.

Resumes and JDs are independently sourced (see src/data/loader.py) - there
is no per-pair ground-truth label, so relevance is category co-membership
(see src/data/pairing.py). A query resume typically has several relevant
JDs in the pool, not exactly one, so Precision@k here is a genuine "fraction
of the top k that are on-category" rather than a disguised hit@k.
"""
from typing import Dict, List, Sequence


def precision_at_k(ranked_relevant_flags: Sequence[bool], k: int) -> float:
    """ranked_relevant_flags[i] is True if the i-th ranked candidate is
    relevant. Precision@k = (# relevant in top k) / k."""
    if k <= 0:
        return 0.0
    top_k = ranked_relevant_flags[:k]
    return sum(top_k) / k


def reciprocal_rank(ranked_relevant_flags: Sequence[bool]) -> float:
    """1 / rank of the first relevant candidate (1-indexed); 0 if none found."""
    for i, is_relevant in enumerate(ranked_relevant_flags, start=1):
        if is_relevant:
            return 1.0 / i
    return 0.0


def aggregate_metrics(per_query_metrics: List[Dict[str, float]], metric_keys: Sequence[str]) -> Dict[str, float]:
    """Mean each of `metric_keys` across all queries. Returns 0.0 for every
    key if there are no queries, rather than dividing by zero."""
    if not per_query_metrics:
        return {k: 0.0 for k in metric_keys}
    return {k: sum(m[k] for m in per_query_metrics) / len(per_query_metrics) for k in metric_keys}
