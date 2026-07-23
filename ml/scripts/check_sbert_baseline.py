"""Sanity-check for src/models/sbert_baseline.py - not part of the Phase 2
pipeline, run on demand to visually/structurally verify the SBERT baseline
before trusting its Precision@k/MRR numbers:

  1. embeddings are L2-normalized before cosine similarity (prints norms)
  2. mean pooling is used, not CLS (prints the model's Pooling config)
  3. prints example resume/JD pairs with their cosine similarity, including
     one weak-positive pair (same category - resumes and JDs are
     independently sourced, see src/data/loader.py, so there is no
     per-pair verified label), so the scores can be eyeballed for
     degeneracy (everything ~1.0 or ~0.0 regardless of text)

Usage (run from the ml/ directory):
    python -m scripts.check_sbert_baseline
"""
import argparse

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import load_config
from src.data.loader import load_job_postings, load_relabeled_resumes
from src.data.split import stratified_split_ids
from src.models.sbert_baseline import SbertMatcher


def _truncate(text: str, n: int = 200) -> str:
    text = text.replace("\n", " / ")
    return text if len(text) <= n else text[:n] + "..."


def main(config_path: str = None) -> None:
    cfg = load_config(config_path)
    matcher = SbertMatcher(checkpoint=cfg["model"]["base_checkpoint"])

    print("=== 1. Pooling configuration (Pooling module of the SentenceTransformer pipeline) ===")
    pooling_module = matcher.model[1]
    pooling_cfg = pooling_module.get_config_dict()
    print(pooling_cfg)
    assert pooling_cfg["pooling_mode_mean_tokens"] is True
    assert pooling_cfg["pooling_mode_cls_token"] is False
    print("-> mean pooling confirmed (pooling_mode_mean_tokens=True, pooling_mode_cls_token=False)\n")

    print("=== 2. L2 normalization check ===")
    probe_vecs = matcher.encode(["Example probe sentence one.", "A different, unrelated probe sentence."])
    norms = np.linalg.norm(probe_vecs, axis=1)
    print(f"Embedding norms: {norms}")
    assert np.allclose(norms, 1.0, atol=1e-5)
    print("-> embeddings are L2-normalized (norm == 1.0), so sklearn's cosine_similarity on them is true cosine\n")

    print("=== 3. Example resume/JD pairs with cosine similarity ===")
    target_categories = cfg["categories"]["target"]
    resumes = load_relabeled_resumes(cfg["paths"]["resumes_relabeled_cache"], target_categories)
    jobs = load_job_postings(cfg["paths"]["postings_cache"], target_categories)
    split_cfg = cfg["split"]
    split_by_id = stratified_split_ids(
        resumes,
        train_frac=split_cfg.get("train_frac", 0.8),
        val_frac=split_cfg.get("val_frac", 0.1),
        test_frac=split_cfg.get("test_frac", 0.1),
        random_seed=split_cfg.get("random_seed", 42),
    )
    test_resumes = resumes[resumes["doc_id"].map(split_by_id) == "test"].reset_index(drop=True)
    # jobs are never split (see src/data/loader.py / src/data/split.py) - the
    # candidate pool is the full target-category job pool.

    first_resume = test_resumes.iloc[0]
    same_category_jobs = jobs[jobs["category"] == first_resume["category"]].reset_index(drop=True)
    other_category_job = jobs[jobs["category"] != first_resume["category"]].iloc[0]

    second_resume = test_resumes.iloc[len(test_resumes) // 2]
    second_same_category_jobs = jobs[jobs["category"] == second_resume["category"]].reset_index(drop=True)

    examples = [
        (
            "WEAK POSITIVE (same category=%s; resumes/JDs are independently sourced, "
            "so this is category co-membership, not a verified per-pair label)"
            % first_resume["category"],
            first_resume["text"],
            same_category_jobs.iloc[0]["text"],
        ),
        (
            "MISMATCH (resume #1's category=%s vs. a JD from a different category=%s)"
            % (first_resume["category"], other_category_job["category"]),
            first_resume["text"],
            other_category_job["text"],
        ),
        (
            "WEAK POSITIVE #2 (same category=%s)" % second_resume["category"],
            second_resume["text"],
            second_same_category_jobs.iloc[0]["text"],
        ),
    ]

    for label, resume_text, jd_text in examples:
        vecs = matcher.encode([resume_text, jd_text])
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0, 0]
        print(f"--- {label} ---")
        print(f"cosine similarity: {sim:.4f}")
        print(f"resume_text: {_truncate(resume_text)}")
        print(f"jd_text:     {_truncate(jd_text)}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check the SBERT baseline (Phase 2).")
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to ml/config/config.yaml)")
    args = parser.parse_args()
    main(args.config)
