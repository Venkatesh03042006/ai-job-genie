"""Phase 1 entry point: load the deduplicated/content-relabeled resume
dataset and the cached/filtered postings dataset, build weak-supervision
(category-based) pairs and training triples, and write them to
data/processed/ for Phase 2 (baselines) and Phase 3 (SBERT fine-tuning) to
consume.

Expects `data/processed/postings_filtered.parquet` (run
`python -m scripts.build_postings_cache` first) and
`data/processed/resumes_relabeled.parquet` (run
`python -m scripts.build_resume_labels_cache` first) to already exist -
both are separate, heavier steps not re-run on every invocation.

Usage (run from the ml/ directory):
    python -m scripts.build_dataset
    python -m scripts.build_dataset --config config/config.yaml
"""
import argparse
import collections
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.config import load_config
from src.data.loader import load_job_postings, load_relabeled_resumes
from src.data.pairing import build_pairs
from src.data.split import stratified_split_ids
from src.data.triples import build_training_triples

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dataset")


def _write_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def main(config_path: str = None) -> None:
    cfg = load_config(config_path)
    target_categories = cfg["categories"]["target"]

    logger.info("Loading relabeled resumes from %s", cfg["paths"]["resumes_relabeled_cache"])
    resumes = load_relabeled_resumes(cfg["paths"]["resumes_relabeled_cache"], target_categories)
    logger.info("Loaded %d resumes across %d categories", len(resumes), resumes["category"].nunique())

    logger.info("Loading cached postings from %s", cfg["paths"]["postings_cache"])
    jobs = load_job_postings(cfg["paths"]["postings_cache"], target_categories)
    logger.info("Loaded %d job postings across %d categories", len(jobs), jobs["category"].nunique())

    for cat in target_categories:
        n_resumes = (resumes["category"] == cat).sum()
        n_jobs = (jobs["category"] == cat).sum()
        logger.info("  %-95s resumes=%-5d postings=%-5d", cat, n_resumes, n_jobs)
        if n_resumes == 0 or n_jobs == 0:
            logger.warning("Category %r has zero resumes or zero postings - it will produce no pairs/triples.", cat)

    split_cfg = cfg["split"]
    split_by_id = stratified_split_ids(
        resumes,
        train_frac=split_cfg.get("train_frac", 0.8),
        val_frac=split_cfg.get("val_frac", 0.1),
        test_frac=split_cfg.get("test_frac", 0.1),
        random_seed=split_cfg.get("random_seed", 42),
    )
    split_counts = collections.Counter(split_by_id.values())
    logger.info(
        "Stratified split by category: train=%d val=%d test=%d",
        split_counts["train"],
        split_counts["val"],
        split_counts["test"],
    )

    logger.info("Building weak-supervision pairs...")
    pairs = build_pairs(resumes, jobs, cfg["pairing"], split_by_id)
    n_pos = sum(1 for p in pairs if p.label == 1)
    pair_split_counts = collections.Counter(p.split for p in pairs)
    logger.info(
        "Built %d pairs (%d positive, %d negative) | split: train=%d val=%d test=%d",
        len(pairs),
        n_pos,
        len(pairs) - n_pos,
        pair_split_counts["train"],
        pair_split_counts["val"],
        pair_split_counts["test"],
    )
    _write_jsonl(pairs, Path(cfg["paths"]["pairs_output"]))

    logger.info("Building training triples...")
    triples, triple_stats = build_training_triples(
        resumes,
        jobs,
        cfg["chunking"],
        cfg["triples"],
        split_by_id,
        normalize_categories=cfg["pairing"].get("normalize_categories", True),
    )
    triple_split_counts = collections.Counter(t.split for t in triples)
    logger.info(
        "Built %d training triples | split: train=%d val=%d test=%d",
        len(triples),
        triple_split_counts["train"],
        triple_split_counts["val"],
        triple_split_counts["test"],
    )
    logger.info(
        "Resumes skipped: %d/%d no resume chunks, %d no same-category postings, %d no cross-category postings",
        triple_stats["skipped_no_resume_chunks"],
        triple_stats["resumes_total"],
        triple_stats["skipped_no_positive_category"],
        triple_stats["skipped_no_negative_category"],
    )
    _write_jsonl(triples, Path(cfg["paths"]["triples_output"]))

    logger.info("Category-level triple counts:")
    triple_cat_counts = collections.Counter(t.category for t in triples)
    for cat in target_categories:
        logger.info("  %-95s %d", cat, triple_cat_counts.get(cat, 0))

    logger.info(
        "Done. Pairs -> %s | Triples -> %s",
        cfg["paths"]["pairs_output"],
        cfg["paths"]["triples_output"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: build weak-supervision pairs and training triples.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to ml/config/config.yaml)")
    args = parser.parse_args()
    main(args.config)
