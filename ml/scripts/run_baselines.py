"""Phase 2 entry point: evaluate Baseline A (TF-IDF) and Baseline B
(pretrained SBERT, no fine-tuning) on the held-out test split, and save a
comparison report to results/ for Phase 3's fine-tuned model to be measured
against.

Resumes and job postings are loaded independently (src/data/loader.py) and
matched only by category (see src/evaluation/evaluate.py) - there is no
per-pair ground truth. The train/val/test split is recomputed here via
src/data/split.py on a freshly loaded resumes DataFrame - deterministic
given the same config.yaml `split:` block, so it reproduces exactly the
split already baked into data/processed/pairs.jsonl / triples.jsonl by
build_dataset.py, without this script needing to parse those files just to
recover it. Only resumes are split (jobs never are - see split.py's
docstring), so the candidate JD pool is the full target-category job pool
regardless of eval_split.

CAVEAT (see docs/PHASE3_FINDINGS.md's "Category label mismatch" section):
resume categories here are the CONSENSUS label assigned by
scripts/build_resume_labels_cache.py, which is a blend of these same three
matchers' own similarity scores. That means every matcher evaluated below
had a hand in defining the ground truth it's being scored against - these
numbers are not a clean apples-to-apples baseline comparison, and should
not be read as such against anything (e.g. a Phase 3 fine-tuned model) that
didn't also contribute to labeling.

Usage (run from the ml/ directory):
    python -m scripts.run_baselines
    python -m scripts.run_baselines --config config/config.yaml
"""
import argparse
import json
import logging
from pathlib import Path

from src.config import load_config
from src.data.loader import load_job_postings, load_relabeled_resumes
from src.data.split import stratified_split_ids
from src.evaluation.evaluate import evaluate_matcher
from src.evaluation.report import format_category_table, format_metrics_table
from src.models.sbert_baseline import SbertMatcher
from src.models.tfidf_baseline import TfidfMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_baselines")


def main(config_path: str = None) -> None:
    cfg = load_config(config_path)
    eval_cfg = cfg["evaluation"]
    ks = eval_cfg.get("ks", [1, 3, 5])
    eval_split = eval_cfg.get("eval_split", "test")
    target_categories = cfg["categories"]["target"]
    normalize_categories = cfg["pairing"].get("normalize_categories", True)

    logger.info("Loading relabeled resumes and job postings...")
    resumes = load_relabeled_resumes(cfg["paths"]["resumes_relabeled_cache"], target_categories)
    jobs = load_job_postings(cfg["paths"]["postings_cache"], target_categories)
    logger.info("Loaded %d resumes and %d job postings across %d target categories", len(resumes), len(jobs), len(target_categories))

    split_cfg = cfg["split"]
    split_by_id = stratified_split_ids(
        resumes,
        train_frac=split_cfg.get("train_frac", 0.8),
        val_frac=split_cfg.get("val_frac", 0.1),
        test_frac=split_cfg.get("test_frac", 0.1),
        random_seed=split_cfg.get("random_seed", 42),
    )

    eval_resumes = resumes[resumes["doc_id"].map(split_by_id) == eval_split].reset_index(drop=True)
    logger.info(
        "Evaluating on split=%s: %d query resumes across %d categories, ranked against the full "
        "%d-JD candidate pool",
        eval_split,
        len(eval_resumes),
        eval_resumes["category"].nunique(),
        len(jobs),
    )

    matchers = [
        TfidfMatcher(),
        SbertMatcher(checkpoint=cfg["model"]["base_checkpoint"], name="sbert_pretrained"),
    ]
    for checkpoint in cfg["model"].get("additional_sbert_checkpoints", []):
        slug = checkpoint.split("/")[-1].lower().replace("-", "_")
        matchers.append(SbertMatcher(checkpoint=checkpoint, name=f"sbert_{slug}"))

    results = []
    for matcher in matchers:
        logger.info("Evaluating %s...", matcher.name)
        result = evaluate_matcher(matcher, eval_resumes, jobs, ks=ks, normalize_categories=normalize_categories)
        logger.info(
            "%s: n_queries=%d %s MRR=%.4f",
            result["matcher"],
            result["overall"]["n_queries"],
            " ".join(f"P@{k}={result['overall'][f'precision_at_{k}']:.4f}" for k in ks),
            result["overall"]["reciprocal_rank"],
        )
        results.append(result)

    report = {
        "config": {
            "eval_split": eval_split,
            "ks": list(ks),
            "split": split_cfg,
            "normalize_categories": normalize_categories,
            "sbert_checkpoints": {m.name: m.checkpoint for m in matchers if isinstance(m, SbertMatcher)},
            "labeling_caveat": (
                "Category labels are the consensus of these same matchers (see "
                "scripts/build_resume_labels_cache.py) - not a clean baseline comparison. "
                "See docs/PHASE3_FINDINGS.md's 'Category label mismatch' section."
            ),
        },
        "results": {r["matcher"]: r for r in results},
    }

    results_json_path = Path(cfg["paths"]["baseline_results_json"])
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    table_path = Path(cfg["paths"]["baseline_results_table"])
    table_md = (
        "# Phase 2 Baseline Results\n\n"
        f"Split: `{eval_split}` | relevance: same (normalized) category | "
        f"queries: `{results[0]['overall']['n_queries']}` | candidate JDs: `{results[0]['overall']['n_candidates']}`\n\n"
        "> **Caveat**: category labels are a consensus of these same three matchers "
        "(`scripts/build_resume_labels_cache.py`) - every matcher below helped define the ground "
        "truth it's scored against, so this is not a clean baseline comparison. See "
        "`docs/PHASE3_FINDINGS.md`'s \"Category label mismatch\" section.\n\n"
        "## Overall\n\n"
        f"{format_metrics_table(results, ks)}\n\n"
        "## By category\n\n"
        f"{format_category_table(results, ks)}\n"
    )
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_md)

    logger.info("Done. Results -> %s | Table -> %s", results_json_path, table_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: evaluate TF-IDF and pretrained SBERT baselines.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to ml/config/config.yaml)")
    args = parser.parse_args()
    main(args.config)
