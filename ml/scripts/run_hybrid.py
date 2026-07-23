"""Phase 3b entry point: build a hybrid TF-IDF + fine-tuned-SBERT matcher
(final_score = alpha * tfidf_score + (1 - alpha) * sbert_score), sweep alpha
on the val split to pick a default, evaluate that alpha on the held-out test
split via the same harness as scripts/run_baselines.py / finetune_sbert.py,
and add it to results/phase3_comparison_table.md.

Also runs a score-normalized variant (each matcher's similarity row min-max
scaled to [0, 1] per query before blending) alongside the raw one: TF-IDF's
cosine scores are near-zero for most pairs with occasional sharp peaks,
while this fine-tuned SBERT's scores cluster tightly near 1.0 (typical
embedding anisotropy) - summing raw scores lets SBERT's small-but-nonzero
variance perturb TF-IDF's ranking regardless of alpha, so the normalized
variant is the fairer blend and is worth comparing directly against the raw
one rather than assuming either wins.

Usage (run from the ml/ directory):
    python -m scripts.run_hybrid
    python -m scripts.run_hybrid --alphas 0.3 0.5 0.7
    python -m scripts.run_hybrid --config config/config.yaml
"""
import argparse
import json
import logging
from pathlib import Path

from src.config import load_config
from src.data.loader import load_paired_dataset
from src.data.split import stratified_split_ids
from src.evaluation.evaluate import evaluate_matcher
from src.evaluation.report import format_category_table, format_metrics_table
from src.models.hybrid_baseline import HybridMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_hybrid")


def sweep_and_evaluate(
    *, finetuned_model_dir, alphas, ks, min_matched_score,
    val_resumes, val_jobs, test_resumes, test_jobs, normalize, matcher_name,
):
    label = "hybrid (normalized)" if normalize else "hybrid (raw)"
    logger.info("Sweeping alpha in %s on the val split - %s...", list(alphas), label)
    sweep_results = []
    for alpha in alphas:
        matcher = HybridMatcher(
            sbert_checkpoint=finetuned_model_dir, alpha=alpha, normalize=normalize,
        )
        result = evaluate_matcher(matcher, val_resumes, val_jobs, min_matched_score=min_matched_score, ks=ks)
        logger.info(
            "[%s] alpha=%.2f: %s MRR=%.4f",
            label, alpha,
            " ".join(f"P@{k}={result['overall'][f'precision_at_{k}']:.4f}" for k in ks),
            result["overall"]["reciprocal_rank"],
        )
        sweep_results.append({"alpha": alpha, **result["overall"]})

    best = max(sweep_results, key=lambda r: r["reciprocal_rank"])
    best_alpha = best["alpha"]
    logger.info("[%s] Best alpha on val (by MRR): %.2f (MRR=%.4f)", label, best_alpha, best["reciprocal_rank"])

    logger.info("[%s] Evaluating alpha=%.2f on the test split...", label, best_alpha)
    matcher = HybridMatcher(
        sbert_checkpoint=finetuned_model_dir, alpha=best_alpha, normalize=normalize, name=matcher_name,
    )
    result = evaluate_matcher(matcher, test_resumes, test_jobs, min_matched_score=min_matched_score, ks=ks)
    logger.info(
        "%s: n_queries=%d %s MRR=%.4f",
        matcher_name,
        result["overall"]["n_queries"],
        " ".join(f"P@{k}={result['overall'][f'precision_at_{k}']:.4f}" for k in ks),
        result["overall"]["reciprocal_rank"],
    )
    return sweep_results, best_alpha, result


def main(config_path: str = None, alphas=(0.3, 0.5, 0.7)) -> None:
    cfg = load_config(config_path)
    ks = cfg["evaluation"].get("ks", [1, 3, 5])
    min_matched_score = cfg["pairing"].get("min_matched_score", 0.7)
    finetuned_model_dir = cfg["paths"]["finetuned_model_dir"]

    if not Path(finetuned_model_dir).exists():
        raise FileNotFoundError(
            f"No fine-tuned model at {finetuned_model_dir} - run scripts/finetune_sbert.py first."
        )

    logger.info("Loading paired dataset...")
    resumes, jobs = load_paired_dataset(cfg["paths"]["paired_dataset"], cfg["columns"]["paired"])

    split_cfg = cfg["split"]
    split_by_id = stratified_split_ids(
        resumes,
        train_frac=split_cfg.get("train_frac", 0.8),
        val_frac=split_cfg.get("val_frac", 0.1),
        test_frac=split_cfg.get("test_frac", 0.1),
        random_seed=split_cfg.get("random_seed", 42),
    )

    def split_slice(split_name: str):
        r = resumes[resumes["doc_id"].map(split_by_id) == split_name].reset_index(drop=True)
        j = jobs[jobs["doc_id"].map(split_by_id) == split_name].reset_index(drop=True)
        return r, j

    val_resumes, val_jobs = split_slice("val")
    test_resumes, test_jobs = split_slice("test")

    common = dict(
        finetuned_model_dir=finetuned_model_dir, alphas=alphas, ks=ks, min_matched_score=min_matched_score,
        val_resumes=val_resumes, val_jobs=val_jobs, test_resumes=test_resumes, test_jobs=test_jobs,
    )
    raw_sweep, raw_best_alpha, raw_result = sweep_and_evaluate(
        **common, normalize=False, matcher_name="hybrid_tfidf_sbert"
    )
    norm_sweep, norm_best_alpha, norm_result = sweep_and_evaluate(
        **common, normalize=True, matcher_name="hybrid_tfidf_sbert_normalized"
    )

    # --- Merge with the existing Phase 3 results for one comparable table ---
    phase3_results_path = Path(cfg["paths"]["phase3_results_json"])
    with open(phase3_results_path, encoding="utf-8") as f:
        phase3_report = json.load(f)

    baseline_results = [
        r for name, r in phase3_report["results"].items()
        if name not in (raw_result["matcher"], norm_result["matcher"])
    ]
    all_results = baseline_results + [raw_result, norm_result]
    phase3_report["results"] = {r["matcher"]: r for r in all_results}
    phase3_report["config"]["hybrid_alpha_sweep_raw"] = {"split": "val", "results": raw_sweep}
    phase3_report["config"]["hybrid_alpha_sweep_normalized"] = {"split": "val", "results": norm_sweep}
    phase3_report["config"]["hybrid_best_alpha_raw"] = raw_best_alpha
    phase3_report["config"]["hybrid_best_alpha_normalized"] = norm_best_alpha

    with open(phase3_results_path, "w", encoding="utf-8") as f:
        json.dump(phase3_report, f, ensure_ascii=False, indent=2)

    table_path = Path(cfg["paths"]["phase3_results_table"])
    cfg_out = phase3_report["config"]
    table_md = (
        "# Phase 3 vs Phase 2 Comparison (test split)\n\n"
        f"Split: `test` | matched_score threshold: `{min_matched_score}` | "
        f"queries: `{raw_result['overall']['n_queries']}` | "
        f"candidate JDs: `{raw_result['overall']['n_candidates']}`\n\n"
        f"Val triplet accuracy: pretrained=`{cfg_out['val_triplet_accuracy_pretrained']:.4f}` -> "
        f"fine-tuned=`{cfg_out['val_triplet_accuracy_finetuned']:.4f}`\n\n"
        f"Hybrid alpha swept on val over {list(alphas)} (selected by MRR): "
        f"raw best alpha=`{raw_best_alpha:.2f}`, normalized best alpha=`{norm_best_alpha:.2f}` "
        f"(final_score = alpha * tfidf_score + (1 - alpha) * sbert_score; \"normalized\" min-max "
        f"scales each matcher's similarity row to [0, 1] per query before blending)\n\n"
        "## Overall\n\n"
        f"{format_metrics_table(all_results, ks)}\n\n"
        "## By category\n\n"
        f"{format_category_table(all_results, ks)}\n"
    )
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_md)

    logger.info("Done. Results -> %s | Table -> %s", phase3_results_path, table_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3b: hybrid TF-IDF + fine-tuned SBERT matcher.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to ml/config/config.yaml)")
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7], help="Alpha values to sweep on the val split."
    )
    args = parser.parse_args()
    main(args.config, alphas=args.alphas)
