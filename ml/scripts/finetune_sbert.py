"""Phase 3 entry point: fine-tune model.base_checkpoint (same checkpoint as
the Phase 2 SBERT baseline) on the resume-JD triples built in Phase 1, using
MultipleNegativesRankingLoss.

Split usage:
  - train: the only data the model's weights are fit on.
  - val:   used only for TripletEvaluator-based best-checkpoint selection
           (load_best_model_at_end) - never backpropagated through.
  - test:  never loaded for training/tuning. This script evaluates the
           final fine-tuned model on it at the end, but only by reusing
           src/evaluation/evaluate.py exactly as scripts/run_baselines.py
           does for the Phase 2 baselines - same code path, so the
           resulting P@k/MRR numbers are directly comparable.

Usage (run from the ml/ directory):
    python -m scripts.finetune_sbert
    python -m scripts.finetune_sbert --config config/config.yaml

    # Resume an interrupted run from its last saved checkpoint:
    python -m scripts.finetune_sbert --resume_from_checkpoint models/finetuned-sbert/checkpoints/checkpoint-<N>
    # ... or let it auto-find the latest checkpoint under that directory:
    python -m scripts.finetune_sbert --resume_from_checkpoint auto
"""
import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from transformers.trainer_utils import get_last_checkpoint

from src.config import load_config
from src.data.loader import load_job_postings, load_relabeled_resumes
from src.data.split import stratified_split_ids
from src.data.triples_dataset import load_triples, triples_to_columns
from src.evaluation.evaluate import evaluate_matcher
from src.evaluation.report import format_category_table, format_metrics_table
from src.models.sbert_baseline import SbertMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_sbert")


def main(config_path: str = None, resume_from_checkpoint: str = None) -> None:
    cfg = load_config(config_path)
    train_cfg = cfg["training"]

    logger.info("Loading train/val triples from %s", cfg["paths"]["triples_output"])
    train_records = load_triples(cfg["paths"]["triples_output"], split="train")
    val_records = load_triples(cfg["paths"]["triples_output"], split="val")
    logger.info("train triples: %d | val triples: %d", len(train_records), len(val_records))
    if not train_records or not val_records:
        raise ValueError("No train/val triples found - run `python -m scripts.build_dataset` first.")

    train_dataset = Dataset.from_dict(triples_to_columns(train_records))
    val_columns = triples_to_columns(val_records)
    val_dataset = Dataset.from_dict(val_columns)

    logger.info("Loading base checkpoint: %s", cfg["model"]["base_checkpoint"])
    model = SentenceTransformer(cfg["model"]["base_checkpoint"])
    loss = MultipleNegativesRankingLoss(model)

    val_evaluator = TripletEvaluator(
        anchors=val_columns["anchor"],
        positives=val_columns["positive"],
        negatives=val_columns["negative"],
        name="val",
        batch_size=train_cfg.get("eval_batch_size", 32),
        show_progress_bar=False,
    )

    pretrained_val_metrics = val_evaluator(model)
    pretrained_val_accuracy = pretrained_val_metrics[val_evaluator.primary_metric]
    logger.info("Pretrained checkpoint val triplet accuracy (before fine-tuning): %.4f", pretrained_val_accuracy)

    output_dir = Path(cfg["paths"]["finetuned_model_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_from_checkpoint == "auto":
        found = get_last_checkpoint(str(checkpoints_dir)) if checkpoints_dir.exists() else None
        if found is None:
            logger.warning("--resume_from_checkpoint auto: no checkpoint found under %s, starting from scratch.", checkpoints_dir)
        resume_from_checkpoint = found

    args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=train_cfg.get("epochs", 4),
        per_device_train_batch_size=train_cfg.get("batch_size", 16),
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 32),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        # Prevents exact-duplicate anchor/positive texts from landing in the
        # same batch, which would otherwise be scored as false in-batch
        # negatives by MultipleNegativesRankingLoss.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Step-based (not epoch-based) so an interrupted run always has a
        # recent checkpoint to resume from, rather than losing up to a full
        # epoch of progress. eval/save steps must match exactly - HF Trainer
        # requires this whenever load_best_model_at_end=True.
        eval_strategy="steps",
        eval_steps=train_cfg.get("save_eval_steps", 200),
        save_strategy="steps",
        save_steps=train_cfg.get("save_eval_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="val_cosine_accuracy",
        greater_is_better=True,
        logging_steps=50,
        seed=train_cfg.get("seed", 42),
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=val_evaluator,
    )

    logger.info(
        "Starting fine-tuning: epochs=%d batch_size=%d lr=%s (train=%d, val=%d triples)%s",
        args.num_train_epochs,
        args.per_device_train_batch_size,
        args.learning_rate,
        len(train_records),
        len(val_records),
        f" - resuming from {resume_from_checkpoint}" if resume_from_checkpoint else "",
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    finetuned_val_metrics = val_evaluator(model)
    finetuned_val_accuracy = finetuned_val_metrics[val_evaluator.primary_metric]
    logger.info(
        "Val triplet accuracy: pretrained=%.4f -> fine-tuned (best checkpoint)=%.4f",
        pretrained_val_accuracy,
        finetuned_val_accuracy,
    )

    model.save(str(output_dir))
    logger.info("Saved fine-tuned model to %s", output_dir)

    # --- Evaluate on the held-out TEST split, via the exact same Phase 2 harness ---
    logger.info("Evaluating fine-tuned model on the test split (untouched until now)...")
    target_categories = cfg["categories"]["target"]
    normalize_categories = cfg["pairing"].get("normalize_categories", True)
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
    # candidate pool is the full target-category job pool, same as run_baselines.py.

    ks = cfg["evaluation"].get("ks", [1, 3, 5])

    finetuned_matcher = SbertMatcher(checkpoint=str(output_dir), name="sbert_finetuned")
    finetuned_result = evaluate_matcher(
        finetuned_matcher, test_resumes, jobs, ks=ks, normalize_categories=normalize_categories
    )
    logger.info(
        "sbert_finetuned: n_queries=%d %s MRR=%.4f",
        finetuned_result["overall"]["n_queries"],
        " ".join(f"P@{k}={finetuned_result['overall'][f'precision_at_{k}']:.4f}" for k in ks),
        finetuned_result["overall"]["reciprocal_rank"],
    )

    # Merge with the already-saved Phase 2 baseline results for one directly comparable table
    all_results = [finetuned_result]
    baseline_results_path = Path(cfg["paths"]["baseline_results_json"])
    if baseline_results_path.exists():
        with open(baseline_results_path, encoding="utf-8") as f:
            baseline_report = json.load(f)
        all_results = list(baseline_report["results"].values()) + [finetuned_result]
    else:
        logger.warning(
            "No Phase 2 baseline results at %s - run scripts/run_baselines.py first for a full comparison.",
            baseline_results_path,
        )

    phase3_report = {
        "config": {
            "eval_split": "test",
            "ks": list(ks),
            "training": train_cfg,
            "base_checkpoint": cfg["model"]["base_checkpoint"],
            "finetuned_model_dir": str(output_dir),
            "val_triplet_accuracy_pretrained": pretrained_val_accuracy,
            "val_triplet_accuracy_finetuned": finetuned_val_accuracy,
        },
        "results": {r["matcher"]: r for r in all_results},
    }

    results_json_path = Path(cfg["paths"]["phase3_results_json"])
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(phase3_report, f, ensure_ascii=False, indent=2)

    table_path = Path(cfg["paths"]["phase3_results_table"])
    table_md = (
        "# Phase 3 vs Phase 2 Comparison (test split)\n\n"
        f"Split: `test` | relevance: same (normalized) category | "
        f"queries: `{finetuned_result['overall']['n_queries']}` | "
        f"candidate JDs: `{finetuned_result['overall']['n_candidates']}`\n\n"
        f"Val triplet accuracy: pretrained=`{pretrained_val_accuracy:.4f}` -> "
        f"fine-tuned=`{finetuned_val_accuracy:.4f}`\n\n"
        "## Overall\n\n"
        f"{format_metrics_table(all_results, ks)}\n\n"
        "## By category\n\n"
        f"{format_category_table(all_results, ks)}\n"
    )
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_md)

    logger.info("Done. Results -> %s | Table -> %s", results_json_path, table_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: fine-tune SBERT with MultipleNegativesRankingLoss.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to ml/config/config.yaml)")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a checkpoint dir under models/finetuned-sbert/checkpoints/ to resume from, "
        "or 'auto' to resume from the latest one found there.",
    )
    args = parser.parse_args()
    main(args.config, resume_from_checkpoint=args.resume_from_checkpoint)
