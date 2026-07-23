"""Phase 4 entry point: demo the chunk-level explainability layer on a
handful of real resume/JD pairs from the test split, using the TF-IDF
matcher (still the strongest matcher for this dataset per
results/phase3_comparison_table.md).

Resumes and JDs are independently sourced (src/data/loader.py) - there is no
per-pair ground truth, so for each demo resume this script explains the
TF-IDF matcher's own top-1 ranked JD (against the full target-category job
pool) rather than a "known" matched row. That mirrors what a user would
actually see: the system's best recommendation for their resume, with a
chunk-level explanation of why.

Usage (run from the ml/ directory):
    python -m scripts.demo_explainability
    python -m scripts.demo_explainability --n-examples 5 --top-k 3
"""
import argparse
import logging
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from src.config import load_config
from src.data.loader import load_job_postings, load_relabeled_resumes
from src.data.split import stratified_split_ids
from src.explainability import count_exact_duplicate_pairs, explain_match, format_explanation
from src.explainability.chunk_similarity import chunk_document
from src.models.tfidf_baseline import TfidfMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo_explainability")


def main(config_path: str = None, n_examples: int = 5, top_k: int = 3) -> None:
    cfg = load_config(config_path)
    eval_split = cfg["evaluation"].get("eval_split", "test")
    target_categories = cfg["categories"]["target"]
    chunking_cfg = cfg["chunking"]

    logger.info("Loading relabeled resumes and job postings...")
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
    eval_resumes = resumes[resumes["doc_id"].map(split_by_id) == eval_split].reset_index(drop=True)
    # jobs are never split (see src/data/loader.py / src/data/split.py) - the
    # candidate pool is the full target-category job pool, same as run_baselines.py.

    # Same TF-IDF vocabulary/IDF used to score this split's document-level
    # ranking (see src/evaluation/evaluate.py) - reused here so chunk-level
    # scores are consistent with the ranking they're explaining.
    matcher = TfidfMatcher()
    matcher.fit(eval_resumes["text"].tolist(), jobs["text"].tolist())
    resume_vecs = matcher.encode(eval_resumes["text"].tolist())
    job_vecs = matcher.encode(jobs["text"].tolist())
    sims = cosine_similarity(resume_vecs, job_vecs)  # (n_resumes, n_jobs)
    top_job_idx = sims.argmax(axis=1)

    shuffled_order = eval_resumes.sample(frac=1.0, random_state=42).index.tolist()

    report_lines = [
        "# Phase 4 Explainability Demo\n",
        f"Split: `{eval_split}` | matcher: `tfidf` | top_k: `{top_k}` | "
        f"\"match\" = the matcher's own top-1 ranked JD from the full {len(jobs)}-JD "
        "target-category pool (see module docstring) | exact-duplicate chunk pairs excluded\n",
    ]

    found = 0
    n_no_genuine_match = 0
    n_scanned = 0

    for i in shuffled_order:
        if found >= n_examples:
            break
        n_scanned += 1

        resume_row = eval_resumes.iloc[i]
        jd_row = jobs.iloc[top_job_idx[i]]
        resume_chunks = chunk_document(resume_row["text"], chunking_cfg)
        jd_chunks = chunk_document(jd_row["text"], chunking_cfg)
        n_duplicates = count_exact_duplicate_pairs(resume_chunks, jd_chunks)

        matches = explain_match(
            resume_row["text"],
            jd_row["text"],
            matcher.vectorizer,
            chunking_cfg,
            top_k=top_k,
            exclude_exact_duplicates=True,
        )

        if not matches:
            n_no_genuine_match += 1
            continue

        found += 1
        same_category = jd_row["category"] == resume_row["category"]
        logger.info(
            "Resume %s (%s) <-> top-ranked JD %s (%s)%s:",
            resume_row["doc_id"], resume_row["category"],
            jd_row["doc_id"], jd_row["category"],
            "" if same_category else " [cross-category match]",
        )
        if n_duplicates:
            logger.info("  (%d exact-duplicate chunk pair(s) hidden)", n_duplicates)
        report_lines.append(
            f"\n## Resume `{resume_row['doc_id']}` ({resume_row['category']}) "
            f"<-> JD `{jd_row['doc_id']}` ({jd_row['category']})"
            f"{'' if same_category else ' _[cross-category match]_'}\n"
        )
        if n_duplicates:
            report_lines.append(
                f"_{n_duplicates} exact-duplicate chunk pair(s) excluded from the ranking below._\n"
            )

        for match in matches:
            sentence = format_explanation(match)
            logger.info("  [%.3f] %s", match.score, sentence)
            report_lines.append(f"- **[{match.score:.3f}]** {sentence}")

    logger.info(
        "Scanned %d resumes to find %d with a genuine (non-duplicate) top-ranked chunk match; "
        "%d had none.",
        n_scanned,
        found,
        n_no_genuine_match,
    )
    report_lines.insert(
        1,
        f"Scanned `{n_scanned}` resumes to find `{found}` with a genuine (non-duplicate) chunk "
        f"match against their top-ranked JD; `{n_no_genuine_match}` had none.\n",
    )
    if found < n_examples:
        report_lines.insert(
            2,
            f"_Only found {found}/{n_examples} requested examples after scanning the entire "
            "eval split._\n",
        )

    report_path = Path(cfg["paths"]["results_dir"]) / "phase4_explainability_demo.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info("Done. Report -> %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: demo chunk-level TF-IDF explainability.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to ml/config/config.yaml)")
    parser.add_argument("--n-examples", type=int, default=5, help="Number of resume/JD example pairs to demo")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top chunk-pair matches to show per example")
    args = parser.parse_args()
    main(args.config, args.n_examples, args.top_k)
