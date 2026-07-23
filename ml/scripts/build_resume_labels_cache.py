"""Phase 1b entry point: deduplicate resume_data.csv down to its ~344 unique
profiles and assign each a content-derived category label, replacing the
raw `job_position_name` cross-product stamp - see docs/PHASE3_FINDINGS.md's
"Category label mismatch" section for why the stamped label can't be
trusted. Writes the relabeled profiles to
data/processed/resumes_relabeled.parquet so build_dataset.py (and every
other script that needs resumes) can load them without re-running the
relabeling - which fits/encodes with every Phase 2 matcher (TF-IDF + all
configured SBERT checkpoints) - on every invocation.

Usage (run from the ml/ directory):
    python -m scripts.build_resume_labels_cache
"""
import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.data.loader import load_job_postings, load_resume_profiles
from src.data.relabel import assign_content_categories
from src.models.sbert_baseline import SbertMatcher
from src.models.tfidf_baseline import TfidfMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_resume_labels_cache")


def main(config_path: str = None) -> None:
    cfg = load_config(config_path)
    # Labeling always considers the FULL category universe, not
    # `categories.target` - the latter may be narrowed downstream (see
    # config.yaml's comment) to categories with enough resulting resumes,
    # but that narrowing must not feed back into which categories a resume
    # is allowed to be labeled as, or it'd bias the labeling itself.
    all_categories = cfg["categories"]["all"]

    logger.info("Loading and deduplicating resume profiles from %s", cfg["paths"]["resumes_dataset"])
    resumes = load_resume_profiles(cfg["paths"]["resumes_dataset"], cfg["columns"]["resumes"])
    logger.info("Deduplicated to %d unique resume profiles", len(resumes))

    logger.info("Loading job postings from %s", cfg["paths"]["postings_cache"])
    jobs = load_job_postings(cfg["paths"]["postings_cache"], all_categories)

    # Same matcher set/naming as scripts/run_baselines.py, so the
    # `<matcher>_category` columns line up with Phase 2's matcher names.
    matchers = [
        TfidfMatcher(),
        SbertMatcher(checkpoint=cfg["model"]["base_checkpoint"], name="sbert_pretrained"),
    ]
    for checkpoint in cfg["model"].get("additional_sbert_checkpoints", []):
        slug = checkpoint.split("/")[-1].lower().replace("-", "_")
        matchers.append(SbertMatcher(checkpoint=checkpoint, name=f"sbert_{slug}"))

    logger.info("Assigning content-derived categories via consensus of: %s", [m.name for m in matchers])
    relabeled = assign_content_categories(resumes, jobs, all_categories, matchers)

    logger.info("Content-derived category distribution:")
    for cat, count in relabeled["category"].value_counts().items():
        logger.info("  %-95s %d", cat, count)

    agreement_cols = [f"{m.name}_category" for m in matchers]
    unanimous = (relabeled[agreement_cols].nunique(axis=1) == 1).sum()
    logger.info("All %d matchers individually agreed on the same category for %d/%d profiles",
                len(matchers), unanimous, len(relabeled))

    out_path = Path(cfg["paths"]["resumes_relabeled_cache"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    relabeled.to_parquet(out_path, index=False)
    logger.info("Done. Relabeled resumes -> %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1b: deduplicate + content-relabel resume profiles.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to ml/config/config.yaml)")
    args = parser.parse_args()
    main(args.config)
