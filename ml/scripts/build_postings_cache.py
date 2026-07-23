"""One-off (or on-demand) heavy step: stream the ~517MB raw LinkedIn Job
Postings CSV, project it down to the columns the pipeline needs, assign
each posting to one of the 14 target resume categories by title keyword
match (src/data/category_keywords.py), drop everything else, and cache the
result as parquet.

This is deliberately kept separate from build_dataset.py: build_dataset.py
(and any future Phase 2/3 re-run) should read the small cached parquet file,
never re-parse the raw CSV.

Usage (run from the ml/ directory):
    python -m scripts.build_postings_cache
    python -m scripts.build_postings_cache --force
"""
import argparse
import collections
import logging
import time
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.category_keywords import assign_category, count_ambiguous_matches
from src.utils.text_cleaning import normalize_whitespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_postings_cache")

CHUNK_SIZE = 50_000


def build_postings_cache(raw_path: Path, cache_path: Path, force: bool = False) -> pd.DataFrame:
    if cache_path.exists() and not force:
        logger.info("Cache already exists at %s (use --force to rebuild). Loading it.", cache_path)
        return pd.read_parquet(cache_path)

    cols = ["job_id", "title", "description", "formatted_experience_level", "location"]
    kept_chunks = []
    total_rows = 0
    ambiguous_titles = 0
    t0 = time.time()

    for chunk in pd.read_csv(raw_path, usecols=cols, chunksize=CHUNK_SIZE, dtype=str):
        total_rows += len(chunk)

        titles = chunk["title"].fillna("")
        chunk = chunk.assign(category=titles.map(assign_category))
        chunk = chunk[chunk["category"].notna()]

        desc = chunk["description"].fillna("").str.strip()
        chunk = chunk[desc.str.len() > 0]

        if len(chunk):
            ambiguous_titles += chunk["title"].map(count_ambiguous_matches).gt(1).sum()
            kept_chunks.append(chunk)

    filtered = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame(columns=cols + ["category"])
    filtered["title"] = filtered["title"].map(normalize_whitespace)
    filtered["description"] = filtered["description"].map(normalize_whitespace)
    filtered = filtered.rename(columns={"job_id": "doc_id"})
    filtered["doc_id"] = filtered["doc_id"].astype(str)

    # A posting can only match >1 category pattern when its title contains
    # more than one keyword (e.g. "Marketing Coordinator, Business
    # Development") - assign_category()'s first-match priority order always
    # picks one, so this is a diagnostic count, not a source of duplicates.
    elapsed = time.time() - t0
    logger.info(
        "Scanned %d raw postings in %.1fs -> kept %d (%.1f%%) with a target-category title match "
        "and non-empty description (%d titles matched >1 category pattern, resolved by priority order)",
        total_rows,
        elapsed,
        len(filtered),
        100 * len(filtered) / total_rows if total_rows else 0,
        ambiguous_titles,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(cache_path, index=False)
    logger.info("Wrote %d rows to %s", len(filtered), cache_path)
    return filtered


def main(config_path: str = None, force: bool = False) -> None:
    cfg = load_config(config_path)
    raw_path = Path(cfg["paths"]["postings_dataset"])
    cache_path = Path(cfg["paths"]["postings_cache"])

    filtered = build_postings_cache(raw_path, cache_path, force=force)

    counts = collections.Counter(filtered["category"])
    logger.info("Category coverage in cached postings:")
    for cat in cfg["categories"]["all"]:
        logger.info("  %6d  %s", counts.get(cat, 0), cat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter+cache postings.csv to the target categories.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to ml/config/config.yaml)")
    parser.add_argument("--force", action="store_true", help="Rebuild the cache even if it already exists")
    args = parser.parse_args()
    main(args.config, force=args.force)
