"""Load the resume side (from data/raw/resume_data.csv, deduplicated and
content-relabeled - see src/data/relabel.py) and the job side (from the
cached, filtered data/processed/postings_filtered.parquet - see
scripts/build_postings_cache.py) into the schema pairing.py/triples.py
expect: two DataFrames (doc_id, category, text, source), independently
sourced - there is no row-to-row correspondence between them, so weak
(category-based) supervision is the only signal, by design.

Both sides are restricted to the 14 categories in config.yaml's
`categories.target` list, listed in src/data/category_keywords.py.

Note: this intentionally drops `responsibilities` (resume side) and
`matched_score` entirely - see docs/PHASE3_FINDINGS.md. `responsibilities`
is a 28-way category template duplicated verbatim into the job side, and
`matched_score` is a formulaic label from a synthetic 344-profile x
28-category cross-product generation process, not a real relevance signal.

The raw `job_position_name` column (the resume side's original "category")
is not used either, for the same underlying reason - see
docs/PHASE3_FINDINGS.md's "Category label mismatch" section:
`load_resume_profiles` deduplicates the cross-product down to the 344
unique profiles, and `scripts/build_resume_labels_cache.py`
(src/data/relabel.py) assigns each one a content-derived category instead
of trusting the stamped label.
"""
import ast
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from src.utils.text_cleaning import normalize_whitespace

_BOM = "﻿"


def _read_csv(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")
    df = pd.read_csv(path, encoding="utf-8-sig")  # strips a leading BOM, if any

    # This dataset carries a stray BOM character embedded mid-file in one
    # header (`﻿job_position_name`), not just at the very start of the
    # file, so strip it from every column name rather than relying on the
    # csv encoding alone.
    df.columns = [str(c).replace(_BOM, "").strip() for c in df.columns]
    return df


def _stringify_field(value) -> str:
    """Fields like `skills`/`positions` are stored as stringified Python
    lists (e.g. "['Python', 'SQL']") - unwrap them into plain text."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            items = ast.literal_eval(text)
            if isinstance(items, (list, tuple)):
                return ", ".join(str(i) for i in items if str(i).strip())
        except (ValueError, SyntaxError):
            pass
    return text


def _combine_fields(row: pd.Series, fields: List[str]) -> str:
    parts = [_stringify_field(row[f]) for f in fields if f in row.index]
    parts = [p for p in parts if p]
    return normalize_whitespace("\n".join(parts))


def load_resume_profiles(path: Union[str, Path], column_cfg: Dict) -> pd.DataFrame:
    """Load resume_data.csv into (doc_id, text, source="resume"), built only
    from `column_cfg["resume_fields"]` (career_objective/skills/positions -
    `responsibilities` is deliberately not in that list, see module
    docstring), and deduplicated down to unique resume profiles.

    resume_data.csv is a synthetic cross-product: the same
    (career_objective, skills, positions) profile is repeated once per
    category (28x in the raw file) with only `job_position_name` (and the
    other per-category-template fields already excluded above) varying -
    see docs/PHASE3_FINDINGS.md. Deduplicating on the constructed `text`
    collapses that back down to the ~344 real, distinct profiles. There is
    no `category` column here on purpose: the raw `job_position_name` stamp
    is exactly what's being discarded (it's the cross-product artifact, not
    a real per-profile label) - see src/data/relabel.py for how a real one
    gets assigned.
    """
    df = _read_csv(path)

    resume_fields = column_cfg["resume_fields"]
    missing = [c for c in resume_fields if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in resumes dataset. "
            f"Available columns: {list(df.columns)}. Fix `columns.resumes` in config.yaml."
        )

    text = df.apply(lambda r: _combine_fields(r, resume_fields), axis=1)

    profiles = pd.DataFrame({"text": text, "source": "resume"})
    profiles = profiles[profiles["text"].str.len() > 0]
    profiles = profiles.drop_duplicates(subset="text").reset_index(drop=True)
    profiles.insert(0, "doc_id", [f"resume-{i}" for i in profiles.index])
    return profiles


def load_relabeled_resumes(cache_path: Union[str, Path], target_categories: List[str]) -> pd.DataFrame:
    """Load the deduplicated, content-relabeled resume profiles cached by
    scripts/build_resume_labels_cache.py (doc_id, category, text, source,
    plus per-matcher category columns kept for inspection) - see
    src/data/relabel.py / docs/PHASE3_FINDINGS.md.

    The cache holds every profile labeled against the full category
    universe (`categories.all`); `target_categories` (typically
    `categories.target`, a narrower, statistically-viable subset - see
    config.yaml's comment) filters it down to the categories actually used
    downstream, mirroring `load_job_postings`'s filtering.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Relabeled resumes cache not found at {cache_path}. Run "
            f"`python -m scripts.build_resume_labels_cache` first."
        )
    resumes = pd.read_parquet(cache_path)
    resumes = resumes[resumes["category"].isin(target_categories)]
    return resumes.reset_index(drop=True)


def load_job_postings(cache_path: Union[str, Path], target_categories: List[str]) -> pd.DataFrame:
    """Load the cached, filtered postings parquet (built by
    scripts/build_postings_cache.py) into (doc_id, category, text,
    source="job_description"). `text` is title + description."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Postings cache not found at {cache_path}. Run "
            f"`python -m scripts.build_postings_cache` first."
        )
    df = pd.read_parquet(cache_path)

    text = (df["title"].fillna("") + "\n" + df["description"].fillna("")).map(normalize_whitespace)
    jobs = pd.DataFrame({
        "doc_id": "job-" + df["doc_id"].astype(str),
        "category": df["category"],
        "text": text,
        "source": "job_description",
    })

    jobs = jobs[jobs["category"].isin(target_categories) & (jobs["text"].str.len() > 0)]
    return jobs.reset_index(drop=True)
