"""The only module that imports ml/src/* - a thin orchestration layer around
existing matching/explainability logic, not a reimplementation of it. Mirrors
exactly what ml/scripts/demo_explainability.py already does per-resume in
batch, just invoked per-request instead (see docs/PHASE3_FINDINGS.md section
7.8 for why TF-IDF, not SBERT, is the matcher here).

Job postings are loaded once per process (module-level singleton), against
`categories.target` - the narrowed, statistically-viable 5-category /
2,174-JD pool (section 7.4), not the full 14-category `categories.all`.
"""
import sys
from typing import Dict, List

import numpy as np
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if str(settings.ML_ROOT) not in sys.path:
    sys.path.insert(0, str(settings.ML_ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import load_job_postings  # noqa: E402
from src.explainability import explain_match  # noqa: E402
from src.models.tfidf_baseline import TfidfMatcher  # noqa: E402

_config = None
_jobs = None
_keyword_analyzer = None


def _get_config():
    global _config
    if _config is None:
        _config = load_config(str(settings.ML_ROOT / "config" / "config.yaml"))
    return _config


def _get_jobs():
    global _jobs
    if _jobs is None:
        cfg = _get_config()
        _jobs = load_job_postings(cfg["paths"]["postings_cache"], cfg["categories"]["target"])
    return _jobs


def _get_keyword_analyzer():
    """Unigram-only tokenizer/stopword-filter (same TfidfVectorizer machinery
    as TfidfMatcher, just ngram_range=(1, 1)) - deliberately not the fitted
    bigram vectorizer used for matching, so ATS keyword lists show single
    terms ("python") instead of stopword-adjacency artifacts like "experience
    building". build_analyzer() needs no fit() call - it's a pure
    tokenization pipeline, not a fitted vocabulary."""
    global _keyword_analyzer
    if _keyword_analyzer is None:
        _keyword_analyzer = TfidfVectorizer(stop_words="english").build_analyzer()
    return _keyword_analyzer


def compute_ats_analysis(resume_text: str, jd_text: str) -> Dict:
    """Keyword-overlap percentage between `resume_text` and `jd_text` (rounded to 1
    decimal place) plus the JD keywords missing from the resume, sorted
    alphabetically for a stable, scannable list."""
    analyze = _get_keyword_analyzer()
    resume_tokens = set(analyze(resume_text))
    jd_tokens = set(analyze(jd_text))

    if not jd_tokens:
        return {"ats_score": 0.0, "skill_gap": []}

    overlap = resume_tokens & jd_tokens
    return {
        "ats_score": round(len(overlap) / len(jd_tokens) * 100, 1),
        "skill_gap": sorted(jd_tokens - resume_tokens),
    }


def get_matches(resume_text: str, top_n: int = 10, explain_top_k: int = 3) -> Dict:
    """Rank `resume_text` against the full target-category job pool with
    TF-IDF (same pattern as src/evaluation/evaluate.py's evaluate_matcher:
    fit fresh per call on resume + candidate pool text), and attach
    chunk-level explanations to each of the top `top_n` matches.

    Returns {"matches": [...], "ats_score": float, "skill_gap": [str]} -
    the ATS fields are keyword-overlap between the resume and only the
    rank-1 match's JD text (see compute_ats_analysis()).
    """
    cfg = _get_config()
    jobs = _get_jobs()
    job_texts = jobs["text"].tolist()

    matcher = TfidfMatcher()
    matcher.fit([resume_text], job_texts)
    resume_vec = matcher.encode([resume_text])
    job_vecs = matcher.encode(job_texts)
    sims = cosine_similarity(resume_vec, job_vecs)[0]

    ranked_idx = np.argsort(-sims)[:top_n]

    results = []
    for rank, idx in enumerate(ranked_idx, start=1):
        job_row = jobs.iloc[idx]
        jd_text = job_row["text"]
        # Display-only split of the title+"\n"+description text
        # load_job_postings already builds - not new matching logic.
        title = jd_text.split("\n", 1)[0]

        matches = explain_match(
            resume_text,
            jd_text,
            matcher.vectorizer,
            cfg["chunking"],
            top_k=explain_top_k,
            exclude_exact_duplicates=True,
        )

        results.append({
            "rank": rank,
            "job_doc_id": job_row["doc_id"],
            "category": job_row["category"],
            "title": title,
            "score": float(sims[idx]),
            "explanations": [
                {
                    "resume_chunk": m.resume_chunk,
                    "jd_chunk": m.jd_chunk,
                    "score": m.score,
                }
                for m in matches
            ],
        })

    ats_analysis = (
        compute_ats_analysis(resume_text, jobs.iloc[ranked_idx[0]]["text"])
        if len(ranked_idx) > 0
        else {"ats_score": 0.0, "skill_gap": []}
    )

    return {
        "matches": results,
        "ats_score": ats_analysis["ats_score"],
        "skill_gap": ats_analysis["skill_gap"],
    }
