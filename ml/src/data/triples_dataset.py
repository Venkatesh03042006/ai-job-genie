"""Load data/processed/triples.jsonl (Phase 1 output) filtered by split, for
Phase 3 fine-tuning (train) and validation (val) - reused instead of
re-deriving triples so training consumes exactly the same examples Phase 1
built and split.
"""
import json
from pathlib import Path
from typing import Dict, List, Union


def load_triples(path: Union[str, Path], split: str) -> List[Dict[str, str]]:
    """Return the raw triple records (see src/data/schema.py's Triple) whose
    `split` field matches, in file order."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["split"] == split:
                records.append(record)
    return records


def triples_to_columns(records: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Reshape triple records into the (anchor, positive, negative) columns
    MultipleNegativesRankingLoss / TripletEvaluator expect."""
    return {
        "anchor": [r["resume_chunk"] for r in records],
        "positive": [r["positive_jd_chunk"] for r in records],
        "negative": [r["negative_jd_chunk"] for r in records],
    }
