"""Typed records produced by the pairing and triple-building steps.

Kept as plain dataclasses (not pydantic/ORM models) since Phase 1 only needs
to serialize these to JSONL - Phase 2/3 can still import them for type hints.
"""
from dataclasses import dataclass


@dataclass
class Pair:
    resume_id: str
    jd_id: str
    category: str
    label: int  # 1 = positive (same category), 0 = negative (cross-category)
    split: str  # "train" / "val" / "test" - stratified by category, see split.py


@dataclass
class Triple:
    resume_chunk: str
    positive_jd_chunk: str
    negative_jd_chunk: str
    category: str
    resume_id: str
    positive_jd_id: str
    negative_jd_id: str
    split: str  # "train" / "val" / "test" - stratified by category, see split.py
