"""Typed record for one explained resume-chunk / JD-chunk match, returned by
src/explainability/chunk_similarity.py - Phase 5's backend can serialize
this directly when returning ranked matches to the frontend.
"""
from dataclasses import dataclass


@dataclass
class ChunkMatch:
    resume_chunk: str
    jd_chunk: str
    score: float
    resume_chunk_idx: int
    jd_chunk_idx: int
