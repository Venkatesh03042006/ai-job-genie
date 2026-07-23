from src.explainability.chunk_similarity import (
    chunk_document,
    chunk_similarity_matrix,
    count_exact_duplicate_pairs,
    explain_match,
    format_explanation,
    top_k_chunk_matches,
)
from src.explainability.schema import ChunkMatch

__all__ = [
    "ChunkMatch",
    "chunk_document",
    "chunk_similarity_matrix",
    "count_exact_duplicate_pairs",
    "explain_match",
    "format_explanation",
    "top_k_chunk_matches",
]
