"""Stage 3 — Feature Engineering: chunk parsed text into retrieval-ready segments."""
from .chunker import chunk_documents_optimized

__all__ = ["chunk_documents_optimized"]
