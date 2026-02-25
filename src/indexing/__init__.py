"""Stage 5 — Indexing: FAISS vector store + document registry."""
from .store import FAISSVectorStore, check_or_create_faiss_index
from .registry import DocumentRegistry

__all__ = [
    "FAISSVectorStore",
    "check_or_create_faiss_index",
    "DocumentRegistry",
]
