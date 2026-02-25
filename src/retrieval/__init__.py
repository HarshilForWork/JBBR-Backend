"""Stage 6 — Retrieval: FAISS ANN search + BGE reranking + adjacent context assembly."""
from .searcher import FAISSQueryProcessor

__all__ = ["FAISSQueryProcessor"]
