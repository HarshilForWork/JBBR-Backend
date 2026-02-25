"""Stage 4 — Embedding: dense vector generation via Pinecone multilingual-e5-large."""
from .embedder import (
    generate_query_embedding_pinecone,
    index_chunks_in_pinecone,
)

__all__ = [
    "generate_query_embedding_pinecone",
    "index_chunks_in_pinecone",
]
