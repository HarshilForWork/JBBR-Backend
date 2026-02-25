"""
Stage 7 — Inference: schema.py
LLMAnswer dataclass — the structured output of every LLM evaluation call.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class LLMAnswer:
    """
    Structured output returned by the Groq evaluator.

    Fields
    ------
    answer          : Natural-language answer extracted from the LLM response.
    confidence      : Confidence score 0.0–1.0 (self-reported by the LLM; 0.5 if not returned).
    llm_available   : Whether a live Groq client was used.
    model_used      : Groq model identifier, or "none".
    search_method   : Always "faiss_vector_search" in this pipeline.
    reranking_method: BGE reranker model name or "none".
    evaluation_method: Evaluation strategy tag (e.g. "llm_with_quotes").
    json_parsed     : True if the LLM response was valid JSON.
    context_length  : Character length of the context fed to the LLM.
    num_sources     : Number of retrieved chunks used as context.
    llm_context     : Full context string that was sent to the LLM.
    source_vectors  : Raw retrieved chunk dicts (for downstream logging).
    error           : Error message if evaluation failed, else None.
    """
    answer:           str
    confidence:       float             = 0.0
    llm_available:    bool              = False
    model_used:       str               = "none"
    search_method:    str               = "faiss_vector_search"
    reranking_method: str               = "none"
    evaluation_method: str             = "llm_with_quotes"
    json_parsed:      bool              = False
    context_length:   int               = 0
    num_sources:      int               = 0
    llm_context:      str               = ""
    source_vectors:   List[Dict[str, Any]] = field(default_factory=list)
    error:            Optional[str]     = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for JSON serialisation."""
        return asdict(self)

    @classmethod
    def unavailable(
        cls,
        reason: str,
        reranking_method: str = "none",
        model_used: str = "none",
        evaluation_method: str = "llm_with_quotes",
    ) -> "LLMAnswer":
        """Factory: returned when the LLM client is not configured."""
        return cls(
            answer=f"LLM evaluation not available — {reason}",
            confidence=0.0,
            llm_available=False,
            model_used=model_used,
            reranking_method=reranking_method,
            evaluation_method=evaluation_method,
        )

    @classmethod
    def rate_limited(
        cls,
        error: str,
        model_used: str,
        reranking_method: str,
        llm_context: str = "",
        source_vectors: Optional[List] = None,
    ) -> "LLMAnswer":
        """Factory: returned when Groq returns a rate-limit / quota error."""
        return cls(
            answer="LLM rate limit reached — please retry shortly.",
            confidence=0.0,
            llm_available=True,
            model_used=model_used,
            reranking_method=reranking_method,
            llm_context=llm_context,
            source_vectors=source_vectors or [],
            error=error,
        )

    @classmethod
    def failed(
        cls,
        error: str,
        model_used: str,
        reranking_method: str,
        evaluation_method: str,
        llm_context: str = "",
        source_vectors: Optional[List] = None,
    ) -> "LLMAnswer":
        """Factory: returned after all retries are exhausted."""
        return cls(
            answer=f"LLM evaluation failed: {error}",
            confidence=0.0,
            llm_available=False,
            model_used=model_used,
            reranking_method=reranking_method,
            evaluation_method=evaluation_method,
            llm_context=llm_context,
            source_vectors=source_vectors or [],
            error=error,
        )
