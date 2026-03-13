"""
src/ops/evaluator.py
====================
RAGAS-style RAG evaluation.  Runs in a background thread after the response
is sent so it adds ZERO latency to the request path.

Metrics computed (no extra LLM call required):
  context_relevance    — cosine similarity between query vector and mean context vector
  faithfulness         — keyword overlap between answer and context (heuristic)
  retrieval_effectiveness — fraction of non-empty top-k slots
  answer_confidence    — LLM-reported confidence (already collected)
  top_similarity_score — top reranker/cosine score from retrieval

Results are written to logs/eval_<timestamp>.jsonl and also returned
as a dict so the experiment tracker can log them to MLflow.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional


# ── Config ────────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

_CFG = _load_cfg()
_EVAL_CFG = _CFG.get("evaluation", {})
_EVAL_LOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "logs"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / norm if norm > 0 else 0.0


def _mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    result = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            result[i] += x
    return [x / len(vectors) for x in result]


def _keyword_overlap(answer: str, context: str) -> float:
    """
    Faithfulness proxy: fraction of content words in `answer` that appear
    in `context`.  Simple stopword filter; no LLM call needed.
    """
    STOP = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "in", "on", "at",
        "by", "for", "with", "of", "and", "or", "but", "not", "this", "that",
        "it", "its", "from", "as", "if", "so", "than", "then",
    }
    tokenize = lambda text: {
        w.lower() for w in re.findall(r"\b[a-z]{3,}\b", text.lower())
    } - STOP

    ans_words = tokenize(answer)
    ctx_words = tokenize(context)

    if not ans_words:
        return 0.0
    overlap = ans_words & ctx_words
    return len(overlap) / len(ans_words)


# ── Main evaluator ─────────────────────────────────────────────────────────────

class RagasEvaluator:
    """
    Computes lightweight RAGAS-style metrics for a single query result.

    Call `evaluate_async(...)` to run in the background (non-blocking).
    Call `evaluate(...)` directly if you need the result synchronously.
    """

    def __init__(self) -> None:
        os.makedirs(_EVAL_LOG_DIR, exist_ok=True)
        self._log_path = os.path.join(_EVAL_LOG_DIR, "eval_results.jsonl")
        self._faithfulness_threshold   = float(
            _EVAL_CFG.get("faithfulness_threshold", 0.40)
        )
        self._context_rel_threshold    = float(
            _EVAL_CFG.get("context_relevance_threshold", 0.50)
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate_async(
        self,
        *,
        query: str,
        query_embedding: Optional[List[float]],
        answer: str,
        context: str,
        source_vectors: List[Dict[str, Any]],
        confidence: float,
        session_id: str,
        pdf_filename: str,
        on_complete: Optional[Any] = None,   # callback(metrics: dict)
    ) -> None:
        """
        Fire-and-forget: runs evaluate() in a daemon thread.
        `on_complete` is called with the metrics dict when done.
        """
        if not _EVAL_CFG.get("enabled", True):
            return

        def _bg():
            try:
                metrics = self.evaluate(
                    query=query,
                    query_embedding=query_embedding,
                    answer=answer,
                    context=context,
                    source_vectors=source_vectors,
                    confidence=confidence,
                    session_id=session_id,
                    pdf_filename=pdf_filename,
                )
                if on_complete:
                    on_complete(metrics)
            except Exception as exc:
                print(f"⚠️  [RagasEvaluator] background eval failed: {exc}")

        t = threading.Thread(target=_bg, daemon=True)
        t.start()

    def evaluate(
        self,
        *,
        query: str,
        query_embedding: Optional[List[float]],
        answer: str,
        context: str,
        source_vectors: List[Dict[str, Any]],
        confidence: float,
        session_id: str,
        pdf_filename: str,
    ) -> Dict[str, float]:
        """
        Compute RAGAS-style metrics synchronously and append result to log.

        Returns
        -------
        dict with keys:
          context_relevance, faithfulness, retrieval_effectiveness,
          answer_confidence, top_similarity_score
        """
        t0 = time.time()

        # 1. Context relevance — cosine(query_vec, mean_chunk_vecs)
        chunk_embeddings = [
            sv["embedding"] for sv in source_vectors if sv.get("embedding")
        ]
        if query_embedding and chunk_embeddings:
            context_relevance = float(
                _cosine(query_embedding, _mean_vector(chunk_embeddings))
            )
        else:
            # Fallback: text-level overlap (lower quality but still useful)
            context_relevance = _keyword_overlap(query, context)

        # 2. Faithfulness — keyword overlap answer ↔ context
        faithfulness = _keyword_overlap(answer, context)

        # 3. Retrieval effectiveness — fraction of top-k that are non-empty
        non_empty = sum(1 for sv in source_vectors if sv.get("text", "").strip())
        retrieval_effectiveness = non_empty / max(len(source_vectors), 1)

        # 4. Top similarity score
        scores = [
            float(sv.get("score", 0.0)) for sv in source_vectors
            if sv.get("score") is not None
        ]
        top_similarity_score = max(scores) if scores else 0.0

        metrics = {
            "context_relevance":       round(context_relevance, 4),
            "faithfulness":            round(faithfulness, 4),
            "retrieval_effectiveness": round(retrieval_effectiveness, 4),
            "answer_confidence":       round(float(confidence), 4),
            "top_similarity_score":    round(top_similarity_score, 4),
        }

        # ── Persist to JSONL ───────────────────────────────────────────────────
        record = {
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id":  session_id,
            "pdf_filename": pdf_filename,
            "query":       query,
            "answer":      answer[:200],
            "eval_duration_ms": round((time.time() - t0) * 1000, 1),
            **metrics,
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            print(f"⚠️  [RagasEvaluator] log write failed: {exc}")

        return metrics
