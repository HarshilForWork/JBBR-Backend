"""
src/ops/metrics.py
==================
Prometheus metrics for the RAG pipeline.

Usage:
    from src.ops.metrics import record_stage, REQUEST_COUNTER, ...

Endpoint:
    Mount metrics_app at /metrics in final_backend.py
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from prometheus_client import (
    Counter, Gauge, Histogram,
    make_asgi_app, REGISTRY,
)

# ── Histograms ─────────────────────────────────────────────────────────────────
PIPELINE_DURATION = Histogram(
    "rag_pipeline_duration_seconds",
    "Duration of each pipeline stage in seconds",
    labelnames=["stage"],        # pdf | chunk | embed | index | retrieve | rerank | llm
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 30, 60],
)

LLM_CONFIDENCE = Histogram(
    "rag_llm_confidence",
    "LLM-reported confidence score distribution",
    labelnames=["model"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Counters ───────────────────────────────────────────────────────────────────
REQUEST_COUNTER = Counter(
    "rag_requests_total",
    "Total HTTP requests processed",
    labelnames=["endpoint", "status"],   # endpoint: upload|url, status: success|error
)

ERROR_COUNTER = Counter(
    "rag_errors_total",
    "Total pipeline errors by stage and type",
    labelnames=["stage", "error_type"],
)

EMPTY_RETRIEVAL_COUNTER = Counter(
    "rag_empty_retrieval_total",
    "Number of queries that returned zero chunks from FAISS",
)

LOW_SIMILARITY_COUNTER = Counter(
    "rag_low_similarity_total",
    "Number of queries where top similarity score was below threshold",
)

TOKEN_USAGE_COUNTER = Counter(
    "rag_token_usage_total",
    "Cumulative token usage",
    labelnames=["model", "token_type"],  # token_type: prompt | completion | embedding
)

# ── Gauges ─────────────────────────────────────────────────────────────────────
ACTIVE_PIPELINES = Gauge(
    "rag_active_pipelines",
    "Number of pipeline workers currently running",
)

# ── ASGI app for /metrics endpoint ────────────────────────────────────────────
metrics_app = make_asgi_app()


# ── Convenience helpers ────────────────────────────────────────────────────────

@contextmanager
def record_stage(stage: str) -> Generator[None, None, None]:
    """
    Context manager that records the wall-clock duration of a pipeline stage.

    Usage:
        with record_stage("pdf"):
            parse_pdf(...)
    """
    with PIPELINE_DURATION.labels(stage=stage).time():
        yield


def inc_tokens(model: str, prompt: int, completion: int = 0) -> None:
    """Increment token usage counters."""
    if prompt:
        TOKEN_USAGE_COUNTER.labels(model=model, token_type="prompt").inc(prompt)
    if completion:
        TOKEN_USAGE_COUNTER.labels(model=model, token_type="completion").inc(completion)


def inc_embedding_tokens(model: str, count: int) -> None:
    TOKEN_USAGE_COUNTER.labels(model=model, token_type="embedding").inc(count)
