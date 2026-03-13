# src/ops/__init__.py
"""
LLMOps package — metrics, experiment tracking, evaluation, alerting, cost.
Import this package to activate all observability instrumentation.
"""
from .metrics import (
    PIPELINE_DURATION, REQUEST_COUNTER, ERROR_COUNTER,
    EMPTY_RETRIEVAL_COUNTER, LOW_SIMILARITY_COUNTER,
    ACTIVE_PIPELINES, TOKEN_USAGE_COUNTER, LLM_CONFIDENCE,
    record_stage, metrics_app,
)
from .experiment_tracker import ExperimentTracker
from .evaluator import RagasEvaluator
from .alerts import AlertManager
from .cost_tracker import CostTracker

__all__ = [
    "PIPELINE_DURATION", "REQUEST_COUNTER", "ERROR_COUNTER",
    "EMPTY_RETRIEVAL_COUNTER", "LOW_SIMILARITY_COUNTER",
    "ACTIVE_PIPELINES", "TOKEN_USAGE_COUNTER", "LLM_CONFIDENCE",
    "record_stage", "metrics_app",
    "ExperimentTracker",
    "RagasEvaluator",
    "AlertManager",
    "CostTracker",
]
