"""
src/ops/alerts.py
=================
Metric-driven alerting for the RAG pipeline.

All thresholds come from config.yaml → alerting:
  max_latency_seconds     (default 30)
  max_errors_per_minute   (default 5)
  ragas_drift_threshold   (default 0.20)
  alert_log               (default logs/alerts.jsonl)

Quality alerts (LOW_SIMILARITY, FAITHFULNESS_DEGRADATION, RAGAS_METRIC_DRIFT)
always include the query text and PDF filename for exact reproducibility.

Alert events are written to a structured JSONL log AND printed as console
warnings.  Extend `_dispatch()` to add Slack/PagerDuty/webhook support.
"""
from __future__ import annotations

import collections
import json
import os
import time
import threading
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

_CFG       = _load_cfg()
_ALERT_CFG = _CFG.get("alerting", {})
_GUARD_CFG = _CFG.get("guardrails", {})

# ── Alert types ────────────────────────────────────────────────────────────────
HIGH_LATENCY              = "HIGH_LATENCY"
EMPTY_RETRIEVAL           = "EMPTY_RETRIEVAL"
LOW_SIMILARITY            = "LOW_SIMILARITY"
LOW_CONFIDENCE            = "LOW_CONFIDENCE"
LLM_FAILURE               = "LLM_FAILURE"
FAITHFULNESS_DEGRADATION  = "FAITHFULNESS_DEGRADATION"
RAGAS_METRIC_DRIFT        = "RAGAS_METRIC_DRIFT"
HIGH_ERROR_RATE           = "HIGH_ERROR_RATE"
EXPENSIVE_QUERY           = "EXPENSIVE_QUERY"


class AlertManager:
    """
    Collects system signals and fires structured alerts when thresholds
    are crossed.  All alert evaluation is metric-driven (not log-driven).
    """

    def __init__(self) -> None:
        enabled = _ALERT_CFG.get("enabled", True)
        self._enabled      = enabled
        self._max_latency  = float(_ALERT_CFG.get("max_latency_seconds", 30))
        self._max_epm      = float(_ALERT_CFG.get("max_errors_per_minute", 5))
        self._drift_thresh = float(_ALERT_CFG.get("ragas_drift_threshold", 0.20))
        self._min_sim      = float(_GUARD_CFG.get("min_similarity_score", 0.30))
        self._min_conf     = float(_GUARD_CFG.get("min_confidence", 0.50))
        self._faith_thresh = float(
            _CFG.get("evaluation", {}).get("faithfulness_threshold", 0.40)
        )

        log_dir  = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        alert_file = _ALERT_CFG.get("alert_log", "logs/alerts.jsonl")
        self._log_path = os.path.join(
            os.path.dirname(__file__), "..", "..", alert_file
        )

        # Rolling error window (timestamps of recent errors)
        self._error_times: collections.deque = collections.deque()
        self._lock = threading.Lock()

        # Rolling RAGAS metric history for drift detection
        # { metric_name: deque of recent values (last 20 runs) }
        self._ragas_history: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=20)
        )

        if enabled:
            print(f"✅ [AlertManager] enabled  |  log → {self._log_path}")

    # ── Public check helpers ───────────────────────────────────────────────────

    def check_latency(
        self,
        total_seconds: float,
        stage_timings: Dict[str, float],
        session_id: str,
    ) -> None:
        if total_seconds > self._max_latency:
            self._dispatch(HIGH_LATENCY, {
                "total_seconds": round(total_seconds, 2),
                "stage_timings": {k: round(v, 2) for k, v in stage_timings.items()},
                "session_id":    session_id,
                "threshold":     self._max_latency,
            })

    def check_retrieval(
        self,
        source_vectors: List[Dict[str, Any]],
        query: str,
        pdf_filename: str,
        session_id: str,
    ) -> None:
        """Check for empty retrieval and low similarity."""
        if not source_vectors:
            self._dispatch(EMPTY_RETRIEVAL, {
                "query":        query,
                "pdf_filename": pdf_filename,
                "session_id":   session_id,
            })
            return

        scores = [float(sv.get("score", 0.0)) for sv in source_vectors if sv.get("score") is not None]
        if scores:
            top = max(scores)
            if top < self._min_sim:
                self._dispatch(LOW_SIMILARITY, {
                    "query":        query,
                    "pdf_filename": pdf_filename,
                    "session_id":   session_id,
                    "top_score":    round(top, 4),
                    "threshold":    self._min_sim,
                })

    def check_confidence(
        self,
        confidence: float,
        query: str,
        pdf_filename: str,
        answer: str,
        session_id: str,
    ) -> None:
        if confidence < self._min_conf:
            self._dispatch(LOW_CONFIDENCE, {
                "query":        query,
                "pdf_filename": pdf_filename,
                "session_id":   session_id,
                "confidence":   round(confidence, 4),
                "threshold":    self._min_conf,
                "answer_preview": answer[:120],
            })

    def check_llm_failure(
        self,
        error: str,
        query: str,
        session_id: str,
        model: str,
    ) -> None:
        self._track_error()
        self._dispatch(LLM_FAILURE, {
            "query":      query,
            "session_id": session_id,
            "model":      model,
            "error":      error[:300],
        })
        self._check_error_rate(session_id)

    def check_eval_metrics(
        self,
        metrics: Dict[str, float],
        query: str,
        pdf_filename: str,
        session_id: str,
    ) -> None:
        """
        Check faithfulness degradation and RAGAS metric drift.
        Called after background evaluation completes.
        """
        # Faithfulness degradation
        faith = metrics.get("faithfulness", 1.0)
        if faith < self._faith_thresh:
            self._dispatch(FAITHFULNESS_DEGRADATION, {
                "query":        query,             # for reproducibility
                "pdf_filename": pdf_filename,      # for reproducibility
                "session_id":   session_id,
                "faithfulness": round(faith, 4),
                "threshold":    self._faith_thresh,
            })

        # RAGAS metric drift — compare current vs rolling avg
        for metric_name, value in metrics.items():
            history = self._ragas_history[metric_name]
            if len(history) >= 5:
                rolling_avg = sum(history) / len(history)
                if rolling_avg > 0:
                    delta = (rolling_avg - value) / rolling_avg  # drop fraction
                    if delta > self._drift_thresh:
                        self._dispatch(RAGAS_METRIC_DRIFT, {
                            "metric":      metric_name,
                            "current":     round(value, 4),
                            "rolling_avg": round(rolling_avg, 4),
                            "drop_pct":    round(delta * 100, 1),
                            "threshold_pct": round(self._drift_thresh * 100, 1),
                            "query":        query,         # for reproducibility
                            "pdf_filename": pdf_filename,  # for reproducibility
                            "session_id":   session_id,
                        })
            history.append(value)

    def check_expensive_query(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        limit: int,
        session_id: str,
        query: str,
    ) -> None:
        total = prompt_tokens + completion_tokens
        if total > limit:
            self._dispatch(EXPENSIVE_QUERY, {
                "session_id":        session_id,
                "query_preview":     query[:80],
                "total_tokens":      total,
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "limit":             limit,
            })

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _track_error(self) -> None:
        """Record an error timestamp for rate tracking."""
        now = time.time()
        with self._lock:
            self._error_times.append(now)
            # Prune events older than 60s
            while self._error_times and self._error_times[0] < now - 60:
                self._error_times.popleft()

    def _check_error_rate(self, session_id: str) -> None:
        with self._lock:
            count = len(self._error_times)
        if count > self._max_epm:
            self._dispatch(HIGH_ERROR_RATE, {
                "errors_last_60s": count,
                "threshold":       self._max_epm,
                "session_id":      session_id,
            })

    def _dispatch(self, alert_type: str, payload: Dict[str, Any]) -> None:
        """Write a structured alert to JSONL log and print warning."""
        if not self._enabled:
            return

        event = {
            "alert":     alert_type,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **payload,
        }
        print(f"🚨 [ALERT:{alert_type}] {json.dumps(payload, ensure_ascii=False)}")

        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"⚠️  [AlertManager] log write failed: {exc}")

        # ── Extend here: send to Slack / webhook ──────────────────────────────
        # self._send_slack(event)
        # self._send_webhook(event)
