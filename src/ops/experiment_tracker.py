"""
src/ops/experiment_tracker.py
==============================
MLflow experiment tracking via DagsHub.

Tracks one MLflow run per HTTP request:
  - params : model config, prompt version, retrieval config
  - metrics: stage latencies, token counts, RAGAS scores, confidence
  - tags   : session_id, user_id, endpoint

DagsHub config (add to .env):
    DAGSHUB_TOKEN=<token>
    DAGSHUB_REPO_OWNER=<github-username>
    DAGSHUB_REPO_NAME=<dagshub-repo-name>
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from typing import Any, Dict, Optional

# ── Optional MLflow / DagsHub ─────────────────────────────────────────────────
try:
    import mlflow
    import dagshub
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    mlflow = None         # type: ignore
    dagshub = None        # type: ignore


def _prompt_version(template: str) -> str:
    """Auto-version a prompt template by hashing its content (sha256[:8])."""
    return hashlib.sha256(template.encode()).hexdigest()[:8]


class ExperimentTracker:
    """
    Thin wrapper around MLflow that logs one run per request to DagsHub.

    If DagsHub credentials are missing or mlflow is not installed,
    all methods become no-ops (fail silently).
    """

    def __init__(self, experiment_name: str = "hackrx-rag") -> None:
        self._enabled = False
        self._experiment_name = experiment_name

        if not _MLFLOW_AVAILABLE:
            print("⚠️  [ExperimentTracker] mlflow/dagshub not installed — tracking disabled.")
            return

        token      = os.getenv("DAGSHUB_TOKEN")
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
        repo_name  = os.getenv("DAGSHUB_REPO_NAME")

        if not all([token, repo_owner, repo_name]):
            print("⚠️  [ExperimentTracker] DAGSHUB_TOKEN / DAGSHUB_REPO_OWNER / "
                  "DAGSHUB_REPO_NAME not set — tracking disabled.")
            return

        try:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            mlflow.set_experiment(experiment_name)
            self._enabled = True
            print(f"✅ [ExperimentTracker] MLflow → DagsHub ({repo_owner}/{repo_name}) "
                  f"| experiment: {experiment_name}")
        except Exception as exc:
            print(f"⚠️  [ExperimentTracker] DagsHub init failed: {exc} — tracking disabled.")

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Recursively flatten a nested dict into dot-separated keys."""
        items: Dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{prefix}{k}"
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, f"{new_key}."))
            else:
                items[new_key] = v
        return items

    # ── Public API ─────────────────────────────────────────────────────────────

    def log_run(
        self,
        *,
        session_id: str,
        user_id: str,
        endpoint: str,
        pipeline_params: Dict[str, Any],
        stage_timings: Dict[str, float],
        token_usage: Dict[str, int],
        eval_metrics: Dict[str, float],
        prompt_template: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        """
        Log a complete pipeline run to MLflow.

        Parameters
        ----------
        pipeline_params : full config.yaml or specific model/retrieval params
        stage_timings   : {"pdf": 8.2, "embed": 0.5, "llm": 2.6, ...}
        token_usage     : {"prompt_tokens": 1200, "completion_tokens": 300}
        eval_metrics    : {"confidence": 0.9, "faithfulness": 0.75, ...}
        prompt_template : Raw prompt string — auto-versioned by sha256[:8] hash.
        """
        if not self._enabled:
            return

        try:
            run_name = run_name or f"{session_id}_{int(time.time())}"

            with mlflow.start_run(run_name=run_name):
                # ── Tags ──────────────────────────────────────────────────────
                mlflow.set_tags({
                    "session_id": session_id,
                    "user_id":    user_id,
                    "endpoint":   endpoint,
                })

                # ── Params (Flattened) ────────────────────────────────────────
                params = self._flatten_dict(pipeline_params)
                if prompt_template:
                    params["prompt_version"] = _prompt_version(prompt_template)
                mlflow.log_params(params)

                # ── Stage timing metrics ─────────────────────────────────────
                for stage, dur in stage_timings.items():
                    if isinstance(dur, (int, float)):
                        mlflow.log_metric(f"latency_{stage}_s", round(float(dur), 3))

                # ── Token usage metrics ───────────────────────────────────────
                for tok_key, tok_val in token_usage.items():
                    if isinstance(tok_val, (int, float)):
                        mlflow.log_metric(tok_key, float(tok_val))

                # ── Evaluation / RAGAS metrics ────────────────────────────────
                if eval_metrics:
                    print(f"📊 [ExperimentTracker] Attempting to log {len(eval_metrics)} eval metrics...")
                for metric_key, metric_val in eval_metrics.items():
                    # More robust check for numeric-like types
                    try:
                        f_val = float(metric_val)
                        mlflow.log_metric(metric_key, round(f_val, 4))
                    except (ValueError, TypeError):
                        # Not a number, skip logging as metric
                        pass

        except Exception as exc:
            print(f"⚠️  [ExperimentTracker] Failed to log run: {exc}")

    def log_artifact_file(self, local_path: str) -> None:
        """Upload a local file as an MLflow artifact to the active run."""
        if not self._enabled or not _MLFLOW_AVAILABLE:
            return
        try:
            mlflow.log_artifact(local_path)
        except Exception as exc:
            print(f"⚠️  [ExperimentTracker] artifact upload failed: {exc}")
