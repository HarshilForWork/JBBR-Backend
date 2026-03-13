"""
src/ops/cost_tracker.py
========================
Token counting + cost estimation for LLM and embedding API calls.

Uses `tiktoken` for accurate prompt/completion token counts.
Falls back to character-based estimation if tiktoken is unavailable.

Thresholds come from config.yaml → cost:
  groq_input_per_1k_tokens
  groq_output_per_1k_tokens
  embedding_per_1k_tokens
  expensive_query_token_limit
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

# ── Optional tiktoken ─────────────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")  # compatible with GPT-4 / Llama
    _TIKTOKEN_AVAILABLE = True
except (ImportError, Exception):
    _TIKTOKEN_AVAILABLE = False
    _enc = None


# ── Config ────────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

_CFG      = _load_cfg()
_COST_CFG = _CFG.get("cost", {})

_GROQ_INPUT_PER_1K  = float(_COST_CFG.get("groq_input_per_1k_tokens",  0.0001))
_GROQ_OUTPUT_PER_1K = float(_COST_CFG.get("groq_output_per_1k_tokens", 0.0002))
_EMB_PER_1K         = float(_COST_CFG.get("embedding_per_1k_tokens",   0.00002))
_EXPENSIVE_LIMIT    = int(_COST_CFG.get("expensive_query_token_limit",  8000))


# ── Helpers ────────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Count tokens in a text string.
    Uses tiktoken cl100k_base encoding if available, else estimates
    via character count (÷ 4 is a common rule of thumb).
    """
    if _TIKTOKEN_AVAILABLE and _enc is not None:
        return len(_enc.encode(text))
    # Fallback: 1 token ≈ 4 chars
    return max(1, len(text) // 4)


def estimate_embedding_tokens(texts: list[str]) -> int:
    """Estimate embedding token count for a list of texts."""
    return sum(count_tokens(t) for t in texts)


# ── Main tracker ───────────────────────────────────────────────────────────────

class CostTracker:
    """
    Tracks token usage and estimates cost for a single pipeline run.

    Usage:
        tracker = CostTracker(model="llama-3.3-70b-versatile")
        tracker.record_llm_call(prompt_text, completion_text)
        tracker.record_embedding(chunks)
        summary = tracker.summary()
    """

    def __init__(self, model: str = "unknown") -> None:
        self.model             = model
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.embedding_tokens  = 0

    # ── Recording methods ──────────────────────────────────────────────────────

    def record_llm_call(self, prompt: str, completion: str = "") -> Tuple[int, int]:
        """
        Count and accumulate tokens for one LLM call.

        Returns (prompt_tokens, completion_tokens) for this call.
        """
        p = count_tokens(prompt)
        c = count_tokens(completion) if completion else 0
        self.prompt_tokens     += p
        self.completion_tokens += c
        return p, c

    def record_embedding(self, texts: list[str]) -> int:
        """Count embedding tokens for a batch of texts."""
        n = estimate_embedding_tokens(texts)
        self.embedding_tokens += n
        return n

    # ── Summary ────────────────────────────────────────────────────────────────

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.embedding_tokens

    def estimated_cost_usd(self) -> float:
        """Estimate total USD cost for this run."""
        llm_input_cost  = (self.prompt_tokens     / 1000) * _GROQ_INPUT_PER_1K
        llm_output_cost = (self.completion_tokens  / 1000) * _GROQ_OUTPUT_PER_1K
        emb_cost        = (self.embedding_tokens   / 1000) * _EMB_PER_1K
        return llm_input_cost + llm_output_cost + emb_cost

    def is_expensive(self) -> bool:
        """Return True if token count exceeds the configurable expensive-query limit."""
        return (self.prompt_tokens + self.completion_tokens) > _EXPENSIVE_LIMIT

    def summary(self) -> Dict[str, object]:
        """Return a dict suitable for logging to MLflow / experiment tracker."""
        return {
            "prompt_tokens":      self.prompt_tokens,
            "completion_tokens":  self.completion_tokens,
            "embedding_tokens":   self.embedding_tokens,
            "total_tokens":       self.total_tokens(),
            "estimated_cost_usd": round(self.estimated_cost_usd(), 6),
            "is_expensive":       self.is_expensive(),
            "model":              self.model,
        }
