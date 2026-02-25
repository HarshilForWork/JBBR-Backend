"""
Stage 7 — Inference: evaluator.py
Multi-provider LLM evaluator (Groq | Gemini) + retry + JSON extraction → LLMAnswer.

Provider is selected via config.yaml:
    llm:
      provider: groq        # or "gemini"
      model: llama-3.3-70b-versatile
      gemini_model: gemini-2.0-flash

API keys are read from environment variables:
    GROQ_API_KEY   — required when provider is "groq"
    GEMINI_API_KEY — required when provider is "gemini"
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

# ── Optional provider imports ─────────────────────────────────────────────────

try:
    from groq import Groq as GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .schema import LLMAnswer
from .prompt import SYSTEM_MESSAGE, build_policy_qa_prompt

# ── Config loader ─────────────────────────────────────────────────────────────
import os as _os
import yaml as _yaml

def _load_cfg() -> dict:
    cfg_path = _os.path.join(_os.path.dirname(__file__), "..", "..", "config.yaml")
    try:
        with open(cfg_path, "r") as _f:
            return _yaml.safe_load(_f) or {}
    except Exception:
        return {}

_CFG     = _load_cfg()
_LLM_CFG = _CFG.get("llm", {})


# ─────────────────────────────────────────────────────────────────────────────
# Public evaluator class
# ─────────────────────────────────────────────────────────────────────────────

class GroqEvaluator:
    """
    Multi-provider LLM evaluator for insurance policy QA.

    Reads `provider` from config.yaml to decide whether to use Groq or Gemini.
    The class name is kept as GroqEvaluator for backward compatibility with
    any existing imports in searcher.py.

    Parameters
    ----------
    groq_api_key     : Groq API key (falls back to GROQ_API_KEY env var).
    reranking_method : Name of the upstream reranker (stored in metadata).
    model            : Override the Groq model (ignores config if given).
    temperature      : Sampling temperature.
    max_tokens       : Max response tokens.
    max_retries      : Retry count on transient errors.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        reranking_method: str = "none",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.reranking_method = reranking_method
        self.temperature = temperature if temperature is not None else float(_LLM_CFG.get("temperature", 0.2))
        # max_tokens: None / null in config.yaml = no cap (omit from LLM call)
        _cfg_max = _LLM_CFG.get("max_tokens", None)
        self.max_tokens  = max_tokens if max_tokens is not None else (_cfg_max if _cfg_max else None)
        self.max_retries = max_retries if max_retries is not None else int(_LLM_CFG.get("max_retries", 2))

        self.provider = _LLM_CFG.get("provider", "groq").lower().strip()
        self._client  = None
        self.available = False

        if self.provider == "gemini":
            self._init_gemini()
        else:
            # Default / fallback: Groq
            self.provider = "groq"
            groq_key = groq_api_key or _os.getenv("GROQ_API_KEY", "")
            self.model_name = model or _LLM_CFG.get("model", "llama-3.3-70b-versatile")
            self._init_groq(groq_key)

    # ── Provider initialisers ─────────────────────────────────────────────────

    def _init_groq(self, key: str) -> None:
        if not GROQ_AVAILABLE:
            print("⚠️  GroqEvaluator: groq package not installed — pip install groq")
            return
        if not key:
            print("⚠️  GroqEvaluator: GROQ_API_KEY not set")
            return
        try:
            self._client = GroqClient(api_key=key)
            self.available = True
            print(f"✅ LLM provider: groq — model: {self.model_name}")
        except Exception as exc:
            print(f"❌ Groq init failed: {exc}")

    def _init_gemini(self) -> None:
        if not GEMINI_AVAILABLE:
            print("⚠️  google-genai not installed — pip install google-genai")
            return
        key = _os.getenv("GEMINI_API_KEY", "")
        if not key:
            print("⚠️  GEMINI_API_KEY not set")
            return
        try:
            self._client = genai.Client(api_key=key)
            self.model_name = _LLM_CFG.get("gemini_model", "gemini-2.0-flash")
            self.available = True
            print(f"✅ LLM provider: gemini — model: {self.model_name}")
        except Exception as exc:
            print(f"❌ Gemini init failed: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        context: str,
        method: str = "llm_with_quotes",
    ) -> LLMAnswer:
        """
        Run LLM evaluation and return a structured LLMAnswer.
        Always returns — never raises.
        """
        if not self.available or self._client is None:
            return LLMAnswer.unavailable(
                reason=f"{self.provider.upper()} API key missing or package not installed",
                reranking_method=self.reranking_method,
                model_used="none",
                evaluation_method=method,
            )

        prompt = build_policy_qa_prompt(query, context)

        for attempt in range(self.max_retries + 1):
            try:
                raw = self._call_llm(prompt)
                parsed = _extract_json(raw)
                answer = parsed["answer"] if (parsed and "answer" in parsed) else raw

                # Read confidence from LLM response; fall back to 0.5 if missing/invalid
                raw_conf = parsed.get("confidence") if parsed else None
                try:
                    confidence = float(raw_conf)
                    confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
                except (TypeError, ValueError):
                    confidence = 0.5   # LLM didn't return a valid number

                return LLMAnswer(
                    answer=answer,
                    confidence=confidence,
                    llm_available=True,
                    model_used=self.model_name,
                    reranking_method=self.reranking_method,
                    evaluation_method=method,
                    json_parsed=parsed is not None,
                    context_length=len(context),
                    num_sources=len(chunks),
                    llm_context=context,
                    source_vectors=chunks,
                )

            except Exception as exc:
                err = str(exc).lower()
                if "rate" in err or "limit" in err or "quota" in err or "429" in err:
                    print(f"⚠️ {self.provider} rate-limit: {exc}")
                    return LLMAnswer.rate_limited(
                        error=str(exc),
                        model_used=self.model_name,
                        reranking_method=self.reranking_method,
                        llm_context=context,
                        source_vectors=chunks,
                    )
                if attempt < self.max_retries:
                    print(f"🔄 {self.provider} retry {attempt + 1}/{self.max_retries}: {exc}")
                    time.sleep(1)
                    continue

                print(f"❌ {self.provider} failed after {self.max_retries + 1} attempts: {exc}")
                return LLMAnswer.failed(
                    error=str(exc),
                    model_used=self.model_name,
                    reranking_method=self.reranking_method,
                    evaluation_method=method,
                    llm_context=context,
                    source_vectors=chunks,
                )

        return LLMAnswer.failed(
            error="Exhausted retries without a result",
            model_used=self.model_name,
            reranking_method=self.reranking_method,
            evaluation_method=method,
            llm_context=context,
            source_vectors=chunks,
        )

    # ── Provider-specific call routing ────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Route to the correct provider SDK and return raw text response."""
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        return self._call_groq(prompt)

    def _call_groq(self, prompt: str) -> str:
        kwargs = dict(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user",   "content": prompt},
            ],
            temperature=self.temperature,
        )
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    def _call_gemini(self, prompt: str) -> str:
        cfg_kwargs = dict(
            system_instruction=SYSTEM_MESSAGE,
            temperature=self.temperature,
        )
        if self.max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = self.max_tokens
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=genai_types.GenerateContentConfig(**cfg_kwargs),
        )
        return response.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract a JSON object from an LLM response string.
    Tries four strategies in order:
      1. Full-string parse
      2. Substring between first '{' and last '}'
      3. Regex for  {"answer": "..."} pattern
      4. Regex to extract just the answer value
    Returns None if all strategies fail.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # 1. Full parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Brace substring
    if "{" in text and "}" in text:
        try:
            start = text.find("{")
            end   = text.rfind("}") + 1
            return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Regex: whole {"answer": ...} block
    for m in re.findall(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue

    # 4. Regex: just the answer value (+ optional confidence)
    m = re.search(r'"answer":\s*"([^"]*)"', text)
    if m:
        result = {"answer": m.group(1)}
        # Also try to pick up confidence if present
        cm = re.search(r'"confidence":\s*([0-9]*\.?[0-9]+)', text)
        if cm:
            try:
                result["confidence"] = float(cm.group(1))
            except ValueError:
                pass
        return result

    return None
