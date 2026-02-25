"""
Stage 7 — Inference: Groq LLM evaluation → structured JSON answer.

Modules
-------
schema.py    : LLMAnswer dataclass — the typed output of every LLM call.
prompt.py    : Prompt templates (SYSTEM_MESSAGE, build_policy_qa_prompt).
evaluator.py : GroqEvaluator — Groq chat completion + retry + JSON extraction.

Usage
-----
    from src.inference import GroqEvaluator, LLMAnswer

    evaluator = GroqEvaluator(groq_api_key=key, reranking_method="bge-reranker-v2-m3")
    answer: LLMAnswer = evaluator.evaluate(query, chunks, context)
    result_dict = answer.to_dict()

Config keys (config.yaml → llm section)
----------------------------------------
    model        : Groq model identifier
    temperature  : Sampling temperature
    max_tokens   : Max response tokens
    max_retries  : Retry count on transient errors
"""

from .schema    import LLMAnswer
from .prompt    import SYSTEM_MESSAGE, build_policy_qa_prompt
from .evaluator import GroqEvaluator

__all__ = ["LLMAnswer", "GroqEvaluator", "SYSTEM_MESSAGE", "build_policy_qa_prompt"]
