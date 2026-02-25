"""
Stage 7 — Inference: prompt.py
Prompt templates for Groq LLM evaluation.
Keeps all prompt engineering in one place so it is easy to iterate
without touching evaluator logic.
"""
from __future__ import annotations

# ── System message ────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = "You are an insurance policy expert. Always return valid JSON."

# ── Policy QA prompt template ─────────────────────────────────────────────────

_POLICY_QA_TEMPLATE = """\
You are an insurance policy expert. Based on the comprehensive context from \
policy documents, provide a clear and concise answer in 2-3 sentences.

QUERY: {query}

CONTEXT FROM POLICY DOCUMENTS:
{context}

Instructions:
- The context vectors were reranked using BGE Reranker-v2-m3 for maximum relevance.
- Consider information from all sections when forming your answer.
- Look for complementary information across different sections.
- For coverage questions, check waiting periods, exclusions, and conditions.
- For amount/limit questions, look for specific numbers in any section.
- Focus on meaning and context, not exact phrase matching.

CRITICAL: Return ONLY valid JSON with exactly these two fields:
{{"answer": "Your detailed answer here", "confidence": 0.95}}

Confidence scoring guide (be honest, not optimistic):
- 0.90–1.00 : Answer stated explicitly and unambiguously in the context
- 0.70–0.89 : Answer clearly implied or can be reliably inferred from context
- 0.50–0.69 : Partial information in context; some inference required
- 0.00–0.49 : Context is vague, contradictory, or largely missing

RULES:
- Yes/no questions: "Yes, [brief reason]" or "No, [brief reason]"
- Infer from context if not explicitly stated.
- RETURN ONLY THE JSON OBJECT — NO OTHER TEXT.\
"""


def build_policy_qa_prompt(query: str, context: str) -> str:
    """
    Render the policy QA prompt template.

    Uses manual string replacement instead of .format() to avoid KeyError when
    the policy PDF text contains literal { } characters (tables, section refs, etc.)
    that Python's str.format() would misinterpret as format fields.

    Parameters
    ----------
    query   : The user's question.
    context : The comprehensive context string built by the retrieval stage.

    Returns
    -------
    str : Fully formatted prompt ready to send to the LLM.
    """
    # Substitute {query} and {context} first, then unescape {{ / }} from the template.
    return (
        _POLICY_QA_TEMPLATE
        .replace("{query}", query)
        .replace("{context}", context)
        .replace("{{", "{")
        .replace("}}", "}")
    )
