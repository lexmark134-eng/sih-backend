"""


Instead of trusting the LLM's own retrieval-grounded answer blindly, every
draft reply is broken into individual factual claims, and each claim is
checked against the retrieved curriculum chunks using a separate,
cheap verification pass. A reply only reaches the student if its aggregate
faithfulness score clears settings.FAITHFULNESS_THRESHOLD — otherwise the
state machine regenerates it with a stricter prompt.

(If you want a heavier-weight, published-metric version later, swap this
for the `ragas` faithfulness metric — this hand-rolled version is faster,
has no extra dependency surface, and is easy to explain in a demo.)
"""
import json
from typing import List, Tuple
from langchain_ollama import ChatOllama
from app.config import settings
from app.schemas import RetrievedChunk

_verifier = ChatOllama(model=settings.CHAT_MODEL, temperature=0, base_url=settings.OLLAMA_BASE_URL)


def _extract_text(content) -> str:
    """Same normalization as state_machine.py """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)

_CLAIM_EXTRACTION_PROMPT = """Break the following tutor response into a list of
distinct factual/mathematical claims it makes. Ignore rhetorical questions
aimed at the student (Socratic prompts) — only extract statements presented
as fact. Return ONLY a JSON list of strings, nothing else.

Tutor response:
{reply}
"""

_VERIFICATION_PROMPT = """You are checking whether a CLAIM is supported by the
CONTEXT below. Answer with exactly one word: "supported", "unsupported", or
"not_applicable" (use not_applicable if the claim is not a factual claim at all).

CONTEXT:
{context}

CLAIM:
{claim}

Answer:
"""


def _extract_claims(reply: str) -> List[str]:
    resp = _verifier.invoke(_CLAIM_EXTRACTION_PROMPT.format(reply=reply))
    text = _extract_text(resp.content)
    try:
        claims = json.loads(text)
        if isinstance(claims, list):
            return [c for c in claims if isinstance(c, str)]
    except json.JSONDecodeError:
        pass
    return []


def score_faithfulness(reply: str, chunks: List[RetrievedChunk]) -> Tuple[float, List[str]]:
    """
    Returns (score in [0,1], list of unsupported claim strings).
    Score is fraction of extracted factual claims that are "supported".
    Claims that are not_applicable (Socratic questions, encouragement) are
    excluded from the denominator entirely.
    """
    context = "\n---\n".join(c.content for c in chunks)
    claims = _extract_claims(reply)
    if not claims:
        # No factual claims made (pure Socratic question) -> trivially faithful
        return 1.0, []

    unsupported = []
    applicable_count = 0
    supported_count = 0

    for claim in claims:
        verdict_resp = _verifier.invoke(
            _VERIFICATION_PROMPT.format(context=context, claim=claim)
        )
        verdict = _extract_text(verdict_resp.content).strip().lower()

        if "not_applicable" in verdict:
            continue
        applicable_count += 1
        if "unsupported" in verdict:
            unsupported.append(claim)
        else:
            supported_count += 1

    if applicable_count == 0:
        return 1.0, []

    return supported_count / applicable_count, unsupported
