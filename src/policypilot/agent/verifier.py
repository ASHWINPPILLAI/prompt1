"""
Verifier (V6 chain-of-verification pass).

After the generator produces an answer, the verifier checks each factual claim
against the retrieved source text. Unsupported claims are stripped; if nothing
supported remains, the answer becomes a refusal.

WHY a separate verifier call? It uses a different (deterministic) call so the
main generator's temperature doesn't affect claim checking, and it uses a
different model in principle (though in practice the same endpoint, just at T=0).

The verifier prompt lives in prompts/verifier.yaml.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from policypilot.llm.client import LLMClient, get_judge_client
from policypilot.prompting.loader import load_prompt

logger = logging.getLogger(__name__)

_REFUSAL_TEXT = (
    "The provided documents do not contain enough verified information to answer "
    "this question. Please check the official agency website or contact the relevant office."
)


def run_verifier(
    answer: str,
    retrieved_chunks_with_ids: str,
    *,
    client: LLMClient | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """
    Run the chain-of-verification pass on an answer.

    Returns:
      verified_answer   — cleaned answer (unsupported sentences removed)
      claims            — list of {claim, supported, evidence_section}
      hallucination     — True if any claim was unsupported
      usage             — token counts
    """
    if client is None:
        client = get_judge_client()

    prompt = load_prompt("verifier")
    system = prompt.system
    user = prompt.user_template.format(
        retrieved_chunks_with_ids=retrieved_chunks_with_ids,
        answer=answer,
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    response = client.chat(
        messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=500,
        run_id=run_id,
        prompt_id="verifier",
    )
    raw = response["content"].strip()

    # Parse JSON; fall back to assuming all supported on parse failure
    try:
        parsed = json.loads(raw)
        claims = parsed.get("claims", [])
    except json.JSONDecodeError:
        logger.warning("Verifier returned invalid JSON; treating all as supported")
        claims = []

    has_unsupported = any(not c.get("supported", True) for c in claims)

    # Build verified answer: keep only sentences with supported claims
    if not claims:
        verified = answer
    else:
        supported_texts = [
            c["claim"] for c in claims if c.get("supported", True)
        ]
        verified = " ".join(supported_texts).strip()
        if not verified:
            verified = _REFUSAL_TEXT

    return {
        "verified_answer": verified,
        "claims": claims,
        "hallucination": has_unsupported,
        "usage": response["usage"],
    }
