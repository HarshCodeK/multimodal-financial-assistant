"""Answer a question using the extracted fields plus retrieved policy.

The grounding rule is in the system prompt: if the retrieved policy does not
cover the question, say so. A model given three irrelevant policy chunks will
happily invent a reason if allowed to, and an invented reason on a financial
question is worse than no answer.
"""
import json
import os
import time

from . import models
from .extract import ExtractionError, extract
from .knowledge_base import search

SYSTEM = """You are a financial document reviewer.

Rules:
- Ground every claim in the POLICY CONTEXT. If the context does not address the
  question, say so plainly and do not speculate.
- Cite the policy document filename when you rely on it, like [refund_policy.txt].
- Be concise. No preamble."""


def answer(document: dict, question: str, model_id: str = None) -> dict:
    """Extract, retrieve, answer. Returns the answer plus its sources."""
    started = time.time()
    model_id = models.resolve(model_id)

    fields = extract(document, model_id)

    # Search on what a human would question: the vendor and the odd charge.
    flagged = fields.get("flagged_charge") or {}
    if isinstance(flagged, dict):
        flagged = flagged.get("description", "")
    query = f"{fields.get('vendor','')} {flagged}".strip()
    hits = search(query)

    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits) or "(none retrieved)"
    prompt = (
        f"EXTRACTED FIELDS:\n{json.dumps(fields, indent=2, default=str)}\n\n"
        f"POLICY CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}"
    )

    from groq import Groq
    resp = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
        model=model_id, temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
    )

    # Dedupe sources while preserving order: top-k retrieval returns chunks,
    # not documents, so a 3-chunk result can be 2 chunks from one policy file.
    # Citing that file twice looks like a bug even though retrieval was correct.
    seen, sources = set(), []
    for h in hits:
        if h["source"] not in seen:
            seen.add(h["source"])
            sources.append(h["source"])

    return {
        "answer": resp.choices[0].message.content,
        "fields": fields,
        "sources": sources,
        "model": model_id,
        "elapsed_ms": round((time.time() - started) * 1000),
    }
