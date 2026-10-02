"""Grounded answer generation: extracted fields + retrieved policy + question.

Why this file changed
---------------------
It hardcoded ``model="llama-3.3-70b-versatile"``, retired by Groq on
2026-07-17. Every answer 404'd. The model now comes from `src.models` and can
be chosen per-request from the UI.

It also now returns the policy *source* filenames alongside the chunk text.
An answer you cannot trace to a document is an answer you cannot audit, and
this project's whole claim is that it is grounded.
"""
from __future__ import annotations

import json
import time

from src import models
from src.knowledge_base import retrieve_with_sources
from src.monitor import log_interaction
from src.vision_extractor import ExtractionError, extract_fields

SYSTEM_PROMPT = """You are a financial assistant. Answer using ONLY the extracted document fields and the policy context provided.

Rules:
- Ground every claim in the context. If the context does not cover the question, say so plainly.
- Cite the policy document filename when you rely on it, like [refund_policy.txt].
- Be concise. One short paragraph plus a citation, no preamble."""

NO_CONTEXT_PROMPT = """You are a financial assistant. No policy context was retrieved for this question.

Say that the available policy documents do not address this question, then
answer only from the extracted document fields. Do not speculate."""


class AnswerError(RuntimeError):
    """Answer generation failed."""


def _build_query(extracted: dict) -> str:
    """Search using the two fields most likely to be questioned.

    Retrieving on what the user actually wants explained beats retrieving on
    the whole document.
    """
    vendor = extracted.get("vendor", "")
    flagged = extracted.get("flagged_charge", "")
    if isinstance(flagged, dict):
        flagged = flagged.get("description", "")
    else:
        flagged = str(flagged)
    return f"{vendor} {flagged}".strip() if vendor else flagged


def answer_question(document: dict, question: str, model: str | None = None) -> dict:
    """Extract, retrieve, answer. Logs the interaction to SQLite."""
    start = time.time()
    model_id = models.resolve(model)

    try:
        extracted = extract_fields(document, model=model_id)
    except ExtractionError as e:
        raise AnswerError(f"extraction failed: {e}") from e

    if "error" in extracted:
        return {
            "answer": "Could not extract fields from the document.",
            "extracted_fields": extracted,
            "policy_sources_used": [],
            "policy_files": [],
            "model": model_id,
        }

    query = _build_query(extracted)
    hits = retrieve_with_sources(query)
    policy_text = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)

    system = SYSTEM_PROMPT if hits else NO_CONTEXT_PROMPT
    prompt = (
        f"EXTRACTED FIELDS:\n{json.dumps(extracted, indent=2)}\n\n"
        f"POLICY CONTEXT:\n{policy_text or '(none retrieved)'}\n\n"
        f"USER QUESTION: {question}"
    )

    from groq import Groq, APIStatusError

    client = Groq(api_key=models.api_key())
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
    except APIStatusError as e:
        if e.status_code == 404:
            raise AnswerError(
                f"model {model_id!r} returned 404 — retired by Groq. Check src/models.py."
            ) from e
        raise AnswerError(f"Groq HTTP {e.status_code}: {e}") from e

    answer = completion.choices[0].message.content
    latency_ms = (time.time() - start) * 1000

    log_interaction(
        document.get("filename", "unknown"),
        question,
        answer,
        extracted,
        latency_ms,
        model=model_id,
        sources=[h["source"] for h in hits],
    )

    return {
        "answer": answer,
        "extracted_fields": extracted,
        "policy_sources_used": [h["text"] for h in hits],
        "policy_files": [h["source"] for h in hits],
        "model": model_id,
        "latency_ms": round(latency_ms, 1),
    }
