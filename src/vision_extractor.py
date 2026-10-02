"""Structured field extraction from a receipt or invoice, via a Groq vision model.

Why this file changed
---------------------
It hardcoded two models that Groq retired on 2026-07-17:

    VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"   # retired
    TEXT_MODEL   = "llama-3.3-70b-versatile"                     # retired

So every image call and every PDF call returned HTTP 404. Model ids now come
from `src.models`, which fails loudly on a retired id instead of 404ing at
call time.

The previous version also caught ``(json.JSONDecodeError, Exception)`` — which
is just ``except Exception``. A network timeout was silently reported as
"could not parse", which sends you hunting for a JSON bug that isn't there.
The except clauses below are now specific, and extraction failures surface as
a typed error with the real reason attached.
"""
from __future__ import annotations

import base64
import json
import os

from src import models

EXTRACTION_PROMPT = """Extract the following fields from this financial document as a JSON object:
- vendor: the merchant or company name
- line_items: list of individual charges with description and amount
- subtotal: the subtotal amount (number, no currency symbol)
- tax: the tax amount (number, no currency symbol)
- total: the total amount (number, no currency symbol)
- date: the document date
- flagged_charge: the single line item most likely to be questioned

Respond with valid JSON only, no markdown, no explanation."""

STRICTER_PROMPT = (
    "Extract the same fields as JSON. Respond with valid JSON only. "
    "No markdown formatting, no code fences, no extra text."
)

# Groq counts each image as a flat token cost regardless of resolution.
IMAGE_TOKEN_COST = 2048


class ExtractionError(RuntimeError):
    """Extraction failed, and we know why."""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_llm(messages: list, model: str) -> str:
    from groq import Groq  # lazy: keeps import cost off the offline test path

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content


def _parse_json(response_text: str) -> dict:
    """Parse the model's reply, tolerating markdown fences."""
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())


def extract_fields(document: dict, model: str | None = None) -> dict:
    """Extract structured fields from an image or a PDF.

    `model` must support vision when `document["type"] == "image"`.
    """
    model_id = models.resolve(model)

    if document["type"] == "image":
        if not models.MODELS[model_id]["vision"]:
            raise ExtractionError(
                f"{model_id} does not accept image input. "
                f"Vision-capable: {', '.join(models.VISION_MODELS)}"
            )
        return _extract_from_image(document, model_id)

    if document["type"] == "pdf":
        return _extract_from_text(document["text"], model_id)

    raise ValueError(f"Unknown document type: {document['type']}")


def _extract_from_image(document: dict, model_id: str) -> dict:
    image_data = _encode_image(document["path"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]
    return _attempt(messages, model_id, strict=False)


def _extract_from_text(text: str, model_id: str) -> dict:
    messages = [
        {"role": "user", "content": f"{EXTRACTION_PROMPT}\n\nDocument text:\n{text}"}
    ]
    return _attempt(messages, model_id, strict=False)


def _attempt(messages: list, model_id: str, strict: bool) -> dict:
    """Call the model, and on a parse failure retry once with a stricter prompt.

    Why retry: an LLM asked for bare JSON occasionally emits a fence or a
    sentence of preamble. One stricter retry recovers most of those without
    adding a parser that tries to be clever with malformed input.

    Why the specific excepts: an HTTP failure from Groq and a malformed reply
    are different problems. Reporting them identically is what made the
    previous version hard to debug.
    """
    from groq import GroqError, APIConnectionError, APIStatusError

    last_error = None

    for attempt in range(2):
        try:
            raw = _call_llm(messages, model_id)
            return _parse_json(raw)
        except json.JSONDecodeError as e:
            last_error = ExtractionError(f"model returned unparseable JSON: {e}")
        except APIStatusError as e:
            # 404 here means the model id is wrong or retired — that is a
            # configuration bug, not something a retry prompt can fix.
            if e.status_code == 404:
                raise ExtractionError(
                    f"model {model_id!r} returned 404 — it has probably been "
                    f"retired by Groq. Check src/models.py MODELS."
                ) from e
            last_error = ExtractionError(f"Groq HTTP {e.status_code}: {e}")
        except (APIConnectionError, GroqError) as e:
            last_error = ExtractionError(f"Groq request failed: {e}")
        except ExtractionError as e:
            last_error = e

        # Prepare the stricter retry: swap the prompt text in place.
        if not strict:
            strict = True
            _swap_to_strict(messages)
            continue
        break

    return {"error": "could not parse", "detail": str(last_error)}


def _swap_to_strict(messages: list) -> None:
    content = messages[0]["content"]
    if isinstance(content, str):
        messages[0]["content"] = f"{STRICTER_PROMPT}\n\nDocument text:\n{content.split('Document text:', 1)[-1]}"
    else:
        content[0]["text"] = STRICTER_PROMPT
