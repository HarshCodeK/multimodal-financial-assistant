"""Extract structured fields from a receipt or invoice with a vision model.

Works for two input types:
  image -> base64 -> multimodal model reads the pixels
  PDF   -> PyMuPDF extracts text -> model reads the text

Why JSON mode: asking a model for JSON and then parsing it is a coin flip.
`response_format={"type": "json_object"}` makes the provider guarantee the
shape, so parsing is not a hopeful `json.loads`. One retry remains, because a
model can still return prose inside a valid-looking envelope.
"""
import base64
import json
import mimetypes
import os

from . import models

EXTRACT_PROMPT = """Extract these fields from this financial document as a JSON object:
- vendor: merchant or company name
- line_items: list of {description, amount} for each charge
- subtotal: number, no currency symbol
- tax: number, no currency symbol
- total: number, no currency symbol
- date: document date
- flagged_charge: {description, amount} for the single line most worth questioning

Respond with the JSON object only."""


class ExtractionError(RuntimeError):
    """Extraction failed, and we know why."""


def _encode_image(path: str) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path)
    if mime not in {"image/jpeg", "image/png"}:
        raise ExtractionError("unsupported image type; use JPEG or PNG")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _call(messages, model_id):
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
        model=model_id, messages=messages,
        temperature=0.1, response_format={"type": "json_object"},
    )


def _parse(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(cleaned.strip())


def extract(document: dict, model_id: str = None) -> dict:
    """`document` is {"type": "image"|"pdf", "path"|"text": ..., "filename": ...}"""
    model_id = models.resolve(model_id)

    if document["type"] == "image":
        if not models.MODELS[model_id][1]:   # index 1 is the vision flag
            raise ExtractionError(
                f"{model_id} cannot read images. Use one of: "
                f"{', '.join(models.VISION_MODELS)}"
            )
        encoded, mime = _encode_image(document["path"])
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": EXTRACT_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{encoded}",
                    "detail": "high"}},
            ],
        }]
    elif document["type"] == "pdf":
        messages = [{"role": "user", "content":
                     f"{EXTRACT_PROMPT}\n\nDocument text:\n{document['text']}"}]
    else:
        raise ValueError(f"unsupported document type: {document['type']}")

    last = None
    for attempt in range(2):
        try:
            return _parse(_call(messages, model_id).choices[0].message.content)
        except json.JSONDecodeError as e:
            last = f"model returned unparseable JSON: {e}"
        except Exception as e:
            if type(e).__name__ == "APIStatusError" and getattr(e, "status_code", None) == 404:
                raise ExtractionError(
                    f"model {model_id!r} returned 404 -- check src/models.py"
                ) from e
            last = f"{type(e).__name__}: {e}"[:160]
        # Second attempt: restate the instruction more forcefully.
        if attempt == 0:
            if isinstance(messages[0]["content"], str):
                messages[0]["content"] = messages[0]["content"] + \
                    "\n\nOutput raw JSON only. No prose, no markdown fences."
            else:
                messages[0]["content"][0]["text"] += \
                    "\n\nOutput raw JSON only. No prose, no markdown fences."
    raise ExtractionError(last or "extraction failed")
