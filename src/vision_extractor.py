import os
import json
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"

EXTRACTION_PROMPT = """Extract the following fields from this financial document as a JSON object:
- vendor: the merchant or company name
- line_items: list of individual charges with description and amount
- subtotal: the subtotal amount (number, no currency symbol)
- tax: the tax amount (number, no currency symbol)
- total: the total amount (number, no currency symbol)
- date: the document date
- flagged_charge: the single line item most likely to be questioned

Respond with valid JSON only, no markdown, no explanation."""

STRICTER_PROMPT = """Extract the same fields as JSON. Respond with valid JSON only. No markdown formatting, no code fences, no extra text."""

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _call_llm(messages: list, model: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    return completion.choices[0].message.content

def _parse_json(response_text: str) -> dict:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())

def extract_fields(document: dict) -> dict:
    if document["type"] == "image":
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
        resp = None
        try:
            resp = _call_llm(messages, VISION_MODEL)
            return _parse_json(resp)
        except (json.JSONDecodeError, Exception):
            try:
                messages[0]["content"][0]["text"] = STRICTER_PROMPT
                resp = _call_llm(messages, VISION_MODEL)
                return _parse_json(resp)
            except (json.JSONDecodeError, Exception):
                return {"error": "could not parse", "raw_response": resp}
    elif document["type"] == "pdf":
        messages = [
            {
                "role": "user",
                "content": f"{EXTRACTION_PROMPT}\n\nDocument text:\n{document['text']}",
            }
        ]
        resp = None
        try:
            resp = _call_llm(messages, TEXT_MODEL)
            return _parse_json(resp)
        except (json.JSONDecodeError, Exception):
            try:
                messages[0]["content"] = f"{STRICTER_PROMPT}\n\nDocument text:\n{document['text']}"
                resp = _call_llm(messages, TEXT_MODEL)
                return _parse_json(resp)
            except (json.JSONDecodeError, Exception):
                return {"error": "could not parse", "raw_response": resp}
    else:
        raise ValueError(f"Unknown document type: {document['type']}")
