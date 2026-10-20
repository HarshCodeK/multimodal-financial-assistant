import json
import os
import re

import httpx
from pydantic import BaseModel, Field

from .models import Answer, RetrievedChunk


class _GroundedLLMResponse(BaseModel):
    supported: bool
    answer: str = Field(min_length=1)


def _parse_grounded_response(raw: str) -> _GroundedLLMResponse:
    """Parse provider JSON, with a small, deterministic repair pass for common formatting errors."""
    candidates = [raw.strip()]
    # Providers occasionally wrap otherwise valid JSON in Markdown fences.
    unfenced = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.IGNORECASE)
    if unfenced != candidates[0]:
        candidates.append(unfenced)

    for candidate in candidates:
        try:
            return _GroundedLLMResponse.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValueError):
            pass

    # Repair only unambiguous trailing commas before a closing brace/bracket.
    # Do not invent missing fields or coerce arbitrary prose into an answer.
    for candidate in candidates:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return _GroundedLLMResponse.model_validate(json.loads(repaired))
        except (json.JSONDecodeError, ValueError):
            pass

    # Some providers prepend text. Extract the first balanced JSON object while
    # respecting quoted strings, then apply the same conservative trailing-comma repair.
    for candidate in candidates:
        start = candidate.find("{")
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    extracted = candidate[start:index + 1]
                    repaired = re.sub(r",\s*([}\]])", r"\1", extracted)
                    try:
                        return _GroundedLLMResponse.model_validate(json.loads(repaired))
                    except (json.JSONDecodeError, ValueError):
                        break

    raise ValueError("Groq returned malformed grounded-answer JSON")


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, vision_model: str | None = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.vision_model = vision_model or os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

    def _request(self, payload: dict) -> dict:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        response = httpx.post(self.base_url, headers={"Authorization": f"Bearer {self.api_key}"}, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def answer(self, question: str, evidence: list[RetrievedChunk]) -> Answer:
        if not evidence:
            return Answer(supported=False, answer="I don't have enough evidence in the indexed documents to answer that.")
        context = "\n\n".join(f"[{i+1}] {item.text}" for i, item in enumerate(evidence))
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a grounded financial-policy assistant. Use only the supplied evidence. Return JSON with exactly two fields: supported (boolean) and answer (string). Set supported=false when the evidence does not directly support the answer. Never invent policy details."},
                {"role": "user", "content": f"Evidence:\n{context}\n\nQuestion: {question}"},
            ],
            "response_format": {"type": "json_object"},
        }
        raw = self._request(payload)["choices"][0]["message"]["content"]
        parsed = _parse_grounded_response(raw)
        return Answer(supported=parsed.supported, answer=parsed.answer, sources=evidence if parsed.supported else [])

    def describe_image(self, question: str, image_data_url: str) -> str:
        payload = {"model": self.vision_model, "temperature": 0, "messages": [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": image_data_url}}]}]}
        return self._request(payload)["choices"][0]["message"]["content"].strip()
