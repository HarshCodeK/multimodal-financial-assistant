"""Local OCR for image questions; only extracted text is sent to the text LLM."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .generation import GroundedAnswerGenerator
from .models import RetrievedChunk, SourceType

MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _validate_image(raw: bytes) -> None:
    if not raw:
        raise ValueError("The uploaded image is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than the 5 MiB limit.")
    if not (
        raw.startswith(b"\x89PNG\r\n\x1a\n")
        or raw.startswith(b"\xff\xd8\xff")
        or (len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP")
    ):
        raise ValueError("Unsupported or invalid image data. Upload a PNG, JPEG, or WebP image.")
    try:
        with Image.open(BytesIO(raw)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc


@lru_cache(maxsize=1)
def _get_ocr_engine():
    """Load/cache local OCR engine. Its model files are downloaded on first use if needed."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "Local OCR is unavailable. Install the project dependencies with "
            "the documented install command, then restart Streamlit."
        ) from exc
    return RapidOCR()


def extract_image_text(image_bytes: bytes, filename: str = "image") -> dict:
    """Run local OCR and return extracted text plus mean OCR confidence."""
    _validate_image(image_bytes)
    with Image.open(BytesIO(image_bytes)) as image:
        pixels = np.asarray(image.convert("RGB"))

    result, _elapsed = _get_ocr_engine()(pixels)
    lines: list[str] = []
    confidences: list[float] = []
    for item in result or []:
        if len(item) < 2:
            continue
        text = str(item[1]).strip()
        if not text:
            continue
        lines.append(text)
        if len(item) > 2:
            try:
                confidences.append(float(item[2]))
            except (TypeError, ValueError):
                pass

    extracted_text = "\n".join(lines).strip()
    if not extracted_text:
        raise ValueError(
            "OCR could not find readable text in this image. Try a sharper, well-lit image. "
            "This text-only workflow cannot interpret charts or visual layouts without readable text."
        )
    return {
        "filename": Path(filename).name,
        "text": extracted_text,
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
    }


def answer_ocr_text(
    question: str,
    extracted_text: str,
    filename: str = "image",
    *,
    generator: GroundedAnswerGenerator | None = None,
) -> dict:
    """Answer using OCR text and the configured text model; never sends image bytes to Groq."""
    normalized_question = question.strip()
    text = extracted_text.strip()
    if not normalized_question:
        raise ValueError("Enter a question about the image.")
    if not text:
        raise ValueError("No OCR text is available to answer from.")

    answer_generator = generator or GroundedAnswerGenerator()
    source = RetrievedChunk(
        source=Path(filename).name,
        source_type=SourceType.image,
        text=text,
        page=None,
        distance=0.0,
    )
    result = answer_generator.answer(normalized_question, [source])
    return {
        "filename": Path(filename).name,
        "question": normalized_question,
        "ocr_text": text,
        "answer": result["answer"],
        "supported": result["supported"],
        "citations": result["citations"],
        "model": result["model"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "prompt_preview": result.get("prompt_preview", ""),
    }
