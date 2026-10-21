"""Small, testable helpers for session-scoped Streamlit UI state."""

from __future__ import annotations

from hashlib import sha256
from typing import Callable, MutableMapping, Any


def image_fingerprint(image_bytes: bytes) -> str:
    """Return a stable content key so identical image bytes share one OCR result."""
    return sha256(image_bytes).hexdigest()


def get_or_extract_image_text(
    session_state: MutableMapping[str, Any],
    image_bytes: bytes,
    filename: str,
    extractor: Callable[[bytes, str], dict],
) -> dict:
    """Reuse OCR for the same image; replace the single-image cache when bytes change."""
    fingerprint = image_fingerprint(image_bytes)
    cached = session_state.get("image_ocr_cache")
    if cached and cached.get("fingerprint") == fingerprint:
        return cached

    extracted = extractor(image_bytes, filename)
    cached = {
        "fingerprint": fingerprint,
        "filename": extracted.get("filename", filename),
        "text": extracted["text"],
        "mean_confidence": extracted.get("mean_confidence"),
    }
    session_state["image_ocr_cache"] = cached
    return cached


def active_pdf_sources(persisted_sources: list[str], indexed_this_session: bool) -> list[str]:
    """Only expose persistent Chroma sources as active after this session indexes a PDF."""
    return list(persisted_sources) if indexed_this_session else []
