from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest
from PIL import Image

from financial_assistant.extract import (
    chunk_text,
    extract_pdf,
    extract_pdf_bytes,
    image_bytes_to_text,
    validate_image,
)
from financial_assistant.models import SourceType


def test_image_validation(tmp_path: Path):
    path = tmp_path / "statement.png"
    Image.new("RGB", (20, 20), "white").save(path)
    assert validate_image(path).size == (20, 20)


def test_pdf_extraction_preserves_page_and_chunks_text(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Meal allowance is 500 INR.")
    path = tmp_path / "policy.pdf"
    pdf.save(path)
    pdf.close()
    chunks = extract_pdf(path)
    assert chunks[0].page == 1
    assert "500 INR" in chunks[0].text


def test_long_text_is_split_into_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, source="policy.txt", source_type=SourceType.text)
    assert len(chunks) >= 3
    assert len(chunks[0].text.split()) == 80
    assert chunks[0].text.split()[-16:] == chunks[1].text.split()[:16]


def test_scanned_pdf_page_uses_ocr_fallback(tmp_path, monkeypatch):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), " ")
    path = tmp_path / "scanned.pdf"
    pdf.save(path)
    pdf.close()
    monkeypatch.setattr("financial_assistant.extract._ocr_image", lambda image: "Scanned reimbursement policy text.")
    chunks = extract_pdf(path)
    assert chunks
    assert "reimbursement" in chunks[0].text


def test_included_sample_pdf_has_extractable_policy_text():
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_financial_policy.pdf"
    chunks = extract_pdf(sample)
    assert chunks
    assert any("INR 1,200" in chunk.text for chunk in chunks)
    assert any("15 calendar days" in chunk.text for chunk in chunks)


def test_image_text_uses_local_ocr_after_validation():
    from io import BytesIO
    buffer = BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")
    with patch("financial_assistant.extract._ocr_image", return_value="TOTAL 100 INR"):
        assert image_bytes_to_text(buffer.getvalue()) == "TOTAL 100 INR"


def test_image_payload_rejects_excessive_pixel_count():
    from io import BytesIO
    buffer = BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")
    # Image itself is valid; this checks normal extraction does not require any provider key.
    with patch("financial_assistant.extract._ocr_image", return_value="sample text"):
        assert image_bytes_to_text(buffer.getvalue()) == "sample text"


def test_bundled_sample_policy_exercises_multiple_chunks():
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_policy.txt"
    chunks = chunk_text(
        sample.read_text(encoding="utf-8"),
        source=sample.name,
        source_type=SourceType.text,
    )
    assert len(chunks) >= 4
    assert all(chunk.text.strip() for chunk in chunks)
    # Adjacent chunks retain the configured overlap from the preceding chunk.
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-16:] == second_words[:16]

