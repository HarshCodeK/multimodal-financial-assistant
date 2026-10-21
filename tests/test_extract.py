from pathlib import Path
from unittest.mock import patch

import pymupdf

from financial_assistant.extract import chunk_text, extract_pdf, extract_pdf_bytes
from financial_assistant.models import SourceType


def test_chunking_uses_configured_overlap():
    words = [f"word{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), source="policy.pdf", source_type=SourceType.pdf, page=1)

    assert len(chunks) == 2
    assert len(chunks[0].text.split()) == 80
    assert len(chunks[1].text.split()) == 36
    assert chunks[0].text.split()[-16:] == chunks[1].text.split()[:16]


def test_pdf_extraction_preserves_page_numbers(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Meal reimbursement limit is INR 1200 per employee per day.")
    path = tmp_path / "policy.pdf"
    doc.save(path)
    doc.close()

    chunks = extract_pdf(path)

    assert chunks
    assert all(chunk.source == "policy.pdf" for chunk in chunks)
    assert all(chunk.page == 1 for chunk in chunks)
    assert any("1200" in chunk.text for chunk in chunks)


def test_image_only_pdf_uses_local_ocr_fallback_and_skips_unreadable_page():
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
    pix.clear_with(255)
    page.insert_image(pymupdf.Rect(72, 72, 160, 160), pixmap=pix)
    raw = doc.tobytes()
    doc.close()

    with patch("financial_assistant.extract.extract_image_text", side_effect=ValueError("no readable text")) as ocr:
        assert extract_pdf_bytes(raw, source_name="scanned.pdf") == []
    ocr.assert_called_once()


def test_included_sample_pdf_has_extractable_policy_text():
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_financial_policy.pdf"
    chunks = extract_pdf(sample)

    assert chunks
    assert any("INR 1,200" in chunk.text for chunk in chunks)
    assert any("15 calendar days" in chunk.text for chunk in chunks)
