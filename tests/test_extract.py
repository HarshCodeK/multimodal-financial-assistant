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


def test_image_only_pdf_is_not_silently_ocr_processed():
    doc = pymupdf.open()
    doc.new_page()
    raw = doc.tobytes()
    doc.close()

    assert extract_pdf_bytes(raw, source_name="scanned.pdf") == []


def test_included_sample_pdf_has_extractable_policy_text():
    sample = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "sample_financial_policy.pdf"
    chunks = extract_pdf(sample)

    assert chunks
    assert any("INR 1,200" in chunk.text for chunk in chunks)
    assert any("15 calendar days" in chunk.text for chunk in chunks)
