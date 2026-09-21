"""Offline smoke tests — no network calls, LLM layers stubbed.

Run from repo root:  python -m pytest tests/ -q
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_parser import load_document  # noqa: E402
from src.vision_extractor import _parse_json  # noqa: E402
import src.monitor as monitor  # noqa: E402
from src.monitor import init_db, log_interaction, get_recent_logs  # noqa: E402


def test_parser_rejects_unsupported_format(tmp_path):
    bad = tmp_path / "notes.docx"
    bad.write_text("hello")
    try:
        load_document(str(bad))
        assert False, "should have raised"
    except ValueError as e:
        assert "Unsupported" in str(e)


def test_parser_handles_image_path(tmp_path):
    img = tmp_path / "receipt.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # stub bytes; parser only routes by extension
    doc = load_document(str(img))
    assert doc["type"] == "image"
    assert doc["filename"] == "receipt.png"


def test_parser_extracts_pdf_text(tmp_path):
    try:
        fitz = __import__("fitz")  # pymupdf
    except ImportError:
        pytest.skip("pymupdf (fitz) not installed — PDF parsing test skipped")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "TOTAL AMOUNT DUE: $4512.78")
    path = tmp_path / "invoice.pdf"
    doc.save(str(path))
    doc.close()

    parsed = load_document(str(path))
    assert parsed["type"] == "pdf"
    assert "4512.78" in parsed["text"]


def test_json_repair_strips_code_fences():
    fenced = '```json\n{"vendor": "Acme", "total": 100}\n```'
    assert _parse_json(fenced) == {"vendor": "Acme", "total": 100}
    plain = '  {"vendor": "Acme", "total": 100}  '
    assert _parse_json(plain) == {"vendor": "Acme", "total": 100}


def test_sqlite_log_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "DB_PATH", str(tmp_path / "test.db"))
    init_db()
    log_interaction("invoice.png", "Why this charge?", "Because policy X", {"total": 10}, 123.4)
    rows = get_recent_logs(5)
    assert len(rows) == 1
    assert rows[0][2] == "invoice.png"
    assert rows[0][3] == "Why this charge?"
    assert rows[0][6] == 123.4
