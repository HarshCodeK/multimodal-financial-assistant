from types import SimpleNamespace

import pytest

from financial_assistant.ocr import answer_ocr_text, extract_image_text


PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload"


class FakeOCR:
    def __call__(self, _pixels):
        return [
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "TOTAL INR 420", 0.98),
            ([[0, 2], [1, 2], [1, 3], [0, 3]], "Date 2026-10-21", 0.92),
        ], 0.01


class FakeGenerator:
    def __init__(self):
        self.question = None
        self.chunks = None

    def answer(self, question, chunks):
        self.question = question
        self.chunks = chunks
        return {
            "supported": True,
            "answer": "The total is INR 420.",
            "citations": ["S1"],
            "model": "openai/gpt-oss-20b",
            "prompt_tokens": 80,
            "completion_tokens": 12,
            "prompt_preview": "question plus OCR text",
        }


def test_extract_image_text_uses_local_ocr(monkeypatch):
    from PIL import Image
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    monkeypatch.setattr("financial_assistant.ocr._get_ocr_engine", lambda: FakeOCR())
    result = extract_image_text(buffer.getvalue(), "receipt.png")
    assert result["filename"] == "receipt.png"
    assert "TOTAL INR 420" in result["text"]
    assert result["mean_confidence"] == 0.95


def test_ocr_answer_passes_extracted_text_not_image_to_text_generator():
    generator = FakeGenerator()
    result = answer_ocr_text("What is the total?", "TOTAL INR 420", "receipt.png", generator=generator)
    assert result["answer"] == "The total is INR 420."
    assert result["model"] == "openai/gpt-oss-20b"
    assert generator.question == "What is the total?"
    assert generator.chunks[0].text == "TOTAL INR 420"
    assert generator.chunks[0].source == "receipt.png"


def test_ocr_answer_rejects_blank_question():
    with pytest.raises(ValueError, match="Enter a question"):
        answer_ocr_text(" ", "TOTAL INR 420", generator=FakeGenerator())


def test_ocr_answer_rejects_empty_extracted_text():
    with pytest.raises(ValueError, match="No OCR text"):
        answer_ocr_text("What is the total?", " ", generator=FakeGenerator())


def test_extract_image_text_rejects_invalid_image():
    with pytest.raises(ValueError, match="Unsupported or invalid image"):
        extract_image_text(b"not-an-image", "receipt.png")
