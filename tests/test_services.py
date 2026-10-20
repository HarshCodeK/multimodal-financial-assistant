from unittest.mock import patch

import pytest

from financial_assistant.models import Answer, DocumentChunk, RetrievedChunk, SourceType
from financial_assistant.services import FinancialAssistant


class FakeRetriever:
    def __init__(self):
        self.upserted = []
        self.evidence = [
            RetrievedChunk(
                source="policy.pdf",
                source_type=SourceType.pdf,
                text="Meals are reimbursed up to 1200 INR per day.",
                page=2,
                distance=0.1,
            )
        ]

    def upsert(self, chunks):
        self.upserted.extend(chunks)

    def search(self, question):
        return self.evidence


class FakeLLM:
    def __init__(self):
        self.image_calls = []

    def answer(self, question, evidence):
        return Answer(
            supported=bool(evidence),
            answer=evidence[0].text if evidence else "I don't have enough evidence.",
            sources=evidence,
        )

    def describe_image(self, question, image_data_url):
        self.image_calls.append((question, image_data_url))
        return "Receipt total: 892.50 INR."


class FakeStore:
    def __init__(self):
        self.rows = []

    def record(self, question, supported, answer, latency_ms=None, extraction_result=None):
        self.rows.append((question, supported, answer, latency_ms, extraction_result))


def make_assistant():
    retriever, llm, store = FakeRetriever(), FakeLLM(), FakeStore()
    return FinancialAssistant(retriever, llm, store), retriever, llm, store


def test_pdf_indexing_logs_metadata_and_latency():
    assistant, retriever, _, store = make_assistant()
    chunks = [
        DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text="Meal limit is 1200 INR.", page=1),
        DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text="Claims are due in 15 days.", page=2),
    ]
    with patch("financial_assistant.services.extract_pdf_bytes", return_value=chunks):
        result = assistant.index_pdf("C:/fake/policy.pdf", b"%PDF-test")

    assert result == {
        "source": "policy.pdf",
        "chunks_indexed": 2,
        "pages_extracted": 2,
        "source_type": "pdf",
    }
    assert len(retriever.upserted) == 2
    assert store.rows[0][3] >= 0
    assert store.rows[0][4] == result


def test_pdf_with_no_selectable_text_explains_ocr_limitation():
    assistant, _, _, store = make_assistant()
    with patch("financial_assistant.services.extract_pdf_bytes", return_value=[]):
        with pytest.raises(ValueError, match="not OCR-processed"):
            assistant.index_pdf("scanned.pdf", b"%PDF-test")
    assert store.rows == []


def test_question_uses_retrieved_evidence_and_logs_latency():
    assistant, retriever, _, store = make_assistant()
    result = assistant.ask("What is the meal limit?")
    assert "1200 INR" in result.answer
    assert result.sources == retriever.evidence
    assert store.rows[0][3] >= 0


def test_empty_question_is_rejected():
    assistant, _, _, _ = make_assistant()
    with pytest.raises(ValueError, match="Enter a question"):
        assistant.ask("  ")


def test_image_inspection_uses_vision_and_logs_result():
    assistant, _, llm, store = make_assistant()
    with patch("financial_assistant.services.image_bytes_as_data_url", return_value="data:image/jpeg;base64,abc"):
        answer = assistant.inspect_image(b"fake-image", "Read the receipt total")
    assert answer == "Receipt total: 892.50 INR."
    assert llm.image_calls == [("Read the receipt total", "data:image/jpeg;base64,abc")]
    assert store.rows[0][3] >= 0
