from unittest.mock import patch

import pytest

from financial_assistant.models import DocumentChunk, RetrievedChunk, SourceType
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

    def search(self, question, k=4):
        return self.evidence


class FakeStore:
    def __init__(self):
        self.rows = []

    def record(self, question, supported, answer, latency_ms=None, extraction_result=None):
        self.rows.append((question, supported, answer, latency_ms, extraction_result))


def make_assistant():
    retriever, store = FakeRetriever(), FakeStore()
    return FinancialAssistant(retriever, store), retriever, store


def test_pdf_indexing_logs_metadata_and_latency():
    assistant, retriever, store = make_assistant()
    chunks = [
        DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text="Meal limit is 1200 INR.", page=1),
        DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text="Claims are due in 15 days.", page=2),
    ]
    with patch("financial_assistant.services.extract_pdf_bytes", return_value=chunks):
        result = assistant.index_pdf("C:/fake/policy.pdf", b"%PDF-test")

    assert result["chunks_indexed"] == 2
    assert result["pages_extracted"] == 2
    assert len(retriever.upserted) == 2
    assert store.rows[0][3] >= 0
    assert store.rows[0][4] == result


def test_pdf_without_readable_text_gives_actionable_error():
    assistant, _, store = make_assistant()
    with patch("financial_assistant.services.extract_pdf_bytes", return_value=[]):
        with pytest.raises(ValueError, match="No readable text"):
            assistant.index_pdf("scanned.pdf", b"%PDF-test")
    assert store.rows == []


def test_question_returns_retrieved_evidence_without_external_api():
    assistant, retriever, store = make_assistant()
    result = assistant.ask("What is the meal limit?")
    assert result.supported is True
    assert "1200 INR" in result.answer
    assert result.sources == retriever.evidence
    assert store.rows[0][3] >= 0


def test_irrelevant_question_does_not_claim_support():
    assistant, retriever, _ = make_assistant()
    retriever.evidence[0] = retriever.evidence[0].model_copy(update={"distance": 0.9})
    result = assistant.ask("What is the international tax policy?")
    assert result.supported is False
    assert result.sources == []


def test_empty_question_is_rejected():
    assistant, _, _ = make_assistant()
    with pytest.raises(ValueError, match="Enter a question"):
        assistant.ask("  ")


def test_image_inspection_uses_local_ocr_and_logs_result():
    assistant, _, store = make_assistant()
    with patch("financial_assistant.services.image_bytes_to_text", return_value="TOTAL INR 892.50"):
        answer = assistant.inspect_image(b"fake-image", "Read the receipt total")
    assert "TOTAL INR 892.50" in answer
    assert store.rows[0][3] >= 0
