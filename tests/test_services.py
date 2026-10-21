from unittest.mock import patch

import pytest

from financial_assistant.models import DocumentChunk, RetrievedChunk, SourceType
from financial_assistant.services import FinancialAssistant, _estimate_original_word_count


class FakeRetriever:
    def __init__(self):
        self.indexed = []
        self.evidence = [
            RetrievedChunk(
                source="policy.pdf",
                source_type=SourceType.pdf,
                text="Meals are reimbursed up to 1200 INR per day.",
                page=2,
                distance=0.1,
            )
        ]

    def replace_all(self, chunks):
        self.indexed = list(chunks)

    def indexed_sources(self):
        return sorted({chunk.source for chunk in self.indexed})

    def search(self, question, k=5):
        return self.evidence[:k]


class FakeStore:
    def __init__(self):
        self.rows = []

    def record(self, question, supported, answer, latency_ms=None, extraction_result=None, **kwargs):
        self.rows.append((question, supported, answer, latency_ms, extraction_result, kwargs))


class FakeGenerator:
    def __init__(self, supported=True):
        self.supported = supported
        self.questions = []
        self.evidence = None

    def answer(self, question, chunks):
        self.questions.append(question)
        self.evidence = chunks
        return {
            "supported": self.supported,
            "answer": "The meal limit is INR 1,200 per employee per day." if self.supported else "The document does not specify this.",
            "citations": ["S1"] if self.supported else [],
            "sources": chunks,
            "model": "fake-test-model",
            "prompt_tokens": 120,
            "completion_tokens": 18,
            "context_word_count": 50,
            "context_token_estimate": 67,
            "prompt_preview": "QUESTION: ... RETRIEVED EVIDENCE: ...",
        }


def make_assistant(generator=None):
    retriever, store = FakeRetriever(), FakeStore()
    return FinancialAssistant(retriever, store, generator), retriever, store


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
    assert len(retriever.indexed) == 2
    assert result["chunk_overlap_words"] == 16
    assert result["chunk_words"] == 80
    assert store.rows[0][3] >= 0
    assert store.rows[0][4]["chunks_indexed"] == 2


def test_pdf_without_readable_text_gives_actionable_error():
    assistant, _, store = make_assistant()
    with patch("financial_assistant.services.extract_pdf_bytes", return_value=[]):
        with pytest.raises(ValueError, match="No readable text"):
            assistant.index_pdf("scanned.pdf", b"%PDF-test")
    assert store.rows == []


def test_question_uses_llm_and_returns_direct_answer_with_citations():
    generator = FakeGenerator()
    assistant, retriever, store = make_assistant(generator)
    result = assistant.ask("What is the meal limit?")

    assert result.supported is True
    assert result.answer == "The meal limit is INR 1,200 per employee per day."
    assert result.citations == ["S1"]
    assert result.sources == retriever.evidence
    assert generator.questions == ["What is the meal limit?"]
    assert generator.evidence == retriever.evidence
    assert result.prompt_tokens == 120
    assert result.prompt_preview
    assert store.rows[0][3] >= 0
    assert store.rows[0][5]["workflow"] == "pdf_rag"
    assert store.rows[0][5]["prompt_tokens"] == 120
    assert store.rows[0][5]["estimated_cost_usd"] is not None


def test_question_passes_only_retrieved_evidence_to_generator():
    generator = FakeGenerator()
    assistant, retriever, _ = make_assistant(generator)

    assistant.ask("What is the meal limit?")

    assert generator.evidence == retriever.evidence
    assert all(source.source == "policy.pdf" for source in generator.evidence)
    assert all(source.text for source in generator.evidence)


def test_question_abstains_when_generator_cannot_support_answer():
    generator = FakeGenerator(supported=False)
    assistant, _, _ = make_assistant(generator)
    result = assistant.ask("What is the international tax policy?")

    assert result.supported is False
    assert result.citations == []
    assert "does not specify" in result.answer


def test_empty_question_is_rejected():
    assistant, _, _ = make_assistant(FakeGenerator())
    with pytest.raises(ValueError, match="Enter a question"):
        assistant.ask("  ")


def test_estimated_original_word_count_removes_chunk_overlap():
    chunks = [
        DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text=" ".join(["word"] * 80), page=1),
        DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text=" ".join(["word"] * 36), page=1),
    ]
    assert _estimate_original_word_count(chunks) == 100


def test_default_assistant_does_not_silently_index_bundled_sample_policy(tmp_path):
    from financial_assistant import services

    fake_retriever = FakeRetriever()
    fake_store = FakeStore()

    with patch.object(services, "SentenceTransformerEmbedder", return_value=object()), \
         patch.object(services, "ChromaRetriever", return_value=fake_retriever) as retriever_factory, \
         patch.object(services, "InteractionStore", return_value=fake_store):
        assistant = services.create_default_assistant(tmp_path)

    retriever_factory.assert_called_once()
    assert retriever_factory.call_args.kwargs["collection_name"] == "financial_assistant_pdf_rag_v5"
    assert assistant.indexed_sources() == []
    assert fake_retriever.indexed == []
