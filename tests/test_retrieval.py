import pytest

chromadb = pytest.importorskip("chromadb")

from financial_assistant.models import DocumentChunk, SourceType
from financial_assistant.retrieval import ChromaRetriever


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] if "meal" in text.lower() else [0.0, 1.0] for text in texts]


def test_chroma_round_trip(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="test")
    retriever.upsert([
        DocumentChunk(
            source="p.pdf",
            source_type=SourceType.pdf,
            text="Meal allowance is 500 INR.",
            page=1,
        )
    ])
    results = retriever.search("meal allowance", k=1)
    assert results[0].text.startswith("Meal allowance")


def test_reingesting_source_replaces_stale_chunks(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="replace_test")
    retriever.upsert([
        DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text="Old limit is 300 INR.", page=1),
        DocumentChunk(source="other.pdf", source_type=SourceType.pdf, text="Meals cover lunch.", page=2),
    ])

    retriever.upsert([
        DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text="New limit is 500 INR.", page=1),
    ])

    assert retriever.collection.count() == 2
    results = retriever.search("new limit", k=2)
    assert any("New limit is 500 INR." == result.text for result in results)
    assert all("Old limit is 300 INR." != result.text for result in results)


def test_invalid_k_is_rejected(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="invalid_k_test")
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.search("meal", k=0)
