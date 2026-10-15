import pytest

chromadb = pytest.importorskip("chromadb")
from financial_assistant.models import DocumentChunk, SourceType
from financial_assistant.retrieval import ChromaRetriever


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] if "meal" in t.lower() else [0.0, 1.0] for t in texts]


def test_chroma_round_trip(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="round_trip")
    retriever.upsert([DocumentChunk(source="p.pdf", source_type=SourceType.pdf, text="Meal allowance is 500 INR.", page=1)])
    results = retriever.search("meal allowance", k=1)
    assert results[0].text.startswith("Meal allowance")
    assert results[0].page == 1


def test_reingesting_source_replaces_old_chunks(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="replace_source")
    retriever.upsert([
        DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text="Old allowance is 100.", page=1),
        DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text="Old travel policy.", page=2),
    ])
    retriever.upsert([DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text="New allowance is 250.", page=1)])
    result = retriever.search("allowance", k=10)
    assert len(result) == 1
    assert result[0].text == "New allowance is 250."


def test_search_rejects_empty_questions_and_unbounded_k(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="query_bounds")
    with pytest.raises(ValueError, match="empty"):
        retriever.search("  ")
    for invalid in (0, 11, True, 1.5):
        with pytest.raises(ValueError, match="k"):
            retriever.search("policy", k=invalid)


def test_embedder_must_return_one_vector_per_input(tmp_path):
    class BrokenEmbedder:
        def embed(self, texts):
            return []

    retriever = ChromaRetriever(str(tmp_path), BrokenEmbedder(), collection_name="bad_embedder")
    with pytest.raises(ValueError, match="different number"):
        retriever.upsert([DocumentChunk(source="p.pdf", source_type=SourceType.pdf, text="Policy.", page=1)])
