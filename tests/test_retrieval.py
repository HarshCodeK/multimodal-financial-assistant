import pytest
chromadb = pytest.importorskip("chromadb")
from financial_assistant.models import DocumentChunk, SourceType
from financial_assistant.retrieval import ChromaRetriever


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] if "meal" in text.lower() else [0.0, 1.0] for text in texts]


def test_chroma_round_trip(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="test")
    retriever.upsert([DocumentChunk(source="p.pdf", source_type=SourceType.pdf, text="Meal allowance is 500 INR.", page=1)])
    results = retriever.search("meal allowance", k=1)
    assert results[0].text.startswith("Meal allowance")


def test_chunk_ids_are_stable_and_content_sensitive():
    first = DocumentChunk(source="p.pdf", source_type=SourceType.pdf, text="Meal limit 500 INR.", page=1)
    same = DocumentChunk(source="p.pdf", source_type=SourceType.pdf, text="Meal limit 500 INR.", page=1)
    changed = DocumentChunk(source="p.pdf", source_type=SourceType.pdf, text="Meal limit 750 INR.", page=1)
    assert ChromaRetriever._chunk_id(first) == ChromaRetriever._chunk_id(same)
    assert ChromaRetriever._chunk_id(first) != ChromaRetriever._chunk_id(changed)


def test_reindex_removes_stale_chunks_from_same_source(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="replace-test")
    retriever.upsert([
        DocumentChunk(source="p.pdf", source_type=SourceType.pdf, text="Meal policy version one.", page=1),
        DocumentChunk(source="p.pdf", source_type=SourceType.pdf, text="Hotel policy old rule.", page=2),
    ])
    assert retriever.collection.count() == 2
    retriever.upsert([DocumentChunk(source="p.pdf", source_type=SourceType.pdf, text="Meal policy version two.", page=1)])
    assert retriever.collection.count() == 1
    result = retriever.search("meal policy", k=2)
    assert len(result) == 1
    assert "version two" in result[0].text
