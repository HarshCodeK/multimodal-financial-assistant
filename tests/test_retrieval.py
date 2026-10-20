import pytest

chromadb = pytest.importorskip("chromadb")

from financial_assistant.models import DocumentChunk, SourceType
from financial_assistant.retrieval import ChromaRetriever


class FakeEmbedder:
    def __init__(self):
        self.fail = False

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("simulated embedding failure")
        return [[1.0, 0.0] if "meal" in text.lower() else [0.0, 1.0] for text in texts]


def chunk(text, page=1, source="p.pdf"):
    return DocumentChunk(source=source, source_type=SourceType.pdf, text=text, page=page)


def test_chroma_round_trip(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="test")
    retriever.replace_all([chunk("Meal allowance is 500 INR.")])
    results = retriever.search("meal allowance", k=1)
    assert results[0].text.startswith("Meal allowance")


def test_chunk_ids_are_stable_and_content_sensitive():
    first = chunk("Meal limit 500 INR.")
    same = chunk("Meal limit 500 INR.")
    changed = chunk("Meal limit 750 INR.")
    assert ChromaRetriever._chunk_id(first) == ChromaRetriever._chunk_id(same)
    assert ChromaRetriever._chunk_id(first) != ChromaRetriever._chunk_id(changed)


def test_reindex_replaces_the_previous_document(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="replace-test")
    retriever.replace_all([
        chunk("Meal policy version one.", page=1),
        chunk("Hotel policy old rule.", page=2),
    ])
    retriever.replace_all([chunk("Meal policy version two.", page=1, source="new-policy.pdf")])
    assert retriever.collection.count() == 1
    assert retriever.indexed_sources() == ["new-policy.pdf"]
    result = retriever.search("meal policy", k=2)
    assert len(result) == 1
    assert "version two" in result[0].text


def test_embedding_failure_preserves_previous_index(tmp_path):
    embedder = FakeEmbedder()
    retriever = ChromaRetriever(str(tmp_path), embedder, collection_name="failure-test")
    retriever.replace_all([chunk("Existing policy is still available.")])
    embedder.fail = True

    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        retriever.replace_all([chunk("Replacement policy.")])

    embedder.fail = False
    result = retriever.search("existing policy", k=1)
    assert len(result) == 1
    assert result[0].text == "Existing policy is still available."


def test_clear_index_removes_all_indexed_sources(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="clear-test")
    retriever.replace_all([
        chunk("Meal allowance is 500 INR.", source="old-policy.pdf"),
        chunk("Hotel allowance is 900 INR.", page=2, source="old-policy.pdf"),
    ])

    retriever.clear_index()

    assert retriever.collection.count() == 0
    assert retriever.indexed_sources() == []
    assert retriever.search("meal allowance") == []


def test_empty_collection_returns_no_results(tmp_path):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name="empty-test")
    assert retriever.search("meal allowance") == []


@pytest.mark.parametrize("k", [0, -1, 11, True, 1.5])
def test_invalid_result_count_is_rejected(tmp_path, k):
    retriever = ChromaRetriever(str(tmp_path), FakeEmbedder(), collection_name=f"k-test-{str(k)}")
    with pytest.raises(ValueError, match="k must be an integer"):
        retriever.search("meal allowance", k=k)
