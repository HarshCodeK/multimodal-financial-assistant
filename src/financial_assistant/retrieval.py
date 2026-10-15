import hashlib
from threading import RLock
from typing import Protocol

from .models import DocumentChunk, RetrievedChunk


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class ChromaRetriever:
    """Chroma-backed retrieval with source replacement and bounded queries."""

    MAX_QUERY_RESULTS = 10

    def __init__(self, persist_dir: str, embedder: Embedder, collection_name: str = "financial_policy"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = embedder
        self._write_lock = RLock()

    @staticmethod
    def _chunk_id(chunk: DocumentChunk, ordinal: int) -> str:
        identity = "\0".join((chunk.source, chunk.source_type.value, str(chunk.page or 0), str(ordinal)))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed(texts)
        if len(embeddings) != len(chunks):
            raise ValueError("embedder returned a different number of vectors than input texts")

        ordinals: dict[tuple[str, str, int], int] = {}
        ids = []
        for chunk in chunks:
            key = (chunk.source, chunk.source_type.value, chunk.page or 0)
            ordinal = ordinals.get(key, 0)
            ordinals[key] = ordinal + 1
            ids.append(self._chunk_id(chunk, ordinal))

        sources = sorted({chunk.source for chunk in chunks})
        metadatas = [
            {"source": chunk.source, "source_type": chunk.source_type.value, "page": chunk.page or 0}
            for chunk in chunks
        ]
        with self._write_lock:
            # Compute vectors first so a model error cannot erase the current source.
            for source in sources:
                self.collection.delete(where={"source": source})
            self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def search(self, question: str, k: int = 4) -> list[RetrievedChunk]:
        normalized = question.strip() if isinstance(question, str) else ""
        if not normalized:
            raise ValueError("question cannot be empty")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= self.MAX_QUERY_RESULTS:
            raise ValueError(f"k must be an integer between 1 and {self.MAX_QUERY_RESULTS}")
        vectors = self.embedder.embed([normalized])
        if len(vectors) != 1:
            raise ValueError("embedder must return exactly one vector for a single query")
        result = self.collection.query(query_embeddings=vectors, n_results=k)
        docs = result.get("documents", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []
        metas = result.get("metadatas", [[]])[0] or []
        found = []
        for index, document in enumerate(docs):
            if index >= len(metas) or index >= len(distances):
                continue
            metadata = metas[index] or {}
            found.append(RetrievedChunk(
                text=document,
                distance=float(distances[index]),
                source=str(metadata.get("source", "unknown")),
                source_type=metadata.get("source_type", "pdf"),
                page=(metadata.get("page") or None),
            ))
        return found
