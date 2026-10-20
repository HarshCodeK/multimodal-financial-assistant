import hashlib
import json
from threading import RLock
from typing import Protocol

from .models import DocumentChunk, RetrievedChunk, SourceType


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Create semantic vectors with a pretrained Sentence Transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimensions(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class ChromaRetriever:
    """Chroma-backed semantic retrieval for one explicitly indexed PDF at a time."""

    MAX_QUERY_RESULTS = 10

    def __init__(self, persist_dir: str, embedder: Embedder, collection_name: str = "financial_policy"):
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder
        self._write_lock = RLock()

    @staticmethod
    def _chunk_id(chunk: DocumentChunk) -> str:
        identity = json.dumps(
            [chunk.source_type.value, chunk.source, chunk.page or 0, chunk.text],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def replace_all(self, chunks: list[DocumentChunk]) -> None:
        """Replace the index with the one PDF the user just explicitly indexed."""
        if not chunks:
            raise ValueError("There are no text chunks to index.")

        unique = {self._chunk_id(chunk): chunk for chunk in chunks}
        ids = list(unique)
        chunks = list(unique.values())
        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("embedder returned a different number of vectors than input texts")

        metadatas = [
            {"source": chunk.source, "source_type": chunk.source_type.value, "page": chunk.page or 0}
            for chunk in chunks
        ]

        with self._write_lock:
            existing = self.collection.get(include=["metadatas"])
            old_ids = existing.get("ids", [])
            if old_ids:
                self.collection.delete(ids=old_ids)
            self.collection.add(
                ids=ids,
                documents=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def clear_index(self) -> None:
        """Delete all indexed chunks from the active PDF collection."""
        with self._write_lock:
            existing = self.collection.get(include=["metadatas"])
            ids = existing.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)

    def indexed_sources(self) -> list[str]:
        result = self.collection.get(include=["metadatas"])
        return sorted({
            str(metadata.get("source", "unknown"))
            for metadata in (result.get("metadatas") or [])
            if metadata
        })

    def search(self, question: str, k: int = 5) -> list[RetrievedChunk]:
        normalized = question.strip() if isinstance(question, str) else ""
        if not normalized:
            raise ValueError("question cannot be empty")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= self.MAX_QUERY_RESULTS:
            raise ValueError(f"k must be an integer between 1 and {self.MAX_QUERY_RESULTS}")

        count = self.collection.count()
        if count == 0:
            return []

        vectors = self.embedder.embed([normalized])
        if len(vectors) != 1:
            raise ValueError("embedder must return exactly one vector for a single query")
        result = self.collection.query(query_embeddings=vectors, n_results=min(k, count))
        docs = result.get("documents", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []
        metas = result.get("metadatas", [[]])[0] or []
        found = []
        for index, document in enumerate(docs):
            if index >= len(metas) or index >= len(distances):
                continue
            metadata = metas[index] or {}
            found.append(
                RetrievedChunk(
                    text=document,
                    distance=float(distances[index]),
                    source=str(metadata.get("source", "unknown")),
                    source_type=SourceType(metadata.get("source_type", "pdf")),
                    page=(metadata.get("page") or None),
                )
            )
        return found
