import hashlib
import json
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
    """Chroma-backed retrieval with bounded queries and safer source replacement."""

    MAX_QUERY_RESULTS = 10

    def __init__(self, persist_dir: str, embedder: Embedder, collection_name: str = "financial_policy"):
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = embedder
        self._write_lock = RLock()

    @staticmethod
    def _chunk_id(chunk: DocumentChunk) -> str:
        # Content-sensitive IDs allow new chunks to be written before old IDs are removed.
        identity = json.dumps(
            [chunk.source_type.value, chunk.source, chunk.page or 0, chunk.text],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        # Chroma requires unique IDs per batch; identical duplicate chunks are redundant.
        unique = {self._chunk_id(chunk): chunk for chunk in chunks}
        ids = list(unique)
        chunks = list(unique.values())
        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("embedder returned a different number of vectors than input texts")

        sources = sorted({chunk.source for chunk in chunks})
        metadatas = [
            {"source": chunk.source, "source_type": chunk.source_type.value, "page": chunk.page or 0}
            for chunk in chunks
        ]

        with self._write_lock:
            old_ids_by_source: dict[str, set[str]] = {}
            for source in sources:
                existing = self.collection.get(where={"source": source}, include=["metadatas"])
                old_ids_by_source[source] = set(existing.get("ids", []))

            # Upsert first: if Chroma rejects the write, the previous source is not
            # deliberately deleted first. This is still not a cross-process transaction.
            self.collection.upsert(
                ids=ids,
                documents=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=metadatas,
            )

            new_ids_by_source: dict[str, set[str]] = {}
            for chunk, chunk_id in zip(chunks, ids):
                new_ids_by_source.setdefault(chunk.source, set()).add(chunk_id)

            # Delete stale records only after the replacement write succeeds.
            for source, old_ids in old_ids_by_source.items():
                stale_ids = sorted(old_ids - new_ids_by_source.get(source, set()))
                if stale_ids:
                    self.collection.delete(ids=stale_ids)

    def search(self, question: str, k: int = 4) -> list[RetrievedChunk]:
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
                    source_type=metadata.get("source_type", "pdf"),
                    page=(metadata.get("page") or None),
                )
            )
        return found
