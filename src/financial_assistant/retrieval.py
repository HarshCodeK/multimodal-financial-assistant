import hashlib
from collections import defaultdict
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
    def __init__(self, persist_dir: str, embedder: Embedder, collection_name: str = "financial_policy"):
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = embedder

    @staticmethod
    def _chunk_id(chunk: DocumentChunk) -> str:
        # Stable across process runs and repeated ingestion of the same content.
        identity = "\\0".join(
            (chunk.source, chunk.source_type.value, str(chunk.page or 0), chunk.text)
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        # Compute embeddings before mutating the index so an embedding failure
        # cannot delete the existing source's indexed content.
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed(texts)
        if len(embeddings) != len(chunks):
            raise ValueError("embedder returned a different number of vectors than input texts")

        grouped: dict[str, list[tuple[int, DocumentChunk]]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            grouped[chunk.source].append((index, chunk))

        # Re-ingestion replaces a source's old chunks instead of leaving stale
        # passages behind. Each source is replaced independently.
        for source, indexed_chunks in grouped.items():
            self.collection.delete(where={"source": source})
            self.collection.upsert(
                ids=[self._chunk_id(chunk) for _, chunk in indexed_chunks],
                documents=[chunk.text for _, chunk in indexed_chunks],
                embeddings=[embeddings[index] for index, _ in indexed_chunks],
                metadatas=[
                    {
                        "source": chunk.source,
                        "source_type": chunk.source_type.value,
                        "page": chunk.page or 0,
                    }
                    for _, chunk in indexed_chunks
                ],
            )

    def search(self, question: str, k: int = 4) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be at least 1")
        result = self.collection.query(query_embeddings=self.embedder.embed([question]), n_results=k)
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        return [
            RetrievedChunk(
                text=document,
                distance=float(distances[index]),
                source=metas[index]["source"],
                source_type=metas[index]["source_type"],
                page=(metas[index].get("page") or None),
            )
            for index, document in enumerate(docs)
        ]
