import hashlib
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
        stable_key = "\x1f".join((chunk.source_type.value, chunk.source, str(chunk.page or 0), chunk.text))
        return hashlib.sha256(stable_key.encode("utf-8")).hexdigest()

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        # Remove duplicate IDs inside one batch, which can occur with repeated identical chunks.
        unique = {self._chunk_id(chunk): chunk for chunk in chunks}
        ids = list(unique)
        chunks = list(unique.values())
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        new_ids_by_source: dict[str, set[str]] = {}
        for chunk, chunk_id in zip(chunks, ids):
            new_ids_by_source.setdefault(chunk.source, set()).add(chunk_id)
        old_ids_by_source: dict[str, set[str]] = {}
        for source in new_ids_by_source:
            existing = self.collection.get(where={"source": source}, include=["metadatas"])
            old_ids_by_source[source] = set(existing.get("ids", []))
        self.collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=vectors,
            metadatas=[{"source": chunk.source, "source_type": chunk.source_type.value, "page": chunk.page or 0} for chunk in chunks],
        )
        # Clean stale chunks only after the new content was embedded and upserted.
        for source, old_ids in old_ids_by_source.items():
            stale_ids = sorted(old_ids - new_ids_by_source[source])
            if stale_ids:
                self.collection.delete(ids=stale_ids)

    def search(self, question: str, k: int = 4) -> list[RetrievedChunk]:
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(query_embeddings=self.embedder.embed([question]), n_results=min(k, count))
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        return [RetrievedChunk(text=document, distance=float(distances[i]), source=metas[i]["source"], source_type=metas[i]["source_type"], page=(metas[i].get("page") or None)) for i, document in enumerate(docs)]
