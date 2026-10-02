"""Policy retrieval over ChromaDB with local sentence-transformer embeddings.

Why embeddings rather than keyword matching: the question is "why was this
charge deducted" but the policy says "fees assessed after the due date". Those
share almost no words. Keyword search needs the query to use the document's
vocabulary; embeddings do not.

Why a LOCAL embedding model: `all-MiniLM-L6-v2` is 22M parameters and runs on
CPU. An embedding API would be marginally better and would make retrieval
require a network round-trip for something deterministic and free offline.

Why ChromaDB: it persists to disk, so the index survives a restart, and the
query API returns documents with their source file attached -- which is what
lets an answer cite where it came from.
"""
import os

from .config import CHROMA_DIR, CHUNK_WORDS, POLICY_DIR, TOP_K

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "policy_docs"

_client = None
_embedder = None


def _get_client():
    global _client
    if _client is None:
        import chromadb
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def chunk(text: str, size: int = CHUNK_WORDS):
    """Split on word count."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def ingest(directory: str = None) -> int:
    """Embed every .txt in the policy directory and store the chunks.

    Chunk ids are deterministic (`filename::index`), so re-ingesting replaces
    rather than duplicates, and deleting first drops chunks from files that were
    edited or removed.
    """
    directory = directory or POLICY_DIR
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"policy directory not found: {directory}")

    chunks, ids, metadatas = [], [], []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            for i, piece in enumerate(chunk(f.read().strip())):
                if piece:
                    chunks.append(piece)
                    ids.append(f"{name}::{i}")
                    metadatas.append({"source": name, "chunk": i})

    if not chunks:
        raise RuntimeError(f"no .txt policy documents in {directory}")

    coll = _get_client().get_or_create_collection(COLLECTION)
    for cid in ids:
        try:
            coll.delete(ids=[cid])
        except Exception:
            pass
    coll.add(
        embeddings=_get_embedder().encode(chunks).tolist(),
        documents=chunks, ids=ids, metadatas=metadatas,
    )
    return len(chunks)


def size() -> int:
    try:
        return _get_client().get_or_create_collection(COLLECTION).count()
    except Exception:
        return 0


def ensure_index() -> int:
    """Build the index on first use. Returns the chunk count."""
    return size() or ingest()


def search(query: str, k: int = TOP_K):
    """Top-k policy chunks, each with the filename it came from."""
    coll = _get_client().get_or_create_collection(COLLECTION)
    if coll.count() == 0:
        ingest()
        coll = _get_client().get_or_create_collection(COLLECTION)
    if coll.count() == 0:
        return []
    res = coll.query(
        query_embeddings=_get_embedder().encode([query]).tolist(),
        n_results=min(k, coll.count()),
    )
    return [
        {"text": doc, "source": meta.get("source", "unknown")}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]


def stats() -> dict:
    return {"chunks": size(), "embeddings": EMBEDDING_MODEL, "store": "ChromaDB"}
