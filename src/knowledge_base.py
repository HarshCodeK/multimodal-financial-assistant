import os
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import POLICY_DOCS_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL, COLLECTION_NAME

_embedding_model = None
_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

def build_knowledge_base():
    model = _get_embedding_model()
    collection = _get_collection()
    all_chunks = []
    all_ids = []
    doc_id = 0

    for fname in os.listdir(POLICY_DOCS_PATH):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(POLICY_DOCS_PATH, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        chunks = []
        for i in range(0, len(words), 100):
            chunk = " ".join(words[i:i+100])
            chunks.append(chunk)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"{fname}_{doc_id}")
            doc_id += 1

    embeddings = model.encode(all_chunks).tolist()
    collection.add(embeddings=embeddings, documents=all_chunks, ids=all_ids)
    print(f"Stored {len(all_chunks)} chunks in the knowledge base.")

def retrieve_policy_context(query: str, k: int = 3) -> list[str]:
    model = _get_embedding_model()
    collection = _get_collection()
    q_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k)
    if results and results["documents"]:
        return results["documents"][0]
    return []
