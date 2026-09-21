import os
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import POLICY_DOCS_PATH, TOP_K

_documents = []
_vectorizer = None
_matrix = None

def build_knowledge_base():
    global _documents, _vectorizer, _matrix
    documents = []
    for fname in sorted(os.listdir(POLICY_DOCS_PATH)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(POLICY_DOCS_PATH, fname), "r", encoding="utf-8") as f:
            text = f.read().strip()
        words = text.split()
        for i in range(0, len(words), 100):
            chunk = " ".join(words[i:i + 100]).strip()
            if chunk:
                documents.append({"source": fname, "text": chunk})
    if not documents:
        raise RuntimeError("No policy documents found.")
    _documents = documents
    _vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    _matrix = _vectorizer.fit_transform([d["text"] for d in documents])
    return len(documents)

def retrieve_policy_context(query: str, k: int = TOP_K) -> list[str]:
    global _vectorizer, _matrix
    if _vectorizer is None or _matrix is None:
        build_knowledge_base()
    query_vector = _vectorizer.transform([query])
    scores = (_matrix @ query_vector.T).toarray().ravel()
    indexes = scores.argsort()[::-1][:k]
    return [_documents[i]["text"] for i in indexes if scores[i] > 0]
