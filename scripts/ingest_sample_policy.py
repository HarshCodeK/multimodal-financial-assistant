"""Index the included sample policy into the configured Chroma collection."""
import os
from pathlib import Path
from financial_assistant.models import DocumentChunk, SourceType
from financial_assistant.retrieval import ChromaRetriever, SentenceTransformerEmbedder
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "data" / "sample_policy.txt"
    persist_dir = os.getenv("CHROMA_DIR", str(ROOT / ".chroma"))
    retriever = ChromaRetriever(persist_dir, SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")))
    retriever.upsert([DocumentChunk(source=source.name, source_type=SourceType.text, text=source.read_text(encoding="utf-8").strip())])
    print(f"Indexed sample policy into {persist_dir}")


if __name__ == "__main__":
    main()
