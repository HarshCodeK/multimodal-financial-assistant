"""Local application services for the Streamlit financial assistant."""

import os
from pathlib import Path
from time import perf_counter

from .extract import extract_pdf_bytes, image_bytes_as_data_url
from .llm import GroqClient
from .models import DocumentChunk, SourceType
from .retrieval import ChromaRetriever, SentenceTransformerEmbedder
from .store import InteractionStore


class FinancialAssistant:
    """Coordinates PDF indexing, grounded Q&A, image analysis, and interaction logs."""

    def __init__(self, retriever, llm, store):
        self.retriever = retriever
        self.llm = llm
        self.store = store

    def index_pdf(self, filename: str, raw: bytes) -> dict:
        started = perf_counter()
        safe_name = Path(filename).name
        chunks = extract_pdf_bytes(raw, source_name=safe_name)
        if not chunks:
            raise ValueError(
                "This PDF contains no selectable text. Scanned PDFs are not OCR-processed yet."
            )
        self.retriever.upsert(chunks)
        pages = len({chunk.page for chunk in chunks})
        result = {
            "source": safe_name,
            "chunks_indexed": len(chunks),
            "pages_extracted": pages,
            "source_type": "pdf",
        }
        self.store.record(
            question=f"PDF extraction: {safe_name}",
            supported=True,
            answer=f"Indexed {len(chunks)} text chunks across {pages} pages.",
            latency_ms=round((perf_counter() - started) * 1000, 2),
            extraction_result=result,
        )
        return result

    def ask(self, question: str):
        normalized = question.strip()
        if not normalized:
            raise ValueError("Enter a question.")
        started = perf_counter()
        evidence = self.retriever.search(normalized)
        result = self.llm.answer(normalized, evidence)
        self.store.record(
            normalized,
            result.supported,
            result.answer,
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )
        return result

    def inspect_image(self, raw: bytes, question: str):
        normalized = question.strip() or (
            "Read the financial information visible in this image. Extract legible "
            "merchant/date/items/totals where possible, distinguish uncertain text, "
            "and do not guess unreadable numbers."
        )
        started = perf_counter()
        data_url = image_bytes_as_data_url(raw)
        answer = self.llm.describe_image(normalized, data_url)
        self.store.record(
            normalized,
            True,
            answer,
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )
        return answer


def create_default_assistant(project_root: str | Path | None = None) -> FinancialAssistant:
    """Build the local persistent services used by the Streamlit UI."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    chroma_dir = os.getenv("CHROMA_DIR", str(root / ".chroma"))
    sqlite_path = os.getenv("SQLITE_PATH", str(root / "data" / "interactions.db"))
    embedder = SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    retriever = ChromaRetriever(chroma_dir, embedder)
    store = InteractionStore(sqlite_path)
    if retriever.collection.count() == 0:
        sample_path = root / "data" / "sample_policy.txt"
        if sample_path.exists():
            sample_text = sample_path.read_text(encoding="utf-8").strip()
            if sample_text:
                retriever.upsert([
                    DocumentChunk(
                        source=sample_path.name,
                        source_type=SourceType.text,
                        text=sample_text,
                    )
                ])
    return FinancialAssistant(retriever, GroqClient(), store)
