"""Coordinate PDF extraction, chunk indexing, semantic retrieval, and grounded answers."""

import os
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from .costing import DEFAULT_TEXT_RATES, token_cost_usd
from .extract import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, extract_pdf_bytes
from .generation import GroundedAnswerGenerator
from .models import Answer
from .retrieval import ChromaRetriever, SentenceTransformerEmbedder
from .store import InteractionStore


def _estimate_original_word_count(chunks) -> int:
    """Estimate document words from per-page overlapping chunks, removing expected overlap."""
    by_page = defaultdict(list)
    for chunk in chunks:
        by_page[chunk.page].append(chunk)
    total = 0
    for page_chunks in by_page.values():
        observed = sum(len(chunk.text.split()) for chunk in page_chunks)
        total += max(0, observed - CHUNK_OVERLAP_WORDS * max(0, len(page_chunks) - 1))
    return total


class FinancialAssistant:
    """Single-document RAG service. A successful index operation replaces the previous PDF."""

    def __init__(self, retriever, store, generator=None):
        self.retriever = retriever
        self.store = store
        self.generator = generator

    def indexed_sources(self) -> list[str]:
        return self.retriever.indexed_sources()

    def clear_index(self) -> None:
        """Clear the active PDF index without touching uploaded/original files."""
        self.retriever.clear_index()

    def index_pdf(self, filename: str, raw: bytes) -> dict:
        started = perf_counter()
        safe_name = Path(filename).name
        chunks = extract_pdf_bytes(raw, source_name=safe_name)
        if not chunks:
            raise ValueError(
                "No readable text was found in this PDF, even after local OCR. Try a clearer PDF or OCR the file separately."
            )

        # Embed first and replace the old collection only after extraction and embedding succeed.
        self.retriever.replace_all(chunks)
        pages = len({chunk.page for chunk in chunks if chunk.page is not None})
        total_words = _estimate_original_word_count(chunks)
        embedder = getattr(self.retriever, "embedder", None)
        model_name = getattr(embedder, "model_name", os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        dimensions = getattr(embedder, "dimensions", None)
        vector_preview = []
        if embedder is not None:
            vector = embedder.embed([chunks[0].text])[0]
            dimensions = dimensions or len(vector)
            vector_preview = [round(float(value), 6) for value in vector[:8]]

        chunk_details = [
            {
                "number": index,
                "page": chunk.page,
                "words": len(chunk.text.split()),
                "text": chunk.text,
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
        result = {
            "source": safe_name,
            "chunks_indexed": len(chunks),
            "pages_extracted": pages,
            "source_type": "pdf",
            "chunk_words": CHUNK_WORDS,
            "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
            "estimated_document_words": total_words,
            "estimated_document_tokens": round(total_words * 1.33),
            "embedding_model": model_name,
            "embedding_dimensions": dimensions,
            "embedding_preview": vector_preview,
            "chunks": chunk_details,
        }
        self.store.record(
            question=f"PDF extraction and indexing: {safe_name}",
            supported=True,
            answer=f"Indexed {len(chunks)} chunks across {pages} pages.",
            latency_ms=round((perf_counter() - started) * 1000, 2),
            extraction_result={
                key: value for key, value in result.items() if key != "chunks"
            },
        )
        return result

    def ask(self, question: str) -> Answer:
        normalized = question.strip()
        if not normalized:
            raise ValueError("Enter a question.")
        started = perf_counter()
        evidence = self.retriever.search(normalized, k=5)
        if self.generator is None:
            self.generator = GroundedAnswerGenerator()
        generated = self.generator.answer(normalized, evidence)
        result = Answer(
            supported=generated["supported"],
            answer=generated["answer"],
            sources=generated.get("sources", evidence),
            citations=generated.get("citations", []),
            model=generated.get("model"),
            prompt_tokens=generated.get("prompt_tokens"),
            completion_tokens=generated.get("completion_tokens"),
            context_word_count=generated.get("context_word_count", 0),
            context_token_estimate=generated.get("context_token_estimate", 0),
            prompt_preview=generated.get("prompt_preview", ""),
        )
        usage_cost = (
            token_cost_usd(result.prompt_tokens, result.completion_tokens, DEFAULT_TEXT_RATES)
            if result.prompt_tokens is not None and result.completion_tokens is not None
            else None
        )
        prompt_length = len(result.prompt_preview)
        self.store.record(
            normalized,
            result.supported,
            result.answer,
            latency_ms=round((perf_counter() - started) * 1000, 2),
            workflow="pdf_rag",
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            estimated_cost_usd=usage_cost,
            input_chars=prompt_length,
            input_words=len(result.prompt_preview.split()),
        )
        return result


def create_default_assistant(project_root: str | Path | None = None) -> FinancialAssistant:
    """Create local extraction/retrieval services; load the LLM only when a question is asked."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    chroma_dir = os.getenv("CHROMA_DIR", str(root / ".chroma"))
    sqlite_path = os.getenv("SQLITE_PATH", str(root / "data" / "interactions.db"))
    embedder = SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    retriever = ChromaRetriever(
        chroma_dir,
        embedder,
        collection_name="financial_assistant_pdf_rag_v5",
    )
    store = InteractionStore(sqlite_path)
    # No bundled data is indexed. The collection changes only when the user explicitly indexes a PDF.
    return FinancialAssistant(retriever, store)
