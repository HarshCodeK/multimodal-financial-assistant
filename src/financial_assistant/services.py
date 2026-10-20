"""Small local services: OCR, overlapping text chunks, embeddings, and evidence retrieval."""

import os
import re
from pathlib import Path
from time import perf_counter

from .extract import extract_pdf_bytes, image_bytes_to_text, chunk_text
from .models import Answer, DocumentChunk, RetrievedChunk, SourceType
from .retrieval import ChromaRetriever, SentenceTransformerEmbedder
from .store import InteractionStore


class FinancialAssistant:
    """Coordinates local document indexing, retrieval, and SQLite logging."""

    def __init__(self, retriever, store):
        self.retriever = retriever
        self.store = store

    def index_pdf(self, filename: str, raw: bytes) -> dict:
        started = perf_counter()
        safe_name = Path(filename).name
        chunks = extract_pdf_bytes(raw, source_name=safe_name)
        if not chunks:
            raise ValueError(
                "No readable text was found. If this is a scanned PDF, check that RapidOCR installed correctly."
            )
        self.retriever.upsert(chunks)
        pages = len({chunk.page for chunk in chunks if chunk.page is not None})
        result = {
            "source": safe_name,
            "chunks_indexed": len(chunks),
            "pages_extracted": pages,
            "source_type": "pdf",
        }
        self.store.record(
            question=f"PDF extraction: {safe_name}",
            supported=True,
            answer=f"Indexed {len(chunks)} overlapping text chunks across {pages} pages.",
            latency_ms=round((perf_counter() - started) * 1000, 2),
            extraction_result=result,
        )
        return result

    def ask(self, question: str) -> Answer:
        normalized = question.strip()
        if not normalized:
            raise ValueError("Enter a question.")
        started = perf_counter()
        evidence = self.retriever.search(normalized, k=5)
        # Cosine distance is 0 for identical vectors. Avoid presenting unrelated
        # nearest-neighbours as evidence when the query is outside the corpus.
        relevant = [item for item in evidence if item.distance <= 0.75]
        if relevant:
            answer_text = "Most relevant passages from your indexed documents:\n\n" + "\n\n".join(
                f"[{item.source}" + (f", page {item.page}" if item.page else "") + f"]\n{item.text}"
                for item in relevant[:3]
            )
            result = Answer(supported=True, answer=answer_text, sources=relevant)
        else:
            result = Answer(
                supported=False,
                answer="I couldn't find a sufficiently close passage in the indexed documents. Try a more specific question or index the relevant policy first.",
                sources=[],
            )
        self.store.record(
            normalized,
            result.supported,
            result.answer,
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )
        return result

    def inspect_image(self, raw: bytes, question: str = "") -> str:
        """Compatibility helper: OCR an image and optionally answer from its text."""
        started = perf_counter()
        extracted = image_bytes_to_text(raw)
        if question.strip() and extracted:
            result = answer_image_question(question, extracted)
            answer = result["answer"]
        else:
            answer = extracted or "No readable text was detected. Try a sharper, brighter image with the receipt filling more of the frame."
        self.store.record(
            question=question.strip() or "OCR image extraction",
            supported=bool(extracted),
            answer=answer,
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )
        return answer


def answer_image_question(question: str, ocr_text: str) -> dict:
    """Select relevant OCR line windows using transparent lexical matching; no LLM is used."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a question about the image.")
    if not isinstance(ocr_text, str) or not ocr_text.strip():
        return {
            "supported": False,
            "answer": "No readable text is available. Extract text from the image first.",
            "evidence": [],
        }

    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did",
        "do", "does", "for", "from", "how", "i", "in", "is", "it", "me", "my",
        "of", "on", "or", "please", "show", "tell", "that", "the", "this", "to",
        "was", "what", "when", "where", "which", "who", "why", "with", "would",
        "you", "your", "much", "value",
    }
    token_pattern = re.compile(r"[a-zA-Z]{2,}|\d+(?:[.,:/-]\d+)*")
    aliases = {
        "price": {"price", "cost", "amount", "total", "inr", "rs", "rupees", "₹"},
        "cost": {"price", "cost", "amount", "total", "inr", "rs", "rupees", "₹"},
        "amount": {"price", "cost", "amount", "total", "inr", "rs", "rupees", "₹"},
        "total": {"price", "cost", "amount", "total", "inr", "rs", "rupees", "₹"},
        "robotics": {"robotics", "robot", "kit"},
        "kit": {"kit", "product", "item"},
        "product": {"product", "item", "kit"},
        "item": {"product", "item", "kit"},
        "date": {"date", "dated", "day"},
        "merchant": {"merchant", "store", "seller", "shop"},
    }

    def terms(value: str) -> set[str]:
        found = {
            token.lower().strip(".,:/-")
            for token in token_pattern.findall(value)
            if token.lower() not in stopwords
        }
        expanded = set(found)
        for token in found:
            expanded.update(aliases.get(token, set()))
        return expanded

    query_terms = terms(question)
    # For generic price questions, avoid requiring a literal word such as "price"
    # to appear on the receipt; the product name and adjacent amount line may be enough.
    if not query_terms:
        return {"supported": False, "answer": "Please ask a more specific question.", "evidence": []}

    lines = [line.strip() for line in re.split(r"[\r\n]+|(?<=[.!?])\s+", ocr_text) if line.strip()]
    candidates = []
    # Receipt OCR often puts an item name and its price on adjacent lines.
    # Score windows of up to three lines so a product line can be paired with its amount.
    for index in range(len(lines)):
        for width in (1, 2, 3):
            window = lines[index:index + width]
            if not window:
                continue
            combined = " | ".join(window)
            line_terms = terms(combined)
            overlap = query_terms & line_terms
            if overlap:
                score = len(overlap) / len(query_terms)
                candidates.append((score, len(overlap), combined))

    # De-duplicate repeated windows while preserving the strongest first.
    candidates.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
    evidence = []
    seen = set()
    for score, _, passage in candidates:
        if score < 0.20 or passage in seen:
            continue
        seen.add(passage)
        evidence.append(passage)
        if len(evidence) == 3:
            break

    if not evidence:
        return {
            "supported": False,
            "answer": "I couldn't find text in this image that clearly answers that question. Review the recognized OCR text; the image may be blurry or the item and price may have been read incorrectly.",
            "evidence": [],
        }

    return {
        "supported": True,
        "answer": "Relevant text recognized in the image (extractive match, not a verified calculation):\n" + "\n".join(evidence),
        "evidence": evidence,
    }


def create_default_assistant(project_root: str | Path | None = None) -> FinancialAssistant:
    """Create local services. Called lazily when the user first indexes or asks."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    chroma_dir = os.getenv("CHROMA_DIR", str(root / ".chroma"))
    sqlite_path = os.getenv("SQLITE_PATH", str(root / "data" / "interactions.db"))
    embedder = SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    retriever = ChromaRetriever(
        chroma_dir,
        embedder,
        collection_name="financial_assistant_local_v2",
    )
    store = InteractionStore(sqlite_path)
    # Refresh the built-in sample once when the local services are first used.
    sample_path = root / "data" / "sample_policy.txt"
    if sample_path.exists():
        sample_text = sample_path.read_text(encoding="utf-8").strip()
        if sample_text:
            retriever.upsert(
                chunk_text(
                    sample_text,
                    source=sample_path.name,
                    source_type=SourceType.text,
                )
            )
    return FinancialAssistant(retriever, store)



def read_image_text(raw: bytes, project_root: str | Path | None = None) -> str:
    """OCR an image without loading the embedding model and log the result."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    store = InteractionStore(os.getenv("SQLITE_PATH", str(root / "data" / "interactions.db")))
    started = perf_counter()
    text = image_bytes_to_text(raw)
    answer = text or "No readable text was detected."
    store.record(
        question="OCR image extraction",
        supported=bool(text),
        answer=answer,
        latency_ms=round((perf_counter() - started) * 1000, 2),
    )
    return text
