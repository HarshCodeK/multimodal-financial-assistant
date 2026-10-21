"""PDF extraction with local OCR fallback and overlapping word-based chunking."""

from pathlib import Path
import re

import pymupdf

from .models import DocumentChunk, SourceType
from .ocr import extract_image_text

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_TOTAL_EXTRACTED_CHARS = 200_000
CHUNK_WORDS = 80
CHUNK_OVERLAP_WORDS = 16


def chunk_text(
    text: str, *, source: str, source_type: SourceType, page: int | None = None
) -> list[DocumentChunk]:
    """Split text into 80-word windows with 16 words of overlap."""
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        part = " ".join(words[start:start + CHUNK_WORDS]).strip()
        if part:
            chunks.append(DocumentChunk(source=source, source_type=source_type, text=part, page=page))
        if start + CHUNK_WORDS >= len(words):
            break
        start += CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    return chunks


def extract_pdf(path: str | Path) -> list[DocumentChunk]:
    file_path = Path(path)
    return extract_pdf_bytes(file_path.read_bytes(), source_name=file_path.name)


def extract_pdf_bytes(raw: bytes, source_name: str) -> list[DocumentChunk]:
    """Extract selectable text page-by-page. Scanned/image-only PDFs need OCR and are rejected."""
    if not raw or len(raw) > MAX_PDF_BYTES:
        raise ValueError("PDF is empty or larger than the 10 MiB limit")
    if not raw.startswith(b"%PDF-"):
        raise ValueError("uploaded file is not a valid PDF")

    chunks: list[DocumentChunk] = []
    extracted_total = 0
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as doc:
            if len(doc) > MAX_PDF_PAGES:
                raise ValueError(f"PDF has more than {MAX_PDF_PAGES} pages")
            for page_number, page in enumerate(doc, start=1):
                if extracted_total >= MAX_TOTAL_EXTRACTED_CHARS:
                    break
                text = " ".join(page.get_text("text").split())
                if not text:
                    # OCR only pages without selectable text; page images remain local.
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    try:
                        text = " ".join(extract_image_text(pixmap.tobytes("png"), source_name)["text"].split())
                    except ValueError:
                        # A blank or unreadable page contributes no chunks.
                        continue
                remaining = MAX_TOTAL_EXTRACTED_CHARS - extracted_total
                text = text[:remaining]
                chunks.extend(
                    chunk_text(text, source=source_name, source_type=SourceType.pdf, page=page_number)
                )
                extracted_total += len(text)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise ValueError("PDF could not be opened") from exc

    return chunks
