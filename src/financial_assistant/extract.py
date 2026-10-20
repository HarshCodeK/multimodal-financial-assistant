"""Local PDF/image text extraction with OCR fallback for scanned pages."""

from pathlib import Path
import io
import re
from functools import lru_cache

import pymupdf
from PIL import Image

from .models import DocumentChunk, SourceType

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_TOTAL_EXTRACTED_CHARS = 200_000
MAX_IMAGE_PIXELS = 20_000_000
CHUNK_WORDS = 80
CHUNK_OVERLAP_WORDS = 16


def chunk_text(text: str, *, source: str, source_type: SourceType, page: int | None = None) -> list[DocumentChunk]:
    """Split extracted text into overlapping word chunks for semantic retrieval."""
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


@lru_cache(maxsize=1)
def _get_ocr_engine():
    """Load the OCR model once, only when OCR is first requested."""
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "Local OCR is not installed. Run: python -m pip install rapidocr onnxruntime"
        ) from exc
    return RapidOCR()


def _ocr_image(image: Image.Image) -> str:
    """Run local OCR. The OCR engine is imported only when needed."""
    import numpy as np

    result = _get_ocr_engine()(np.asarray(image.convert("RGB")))
    recognized = getattr(result, "txts", None)
    if recognized is None:
        # Compatibility fallback for older RapidOCR result formats.
        if isinstance(result, tuple) and result:
            recognized = [line[1] for line in (result[0] or []) if len(line) > 1]
        else:
            recognized = []
    # Keep OCR line boundaries so receipt fields remain independently searchable.
    return "\n".join(str(line).strip() for line in recognized if str(line).strip())


def extract_pdf(path: str | Path) -> list[DocumentChunk]:
    file_path = Path(path)
    return extract_pdf_bytes(file_path.read_bytes(), source_name=file_path.name)


def extract_pdf_bytes(raw: bytes, source_name: str) -> list[DocumentChunk]:
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
                # A nearly empty text layer usually indicates a scanned page.
                if len(text) < 25:
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                    text = _ocr_image(image)
                if not text:
                    continue
                remaining = MAX_TOTAL_EXTRACTED_CHARS - extracted_total
                text = text[:remaining]
                chunks.extend(chunk_text(text, source=source_name, source_type=SourceType.pdf, page=page_number))
                extracted_total += len(text)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise ValueError("PDF could not be opened") from exc
    return chunks


def image_bytes_to_text(raw: bytes) -> str:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image is empty or larger than the 8 MiB limit")
    try:
        image = Image.open(io.BytesIO(raw))
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError("image dimensions exceed the supported pixel limit")
        image.verify()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("uploaded file is not a valid supported image") from exc
    return _ocr_image(image)


def validate_image(path: str | Path) -> Image.Image:
    raw = Path(path).read_bytes()
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image is empty or larger than the 8 MiB limit")
    try:
        image = Image.open(io.BytesIO(raw))
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError("image dimensions exceed the supported pixel limit")
        image.verify()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("uploaded file is not a valid supported image") from exc
