from pathlib import Path
import base64
import io

import fitz
from PIL import Image

from .models import DocumentChunk, SourceType

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_TOTAL_EXTRACTED_CHARS = 200_000
MAX_CHARS_PER_PAGE = 8_000


def extract_pdf(path: str | Path) -> list[DocumentChunk]:
    file_path = Path(path)
    return extract_pdf_bytes(file_path.read_bytes(), source_name=file_path.name)


def extract_pdf_bytes(raw: bytes, source_name: str) -> list[DocumentChunk]:
    if not raw or len(raw) > MAX_PDF_BYTES:
        raise ValueError("PDF is empty or larger than the 10 MiB limit")
    if not raw.startswith(b"%PDF-"):
        raise ValueError("uploaded file is not a valid PDF")
    chunks: list[DocumentChunk] = []
    remaining_chars = MAX_TOTAL_EXTRACTED_CHARS
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            if len(doc) > MAX_PDF_PAGES:
                raise ValueError(f"PDF has more than {MAX_PDF_PAGES} pages")
            for page_number, page in enumerate(doc, start=1):
                if remaining_chars <= 0:
                    break
                text = " ".join(page.get_text("text").split())[:MAX_CHARS_PER_PAGE]
                text = text[:remaining_chars]
                if text:
                    chunks.append(DocumentChunk(source=source_name, source_type=SourceType.pdf, text=text, page=page_number))
                    remaining_chars -= len(text)
    except (fitz.FileDataError, fitz.EmptyFileError) as exc:
        raise ValueError("PDF could not be opened") from exc
    return chunks


def validate_image(path: str | Path) -> Image.Image:
    raw = Path(path).read_bytes()
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image is larger than the 8 MiB limit")
    image = Image.open(io.BytesIO(raw))
    image.verify()
    return Image.open(io.BytesIO(raw)).convert("RGB")


def image_bytes_as_data_url(raw: bytes) -> str:
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image is larger than the 8 MiB limit")
    image = Image.open(io.BytesIO(raw))
    image.verify()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_as_data_url(path: str | Path) -> str:
    image = validate_image(path)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_description_chunk(path: str | Path, description: str) -> DocumentChunk:
    text = " ".join(description.split())
    if not text:
        raise ValueError("image description cannot be empty")
    return DocumentChunk(source=str(path), source_type=SourceType.image, text=text)
