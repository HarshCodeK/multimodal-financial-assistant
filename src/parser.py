"""Turn an uploaded file into the dict the extractor expects.

PyMuPDF is imported lazily inside the function: it is a ~50MB binary
dependency, and an image upload should not pay for loading it.
"""
import os


def load(file_path: str) -> dict:
    lower = file_path.lower()
    name = os.path.basename(file_path)

    if lower.endswith(".pdf"):
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        text = text.strip()
        if not text:
            raise ValueError(
                f"PDF {name!r} has no extractable text. Scanned/image-only PDFs "
                "are not supported by the current parser."
            )
        return {"type": "pdf", "text": text, "filename": name}

    if lower.endswith((".jpg", ".jpeg", ".png")):
        return {"type": "image", "path": file_path, "filename": name}

    raise ValueError(
        f"unsupported file type: {name}. Supported: .pdf, .jpg, .jpeg, .png"
    )
