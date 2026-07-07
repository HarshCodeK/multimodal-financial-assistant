import fitz

def load_document(file_path: str) -> dict:
    lower = file_path.lower()
    if lower.endswith(".pdf"):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return {"type": "pdf", "text": text.strip()}
    elif lower.endswith((".jpg", ".jpeg", ".png")):
        return {"type": "image", "path": file_path}
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Supported: .pdf, .jpg, .jpeg, .png")
