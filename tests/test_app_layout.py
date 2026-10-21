from pathlib import Path


def test_pdf_and_image_uploaders_are_on_main_screen_not_sidebar():
    app_path = Path(__file__).resolve().parents[1] / "src" / "financial_assistant" / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert 'st.tabs(["📄 PDF RAG Explorer", "🧾 OCR image Q&A"])' in source
    assert 'key="pdf_upload_main"' in source
    assert 'key="image_upload_main"' in source
    assert source.count("st.file_uploader(") == 2
    assert "st.sidebar" not in source
    assert "answer_image_question" not in source
    assert "extract_image_text" in source
    assert "answer_ocr_text" in source
