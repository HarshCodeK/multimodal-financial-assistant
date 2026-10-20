from pathlib import Path
from PIL import Image
from financial_assistant.extract import image_as_data_url, image_bytes_as_data_url, validate_image

def test_image_validation_and_data_url(tmp_path: Path):
    path = tmp_path / "statement.png"
    Image.new("RGB", (20, 20), "white").save(path)
    assert validate_image(path).size == (20, 20)
    assert image_as_data_url(path).startswith("data:image/jpeg;base64,")
    assert image_bytes_as_data_url(path.read_bytes()).startswith("data:image/jpeg;base64,")

def test_pdf_extraction_preserves_page(tmp_path):
    import pymupdf
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Meal allowance is 500 INR.")
    path = tmp_path / "policy.pdf"
    pdf.save(path)
    pdf.close()
    from financial_assistant.extract import extract_pdf
    chunks = extract_pdf(path)
    assert chunks[0].page == 1
    assert "500 INR" in chunks[0].text


def test_included_sample_pdf_has_extractable_policy_text():
    from financial_assistant.extract import extract_pdf

    sample = Path(__file__).resolve().parents[1] / "data" / "sample_financial_policy.pdf"
    chunks = extract_pdf(sample)
    assert chunks
    assert any("INR 1,200" in chunk.text for chunk in chunks)
    assert any("15 calendar days" in chunk.text for chunk in chunks)
