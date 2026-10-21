from financial_assistant.ui_state import active_pdf_sources, get_or_extract_image_text


def test_ocr_is_reused_for_multiple_questions_about_same_image():
    state = {}
    calls = []

    def fake_extractor(image_bytes, filename):
        calls.append((image_bytes, filename))
        return {"filename": filename, "text": "TOTAL INR 420", "mean_confidence": 0.95}

    first = get_or_extract_image_text(state, b"image-one", "receipt.png", fake_extractor)
    second = get_or_extract_image_text(state, b"image-one", "receipt.png", fake_extractor)

    assert first is second
    assert first["text"] == "TOTAL INR 420"
    assert len(calls) == 1


def test_ocr_cache_recomputes_when_image_bytes_change():
    state = {}
    calls = []

    def fake_extractor(image_bytes, filename):
        calls.append(image_bytes)
        return {"filename": filename, "text": image_bytes.decode(), "mean_confidence": None}

    first = get_or_extract_image_text(state, b"image-one", "one.png", fake_extractor)
    second = get_or_extract_image_text(state, b"image-two", "two.png", fake_extractor)

    assert first["text"] == "image-one"
    assert second["text"] == "image-two"
    assert calls == [b"image-one", b"image-two"]


def test_persisted_pdf_index_is_not_active_until_indexed_in_current_session():
    saved = ["sample_financial_policy.pdf"]

    assert active_pdf_sources(saved, indexed_this_session=False) == []
    assert active_pdf_sources(saved, indexed_this_session=True) == saved
