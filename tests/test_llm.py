from financial_assistant.llm import GroqClient, _parse_grounded_response
from financial_assistant.models import RetrievedChunk, SourceType


def test_repair_accepts_trailing_comma():
    parsed = _parse_grounded_response('{"supported": true, "answer": "Limit is 500 INR.",}')
    assert parsed.supported is True
    assert parsed.answer == "Limit is 500 INR."


def test_repair_accepts_fenced_json_with_preface():
    parsed = _parse_grounded_response('Result:\n```json\n{"supported": false, "answer": "Not found."}\n```')
    assert parsed.supported is False
    assert parsed.answer == "Not found."


def test_repair_does_not_invent_missing_fields():
    import pytest

    with pytest.raises(ValueError, match="malformed grounded-answer JSON"):
        _parse_grounded_response('{"supported": true}')


def test_groq_answer_uses_repaired_json_and_preserves_sources(monkeypatch):
    client = GroqClient(api_key="test-key")
    monkeypatch.setattr(
        client,
        "_request",
        lambda payload: {"choices": [{"message": {"content": '{"supported": true, "answer": "500 INR",}'}}]},
    )
    evidence = [RetrievedChunk(source="policy.pdf", source_type=SourceType.pdf, text="Meal limit is 500 INR.", page=1, distance=0.1)]

    result = client.answer("What is the meal limit?", evidence)

    assert result.supported is True
    assert result.answer == "500 INR"
    assert result.sources == evidence
