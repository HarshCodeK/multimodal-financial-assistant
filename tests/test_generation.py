from types import SimpleNamespace

from financial_assistant.generation import GroundedAnswerGenerator
from financial_assistant.models import RetrievedChunk, SourceType


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=250, completion_tokens=24),
        )


def test_generation_returns_concise_answer_and_only_valid_citations():
    completions = FakeCompletions(
        '{"supported": true, "answer": "Meals are capped at INR 1,200 per day.", "citations": ["S1", "S99"]}'
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generator = GroundedAnswerGenerator(api_key="test-key", model="test-model", client=client)
    evidence = [
        RetrievedChunk(
            source="policy.pdf",
            source_type=SourceType.pdf,
            text="Meals are reimbursed up to INR 1,200 per employee per day.",
            page=1,
            distance=0.12,
        )
    ]

    result = generator.answer("What is the meal limit?", evidence)

    assert result["supported"] is True
    assert result["answer"] == "Meals are capped at INR 1,200 per day."
    assert result["citations"] == ["S1"]
    assert result["sources"] == evidence
    assert result["prompt_tokens"] == 250
    assert "INR 1,200" in result["prompt_preview"]
    assert completions.kwargs["response_format"]["type"] == "json_schema"
    assert completions.kwargs["response_format"]["json_schema"]["strict"] is True
    assert completions.kwargs["temperature"] == 0


def test_generation_rejects_supported_answer_without_valid_citation():
    completions = FakeCompletions(
        '{"supported": true, "answer": "The policy says 10 days.", "citations": ["UNKNOWN"]}'
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generator = GroundedAnswerGenerator(api_key="test-key", client=client)
    evidence = [
        RetrievedChunk(
            source="policy.pdf",
            source_type=SourceType.pdf,
            text="Claims are due in 15 days.",
            page=2,
            distance=0.2,
        )
    ]

    result = generator.answer("What is the deadline?", evidence)

    assert result["supported"] is False
    assert result["citations"] == []
    assert result["sources"] == evidence
    assert "not provide enough evidence" in result["answer"]


def test_generation_abstains_without_retrieved_passages():
    generator = GroundedAnswerGenerator(api_key="test-key", client=object())

    result = generator.answer("What is the deadline?", [])

    assert result["supported"] is False
    assert result["sources"] == []
    assert "index" in result["answer"].lower()
