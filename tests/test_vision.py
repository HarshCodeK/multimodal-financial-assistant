from types import SimpleNamespace

import pytest

from financial_assistant.vision import answer_image_question


PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload"


class FakeCompletions:
    def __init__(self, content="The image shows a receipt total of INR 420."):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=2200, completion_tokens=24),
        )


def make_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_image_question_sends_image_and_question_to_vision_model():
    completions = FakeCompletions()
    result = answer_image_question(
        PNG,
        "receipt.png",
        "What is the total?",
        api_key="test-key",
        model="test-vision-model",
        client=make_client(completions),
    )

    assert result["answer"] == "The image shows a receipt total of INR 420."
    assert result["filename"] == "receipt.png"
    assert result["model"] == "test-vision-model"
    assert result["prompt_tokens"] == 2200
    assert result["completion_tokens"] == 24
    assert completions.kwargs["messages"][1]["content"][0]["text"] == "What is the total?"
    image_part = completions.kwargs["messages"][1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert completions.kwargs["temperature"] == 0


def test_image_question_rejects_empty_question():
    with pytest.raises(ValueError, match="Enter a question"):
        answer_image_question(PNG, "receipt.png", " ", api_key="test-key", client=object())


def test_image_question_rejects_invalid_image_bytes():
    with pytest.raises(ValueError, match="Unsupported or invalid image"):
        answer_image_question(b"not-an-image", "image.png", "Describe this", api_key="test-key", client=object())


def test_image_question_rejects_oversized_image():
    too_large = b"\x89PNG\r\n\x1a\n" + b"x" * (5 * 1024 * 1024)
    with pytest.raises(ValueError, match="5 MiB"):
        answer_image_question(too_large, "image.png", "Describe this", api_key="test-key", client=object())


def test_image_question_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY is missing"):
        answer_image_question(PNG, "image.png", "Describe this")
