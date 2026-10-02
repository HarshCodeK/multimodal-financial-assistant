"""Which models exist, what they cost, and which can read images.

One file, because a model name hardcoded anywhere else is a bug waiting to
happen. Groq retires models without a deprecation warning -- an outdated id
returns a hard 404 -- so every model used anywhere is named here and nowhere else.

Checked against https://console.groq.com/docs/models on 2026-10-02.
"""

# id: (label, vision, input $/1M, output $/1M)
MODELS = {
    "qwen/qwen3.8-27b": ("Qwen 3.8 27B", True, 0.80, 4.00),
    "openai/gpt-oss-120b": ("GPT-OSS 120B", False, 0.15, 0.60),
    "openai/gpt-oss-20b": ("GPT-OSS 20B", False, 0.075, 0.30),
}

DEFAULT_MODEL = "qwen/qwen3.8-27b"

# qwen/qwen3.8-27b is currently the only model on the account that accepts
# images, which is why it is the default: a photo receipt is the primary input.
VISION_MODELS = [m for m, (_, v, _, _) in MODELS.items() if v]


class UnknownModel(ValueError):
    """A model id that is not in the table above."""


def resolve(model_id: str = None) -> str:
    """Return a model id that is known to exist."""
    candidate = model_id or DEFAULT_MODEL
    if candidate not in MODELS:
        raise UnknownModel(
            f"{candidate!r} is not a known model. Available: {', '.join(MODELS)}"
        )
    return candidate


def estimate_cost_usd(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Rough cost. Unknown models are refused rather than guessed at."""
    _, _, per_in, per_out = MODELS[resolve(model_id)]
    return (prompt_tokens * per_in + completion_tokens * per_out) / 1_000_000
