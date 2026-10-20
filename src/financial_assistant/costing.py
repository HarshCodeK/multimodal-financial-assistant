"""Transparent API-cost estimates; prices are editable UI assumptions, not invoices."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRates:
    """USD per one million input/output tokens."""

    input_per_million: float
    output_per_million: float


DEFAULT_VISION_RATES = TokenRates(input_per_million=0.60, output_per_million=3.00)
DEFAULT_TEXT_RATES = TokenRates(input_per_million=0.075, output_per_million=0.30)
GOOGLE_VISION_OCR_USD_PER_IMAGE = 0.0015
GROQ_QWEN_IMAGE_INPUT_TOKENS = 2048


def token_cost_usd(input_tokens: int, output_tokens: int, rates: TokenRates) -> float:
    """Estimate token-billed API cost in USD."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")
    if rates.input_per_million < 0 or rates.output_per_million < 0:
        raise ValueError("Token prices cannot be negative.")
    return (input_tokens * rates.input_per_million + output_tokens * rates.output_per_million) / 1_000_000


def estimate_vision_cost_usd(
    input_tokens: int | None,
    output_tokens: int | None,
    rates: TokenRates = DEFAULT_VISION_RATES,
) -> float | None:
    """Estimate cost only when provider-reported usage is available."""
    if input_tokens is None or output_tokens is None:
        return None
    return token_cost_usd(input_tokens, output_tokens, rates)


def estimate_ocr_then_text_cost_usd(
    extracted_text_tokens: int,
    output_tokens: int,
    *,
    text_rates: TokenRates = DEFAULT_TEXT_RATES,
    ocr_cost_usd: float = GOOGLE_VISION_OCR_USD_PER_IMAGE,
    prompt_overhead_tokens: int = 250,
) -> float:
    """Estimate paid cloud OCR + text LLM cost for one image, before tax/free tiers."""
    if extracted_text_tokens < 0 or output_tokens < 0 or prompt_overhead_tokens < 0:
        raise ValueError("Token counts cannot be negative.")
    if ocr_cost_usd < 0:
        raise ValueError("OCR price cannot be negative.")
    return ocr_cost_usd + token_cost_usd(
        extracted_text_tokens + prompt_overhead_tokens,
        output_tokens,
        text_rates,
    )
