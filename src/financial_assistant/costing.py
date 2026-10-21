"""Transparent text-model API-cost estimates; prices are assumptions, not invoices."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRates:
    """USD per one million input/output tokens."""

    input_per_million: float
    output_per_million: float


DEFAULT_TEXT_RATES = TokenRates(input_per_million=0.075, output_per_million=0.30)


def token_cost_usd(input_tokens: int, output_tokens: int, rates: TokenRates) -> float:
    """Estimate token-billed API cost in USD."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")
    if rates.input_per_million < 0 or rates.output_per_million < 0:
        raise ValueError("Token prices cannot be negative.")
    return (input_tokens * rates.input_per_million + output_tokens * rates.output_per_million) / 1_000_000
