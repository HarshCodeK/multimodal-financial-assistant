import pytest

from financial_assistant.costing import DEFAULT_TEXT_RATES, token_cost_usd


def test_token_cost_uses_input_and_output_rates():
    assert token_cost_usd(1_000_000, 1_000_000, DEFAULT_TEXT_RATES) == pytest.approx(0.375)


def test_costing_rejects_negative_token_counts():
    with pytest.raises(ValueError, match="cannot be negative"):
        token_cost_usd(-1, 1, DEFAULT_TEXT_RATES)
