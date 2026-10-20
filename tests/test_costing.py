import pytest

from financial_assistant.costing import (
    DEFAULT_TEXT_RATES,
    DEFAULT_VISION_RATES,
    estimate_ocr_then_text_cost_usd,
    estimate_vision_cost_usd,
    token_cost_usd,
)


def test_token_cost_uses_input_and_output_rates():
    assert token_cost_usd(1_000_000, 1_000_000, DEFAULT_TEXT_RATES) == pytest.approx(0.375)


def test_vision_cost_requires_provider_usage():
    assert estimate_vision_cost_usd(None, 100) is None
    assert estimate_vision_cost_usd(2048, 100, DEFAULT_VISION_RATES) == pytest.approx(0.0015288)


def test_ocr_plus_text_cost_includes_cloud_ocr_and_text_tokens():
    cost = estimate_ocr_then_text_cost_usd(300, 100, prompt_overhead_tokens=250)
    assert cost == pytest.approx(0.00157125)


def test_costing_rejects_negative_token_counts():
    with pytest.raises(ValueError, match="cannot be negative"):
        token_cost_usd(-1, 1, DEFAULT_TEXT_RATES)
