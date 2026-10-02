"""Model registry — one place that knows which models are live and what each costs.

Why this file exists
--------------------
The previous version hardcoded ``llama-3.3-70b-versatile`` and
``meta-llama/llama-4-scout-17b-16e-instruct``. Groq retired BOTH on
2026-07-17, so every call returned a 404 and the app was dead on arrival.
Hardcoding a model name is how you get 404'd with no warning.

Everything now resolves through here:

  * ``list_models()``  — what the UI dropdown is built from
  * ``resolve()``      — turn a user-facing name into a live model id
  * ``price()``        — USD per 1M tokens, keyed by model id

Model ids can change under you again. They live here and in one env var,
not scattered through the codebase.

Sources (checked 2026-10-02):
  https://console.groq.com/docs/models
  https://console.groq.com/docs/deprecations
"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
# `vision` matters: the financial assistant sends a base64 image and the
# model has to accept it. Only qwen3.8-27b does on the current catalogue.
# `deprecates` records what this model replaced, so the git history explains
# the migration instead of the commit message alone.
MODELS = {
    "qwen/qwen3.8-27b": {
        "label": "Qwen 3.8 27B",
        "vision": True,
        "tools": True,
        "json": True,
        "context": 131_072,
        "in_per_1m": 0.80,
        "out_per_1m": 4.00,
        "speed_tps": 450,
        "note": "Multimodal. Only model here that reads images.",
        "replaces": "meta-llama/llama-4-scout-17b-16e-instruct, llama-3.3-70b-versatile",
    },
    "openai/gpt-oss-120b": {
        "label": "GPT-OSS 120B",
        "vision": False,
        "tools": True,
        "json": True,
        "context": 131_072,
        "in_per_1m": 0.15,
        "out_per_1m": 0.60,
        "speed_tps": 500,
        "note": "Cheapest production-grade text model. No vision.",
        "replaces": "llama-3.3-70b-versatile",
    },
    "openai/gpt-oss-20b": {
        "label": "GPT-OSS 20B",
        "vision": False,
        "tools": True,
        "json": True,
        "context": 131_072,
        "in_per_1m": 0.075,
        "out_per_1m": 0.30,
        "speed_tps": 1000,
        "note": "Fastest and cheapest. Weakest at closed-set tasks.",
        "replaces": "llama-3.1-8b-instant",
    },
}

# Retired on Groq. Kept here so the UI can explain WHY a model is missing
# instead of silently dropping it from the dropdown.
RETIRED = {
    "llama-3.3-70b-versatile": "2026-07-17",
    "meta-llama/llama-4-scout-17b-16e-instruct": "2026-07-17",
    "llama-3.1-8b-instant": "2026-07-17",
    "qwen/qwen3-32b": "2026-07-17",
    "qwen/qwen3.6-27b": "2026-09-14",
    "groq/compound": "2026-09-21",
    "groq/compound-mini": "2026-09-21",
}

DEFAULT_MODEL = os.environ.get("MFA_MODEL", "qwen/qwen3.8-27b")

# Convenience for error messages and UI hints.
VISION_MODELS = [mid for mid, m in MODELS.items() if m["vision"]]
TEXT_MODELS = [mid for mid, m in MODELS.items() if not m["vision"]]

# Models that Groq still lists for enterprise contracts but which 404 on
# free/developer tiers. Asserted in tests so a stale pin fails loudly.
FREE_TIER_SAFE = {"qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"}


class ModelRetiredError(ValueError):
    """Raised when code asks for a model Groq has shut down."""


def list_models(vision: bool | None = None) -> list[dict]:
    """Models available for the dropdown, optionally filtered by capability."""
    out = []
    for model_id, meta in MODELS.items():
        if vision is not None and meta["vision"] != vision:
            continue
        out.append({"id": model_id, **{k: v for k, v in meta.items() if k != "replaces"}})
    return out


def resolve(name: str | None = None) -> str:
    """Turn a model id or short label into a live model id.

    Why: the UI passes what the user picked; config passes an env var; a
    stale value from either must fail loudly instead of 404ing at call time.
    """
    candidate = name or DEFAULT_MODEL

    if candidate in RETIRED:
        raise ModelRetiredError(
            f"{candidate} was retired by Groq on {RETIRED[candidate]}. "
            f"Live models: {', '.join(MODELS)}. "
            f"It replaced by: see MODELS[...]['replaces']."
        )
    if candidate not in MODELS:
        raise ModelRetiredError(
            f"unknown model {candidate!r}. Live models: {', '.join(MODELS)}"
        )
    return candidate


def requires_vision(name: str | None = None) -> bool:
    """True when the resolved model accepts image input."""
    return MODELS[resolve(name)]["vision"]


def price(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost.

    Why: the dashboard shows what a run costs, and a stale price table is a
    silent lie. Prices are looked up by model id, never hardcoded per call site.
    """
    meta = MODELS.get(model_id)
    if meta is None:
        raise ModelRetiredError(f"no price for unknown model {model_id!r}")
    return (
        prompt_tokens / 1_000_000 * meta["in_per_1m"]
        + completion_tokens / 1_000_000 * meta["out_per_1m"]
    )


def api_key() -> str | None:
    return os.environ.get("GROQ_API_KEY")


def key_configured() -> bool:
    return bool(api_key())
