"""Safe diagnostics for local Groq API-key configuration; never expose secrets."""

import os
from pathlib import Path

from dotenv import dotenv_values

PLACEHOLDER_KEY = "replace_with_your_groq_api_key"


def normalize_api_key(value: str | None) -> str:
    """Trim accidental whitespace and one matching pair of surrounding quotes."""
    key = (value or "").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1].strip()
    return key


def groq_key_diagnostics(project_root: str | Path, process_env_preexisting: bool = False) -> dict:
    """Prefer a usable project-root .env key; fall back to process env only if absent."""
    env_file = Path(project_root) / ".env"
    values = dotenv_values(env_file) if env_file.is_file() else {}
    raw_file_key = str(values.get("GROQ_API_KEY") or "")
    file_key = normalize_api_key(raw_file_key)
    process_raw = os.environ.get("GROQ_API_KEY", "")
    process_key = normalize_api_key(process_raw)
    usable_file_key = bool(file_key) and file_key != PLACEHOLDER_KEY
    effective_key = file_key if usable_file_key else process_key
    source = (
        "project-root .env file (takes precedence)"
        if usable_file_key
        else "process environment (no usable key in .env)"
        if process_key and process_key != PLACEHOLDER_KEY
        else "not found"
    )
    placeholder = not effective_key or effective_key == PLACEHOLDER_KEY
    effective_raw = raw_file_key if usable_file_key else process_raw
    return {
        "env_file_exists": env_file.is_file(),
        "file_has_key": bool(file_key),
        "process_env_has_key": bool(process_key),
        "source": source,
        "configured": bool(effective_key) and not placeholder,
        "has_outer_whitespace": bool(effective_raw) and effective_raw != effective_raw.strip(),
        "looks_placeholder": placeholder,
        "key_length": len(effective_key) if not placeholder else 0,
    }


def friendly_groq_error(exc: Exception, workflow: str = "request") -> str:
    """Return actionable provider errors without echoing credentials or raw headers."""
    status_code = getattr(exc, "status_code", None)
    error_code = getattr(exc, "code", None)
    message = str(exc).lower()
    is_auth_error = (
        status_code == 401
        or error_code == "invalid_api_key"
        or "invalid_api_key" in message
        or "authenticationerror" in type(exc).__name__.lower()
    )
    if "groq_api_key is missing" in message:
        return (
            f"No Groq API key was available for this {workflow}. Put a valid GROQ_API_KEY in the project-root "
            ".env file and restart Streamlit. The key is never shown by this diagnostic."
        )
    if is_auth_error:
        return (
            f"Groq rejected the API key for this {workflow} (HTTP 401 / invalid_api_key). The app's key source "
            "is shown in the configuration panel. A usable project-root .env key takes precedence over inherited "
            "Windows environment variables. Verify the key in Groq Console; the key is never displayed."
        )
    if status_code == 429 or "rate_limit" in message:
        return (
            f"Groq rate-limited this {workflow}. Check model limits and account quota in Groq Console, "
            "then retry after the limit resets."
        )
    if status_code in (400, 404) or "model_not_found" in message:
        return (
            f"Groq rejected this {workflow} request or model. Check that the configured text model is available "
            "to your account. Details are intentionally redacted."
        )
    return (
        f"The Groq {workflow} request failed ({type(exc).__name__}). Check network access, text-model availability, "
        "and Groq Console. Sensitive request details are not displayed."
    )
