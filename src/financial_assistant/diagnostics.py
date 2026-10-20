"""Safe diagnostics for local Groq API-key configuration.

Never return or log the API key itself.
"""

import os
from pathlib import Path

from dotenv import dotenv_values


def groq_key_diagnostics(project_root: str | Path) -> dict:
    root = Path(project_root)
    env_file = root / ".env"
    process_key = os.environ.get("GROQ_API_KEY", "")
    file_values = dotenv_values(env_file) if env_file.is_file() else {}
    file_key = str(file_values.get("GROQ_API_KEY") or "")
    effective_key = process_key or os.getenv("GROQ_API_KEY", "")
    source = (
        "process environment (takes precedence over .env)"
        if process_key
        else ".env file"
        if file_key
        else "not found"
    )
    stripped = effective_key.strip()
    looks_placeholder = not stripped or stripped == "replace_with_your_groq_api_key"
    return {
        "env_file_exists": env_file.is_file(),
        "file_has_key": bool(file_key.strip()),
        "process_env_has_key": bool(process_key),
        "source": source,
        "configured": bool(stripped) and not looks_placeholder,
        "has_outer_whitespace": effective_key != stripped,
        "looks_placeholder": looks_placeholder,
        "key_length": len(stripped) if not looks_placeholder else 0,
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
    if is_auth_error:
        return (
            f"Groq rejected the API key for this {workflow} (HTTP 401 / invalid_api_key). "
            "The app can read a key, but Groq does not accept it. Check the exact project-root .env file, "
            "paste the complete current key without extra characters, remove the example placeholder, and "
            "restart Streamlit. If GROQ_API_KEY is also set in Windows/user/system environment variables, "
            "that value takes precedence over .env; update or remove the stale variable. If unsure, revoke the "
            "old key and create a new one in the Groq Console. The key is never shown by this diagnostic."
        )
    if status_code == 429 or "rate_limit" in message:
        return (
            f"Groq rate-limited this {workflow}. Check your model limits and account quota in the Groq Console, "
            "then retry after the limit resets."
        )
    if status_code in (400, 404) or "model_not_found" in message:
        return (
            f"Groq rejected this {workflow} request or model. Check that the configured model ID is currently "
            "available to your account and supports this input type. Details are intentionally redacted."
        )
    return (
        f"The Groq {workflow} request failed ({type(exc).__name__}). Check network access, model availability, "
        "and the Groq Console. Sensitive request details are not displayed."
    )
