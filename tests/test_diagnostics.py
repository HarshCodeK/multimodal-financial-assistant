from financial_assistant.diagnostics import friendly_groq_error, groq_key_diagnostics, normalize_api_key


def test_normalize_api_key_trims_spaces_and_matching_quotes():
    assert normalize_api_key('  "gsk-test-key"  ') == "gsk-test-key"


def test_diagnostics_report_env_file_without_revealing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk-super-secret\n", encoding="utf-8")

    status = groq_key_diagnostics(tmp_path, process_env_preexisting=False)

    assert status["configured"] is True
    assert status["source"] == "project-root .env file (takes precedence)"
    assert status["key_length"] == len("gsk-super-secret")
    assert "gsk-super-secret" not in str(status)


def test_project_env_takes_precedence_over_stale_process_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "stale-invalid-key")
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk-valid-file-key\n", encoding="utf-8")

    status = groq_key_diagnostics(tmp_path, process_env_preexisting=True)

    assert status["source"] == "project-root .env file (takes precedence)"
    assert status["configured"] is True
    assert "stale-invalid-key" not in str(status)


def test_invalid_key_error_is_actionable_and_does_not_echo_secret():
    class FakeAuthError(Exception):
        status_code = 401
        code = "invalid_api_key"

    error = friendly_groq_error(FakeAuthError("invalid_api_key: do not echo gsk-secret"), "PDF answer")

    assert "HTTP 401" in error
    assert "Windows environment variables" in error
    assert "gsk-secret" not in error
