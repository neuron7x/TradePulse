from __future__ import annotations

from application.settings import AdminApiSettings


def test_admin_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("TRADEPULSE_AUDIT_SECRET_ID", "env/audit")
    monkeypatch.setenv("TRADEPULSE_ADMIN_TOKEN_ID", "env/token")
    monkeypatch.setenv("TRADEPULSE_ADMIN_SUBJECT", "env-operator")
    monkeypatch.setenv("TRADEPULSE_ADMIN_RATE_LIMIT_MAX_ATTEMPTS", "7")
    monkeypatch.setenv("TRADEPULSE_ADMIN_RATE_LIMIT_INTERVAL_SECONDS", "15")
    monkeypatch.setenv("TRADEPULSE_AUDIT_WEBHOOK_URL", "https://audit.example.com/ingest")

    settings = AdminApiSettings()

    assert settings.audit_secret_id == "env/audit"
    assert settings.admin_token_id == "env/token"
    assert settings.admin_subject == "env-operator"
    assert settings.admin_rate_limit_max_attempts == 7
    assert settings.admin_rate_limit_interval_seconds == 15.0
    assert str(settings.audit_webhook_url) == "https://audit.example.com/ingest"


def test_admin_settings_accepts_explicit_values(monkeypatch):
    monkeypatch.delenv("TRADEPULSE_AUDIT_SECRET_ID", raising=False)
    monkeypatch.delenv("TRADEPULSE_ADMIN_TOKEN_ID", raising=False)

    settings = AdminApiSettings(
        audit_secret_id="explicit-secret",
        admin_token_id="explicit-token",
        admin_subject="explicit",
        admin_rate_limit_max_attempts=3,
        admin_rate_limit_interval_seconds=45.0,
        audit_webhook_url="https://audit.example.com/explicit",
    )

    assert settings.audit_secret_id == "explicit-secret"
    assert settings.admin_token_id == "explicit-token"
    assert settings.admin_subject == "explicit"
    assert settings.admin_rate_limit_max_attempts == 3
    assert settings.admin_rate_limit_interval_seconds == 45.0
    assert str(settings.audit_webhook_url) == "https://audit.example.com/explicit"
