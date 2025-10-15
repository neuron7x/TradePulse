from __future__ import annotations

import pytest

from application.settings import AdminApiSettings


def test_admin_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("TRADEPULSE_AUDIT_SECRET", "env-secret-value")
    monkeypatch.setenv("TRADEPULSE_ADMIN_SUBJECT", "env-operator")
    monkeypatch.setenv("TRADEPULSE_ADMIN_RATE_LIMIT_MAX_ATTEMPTS", "7")
    monkeypatch.setenv("TRADEPULSE_ADMIN_RATE_LIMIT_INTERVAL_SECONDS", "15")
    monkeypatch.setenv("TRADEPULSE_AUDIT_WEBHOOK_URL", "https://audit.example.com/ingest")

    settings = AdminApiSettings()

    assert settings.audit_secret.get_secret_value() == "env-secret-value"
    assert settings.admin_subject == "env-operator"
    assert settings.admin_rate_limit_max_attempts == 7
    assert settings.admin_rate_limit_interval_seconds == 15.0
    assert str(settings.audit_webhook_url) == "https://audit.example.com/ingest"


def test_admin_settings_accepts_explicit_values(monkeypatch):
    monkeypatch.delenv("TRADEPULSE_AUDIT_SECRET", raising=False)

    settings = AdminApiSettings(
        audit_secret="explicit-secret-value",
        admin_subject="explicit",
        admin_rate_limit_max_attempts=3,
        admin_rate_limit_interval_seconds=45.0,
        audit_webhook_url="https://audit.example.com/explicit",
    )

    assert settings.audit_secret.get_secret_value() == "explicit-secret-value"
    assert settings.admin_subject == "explicit"
    assert settings.admin_rate_limit_max_attempts == 3
    assert settings.admin_rate_limit_interval_seconds == 45.0
    assert str(settings.audit_webhook_url) == "https://audit.example.com/explicit"


def test_secret_manager_prefers_file_value(tmp_path):
    path = tmp_path / "audit_secret"
    path.write_text("file-backed-secret-value", encoding="utf-8")
    settings = AdminApiSettings(
        audit_secret="fallback-secret-value",
        audit_secret_path=path,
        secret_refresh_interval_seconds=0.1,
    )

    manager = settings.build_secret_manager()

    assert manager.get("audit_secret") == "file-backed-secret-value"
    path.write_text("rotated-file-secret-value", encoding="utf-8")
    manager.force_refresh("audit_secret")
    assert manager.get("audit_secret") == "rotated-file-secret-value"


def test_siem_secret_path_satisfies_validation(tmp_path):
    secret_path = tmp_path / "siem_secret"
    secret_path.write_text("siem-client-secret-value", encoding="utf-8")

    settings = AdminApiSettings(
        audit_secret="explicit-secret-value",
        siem_endpoint="https://siem.example.com/ingest",
        siem_client_id="siem-client",
        siem_client_secret_path=secret_path,
    )

    manager = settings.build_secret_manager()
    assert manager.get("siem_client_secret") == "siem-client-secret-value"


def test_missing_siem_secret_raises(monkeypatch):
    monkeypatch.delenv("TRADEPULSE_AUDIT_SECRET", raising=False)
    with pytest.raises(ValueError):
        AdminApiSettings(
            audit_secret="explicit-secret-value",
            siem_endpoint="https://siem.example.com/ingest",
            siem_client_id="siem-client",
        )
