from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from interfaces.secrets.manager import SecretManager


@contextmanager
def _dashboard_module() -> Iterator[object]:
    module = importlib.import_module("interfaces.dashboard_streamlit")
    try:
        yield module
    finally:
        module._set_secret_manager(None)  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_dashboard_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in {
        "DASHBOARD_SECRET_BACKEND",
        "DASHBOARD_SECRET_RESOLVER",
        "DASHBOARD_SECRET_PATH",
        "DASHBOARD_SECRET_PATH_ENV",
        "DASHBOARD_SECRET_ALIAS",
        "DASHBOARD_ADMIN_USERNAME",
        "DASHBOARD_ADMIN_PASSWORD_HASH",
        "DASHBOARD_COOKIE_NAME",
        "DASHBOARD_COOKIE_KEY",
        "DASHBOARD_COOKIE_EXPIRY_DAYS",
        "TRADEPULSE_SECRET_MANAGER_FACTORY",
    }:
        monkeypatch.delenv(key, raising=False)
    with _dashboard_module() as module:
        module._set_secret_manager(None)  # type: ignore[attr-defined]
    yield


def test_load_secrets_returns_empty_without_backend() -> None:
    with _dashboard_module() as module:
        assert module._load_secrets() == {}  # type: ignore[attr-defined]


def test_load_secrets_uses_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolver(path: str) -> dict[str, str]:
        assert path == "kv/tradepulse/dashboard"
        return {"username": "vault", "password_hash": "hash"}

    backend = SecretManager({"vault": resolver})

    with _dashboard_module() as module:
        module._set_secret_manager(backend)  # type: ignore[attr-defined]
        monkeypatch.setenv("DASHBOARD_SECRET_BACKEND", "vault")
        monkeypatch.setenv("DASHBOARD_SECRET_PATH_ENV", "DASHBOARD_SECRET_ALIAS")
        monkeypatch.setenv("DASHBOARD_SECRET_ALIAS", "kv/tradepulse/dashboard")

        secrets = module._load_secrets()  # type: ignore[attr-defined]

    assert secrets["DASHBOARD_ADMIN_USERNAME"] == "vault"
    assert secrets["DASHBOARD_ADMIN_PASSWORD_HASH"] == "hash"


def test_load_auth_config_prefers_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolver(path: str) -> dict[str, str]:
        assert path == "kv/tradepulse/dashboard"
        return {
            "username": "vault-admin",
            "password_hash": "vault-hash",
            "cookie_key": "vault-cookie",
            "cookie_expiry_days": "45",
        }

    backend = SecretManager({"vault": resolver})

    with _dashboard_module() as module:
        module._set_secret_manager(backend)  # type: ignore[attr-defined]
        monkeypatch.setenv("DASHBOARD_SECRET_BACKEND", "vault")
        monkeypatch.setenv("DASHBOARD_SECRET_PATH", "kv/tradepulse/dashboard")
        monkeypatch.setenv("DASHBOARD_ADMIN_USERNAME", "env-admin")
        monkeypatch.setenv("DASHBOARD_COOKIE_KEY", "env-cookie")
        config = module.load_auth_config()

    usernames = config["credentials"]["usernames"]
    assert "vault-admin" in usernames
    assert usernames["vault-admin"]["password"] == "vault-hash"
    assert config["cookie"]["key"] == "vault-cookie"
    assert config["cookie"]["expiry_days"] == 45
