# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Mapping
from typing import Any, Callable

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth

from core.indicators.entropy import delta_entropy, entropy
from core.indicators.kuramoto import compute_phase, kuramoto_order
from interfaces.secrets.manager import SecretManager, SecretManagerError

LOGGER = logging.getLogger(__name__)

_SECRET_MANAGER: SecretManager | None = None
_SECRET_FIELD_ALIASES = {
    "USERNAME": "DASHBOARD_ADMIN_USERNAME",
    "ADMIN_USERNAME": "DASHBOARD_ADMIN_USERNAME",
    "PASSWORD_HASH": "DASHBOARD_ADMIN_PASSWORD_HASH",
    "ADMIN_PASSWORD_HASH": "DASHBOARD_ADMIN_PASSWORD_HASH",
    "COOKIE_NAME": "DASHBOARD_COOKIE_NAME",
    "COOKIE_KEY": "DASHBOARD_COOKIE_KEY",
    "COOKIE_SECRET": "DASHBOARD_COOKIE_KEY",
    "COOKIE_EXPIRY_DAYS": "DASHBOARD_COOKIE_EXPIRY_DAYS",
}


def _import_callable(spec: str) -> Callable[..., Any]:
    """Return the callable identified by ``spec``."""

    module_path: str
    attribute: str
    if ":" in spec:
        module_path, attribute = spec.split(":", 1)
    else:
        module_path, _, attribute = spec.rpartition(".")
    if not module_path or not attribute:
        raise ImportError(f"Invalid resolver specification '{spec}'")
    module = importlib.import_module(module_path)
    candidate = getattr(module, attribute)
    if not callable(candidate):
        raise TypeError(f"Resolver '{spec}' is not callable")
    return candidate


def _get_secret_manager() -> SecretManager | None:
    """Initialise or reuse the dashboard secret manager."""

    global _SECRET_MANAGER  # noqa: PLW0603 - module level cache for runtime reuse
    if _SECRET_MANAGER is not None:
        return _SECRET_MANAGER

    factory_spec = os.getenv("TRADEPULSE_SECRET_MANAGER_FACTORY")
    manager: SecretManager | None = None

    if factory_spec:
        try:
            factory = _import_callable(factory_spec)
            candidate = factory()  # type: ignore[no-any-return]
            if isinstance(candidate, SecretManager):
                manager = candidate
            else:
                LOGGER.error(
                    "Secret manager factory %s returned %r instead of SecretManager",
                    factory_spec,
                    candidate,
                )
        except Exception:  # pragma: no cover - defensive guard around optional dependency
            LOGGER.exception("Failed to build secret manager via %s", factory_spec)

    if manager is None:
        backend_name = os.getenv("DASHBOARD_SECRET_BACKEND")
        resolver_spec = os.getenv("DASHBOARD_SECRET_RESOLVER")
        if backend_name and resolver_spec:
            try:
                resolver = _import_callable(resolver_spec)
            except Exception:  # pragma: no cover - defensive guard around optional dependency
                LOGGER.exception("Failed to import resolver %s", resolver_spec)
            else:
                manager = SecretManager({backend_name: resolver})
        elif resolver_spec and not backend_name:
            LOGGER.warning(
                "DASHBOARD_SECRET_RESOLVER is set but DASHBOARD_SECRET_BACKEND is missing",
            )

    _SECRET_MANAGER = manager
    return manager


def _set_secret_manager(manager: SecretManager | None) -> None:
    """Override the cached secret manager instance (primarily for tests)."""

    global _SECRET_MANAGER  # noqa: PLW0603 - test hook
    _SECRET_MANAGER = manager


def _load_secrets() -> Mapping[str, str]:
    """Load dashboard credentials from the configured secret backend."""

    backend = os.getenv("DASHBOARD_SECRET_BACKEND")
    if not backend:
        return {}

    manager = _get_secret_manager()
    if manager is None:
        LOGGER.warning("Secret backend '%s' configured but no resolver is registered", backend)
        return {}

    path = os.getenv("DASHBOARD_SECRET_PATH")
    if not path:
        path_env = os.getenv("DASHBOARD_SECRET_PATH_ENV")
        if path_env:
            path = os.getenv(path_env)
    if not path:
        LOGGER.warning("Secret backend '%s' configured but no path provided", backend)
        return {}

    try:
        payload = manager.resolve(backend, path)
    except SecretManagerError:
        LOGGER.exception("Unable to resolve secrets for backend '%s'", backend)
        return {}

    if not isinstance(payload, Mapping):
        LOGGER.error(
            "Secret resolver for backend '%s' must return a mapping, got %s",
            backend,
            type(payload).__name__,
        )
        return {}

    normalised: dict[str, str] = {}
    for key, value in payload.items():
        canonical = _SECRET_FIELD_ALIASES.get(str(key).upper(), str(key).upper())
        normalised[canonical] = str(value)
    return normalised


def load_auth_config() -> dict[str, Any]:
    """Load authentication configuration from environment variables or Vault."""

    secrets = _load_secrets()

    username = secrets.get("DASHBOARD_ADMIN_USERNAME") or os.getenv(
        "DASHBOARD_ADMIN_USERNAME",
        "admin",
    )
    password_hash = secrets.get("DASHBOARD_ADMIN_PASSWORD_HASH") or os.getenv(
        "DASHBOARD_ADMIN_PASSWORD_HASH",
        # Default hash for 'admin123' (ONLY for development/example)
        "$2b$12$EixZaYVK1fsbw1ZfbX3OXe.RKjKWbFUZYWbAKpKnvGmcPNW3OL2K6",
    )
    cookie_name = secrets.get("DASHBOARD_COOKIE_NAME") or os.getenv(
        "DASHBOARD_COOKIE_NAME",
        "tradepulse_auth",
    )
    cookie_key = secrets.get("DASHBOARD_COOKIE_KEY") or os.getenv(
        "DASHBOARD_COOKIE_KEY",
        "default_cookie_key_change_in_production",
    )
    cookie_expiry_value = secrets.get("DASHBOARD_COOKIE_EXPIRY_DAYS") or os.getenv(
        "DASHBOARD_COOKIE_EXPIRY_DAYS",
        "30",
    )
    try:
        cookie_expiry_days = int(cookie_expiry_value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        LOGGER.warning("Invalid cookie expiry '%s', falling back to 30 days", cookie_expiry_value)
        cookie_expiry_days = 30

    return {
        "credentials": {
            "usernames": {
                username: {
                    "name": username.capitalize(),
                    "password": password_hash,
                }
            }
        },
        "cookie": {
            "name": cookie_name,
            "key": cookie_key,
            "expiry_days": cookie_expiry_days,
        },
        "preauthorized": [],
    }


def _build_authenticator(config: Mapping[str, Any] | None = None) -> stauth.Authenticate:
    """Construct a Streamlit authenticator using the provided or discovered config."""

    cfg = config if config is not None else load_auth_config()
    return stauth.Authenticate(
        cfg["credentials"],
        cfg["cookie"]["name"],
        cfg["cookie"]["key"],
        cfg["cookie"]["expiry_days"],
        cfg["preauthorized"],
    )


def main(config: Mapping[str, Any] | None = None) -> None:
    """Entry point executed by Streamlit to render the dashboard."""

    authenticator = _build_authenticator(config)
    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status is False:
        st.error("Username/password is incorrect")
        return
    if authentication_status is None:
        st.warning("Please enter your username and password")
        return

    authenticator.logout("Logout", "sidebar")
    st.sidebar.write(f"Welcome *{name}*")

    st.title("TradePulse — Real-time Indicators")
    uploaded = st.file_uploader("Upload CSV with columns: ts, price, volume", type=["csv"])
    if not uploaded:
        return

    df = pd.read_csv(uploaded)
    phases = compute_phase(df["price"].to_numpy())
    R = kuramoto_order(phases[-200:])
    H = entropy(df["price"].to_numpy()[-200:])
    dH = delta_entropy(df["price"].to_numpy(), window=200)
    st.metric("Kuramoto R", f"{R:.3f}")
    st.metric("Entropy H(200)", f"{H:.3f}")
    st.metric("ΔH(200)", f"{dH:.3f}")
    st.line_chart(df[["price"]])


if __name__ == "__main__":
    main()
