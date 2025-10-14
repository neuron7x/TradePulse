"""Utilities for resolving Hydra secret references in local environments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .settings import SecretLoader, SecretRef, SecretResolutionError

ENV_PREFIX = "SECRETS"


def _normalize_segment(value: str) -> str:
    return value.replace("/", "__").replace("-", "_").replace(".", "_").upper()


def secret_env_var_name(ref: SecretRef) -> str:
    """Return the environment variable name expected for the given secret reference."""

    backend = _normalize_segment(ref.backend)
    path = _normalize_segment(ref.path.strip("/"))
    key = _normalize_segment(ref.key)
    if ref.version:
        version = _normalize_segment(ref.version)
        return f"{ENV_PREFIX}__{backend}__{path}__{key}__{version}"
    return f"{ENV_PREFIX}__{backend}__{path}__{key}"


@dataclass(slots=True)
class EnvironmentSecretLoader:
    """Resolve secrets from environment variables in CI and developer machines."""

    environ: Mapping[str, str] | None = None

    def __call__(self, ref: SecretRef) -> str:
        env = self.environ if self.environ is not None else os.environ
        key = secret_env_var_name(ref)
        value = env.get(key)
        if value is None:
            raise SecretResolutionError(
                f"Secret {ref.backend}:{ref.path}#{ref.key} not found in environment variable {key}"
            )
        return value


def default_secret_loader() -> SecretLoader:
    """Return the default secret loader used by the analytics runner."""

    return EnvironmentSecretLoader()
