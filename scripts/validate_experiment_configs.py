#!/usr/bin/env python3
"""Validate Hydra experiment configurations using the Pydantic schemas."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.secrets import EnvironmentSecretLoader, secret_env_var_name
from configs.settings import (
    SecretRef,
    SecretResolutionError,
    load_experiment_settings,
    register_structured_configs,
)

_ENVIRONMENTS = ("ci", "stage", "prod", "local")


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "conf"


def _load_experiment(name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(_config_dir())):
        return compose(config_name="config", overrides=[f"experiment={name}"])


def main() -> int:
    register_structured_configs()
    loader = EnvironmentSecretLoader()
    errors: list[str] = []

    for name in _ENVIRONMENTS:
        cfg = _load_experiment(name)
        try:
            load_experiment_settings(cfg, secret_loader=loader)
        except SecretResolutionError as exc:
            database_cfg = cfg.experiment.database
            if isinstance(database_cfg, DictConfig):
                password_cfg = database_cfg.get("password")
            else:  # pragma: no cover - defensive fallback
                password_cfg = getattr(database_cfg, "password", None)
            try:
                payload = OmegaConf.to_container(password_cfg, resolve=True)
                ref = SecretRef.model_validate(payload)
                env_key = secret_env_var_name(ref)
            except Exception:  # pragma: no cover - fall back to raw message
                env_key = str(exc)
            errors.append(f"{name}: missing secret {env_key}")
        except ValidationError as exc:
            errors.append(f"{name}: {exc}")

    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        return 1

    print("All experiment configurations validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
