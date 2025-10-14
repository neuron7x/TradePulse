"""Tests for Hydra-backed experiment configuration validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from configs.secrets import EnvironmentSecretLoader, secret_env_var_name
from configs.settings import (
    ExperimentSettings,
    SecretRef,
    SecretResolutionError,
    load_experiment_settings,
    register_structured_configs,
)


def _compose_experiment(name: str) -> DictConfig:
    """Compose a Hydra configuration for the requested experiment."""

    register_structured_configs()
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parents[3] / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    return cfg


def test_local_configuration_validates_without_secrets() -> None:
    cfg = _compose_experiment("local")
    settings = load_experiment_settings(cfg, secret_loader=EnvironmentSecretLoader({}))
    assert isinstance(settings, ExperimentSettings)
    assert str(settings.database.uri).startswith("sqlite:///")
    assert settings.analytics.window >= 32


def test_stage_configuration_requires_secret() -> None:
    cfg = _compose_experiment("stage")
    loader = EnvironmentSecretLoader({})
    with pytest.raises(SecretResolutionError):
        load_experiment_settings(cfg, secret_loader=loader)


def test_stage_configuration_resolves_secret_when_present() -> None:
    cfg = _compose_experiment("stage")
    password_cfg = cfg.experiment.database.password
    secret_payload = OmegaConf.to_container(password_cfg, resolve=True)
    secret_ref = SecretRef.model_validate(secret_payload)
    env_key = secret_env_var_name(secret_ref)

    loader = EnvironmentSecretLoader({env_key: "supersafe"})
    settings = load_experiment_settings(cfg, secret_loader=loader)

    assert settings.database.driver == "postgresql"
    assert "supersafe" in str(settings.database.uri)
