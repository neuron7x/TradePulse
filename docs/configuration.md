# Configuration System

TradePulse uses a unified configuration layer powered by
[`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
Every service, domain module and CLI tool now reads its settings through the same
`TradePulseSettings` model which merges values from multiple sources.

## Source precedence

Configuration values are resolved in the following order (highest priority first):

1. **Command line overrides** – values passed to `load_kuramoto_ricci_config(..., cli_overrides=...)`
   or via the new `--config-override` flag in CLI utilities.
2. **Process environment variables** – variables prefixed with `TRADEPULSE_`.
3. **`.env` files** – the loader automatically reads a `.env` file from the current
   working directory if present.
4. **YAML files** – the base configuration is read from the file referenced by
   `config_file` (defaults to `configs/kuramoto_ricci_composite.yaml`).

Missing sources are skipped gracefully, so removing a `.env` file or YAML preset simply
falls back to default values embedded in the models.

## YAML layout

The YAML structure mirrors the Pydantic models. A minimal example looks like:

```yaml
kuramoto:
  timeframes: ["M1", "M5", "M15", "H1"]
  adaptive_window:
    enabled: true
    base_window: 200
  min_samples_per_scale: 64
ricci:
  temporal:
    window_size: 100
    n_snapshots: 8
    retain_history: true
  graph:
    n_levels: 20
    connection_threshold: 0.1
composite:
  thresholds:
    R_strong_emergent: 0.8
    R_proto_emergent: 0.4
    coherence_min: 0.6
    ricci_negative: -0.3
    temporal_ricci: -0.2
    topological_transition: 0.7
  signals:
    min_confidence: 0.5
```

All values are validated by the Pydantic models; invalid inputs raise a
`ConfigError` with a descriptive message.

## CLI usage

The Kuramoto–Ricci integration script demonstrates CLI overrides:

```bash
python scripts/integrate_kuramoto_ricci.py \
  --data sample.csv \
  --config configs/kuramoto_ricci_composite.yaml \
  --config-override kuramoto.base_window=256 \
  --config-override composite.thresholds.R_strong_emergent=0.9
```

`--config-override` accepts dot-delimited keys and `YAML` expressions for values. Lists
and booleans can be expressed naturally, for example
`--config-override kuramoto.timeframes=['M1','M5','H1']`.

## Environment variables and `.env`

Environment variables use the `TRADEPULSE_` prefix and `__` as the nested delimiter:

```bash
export TRADEPULSE_KURAMOTO__BASE_WINDOW=300
export TRADEPULSE_COMPOSITE__THRESHOLDS__R_STRONG_EMERGENT=0.85
```

The same syntax works inside a `.env` file located in the working directory. When present,
values inside `.env` override the YAML baseline but remain below live environment variables
and CLI overrides.

## Programmatic access

Python modules should use `load_kuramoto_ricci_config` to obtain a fully merged
`KuramotoRicciIntegrationConfig` instance:

```python
from core.config import load_kuramoto_ricci_config

cfg = load_kuramoto_ricci_config("configs/kuramoto_ricci_composite.yaml")
engine_kwargs = cfg.to_engine_kwargs()
```

For scripts that parse their own CLI arguments, convert `key=value` pairs with
`parse_cli_overrides` and forward the mapping to the loader:

```python
from core.config import load_kuramoto_ricci_config, parse_cli_overrides

overrides = parse_cli_overrides(["kuramoto.base_window=512"])
cfg = load_kuramoto_ricci_config("configs/custom.yaml", cli_overrides=overrides)
```

This keeps all configuration handling centralised and ensures new sources or validation
rules are automatically applied across the code base.
