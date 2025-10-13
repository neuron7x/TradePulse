# Scripts

This directory hosts the consolidated command line tooling for TradePulse.
All commands are exposed through the Python module `scripts.cli`, which can be
invoked with `python -m scripts` from the repository root.  The CLI provides
consistent logging, deterministic defaults and optional environment variable
loading so that workflows behave identically across Linux, macOS and Windows.

## Quick start

```bash
python -m scripts --help
python -m scripts lint --verbose
python -m scripts test --pytest-args -k smoke
python -m scripts gen-proto
python -m scripts dev-up
python -m scripts dev-down
python -m scripts fpma graph
```

### Logging controls

Use `--verbose` (repeatable) to increase logging verbosity and `--quiet`
(repeatable) to reduce noise.  Timestamps are always emitted in ISO 8601 format
using UTC to avoid timezone ambiguity.

### Environment variables

Scripts load configuration from `scripts/.env` when present.  Secrets should not
be committed to the repository; copy [`scripts/.env.example`](./.env.example)
and provide your own values locally.  The loader never echoes variable values to
avoid leaking credentials.

### Deterministic behaviour

On startup the CLI configures the locale and random seeds to deterministic
defaults.  This keeps tool output stable across machines, which in turn
simplifies debugging and makes regression tests easier to reproduce.

## Modules

Each top-level command is implemented in a dedicated module under
[`scripts/commands`](./commands).  The modules provide reusable functions that
can be imported from unit tests or other automation code.

