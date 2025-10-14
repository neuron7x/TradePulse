# Golden Datasets

This directory contains the minimal, versioned datasets that anchor
regression checks for TradePulse. The files are intentionally small so they can
be shipped with the repository and exercised in unit tests or CLI sanity checks
without external dependencies.

## Available Baselines

| Dataset | Description |
| --- | --- |
| `indicator_macd_baseline.csv` | Five minute OHLC close snapshots with pre-computed MACD, signal, and histogram components. |

## Usage

* Run `python scripts/data_sanity.py data/golden` to verify duplicates, missing
  values, spike counts, and timestamp gaps before making changes.
* Use these files to validate indicator pipelines locally – results should match
  the expected MACD, MACD signal line, and MACD histogram values exactly.
* Add new golden files sparingly and prefer the smallest dataset that covers the
  behaviour under test. Document each addition in this README.
