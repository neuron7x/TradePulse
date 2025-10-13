"""Generate tutorial-friendly sample datasets with manifest tracking.

The script wraps :mod:`scripts.gen_synth_amm_data` and complements it with a
structured manifest describing the generated artefacts.  It doubles as a
walk-through for newcomers: reading the manifest illustrates how the wider
TradePulse platform keeps artefacts versioned and discoverable.
"""

from __future__ import annotations

# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

try:  # pragma: no cover - executed when called via ``python scripts/...``
    from . import ScriptRegistry, ScriptRunner
except ImportError:  # pragma: no cover - fallback when package context is absent
    import sys
    from pathlib import Path

    _MODULE_ROOT = Path(__file__).resolve().parent.parent
    if str(_MODULE_ROOT) not in sys.path:
        sys.path.insert(0, str(_MODULE_ROOT))
    from scripts import ScriptRegistry, ScriptRunner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tutorial/output"),
        help="Directory where tutorial artefacts will be written.",
    )
    parser.add_argument(
        "--format",
        choices={"csv", "parquet"},
        default="csv",
        help="File format to produce (default: csv).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1_000,
        help="Number of rows to synthesise for each dataset (default: 1000).",
    )
    return parser.parse_args(argv)


def _generate_price_series(rows: int) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="min")
    df = pd.DataFrame(
        {
            "timestamp": index,
            "price": 100 + pd.Series(range(rows)).rolling(15, min_periods=1).mean(),
            "volume": pd.Series(range(rows)).pow(0.5),
        }
    )
    return df


def _write_dataset(df: pd.DataFrame, path: Path, *, file_format: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    price_series = _generate_price_series(args.rows)
    price_path = _write_dataset(
        price_series, args.output / f"price_series.{args.format}", file_format=args.format
    )

    # Leverage the registry infrastructure to demonstrate cross-script reuse.
    registry = ScriptRegistry.from_path()
    runner = ScriptRunner(registry)
    synth_manifest = runner.run(
        "gen_synth_amm_data",
        extra_args=[],
        dry_run=True,
        manifest_path=args.output / "amm_manifest.json",
        metrics_path=args.output / "amm_metrics.prom",
    )

    manifest = {
        "price_series": str(price_path),
        "amm_data": str(Path("/mnt/data/amm_synth.csv")),
        "format": args.format,
        "rows": args.rows,
        "registry_version": registry.version,
        "synth_manifest": str(synth_manifest.manifest_path) if synth_manifest.manifest_path else None,
    }

    manifest_path = args.output / "tutorial_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Tutorial artefacts written to {args.output}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
