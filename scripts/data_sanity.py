"""Quick data quality sanity checks for CSV files.

This module started life as a small ad-hoc script but has since been tidied up
so it is easier to extend, test and run as part of CI workflows.  The
improvements include:

* Fully typed, documented helper functions that can be imported from tests.
* Structured analysis results that make downstream processing easier.
* Extra quality metrics including constant column detection and dtype previews.
* A tiny CLI powered by :mod:`argparse` for filtering input files and tweaking
  behaviour without editing the script.

Typical usage from the repository root::

    python scripts/data_sanity.py data --pattern "**/*.csv"

"""

from __future__ import annotations

# SPDX-License-Identifier: MIT

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class TimestampGapStats:
    """Summary statistics describing gaps between consecutive timestamps."""

    median_seconds: float
    max_seconds: float
    min_seconds: float
    negative_gap_count: int


@dataclass(frozen=True)
class CSVAnalysis:
    """Container for the computed quality metrics of a CSV file."""

    path: Path
    row_count: int
    column_count: int
    nan_ratio: float
    row_nan_ratio: float
    duplicate_rows: int
    column_nan_ratios: dict[str, float]
    constant_columns: list[str]
    dtypes: dict[str, str]
    timestamp_gap_stats: TimestampGapStats | None


def _iter_csv_files(paths: Sequence[Path], pattern: str) -> Iterable[Path]:
    """Yield CSV files contained in *paths*.

    When *paths* is empty the function falls back to the default ``data``
    directory.  Directory arguments are walked recursively using :meth:`Path.rglob`.
    """

    if not paths:
        default_root = Path("data")
        if not default_root.exists():
            return
        paths = [default_root]

    seen: set[Path] = set()
    for path in paths:
        if path.is_dir():
            candidates = path.rglob(pattern)
        else:
            candidates = [path]

        for candidate in candidates:
            if candidate.suffix.lower() != ".csv":
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def analyze_csv(path: Path, timestamp_column: str | None = "ts") -> CSVAnalysis:
    """Inspect a CSV file and derive lightweight quality metrics."""

    df = pd.read_csv(path)
    row_count, column_count = df.shape
    nan_ratio = float(df.isna().mean().mean()) if column_count else 0.0
    row_nan_ratio = float(df.isna().any(axis=1).mean()) if row_count else 0.0
    duplicate_rows = int(df.duplicated().sum())

    timestamp_gap_stats: TimestampGapStats | None = None
    if timestamp_column and timestamp_column in df.columns:
        ts = pd.to_datetime(df[timestamp_column], errors="coerce")
        gaps = ts.diff().dt.total_seconds().dropna()
        if not gaps.empty:
            negative_gap_count = int((gaps < 0).sum())
            timestamp_gap_stats = TimestampGapStats(
                median_seconds=float(gaps.median()),
                max_seconds=float(gaps.max()),
                min_seconds=float(gaps.min()),
                negative_gap_count=negative_gap_count,
            )

    per_column_nan = (
        df.isna()
        .mean()
        .loc[lambda series: series.gt(0)]
        .sort_values(ascending=False)
        .to_dict()
    )

    non_null_uniques = df.nunique(dropna=True)
    constant_columns = [
        column
        for column, unique_count in non_null_uniques.items()
        if unique_count <= 1 and df[column].notna().any()
    ]

    dtypes = {column: str(dtype) for column, dtype in df.dtypes.items()}

    return CSVAnalysis(
        path=Path(path),
        row_count=int(row_count),
        column_count=int(column_count),
        nan_ratio=nan_ratio,
        row_nan_ratio=row_nan_ratio,
        duplicate_rows=duplicate_rows,
        column_nan_ratios=per_column_nan,
        constant_columns=constant_columns,
        dtypes=dtypes,
        timestamp_gap_stats=timestamp_gap_stats,
    )


def _format_column_nan_ratios(column_nan_ratios: dict[str, float], limit: int) -> str | None:
    if not column_nan_ratios:
        return None

    items = list(column_nan_ratios.items())[:limit]
    formatted = ", ".join(f"{column}={ratio:.2%}" for column, ratio in items)
    if len(column_nan_ratios) > limit:
        formatted += ", …"
    return formatted


def _format_constant_columns(columns: Sequence[str], limit: int) -> str | None:
    if not columns:
        return None

    columns = list(columns)
    displayed = ", ".join(columns[:limit])
    if len(columns) > limit:
        displayed += ", …"
    return displayed


def format_analysis(
    analysis: CSVAnalysis,
    *,
    max_column_details: int = 5,
    max_constant_columns: int = 5,
) -> str:
    """Convert :class:`CSVAnalysis` into the original human readable format."""

    report_lines = [
        f"# File: {analysis.path}",
        f"- rows: {analysis.row_count}; cols: {analysis.column_count}",
        f"- NaN ratio (avg): {analysis.nan_ratio:.4f}",
        f"- rows with any NaNs: {analysis.row_nan_ratio:.2%}",
    ]

    column_details = _format_column_nan_ratios(
        analysis.column_nan_ratios, max_column_details
    )
    if column_details:
        report_lines.append(f"- column NaN ratios: {column_details}")

    constant_columns = _format_constant_columns(
        analysis.constant_columns, max_constant_columns
    )
    if constant_columns:
        report_lines.append(f"- constant columns: {constant_columns}")

    if analysis.timestamp_gap_stats:
        report_lines.append(
            "- median gap (s): "
            f"{analysis.timestamp_gap_stats.median_seconds:.3f}; "
            "max gap (s): "
            f"{analysis.timestamp_gap_stats.max_seconds:.3f}; "
            "min gap (s): "
            f"{analysis.timestamp_gap_stats.min_seconds:.3f}; "
            "negative gaps: "
            f"{analysis.timestamp_gap_stats.negative_gap_count}"
        )

    report_lines.append(f"- duplicates: {analysis.duplicate_rows}")

    if analysis.dtypes:
        dtype_preview = ", ".join(
            f"{column}={dtype}" for column, dtype in list(analysis.dtypes.items())[:5]
        )
        report_lines.append(f"- dtypes: {dtype_preview}")
    return "\n".join(report_lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "CSV files or directories to inspect.  When omitted the script "
            "searches the 'data' directory."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="**/*.csv",
        help="Glob-style pattern used when walking directories (default: **/*.csv).",
    )
    parser.add_argument(
        "--timestamp-column",
        default="ts",
        help="Column containing timestamps used for gap statistics (default: ts).",
    )
    parser.add_argument(
        "--max-column-details",
        type=int,
        default=5,
        help="Maximum number of per-column NaN ratios to display (default: 5).",
    )
    parser.add_argument(
        "--max-constant-columns",
        type=int,
        default=5,
        help="Maximum number of constant columns to display (default: 5).",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a non-zero exit status if any file fails to parse.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    analyses: List[str] = []
    exit_code = 0

    csv_files = sorted(_iter_csv_files(args.paths, args.pattern) or [])

    if not csv_files:
        print("No CSV files found — nothing to check (OK).")
        return exit_code

    for csv_file in csv_files:
        try:
            analysis = analyze_csv(csv_file, args.timestamp_column)
        except Exception as exc:  # pragma: no cover - defensive: pandas error message varies
            analyses.append(f"# File: {csv_file}\n- ERROR: {exc}")
            if args.fail_on_error:
                exit_code = 1
        else:
            analyses.append(
                format_analysis(
                    analysis,
                    max_column_details=max(1, args.max_column_details),
                    max_constant_columns=max(1, args.max_constant_columns),
                )
            )

    print("\n\n".join(analyses))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
