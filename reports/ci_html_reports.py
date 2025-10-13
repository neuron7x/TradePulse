"""Generate lightweight HTML dashboards for CI research jobs."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from cli.tradepulse_cli import _run_backtest
from core.config.cli_models import BacktestConfig, DataSourceConfig, ExecutionConfig, StrategyConfig


@dataclass
class HtmlArtifact:
    """Metadata describing a generated HTML report."""

    name: str
    path: Path
    metrics: dict[str, float | int | str]


def _sparkline(values: Sequence[float], *, width: int = 480, height: int = 160, color: str = "#2563eb") -> str:
    """Return a simple inline SVG sparkline for ``values``.

    The implementation avoids heavyweight plotting dependencies so it can run
    inside CI jobs without additional wheels. Values are normalised into the
    provided viewport while preserving proportional changes.
    """

    if not values:
        return "<p>No data available</p>"

    min_v = min(values)
    max_v = max(values)
    span = max(max_v - min_v, 1e-9)
    scale_x = width / max(len(values) - 1, 1)
    points = []
    for idx, value in enumerate(values):
        x = idx * scale_x
        y = height - ((value - min_v) / span) * height
        points.append(f"{x:.2f},{y:.2f}")
    polyline = " ".join(points)
    baseline = height - ((0 - min_v) / span) * height if min_v <= 0 <= max_v else None
    baseline_line = (
        f'<line x1="0" y1="{baseline:.2f}" x2="{width:.2f}" y2="{baseline:.2f}" '
        f'stroke="#e11d48" stroke-width="1" stroke-dasharray="4,4" />'
        if baseline is not None
        else ""
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}" />'
        f"{baseline_line}</svg>"
    )


def _write_html(artifact: HtmlArtifact, body: str) -> Path:
    artifact.path.parent.mkdir(parents=True, exist_ok=True)
    html = (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "  <head>\n"
        "    <meta charset=\"utf-8\" />\n"
        f"    <title>{artifact.name} — CI Report</title>\n"
        "    <style>\n"
        "      body { font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; margin: 2rem; color: #111827; }\n"
        "      h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }\n"
        "      h2 { margin-top: 2rem; font-size: 1.3rem; color: #1f2937; }\n"
        "      table { border-collapse: collapse; margin-top: 1rem; width: 100%; max-width: 680px; }\n"
        "      th, td { border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }\n"
        "      th { background: #f3f4f6; font-weight: 600; }\n"
        "      .metric { font-size: 1.1rem; font-weight: 600; }\n"
        "      .sparkline { margin: 1.5rem 0; }\n"
        "      .summary { background: #f9fafb; border-left: 4px solid #2563eb; padding: 1rem; margin-top: 1.5rem; }\n"
        "      code { background: #eef2ff; padding: 0.2rem 0.35rem; border-radius: 4px; }\n"
        "    </style>\n"
        "  </head>\n"
        "  <body>\n"
        f"    <h1>{artifact.name}</h1>\n"
        f"    <p class=\"summary\">{json.dumps(artifact.metrics, indent=2)}</p>\n"
        f"    {body}\n"
        "  </body>\n"
        "</html>\n"
    )
    artifact.path.write_text(html, encoding="utf-8")
    return artifact.path


def _generate_prices(periods: int = 240, seed: int = 1337) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="1min")
    prices = 100 + rng.normal(0, 0.45, size=periods).cumsum()
    volumes = rng.uniform(80, 250, size=periods)
    return pd.DataFrame({"timestamp": timestamps, "price": prices, "volume": volumes})


def _run_sample_backtest(output_dir: Path) -> HtmlArtifact:
    frame = _generate_prices()
    data_path = output_dir / "backtest_input.csv"
    frame.to_csv(data_path, index=False)

    cfg = BacktestConfig(
        name="ci-backtest",
        data=DataSourceConfig(kind="csv", path=data_path, timestamp_field="timestamp", value_field="price"),
        strategy=StrategyConfig(entrypoint="core.strategies.signals:moving_average_signal", parameters={"window": 12}),
        execution=ExecutionConfig(starting_cash=100_000.0),
        results_path=output_dir / "backtest.json",
    )
    result = _run_backtest(cfg)
    cfg.results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    equity_curve = np.array(result["returns"], dtype=float).cumsum()
    metrics = {
        "total_return": round(float(result["stats"]["total_return"]), 6),
        "max_drawdown": round(float(result["stats"]["max_drawdown"]), 6),
        "trades": int(result["stats"]["trades"]),
    }

    body = """
    <h2>Equity Curve</h2>
    <div class=\"sparkline\">{sparkline}</div>
    <h2>Backtest Metrics</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Total Return</td><td class=\"metric\">{total_return}</td></tr>
        <tr><td>Max Drawdown</td><td class=\"metric\">{max_drawdown}</td></tr>
        <tr><td>Trades</td><td class=\"metric\">{trades}</td></tr>
      </tbody>
    </table>
    <p>The curve renders the cumulative return progression for the synthetic walk-forward sample used in CI.</p>
    """.format(
        sparkline=_sparkline(equity_curve.tolist(), color="#10b981"),
        total_return=metrics["total_return"],
        max_drawdown=metrics["max_drawdown"],
        trades=metrics["trades"],
    )

    artifact = HtmlArtifact(name="Backtest Job Report", path=output_dir / "backtest_report.html", metrics=metrics)
    _write_html(artifact, body)
    return artifact


def _run_sample_training(output_dir: Path) -> HtmlArtifact:
    frame = _generate_prices(seed=4242)
    data_path = output_dir / "train_input.csv"
    frame.to_csv(data_path, index=False)

    backtest_cfg = BacktestConfig(
        name="ci-train-backtest",
        data=DataSourceConfig(kind="csv", path=data_path, timestamp_field="timestamp", value_field="price"),
        strategy=StrategyConfig(entrypoint="core.strategies.signals:moving_average_signal", parameters={"window": 8}),
        execution=ExecutionConfig(starting_cash=50_000.0),
        results_path=output_dir / "train_backtest.json",
    )

    window_options = [4, 6, 8, 12, 16]
    trials: list[dict[str, float | dict[str, int]]] = []
    for window in window_options:
        candidate_cfg = backtest_cfg.model_copy(deep=True)
        candidate_cfg.strategy.parameters["window"] = window
        result = _run_backtest(candidate_cfg)
        returns = np.asarray(result["returns"], dtype=float)
        score = float(np.mean(returns) / (np.std(returns) + 1e-9))
        trials.append({"window": window, "score": score, "total_return": float(np.sum(returns))})

    best = max(trials, key=lambda trial: trial["score"])
    sparkline = _sparkline([trial["score"] for trial in trials], color="#7c3aed")
    body_rows = "\n".join(
        f"<tr><td>{trial['window']}</td><td>{trial['score']:.6f}</td><td>{trial['total_return']:.6f}</td></tr>"
        for trial in trials
    )
    body = f"""
    <h2>Optimisation Score Distribution</h2>
    <div class=\"sparkline\">{sparkline}</div>
    <h2>Trial Summary</h2>
    <table>
      <thead><tr><th>Window</th><th>Sharpe Proxy</th><th>Total Return</th></tr></thead>
      <tbody>
        {body_rows}
      </tbody>
    </table>
    <p>Scores are computed using a simple Sharpe proxy on the generated walk-forward returns.</p>
    """

    artifact = HtmlArtifact(
        name="Training Job Report",
        path=output_dir / "train_report.html",
        metrics={
            "best_window": best["window"],
            "best_score": round(float(best["score"]), 6),
            "mean_score": round(statistics.mean(trial["score"] for trial in trials), 6),
            "trials": len(trials),
        },
    )
    _write_html(artifact, body)
    return artifact


def _load_release_metadata() -> dict[str, str | int]:
    version = Path("VERSION").read_text(encoding="utf-8").strip()
    pyproject_path = Path("pyproject.toml")
    dependencies = 0
    if pyproject_path.exists():
        import tomllib

        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        dependencies = len(data.get("project", {}).get("dependencies", []))
    wheel_targets = len(list(Path("tools").rglob("*.py")))
    return {"version": version, "declared_dependencies": dependencies, "tooling_modules": wheel_targets}


def _run_publish_snapshot(output_dir: Path) -> HtmlArtifact:
    metadata = _load_release_metadata()
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    requirement_count = sum(1 for line in requirements if line.strip() and not line.startswith("#"))
    metadata["locked_requirements"] = requirement_count
    metadata["sbom_target"] = "sbom/cyclonedx-sbom.json"

    dependency_preview = "".join(f"<li><code>{line}</code></li>" for line in requirements[:10] if line.strip())
    body = f"""
    <h2>Release Snapshot</h2>
    <table>
      <thead><tr><th>Property</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Version</td><td class=\"metric\">{metadata['version']}</td></tr>
        <tr><td>Declared Dependencies</td><td class=\"metric\">{metadata['declared_dependencies']}</td></tr>
        <tr><td>Locked Requirements</td><td class=\"metric\">{metadata['locked_requirements']}</td></tr>
        <tr><td>Tooling Modules</td><td class=\"metric\">{metadata['tooling_modules']}</td></tr>
        <tr><td>SBOM Artifact</td><td><code>{metadata['sbom_target']}</code></td></tr>
      </tbody>
    </table>
    <h2>Dependency Preview</h2>
    <ul>{dependency_preview}</ul>
    <p>Full dependency inventory is captured in the CycloneDX SBOM generated during release verification.</p>
    """

    artifact = HtmlArtifact(name="Publish Job Report", path=output_dir / "publish_report.html", metrics=metadata)
    _write_html(artifact, body)
    return artifact


def generate_reports(target: str, output_dir: Path) -> list[HtmlArtifact]:
    output_dir = output_dir.resolve()
    generators = {
        "backtest": _run_sample_backtest,
        "train": _run_sample_training,
        "publish": _run_publish_snapshot,
    }
    if target == "all":
        return [generators[name](output_dir) for name in ("backtest", "train", "publish")]
    if target not in generators:
        raise ValueError(f"Unknown target '{target}'")
    return [generators[target](output_dir)]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CI HTML reports")
    parser.add_argument("--target", choices=["backtest", "train", "publish", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ci-html"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> list[HtmlArtifact]:
    args = _parse_args(argv)
    return generate_reports(args.target, args.output_dir)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
