
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import load_kuramoto_ricci_config, parse_cli_overrides
from core.indicators.kuramoto_ricci_composite import TradePulseCompositeEngine

def main():
    ap = argparse.ArgumentParser(description="Kuramoto–Ricci Composite Integration")
    ap.add_argument("--data", type=str, required=True, help="CSV with at least 'close', optionally 'volume'")
    ap.add_argument("--config", type=str, default="configs/kuramoto_ricci_composite.yaml")
    ap.add_argument("--mode", choices=["analyze","backtest"], default="analyze")
    ap.add_argument("--output", type=str, default="outputs")
    ap.add_argument(
        "--config-override",
        action="append",
        dest="config_overrides",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override configuration values using dot-delimited keys. "
            "Example: --config-override kuramoto.base_window=256"
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    if "volume" not in df.columns:
        df["volume"] = 1.0

    overrides = parse_cli_overrides(args.config_overrides)
    cfg = load_kuramoto_ricci_config(args.config, cli_overrides=overrides)
    engine = TradePulseCompositeEngine(**cfg.to_engine_kwargs())
    sig = engine.analyze_market(df)

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    # Save outputs
    engine.get_signal_dataframe().to_csv(outdir / "signal_history.csv", index=False)
    df_out = df.copy()
    df_out.loc[df_out.index[-1],"phase"] = sig.phase.value
    df_out.loc[df_out.index[-1],"entry_signal"] = sig.entry_signal
    df_out.loc[df_out.index[-1],"confidence"] = sig.confidence
    df_out.to_csv(outdir / "enhanced_features.csv")

    print(f"Phase: {sig.phase.value}")
    print(f"Confidence: {sig.confidence:.3f}")
    print(f"Entry: {sig.entry_signal:.3f} | Exit: {sig.exit_signal:.3f} | Risk: {sig.risk_multiplier:.3f}")
    print(f"Kuramoto R: {sig.kuramoto_R:.3f}, Coherence: {sig.cross_scale_coherence:.3f}")
    print(f"Static κ: {sig.static_ricci:.4f}, Temporal κ_t: {sig.temporal_ricci:.4f}, Transition: {sig.topological_transition:.3f}")

if __name__ == "__main__":
    main()
