
import argparse, pandas as pd, numpy as np, yaml, os
from pathlib import Path
from typing import Dict, Optional
from core.indicators.kuramoto_ricci_composite import TradePulseCompositeEngine

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Kuramoto–Ricci Composite Integration")
    ap.add_argument("--data", type=str, required=True, help="CSV with at least 'close', optionally 'volume'")
    ap.add_argument("--config", type=str, default="configs/kuramoto_ricci_composite.yaml")
    ap.add_argument("--mode", choices=["analyze","backtest"], default="analyze")
    ap.add_argument("--output", type=str, default="outputs")
    args = ap.parse_args()

    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    if "volume" not in df.columns:
        df["volume"] = 1.0

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    kcfg = {"timeframes": cfg.get("kuramoto",{}).get("timeframes",[60,300,900,3600]),
            "use_adaptive_window": cfg.get("kuramoto",{}).get("adaptive_window",{}).get("enabled", True),
            "base_window": cfg.get("kuramoto",{}).get("adaptive_window",{}).get("base_window", 200)}
    rcfg = {"window_size": cfg.get("ricci",{}).get("temporal",{}).get("window_size",100),
            "n_snapshots": cfg.get("ricci",{}).get("temporal",{}).get("n_snapshots",8),
            "n_levels": cfg.get("ricci",{}).get("graph",{}).get("n_levels",20)}
    ccfg = {"R_strong_emergent": cfg.get("composite",{}).get("thresholds",{}).get("R_strong_emergent",0.8),
            "R_proto_emergent":  cfg.get("composite",{}).get("thresholds",{}).get("R_proto_emergent",0.4),
            "coherence_threshold": cfg.get("composite",{}).get("thresholds",{}).get("coherence_min",0.6),
            "ricci_negative_threshold": cfg.get("composite",{}).get("thresholds",{}).get("ricci_negative",-0.3),
            "temporal_ricci_threshold": cfg.get("composite",{}).get("thresholds",{}).get("temporal_ricci",-0.2),
            "transition_threshold": cfg.get("composite",{}).get("thresholds",{}).get("topological_transition",0.7),
            "min_confidence": cfg.get("composite",{}).get("signals",{}).get("min_confidence",0.5)}
    engine = TradePulseCompositeEngine(kuramoto_config=kcfg, ricci_config=rcfg, composite_config=ccfg)
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
