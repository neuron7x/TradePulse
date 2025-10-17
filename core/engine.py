"""Hydra orchestrated trading engine entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from core.backtest import Backtester, BarData
from core.execution import ExecutionModel
from core.mlflow_utils import mlflow
from core.tca import TCA
from data.loaders import CSVLoaderConfig, load_csv_bars
from risk.limits import LimitConfig
from risk.manager import RiskManager
from strategies import (
    AMMConfig,
    AMMStrategy,
    MeanReversionConfig,
    MeanReversionStrategy,
    TrendStrategy,
    TrendStrategyConfig,
)


@dataclass
class TradingConfig:
    strategy: Dict[str, Any]
    risk: Dict[str, Any]
    exec: Dict[str, Any]
    data: Dict[str, Any]


ConfigStore.instance().store(name="trading_config", node=TradingConfig)


class TradingEngine:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def _build_strategy(self):
        kind = self.cfg.strategy.get("kind")
        params = {k: v for k, v in self.cfg.strategy.items() if k != "kind"}
        if kind == "trend":
            return TrendStrategy(TrendStrategyConfig(**params))
        if kind == "meanrev":
            return MeanReversionStrategy(MeanReversionConfig(**params))
        if kind == "amm":
            return AMMStrategy(AMMConfig(**params))
        raise ValueError(f"Unsupported strategy kind: {kind}")

    def _build_risk(self) -> RiskManager:
        cfg = LimitConfig(**self.cfg.risk)
        return RiskManager(cfg)

    def _build_execution(self) -> ExecutionModel:
        exec_cfg = self.cfg.exec
        return ExecutionModel(
            venue=exec_cfg.get("venue", "backtest"),
            order_type=exec_cfg.get("order_type", "market"),
            slippage_bps=exec_cfg.get("slippage_bps", 3.0),
        )

    def _load_data(self) -> list[BarData]:
        data_cfg = self.cfg.data
        if "path" not in data_cfg:
            raise ValueError("Data path must be provided")
        csv_cfg = CSVLoaderConfig(path=Path(data_cfg["path"]))
        return load_csv_bars(csv_cfg)

    def run_backtest(self) -> Dict[str, Any]:
        strategy = self._build_strategy()
        risk_manager = self._build_risk()
        execution_model = self._build_execution()
        backtester = Backtester(execution_model, risk_manager, TCA())
        bars = self._load_data()
        result = backtester.run(strategy, bars)
        tca_summary = backtester.tca.summary()
        payload = {
            "pnl": result.pnl,
            "cagr": result.cagr,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "calmar": result.calmar,
            "hit_rate": result.hit_rate,
            "tca": tca_summary,
        }
        return payload


@hydra.main(config_path="conf", config_name="trading", version_base=None)
def run(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("core_strats_2025")
    cfg_flat = OmegaConf.to_container(cfg, resolve=True)
    engine = TradingEngine(cfg)
    with mlflow.start_run(run_name=cfg.strategy.get("kind", "strategy")):
        mlflow.log_params({"git_sha": hydra.utils.get_original_cwd()})
        mlflow.log_dict(cfg_flat, "hydra_cfg.json")
        result = engine.run_backtest()
        mlflow.log_metrics({k: float(v) for k, v in result.items() if isinstance(v, (int, float))})
        mlflow.log_dict(result["tca"], "tca_summary.json")


if __name__ == "__main__":
    run()

