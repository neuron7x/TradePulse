from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .amm import AdaptiveMarketMind, AMMConfig

Float = np.float32

@dataclass
class CalibConfig:
    iters: int = 200
    seed: int = 7
    ema_span: tuple[int,int] = (8, 96)
    vol_lambda: tuple[float,float] = (0.86, 0.98)
    alpha: tuple[float,float] = (0.2, 5.0)
    beta: tuple[float,float] = (0.1, 2.0)
    lambda_sync: tuple[float,float] = (0.2, 1.2)
    eta_ricci: tuple[float,float] = (0.1, 1.0)
    rho: tuple[float,float] = (0.01, 0.12)

def _rand(rng: np.random.Generator, lo_hi: tuple[float,float], is_int=False):
    lo, hi = lo_hi
    if is_int: return int(rng.integers(lo, hi+1))
    return float(rng.uniform(lo, hi))

def calibrate_random(x: np.ndarray, R: np.ndarray, kappa: np.ndarray, cfg: CalibConfig) -> AMMConfig:
    rng = np.random.default_rng(cfg.seed)
    best_val = -1e18; best: AMMConfig | None = None
    for _ in range(cfg.iters):
        c = AMMConfig(
            ema_span=_rand(rng, cfg.ema_span, is_int=True),
            vol_lambda=_rand(rng, cfg.vol_lambda),
            alpha=_rand(rng, cfg.alpha),
            beta=_rand(rng, cfg.beta),
            lambda_sync=_rand(rng, cfg.lambda_sync),
            eta_ricci=_rand(rng, cfg.eta_ricci),
            rho=_rand(rng, cfg.rho),
        )
        amm = AdaptiveMarketMind(c)
        S,P,PE = [],[],[]
        for i in range(len(x)):
            o = amm.update(float(x[i]), float(R[i]), float(kappa[i]), None)
            S.append(o["amm_pulse"]); P.append(o["amm_precision"]); PE.append(abs(o["pe"]))
        S = np.asarray(S, dtype=Float); P = np.asarray(P, dtype=Float); PE = np.asarray(PE, dtype=Float)
        val = float(np.corrcoef(PE, S)[0,1]) * float(np.mean(np.clip(P, 0.01, 100.0)))
        if np.isnan(val): continue
        if val > best_val:
            best_val = val; best = c
    return best if best is not None else AMMConfig()
