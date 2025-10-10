from __future__ import annotations
import asyncio
import numpy as np
from core.neuro.amm import AMMConfig, AdaptiveMarketMind

def test_precision_drops_with_entropy_and_vol():
    cfg = AMMConfig()
    amm = AdaptiveMarketMind(cfg, use_internal_entropy=False)
    o1 = amm.update(0.001, R_t=0.8, kappa_t=0.2, H_t=0.1)
    o = None
    for _ in range(256):
        o = amm.update(0.02, R_t=0.8, kappa_t=0.2, H_t=1.2)
    assert o["amm_precision"] < o1["amm_precision"]

def test_bursts_trigger_after_shock():
    cfg = AMMConfig()
    amm = AdaptiveMarketMind(cfg)
    pulses = []
    for i in range(600):
        x = 0.0005 if i < 500 else (0.02 if i % 2 == 0 else -0.02)
        out = amm.update(x, R_t=0.7, kappa_t=0.1)
        pulses.append(out["amm_pulse"])
    import numpy as np
    q_hi = np.quantile(np.asarray(pulses[-256:], dtype=np.float32), 0.8)
    assert pulses[-1] >= q_hi * 0.8

def test_async_interface_exists_and_returns():
    cfg = AMMConfig()
    amm = AdaptiveMarketMind(cfg)
    sync_out = amm.update(0.001, 0.6, 0.1, None)
    async def run():
        return await amm.aupdate(0.001, 0.6, 0.1, None)
    async_out = asyncio.get_event_loop().run_until_complete(run())
    for k in ("amm_pulse","amm_precision","amm_valence","pred","pe","entropy"):
        assert k in async_out and isinstance(async_out[k], float)
