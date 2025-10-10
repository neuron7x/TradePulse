from __future__ import annotations

import asyncio
import numpy as np
import pytest

from core.neuro.amm import AMMConfig, AdaptiveMarketMind


def test_precision_penalized_by_entropy_and_desync():
    cfg = AMMConfig(alpha=0.05, beta=1.0, lambda_sync=0.5, eta_ricci=0.3, pi_max=100.0, pi_min=1e-6)
    amm = AdaptiveMarketMind(cfg, use_internal_entropy=False, R_bar=0.5)
    rng = np.random.default_rng(1)
    for _ in range(500):
        amm.update(float(rng.normal(0.0, 0.05)), R_t=0.5, kappa_t=0.0, H_t=0.0)

    baseline = amm.update(0.001, R_t=0.5, kappa_t=0.0, H_t=0.0)["amm_precision"]
    stressed = amm.update(0.001, R_t=0.2, kappa_t=-0.5, H_t=1.5)["amm_precision"]
    assert stressed < baseline


def test_bursts_trigger_after_shock():
    cfg = AMMConfig()
    amm = AdaptiveMarketMind(cfg)
    pulses = []
    for i in range(600):
        x = 0.0005 if i < 500 else (0.02 if i % 2 == 0 else -0.02)
        out = amm.update(x, R_t=0.7, kappa_t=0.1)
        pulses.append(out["amm_pulse"])
    q_hi = np.quantile(np.asarray(pulses[-256:], dtype=np.float32), 0.8)
    assert pulses[-1] >= q_hi * 0.8


def test_async_interface_exists_and_returns():
    cfg = AMMConfig()

    async def run() -> tuple[dict, dict]:
        amm_sync = AdaptiveMarketMind(cfg)
        amm_async = AdaptiveMarketMind(cfg)
        sync_out = amm_sync.update(0.001, 0.6, 0.1, None)
        async_out = await amm_async.aupdate(0.001, 0.6, 0.1, None)
        return sync_out, async_out

    sync_out, async_out = asyncio.run(run())
    for k in ("amm_pulse", "amm_precision", "amm_valence", "pred", "pe", "entropy"):
        assert k in async_out and isinstance(async_out[k], float)
        assert pytest.approx(sync_out[k], rel=1e-5, abs=1e-5) == async_out[k]


def test_batch_matches_stream_updates():
    cfg = AMMConfig()
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 0.01, 64).astype(np.float32)
    R = rng.uniform(0.3, 0.7, 64).astype(np.float32)
    kappa = rng.normal(0.0, 0.2, 64).astype(np.float32)

    amm = AdaptiveMarketMind(cfg)
    seq = {k: [] for k in ("amm_pulse", "amm_precision", "amm_valence", "pred", "pe", "entropy")}
    for i in range(len(x)):
        out = amm.update(float(x[i]), float(R[i]), float(kappa[i]), None)
        for k in seq:
            seq[k].append(out[k])

    batched = AdaptiveMarketMind.batch(cfg, x, R, kappa, None)
    for k in seq:
        assert np.allclose(batched[k], np.asarray(seq[k], dtype=np.float32), atol=1e-6)
