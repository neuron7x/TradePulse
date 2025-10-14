import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from analytics import amm_metrics


@pytest.fixture
def metrics_registry(monkeypatch):
    registry = CollectorRegistry()
    g_pulse = Gauge("amm_pulse", "AMM pulse intensity", ["symbol", "tf"], registry=registry)
    g_prec = Gauge("amm_precision", "AMM precision", ["symbol", "tf"], registry=registry)
    g_gain = Gauge("amm_gain", "AMM adaptive gain k", ["symbol", "tf"], registry=registry)
    g_theta = Gauge("amm_threshold", "AMM adaptive threshold theta", ["symbol", "tf"], registry=registry)
    c_burst = Counter("amm_bursts_total", "AMM high-pulse bursts", ["symbol", "tf"], registry=registry)
    h_update = Histogram(
        "amm_update_seconds", "AMM update latency seconds", ["symbol", "tf"], registry=registry
    )

    monkeypatch.setattr(amm_metrics, "_g_pulse", g_pulse)
    monkeypatch.setattr(amm_metrics, "_g_prec", g_prec)
    monkeypatch.setattr(amm_metrics, "_g_gain", g_gain)
    monkeypatch.setattr(amm_metrics, "_g_theta", g_theta)
    monkeypatch.setattr(amm_metrics, "_c_burst", c_burst)
    monkeypatch.setattr(amm_metrics, "_h_update", h_update)

    return {
        "registry": registry,
        "pulse": g_pulse,
        "precision": g_prec,
        "gain": g_gain,
        "theta": g_theta,
        "burst": c_burst,
        "update": h_update,
    }


def test_timed_update_records_single_observation(monkeypatch, metrics_registry):
    symbol = "BTC"
    tf = "1h"
    perf_values = iter([1.0, 3.5])
    monkeypatch.setattr(amm_metrics.time, "perf_counter", lambda: next(perf_values))

    with amm_metrics.timed_update(symbol, tf):
        pass

    count = metrics_registry["registry"].get_sample_value(
        "amm_update_seconds_count", {"symbol": symbol, "tf": tf}
    )
    assert count == 1.0


def test_publish_metrics_sets_gauges_and_increments_counter(metrics_registry):
    symbol = "ETH"
    tf = "5m"
    out = {"amm_pulse": 2.5, "amm_precision": 0.85}
    amm_metrics.publish_metrics(symbol, tf, out, k=1.2, theta=0.4, q_hi=2.0)

    registry = metrics_registry["registry"]
    assert registry.get_sample_value("amm_pulse", {"symbol": symbol, "tf": tf}) == out["amm_pulse"]
    assert registry.get_sample_value("amm_precision", {"symbol": symbol, "tf": tf}) == out["amm_precision"]
    assert registry.get_sample_value("amm_gain", {"symbol": symbol, "tf": tf}) == 1.2
    assert registry.get_sample_value("amm_threshold", {"symbol": symbol, "tf": tf}) == 0.4
    burst_count = registry.get_sample_value("amm_bursts_total", {"symbol": symbol, "tf": tf})
    assert burst_count == 1.0
