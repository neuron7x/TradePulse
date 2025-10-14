# SPDX-License-Identifier: MIT
"""Streamlit dashboard for TradePulse indicator diagnostics."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from interfaces.cli import compute_indicator_metrics
from interfaces.dashboard_telemetry import post_metrics
from observability.tracing import get_tracer


def _prepare_dataframe(
    df: pd.DataFrame, window: int
) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Validate and normalise uploaded price data."""

    errors: list[str] = []
    warnings: list[str] = []

    required = {"ts", "price", "volume"}
    missing = required.difference(df.columns)
    if missing:
        errors.append(
            "Missing required columns: " + ", ".join(sorted(missing)) + "."
        )
        return None, errors, warnings

    working = df.copy()

    working["ts"] = pd.to_datetime(working["ts"], errors="coerce")
    invalid_ts = working["ts"].isna().sum()
    if invalid_ts:
        warnings.append(f"Dropped {invalid_ts} rows with invalid timestamps.")
    numeric_cols = ["price", "volume"]
    for col in numeric_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")
        invalid_values = working[col].isna().sum()
        if invalid_values:
            warnings.append(f"Dropped {invalid_values} rows with non-numeric {col} values.")

    working = working.dropna(subset=["ts", "price", "volume"])
    working = working.sort_values("ts").reset_index(drop=True)

    if working.empty:
        errors.append("No valid rows remained after cleansing the dataset.")
        return None, errors, warnings

    if len(working) < window:
        errors.append(
            (
                f"Dataset length ({len(working)}) is smaller than the configured "
                f"lookback window ({window})."
            )
        )

    return working, errors, warnings
st.set_page_config(page_title="TradePulse — Real-time Indicators", layout="wide")
st.title("TradePulse — Real-time Indicators")

with st.sidebar:
    st.header("Configuration")
    window = st.slider("Lookback window", min_value=64, max_value=512, value=200, step=16)
    bins = st.slider("Entropy bins", min_value=16, max_value=256, value=64, step=8)
    delta = st.number_input(
        "Ricci delta",
        min_value=0.0001,
        max_value=0.1,
        value=0.005,
        step=0.0005,
        format="%0.4f",
    )
    use_gpu = st.checkbox(
        "Use GPU acceleration",
        value=False,
        help="Requires CuPy support at runtime",
    )
    backend_url_input = st.text_input(
        "Telemetry API endpoint",
        placeholder="https://telemetry.example.com/ingest",
        help="Optional. When provided the dashboard will POST metrics along with trace context.",
    )
    send_to_backend = st.checkbox(
        "Send metrics to backend",
        value=False,
        help="Enable to POST computed metrics to the configured backend endpoint.",
    )

backend_url = backend_url_input.strip()

uploaded = st.file_uploader(
    "Upload CSV with columns: ts, price, volume",
    type=["csv"],
    key="price_upload",
)

fixture_override = os.getenv("TRADEPULSE_DASHBOARD_TEST_UPLOAD")
fixture_bytes: bytes | None = None
fixture_name: str | None = None
if not uploaded and fixture_override:
    fixture_path = Path(fixture_override).expanduser()
    try:
        fixture_bytes = fixture_path.read_bytes()
    except OSError as exc:
        st.error(f"Failed to read fixture CSV: {exc}")
    else:
        fixture_name = fixture_path.name
        st.info(
            "Loaded fixture data for automated testing: %s" % fixture_name
        )

data_source = uploaded
if not data_source and fixture_bytes is not None:
    buffer = io.BytesIO(fixture_bytes)
    if fixture_name:
        setattr(buffer, "name", fixture_name)
    data_source = buffer

if data_source is not None:
    try:
        raw_df = pd.read_csv(data_source)
    except Exception as exc:  # pragma: no cover - pandas raises specific subclasses
        st.error(f"Failed to read CSV: {exc}")
    else:
        prepared, errors, warnings = _prepare_dataframe(raw_df, window)
        for message in errors:
            st.error(message)
        for message in warnings:
            st.warning(message)

        if errors:
            st.info("Resolve the errors above to inspect indicator diagnostics.")
        else:
            prices = prepared["price"].to_numpy(dtype=float)
            tracer = get_tracer("tradepulse.dashboard")
            with tracer.start_as_current_span(
                "dashboard.compute_metrics",
                attributes={
                    "window": window,
                    "bins": bins,
                    "delta": delta,
                    "gpu_requested": use_gpu,
                },
            ) as span:
                try:
                    metrics = compute_indicator_metrics(
                        prices,
                        window=window,
                        bins=bins,
                        delta=delta,
                        use_gpu=use_gpu,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    span.record_exception(exc)
                    span.set_attributes({"error": True})
                    st.stop()
                span.set_attributes(
                    {
                        "metrics.phase": metrics["phase"],
                        "metrics.gpu_used": metrics["gpu_used"],
                    }
                )

            if use_gpu and not metrics["gpu_used"]:
                st.warning(
                    "GPU acceleration was requested but the CPU implementation was used instead."
                )

            primary_cols = st.columns(3)
            primary_cols[0].metric("Kuramoto R", f"{metrics['R']:.3f}")
            primary_cols[1].metric("Entropy H", f"{metrics['H']:.3f}")
            primary_cols[2].metric("ΔH", f"{metrics['delta_H']:.3f}")

            with st.expander("Advanced metrics", expanded=True):
                advanced_cols = st.columns(3)
                advanced_cols[0].metric("Ricci κ̄", f"{metrics['kappa_mean']:.3f}")
                advanced_cols[1].metric("Hurst exponent", f"{metrics['Hurst']:.3f}")
                advanced_cols[2].metric("Phase regime", metrics["phase"])
                st.caption(
                    "Window: %(window)d · Bins: %(bins)d · Δ: %(delta).4f" % metrics
                )

            chart_df = prepared.set_index("ts")["price"]
            st.line_chart(chart_df, height=320)

            if send_to_backend and not backend_url:
                st.warning("Provide a telemetry endpoint URL to enable metric forwarding.")

            if backend_url and send_to_backend:
                with tracer.start_as_current_span(
                    "dashboard.telemetry_post",
                    attributes={"backend_url": backend_url},
                ) as span:
                    success, status = post_metrics(
                        backend_url,
                        {key: value for key, value in metrics.items() if key != "gpu_used"}
                        | {"gpu_requested": use_gpu},
                    )
                    span.set_attributes({"success": success})
                if success:
                    st.success(status)
                else:
                    st.error(status)
