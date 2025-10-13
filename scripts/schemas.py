"""Shared schema helpers for repository scripts."""

from __future__ import annotations

# SPDX-License-Identifier: MIT

from typing import Any

try:  # pragma: no cover - exercised only when pandera is available
    import pandera as pa
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "scripts.schemas requires pandera. Install TradePulse's base requirements first."
    ) from exc


def get_data_sanity_input_schema() -> pa.DataFrameSchema:
    """Return a strict schema for CSV inspection DataFrames."""

    return pa.DataFrameSchema(
        {
            "ts": pa.Column(pa.DateTime, nullable=True),
            "price": pa.Column(pa.Float, nullable=True),
            "volume": pa.Column(pa.Float, nullable=True),
        },
        coerce=True,
        strict=False,
        name="data_sanity_input",
    )


def validate_payload(schema_ref: Any, payload: Any) -> None:
    """Validate *payload* against a :mod:`pandera` schema."""

    schema_ref.validate(payload)
