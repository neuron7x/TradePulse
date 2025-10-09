# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

try:
    from hypothesis import given, strategies as st
except Exception:  # pragma: no cover - hypothesis optional in runtime envs
    pytest.skip("hypothesis not installed", allow_module_level=True)

from core.utils.schemas import dataclass_to_json_schema, validate_against_schema


@dataclass
class PropertySample:
    timestamp: float
    price: float
    symbol: str
    tags: list[str]
    metadata: dict[str, int] = field(default_factory=dict)
    active: bool = True
    note: Optional[str] = None


SCHEMA = dataclass_to_json_schema(PropertySample)


@given(
    timestamp=st.floats(allow_nan=False, allow_infinity=False, width=32),
    price=st.floats(allow_nan=False, allow_infinity=False, width=32),
    symbol=st.text(min_size=1, max_size=12),
    tags=st.lists(st.text(min_size=1, max_size=8), min_size=0, max_size=5),
    metadata=st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.integers(min_value=0, max_value=1000),
        max_size=5,
    ),
    active=st.booleans(),
    note=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
)
def test_generated_payloads_validate_against_schema(
    timestamp: float,
    price: float,
    symbol: str,
    tags: list[str],
    metadata: dict[str, int],
    active: bool,
    note: Optional[str],
) -> None:
    payload = {
        "timestamp": float(timestamp),
        "price": float(price),
        "symbol": symbol,
        "tags": tags,
        "metadata": metadata,
        "active": active,
    }
    if note is not None:
        payload["note"] = note

    assert validate_against_schema(payload, SCHEMA)
