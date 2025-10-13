# SPDX-License-Identifier: MIT
"""Property-based tests covering resampling utilities under irregular inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)

from core.data.resampling import align_timeframes, resample_ticks_to_l1


@st.composite
def irregular_tick_frames(draw, *, max_period: int = 3600) -> pd.DataFrame:
    """Generate tick data with irregular spacing, gaps and unsorted timestamps."""

    size = draw(st.integers(min_value=5, max_value=60))
    # Positive step sizes to ensure a strictly increasing canonical index while still
    # allowing large gaps between observations.
    increments = draw(
        st.lists(
            st.integers(min_value=1, max_value=max_period // 2),
            min_size=size,
            max_size=size,
        )
    )
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    offsets = np.cumsum(np.asarray(increments, dtype=np.int64))
    index = base + pd.to_timedelta(offsets, unit="s")

    # Introduce gaps by randomly dropping up to half of the timestamps. Hypothesis
    # handles deduplication of the sampled indices for us.
    keep_positions = draw(
        st.sets(st.integers(min_value=0, max_value=len(index) - 1), min_size=2, max_size=len(index))
    )
    if len(keep_positions) < len(index):
        index = index[sorted(keep_positions)]

    prices = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False),
            min_size=len(index),
            max_size=len(index),
        )
    )
    sizes = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=5_000.0, allow_nan=False, allow_infinity=False),
            min_size=len(index),
            max_size=len(index),
        )
    )

    frame = pd.DataFrame({"price": prices, "size": sizes}, index=index)

    # Shuffle to ensure the implementation robustly sorts the index.
    perm = np.asarray(draw(st.permutations(tuple(range(len(frame))))), dtype=int)
    shuffled = frame.iloc[perm]
    return shuffled


@settings(max_examples=100, deadline=None)
@given(frame=irregular_tick_frames(), freq=st.sampled_from(["15S", "1T", "5T", "15T"]))
def test_resample_ticks_handles_irregular_series(frame: pd.DataFrame, freq: str) -> None:
    """Resampling should tolerate irregular spacing and preserve ordering."""

    resampled = resample_ticks_to_l1(frame, freq=freq)

    assert isinstance(resampled.index, pd.DatetimeIndex)
    assert resampled.index.is_monotonic_increasing
    assert not resampled.index.has_duplicates

    # Values should be numeric and finite for the resampled periods that contain data.
    assert resampled.columns.tolist() == ["mid_price", "last_size"]
    mid_values = resampled["mid_price"].dropna().to_numpy(dtype=float, copy=False)
    size_values = resampled["last_size"].dropna().to_numpy(dtype=float, copy=False)
    assert np.isfinite(mid_values).all()
    assert np.isfinite(size_values).all()


@settings(max_examples=75, deadline=None)
@given(frame=irregular_tick_frames(), freq=st.sampled_from(["30S", "1T", "2T"]))
def test_resample_ticks_is_idempotent(frame: pd.DataFrame, freq: str) -> None:
    """Applying the tick-to-L1 resampler twice should not change the output."""

    first = resample_ticks_to_l1(frame, freq=freq)
    renamed = first.rename(columns={"mid_price": "price", "last_size": "size"})
    second = resample_ticks_to_l1(renamed, freq=freq)
    pd.testing.assert_frame_equal(second, first)


@st.composite
def timeframe_mapping(draw) -> dict[str, pd.DataFrame]:
    size = draw(st.integers(min_value=5, max_value=40))
    increments = draw(
        st.lists(st.integers(min_value=1, max_value=900), min_size=size, max_size=size)
    )
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    index = base + pd.to_timedelta(np.cumsum(np.asarray(increments, dtype=np.int64)), unit="s")

    values = draw(
        st.lists(
            st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
            min_size=len(index),
            max_size=len(index),
        )
    )
    frames: dict[str, pd.DataFrame] = {
        "reference": pd.DataFrame({"value": values}, index=index)
    }
    base_values = np.asarray(values, dtype=float)

    extra_frames = draw(st.integers(min_value=1, max_value=3))
    for i in range(extra_frames):
        keep = sorted(
            draw(
                st.sets(
                    st.integers(min_value=0, max_value=len(index) - 1),
                    min_size=1,
                    max_size=len(index),
                )
            )
        )
        sub_index = index[keep]
        jitter = draw(
            st.lists(
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=len(sub_index),
                max_size=len(sub_index),
            )
        )
        frames[f"alt_{i}"] = pd.DataFrame(
            {"value": base_values[keep] + np.asarray(jitter, dtype=float)},
            index=sub_index,
        )

    return frames


@settings(max_examples=75, deadline=None)
@given(frames=timeframe_mapping())
def test_align_timeframes_is_idempotent(frames: dict[str, pd.DataFrame]) -> None:
    """Re-aligning already aligned frames should not alter data or ordering."""

    aligned = align_timeframes(frames, reference="reference")
    realigned = align_timeframes(aligned, reference="reference")

    for key, frame in aligned.items():
        pd.testing.assert_frame_equal(realigned[key], frame)
        assert frame.index.equals(aligned["reference"].index)
        assert frame.index.is_monotonic_increasing
        assert not frame.index.has_duplicates

