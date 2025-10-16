"""Additional tests for remote_control module focusing on edge cases."""

from __future__ import annotations

from collections import deque

import pytest

from src.admin.remote_control import AdminRateLimiter, AdminRateLimiterSnapshot


@pytest.mark.asyncio
async def test_snapshot_ignores_injected_stale_buckets() -> None:
    """Test that snapshot() correctly ignores buckets with invalid timestamps."""
    limiter = AdminRateLimiter(max_attempts=2, interval_seconds=60.0)

    # Add some valid activity
    await limiter.check("active_user")

    # Inject a stale bucket with invalid timestamp (<= 0.0)
    limiter._records["stale"] = deque([0.0])  # type: ignore[attr-defined]

    # Get snapshot and verify it only reflects active identifiers
    snapshot: AdminRateLimiterSnapshot = await limiter.snapshot()

    # Should only count the active_user, not the stale bucket
    assert snapshot.tracked_identifiers == 1
    assert "stale" not in [id for id in limiter._records.keys() if limiter._records[id]]  # type: ignore[attr-defined]
