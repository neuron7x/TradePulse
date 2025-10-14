"""Property-based tests for remote control schemas."""

from __future__ import annotations

import string

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.admin.remote_control import KillSwitchRequest, RemoteCommand


reason_strategy = st.text(
    alphabet=string.ascii_letters + string.whitespace,
    min_size=3,
    max_size=50,
).filter(lambda value: len(value.strip()) >= 3)


@given(reason_strategy)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_kill_switch_request_trims_reason(reason: str) -> None:
    request = KillSwitchRequest(reason=reason)
    assert request.reason == reason.strip()
    assert request.command is RemoteCommand.ENGAGE_KILL_SWITCH


@pytest.mark.parametrize(
    "reason",
    ["halt", "   halt", "halt   "],
)
def test_kill_switch_request_length_validation(reason: str) -> None:
    request = KillSwitchRequest(reason=reason)
    assert len(request.reason) >= 3


def test_kill_switch_request_rejects_short_reason() -> None:
    with pytest.raises(ValueError):
        KillSwitchRequest(reason=" a ")
