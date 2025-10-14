from __future__ import annotations

from execution.risk import RiskLimits, RiskManager
from src.risk.risk_manager import KillSwitchState, RiskManagerFacade


def test_facade_engages_kill_switch_with_reason() -> None:
    manager = RiskManager(RiskLimits())
    facade = RiskManagerFacade(manager)

    state = facade.engage_kill_switch("manual intervention")

    assert isinstance(state, KillSwitchState)
    assert state.engaged is True
    assert state.reason == "manual intervention"
    assert state.already_engaged is False
    assert manager.kill_switch.is_triggered() is True


def test_facade_engage_reaffirms_without_new_reason() -> None:
    manager = RiskManager(RiskLimits())
    facade = RiskManagerFacade(manager)

    first = facade.engage_kill_switch("initial reason")
    assert first.engaged is True

    reaffirmed = facade.engage_kill_switch("")
    assert reaffirmed.already_engaged is True
    assert reaffirmed.reason == "initial reason"
    assert manager.kill_switch.reason == "initial reason"


def test_facade_reset_returns_previous_reason() -> None:
    manager = RiskManager(RiskLimits())
    facade = RiskManagerFacade(manager)

    facade.engage_kill_switch("investigation")
    reset_state = facade.reset_kill_switch()

    assert reset_state.engaged is False
    assert reset_state.reason == "investigation"
    assert reset_state.already_engaged is True
    assert manager.kill_switch.is_triggered() is False


def test_facade_state_reflects_kill_switch() -> None:
    manager = RiskManager(RiskLimits())
    facade = RiskManagerFacade(manager)

    initial = facade.kill_switch_state()
    assert initial.engaged is False
    assert initial.reason == ""

    manager.kill_switch.trigger("ops override")
    current = facade.kill_switch_state()
    assert current.engaged is True
    assert current.reason == "ops override"
