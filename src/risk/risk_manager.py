"""Risk manager façade exposing kill-switch controls to administrative APIs."""

from __future__ import annotations

from dataclasses import dataclass

from execution.risk import RiskManager

__all__ = ["RiskManagerFacade", "KillSwitchState"]


@dataclass(slots=True)
class KillSwitchState:
    """Snapshot of the kill-switch status."""

    engaged: bool
    reason: str
    already_engaged: bool = False


class RiskManagerFacade:
    """Thin wrapper that exposes high-level risk management operations."""

    def __init__(self, risk_manager: RiskManager) -> None:
        self._risk_manager = risk_manager

    @property
    def risk_manager(self) -> RiskManager:
        """Return the underlying risk manager instance."""

        return self._risk_manager

    def engage_kill_switch(self, reason: str) -> KillSwitchState:
        """Engage the global kill-switch with the provided reason."""

        kill_switch = self._risk_manager.kill_switch
        already_engaged = kill_switch.is_triggered()
        kill_switch.trigger(reason)
        current_reason = reason or kill_switch.reason
        return KillSwitchState(
            engaged=True,
            reason=current_reason,
            already_engaged=already_engaged,
        )

    def reset_kill_switch(self) -> KillSwitchState:
        """Reset the kill-switch state and return the new snapshot."""

        kill_switch = self._risk_manager.kill_switch
        kill_switch.reset()
        return KillSwitchState(engaged=False, reason="", already_engaged=False)

    def kill_switch_state(self) -> KillSwitchState:
        """Return the current kill-switch status."""

        kill_switch = self._risk_manager.kill_switch
        return KillSwitchState(
            engaged=kill_switch.is_triggered(),
            reason=kill_switch.reason,
            already_engaged=kill_switch.is_triggered(),
        )
