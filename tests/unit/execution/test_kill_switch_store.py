from __future__ import annotations

from execution.risk import KillSwitch, SQLiteKillSwitchStateStore


def test_kill_switch_persists_state_across_instances(tmp_path) -> None:
    store_path = tmp_path / "state" / "kill_switch.sqlite"
    store = SQLiteKillSwitchStateStore(store_path)

    first = KillSwitch(store=store)
    assert first.is_triggered() is False

    first.trigger("incident response")
    assert store.load() == (True, "incident response")

    restored = KillSwitch(store=store)
    assert restored.is_triggered() is True
    assert restored.reason == "incident response"


def test_kill_switch_reset_persists_clear_state(tmp_path) -> None:
    store = SQLiteKillSwitchStateStore(tmp_path / "kill_switch.sqlite")
    kill_switch = KillSwitch(store=store)
    kill_switch.trigger("maintenance")
    kill_switch.reset()

    persisted = store.load()
    assert persisted == (False, "")

    reloaded = KillSwitch(store=store)
    assert reloaded.is_triggered() is False
    assert reloaded.reason == ""
