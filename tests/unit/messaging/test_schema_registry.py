from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from core.messaging.schema_registry import (
    EventSchemaRegistry,
    SchemaCompatibilityError,
    SchemaFormat,
)


def test_registry_loads_known_events() -> None:
    registry = EventSchemaRegistry.from_directory("schemas/events")
    assert "ticks" in set(registry.available_events())
    latest = registry.latest("ticks", SchemaFormat.AVRO)
    schema = latest.load()
    assert schema["name"] == "TickEvent"


def test_backward_compatibility_violation_detected(tmp_path: Path) -> None:
    source_dir = Path("schemas/events")
    target = tmp_path / "events"
    shutil.copytree(source_dir, target)

    tick_v1_path = target / "avro" / "v1" / "tick.avsc"
    tick_v2_path = target / "avro" / "v2" / "tick.avsc"
    tick_v2_path.parent.mkdir(parents=True, exist_ok=True)
    with tick_v1_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    schema["fields"] = [field for field in schema["fields"] if field["name"] != "symbol"]
    with tick_v2_path.open("w", encoding="utf-8") as handle:
        json.dump(schema, handle)

    registry_path = target / "registry.json"
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_payload["events"]["ticks"]["versions"].append(
        {
            "version": 2,
            "avro": "avro/v2/tick.avsc",
            "protobuf": "../../libs/proto/events.proto",
        }
    )
    registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")

    registry = EventSchemaRegistry.from_directory(target)
    with pytest.raises(SchemaCompatibilityError):
        registry.validate_backward_and_forward("ticks")
