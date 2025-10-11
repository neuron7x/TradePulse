"""Local schema registry with compatibility validation for TradePulse events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


class SchemaFormat(str, Enum):
    """Supported serialization formats."""

    AVRO = "avro"
    PROTOBUF = "protobuf"


class SchemaCompatibilityError(RuntimeError):
    """Raised when schema compatibility validation fails."""


@dataclass(frozen=True)
class SchemaVersionInfo:
    """Metadata describing a concrete schema version."""

    version: int
    path: Path
    format: SchemaFormat

    def load(self) -> Mapping[str, Any]:
        """Return the parsed schema document."""

        if self.format is SchemaFormat.AVRO:
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        raise ValueError(f"Unsupported load operation for {self.format}")


class EventSchemaRegistry:
    """Local schema registry backed by JSON metadata files."""

    def __init__(self, base_path: Path, registry: Dict[str, List[SchemaVersionInfo]]):
        self._base_path = base_path
        self._registry = registry

    @classmethod
    def from_directory(cls, base_path: str | Path) -> "EventSchemaRegistry":
        """Build a registry instance from the canonical registry.json file."""

        root = Path(base_path)
        registry_path = root / "registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"Schema registry descriptor not found: {registry_path}")
        with registry_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        events: Dict[str, List[SchemaVersionInfo]] = {}
        for event_type, event_data in payload.get("events", {}).items():
            versions: List[SchemaVersionInfo] = []
            for version_info in event_data.get("versions", []):
                avro_path = root / version_info[SchemaFormat.AVRO.value]
                versions.append(
                    SchemaVersionInfo(
                        version=version_info["version"],
                        path=avro_path,
                        format=SchemaFormat.AVRO,
                    )
                )
                versions.append(
                    SchemaVersionInfo(
                        version=version_info["version"],
                        path=(root / version_info[SchemaFormat.PROTOBUF.value]).resolve(),
                        format=SchemaFormat.PROTOBUF,
                    )
                )
            events[event_type] = versions
        return cls(root, events)

    def available_events(self) -> Iterable[str]:
        return self._registry.keys()

    @property
    def base_path(self) -> Path:
        """Return the root directory containing schema definitions."""

        return self._base_path

    def get_versions(self, event_type: str, fmt: SchemaFormat) -> List[SchemaVersionInfo]:
        if event_type not in self._registry:
            raise KeyError(f"Unknown event type '{event_type}'")
        return [info for info in self._registry[event_type] if info.format is fmt]

    def latest(self, event_type: str, fmt: SchemaFormat) -> SchemaVersionInfo:
        versions = self.get_versions(event_type, fmt)
        if not versions:
            raise KeyError(f"No {fmt.value} schema registered for '{event_type}'")
        return max(versions, key=lambda info: info.version)

    def validate_backward_and_forward(self, event_type: str) -> None:
        """Ensure all registered versions are backward and forward compatible."""

        avro_versions = sorted(self.get_versions(event_type, SchemaFormat.AVRO), key=lambda info: info.version)
        if not avro_versions:
            return
        schemas = [version.load() for version in avro_versions]
        self._validate_sequential_backward(schemas)
        self._validate_sequential_forward(schemas)

    def validate_all(self) -> None:
        for event in self.available_events():
            self.validate_backward_and_forward(event)

    def _validate_sequential_backward(self, schemas: List[Mapping[str, Any]]) -> None:
        for previous, current in zip(schemas, schemas[1:]):
            missing = [field["name"] for field in previous.get("fields", []) if field["name"] not in _field_name_index(current)]
            if missing:
                raise SchemaCompatibilityError(
                    f"Backward compatibility broken: fields {missing} removed or renamed"
                )
            for field in previous.get("fields", []):
                name = field["name"]
                prev_type = _normalise_avro_type(field["type"])
                curr_field = _field_name_index(current)[name]
                curr_type = _normalise_avro_type(curr_field["type"])
                if prev_type != curr_type:
                    raise SchemaCompatibilityError(
                        f"Backward compatibility broken for field '{name}': {prev_type} != {curr_type}"
                    )

    def _validate_sequential_forward(self, schemas: List[Mapping[str, Any]]) -> None:
        for previous, current in zip(schemas, schemas[1:]):
            for field in current.get("fields", []):
                name = field["name"]
                if name not in _field_name_index(previous):
                    if not _is_nullable(field) and "default" not in field:
                        raise SchemaCompatibilityError(
                            f"Forward compatibility broken: new field '{name}' missing default or nullable"
                        )


def _field_name_index(schema: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {field["name"]: field for field in schema.get("fields", [])}


def _normalise_avro_type(avro_type: Any) -> Tuple:
    """Normalise Avro type declarations to tuples for comparison."""

    if isinstance(avro_type, str):
        return (avro_type,)
    if isinstance(avro_type, list):
        return tuple(sorted(_normalise_union_member(member) for member in avro_type))
    if isinstance(avro_type, Mapping):
        type_name = avro_type.get("type")
        if type_name == "record":
            return (
                "record",
                avro_type.get("name"),
                tuple((field["name"], _normalise_avro_type(field["type"])) for field in avro_type.get("fields", [])),
            )
        if type_name == "enum":
            return ("enum", avro_type.get("name"), tuple(avro_type.get("symbols", [])))
        if type_name == "array":
            return ("array", _normalise_avro_type(avro_type.get("items")))
        if type_name == "map":
            return ("map", _normalise_avro_type(avro_type.get("values")))
        if type_name == "fixed":
            return ("fixed", avro_type.get("name"), avro_type.get("size"))
        return (type_name,)
    raise TypeError(f"Unsupported Avro type declaration: {avro_type!r}")


def _normalise_union_member(member: Any) -> Tuple:
    if isinstance(member, str):
        return (member,)
    return _normalise_avro_type(member)


def _is_nullable(field: Mapping[str, Any]) -> bool:
    avro_type = field["type"]
    if isinstance(avro_type, list):
        return any(member == "null" or (isinstance(member, Mapping) and member.get("type") == "null") for member in avro_type)
    return False
