"""Runtime support for orchestrating repository scripts.

The :mod:`scripts` directory historically contained a loose collection of
standalone utilities.  As the catalogue grew, running the right script with the
appropriate arguments became increasingly error prone.  This module introduces a
structured registry that documents each script, validates inputs, exports
manifest metadata and records Prometheus metrics so executions remain auditable
and predictable.

Only a single lightweight dependency—PyYAML—is required at import time.  Optional
capabilities such as Pandera- or JSON-schema validation and Prometheus metrics
are enabled automatically when their respective libraries are installed.  The
design keeps developer workflows frictionless while providing the guard rails
expected in production-style automation.
"""

from __future__ import annotations

# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap
import time
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class ScriptRegistryError(RuntimeError):
    """Base class for registry related errors."""


class MissingDependencyError(ScriptRegistryError):
    """Raised when optional dependencies are missing."""


class ConfirmationRequiredError(ScriptRegistryError):
    """Raised when a script requires explicit confirmation before running."""


class SchemaValidationError(ScriptRegistryError):
    """Raised when structured data fails schema validation."""


@dataclass(slots=True)
class ParameterSpec:
    """Declarative description of an individual command line parameter."""

    name: str
    description: str
    required: bool = False
    type: str | None = None
    default: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterSpec":
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            required=bool(payload.get("required", False)),
            type=payload.get("type"),
            default=payload.get("default"),
        )


@dataclass(slots=True)
class SchemaRef:
    """Reference to a structured schema used for validation."""

    kind: str
    target: str
    description: str | None = None
    applies_to: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchemaRef":
        return cls(
            kind=str(payload["kind"]).lower(),
            target=str(payload["target"]),
            description=payload.get("description"),
            applies_to=payload.get("applies_to"),
        )

    def _load_pandera_schema(self) -> Any:
        try:  # pragma: no cover - exercised in integration environments
            import pandera as pa
        except Exception as exc:  # pragma: no cover - import guard
            raise MissingDependencyError(
                "pandera is required for DataFrame schema validation"
            ) from exc

        module_path, _, attr = self.target.partition(":")
        if not module_path:
            raise SchemaValidationError(
                f"Invalid pandera schema reference '{self.target}': missing module"
            )
        module = importlib.import_module(module_path)
        schema_factory = getattr(module, attr) if attr else None
        if schema_factory is None:
            raise SchemaValidationError(
                f"Could not resolve pandera schema callable '{self.target}'"
            )
        schema = schema_factory()
        if not isinstance(schema, pa.DataFrameSchema):
            raise SchemaValidationError(
                f"Callable '{self.target}' did not return a pandera.DataFrameSchema"
            )
        return schema

    def _load_json_schema(self) -> Mapping[str, Any]:
        schema_path = Path(self.target)
        if not schema_path.is_absolute():
            schema_path = REPO_ROOT / schema_path
        if not schema_path.exists():
            raise SchemaValidationError(
                f"JSON schema file '{schema_path}' does not exist"
            )
        return json.loads(schema_path.read_text(encoding="utf-8"))

    def _validate_json(self, payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
        required = schema.get("required", [])
        properties: Mapping[str, Mapping[str, Any]] = schema.get("properties", {})

        missing = [field for field in required if field not in payload]
        if missing:
            raise SchemaValidationError(
                f"Missing required properties: {', '.join(sorted(missing))}"
            )

        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": Mapping,
            "array": (list, tuple),
        }

        for key, value in payload.items():
            if key not in properties:
                continue
            expected = properties[key].get("type")
            if not expected:
                continue
            expected_type = type_mapping.get(expected)
            if expected_type is None:
                continue
            if not isinstance(value, expected_type):
                raise SchemaValidationError(
                    f"Property '{key}' expected type '{expected}', got '{type(value).__name__}'"
                )

    def validate(self, payload: Any) -> None:
        """Validate *payload* using the configured schema reference."""

        if self.kind == "pandera":
            schema = self._load_pandera_schema()
            schema.validate(payload)
            return

        if self.kind == "jsonschema":
            if not isinstance(payload, Mapping):
                raise SchemaValidationError(
                    "JSON schema validation requires a mapping payload"
                )
            schema = self._load_json_schema()
            self._validate_json(payload, schema)
            return

        raise SchemaValidationError(f"Unknown schema kind '{self.kind}'")


@dataclass(slots=True)
class ArtifactSpec:
    """Describe an artefact emitted by a script run."""

    name: str
    path_template: str
    description: str | None = None
    schema: SchemaRef | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactSpec":
        schema = payload.get("schema")
        return cls(
            name=str(payload["name"]),
            path_template=str(payload.get("path_template", "")),
            description=payload.get("description"),
            schema=SchemaRef.from_dict(schema) if schema else None,
        )


@dataclass(slots=True)
class ScriptSpec:
    """Complete metadata specification for a repository script."""

    name: str
    description: str
    command: list[str]
    category: str | None = None
    parameters: list[ParameterSpec] = field(default_factory=list)
    python_dependencies: list[str] = field(default_factory=list)
    container_image: str | None = None
    tags: list[str] = field(default_factory=list)
    estimated_runtime_seconds: float | None = None
    estimated_cost_usd: float | None = None
    requires_confirmation: bool = False
    input_schemas: list[SchemaRef] = field(default_factory=list)
    output_artifacts: list[ArtifactSpec] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScriptSpec":
        raw_command = payload.get("command")
        if isinstance(raw_command, str):
            command = shlex.split(raw_command)
        elif isinstance(raw_command, Sequence):
            command = [str(part) for part in raw_command]
        else:
            raise ScriptRegistryError(
                f"Script '{payload.get('name')}' has invalid command declaration"
            )

        parameters = [
            ParameterSpec.from_dict(item) for item in payload.get("parameters", [])
        ]
        input_schemas = [
            SchemaRef.from_dict(item) for item in payload.get("input_schemas", [])
        ]
        output_artifacts = [
            ArtifactSpec.from_dict(item)
            for item in payload.get("output_artifacts", [])
        ]

        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            command=command,
            category=payload.get("category"),
            parameters=parameters,
            python_dependencies=[str(dep) for dep in payload.get("python_dependencies", [])],
            container_image=payload.get("container_image"),
            tags=[str(tag) for tag in payload.get("tags", [])],
            estimated_runtime_seconds=(
                float(payload["estimated_runtime_seconds"])
                if payload.get("estimated_runtime_seconds") is not None
                else None
            ),
            estimated_cost_usd=(
                float(payload["estimated_cost_usd"])
                if payload.get("estimated_cost_usd") is not None
                else None
            ),
            requires_confirmation=bool(payload.get("requires_confirmation", False)),
            input_schemas=input_schemas,
            output_artifacts=output_artifacts,
            environment={
                str(key): str(value)
                for key, value in (payload.get("environment") or {}).items()
            },
        )

    def ensure_dependencies(self, *, auto_install: bool = False) -> None:
        """Check optional Python dependencies are available.

        When ``auto_install`` is ``True`` the interpreter invokes ``pip`` to
        install missing packages automatically.  This keeps first-run experiences
        smooth for users who have not yet prepared a virtual environment.
        """

        missing: list[str] = []
        for dependency in self.python_dependencies:
            if importlib.util.find_spec(dependency) is None:
                missing.append(dependency)

        if not missing:
            return

        if not auto_install:
            raise MissingDependencyError(
                "Missing Python dependencies: " + ", ".join(missing)
            )

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            *missing,
        ]
        subprocess.run(cmd, check=True)

    def build_command(self, extra_args: Sequence[str] | None = None) -> list[str]:
        command = list(self.command)
        if extra_args:
            command.extend(str(arg) for arg in extra_args)
        return command

    def summarise_parameters(self) -> str:
        if not self.parameters:
            return "—"
        items = []
        for parameter in self.parameters:
            summary = f"{parameter.name}"
            if parameter.required:
                summary += " (required)"
            if parameter.description:
                summary += f": {parameter.description}"
            items.append(summary)
        return "<br>".join(items)


@dataclass(slots=True)
class ScriptRunResult:
    """Details describing a script execution."""

    spec: ScriptSpec
    command: list[str]
    duration_seconds: float
    success: bool
    metrics_path: Path | None
    manifest_path: Path | None


class ScriptMetricsCollector:
    """Minimal Prometheus-compatible metrics collector."""

    def __init__(self, script_name: str):
        self.script_name = script_name
        self.registry = None
        self._runs_total = 0
        self._runs_success = 0
        self._total_duration = 0.0
        self._last_duration = 0.0
        self._last_cost = None
        self._prometheus_unavailable_reason: str | None = None

        try:  # pragma: no cover - optional dependency branch
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        except Exception as exc:  # pragma: no cover - import guard
            self._prometheus_unavailable_reason = str(exc)
            self.registry = None
            self._counter_runs = None
            self._counter_success = None
            self._hist_duration = None
            self._gauge_last_cost = None
            return

        self.registry = CollectorRegistry()
        self._counter_runs = Counter(
            "tradepulse_script_runs_total",
            "Total number of script executions.",
            ["script"],
            registry=self.registry,
        )
        self._counter_success = Counter(
            "tradepulse_script_runs_success_total",
            "Total number of successful script executions.",
            ["script"],
            registry=self.registry,
        )
        self._hist_duration = Histogram(
            "tradepulse_script_run_duration_seconds",
            "Duration of script executions in seconds.",
            ["script"],
            registry=self.registry,
        )
        self._gauge_last_cost = Gauge(
            "tradepulse_script_last_run_cost_usd",
            "Cost of the most recent script execution in USD.",
            ["script"],
            registry=self.registry,
        )

    def record(self, *, success: bool, duration_seconds: float, cost_usd: float | None) -> None:
        self._runs_total += 1
        self._total_duration += duration_seconds
        self._last_duration = duration_seconds
        if success:
            self._runs_success += 1
        if cost_usd is not None:
            self._last_cost = float(cost_usd)

        if self.registry is None:
            return

        assert self._counter_runs is not None
        assert self._counter_success is not None
        assert self._hist_duration is not None
        assert self._gauge_last_cost is not None

        labels = (self.script_name,)
        self._counter_runs.labels(*labels).inc()
        if success:
            self._counter_success.labels(*labels).inc()
        self._hist_duration.labels(*labels).observe(duration_seconds)
        if cost_usd is not None:
            self._gauge_last_cost.labels(*labels).set(cost_usd)

    def render(self) -> str:
        if self.registry is not None:  # pragma: no cover - optional dependency branch
            from prometheus_client import generate_latest

            return generate_latest(self.registry).decode("utf-8")

        lines = [
            "# Prometheus client unavailable; falling back to text metrics",
            f"script_runs_total{{script=\"{self.script_name}\"}} {self._runs_total}",
            f"script_runs_success_total{{script=\"{self.script_name}\"}} {self._runs_success}",
            f"script_last_duration_seconds{{script=\"{self.script_name}\"}} {self._last_duration}",
        ]
        if self._last_cost is not None:
            lines.append(
                "script_last_run_cost_usd{script=\"%s\"} %s"
                % (self.script_name, self._last_cost)
            )
        if self._prometheus_unavailable_reason:
            lines.append(f"# reason: {self._prometheus_unavailable_reason}")
        return "\n".join(lines) + "\n"

    def as_dict(self) -> dict[str, Any]:
        return {
            "script": self.script_name,
            "runs_total": self._runs_total,
            "runs_success": self._runs_success,
            "total_duration_seconds": self._total_duration,
            "last_duration_seconds": self._last_duration,
            "last_cost_usd": self._last_cost,
        }


class ScriptRegistry:
    """In-memory representation of :mod:`scripts` metadata."""

    def __init__(self, specs: Iterable[ScriptSpec], *, version: int | None = None):
        self._specs = {spec.name: spec for spec in specs}
        self.version = version

    @classmethod
    def from_yaml(cls, payload: Mapping[str, Any]) -> "ScriptRegistry":
        scripts = [ScriptSpec.from_dict(item) for item in payload.get("scripts", [])]
        return cls(scripts, version=payload.get("version"))

    @classmethod
    def from_path(cls, path: Path | str | None = None) -> "ScriptRegistry":
        if path is None:
            path = REPO_ROOT / "scripts" / "registry.yaml"
        path = Path(path)
        if not path.exists():
            raise ScriptRegistryError(f"Registry file '{path}' does not exist")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_yaml(payload)

    def get(self, name: str) -> ScriptSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ScriptRegistryError(f"Unknown script '{name}'") from exc

    def __iter__(self):
        return iter(self._specs.values())

    def generate_markdown_index(self) -> str:
        header = [
            "# TradePulse Script Index",
            "",
            "The table below is auto-generated from ``scripts/registry.yaml`` to",
            "document the supported automation entry points.  Update the YAML file",
            "to keep the index in sync.",
            "",
            "| Script | Description | Command | Parameters | Container | Dependencies |",
            "| --- | --- | --- | --- | --- | --- |",
        ]

        rows = []
        for spec in sorted(self._specs.values(), key=lambda item: item.name):
            command = f"`{' '.join(spec.command)}`"
            params = spec.summarise_parameters()
            container = spec.container_image or "—"
            dependencies = ", ".join(spec.python_dependencies) or "—"
            rows.append(
                "| {name} | {description} | {command} | {params} | {container} | {deps} |".format(
                    name=spec.name,
                    description=spec.description.replace("|", "\\|"),
                    command=command,
                    params=params.replace("|", "\\|"),
                    container=container,
                    deps=dependencies,
                )
            )

        body = header + rows + ["", "Generated by ``scripts.manage_scripts``."]
        return "\n".join(body) + "\n"


class ScriptRunner:
    """Execute registered scripts with validation and observability hooks."""

    def __init__(self, registry: ScriptRegistry):
        self.registry = registry

    def _check_confirmation(self, spec: ScriptSpec, *, force: bool) -> None:
        if not spec.requires_confirmation:
            return
        if force or os.environ.get("TRADEPULSE_ALLOW_LIVE") == "1":
            return
        raise ConfirmationRequiredError(
            textwrap.dedent(
                f"""
                Script '{spec.name}' performs actions flagged as potentially live.
                Re-run with ``--force`` or set TRADEPULSE_ALLOW_LIVE=1 after
                verifying the target environment.
                """
            ).strip()
        )

    def _build_environment(
        self, spec: ScriptSpec, overrides: Mapping[str, str] | None
    ) -> MutableMapping[str, str]:
        env = dict(os.environ)
        env.update(spec.environment)
        if overrides:
            env.update({str(key): str(value) for key, value in overrides.items()})
        return env

    def _write_text(self, path: Path | None, content: str) -> Path | None:
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _write_json(self, path: Path | None, payload: Mapping[str, Any]) -> Path | None:
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def run(
        self,
        script_name: str,
        *,
        extra_args: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        auto_install_dependencies: bool = False,
        dry_run: bool = False,
        force: bool = False,
        metrics_path: Path | None = None,
        manifest_path: Path | None = None,
    ) -> ScriptRunResult:
        spec = self.registry.get(script_name)
        self._check_confirmation(spec, force=force)
        spec.ensure_dependencies(auto_install=auto_install_dependencies)

        metrics = ScriptMetricsCollector(spec.name)
        command = spec.build_command(extra_args)
        start_time = time.perf_counter()
        success = True

        if not dry_run:
            completed = subprocess.run(
                command,
                env=self._build_environment(spec, env),
                check=False,
            )
            success = completed.returncode == 0

        duration = time.perf_counter() - start_time if not dry_run else 0.0
        metrics.record(
            success=success,
            duration_seconds=duration,
            cost_usd=spec.estimated_cost_usd,
        )

        default_metrics = REPO_ROOT / "reports" / "metrics" / f"{spec.name}.prom"
        metrics_path = metrics_path or default_metrics
        metrics_written = self._write_text(metrics_path, metrics.render())

        manifest_payload = {
            "script": spec.name,
            "description": spec.description,
            "command": command,
            "success": success,
            "duration_seconds": duration,
            "estimated_cost_usd": spec.estimated_cost_usd,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": spec.environment,
            "metrics": metrics.as_dict(),
            "output_artifacts": [
                {
                    "name": artifact.name,
                    "path_template": artifact.path_template,
                    "description": artifact.description,
                }
                for artifact in spec.output_artifacts
            ],
        }

        default_manifest = (
            REPO_ROOT
            / "reports"
            / "script_manifests"
            / f"{spec.name}-{int(time.time())}.json"
        )
        manifest_path = manifest_path or default_manifest
        manifest_written = self._write_json(manifest_path, manifest_payload)

        return ScriptRunResult(
            spec=spec,
            command=command,
            duration_seconds=duration,
            success=success,
            metrics_path=metrics_written,
            manifest_path=manifest_written,
        )
