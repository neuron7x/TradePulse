"""Utilities that keep TradePulse's software supply chain trustworthy."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import yaml  # type: ignore[import-untyped]
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

LOGGER = logging.getLogger(__name__)


class DependencyError(RuntimeError):
    """Raised when dependency metadata cannot be processed."""


@dataclass(frozen=True)
class Dependency:
    """Represents a pinned dependency from a requirements lock file."""

    name: str
    raw_requirement: str
    source: Path
    requirement: Requirement

    @property
    def canonical_name(self) -> str:
        return canonicalize_name(self.name)

    @property
    def version(self) -> str | None:
        specifiers = list(self.requirement.specifier)
        if len(specifiers) != 1:
            return None
        spec = specifiers[0]
        if spec.operator in {"==", "==="} and not spec.version.endswith(".*"):
            return spec.version
        return None

    @property
    def parsed_version(self) -> Version | None:
        value = self.version
        if not value:
            return None
        try:
            return Version(value)
        except InvalidVersion:
            LOGGER.debug("Unable to parse version '%s' for %s", value, self.name)
            return None

    @property
    def is_pinned(self) -> bool:
        if self.requirement.url:
            return True
        specifiers = list(self.requirement.specifier)
        if len(specifiers) != 1:
            return False
        spec = specifiers[0]
        return spec.operator in {"==", "==="} and not spec.version.endswith(".*")

    def to_component(self) -> dict[str, object]:
        component: dict[str, object] = {
            "type": "library",
            "name": self.name,
            "purl": f"pkg:pypi/{self.canonical_name}",
            "properties": [
                {
                    "name": "dependency.source",
                    "value": str(self.source),
                },
            ],
        }
        if self.version:
            component["version"] = self.version
            component["purl"] = f"pkg:pypi/{self.canonical_name}@{self.version}"
        if self.requirement.extras:
            component["properties"].append(
                {
                    "name": "python.extras",
                    "value": ",".join(sorted(self.requirement.extras)),
                }
            )
        if self.requirement.marker:
            component["properties"].append(
                {
                    "name": "python.marker",
                    "value": str(self.requirement.marker),
                }
            )
        return component


@dataclass(frozen=True)
class DenylistEntry:
    """Represents a compromised package rule."""

    name: str
    specifier: SpecifierSet | None
    reason: str
    references: tuple[str, ...]
    cves: tuple[str, ...]

    @property
    def canonical_name(self) -> str:
        return canonicalize_name(self.name)

    def matches(self, dependency: Dependency) -> bool:
        if dependency.canonical_name != self.canonical_name:
            return False
        if not self.specifier:
            return True
        version = dependency.parsed_version
        if version is None:
            # When the dependency is unpinned, err on the safe side.
            return True
        return version in self.specifier


class Severity:
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class VerificationIssue:
    dependency: Dependency
    message: str
    severity: str
    references: tuple[str, ...] = ()
    cves: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dependency": {
                "name": self.dependency.name,
                "version": self.dependency.version,
                "source": str(self.dependency.source),
            },
            "message": self.message,
            "severity": self.severity,
        }
        if self.references:
            payload["references"] = list(self.references)
        if self.cves:
            payload["cves"] = list(self.cves)
        return payload


@dataclass(frozen=True)
class DependencyVerificationReport:
    generated_at: datetime
    dependencies: tuple[Dependency, ...]
    issues: tuple[VerificationIssue, ...]

    def has_failures(self) -> bool:
        return any(
            issue.severity in {Severity.ERROR, Severity.CRITICAL}
            for issue in self.issues
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.replace(microsecond=0).isoformat(),
            "dependency_count": len(self.dependencies),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _parse_requirement(line: str, source: Path) -> Dependency:
    try:
        requirement = Requirement(line)
    except InvalidRequirement as exc:  # pragma: no cover - defensive
        raise DependencyError(f"Invalid requirement '{line}' in {source}") from exc
    name = requirement.name
    if not name:
        raise DependencyError(
            f"Missing package name in requirement '{line}' from {source}"
        )
    return Dependency(
        name=name, raw_requirement=line, source=source, requirement=requirement
    )


IGNORED_PREFIXES = ("-r ", "--", "http://", "https://", "git+", "svn+", "hg+")


def load_dependencies(requirement_files: Sequence[Path]) -> list[Dependency]:
    dependencies: list[Dependency] = []
    for path in requirement_files:
        if not path.exists():
            raise DependencyError(f"Requirements file not found: {path}")
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(IGNORED_PREFIXES):
                LOGGER.debug("Skipping non-package requirement line '%s'", line)
                continue
            dependency = _parse_requirement(line, path)
            dependencies.append(dependency)
    dependencies.sort(key=lambda dep: (dep.canonical_name, dep.version or ""))
    return dependencies


def load_denylist(path: Path) -> list[DenylistEntry]:
    if not path.exists():
        LOGGER.debug(
            "No denylist found at %s; skipping compromised dependency checks.", path
        )
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_entries = data.get("compromised", []) or []
    entries: list[DenylistEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            LOGGER.debug("Skipping malformed denylist entry: %s", item)
            continue
        name = item.get("name")
        if not name:
            LOGGER.debug("Skipping denylist entry without a name: %s", item)
            continue
        specifier_text = item.get("specifier")
        specifier = SpecifierSet(specifier_text) if specifier_text else None
        reason = item.get("reason") or "Flagged as compromised by security policy."
        references = tuple(item.get("references", []) or [])
        cves = tuple(item.get("cves", []) or [])
        entries.append(
            DenylistEntry(
                name=str(name),
                specifier=specifier,
                reason=str(reason),
                references=tuple(str(ref) for ref in references),
                cves=tuple(str(cve) for cve in cves),
            )
        )
    return entries


def verify_dependencies(
    dependencies: Sequence[Dependency],
    denylist: Sequence[DenylistEntry],
    *,
    require_pins: bool = True,
) -> DependencyVerificationReport:
    issues: list[VerificationIssue] = []
    seen: dict[str, Dependency] = {}

    for dependency in dependencies:
        key = dependency.canonical_name
        if key in seen and seen[key].version != dependency.version:
            issues.append(
                VerificationIssue(
                    dependency=dependency,
                    message=(
                        "Multiple versions detected for package "
                        f"'{dependency.name}' ({seen[key].version} and {dependency.version})."
                    ),
                    severity=Severity.ERROR,
                )
            )
        else:
            seen[key] = dependency

        if require_pins and not dependency.is_pinned:
            issues.append(
                VerificationIssue(
                    dependency=dependency,
                    message="Dependency is not pinned to an exact version.",
                    severity=Severity.ERROR,
                )
            )

    denylist_map: dict[str, list[DenylistEntry]] = defaultdict(list)
    for entry in denylist:
        denylist_map[entry.canonical_name].append(entry)

    for dependency in dependencies:
        for entry in denylist_map.get(dependency.canonical_name, []):
            if entry.matches(dependency):
                issues.append(
                    VerificationIssue(
                        dependency=dependency,
                        message=entry.reason,
                        severity=Severity.CRITICAL,
                        references=entry.references,
                        cves=entry.cves,
                    )
                )

    report = DependencyVerificationReport(
        generated_at=datetime.now(tz=timezone.utc),
        dependencies=tuple(dependencies),
        issues=tuple(issues),
    )
    return report


def build_cyclonedx_sbom(
    dependencies: Sequence[Dependency],
    *,
    component_name: str,
    component_version: str,
    source_description: str | None = None,
) -> dict[str, object]:
    components = [dependency.to_component() for dependency in dependencies]
    metadata: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "component": {
            "type": "application",
            "name": component_name,
            "version": component_version,
        },
        "tools": [
            {
                "vendor": "TradePulse",
                "name": "SupplyChainToolkit",
                "version": "2025.1",
            }
        ],
    }
    if source_description:
        metadata["component"]["description"] = source_description

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid4()}",
        "version": 1,
        "metadata": metadata,
        "components": components,
    }
    return sbom


def write_json_document(document: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote JSON document to %s", destination)


def write_verification_report(
    report: DependencyVerificationReport, destination: Path
) -> None:
    write_json_document(report.to_dict(), destination)


def write_sbom(sbom: dict[str, object], destination: Path) -> None:
    write_json_document(sbom, destination)
