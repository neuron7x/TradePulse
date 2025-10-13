"""Validate SBOM licenses against an allow-list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DISALLOWED = {
    "GPL-3.0",
    "AGPL-3.0",
    "SSPL-1.0",
    "GPL-2.0",
}


def load_components(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    components = data.get("components", [])
    if not isinstance(components, list):
        raise ValueError("SBOM components payload must be a list")
    return [component for component in components if isinstance(component, dict)]


def extract_license(component: dict[str, object]) -> set[str]:
    licenses = component.get("licenses", [])
    results: set[str] = set()
    if isinstance(licenses, list):
        for entry in licenses:
            if not isinstance(entry, dict):
                continue
            license_id = entry.get("license", {}).get("id")
            if isinstance(license_id, str):
                results.add(license_id)
            name = entry.get("license", {}).get("name")
            if isinstance(name, str):
                results.add(name)
            expression = entry.get("expression")
            if isinstance(expression, str):
                results.add(expression)
    return results


def validate_sbom(path: Path) -> tuple[int, list[str], list[tuple[str, list[str]]]]:
    components = load_components(path)
    missing_license = []
    violations = []
    for component in components:
        name = component.get("name", "<unknown>")
        licenses = extract_license(component)
        if not licenses:
            missing_license.append(str(name))
            continue
        if DISALLOWED.intersection(licenses):
            violations.append((str(name), sorted(DISALLOWED.intersection(licenses))))

    exit_code = 0
    if violations:
        exit_code = 1
    return exit_code, missing_license, violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SBOM license metadata")
    parser.add_argument("sbom", type=Path, help="Path to CycloneDX JSON SBOM")
    args = parser.parse_args(argv)

    exit_code, missing_license, violations = validate_sbom(args.sbom)

    if missing_license:
        print("::warning::Components missing license metadata:")
        for name in sorted(missing_license):
            print(f"  - {name}")

    if violations:
        print("::error::Detected disallowed licenses:")
        for name, bad in violations:
            print(f"  - {name}: {', '.join(bad)}")
    else:
        print("Validated SBOM components with approved licenses.")
    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
