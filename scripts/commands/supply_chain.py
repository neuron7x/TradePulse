"""Supply chain security automation commands."""

from __future__ import annotations

import logging
from argparse import Namespace, _SubParsersAction
from pathlib import Path
from typing import Iterable, Sequence

from scripts.commands.base import (
    CommandError,
    ensure_tools_exist,
    register,
    run_subprocess,
)
from scripts.supply_chain import (
    DependencyError,
    Severity,
    build_cyclonedx_sbom,
    load_denylist,
    load_dependencies,
    verify_dependencies,
    write_sbom,
    write_verification_report,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_REQUIREMENTS = (Path("requirements.lock"),)
DEFAULT_DEV_REQUIREMENTS = (Path("requirements-dev.lock"),)
DEFAULT_DENYLIST = Path("configs/security/denylist.yaml")
DEFAULT_SBOM_PATH = Path("sbom/cyclonedx-sbom.json")
DEFAULT_REPORT_PATH = Path("reports/supply_chain/dependency-verification.json")


def build_parser(subparsers: _SubParsersAction[object]) -> None:
    parser = subparsers.add_parser(
        "supply-chain",
        help="Generate SBOMs, sign containers, and verify dependency hygiene.",
    )
    parser.set_defaults(command="supply-chain", handler=handle)

    subparsers = parser.add_subparsers(dest="action", required=True)

    build_generate_parser(subparsers)
    build_verify_parser(subparsers)
    build_sign_parser(subparsers)
    build_verify_image_parser(subparsers)


def _parse_requirement_args(namespace: Namespace) -> Sequence[Path]:
    requirement_args: Iterable[Path] | None = getattr(namespace, "requirements", None)
    include_dev: bool = bool(getattr(namespace, "include_dev", False))

    paths: list[Path] = []
    if requirement_args:
        paths.extend(requirement_args)
    else:
        paths.extend(DEFAULT_REQUIREMENTS)

    if include_dev:
        paths.extend(DEFAULT_DEV_REQUIREMENTS)

    # Deduplicate while preserving order
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path)
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _resolve_version(namespace: Namespace) -> str:
    project_version = getattr(namespace, "project_version", None)
    if project_version:
        return str(project_version)
    version_file = Path("VERSION")
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip()
    return "0.0.0-dev"


def build_generate_parser(subparsers: _SubParsersAction[object]) -> None:
    sbom = subparsers.add_parser(
        "generate-sbom",
        help="Produce a CycloneDX SBOM from pinned dependency manifests.",
    )
    sbom.add_argument(
        "--requirements",
        action="append",
        type=Path,
        help="Path(s) to requirement files. Defaults to requirements.lock.",
    )
    sbom.add_argument(
        "--include-dev",
        action="store_true",
        help="Include development requirements (requirements-dev.lock).",
    )
    sbom.add_argument(
        "--project-name",
        default="TradePulse",
        help="Name of the root component represented by the SBOM.",
    )
    sbom.add_argument(
        "--project-version",
        default=None,
        help="Version of the root component. Defaults to contents of VERSION file.",
    )
    sbom.add_argument(
        "--description",
        default=None,
        help="Optional description included in the SBOM metadata.",
    )
    sbom.add_argument(
        "--output",
        default=DEFAULT_SBOM_PATH,
        type=Path,
        help=f"Destination file for the SBOM (default: {DEFAULT_SBOM_PATH}).",
    )
    sbom.set_defaults(action="generate-sbom")


def build_verify_parser(subparsers: _SubParsersAction[object]) -> None:
    verify = subparsers.add_parser(
        "verify",
        help="Validate dependency manifests against policy and deny lists.",
    )
    verify.add_argument(
        "--requirements",
        action="append",
        type=Path,
        help="Path(s) to requirement files. Defaults to requirements.lock.",
    )
    verify.add_argument(
        "--include-dev",
        action="store_true",
        help="Include development requirements in verification.",
    )
    verify.add_argument(
        "--denylist",
        default=DEFAULT_DENYLIST,
        type=Path,
        help="Path to compromised package denylist (YAML).",
    )
    verify.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Allow dependencies without exact version pins.",
    )
    verify.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=(
            "Path to write a JSON transparency report. Use '-' to disable file output."
        ),
    )
    verify.set_defaults(action="verify")


def build_sign_parser(subparsers: _SubParsersAction[object]) -> None:
    sign = subparsers.add_parser(
        "sign-image",
        help="Sign a container image with cosign to guarantee provenance.",
    )
    sign.add_argument("image", help="Fully-qualified container image reference (e.g., repo/image:tag).")
    sign.add_argument(
        "--key",
        default=None,
        type=Path,
        help="Path to the private key used for signing. Omit for keyless workflows.",
    )
    sign.add_argument(
        "--cosign-binary",
        default="cosign",
        help="Executable name or path for cosign.",
    )
    sign.add_argument(
        "--annotation",
        action="append",
        default=(),
        help="Additional annotations in the form key=value to attach to the signature.",
    )
    sign.add_argument(
        "--identity-token",
        default=None,
        help="OIDC identity token to use for keyless signing.",
    )
    sign.set_defaults(action="sign-image")


def build_verify_image_parser(subparsers: _SubParsersAction[object]) -> None:
    verify_image = subparsers.add_parser(
        "verify-image",
        help="Verify the signature on a container image using cosign.",
    )
    verify_image.add_argument("image", help="Container image reference to verify.")
    verify_image.add_argument(
        "--cosign-binary",
        default="cosign",
        help="Executable name or path for cosign.",
    )
    verify_image.add_argument(
        "--key",
        default=None,
        type=Path,
        help="Path to a public key if verifying key-based signatures.",
    )
    verify_image.add_argument(
        "--certificate-identity",
        default=None,
        help="Expected identity (email/URI) when verifying keyless signatures.",
    )
    verify_image.add_argument(
        "--certificate-oidc-issuer",
        default=None,
        help="Expected OIDC issuer for keyless signatures.",
    )
    verify_image.set_defaults(action="verify-image")


def _log_issues(report) -> None:
    for issue in report.issues:
        if issue.severity == Severity.WARNING:
            LOGGER.warning("%s – %s", issue.dependency.name, issue.message)
        elif issue.severity == Severity.ERROR:
            LOGGER.error("%s – %s", issue.dependency.name, issue.message)
        else:
            LOGGER.critical("%s – %s", issue.dependency.name, issue.message)
        if issue.references:
            LOGGER.debug("References: %s", ", ".join(issue.references))
        if issue.cves:
            LOGGER.debug("Associated CVEs: %s", ", ".join(issue.cves))


@register("supply-chain")
def handle(args: object) -> int:
    namespace = getattr(args, "__dict__", args)
    action = namespace.get("action")

    if action == "generate-sbom":
        return _handle_generate_sbom(Namespace(**namespace))
    if action == "verify":
        return _handle_verify(Namespace(**namespace))
    if action == "sign-image":
        return _handle_sign(Namespace(**namespace))
    if action == "verify-image":
        return _handle_verify_image(Namespace(**namespace))
    raise CommandError(f"Unknown supply-chain action: {action}")


def _handle_generate_sbom(namespace: Namespace) -> int:
    requirements = _parse_requirement_args(namespace)
    try:
        dependencies = load_dependencies(requirements)
    except DependencyError as exc:
        raise CommandError(str(exc)) from exc

    project_name = getattr(namespace, "project_name", "TradePulse")
    project_version = _resolve_version(namespace)
    description = getattr(namespace, "description", None)
    output_path: Path = Path(getattr(namespace, "output", DEFAULT_SBOM_PATH))

    sbom = build_cyclonedx_sbom(
        dependencies,
        component_name=str(project_name),
        component_version=project_version,
        source_description=str(description) if description else None,
    )
    write_sbom(sbom, output_path)

    LOGGER.info(
        "Generated SBOM containing %d component(s) at %s.",
        len(dependencies),
        output_path,
    )
    return 0


def _handle_verify(namespace: Namespace) -> int:
    requirements = _parse_requirement_args(namespace)
    try:
        dependencies = load_dependencies(requirements)
    except DependencyError as exc:
        raise CommandError(str(exc)) from exc

    denylist_path: Path = Path(getattr(namespace, "denylist", DEFAULT_DENYLIST))
    denylist = load_denylist(denylist_path)

    allow_unpinned: bool = bool(getattr(namespace, "allow_unpinned", False))
    report = verify_dependencies(
        dependencies,
        denylist,
        require_pins=not allow_unpinned,
    )

    _log_issues(report)

    report_path = getattr(namespace, "report", None)
    if report_path and str(report_path) != "-":
        destination = Path(report_path)
        write_verification_report(report, destination)
        LOGGER.info("Wrote dependency transparency report to %s", destination)

    if report.has_failures():
        raise CommandError(
            "Dependency verification failed. Review the issues above and the optional report if generated."
        )

    LOGGER.info(
        "Dependency verification succeeded for %d dependencies (issues: %d).",
        len(report.dependencies),
        len(report.issues),
    )
    return 0


def _build_cosign_command(binary: str) -> str:
    if not binary:
        raise CommandError("Cosign binary cannot be empty.")
    return binary


def _handle_sign(namespace: Namespace) -> int:
    cosign_binary = _build_cosign_command(getattr(namespace, "cosign_binary", "cosign"))
    ensure_tools_exist([cosign_binary])

    command = [cosign_binary, "sign", "--yes"]
    identity_token = getattr(namespace, "identity_token", None)
    if identity_token:
        command.extend(["--identity-token", identity_token])

    key = getattr(namespace, "key", None)
    if key:
        command.extend(["--key", str(Path(key))])

    for annotation in getattr(namespace, "annotation", ()) or ():
        command.extend(["--annotation", str(annotation)])

    image = getattr(namespace, "image")
    if not image:
        raise CommandError("Container image reference is required for signing.")
    command.append(str(image))

    LOGGER.info("Signing container image %s using %s", image, cosign_binary)
    run_subprocess(command)
    LOGGER.info("Successfully signed %s", image)
    return 0


def _handle_verify_image(namespace: Namespace) -> int:
    cosign_binary = _build_cosign_command(getattr(namespace, "cosign_binary", "cosign"))
    ensure_tools_exist([cosign_binary])

    command = [cosign_binary, "verify"]
    key = getattr(namespace, "key", None)
    if key:
        command.extend(["--key", str(Path(key))])
    identity = getattr(namespace, "certificate_identity", None)
    if identity:
        command.extend(["--certificate-identity", identity])
    issuer = getattr(namespace, "certificate_oidc_issuer", None)
    if issuer:
        command.extend(["--certificate-oidc-issuer", issuer])

    image = getattr(namespace, "image")
    if not image:
        raise CommandError("Container image reference is required for verification.")
    command.append(str(image))

    LOGGER.info("Verifying signature for container image %s", image)
    run_subprocess(command)
    LOGGER.info("Signature verification completed for %s", image)
    return 0
