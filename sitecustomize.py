"""TradePulse site customizations.

This module is automatically imported by Python when present on the import
path. We use it to apply security hardening patches as soon as an interpreter
starts. Keeping the logic in ``sitecustomize`` ensures that both local
development workflows and CI environments benefit from the same mitigation
without requiring manual steps from contributors.
"""

from __future__ import annotations

import os
import shutil
import tarfile
from typing import TYPE_CHECKING


def _patch_pip_symlink_extraction() -> None:
    """Mitigate GHSA-4xh5-x5gv-qwph for pip < 25.3.

    ``pip``'s fallback tar extraction path does not guard against symlinks that
    escape the target directory until the upstream patch scheduled for the 25.3
    release. A malicious sdist could therefore overwrite arbitrary files on the
    host running ``pip install``. We eagerly monkey patch pip's
    ``_untar_without_filter`` implementation with the logic from the upstream
    fix so that the fallback path validates symlink targets.
    """

    try:  # Lazy import so regular interpreters without pip keep working.
        import pip
        from packaging.version import InvalidVersion, Version
        from pip._internal.exceptions import InstallationError
        from pip._internal.utils import unpacking
    except Exception:  # pragma: no cover - defensive guard; pip may be absent.
        return

    try:
        version = Version(pip.__version__)
    except InvalidVersion:
        # Unknown version format – assume nothing to do instead of risking
        # breaking pip in unusual environments.
        return

    # pip 25.3 and newer ship the upstream fix; do not touch those versions.
    if version >= Version("25.3"):
        return

    # Avoid re-applying the patch when sitecustomize is imported repeatedly.
    if getattr(unpacking, "_tradepulse_symlink_patch", False):
        return

    ensure_dir = unpacking.ensure_dir
    installation_error = InstallationError
    is_within_directory = unpacking.is_within_directory
    logger = unpacking.logger
    set_mode = unpacking.set_extracted_file_to_default_mode_plus_executable
    split_leading_dir = unpacking.split_leading_dir

    def _is_symlink_target_in_tar(
        tar: tarfile.TarFile, tarinfo: tarfile.TarInfo
    ) -> bool:
        linkname = os.path.join(os.path.dirname(tarinfo.name), tarinfo.linkname)
        linkname = os.path.normpath(linkname)
        # Normalise backslashes so Windows style separators are handled.
        linkname = linkname.replace("\\", "/")
        try:
            tar.getmember(linkname)
            return True
        except KeyError:
            return False

    def _patched_untar_without_filter(
        filename: str,
        location: str,
        tar: tarfile.TarFile,
        leading: bool,
    ) -> None:
        for member in tar.getmembers():
            fn = member.name
            if leading:
                fn = split_leading_dir(fn)[1]
            path = os.path.join(location, fn)
            if not is_within_directory(location, path):
                message = (
                    "The tar file ({}) has a file ({}) trying to install "
                    "outside target directory ({})"
                )
                raise installation_error(message.format(filename, path, location))

            if member.isdir():
                ensure_dir(path)
                continue

            if member.issym():
                if not _is_symlink_target_in_tar(tar, member):
                    message = (
                        "The tar file ({}) has a file ({}) trying to install "
                        "outside target directory ({})"
                    )
                    raise installation_error(
                        message.format(filename, member.name, member.linkname)
                    )
                try:
                    tar._extract_member(member, path)
                except Exception as exc:  # pragma: no cover - mirrors pip logic.
                    logger.warning(
                        "In the tar file %s the member %s is invalid: %s",
                        filename,
                        member.name,
                        exc,
                    )
                continue

            try:
                fp = tar.extractfile(member)
            except (KeyError, AttributeError) as exc:
                logger.warning(
                    "In the tar file %s the member %s is invalid: %s",
                    filename,
                    member.name,
                    exc,
                )
                continue

            ensure_dir(os.path.dirname(path))
            assert fp is not None
            with open(path, "wb") as destfp:
                shutil.copyfileobj(fp, destfp)
            fp.close()
            tar.utime(member, path)
            if member.mode & 0o111:
                set_mode(path)

    # Publish helper to mirror upstream naming so introspection behaves as
    # expected and pip unit tests (if run) continue to function.
    unpacking.is_symlink_target_in_tar = _is_symlink_target_in_tar  # type: ignore[attr-defined]
    unpacking._untar_without_filter = _patched_untar_without_filter  # type: ignore[attr-defined]
    unpacking._tradepulse_symlink_patch = True


def _apply_patches() -> None:
    _patch_pip_symlink_extraction()


if not TYPE_CHECKING:
    _apply_patches()
