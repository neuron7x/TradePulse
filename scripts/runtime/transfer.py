"""Reliable transfer helpers supporting resume and checksum verification."""
from __future__ import annotations

# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

import requests

from .checksum import verify_checksum
from .progress import ProgressBar

_CHUNK_SIZE = 1024 * 512


class TransferError(RuntimeError):
    """Raised when a download or copy fails."""


def _is_url(source: str | os.PathLike[str]) -> bool:
    parsed = urlparse(str(source))
    return parsed.scheme in {"http", "https", "file"}


def _local_path_from_url(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise TransferError(f"Unsupported URL scheme for local path conversion: {parsed.scheme}")
    return Path(parsed.path)


def _open_source_file(path: Path, *, offset: int = 0) -> BinaryIO:
    handle = path.open("rb")
    if offset:
        handle.seek(offset)
    return handle


def _prepare_destination(destination: Path, *, expected_size: int | None) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing_size = destination.stat().st_size
        if expected_size is not None and existing_size > expected_size:
            destination.unlink()
            return 0
        return existing_size
    return 0


def transfer_with_resume(
    source: str | os.PathLike[str],
    destination: Path | str,
    *,
    session: requests.Session | None = None,
    expected_checksum: str | None = None,
    checksum_algorithm: str = "sha256",
    progress: ProgressBar | None = None,
) -> Path:
    """Copy *source* to *destination* with resume support and optional checksum validation."""

    destination_path = Path(destination)

    if not _is_url(source):
        source_path = Path(source)
        if not source_path.exists():
            raise TransferError(f"Source file does not exist: {source_path}")
        total_size = source_path.stat().st_size
        start_offset = _prepare_destination(destination_path, expected_size=total_size)
        with _open_source_file(source_path, offset=start_offset) as input_handle, destination_path.open(
            "ab" if start_offset else "wb"
        ) as output_handle:
            if progress:
                progress.total = total_size
                progress.update(start_offset)
            while chunk := input_handle.read(_CHUNK_SIZE):
                output_handle.write(chunk)
                if progress:
                    progress.advance(len(chunk))
        if expected_checksum:
            verify_checksum(destination_path, expected_checksum, algorithm=checksum_algorithm)
        return destination_path

    session = session or requests.Session()
    parsed = urlparse(str(source))
    if parsed.scheme == "file":
        return transfer_with_resume(
            _local_path_from_url(str(source)),
            destination_path,
            expected_checksum=expected_checksum,
            checksum_algorithm=checksum_algorithm,
            progress=progress,
        )

    head = session.head(str(source), allow_redirects=True, timeout=30)
    if head.status_code >= 400:
        raise TransferError(f"HEAD request failed with status {head.status_code}")
    total_size = int(head.headers.get("Content-Length", "0")) or None
    start_offset = _prepare_destination(destination_path, expected_size=total_size)

    headers = {}
    if start_offset and total_size and start_offset < total_size:
        headers["Range"] = f"bytes={start_offset}-"
    mode = "ab" if start_offset else "wb"
    response = session.get(str(source), stream=True, headers=headers, timeout=60)
    if response.status_code in {206, 200}:
        pass
    elif response.status_code in {429, 500, 502, 503, 504}:
        raise TransferError(f"Remote server returned retryable status {response.status_code}")
    elif response.status_code >= 400:
        raise TransferError(f"Download failed with status {response.status_code}")

    if progress:
        progress.total = total_size
        progress.update(start_offset)

    with destination_path.open(mode) as output_handle:
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
            if not chunk:
                continue
            output_handle.write(chunk)
            if progress:
                progress.advance(len(chunk))
    response.close()

    if expected_checksum:
        verify_checksum(destination_path, expected_checksum, algorithm=checksum_algorithm)

    return destination_path
