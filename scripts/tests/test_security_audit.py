from __future__ import annotations

# SPDX-License-Identifier: MIT

from pathlib import Path

from scripts.security_audit import scan_paths


def test_scan_paths_detects_tokens(tmp_path: Path) -> None:
    sample = tmp_path / "credentials.txt"
    sample.write_text(
        """
        API_KEY=not-a-match
        AWS_KEY=AKIA1234567890ABCDEF
        placeholder=changeme
        GH_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz1234567890
        """.strip()
    )

    findings = scan_paths([tmp_path])

    assert len(findings) == 2
    patterns = {finding.pattern for finding in findings}
    assert "AWS access key" in patterns
    assert "GitHub personal access token" in patterns


def test_scan_paths_skips_placeholders(tmp_path: Path) -> None:
    sample = tmp_path / "env.txt"
    sample.write_text(
        """
        AWS_ACCESS_KEY_ID=your_aws_access_key
        STRIPE_KEY=sk_test_placeholder
        """.strip()
    )

    findings = scan_paths([tmp_path])

    assert findings == []
