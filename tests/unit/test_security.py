# SPDX-License-Identifier: MIT
"""Tests for security scanning and secret detection module."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from core.utils.security import (
    SecretDetector,
    check_for_hardcoded_secrets,
    SECRET_PATTERNS,
)


class TestSecretPatterns:
    """Test secret pattern matching."""

    def test_api_key_pattern(self) -> None:
        """Should match API key patterns."""
        pattern = SECRET_PATTERNS["api_key"]
        
        assert pattern.search('api_key = "abc123def456"')
        assert pattern.search('API_KEY = "xyz789"')
        assert pattern.search('apikey: "test123"')
        assert not pattern.search('api_key = variable')

    def test_password_pattern(self) -> None:
        """Should match password patterns."""
        pattern = SECRET_PATTERNS["password"]
        
        assert pattern.search('password = "secret123"')
        assert pattern.search('PASSWORD: "p@ssw0rd"')
        assert pattern.search('pwd = "test"')

    def test_aws_key_pattern(self) -> None:
        """Should match AWS access key patterns."""
        pattern = SECRET_PATTERNS["aws_key"]
        
        assert pattern.search("AKIAIOSFODNN7EXAMPLE")
        assert pattern.search("ASIATESTACCESSKEY123")

    def test_github_token_pattern(self) -> None:
        """Should match GitHub personal access tokens."""
        pattern = SECRET_PATTERNS["github_token"]
        
        assert pattern.search("ghp_" + "x" * 36)

    def test_jwt_token_pattern(self) -> None:
        """Should match JWT tokens."""
        pattern = SECRET_PATTERNS["jwt_token"]
        
        assert pattern.search("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U")

    def test_private_key_pattern(self) -> None:
        """Should match private key headers."""
        pattern = SECRET_PATTERNS["private_key"]
        
        assert pattern.search("-----BEGIN RSA PRIVATE KEY-----")
        assert pattern.search("-----BEGIN PRIVATE KEY-----")
        assert pattern.search("-----BEGIN EC PRIVATE KEY-----")


class TestSecretDetector:
    """Test SecretDetector class."""

    def test_init_default_patterns(self) -> None:
        """Should initialize with default patterns."""
        detector = SecretDetector()
        assert len(detector.patterns) == len(SECRET_PATTERNS)

    def test_init_custom_patterns(self) -> None:
        """Should accept custom patterns."""
        custom = {"custom": re.compile(r"custom_pattern")}
        detector = SecretDetector(custom_patterns=custom)
        
        assert "custom" in detector.patterns
        assert len(detector.patterns) == len(SECRET_PATTERNS) + 1

    def test_scan_file_detects_secrets(self) -> None:
        """Should detect secrets in a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('api_key = "secret123"\n')
            f.write('password = "p@ssw0rd"\n')
            f.flush()
            filepath = f.name
        
        try:
            detector = SecretDetector()
            findings = detector.scan_file(filepath)
            
            assert len(findings) >= 2
            secret_types = [finding[0] for finding in findings]
            assert "api_key" in secret_types
            assert "password" in secret_types
        finally:
            Path(filepath).unlink()

    def test_scan_file_returns_line_numbers(self) -> None:
        """Should return correct line numbers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('# Line 1\n')
            f.write('api_key = "secret123"\n')  # Line 2
            f.write('# Line 3\n')
            f.flush()
            filepath = f.name
        
        try:
            detector = SecretDetector()
            findings = detector.scan_file(filepath)
            
            assert len(findings) >= 1
            _, line_num, _ = findings[0]
            assert line_num == 2
        finally:
            Path(filepath).unlink()

    def test_scan_file_masks_secrets(self) -> None:
        """Should mask secrets in output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('api_key = "verylongsecretkey123"\n')
            f.flush()
            filepath = f.name
        
        try:
            detector = SecretDetector()
            findings = detector.scan_file(filepath)
            
            assert len(findings) >= 1
            _, _, masked_line = findings[0]
            # Secret should be masked
            assert "verylongsecretkey123" not in masked_line
            assert "********" in masked_line
        finally:
            Path(filepath).unlink()

    def test_scan_file_ignores_test_files(self) -> None:
        """Should ignore test files."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="test_", delete=False
        ) as f:
            f.write('api_key = "secret123"\n')
            f.flush()
            filepath = f.name
        
        try:
            detector = SecretDetector()
            findings = detector.scan_file(filepath)
            
            # Test files should be ignored
            assert len(findings) == 0
        finally:
            Path(filepath).unlink()

    def test_scan_file_ignores_example_env(self) -> None:
        """Should ignore .env.example files."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env.example", delete=False
        ) as f:
            f.write('API_KEY=secret123\n')
            f.flush()
            filepath = f.name
        
        try:
            detector = SecretDetector()
            findings = detector.scan_file(filepath)
            
            # .env.example should be ignored
            assert len(findings) == 0
        finally:
            Path(filepath).unlink()

    def test_scan_file_handles_unreadable_files(self) -> None:
        """Should gracefully handle unreadable files."""
        detector = SecretDetector()
        findings = detector.scan_file("/nonexistent/file.py")
        
        # Should return empty list, not raise
        assert findings == []

    def test_scan_directory_finds_secrets(self) -> None:
        """Should scan directory recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with secret
            secret_file = Path(tmpdir, "config.py")
            secret_file.write_text('api_key = "secret123"\n')
            
            # Create subdirectory with secret
            subdir = Path(tmpdir, "subdir")
            subdir.mkdir()
            sub_secret = Path(subdir, "auth.py")
            sub_secret.write_text('password = "p@ssw0rd"\n')
            
            detector = SecretDetector()
            results = detector.scan_directory(tmpdir)
            
            assert len(results) >= 2
            assert "config.py" in results
            assert "subdir/auth.py" in results or "subdir\\auth.py" in results

    def test_scan_directory_filters_extensions(self) -> None:
        """Should filter by file extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Python file with secret
            py_file = Path(tmpdir, "config.py")
            py_file.write_text('api_key = "secret123"\n')
            
            # Create text file with secret (should be ignored)
            txt_file = Path(tmpdir, "config.txt")
            txt_file.write_text('api_key = "secret123"\n')
            
            detector = SecretDetector()
            results = detector.scan_directory(tmpdir, extensions=[".py"])
            
            assert "config.py" in results
            assert "config.txt" not in results

    def test_scan_directory_empty_for_clean_dir(self) -> None:
        """Should return empty dict for clean directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create clean file
            clean_file = Path(tmpdir, "clean.py")
            clean_file.write_text('# No secrets here\nprint("Hello")\n')
            
            detector = SecretDetector()
            results = detector.scan_directory(tmpdir)
            
            assert len(results) == 0

    def test_scan_directory_skips_subdirectories(self) -> None:
        """Should skip git and cache directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .git directory with "secrets"
            git_dir = Path(tmpdir, ".git")
            git_dir.mkdir()
            git_file = Path(git_dir, "config")
            git_file.write_text('api_key = "secret123"\n')
            
            detector = SecretDetector()
            results = detector.scan_directory(tmpdir)
            
            # .git files should be ignored
            assert len(results) == 0


class TestCheckForHardcodedSecrets:
    """Test check_for_hardcoded_secrets function."""

    def test_returns_false_for_clean_directory(self) -> None:
        """Should return False when no secrets found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_file = Path(tmpdir, "clean.py")
            clean_file.write_text('print("Hello, World!")\n')
            
            result = check_for_hardcoded_secrets(tmpdir)
            assert result is False

    def test_returns_true_when_secrets_found(self) -> None:
        """Should return True when secrets detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_file = Path(tmpdir, "config.py")
            secret_file.write_text('api_key = "secret123"\n')
            
            result = check_for_hardcoded_secrets(tmpdir)
            assert result is True

    def test_prints_findings(self, capsys) -> None:
        """Should print findings to stdout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_file = Path(tmpdir, "config.py")
            secret_file.write_text('api_key = "secret123"\n')
            
            check_for_hardcoded_secrets(tmpdir)
            captured = capsys.readouterr()
            
            assert "config.py" in captured.out
            assert "api_key" in captured.out

    def test_prints_success_for_clean_dir(self, capsys) -> None:
        """Should print success message for clean directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_file = Path(tmpdir, "clean.py")
            clean_file.write_text('print("Hello")\n')
            
            check_for_hardcoded_secrets(tmpdir)
            captured = capsys.readouterr()
            
            assert "No hardcoded secrets detected" in captured.out
            assert "✓" in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_detector_handles_binary_files(self) -> None:
        """Should handle binary files gracefully."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b'\x00\x01\x02\x03')
            f.flush()
            filepath = f.name
        
        try:
            detector = SecretDetector()
            findings = detector.scan_file(filepath)
            # Should not crash, returns empty
            assert isinstance(findings, list)
        finally:
            Path(filepath).unlink()

    def test_detector_handles_empty_file(self) -> None:
        """Should handle empty files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.flush()
            filepath = f.name
        
        try:
            detector = SecretDetector()
            findings = detector.scan_file(filepath)
            assert findings == []
        finally:
            Path(filepath).unlink()

    def test_mask_line_preserves_short_strings(self) -> None:
        """Should not mask short strings."""
        detector = SecretDetector()
        line = 'key = "short"'
        masked = detector._mask_line(line)
        # Short strings (< 8 chars) should not be masked
        assert "short" in masked

    def test_multiple_secrets_same_line(self) -> None:
        """Should detect multiple secrets on same line."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('api_key = "secret123" and password = "p@ssw0rd"\n')
            f.flush()
            filepath = f.name
        
        try:
            detector = SecretDetector()
            findings = detector.scan_file(filepath)
            
            # Should find both secrets
            assert len(findings) >= 2
        finally:
            Path(filepath).unlink()
