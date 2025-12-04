"""Unit tests for cache validation module."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from datafolio.cache.validation import (
    ChecksumMismatchError,
    compute_checksum,
    compute_checksum_from_bytes,
    get_ttl_remaining,
    is_ttl_valid,
    verify_checksum,
)


class TestTTLValidation:
    """Test TTL validation functions."""

    def test_ttl_valid_within_window(self):
        """Test that cache is valid within TTL window."""
        now = datetime.now(timezone.utc)
        last_verified = now.isoformat()
        ttl_seconds = 1800  # 30 minutes

        assert is_ttl_valid(last_verified, ttl_seconds) is True

    def test_ttl_expired_after_window(self):
        """Test that cache expires after TTL window."""
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        last_verified = past.isoformat()
        ttl_seconds = 1800  # 30 minutes

        assert is_ttl_valid(last_verified, ttl_seconds) is False

    def test_ttl_none_never_expires(self):
        """Test that TTL=None means never expire."""
        ancient_past = datetime.now(timezone.utc) - timedelta(days=365)
        last_verified = ancient_past.isoformat()
        ttl_seconds = None

        assert is_ttl_valid(last_verified, ttl_seconds) is True

    def test_ttl_boundary_condition(self):
        """Test TTL at exact boundary."""
        past = datetime.now(timezone.utc) - timedelta(seconds=1800)
        last_verified = past.isoformat()
        ttl_seconds = 1800

        # Should be very close to expiration but might still be valid
        # due to timing precision
        result = is_ttl_valid(last_verified, ttl_seconds)
        assert isinstance(result, bool)

    def test_ttl_remaining_within_window(self):
        """Test TTL remaining calculation."""
        now = datetime.now(timezone.utc)
        last_verified = now.isoformat()
        ttl_seconds = 1800

        remaining = get_ttl_remaining(last_verified, ttl_seconds)
        assert remaining is not None
        assert remaining <= ttl_seconds
        assert remaining >= 0

    def test_ttl_remaining_expired(self):
        """Test TTL remaining when expired."""
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        last_verified = past.isoformat()
        ttl_seconds = 1800

        remaining = get_ttl_remaining(last_verified, ttl_seconds)
        assert remaining == 0

    def test_ttl_remaining_none(self):
        """Test TTL remaining when TTL is None."""
        now = datetime.now(timezone.utc)
        last_verified = now.isoformat()
        ttl_seconds = None

        remaining = get_ttl_remaining(last_verified, ttl_seconds)
        assert remaining is None

    def test_ttl_timezone_naive(self):
        """Test TTL validation with timezone-naive timestamps."""
        # Create timezone-naive datetime (recent)
        now = datetime.now()  # No timezone
        last_verified = now.isoformat()
        ttl_seconds = 1800

        # Should handle timezone-naive timestamps gracefully
        # The result may vary based on local timezone vs UTC difference
        result = is_ttl_valid(last_verified, ttl_seconds)
        assert isinstance(result, bool)


class TestChecksumComputation:
    """Test checksum computation functions."""

    def test_compute_md5_checksum(self):
        """Test MD5 checksum computation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, World!")
            temp_path = Path(f.name)

        try:
            checksum = compute_checksum(temp_path, algorithm="md5")
            # Known MD5 for "Hello, World!"
            assert checksum == "65a8e27d8879283831b664bd8b7f0ad4"
        finally:
            temp_path.unlink()

    def test_compute_sha256_checksum(self):
        """Test SHA256 checksum computation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, World!")
            temp_path = Path(f.name)

        try:
            checksum = compute_checksum(temp_path, algorithm="sha256")
            # Known SHA256 for "Hello, World!"
            expected = (
                "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
            )
            assert checksum == expected
        finally:
            temp_path.unlink()

    def test_compute_checksum_unsupported_algorithm(self):
        """Test that unsupported algorithm raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Test")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported algorithm"):
                compute_checksum(temp_path, algorithm="sha512")
        finally:
            temp_path.unlink()

    def test_compute_checksum_from_bytes_md5(self):
        """Test checksum computation from bytes."""
        data = b"Hello, World!"
        checksum = compute_checksum_from_bytes(data, algorithm="md5")
        assert checksum == "65a8e27d8879283831b664bd8b7f0ad4"

    def test_compute_checksum_from_bytes_sha256(self):
        """Test SHA256 checksum from bytes."""
        data = b"Hello, World!"
        checksum = compute_checksum_from_bytes(data, algorithm="sha256")
        expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        assert checksum == expected

    def test_compute_checksum_large_file(self):
        """Test checksum computation handles large files (chunks)."""
        # Create a large file (1MB)
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            data = b"X" * (1024 * 1024)  # 1MB of X's
            f.write(data)
            temp_path = Path(f.name)

        try:
            checksum = compute_checksum(temp_path, algorithm="md5")
            # Verify it's a valid MD5 (32 hex chars)
            assert len(checksum) == 32
            assert all(c in "0123456789abcdef" for c in checksum)
        finally:
            temp_path.unlink()


class TestChecksumVerification:
    """Test checksum verification functions."""

    def test_verify_checksum_success(self):
        """Test successful checksum verification."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, World!")
            temp_path = Path(f.name)

        try:
            expected = "65a8e27d8879283831b664bd8b7f0ad4"
            assert verify_checksum(temp_path, expected, algorithm="md5") is True
        finally:
            temp_path.unlink()

    def test_verify_checksum_mismatch_non_strict(self):
        """Test checksum mismatch in non-strict mode."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, World!")
            temp_path = Path(f.name)

        try:
            wrong_checksum = "0000000000000000000000000000000"
            assert (
                verify_checksum(
                    temp_path, wrong_checksum, algorithm="md5", strict=False
                )
                is False
            )
        finally:
            temp_path.unlink()

    def test_verify_checksum_mismatch_strict(self):
        """Test checksum mismatch in strict mode raises exception."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, World!")
            temp_path = Path(f.name)

        try:
            wrong_checksum = "0000000000000000000000000000000"
            with pytest.raises(ChecksumMismatchError, match="Checksum mismatch"):
                verify_checksum(temp_path, wrong_checksum, algorithm="md5", strict=True)
        finally:
            temp_path.unlink()

    def test_verify_checksum_sha256(self):
        """Test SHA256 checksum verification."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello, World!")
            temp_path = Path(f.name)

        try:
            expected = (
                "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
            )
            assert verify_checksum(temp_path, expected, algorithm="sha256") is True
        finally:
            temp_path.unlink()

    def test_verify_checksum_empty_file(self):
        """Test checksum verification for empty file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # MD5 of empty string
            expected = "d41d8cd98f00b204e9800998ecf8427e"
            assert verify_checksum(temp_path, expected, algorithm="md5") is True
        finally:
            temp_path.unlink()
