"""Cache validation utilities for TTL and checksum verification."""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


class CacheValidationError(Exception):
    """Base exception for cache validation errors."""

    pass


class TTLExpiredError(CacheValidationError):
    """Raised when cache item TTL has expired."""

    pass


class ChecksumMismatchError(CacheValidationError):
    """Raised when checksum validation fails."""

    pass


def is_ttl_valid(last_verified: str, ttl_seconds: int) -> bool:
    """Check if cache item is still valid based on TTL.

    Args:
        last_verified: ISO format timestamp of last verification
        ttl_seconds: Time-to-live in seconds

    Returns:
        True if cache is still valid, False if expired
    """
    if ttl_seconds is None:
        # None means never expire
        return True

    last_verified_dt = datetime.fromisoformat(last_verified)
    now = datetime.now(timezone.utc)

    # Handle timezone-naive datetimes
    if last_verified_dt.tzinfo is None:
        last_verified_dt = last_verified_dt.replace(tzinfo=timezone.utc)

    elapsed = (now - last_verified_dt).total_seconds()
    return elapsed < ttl_seconds


def compute_checksum(file_path: Path, algorithm: str = "md5") -> str:
    """Compute checksum for a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256')

    Returns:
        Hex digest of checksum

    Raises:
        ValueError: If algorithm not supported
    """
    if algorithm not in ("md5", "sha256"):
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if algorithm == "md5":
        hasher = hashlib.md5()
    else:
        hasher = hashlib.sha256()

    # Read file in chunks to handle large files
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def verify_checksum(
    file_path: Path,
    expected_checksum: str,
    algorithm: str = "md5",
    strict: bool = False,
) -> bool:
    """Verify file checksum matches expected value.

    Args:
        file_path: Path to file
        expected_checksum: Expected checksum value
        algorithm: Hash algorithm ('md5', 'sha256')
        strict: If True, raise exception on mismatch; if False, return False

    Returns:
        True if checksums match, False otherwise

    Raises:
        ChecksumMismatchError: If strict=True and checksums don't match
    """
    actual_checksum = compute_checksum(file_path, algorithm)

    if actual_checksum != expected_checksum:
        if strict:
            raise ChecksumMismatchError(
                f"Checksum mismatch for {file_path}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
        return False

    return True


def compute_checksum_from_bytes(data: bytes, algorithm: str = "md5") -> str:
    """Compute checksum from bytes.

    Args:
        data: Bytes to hash
        algorithm: Hash algorithm ('md5', 'sha256')

    Returns:
        Hex digest of checksum

    Raises:
        ValueError: If algorithm not supported
    """
    if algorithm not in ("md5", "sha256"):
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if algorithm == "md5":
        hasher = hashlib.md5()
    else:
        hasher = hashlib.sha256()

    hasher.update(data)
    return hasher.hexdigest()


def get_ttl_remaining(last_verified: str, ttl_seconds: int) -> Optional[int]:
    """Get remaining seconds until TTL expires.

    Args:
        last_verified: ISO format timestamp of last verification
        ttl_seconds: Time-to-live in seconds

    Returns:
        Seconds remaining, or None if never expires
    """
    if ttl_seconds is None:
        return None

    last_verified_dt = datetime.fromisoformat(last_verified)
    now = datetime.now(timezone.utc)

    # Handle timezone-naive datetimes
    if last_verified_dt.tzinfo is None:
        last_verified_dt = last_verified_dt.replace(tzinfo=timezone.utc)

    elapsed = (now - last_verified_dt).total_seconds()
    remaining = ttl_seconds - elapsed

    return max(0, int(remaining))
