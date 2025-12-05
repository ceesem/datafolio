"""Cache manager for handling local caching of remote bundles."""

import logging
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from filelock import FileLock, Timeout

from datafolio.cache.config import CacheConfig
from datafolio.cache.metadata import CacheMetadata
from datafolio.cache.validation import (
    ChecksumMismatchError,
    compute_checksum,
    get_ttl_remaining,
    is_ttl_valid,
    verify_checksum,
)

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base exception for cache-related errors."""

    pass


class CacheDiskFullError(CacheError):
    """Raised when disk is full and cannot write to cache."""

    pass


class CachePermissionError(CacheError):
    """Raised when cache directory permissions are insufficient."""

    pass


class CacheLockError(CacheError):
    """Raised when unable to acquire cache lock."""

    pass


class CacheManager:
    """Manages local caching for a DataFolio bundle.

    Provides TTL-based validation and checksum verification for cached items.
    Thread-safe using file-based locking.
    """

    def __init__(
        self,
        bundle_path: str,
        config: Optional[CacheConfig] = None,
        ttl_override: Optional[int] = None,
    ):
        """Initialize cache manager.

        Args:
            bundle_path: Original bundle path (e.g., 'gs://bucket/bundle')
            config: Cache configuration (uses global if None)
            ttl_override: Override default TTL for this bundle
        """
        self.bundle_path = bundle_path
        self.config = config or CacheConfig()

        # TTL can be overridden per bundle
        self.ttl = ttl_override if ttl_override is not None else self.config.default_ttl

        # Normalize bundle path to filesystem-safe directory name
        self.bundle_id = self._normalize_bundle_path(bundle_path)

        # If cache_dir was explicitly provided, use it directly as the bundle cache directory.
        # Otherwise, use a multi-bundle structure with bundles/{bundle_id} subdirectories.
        if self.config._is_explicit_cache_dir:
            # Explicit cache_dir: use it directly for this bundle
            self.cache_dir = self.config.cache_dir
            self.lock_dir = self.config.cache_dir / ".locks"
        else:
            # Default cache_dir: use multi-bundle structure
            self.cache_dir = self.config.cache_dir / "bundles" / self.bundle_id
            self.lock_dir = self.config.cache_dir / ".locks"
        try:
            self.lock_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise CachePermissionError(
                f"Cannot create cache lock directory at {self.lock_dir}: {e}"
            ) from e
        except OSError as e:
            logger.warning(f"Error creating cache lock directory: {e}")
            raise CacheError(
                f"Cannot access cache directory at {self.lock_dir}: {e}"
            ) from e

        # Initialize metadata
        try:
            self.metadata = CacheMetadata(self.cache_dir, bundle_path)
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            raise CacheError(f"Cannot initialize cache metadata: {e}") from e

    @staticmethod
    def _normalize_bundle_path(bundle_path: str) -> str:
        """Convert bundle path to filesystem-safe identifier.

        Args:
            bundle_path: Original bundle path

        Returns:
            Normalized identifier safe for filesystem

        Examples:
            >>> CacheManager._normalize_bundle_path('gs://bucket/path/bundle')
            'gs_bucket_path_bundle'
            >>> CacheManager._normalize_bundle_path('/local/path/bundle')
            'local_path_bundle'
        """
        # Remove protocol and replace slashes with underscores
        normalized = bundle_path.replace("://", "_").replace("/", "_")
        # Remove leading/trailing underscores
        normalized = normalized.strip("_")
        return normalized

    def _get_lock_path(self, item_name: str) -> Path:
        """Get lock file path for an item.

        Args:
            item_name: Name of the item

        Returns:
            Path to lock file
        """
        return self.lock_dir / f"{self.bundle_id}_{item_name}.lock"

    def _get_item_cache_path(
        self, item_name: str, item_type: str, filename: str
    ) -> Path:
        """Get cache path for an item.

        Args:
            item_name: Name of the item
            item_type: Type of item (determines subdirectory)
            filename: Filename to use

        Returns:
            Path where item should be cached
        """
        # Map item types to subdirectories (matches DataFolio structure)
        type_to_subdir = {
            "included_table": "tables",
            "referenced_table": "tables",
            "model": "models",
            "json": "artifacts",
            "yaml": "artifacts",
            "text": "artifacts",
            "binary": "artifacts",
            "image": "artifacts",
        }

        subdir = type_to_subdir.get(item_type, "artifacts")
        return self.cache_dir / subdir / filename

    def is_cached(self, item_name: str) -> bool:
        """Check if item exists in cache.

        Args:
            item_name: Name of the item

        Returns:
            True if item is cached
        """
        item_meta = self.metadata.get_item(item_name)
        if item_meta is None:
            return False

        # Check if file actually exists
        cache_path = self._get_item_cache_path(
            item_name, item_meta["item_type"], item_meta["filename"]
        )
        return cache_path.exists()

    def is_valid(self, item_name: str, remote_checksum: Optional[str] = None) -> bool:
        """Check if cached item is valid (TTL not expired).

        Args:
            item_name: Name of the item
            remote_checksum: Optional remote checksum to compare

        Returns:
            True if cache is valid and can be used
        """
        if not self.is_cached(item_name):
            return False

        item_meta = self.metadata.get_item(item_name)

        # Check TTL
        if not is_ttl_valid(item_meta["last_verified"], self.ttl):
            return False

        # Check checksum if provided
        if remote_checksum and item_meta["remote_checksum"] != remote_checksum:
            return False

        return True

    def _check_disk_space(self, required_bytes: int) -> None:
        """Check if sufficient disk space is available.

        Args:
            required_bytes: Number of bytes needed

        Raises:
            CacheDiskFullError: If insufficient disk space
        """
        try:
            stat = shutil.disk_usage(self.cache_dir.parent)
            available = stat.free
            # Require 10% buffer beyond required bytes
            required_with_buffer = int(required_bytes * 1.1)

            if available < required_with_buffer:
                raise CacheDiskFullError(
                    f"Insufficient disk space: {available / (1024**3):.2f} GB available, "
                    f"{required_with_buffer / (1024**3):.2f} GB required"
                )
        except CacheDiskFullError:
            raise
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            # Don't fail if we can't check disk space

    def get(
        self,
        item_name: str,
        remote_fetch_fn: Optional[Callable] = None,
        remote_checksum: Optional[str] = None,
    ) -> Optional[Path]:
        """Get cached item path.

        Args:
            item_name: Name of the item
            remote_fetch_fn: Function to fetch from remote if cache miss
            remote_checksum: Checksum from remote manifest

        Returns:
            Path to cached file, or None if not cached and no fetch function

        Raises:
            ChecksumMismatchError: If checksum validation fails in strict mode
            CacheLockError: If unable to acquire lock
            CacheDiskFullError: If insufficient disk space
            CachePermissionError: If cache directory not writable
        """
        lock_path = self._get_lock_path(item_name)

        try:
            with FileLock(lock_path, timeout=30):
                return self._get_locked(item_name, remote_fetch_fn, remote_checksum)
        except Timeout as e:
            raise CacheLockError(
                f"Timeout acquiring lock for {item_name} after 30 seconds"
            ) from e

    def _get_locked(
        self,
        item_name: str,
        remote_fetch_fn: Optional[Callable],
        remote_checksum: Optional[str],
    ) -> Optional[Path]:
        """Get cached item with lock already acquired."""
        # Check if cached and valid
        if self.is_valid(item_name, remote_checksum):
            # Cache hit
            item_meta = self.metadata.get_item(item_name)
            cache_path = self._get_item_cache_path(
                item_name, item_meta["item_type"], item_meta["filename"]
            )

            # Update access stats
            self.metadata.update_access(item_name)
            self.metadata.record_cache_hit()

            return cache_path

        # Cache miss or invalid - need to fetch
        if remote_fetch_fn is None:
            self.metadata.record_cache_miss()
            return None

        # Fetch from remote
        return self._fetch_and_cache(item_name, remote_fetch_fn, remote_checksum)

    def _fetch_and_cache(
        self,
        item_name: str,
        remote_fetch_fn: Callable,
        remote_checksum: Optional[str] = None,
    ) -> Path:
        """Fetch item from remote and cache it.

        Args:
            item_name: Name of the item
            remote_fetch_fn: Function to fetch data from remote
            remote_checksum: Expected checksum

        Returns:
            Path to cached file

        Raises:
            ChecksumMismatchError: If checksum validation fails
        """
        self.metadata.record_cache_miss()

        # Fetch data from remote
        # remote_fetch_fn should return (data_bytes, item_type, filename)
        data_bytes, item_type, filename = remote_fetch_fn()

        # Check file size limit
        data_size = len(data_bytes)
        if data_size > self.config.max_item_size:
            warnings.warn(
                f"Item {item_name} size ({data_size / (1024 * 1024):.2f} MB) exceeds "
                f"max_item_size ({self.config.max_item_size / (1024 * 1024):.2f} MB). "
                f"Skipping cache."
            )
            # Raise exception to indicate item cannot be cached
            raise ValueError(
                f"Item {item_name} exceeds maximum cache size "
                f"({data_size} > {self.config.max_item_size} bytes)"
            )

        # Check available disk space
        try:
            self._check_disk_space(data_size)
        except CacheDiskFullError as e:
            logger.error(f"Cannot cache {item_name}: {e}")
            raise

        # Get cache path
        cache_path = self._get_item_cache_path(item_name, item_type, filename)

        # Create parent directory with error handling
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise CachePermissionError(
                f"Cannot create cache directory {cache_path.parent}: {e}"
            ) from e
        except OSError as e:
            logger.error(f"OS error creating cache directory: {e}")
            raise CacheError(f"Cannot create cache directory: {e}") from e

        # Write to temp file first (atomic write)
        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")

        try:
            # Write to temp file
            try:
                with open(temp_path, "wb") as f:
                    f.write(data_bytes)
            except PermissionError as e:
                raise CachePermissionError(
                    f"Cannot write to cache file {temp_path}: {e}"
                ) from e
            except OSError as e:
                if e.errno == 28:  # ENOSPC - No space left on device
                    raise CacheDiskFullError(
                        f"Disk full while writing {item_name} to cache"
                    ) from e
                logger.error(f"OS error writing cache file: {e}")
                raise CacheError(f"Cannot write cache file: {e}") from e

            # Compute checksum
            try:
                actual_checksum = compute_checksum(
                    temp_path, self.config.checksum_algorithm
                )
            except Exception as e:
                logger.error(f"Error computing checksum for {item_name}: {e}")
                raise CacheError(f"Checksum computation failed: {e}") from e

            # Verify checksum if provided
            if remote_checksum and actual_checksum != remote_checksum:
                if self.config.strict_mode:
                    temp_path.unlink()
                    raise ChecksumMismatchError(
                        f"Checksum mismatch for {item_name}: "
                        f"expected {remote_checksum}, got {actual_checksum}"
                    )
                else:
                    warnings.warn(
                        f"Checksum mismatch for {item_name}, but continuing in non-strict mode"
                    )

            # Atomic rename
            try:
                temp_path.rename(cache_path)
            except OSError as e:
                logger.error(f"Error renaming temp file to cache path: {e}")
                raise CacheError(f"Cannot finalize cache file: {e}") from e

            # Update metadata
            try:
                self.metadata.set_item(
                    item_name,
                    item_type,
                    filename,
                    remote_checksum or actual_checksum,
                    cache_path.stat().st_size,
                )
            except Exception as e:
                logger.error(f"Error updating cache metadata: {e}")
                # Not critical - file is cached even if metadata update fails
                warnings.warn(f"Cache metadata update failed for {item_name}: {e}")

            return cache_path

        except (CacheError, ChecksumMismatchError, ValueError):
            # Re-raise our custom exceptions
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_path}: {e}")
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error caching {item_name}: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {cleanup_error}")
            raise CacheError(f"Unexpected error caching item: {e}") from e

    def invalidate(self, item_name: str) -> None:
        """Invalidate cache for an item (mark as needing refresh).

        Args:
            item_name: Name of the item
        """
        # Simply remove from metadata (keeps file for now)
        self.metadata.remove_item(item_name)

    def clear_item(self, item_name: str) -> None:
        """Remove item from cache entirely.

        Args:
            item_name: Name of the item
        """
        item_meta = self.metadata.get_item(item_name)
        if item_meta:
            cache_path = self._get_item_cache_path(
                item_name, item_meta["item_type"], item_meta["filename"]
            )
            if cache_path.exists():
                cache_path.unlink()

            self.metadata.remove_item(item_name)

    def clear_all(self) -> None:
        """Clear entire cache for this bundle."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.metadata = CacheMetadata(self.cache_dir, self.bundle_path)

    def get_status(self, item_name: str) -> Optional[Dict[str, Any]]:
        """Get cache status for an item.

        Args:
            item_name: Name of the item

        Returns:
            Status dict with cache information, or None if not cached
        """
        if not self.is_cached(item_name):
            return None

        item_meta = self.metadata.get_item(item_name)
        cache_path = self._get_item_cache_path(
            item_name, item_meta["item_type"], item_meta["filename"]
        )

        return {
            "cached": True,
            "cache_path": str(cache_path),
            "size_bytes": item_meta["size_bytes"],
            "cached_at": item_meta["cached_at"],
            "last_verified": item_meta["last_verified"],
            "last_accessed": item_meta["last_accessed"],
            "access_count": item_meta["access_count"],
            "ttl_remaining": get_ttl_remaining(item_meta["last_verified"], self.ttl),
            "remote_checksum": item_meta["remote_checksum"],
            "local_checksum": item_meta["local_checksum"],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Statistics dict
        """
        stats = self.metadata.get_stats()
        stats["bundle_path"] = self.bundle_path
        stats["cache_dir"] = str(self.cache_dir)
        stats["ttl_seconds"] = self.ttl

        # Calculate hit rate
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        stats["cache_hit_rate"] = (
            stats["cache_hits"] / total_requests if total_requests > 0 else 0.0
        )

        return stats
