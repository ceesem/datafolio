"""Cache metadata management."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class CacheMetadata:
    """Manages cache metadata for a bundle.

    The metadata file (.cache_meta.json) tracks:
    - Bundle information (path, checksum)
    - Per-item cache status (checksums, timestamps, stats)
    - Cache statistics (hits, misses, size)
    """

    def __init__(self, cache_dir: Path, bundle_path: str):
        """Initialize cache metadata manager.

        Args:
            cache_dir: Directory where cache metadata is stored
            bundle_path: Original bundle path (e.g., 'gs://bucket/bundle')
        """
        self.cache_dir = Path(cache_dir)
        self.bundle_path = bundle_path
        self.meta_path = self.cache_dir / ".cache_meta.json"
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load metadata from file or create new."""
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                # Corrupted metadata, start fresh
                self._initialize_new()
        else:
            self._initialize_new()

    def _initialize_new(self) -> None:
        """Initialize new metadata structure."""
        self._data = {
            "schema_version": "1.0",
            "bundle_path": self.bundle_path,
            "bundle_checksum": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_manifest_check": None,
            "cache_config": {
                "ttl_seconds": 1800,
                "checksum_algorithm": "md5",
                "enabled": True,
            },
            "items": {},
            "stats": {
                "total_size_bytes": 0,
                "total_items": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "remote_checks": 0,
            },
        }

    def save(self) -> None:
        """Save metadata to file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get_item(self, item_name: str) -> Optional[Dict[str, Any]]:
        """Get cache metadata for an item.

        Args:
            item_name: Name of the item

        Returns:
            Item metadata dict or None if not cached
        """
        return self._data["items"].get(item_name)

    def set_item(
        self,
        item_name: str,
        item_type: str,
        filename: str,
        remote_checksum: str,
        size_bytes: int,
    ) -> None:
        """Set cache metadata for an item.

        Args:
            item_name: Name of the item
            item_type: Type of item (e.g., 'included_table', 'model')
            filename: Filename in cache
            remote_checksum: Checksum from remote manifest
            size_bytes: Size of cached file
        """
        now = datetime.now(timezone.utc).isoformat()

        # Get existing item or create new
        item = self._data["items"].get(
            item_name,
            {
                "access_count": 0,
            },
        )

        # Update item metadata
        item.update(
            {
                "item_type": item_type,
                "filename": filename,
                "remote_checksum": remote_checksum,
                "local_checksum": remote_checksum,  # Assume valid on write
                "cached_at": now,
                "last_verified": now,
                "last_accessed": now,
                "size_bytes": size_bytes,
            }
        )

        self._data["items"][item_name] = item

        # Update stats
        if item["access_count"] == 0:
            self._data["stats"]["total_items"] += 1
            self._data["stats"]["total_size_bytes"] += size_bytes

        self.save()

    def update_access(self, item_name: str) -> None:
        """Update access statistics for an item.

        Args:
            item_name: Name of the item
        """
        if item_name in self._data["items"]:
            self._data["items"][item_name]["access_count"] += 1
            self._data["items"][item_name]["last_accessed"] = datetime.now(
                timezone.utc
            ).isoformat()
            self.save()

    def update_verification(self, item_name: str) -> None:
        """Update last verification timestamp.

        Args:
            item_name: Name of the item
        """
        if item_name in self._data["items"]:
            self._data["items"][item_name]["last_verified"] = datetime.now(
                timezone.utc
            ).isoformat()
            self.save()

    def record_cache_hit(self) -> None:
        """Record a cache hit in statistics."""
        self._data["stats"]["cache_hits"] += 1
        self.save()

    def record_cache_miss(self) -> None:
        """Record a cache miss in statistics."""
        self._data["stats"]["cache_misses"] += 1
        self.save()

    def record_remote_check(self) -> None:
        """Record a remote manifest check."""
        self._data["stats"]["remote_checks"] += 1
        self._data["last_manifest_check"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def remove_item(self, item_name: str) -> None:
        """Remove item from cache metadata.

        Args:
            item_name: Name of the item to remove
        """
        if item_name in self._data["items"]:
            item = self._data["items"][item_name]
            self._data["stats"]["total_items"] -= 1
            self._data["stats"]["total_size_bytes"] -= item.get("size_bytes", 0)
            del self._data["items"][item_name]
            self.save()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Statistics dict with cache metrics
        """
        return self._data["stats"].copy()

    def get_all_items(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached items.

        Returns:
            Dict mapping item names to metadata
        """
        return self._data["items"].copy()

    def set_bundle_checksum(self, checksum: str) -> None:
        """Set bundle manifest checksum.

        Args:
            checksum: MD5 checksum of manifest
        """
        self._data["bundle_checksum"] = checksum
        self.save()

    def get_bundle_checksum(self) -> Optional[str]:
        """Get bundle manifest checksum.

        Returns:
            Checksum string or None
        """
        return self._data.get("bundle_checksum")
