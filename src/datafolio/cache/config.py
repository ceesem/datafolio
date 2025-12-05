"""Cache configuration management."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CacheConfig:
    """Configuration for local caching system.

    Attributes:
        enabled: Whether caching is enabled
        cache_dir: Root directory for cache storage. If not explicitly provided,
            defaults to ~/.datafolio_cache and will use a multi-bundle structure
            with bundles/{bundle_id} subdirectories. If explicitly provided,
            will be used as the exact cache directory for a single bundle.
        default_ttl: Default time-to-live in seconds (30 minutes)
        max_cache_size: Maximum total cache size in bytes (None = unlimited)
        max_item_size: Maximum size per item in bytes (250 MB default)
        checksum_algorithm: Algorithm for checksum validation ('md5', 'sha256')
        strict_mode: If True, fail on checksum mismatch; if False, warn and re-download
        fallback_to_stale: Use stale cache on network error
        max_stale_age: Maximum age for stale cache in seconds (24 hours)
        _is_explicit_cache_dir: Internal flag tracking if cache_dir was explicitly provided
    """

    enabled: bool = False
    cache_dir: Path = Path.home() / ".datafolio_cache"
    default_ttl: int = 1800  # 30 minutes
    max_cache_size: Optional[int] = None  # bytes (total cache size)
    max_item_size: int = 250 * 1024 * 1024  # 250 MB per item
    checksum_algorithm: str = "md5"
    strict_mode: bool = False
    fallback_to_stale: bool = True
    max_stale_age: int = 86400  # 24 hours
    _is_explicit_cache_dir: bool = False

    def __post_init__(self):
        """Ensure cache_dir is a Path object and track if it was explicitly provided."""
        if self.cache_dir is None:
            # Use default if None was explicitly passed
            self.cache_dir = Path.home() / ".datafolio_cache"
            self._is_explicit_cache_dir = False
        elif not isinstance(self.cache_dir, Path):
            # String path provided - convert and mark as explicit
            self.cache_dir = Path(self.cache_dir).expanduser()
            self._is_explicit_cache_dir = True
        else:
            # Path object provided - check if it's the default
            expanded = self.cache_dir.expanduser()
            default_path = Path.home() / ".datafolio_cache"
            # Mark as explicit only if it's not the default path
            self._is_explicit_cache_dir = expanded != default_path
            self.cache_dir = expanded

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "CacheConfig":
        """Load configuration from file.

        Args:
            config_path: Path to config file. If None, uses default location.

        Returns:
            CacheConfig instance
        """
        if config_path is None:
            config_path = Path.home() / ".datafolio_cache" / "config.json"

        if not config_path.exists():
            return cls()

        with open(config_path, "r") as f:
            data = json.load(f)

        # Convert cache_dir string to Path
        if "cache_dir" in data:
            data["cache_dir"] = Path(data["cache_dir"])

        return cls(**data)

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to file.

        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            config_path = self.cache_dir / "config.json"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir),
            "default_ttl": self.default_ttl,
            "max_cache_size": self.max_cache_size,
            "max_item_size": self.max_item_size,
            "checksum_algorithm": self.checksum_algorithm,
            "strict_mode": self.strict_mode,
            "fallback_to_stale": self.fallback_to_stale,
            "max_stale_age": self.max_stale_age,
        }

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create configuration from environment variables.

        Environment variables:
            DATAFOLIO_CACHE_ENABLED: Enable caching (true/false)
            DATAFOLIO_CACHE_DIR: Cache directory path
            DATAFOLIO_CACHE_TTL: Default TTL in seconds
            DATAFOLIO_CACHE_MAX_ITEM_SIZE: Maximum item size in bytes

        Returns:
            CacheConfig instance
        """
        config = cls()

        if os.getenv("DATAFOLIO_CACHE_ENABLED"):
            config.enabled = os.getenv("DATAFOLIO_CACHE_ENABLED", "").lower() == "true"

        if os.getenv("DATAFOLIO_CACHE_DIR"):
            config.cache_dir = Path(os.getenv("DATAFOLIO_CACHE_DIR"))

        if os.getenv("DATAFOLIO_CACHE_TTL"):
            config.default_ttl = int(os.getenv("DATAFOLIO_CACHE_TTL"))

        if os.getenv("DATAFOLIO_CACHE_MAX_ITEM_SIZE"):
            config.max_item_size = int(os.getenv("DATAFOLIO_CACHE_MAX_ITEM_SIZE"))

        return config


# Global cache configuration instance
_global_config: Optional[CacheConfig] = None


def get_global_config() -> CacheConfig:
    """Get global cache configuration.

    Returns:
        Global CacheConfig instance
    """
    global _global_config
    if _global_config is None:
        # Try loading from file, then env, then defaults
        try:
            _global_config = CacheConfig.load()
        except Exception:
            _global_config = CacheConfig.from_env()
    return _global_config


def set_global_config(config: CacheConfig) -> None:
    """Set global cache configuration.

    Args:
        config: CacheConfig instance to use globally
    """
    global _global_config
    _global_config = config
