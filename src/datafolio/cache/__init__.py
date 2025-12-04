"""Local caching system for cloud-based DataFolios.

This module provides transparent caching of remote bundle data with TTL-based
validation and checksum verification.

Key components:
- CacheManager: Main cache interface
- CacheConfig: Configuration management
- CacheMetadata: Metadata operations
- CacheValidator: TTL and checksum validation
"""

from datafolio.cache.config import CacheConfig
from datafolio.cache.manager import CacheManager

__all__ = ["CacheManager", "CacheConfig"]
