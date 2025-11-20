"""Storage backend for file I/O operations.

This module provides abstraction for file system operations,
supporting both local and cloud storage, along with storage
category definitions for organizing data types.
"""

from datafolio.storage.backend import StorageBackend
from datafolio.storage.categories import (
    ITEM_TYPE_TO_CATEGORY,
    StorageCategory,
    get_storage_category,
    get_storage_directory,
)

__all__ = [
    "StorageBackend",
    "StorageCategory",
    "ITEM_TYPE_TO_CATEGORY",
    "get_storage_category",
    "get_storage_directory",
]
