"""Storage category definitions for data organization.

This module defines the storage categories used to organize different data types
within a DataFolio bundle. Each category maps to a subdirectory and can have
category-specific configuration.
"""

from enum import Enum
from typing import Dict


class StorageCategory(Enum):
    """Storage categories for organizing data types.

    Each category represents a logical grouping of related data types
    and maps to a subdirectory within the bundle.

    Categories:
        TABLES: Tabular data (pandas DataFrames, references)
        MODELS: Machine learning models (sklearn, PyTorch)
        ARTIFACTS: Miscellaneous data (arrays, JSON, files, timestamps)
        VIEWS: Saved queries and virtual tables (reserved for future use)

    Examples:
        >>> StorageCategory.TABLES.directory
        'tables'

        >>> StorageCategory.MODELS.directory
        'models'
    """

    TABLES = "tables"
    MODELS = "models"
    ARTIFACTS = "artifacts"
    VIEWS = "views"

    @property
    def directory(self) -> str:
        """Get the directory name for this category.

        Returns:
            Directory name as string

        Examples:
            >>> StorageCategory.TABLES.directory
            'tables'
        """
        return self.value


# Mapping from item_type to storage category
# This is the single source of truth for how item types are organized
ITEM_TYPE_TO_CATEGORY: Dict[str, StorageCategory] = {
    # Tables
    "included_table": StorageCategory.TABLES,
    "polars_table": StorageCategory.TABLES,
    "referenced_table": StorageCategory.TABLES,
    # Arrays and JSON
    "numpy_array": StorageCategory.ARTIFACTS,
    "json_data": StorageCategory.ARTIFACTS,
    # Models
    "model": StorageCategory.MODELS,
    "pytorch_model": StorageCategory.MODELS,
    # Artifacts and timestamps
    "artifact": StorageCategory.ARTIFACTS,
    "timestamp": StorageCategory.ARTIFACTS,
}


def get_storage_category(item_type: str) -> StorageCategory:
    """Get the storage category for a given item type.

    Args:
        item_type: Item type identifier (e.g., 'included_table')

    Returns:
        StorageCategory enum value

    Raises:
        KeyError: If item_type is not recognized

    Examples:
        >>> get_storage_category('included_table')
        <StorageCategory.TABLES: 'tables'>

        >>> get_storage_category('pytorch_model')
        <StorageCategory.MODELS: 'models'>
    """
    return ITEM_TYPE_TO_CATEGORY[item_type]


def get_storage_directory(item_type: str) -> str:
    """Get the storage directory name for a given item type.

    Convenience function that combines category lookup and directory access.

    Args:
        item_type: Item type identifier (e.g., 'included_table')

    Returns:
        Directory name as string

    Raises:
        KeyError: If item_type is not recognized

    Examples:
        >>> get_storage_directory('included_table')
        'tables'

        >>> get_storage_directory('model')
        'models'
    """
    return get_storage_category(item_type).directory
