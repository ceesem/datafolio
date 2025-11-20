"""Base handler interface for data type handlers.

This module defines the abstract base class that all data type handlers must implement.
Handlers are responsible for serialization, deserialization, and metadata management
for specific data types (e.g., pandas DataFrames, numpy arrays, PyTorch models).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from datafolio.storage.categories import StorageCategory, get_storage_category

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class BaseHandler(ABC):
    """Abstract base class for data type handlers.

    Each handler manages serialization, deserialization, and metadata
    for one data type (tables, arrays, models, etc.).

    Handlers are responsible for:
    1. Type detection (can_handle)
    2. Writing data to storage (add)
    3. Reading data from storage (get)
    4. File cleanup (delete) - optional override
    5. Storage location (get_storage_subdir) - automatically derived from item_type

    The storage location is automatically determined from the item_type using
    the ITEM_TYPE_TO_CATEGORY mapping. Handlers only need to define their
    item_type, and the storage category is looked up automatically.

    Examples:
        Create a custom handler (minimal implementation):
        >>> class MyHandler(BaseHandler):
        ...     @property
        ...     def item_type(self) -> str:
        ...         return "my_type"  # Must be in ITEM_TYPE_TO_CATEGORY
        ...
        ...     def can_handle(self, data: Any) -> bool:
        ...         return isinstance(data, MyDataType)
        ...
        ...     def add(self, folio, name, data, **kwargs):
        ...         # Write data and return metadata
        ...         ...
        ...
        ...     def get(self, folio, name, **kwargs):
        ...         # Read and return data
        ...         ...
        ...
        ...     # No need to implement get_storage_subdir() - it's automatic!
    """

    @property
    @abstractmethod
    def item_type(self) -> str:
        """Unique identifier for this item type.

        Must match the 'item_type' value stored in items.json.

        Returns:
            Item type string (e.g., 'included_table', 'numpy_array', 'pytorch_model')

        Examples:
            >>> handler = PandasHandler()
            >>> handler.item_type
            'included_table'
        """
        pass

    @abstractmethod
    def can_handle(self, data: Any) -> bool:
        """Check if this handler can process the given data.

        Used by add_data() for auto-detection. Return False if the handler
        should not participate in auto-detection (e.g., for reference tables
        or models that require explicit parameters).

        Args:
            data: Data object to check

        Returns:
            True if this handler supports this data type

        Examples:
            >>> handler = PandasHandler()
            >>> handler.can_handle(pd.DataFrame())
            True
            >>> handler.can_handle(np.array([1, 2, 3]))
            False
        """
        pass

    @abstractmethod
    def add(
        self,
        folio: "DataFolio",
        name: str,
        data: Any,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add data to folio and return metadata.

        Responsible for:
        1. Validating data type
        2. Writing data to storage
        3. Building metadata dictionary

        Args:
            folio: DataFolio instance (for accessing storage, bundle_dir, etc.)
            name: Item name
            data: Data to store
            description: Optional description
            inputs: Optional lineage inputs
            **kwargs: Handler-specific options

        Returns:
            Complete metadata dict to store in folio._items[name].
            Must include at minimum: 'name', 'item_type', 'created_at'.
            Should include 'description' and 'inputs' if provided.

        Raises:
            TypeError: If data is wrong type for this handler

        Examples:
            >>> metadata = handler.add(folio, 'results', df, description='Results')
            >>> # Returns: {'name': 'results', 'item_type': 'included_table',
            >>> #           'filename': 'results.parquet', 'num_rows': 100, ...}
        """
        pass

    @abstractmethod
    def get(self, folio: "DataFolio", name: str, **kwargs) -> Any:
        """Load and return data from folio.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Handler-specific options (passed to reader)

        Returns:
            The loaded data object

        Raises:
            KeyError: If item doesn't exist

        Examples:
            >>> df = handler.get(folio, 'results')
            >>> df = handler.get(folio, 'results', columns=['col1', 'col2'])
        """
        pass

    def delete(self, folio: "DataFolio", name: str) -> None:
        """Delete data files for this item.

        Default implementation deletes file at items[name]['filename']
        from the appropriate subdirectory. Override if custom logic needed
        (e.g., for external references that have no local files).

        Args:
            folio: DataFolio instance
            name: Item name

        Examples:
            Default behavior (can be overridden):
            >>> handler.delete(folio, 'results')
            # Deletes: {bundle_dir}/{subdir}/{filename}
        """
        item = folio._items[name]
        if "filename" in item:
            subdir = self.get_storage_subdir()
            filepath = folio._storage.join_paths(
                folio._bundle_dir, subdir, item["filename"]
            )
            folio._storage.delete_file(filepath)

    def get_storage_category(self) -> StorageCategory:
        """Get the storage category for this handler's item type.

        This method looks up the storage category in ITEM_TYPE_TO_CATEGORY
        based on the handler's item_type. Override this method if you need
        custom category logic that differs from the standard mapping.

        Returns:
            StorageCategory enum value

        Examples:
            >>> handler = PandasHandler()
            >>> handler.get_storage_category()
            <StorageCategory.TABLES: 'tables'>

            >>> handler = NumpyHandler()
            >>> handler.get_storage_category()
            <StorageCategory.ARTIFACTS: 'artifacts'>
        """
        return get_storage_category(self.item_type)

    def get_storage_subdir(self) -> str:
        """Return subdirectory for this data type.

        This method is derived from get_storage_category(). In most cases,
        you should not need to override this - override get_storage_category()
        instead if you need custom storage location logic.

        Returns:
            Subdirectory name: 'tables', 'models', or 'artifacts'

        Examples:
            >>> handler = PandasHandler()
            >>> handler.get_storage_subdir()
            'tables'

            >>> handler = SklearnHandler()
            >>> handler.get_storage_subdir()
            'models'
        """
        return self.get_storage_category().directory
