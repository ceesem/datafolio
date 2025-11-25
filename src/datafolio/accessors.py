"""Data accessors for convenient item access.

This module provides ItemProxy and DataAccessor classes that enable
autocomplete-friendly access to DataFolio items using both attribute
and dictionary-style syntax.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class ItemProxy:
    """Proxy for accessing a single item with autocomplete-friendly properties.

    Provides convenient property-based access to item data and metadata.
    """

    def __init__(self, folio: "DataFolio", name: str):
        """Initialize ItemProxy.

        Args:
            folio: Parent DataFolio instance
            name: Name of the item
        """
        self._folio = folio
        self._name = name

    @property
    def content(self) -> Any:
        """Get the content of this item.

        Returns:
            - For tables: DataFrame
            - For numpy arrays: numpy array
            - For JSON data: dict/list/scalar
            - For timestamps: datetime object (UTC-aware)
            - For models: loaded model object
            - For artifacts: file path string (use with open())

        Examples:
            >>> df = folio.data.results.content  # DataFrame
            >>> arr = folio.data.embeddings.content  # numpy array
            >>> cfg = folio.data.config.content  # dict
            >>> ts = folio.data.event_time.content  # datetime
            >>> model = folio.data.classifier.content  # model object
            >>> with open(folio.data.plot.content, 'rb') as f:  # file path
            ...     img = f.read()
        """
        # Auto-refresh before accessing
        self._folio._refresh_if_needed()

        item = self._folio._items[self._name]
        item_type = item.get("item_type")

        # Dispatch to appropriate getter
        if item_type in ("referenced_table", "included_table", "polars_table"):
            return self._folio.get_table(self._name)
        elif item_type == "numpy_array":
            return self._folio.get_numpy(self._name)
        elif item_type == "json_data":
            return self._folio.get_json(self._name)
        elif item_type == "timestamp":
            return self._folio.get_timestamp(self._name)
        elif item_type in ("model", "pytorch_model"):
            return self._folio.get_model(self._name)
        elif item_type == "artifact":
            return self._folio.get_artifact_path(self._name)
        else:
            raise ValueError(f"Unknown item type: {item_type}")

    @property
    def description(self) -> Optional[str]:
        """Get the description of this item.

        Returns:
            Description string or None if not set
        """
        # Auto-refresh before accessing
        self._folio._refresh_if_needed()

        item = self._folio._items[self._name]
        return item.get("description")

    @property
    def type(self) -> str:
        """Get the type of this item.

        Returns:
            Item type string ('referenced_table', 'included_table', 'model',
            'pytorch_model', 'numpy_array', 'json_data', 'artifact')
        """
        # Auto-refresh before accessing
        self._folio._refresh_if_needed()

        item = self._folio._items[self._name]
        return item.get("item_type", "unknown")

    @property
    def path(self) -> Optional[str]:
        """Get the file path for this item.

        Returns:
            - For referenced tables: external file path
            - For artifacts: artifact file path
            - For other types: None

        Examples:
            >>> folio.data.external_data.path  # 's3://bucket/data.parquet'
            >>> folio.data.plot.path  # '/path/to/bundle/artifacts/plot.png'
        """
        # Auto-refresh before accessing
        self._folio._refresh_if_needed()

        item = self._folio._items[self._name]
        item_type = item.get("item_type")

        if item_type == "referenced_table":
            return item.get("path")
        elif item_type == "artifact":
            return self._folio.get_artifact_path(self._name)
        else:
            return None

    @property
    def inputs(self) -> list[str]:
        """Get the list of items this item depends on (lineage).

        Returns:
            List of item names that were used to create this item
        """
        return self._folio.get_inputs(self._name)

    @property
    def dependents(self) -> list[str]:
        """Get the list of items that depend on this item (lineage).

        Returns:
            List of item names that use this item as input
        """
        return self._folio.get_dependents(self._name)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the full metadata dictionary for this item.

        Returns:
            Dictionary containing all item metadata
        """
        # Auto-refresh before accessing
        self._folio._refresh_if_needed()

        return dict(self._folio._items[self._name])

    def __repr__(self) -> str:
        """Return string representation."""
        # Auto-refresh before accessing
        self._folio._refresh_if_needed()

        item = self._folio._items[self._name]
        item_type = item.get("item_type", "unknown")
        desc = item.get("description", "")
        desc_str = f": {desc}" if desc else ""
        return f"ItemProxy('{self._name}', type='{item_type}'{desc_str})"


class DataAccessor:
    """Accessor for autocomplete-friendly item access.

    Supports both attribute-style (folio.data.my_item) and
    dictionary-style (folio.data['my_item']) access.
    """

    def __init__(self, folio: "DataFolio"):
        """Initialize DataAccessor.

        Args:
            folio: Parent DataFolio instance
        """
        self._folio = folio

    def __getattr__(self, name: str) -> ItemProxy:
        """Get item by attribute access.

        Args:
            name: Item name

        Returns:
            ItemProxy for the item

        Raises:
            AttributeError: If item doesn't exist
        """
        if name.startswith("_"):
            # Allow access to private attributes
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Auto-refresh before accessing
        self._folio._refresh_if_needed()

        if name not in self._folio._items:
            raise AttributeError(
                f"Item '{name}' not found in DataFolio. "
                f"Available items: {', '.join(sorted(self._folio._items.keys()))}"
            )

        return ItemProxy(self._folio, name)

    def __getitem__(self, name: str) -> ItemProxy:
        """Get item by dictionary access.

        Args:
            name: Item name

        Returns:
            ItemProxy for the item

        Raises:
            KeyError: If item doesn't exist
        """
        # Auto-refresh before accessing
        self._folio._refresh_if_needed()

        if name not in self._folio._items:
            raise KeyError(f"Item '{name}' not found in DataFolio")

        return ItemProxy(self._folio, name)

    def __dir__(self) -> list[str]:
        """Return list of item names for autocomplete.

        Returns:
            Sorted list of all item names
        """
        # Auto-refresh before accessing
        self._folio._refresh_if_needed()

        # Include item names for autocomplete
        return sorted(self._folio._items.keys())

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String showing available items
        """
        items = sorted(self._folio._items.keys())
        if not items:
            return "DataAccessor(no items)"

        # Group by type
        contents = self._folio.list_contents()
        lines = ["DataAccessor:"]

        for category, item_list in [
            ("Tables", contents["referenced_tables"] + contents["included_tables"]),
            ("Models", contents["models"] + contents["pytorch_models"]),
            ("Numpy Arrays", contents["numpy_arrays"]),
            ("JSON Data", contents["json_data"]),
            ("Artifacts", contents["artifacts"]),
        ]:
            if item_list:
                lines.append(f"  {category}: {', '.join(sorted(item_list))}")

        return "\n".join(lines)
