"""Handler for JSON-serializable data."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from datafolio.base.handler import BaseHandler

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class JsonHandler(BaseHandler):
    """Handler for JSON-serializable data stored in bundle.

    This handler manages JSON data (dicts, lists, scalars):
    - Serializes data to .json format
    - Stores metadata (data type)
    - Handles lineage tracking
    - Deserializes back to Python objects on read

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = JsonHandler()
        >>> register_handler(handler)
        >>>
        >>> # Handler is used automatically by DataFolio
        >>> config = {'learning_rate': 0.01, 'batch_size': 32}
        >>> folio.add_json('config', config)
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "json_data"

    def can_handle(self, data: Any) -> bool:
        """Check if data is JSON-serializable.

        Only handles dict, list, and basic JSON types (int, float, str, bool, None).
        Does NOT handle complex objects, DataFrames, arrays, etc.

        Args:
            data: Data to check

        Returns:
            True if data is a basic JSON-serializable type
        """
        # Only handle dict and list explicitly
        # Exclude primitives (int, float, str, bool, None) to avoid conflicts with other handlers
        # These can still be stored via add_json() explicitly
        if isinstance(data, (dict, list)):
            try:
                import orjson

                orjson.dumps(data)
                return True
            except (TypeError, ValueError, ImportError):
                return False
        return False

    def add(
        self,
        folio: "DataFolio",
        name: str,
        data: Union[dict, list, int, float, str, bool, None],
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add JSON data to folio.

        Writes the data to storage and builds complete metadata including:
        - Basic info: filename, data type
        - Lineage: inputs and description
        - Timestamps: creation time

        Args:
            folio: DataFolio instance
            name: Item name
            data: JSON-serializable data to store
            description: Optional description
            inputs: Optional lineage inputs
            **kwargs: Additional arguments (currently unused)

        Returns:
            Complete metadata dict for this JSON data

        Raises:
            TypeError: If data is not JSON-serializable
        """
        import orjson

        # Validate JSON-serializability
        try:
            orjson.dumps(data)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Data is not JSON-serializable: {e}")

        # Build filename
        filename = f"{name}.json"
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        # Write data to storage
        folio._storage.write_json(filepath, data)

        # Calculate checksum
        checksum = folio._storage.calculate_checksum(filepath)

        # Build comprehensive metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "checksum": checksum,
            "data_type": type(data).__name__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    def get(self, folio: "DataFolio", name: str, **kwargs) -> Any:
        """Load JSON data from folio.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Additional arguments (currently unused)

        Returns:
            Deserialized JSON data (dict, list, scalar, etc.)

        Raises:
            KeyError: If item doesn't exist
        """
        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )
        return folio._storage.read_json(filepath)
