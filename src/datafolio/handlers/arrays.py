"""Handler for numpy arrays."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from datafolio.base.handler import BaseHandler

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class NumpyHandler(BaseHandler):
    """Handler for numpy arrays stored in bundle.

    This handler manages the full lifecycle of numpy array storage:
    - Serializes arrays to .npy format
    - Stores metadata (shape, dtype)
    - Handles lineage tracking
    - Deserializes back to array on read

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = NumpyHandler()
        >>> register_handler(handler)
        >>>
        >>> # Handler is used automatically by DataFolio
        >>> import numpy as np
        >>> folio.add_numpy('embeddings', np.random.randn(100, 128))
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "numpy_array"

    def can_handle(self, data: Any) -> bool:
        """Check if data is a numpy array.

        Args:
            data: Data to check

        Returns:
            True if data is a numpy ndarray
        """
        try:
            import numpy as np

            return isinstance(data, np.ndarray)
        except ImportError:
            return False

    def add(
        self,
        folio: "DataFolio",
        name: str,
        data: Any,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add numpy array to folio.

        Writes the array to storage and builds complete metadata including:
        - Basic info: filename, shape, dtype
        - Lineage: inputs and description
        - Timestamps: creation time

        Args:
            folio: DataFolio instance
            name: Item name
            data: numpy array to store
            description: Optional description
            inputs: Optional lineage inputs
            **kwargs: Additional arguments (currently unused)

        Returns:
            Complete metadata dict for this array

        Raises:
            TypeError: If data is not a numpy array
            ImportError: If numpy is not installed
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "NumPy is required to save numpy arrays. "
                "Install with: pip install numpy"
            )

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data).__name__}")

        # Build filename
        filename = f"{name}.npy"
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        # Write data to storage
        folio._storage.write_numpy(filepath, data)

        # Build comprehensive metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    def get(self, folio: "DataFolio", name: str, **kwargs) -> Any:
        """Load numpy array from folio.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Additional arguments (currently unused)

        Returns:
            numpy array

        Raises:
            KeyError: If item doesn't exist
            ImportError: If numpy is not installed
        """
        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )
        return folio._storage.read_numpy(filepath)
