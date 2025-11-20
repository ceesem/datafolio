"""Handler for PyTorch models."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from datafolio.base.handler import BaseHandler

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class PyTorchHandler(BaseHandler):
    """Handler for PyTorch models stored in bundle.

    This handler manages PyTorch model storage:
    - Serializes models using state_dict to .pt format
    - Stores metadata (model type, PyTorch version)
    - Handles lineage tracking
    - Deserializes back to model on read

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = PyTorchHandler()
        >>> register_handler(handler)
        >>>
        >>> # Handler is used automatically by DataFolio
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 2)
        >>> folio.add_pytorch('model', model)
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "pytorch_model"

    def can_handle(self, data: Any) -> bool:
        """Check if data is a PyTorch model.

        Args:
            data: Data to check

        Returns:
            True if data is a torch.nn.Module
        """
        try:
            import torch.nn as nn

            return isinstance(data, nn.Module)
        except ImportError:
            return False

    def add(
        self,
        folio: "DataFolio",
        name: str,
        model: Any,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add PyTorch model to folio.

        Writes the model to storage and builds complete metadata including:
        - Basic info: filename, model type
        - Model info: PyTorch version
        - Lineage: inputs and description
        - Timestamps: creation time

        Args:
            folio: DataFolio instance
            name: Item name
            model: PyTorch model (nn.Module) to store
            description: Optional description
            inputs: Optional lineage inputs
            **kwargs: Additional arguments (currently unused)

        Returns:
            Complete metadata dict for this model

        Raises:
            TypeError: If model is not a torch.nn.Module
            ImportError: If PyTorch is not installed
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required to save PyTorch models. "
                "Install with: pip install torch"
            )

        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model).__name__}")

        # Build filename
        filename = f"{name}.pt"
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        # Write model to storage
        folio._storage.write_pytorch(filepath, model)

        # Build comprehensive metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "model_type": type(model).__name__,
            "torch_version": torch.__version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    def get(
        self, folio: "DataFolio", name: str, model_class: Optional[Any] = None, **kwargs
    ) -> Any:
        """Load PyTorch model from folio.

        Args:
            folio: DataFolio instance
            name: Item name
            model_class: Model class to instantiate (required for PyTorch)
            **kwargs: Additional arguments passed to model_class constructor

        Returns:
            Deserialized PyTorch model with loaded state_dict

        Raises:
            KeyError: If item doesn't exist
            ValueError: If model_class is not provided
            ImportError: If PyTorch is not installed
        """
        if model_class is None:
            raise ValueError(
                "model_class is required to load PyTorch models. "
                "Example: folio.get_pytorch('model', model_class=MyModel)"
            )

        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )
        return folio._storage.read_pytorch(filepath, model_class, **kwargs)
