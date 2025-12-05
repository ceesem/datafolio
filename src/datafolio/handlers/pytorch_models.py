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
        init_args: Optional[Dict[str, Any]] = None,
        save_class: bool = False,
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
            init_args: Optional dict of arguments needed to reconstruct the model
            save_class: If True, use dill to serialize the model class for reconstruction
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

        # Write model to storage (pass init_args and save_class to storage backend)
        folio._storage.write_pytorch(
            filepath, model, init_args=init_args, save_class=save_class
        )

        # Calculate checksum
        checksum = folio._storage.calculate_checksum(filepath)

        # Build comprehensive metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "checksum": checksum,
            "model_type": type(model).__name__,
            "torch_version": torch.__version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs
        if init_args:
            metadata["init_args"] = init_args

        return metadata

    def get(
        self,
        folio: "DataFolio",
        name: str,
        model_class: Optional[Any] = None,
        reconstruct: bool = True,
        **kwargs,
    ) -> Any:
        """Load PyTorch model from folio.

        Args:
            folio: DataFolio instance
            name: Item name
            model_class: Model class to instantiate (optional if reconstruct=False or model saved with save_class=True)
            reconstruct: If True, reconstruct the model; if False, return state_dict only
            **kwargs: Additional arguments passed to model_class constructor

        Returns:
            If reconstruct=True: Deserialized PyTorch model with loaded state_dict
            If reconstruct=False: Just the state_dict dictionary

        Raises:
            KeyError: If item doesn't exist
            ValueError: If model_class is not provided and reconstruction requires it
            ImportError: If PyTorch is not installed
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required to load PyTorch models. "
                "Install with: pip install torch"
            )

        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )

        # Use cache if available
        filepath = folio._get_file_path_with_cache(name, filepath)

        bundle = folio._storage.read_pytorch(filepath)

        # If not reconstructing, return just the state dict
        if not reconstruct:
            return bundle["state_dict"]

        # Try to reconstruct the model
        # First, check if we have a serialized class (from save_class=True)
        if "serialized_class" in bundle:
            try:
                import dill

                model_cls = dill.loads(bundle["serialized_class"])
                if "init_args" in bundle["metadata"]:
                    model = model_cls(**bundle["metadata"]["init_args"])
                else:
                    model = model_cls(**kwargs)
                model.load_state_dict(bundle["state_dict"])
                return model
            except ImportError:
                pass  # Fall through to other methods

        # If model_class provided, use it
        if model_class is not None:
            # Use init_args from metadata if available, otherwise use kwargs
            if "init_args" in bundle.get("metadata", {}):
                model = model_class(**bundle["metadata"]["init_args"])
            else:
                model = model_class(**kwargs)
            model.load_state_dict(bundle["state_dict"])
            return model

        # Try auto-reconstruction from metadata
        if "metadata" in bundle:
            metadata = bundle["metadata"]
            if "model_module" in metadata and "model_class" in metadata:
                try:
                    # Try to import the class
                    import importlib

                    module = importlib.import_module(metadata["model_module"])
                    model_cls = getattr(module, metadata["model_class"])
                    if "init_args" in metadata:
                        model = model_cls(**metadata["init_args"])
                    else:
                        model = model_cls(**kwargs)
                    model.load_state_dict(bundle["state_dict"])
                    return model
                except (ImportError, AttributeError) as e:
                    # Convert to RuntimeError for compatibility with existing tests
                    raise RuntimeError(
                        f"Failed to auto-reconstruct model: {str(e)}"
                    ) from e

        # If all else fails, raise an error
        raise ValueError(
            "model_class is required to load PyTorch models. "
            "Example: folio.get_pytorch('model', model_class=MyModel)"
        )
