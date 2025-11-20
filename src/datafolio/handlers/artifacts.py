"""Handler for arbitrary file artifacts."""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from datafolio.base.handler import BaseHandler

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class ArtifactHandler(BaseHandler):
    """Handler for arbitrary file artifacts stored in bundle.

    This handler manages file artifacts:
    - Copies files into bundle's artifacts/ directory
    - Stores metadata (original path, file size)
    - Handles lineage tracking
    - Returns paths to artifacts on read

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = ArtifactHandler()
        >>> register_handler(handler)
        >>>
        >>> # Handler is used automatically by DataFolio
        >>> folio.add_artifact('config', '/path/to/config.yaml')
        >>> path = folio.get_artifact_path('config')
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "artifact"

    def can_handle(self, data: Any) -> bool:
        """Check if data is a file path string.

        Args:
            data: Data to check

        Returns:
            True if data is a string representing an existing file
        """
        if isinstance(data, (str, Path)):
            path = Path(data)
            return path.exists() and path.is_file()
        return False

    def add(
        self,
        folio: "DataFolio",
        name: str,
        filepath: str,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add file artifact to folio.

        Copies the file to storage and builds complete metadata including:
        - Basic info: filename, original path, file size
        - Lineage: inputs and description
        - Timestamps: creation time

        Args:
            folio: DataFolio instance
            name: Item name
            filepath: Path to file to copy into bundle
            description: Optional description
            inputs: Optional lineage inputs
            **kwargs: Additional arguments (currently unused)

        Returns:
            Complete metadata dict for this artifact

        Raises:
            FileNotFoundError: If filepath doesn't exist
            IsADirectoryError: If filepath is a directory
        """
        from pathlib import Path

        source_path = Path(filepath)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        if source_path.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {filepath}")

        # Build filename - preserve extension from original file
        extension = source_path.suffix
        filename = f"{name}{extension}"
        subdir = self.get_storage_subdir()
        dest_path = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        # Copy file to storage
        folio._storage.copy_file(str(source_path), dest_path)

        # Calculate checksum
        checksum = folio._storage.calculate_checksum(dest_path)

        # Get file size
        file_size = source_path.stat().st_size

        # Build comprehensive metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "checksum": checksum,
            "original_path": str(source_path),
            "file_size": file_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    def get(self, folio: "DataFolio", name: str, **kwargs) -> str:
        """Get path to artifact file.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Additional arguments (currently unused)

        Returns:
            Absolute path to artifact file in bundle

        Raises:
            KeyError: If item doesn't exist
        """
        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )
        return filepath
