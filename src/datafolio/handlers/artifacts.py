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
        >>> # Files enter the folio via the explicit add_file() verb
        >>> folio.add_file('/path/to/config.yaml', name='config')
        >>> path = folio.get('config')
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "artifact"

    def can_handle(self, data: Any) -> bool:
        """Artifacts are never auto-detected.

        Whether a string is "a file" depends on the current working
        directory's contents, which made ``add()``'s behavior for strings
        nondeterministic. Files enter the folio only through the explicit
        ``add_file()`` verb; strings passed to ``add()`` are always stored
        as JSON data.

        Args:
            data: Data to check

        Returns:
            Always False.
        """
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
        # (folio injects a collision-safe versioned name).
        extension = source_path.suffix
        filename = kwargs.get("_filename") or f"{name}{extension}"
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
            metadata["inputs"] = list(inputs)

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
