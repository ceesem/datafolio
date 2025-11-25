"""Table handlers for pandas DataFrames and external references."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from datafolio.base.handler import BaseHandler
from datafolio.utils import get_file_extension

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class PandasHandler(BaseHandler):
    """Handler for pandas DataFrames (included in bundle).

    This handler manages the full lifecycle of DataFrame storage:
    - Serializes DataFrames to Parquet format
    - Stores metadata (shape, columns, dtypes)
    - Handles lineage tracking
    - Deserializes back to DataFrame on read

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = PandasHandler()
        >>> register_handler(handler)
        >>>
        >>> # Handler is used automatically by DataFolio
        >>> folio.add_table('data', df)  # Uses PandasHandler internally
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "included_table"

    def can_handle(self, data: Any) -> bool:
        """Check if data is a pandas DataFrame.

        Args:
            data: Data to check

        Returns:
            True if data is a pandas DataFrame
        """
        try:
            import pandas as pd

            return isinstance(data, pd.DataFrame)
        except ImportError:
            return False

    def add(
        self,
        folio: "DataFolio",
        name: str,
        data: Any,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        table_format: str = "parquet",
        **kwargs,
    ) -> Dict[str, Any]:
        """Add DataFrame to folio.

        Writes the DataFrame to storage and builds complete metadata including:
        - Basic info: filename, row/column counts
        - Schema: column names and dtypes
        - Lineage: inputs and description
        - Timestamps: creation time

        Args:
            folio: DataFolio instance
            name: Item name
            data: pandas DataFrame to store
            description: Optional description
            inputs: Optional lineage inputs
            table_format: Format for storage (default: 'parquet')
            **kwargs: Additional arguments (currently unused)

        Returns:
            Complete metadata dict for this table

        Raises:
            TypeError: If data is not a pandas DataFrame
        """
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data).__name__}")

        # Build filename
        extension = get_file_extension(table_format)
        filename = f"{name}{extension}"
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        # Write data to storage
        folio._storage.write_parquet(filepath, data)

        # Calculate checksum
        checksum = folio._storage.calculate_checksum(filepath)

        # Build comprehensive metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "table_format": table_format,
            "is_directory": False,
            "checksum": checksum,
            "num_rows": len(data),
            "num_cols": len(data.columns),
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    def get(self, folio: "DataFolio", name: str, **kwargs) -> Any:
        """Load DataFrame from folio.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Additional arguments passed to read_parquet

        Returns:
            pandas DataFrame

        Raises:
            KeyError: If item doesn't exist
        """
        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )
        return folio._storage.read_parquet(filepath, **kwargs)


class PolarsHandler(BaseHandler):
    """Handler for Polars DataFrames (included in bundle).

    This handler manages the full lifecycle of Polars DataFrame storage:
    - Serializes DataFrames to Parquet format (compatible with pandas)
    - Stores metadata (shape, columns, dtypes)
    - Handles lineage tracking
    - Deserializes back to Polars DataFrame on read
    - Supports conversion between Polars and pandas formats

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = PolarsHandler()
        >>> register_handler(handler)
        >>>
        >>> # Handler is used automatically by DataFolio
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> folio.add_table('data', df)  # Uses PolarsHandler internally
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "polars_table"

    def can_handle(self, data: Any) -> bool:
        """Check if data is a Polars DataFrame.

        Args:
            data: Data to check

        Returns:
            True if data is a Polars DataFrame
        """
        try:
            import polars as pl

            return isinstance(data, pl.DataFrame)
        except ImportError:
            return False

    def add(
        self,
        folio: "DataFolio",
        name: str,
        data: Any,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        table_format: str = "parquet",
        **kwargs,
    ) -> Dict[str, Any]:
        """Add Polars DataFrame to folio.

        Writes the DataFrame to storage and builds complete metadata including:
        - Basic info: filename, row/column counts
        - Schema: column names and dtypes
        - Lineage: inputs and description
        - Timestamps: creation time
        - Library: 'polars' marker for proper deserialization

        Args:
            folio: DataFolio instance
            name: Item name
            data: Polars DataFrame to store
            description: Optional description
            inputs: Optional lineage inputs
            table_format: Format for storage (default: 'parquet')
            **kwargs: Additional arguments (currently unused)

        Returns:
            Complete metadata dict for this table

        Raises:
            TypeError: If data is not a Polars DataFrame
        """
        import polars as pl

        if not isinstance(data, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, got {type(data).__name__}")

        # Build filename
        extension = get_file_extension(table_format)
        filename = f"{name}{extension}"
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        # Write data to storage (convert to pandas for compatibility with storage backend)
        pandas_df = data.to_pandas()
        folio._storage.write_parquet(filepath, pandas_df)

        # Calculate checksum
        checksum = folio._storage.calculate_checksum(filepath)

        # Build comprehensive metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "table_format": table_format,
            "dataframe_library": "polars",  # Mark as polars for proper deserialization
            "is_directory": False,
            "checksum": checksum,
            "num_rows": len(data),
            "num_cols": len(data.columns),
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.schema.items()},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    def get(self, folio: "DataFolio", name: str, **kwargs) -> Any:
        """Load Polars DataFrame from folio.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Additional arguments passed to read_parquet

        Returns:
            Polars DataFrame

        Raises:
            KeyError: If item doesn't exist
        """
        import polars as pl

        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )

        # Read as pandas first (using storage backend), then convert to polars
        pandas_df = folio._storage.read_parquet(filepath, **kwargs)
        return pl.from_pandas(pandas_df)


class ReferenceTableHandler(BaseHandler):
    """Handler for external table references (not stored in bundle).

    This handler manages references to external data:
    - Stores only metadata (path, format)
    - Does not copy data into bundle
    - Reads from external location on access
    - Supports cloud and local paths

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = ReferenceTableHandler()
        >>> register_handler(handler)
        >>>
        >>> # Handler is used automatically by DataFolio
        >>> folio.reference_table('external', 's3://bucket/data.parquet')
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "referenced_table"

    def can_handle(self, data: Any) -> bool:
        """Cannot auto-detect references (must use explicit method).

        Returns:
            Always False - references must be explicit via reference_table()
        """
        return False

    def add(
        self,
        folio: "DataFolio",
        name: str,
        reference: str,  # Note: different signature - takes reference path
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        table_format: str = "parquet",
        **kwargs,
    ) -> Dict[str, Any]:
        """Add reference to external table.

        Args:
            folio: DataFolio instance
            name: Item name
            reference: Path to external table (local or cloud)
            description: Optional description
            inputs: Optional lineage inputs
            table_format: Format of the table (default: 'parquet')
            **kwargs: Additional arguments

        Returns:
            Metadata dict for this reference
        """
        from datafolio.utils import is_cloud_path, resolve_path

        # Resolve path (handles local/cloud)
        resolved_path = resolve_path(reference)

        # Check if directory
        is_directory = False

        # Handle local paths (including file://)
        check_path = resolved_path
        if check_path.startswith("file://"):
            check_path = check_path[7:]

        if not is_cloud_path(check_path):
            import os

            is_directory = os.path.isdir(check_path)
        else:
            # For cloud paths, we can't easily check isdir without network calls
            # Heuristic: if it ends with '/', treat as directory
            # Or if table_format implies directory (like 'delta')
            if resolved_path.endswith("/") or table_format in ("delta", "iceberg"):
                is_directory = True

        # Build metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "path": resolved_path,
            "table_format": table_format,
            "is_directory": is_directory,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Optionally read metadata (can be slow for large files)
        if kwargs.get("read_metadata", False) and not is_directory:
            try:
                df = folio._storage.read_table(resolved_path, table_format)
                metadata["num_rows"] = len(df)
            except Exception:
                # Don't fail add() if we can't read the remote file
                pass

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    def get(self, folio: "DataFolio", name: str, **kwargs) -> Any:
        """Load DataFrame from external reference.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Additional arguments passed to reader

        Returns:
            pandas DataFrame loaded from external location
        """
        item = folio._items[name]
        return folio._storage.read_table(
            item["path"], item.get("table_format", "parquet"), **kwargs
        )

    def delete(self, folio: "DataFolio", name: str) -> None:
        """Delete reference metadata (not the external file).

        Override default delete behavior - we only remove metadata,
        not the actual external file.
        """
        # References have no local files to delete
        pass
