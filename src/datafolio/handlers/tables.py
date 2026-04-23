"""Table handlers for DataFrames and external references."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from datafolio.base.handler import BaseHandler
from datafolio.utils import get_file_extension

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class DataframeHandler(BaseHandler):
    """Handler for tabular DataFrames (included in bundle).

    Accepts pandas and Polars DataFrames. Each library has its own private
    conversion path to PyArrow, which is then written directly to Parquet
    without any intermediate type degradation.

    Adding support for a new DataFrame library means implementing a
    ``_<lib>_to_arrow`` classmethod and adding a branch in ``_to_arrow``.

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = DataframeHandler()
        >>> register_handler(handler)
        >>>
        >>> folio.add_table('data', polars_df)   # Polars DataFrame
        >>> folio.add_table('data', pandas_df)   # pandas DataFrame
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "included_table"

    # ── type detection ────────────────────────────────────────────────────────

    @staticmethod
    def _is_pandas(data: Any) -> bool:
        try:
            import pandas as pd

            return isinstance(data, pd.DataFrame)
        except ImportError:
            return False

    @staticmethod
    def _is_polars(data: Any) -> bool:
        try:
            import polars as pl

            return isinstance(data, pl.DataFrame)
        except ImportError:
            return False

    def can_handle(self, data: Any) -> bool:
        """Return True for pandas or Polars DataFrames."""
        return self._is_pandas(data) or self._is_polars(data)

    # ── Arrow conversion ──────────────────────────────────────────────────────

    @staticmethod
    def _pandas_to_arrow(data: Any) -> Any:
        import pyarrow as pa

        return pa.Table.from_pandas(data, preserve_index=False)

    @staticmethod
    def _polars_to_arrow(data: Any) -> Any:
        return data.to_arrow()

    @classmethod
    def _to_arrow(cls, data: Any) -> Any:
        """Convert a supported DataFrame to a PyArrow Table.

        Each library has its own conversion path to preserve exact column types.

        Raises:
            TypeError: If data is not a supported DataFrame type.
        """
        if cls._is_polars(data):
            return cls._polars_to_arrow(data)
        if cls._is_pandas(data):
            return cls._pandas_to_arrow(data)
        raise TypeError(
            f"Expected a pandas or Polars DataFrame, got {type(data).__name__}"
        )

    # ── storage ───────────────────────────────────────────────────────────────

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
        """Add a DataFrame to the folio.

        Converts the DataFrame to a PyArrow Table and writes it to Parquet,
        preserving exact column types (Int64, struct fields, etc.).

        Args:
            folio: DataFolio instance.
            name: Item name.
            data: pandas or Polars DataFrame to store.
            description: Optional description.
            inputs: Optional lineage inputs.
            table_format: Storage format (default: ``'parquet'``).

        Returns:
            Metadata dict for this table.

        Raises:
            TypeError: If data is not a supported DataFrame type.
        """
        arrow_table = self._to_arrow(data)

        extension = get_file_extension(table_format)
        filename = f"{name}{extension}"
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        # Pass the original data so StorageBackend can use the native writer
        # (e.g. Polars' own writer for Polars DataFrames, which produces parquet
        # statistics that Polars' predicate-pushdown engine can parse correctly).
        folio._storage.write_parquet(filepath, data)

        checksum = folio._storage.calculate_checksum(filepath)

        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "table_format": table_format,
            "is_directory": False,
            "checksum": checksum,
            "num_rows": arrow_table.num_rows,
            "num_cols": arrow_table.num_columns,
            "columns": arrow_table.schema.names,
            "dtypes": {field.name: str(field.type) for field in arrow_table.schema},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

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
        remote_path = item["path"]

        return folio._storage.read_table(
            remote_path, item.get("table_format", "parquet"), **kwargs
        )

    def delete(self, folio: "DataFolio", name: str) -> None:
        """Delete reference metadata (not the external file).

        Override default delete behavior - we only remove metadata,
        not the actual external file.
        """
        # References have no local files to delete
        pass


# Backward-compatible alias
PandasHandler = DataframeHandler
