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
        size_bytes = folio._storage.file_size(filepath)

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
        if size_bytes is not None:
            metadata["size_bytes"] = size_bytes

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

    def get_lazy(self, folio: "DataFolio", name: str, **kwargs) -> Any:
        """Lazily scan the bundled table as a polars LazyFrame.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Additional arguments passed to the polars scanner

        Returns:
            polars LazyFrame backed by the bundle's parquet file
        """
        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )

        return folio._storage.scan_table(
            filepath, item.get("table_format", "parquet"), **kwargs
        )


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
        infer_schema: bool = True,
        allow_full_load: bool = False,
        polars_only: Optional[bool] = None,
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
            infer_schema: If True and polars is available, cheaply read the
                parquet footer to populate ``columns``/``dtypes``/``num_rows``
                so a reference carries the same schema metadata as an included
                table. Best-effort: silently skipped on any failure.
            allow_full_load: If True, this reference bypasses the folio's
                ``max_eager_bytes`` guard on eager ``get_table`` reads.
            polars_only: If True, the reference can only be read lazily/via
                polars (``get_lazy`` / ``frame='polars'``); eager pandas reads
                raise a clear error. If None (default), this is inferred: a
                sharded/partitioned directory dataset is polars-only, since
                pandas mishandles such layouts.
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

        if allow_full_load:
            metadata["allow_full_load"] = True

        # Sharded/partitioned directory datasets are polars-only by default:
        # pandas mishandles hive-partitioned/typed layouts (cryptic errors), so
        # eager pandas reads are refused in favor of get_lazy / frame='polars'.
        is_polars_only = is_directory if polars_only is None else polars_only
        if is_polars_only:
            metadata["polars_only"] = True

        # Best-effort size (drives the eager-load guard). Unknown -> unset.
        size_bytes = folio._storage.file_size(resolved_path)
        if size_bytes is not None:
            metadata["size_bytes"] = size_bytes

        # Cheap schema inference via a polars scan (footer-only for a single
        # file; reads shard metadata for a directory dataset), so references
        # carry the same columns/dtypes/num_rows an included table would.
        # Works for both single parquet files and sharded/hive directories.
        if infer_schema and table_format == "parquet":
            self._infer_schema(folio, resolved_path, metadata)

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    @staticmethod
    def _infer_schema(folio: "DataFolio", path: str, metadata: Dict[str, Any]) -> None:
        """Populate columns/dtypes/num_rows from a parquet footer (best-effort).

        Uses a polars lazy scan so only the file footer is read for the schema
        (and, where polars can optimize it, for the row count). Any failure
        (polars missing, unreachable remote, bad file) is swallowed so that
        creating a reference never fails on metadata inference.
        """
        try:
            from datafolio.readers import scan_parquet

            lf = scan_parquet(path, use_https=folio._storage._use_https)
            schema = lf.collect_schema()
            metadata["columns"] = list(schema.names())
            metadata["dtypes"] = {n: str(t) for n, t in schema.items()}
            metadata["num_cols"] = len(schema)
            try:
                import polars as pl

                metadata["num_rows"] = int(lf.select(pl.len()).collect().item())
            except Exception:
                pass
        except Exception:
            # Never fail reference creation on schema inference.
            pass

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

    def get_lazy(self, folio: "DataFolio", name: str, **kwargs) -> Any:
        """Lazily scan the external table as a polars LazyFrame.

        This is the marquee case for references: predicate/projection pushdown
        over a large external parquet (e.g. ``s3://``/``gs://``) without copying
        it into the bundle or pulling it fully into memory.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Additional arguments passed to the polars scanner

        Returns:
            polars LazyFrame backed by the external path
        """
        item = folio._items[name]
        return folio._storage.scan_table(
            item["path"], item.get("table_format", "parquet"), **kwargs
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
