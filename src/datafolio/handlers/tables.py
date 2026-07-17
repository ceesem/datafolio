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
        >>> folio.add('data', polars_df)   # Polars DataFrame
        >>> folio.add('data', pandas_df)   # pandas DataFrame
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

    @staticmethod
    def _is_polars_lazy(data: Any) -> bool:
        try:
            import polars as pl

            return isinstance(data, pl.LazyFrame)
        except ImportError:
            return False

    def can_handle(self, data: Any) -> bool:
        """Return True for pandas or Polars DataFrames (eager or lazy)."""
        return (
            self._is_pandas(data) or self._is_polars(data) or self._is_polars_lazy(data)
        )

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
        """Add a DataFrame (eager) or LazyFrame (streamed) to the folio.

        For pandas/Polars DataFrames the frame is converted to a PyArrow Table
        and written to Parquet, preserving exact column types (Int64, struct
        fields, etc.). For a Polars ``LazyFrame`` the query is materialized
        with a streaming ``sink_parquet`` (bounded memory — the full result is
        never held at once); schema and row count are then read back from the
        written Parquet footer (cheap) rather than from an in-memory result.

        Args:
            folio: DataFolio instance.
            name: Item name.
            data: pandas DataFrame, Polars DataFrame, or Polars LazyFrame.
            description: Optional description.
            inputs: Optional lineage inputs.
            table_format: Storage format (default: ``'parquet'``).

        Returns:
            Metadata dict for this table.

        Raises:
            TypeError: If data is not a supported frame type.
        """
        extension = get_file_extension(table_format)
        # The folio allocates a collision-safe versioned filename and injects it
        # so a new payload never overwrites bytes a committed manifest (or a
        # snapshot) still references. Direct handler calls fall back to a stable
        # name.
        filename = kwargs.get("_filename") or f"{name}{extension}"
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        index_columns = []
        if self._is_polars_lazy(data):
            # Streaming, bounded-memory materialization. The footer is read from
            # the local staging file, so a cloud write never re-downloads the
            # just-uploaded object just to learn its schema/row count.
            arrow_schema, num_rows = folio._storage.sink_parquet_with_footer(
                filepath, data
            )
        else:
            # Eager frame -> Arrow -> Parquet. Pass the original data so the
            # backend can use the native writer (Polars' own writer emits
            # parquet statistics its predicate-pushdown engine can parse).
            preserve_index = bool(kwargs.get("preserve_index", False))
            if self._is_pandas(data):
                import pandas as pd

                default_index = isinstance(
                    data.index, pd.RangeIndex
                ) and data.index.equals(pd.RangeIndex(len(data)))
                if preserve_index and not default_index:
                    # Store the index as ordinary columns (any tool can read
                    # them) and record which they were so the pandas read
                    # path can set_index() them back.
                    original = set(data.columns)
                    data = data.reset_index()
                    index_columns = [c for c in data.columns if c not in original]
                elif not default_index:
                    # A non-default index is silently dropped by the parquet
                    # write; make that visible and offer the escape hatch.
                    import warnings

                    warnings.warn(
                        f"Table '{name}' has a non-default pandas index that "
                        f"will NOT be stored (parquet keeps columns only). "
                        f"Call reset_index() first to keep it as a column, or "
                        f"pass preserve_index=True to store it.",
                        UserWarning,
                        stacklevel=4,
                    )
            arrow_table = self._to_arrow(data)
            folio._storage.write_parquet(filepath, data)
            arrow_schema = arrow_table.schema
            num_rows = arrow_table.num_rows

        checksum = folio._storage.calculate_checksum(filepath)
        size_bytes = folio._storage.file_size(filepath)

        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "table_format": table_format,
            "is_directory": False,
            "checksum": checksum,
            "num_rows": num_rows,
            "num_cols": len(arrow_schema),
            "columns": list(arrow_schema.names),
            "dtypes": {field.name: str(field.type) for field in arrow_schema},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if size_bytes is not None:
            metadata["size_bytes"] = size_bytes
        if index_columns:
            metadata["index_columns"] = index_columns

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

        df = folio._storage.read_parquet(filepath, **kwargs)
        # Restore an index stored via add(..., preserve_index=True). The
        # parquet file itself keeps these as plain columns (readable by any
        # tool); only the pandas read path re-applies them as the index.
        index_columns = item.get("index_columns")
        if index_columns and all(c in df.columns for c in index_columns):
            df = df.set_index(
                index_columns if len(index_columns) > 1 else index_columns[0]
            )
            # An unnamed single index round-trips through reset_index() as a
            # column literally named 'index'; restore it to unnamed.
            if index_columns == ["index"]:
                df.index.name = None
        return df

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
        allow_full_load: bool = False,
        polars_only: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add reference to external table.

        This is a cheap, purely-local manifest operation: it performs **no
        remote I/O**. It does not stat the object, read its schema, count its
        rows, or check that it exists. Use :meth:`DataFolio.inspect_table` to
        enrich the manifest with schema/size/identity, and
        :meth:`DataFolio.validate` to check existence.

        Args:
            folio: DataFolio instance
            name: Item name
            reference: Path to external table (local or cloud)
            description: Optional description
            inputs: Optional lineage inputs
            table_format: Format of the table (default: 'parquet')
            allow_full_load: If True, this reference bypasses the folio's
                ``max_eager_bytes`` guard on eager ``get_table`` reads.
            polars_only: If True, the reference can only be read lazily/via
                polars (``scan_table`` / ``frame='polars'``); eager pandas reads
                raise a clear error. If None (default), this is inferred: a
                sharded/partitioned directory dataset is polars-only, since
                pandas mishandles such layouts.
            **kwargs: Additional arguments

        Returns:
            Metadata dict for this reference
        """
        import os

        from datafolio.utils import is_cloud_path, resolve_path

        # Reference path policy: external references are absolute and static, so
        # that moving/copying a folio preserves the recorded reference exactly
        # (it is never rebased or reinterpreted). Cloud URIs keep their full URI;
        # an absolute local path is normalized to a ``file://`` URI. A *new*
        # relative path is rejected with an actionable error (legacy relative
        # references already in a manifest are still read — resolved against the
        # bundle — for backward compatibility; see _resolve_reference_path). No
        # remote I/O either way.
        ref_str = str(reference)
        if (
            is_cloud_path(ref_str)
            or ref_str.startswith("file://")
            or os.path.isabs(ref_str)
        ):
            stored_path = resolve_path(ref_str)
        else:
            raise ValueError(
                f"Relative reference path {ref_str!r} is not allowed. External "
                f"references must be absolute and static so moving or copying the "
                f"folio preserves the link exactly. Pass an absolute path or a "
                f"file:// URI (e.g. {os.path.abspath(ref_str)!r}) or a cloud URI "
                f"(s3://, gs://, ...)."
            )

        # Layout guess against the effective location — local uses a cheap local
        # stat; cloud is a name-only heuristic. Neither performs remote I/O.
        effective = folio._resolve_reference_path(stored_path)
        check_path = effective[7:] if effective.startswith("file://") else effective

        is_directory = False
        if not is_cloud_path(check_path):
            is_directory = os.path.isdir(check_path)
        elif check_path.endswith("/"):
            # trailing '/' implies a directory (no network call)
            is_directory = True

        # Build metadata (manifest-only; enrichment happens in inspect_table).
        # References are inherently mutable: datafolio links to external data it
        # does not own or copy, so the content at this path can change over time
        # (snapshots preserve the link, not the bytes). inspect_table() records
        # a point-in-time source_identity to make such drift detectable.
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "path": stored_path,
            "table_format": table_format,
            "is_directory": is_directory,
            "mutable": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if allow_full_load:
            metadata["allow_full_load"] = True

        # Sharded/partitioned directory datasets are polars-only by default:
        # pandas mishandles hive-partitioned/typed layouts (cryptic errors), so
        # eager pandas reads are refused in favor of scan_table / frame='polars'.
        is_polars_only = is_directory if polars_only is None else polars_only
        if is_polars_only:
            metadata["polars_only"] = True

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    def inspect(self, folio: "DataFolio", name: str) -> Dict[str, Any]:
        """Read the external object and return enrichment metadata.

        Unlike :meth:`add`, this performs remote I/O. It verifies the object
        exists, records its size, and (for parquet) reads the schema and row
        count via a polars scan. Failures raise actionable errors rather than
        being silently swallowed.

        Args:
            folio: DataFolio instance
            name: Item name

        Returns:
            Dict of fields to merge into the manifest entry (``size_bytes``,
            ``columns``, ``dtypes``, ``num_cols``, ``num_rows``, refreshed
            ``is_directory``).

        Raises:
            FileNotFoundError: If the referenced object does not exist.
            RuntimeError: If the object exists but its schema cannot be read.
        """
        item = folio._items[name]
        path = folio._resolve_reference_path(item["path"])
        table_format = item.get("table_format", "parquet")

        if not folio._storage.exists(path):
            raise FileNotFoundError(f"Referenced table '{name}' not found at {path}")

        updated: Dict[str, Any] = {}

        size_bytes = folio._storage.file_size(path)
        if size_bytes is not None:
            updated["size_bytes"] = size_bytes

        if table_format == "parquet":
            try:
                from datafolio.readers import scan_parquet

                lf = scan_parquet(path, use_https=folio._storage._use_https)
                # Normalize to an Arrow-derived logical schema (same convention
                # as included tables): collect zero rows to get the Arrow schema
                # with column order and types, without reading data.
                arrow_schema = lf.limit(0).collect().to_arrow().schema
                updated["columns"] = list(arrow_schema.names)
                updated["dtypes"] = {
                    field.name: str(field.type) for field in arrow_schema
                }
                updated["num_cols"] = len(arrow_schema)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read schema for reference '{name}' at {path}: {exc}"
                ) from exc
            # Row count is best-effort (may require reading shard metadata).
            try:
                import polars as pl

                updated["num_rows"] = int(lf.select(pl.len()).collect().item())
            except Exception:
                pass

        # Point-in-time source identity so a mutable reference's drift is
        # detectable (informational; datafolio does not freeze external data).
        identity = folio._storage.source_identity(path)
        if identity:
            updated["source_identity"] = identity

        return updated

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
        remote_path = folio._resolve_reference_path(item["path"])

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
            folio._resolve_reference_path(item["path"]),
            item.get("table_format", "parquet"),
            **kwargs,
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
