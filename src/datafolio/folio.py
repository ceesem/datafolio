"""Main DataFolio class for bundling analysis artifacts."""

import contextlib
import fnmatch
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cloudfiles
from typing_extensions import Self

# Import handlers to trigger auto-registration
import datafolio.handlers  # noqa: F401
from datafolio.accessors import DataAccessor, ItemProxy
from datafolio.base.registry import get_registry
from datafolio.display import DisplayFormatter
from datafolio.metadata import MetadataDict
from datafolio.storage import StorageBackend
from datafolio.utils import (
    ARTIFACTS_DIR,
    ITEMS_FILE,
    METADATA_DIR,
    METADATA_FILE,
    MODELS_DIR,
    SNAPSHOTS_FILE,
    TABLES_DIR,
    IncludedItem,
    IncludedTable,
    TableReference,
    TimestampItem,
    is_cloud_path,
    is_http_path,
    make_bundle_name,
    resolve_path,
    validate_item_name,
    validate_snapshot_name,
    validate_table_format,
)

# Current on-disk layout version of items.json. Bump when the manifest's shape
# changes in a way that needs migration on load.
MANIFEST_SCHEMA_VERSION = 1

# Manifest schema versions this build can read. A bare-list manifest and a dict
# manifest without an explicit ``schema_version`` are both treated as the
# pre-versioning format (0) and migrated forward on the next write. A manifest
# whose ``schema_version`` is newer than anything here is refused rather than
# silently reinterpreted (see :meth:`DataFolio._load_manifests`).
SUPPORTED_MANIFEST_VERSIONS = frozenset({0, 1})

# Bounded wait (seconds) for the per-folio local write lock before giving up.
DEFAULT_LOCK_TIMEOUT = 30.0


class ConcurrentWriteError(RuntimeError):
    """Raised when a write would clobber a newer manifest from another writer.

    datafolio supports many readers but a single writer per bundle. If a second
    writer has advanced the manifest revision since this instance last loaded
    it, writing would silently lose their changes; this error surfaces that
    instead. Call :meth:`DataFolio.refresh` and re-apply your change.

    This is also raised when the per-folio local write lock cannot be acquired
    within the bounded timeout (another live writer is holding it).
    """


class UnsupportedManifestVersionError(RuntimeError):
    """Raised when opening a folio written by a newer, unknown Datafolio.

    ``items.json`` records a ``schema_version``. If it is newer than any version
    this build knows how to read, the manifest is refused rather than silently
    reinterpreted — reading it under the wrong assumptions could corrupt data on
    the next write. Upgrade the ``datafolio`` package to open the folio.
    """


def _polars_only_error(name: str) -> ValueError:
    """Build the standard error for pandas access to a polars-only table."""
    return ValueError(
        f"Table '{name}' is a sharded/partitioned (polars-only) dataset and "
        f"cannot be loaded as pandas. Use get_lazy('{name}') for a lazy scan, "
        f"or get_table('{name}', frame='polars') to collect it eagerly."
    )


class SnapshotView:
    """Read-only view of a specific snapshot.

    Provides access to items and metadata as they existed at snapshot time.
    Items are read from their versioned files (e.g., data@v1.0.parquet).

    Examples:
        >>> folio = DataFolio('experiments/my-exp')
        >>> snapshot = folio.snapshots['v1.0']
        >>> df = snapshot.get_table('results')  # Read from snapshot version
        >>> print(snapshot.metadata)  # Get snapshot metadata
    """

    def __init__(self, folio: "DataFolio", snapshot_name: str):
        """Initialize snapshot view.

        Args:
            folio: Parent DataFolio instance
            snapshot_name: Name of the snapshot to view
        """
        self._folio = folio
        self._name = snapshot_name

        if snapshot_name not in folio._snapshots:
            raise ValueError(f"Snapshot '{snapshot_name}' not found")

        self._snapshot_meta = folio._snapshots[snapshot_name]

    @property
    def name(self) -> str:
        """Get snapshot name."""
        return self._name

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata as it existed at snapshot time."""
        return self._snapshot_meta.get("metadata_snapshot", {})

    @property
    def timestamp(self) -> str:
        """Get snapshot creation timestamp."""
        return self._snapshot_meta.get("timestamp", "")

    @property
    def description(self) -> Optional[str]:
        """Get snapshot description."""
        return self._snapshot_meta.get("description")

    @property
    def tags(self) -> list[str]:
        """Get snapshot tags."""
        return self._snapshot_meta.get("tags", [])

    @property
    def item_versions(self) -> Dict[str, int]:
        """Get item versions in this snapshot."""
        return self._snapshot_meta.get("item_versions", {})

    def get_table(self, name: str, frame: str = "pandas") -> Any:
        """Get a table as it existed in this snapshot.

        Args:
            name: Table name
            frame: Output flavor — ``'pandas'`` (default) or ``'polars'`` for an
                eager polars DataFrame.

        Returns:
            Table data (pandas or polars DataFrame)

        Raises:
            KeyError: If table not in snapshot
            ValueError: If ``frame`` is invalid
        """
        if name not in self.item_versions:
            raise KeyError(f"Table '{name}' not found in snapshot '{self._name}'")

        if frame not in ("pandas", "polars"):
            raise ValueError(f"Unknown frame '{frame}'. Use 'pandas' or 'polars'.")

        # Find the snapshot version item
        snapshot_item = self._find_snapshot_item(name)

        # Use handler to read the data
        # Handlers expect to find item in folio._items, so temporarily inject it
        handler = get_registry().get(snapshot_item["item_type"])
        if handler is None:
            raise RuntimeError(
                f"No handler for item type: {snapshot_item['item_type']}"
            )

        # Temporarily inject snapshot item into _items for handler to find
        original_item = self._folio._items.get(name)
        try:
            self._folio._items[name] = snapshot_item
            if frame == "polars":
                return handler.get_lazy(self._folio, name).collect()
            # Apply the same eager-read safeguards as the live get_table():
            # a sharded/partitioned dataset can't be pandas-materialized, and a
            # too-large table should be read lazily via scan_table().
            if snapshot_item.get("polars_only"):
                raise _polars_only_error(name)
            self._folio._check_eager_size(name, snapshot_item, allow_full_load=False)
            return handler.get(self._folio, name)
        finally:
            # Restore original item
            if original_item is not None:
                self._folio._items[name] = original_item
            else:
                self._folio._items.pop(name, None)

    def scan_table(self, name: str, **kwargs) -> Any:  # Returns polars.LazyFrame
        """Scan a snapshotted table as a genuinely lazy polars LazyFrame.

        The snapshot counterpart to :meth:`DataFolio.scan_table`: predicate/
        projection pushdown over the exact version recorded at snapshot time,
        without materializing it. For a referenced table this reads from the
        recorded external path (whose bytes are not owned/frozen — see
        :meth:`DataFolio.mutable_references`). Not subject to the eager-size
        guard (the whole point is to avoid a full read).

        Args:
            name: Table name.
            **kwargs: Passed to the polars scanner (e.g. ``storage_options``).

        Returns:
            polars LazyFrame.

        Raises:
            KeyError: If the table is not in this snapshot.
            ValueError: If the item is not a table.
            NotImplementedError: If its format has no lazy scanner.
        """
        if name not in self.item_versions:
            raise KeyError(f"Table '{name}' not found in snapshot '{self._name}'")

        snapshot_item = self._find_snapshot_item(name)
        item_type = snapshot_item.get("item_type")
        if item_type not in ("included_table", "referenced_table"):
            raise ValueError(f"Item '{name}' is not a table (type: {item_type})")

        handler = get_registry().get(item_type)
        original_item = self._folio._items.get(name)
        try:
            self._folio._items[name] = snapshot_item
            return handler.get_lazy(self._folio, name, **kwargs)
        finally:
            if original_item is not None:
                self._folio._items[name] = original_item
            else:
                self._folio._items.pop(name, None)

    def _find_snapshot_item(self, name: str) -> Dict[str, Any]:
        """Find the item metadata for this snapshot.

        Args:
            name: Item name

        Returns:
            Item metadata dict

        Raises:
            KeyError: If item not found
        """
        # Check if it's the current version that's in this snapshot
        if name in self._folio._items:
            item = self._folio._items[name]
            if self._name in item.get("in_snapshots", []):
                return item

        # Otherwise search snapshot versions
        for item in self._folio._snapshot_versions:
            if item.get("name") == name and self._name in item.get("in_snapshots", []):
                return item

        raise KeyError(f"Item '{name}' not found in snapshot '{self._name}'")


class SnapshotAccessor:
    """Dict-like accessor for snapshots.

    Allows accessing snapshots like: folio.snapshots['v1.0']
    """

    def __init__(self, folio: "DataFolio"):
        """Initialize accessor.

        Args:
            folio: Parent DataFolio instance
        """
        self._folio = folio

    def __getitem__(self, name: str) -> SnapshotView:
        """Get a snapshot by name.

        Args:
            name: Snapshot name

        Returns:
            SnapshotView for the given snapshot

        Raises:
            KeyError: If snapshot not found
        """
        if name not in self._folio._snapshots:
            raise KeyError(f"Snapshot '{name}' not found")

        return SnapshotView(self._folio, name)

    def __contains__(self, name: str) -> bool:
        """Check if snapshot exists.

        Args:
            name: Snapshot name

        Returns:
            True if snapshot exists
        """
        return name in self._folio._snapshots

    def __iter__(self):
        """Iterate over snapshot names."""
        return iter(self._folio._snapshots.keys())

    def __len__(self) -> int:
        """Get number of snapshots."""
        return len(self._folio._snapshots)

    def keys(self):
        """Get snapshot names."""
        return self._folio._snapshots.keys()

    def values(self):
        """Get SnapshotView objects for all snapshots."""
        return [SnapshotView(self._folio, name) for name in self._folio._snapshots]

    def items(self):
        """Get (name, SnapshotView) pairs."""
        return [
            (name, SnapshotView(self._folio, name)) for name in self._folio._snapshots
        ]


class DataFolio:
    """A lightweight bundle for tracking analysis artifacts and metadata.

    DataFolio uses a directory structure that supports incremental writes
    and cloud storage. All operations write immediately to disk.

    Directory structure:
        my-experiment-blue-happy-falcon/
        ├── metadata.json      # User metadata
        ├── items.json         # Unified manifest for all items (tables, models, artifacts)
        ├── tables/
        │   └── results.parquet
        ├── models/
        │   └── classifier.joblib
        └── artifacts/
            └── plot.png

    The items.json manifest uses an 'item_type' field to distinguish between:
    - 'referenced_table': External data not copied to bundle
    - 'included_table': Data stored in bundle
    - 'model': ML models stored in bundle
    - 'artifact': Other files (plots, configs, etc.) stored in bundle

    Examples:
        Create a new bundle:
        >>> folio = DataFolio(
        ...     'experiments/my-exp',
        ...     metadata={'experiment': 'test_001'}
        ... )
        >>> folio.add_table('results', df)  # Writes immediately
        >>> folio.reference_table('raw_data', path='s3://bucket/data.parquet')

        Load an existing bundle:
        >>> folio = DataFolio('experiments/my-exp')
        >>> print(folio.metadata)
        >>> df = folio.get_table('results')
    """

    def __init__(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        random_suffix: bool = False,
        read_only: bool = False,
        use_https: bool = False,
        max_eager_bytes: Optional[int] = 500 * 1024 * 1024,
    ):
        """Initialize a new or open an existing DataFolio.

        If the directory doesn't exist, creates a new bundle.
        If it exists, opens the existing bundle and reads manifests.

        Args:
            path: Full path to bundle directory (local or cloud)
            metadata: Optional dictionary of analysis metadata (for new bundles)
            random_suffix: If True, append random suffix to bundle name (default: False)
            read_only: If True, prevent all write operations (default: False)
            use_https: If True, use HTTPS URLs for CloudFiles (for read-only access to public buckets) (default: False)
            max_eager_bytes: Size ceiling (in bytes) for eager, full-table reads
                via ``get_table``. A table whose recorded ``size_bytes`` exceeds
                this raises unless it is flagged ``allow_full_load`` — use
                ``get_lazy`` instead. Set to ``None`` to disable the guard
                (default: 500 MB).

        Examples:
            Create new bundle with exact name:
            >>> folio = DataFolio('experiments/protein-analysis')
            # Creates: experiments/protein-analysis/

            Create new bundle with random suffix:
            >>> folio = DataFolio(
            ...     'experiments/protein-analysis',
            ...     random_suffix=True
            ... )
            # Creates: experiments/protein-analysis-blue-happy-falcon/

            Open existing bundle:
            >>> folio = DataFolio('experiments/protein-analysis')

            With metadata:
            >>> folio = DataFolio(
            ...     'experiments/my-exp',
            ...     metadata={'date': '2024-01-15', 'scientist': 'Dr. Smith'}
            ... )

            Open existing bundle as read-only (for safe inspection):
            >>> folio = DataFolio('experiments/production-model', read_only=True)
            >>> model = folio.get_model('classifier')  # OK
            >>> folio.add_table('new', df)  # Error: read-only
        """
        # Read-only mode flag
        self._read_only = read_only

        # HTTP/HTTPS paths are always read-only (cannot create a new folio over HTTP)
        if is_http_path(str(path)):
            self._read_only = True

        # HTTPS mode flag (for read-only access to public cloud buckets)
        self._use_https = use_https

        # Ceiling for eager full-table reads (None disables the guard)
        self._max_eager_bytes = max_eager_bytes

        # Snapshot mode flags (set by load_snapshot())
        self._in_snapshot_mode = False
        self._loaded_snapshot: Optional[str] = None

        # Storage backend for all I/O operations
        self._storage = StorageBackend(use_https=use_https)

        # Unified items dictionary - current versions only (for fast lookup)
        self._items: Dict[str, Union[TableReference, IncludedTable, IncludedItem]] = {}

        # Snapshot versions - non-current versions referenced by snapshots
        self._snapshot_versions: list[
            Union[TableReference, IncludedTable, IncludedItem]
        ] = []

        # For storing models/artifacts before writing
        self._models_data: Dict[str, Any] = {}  # Model storage
        self._artifacts_paths: Dict[str, Union[str, Path]] = {}  # Artifact file paths

        # Store random suffix setting for collision retry
        self._use_random_suffix = random_suffix

        # Auto-refresh tracking for multi-instance consistency
        self._auto_refresh_enabled: bool = True  # Can be disabled if needed
        self._in_save_operation: bool = False  # Prevent refresh during saves

        # Batch mode flag
        self._batch_mode = False

        # Per-folio local write lock (created lazily; reused so nested
        # acquisitions within one instance are reentrant rather than
        # self-deadlocking). Guards the full mutation: payload write +
        # manifest publish. Readers never touch it.
        self._file_lock: Any = None
        self._lock_timeout: float = DEFAULT_LOCK_TIMEOUT
        # Depth of the active mutation guard (reentrancy counter).
        self._mutation_depth: int = 0
        # Version ids / payload filenames reserved during this instance's
        # lifetime, so repeated replacements of the same logical item (e.g.
        # within a single batch, before the revision advances) never collide.
        self._reserved_version_ids: set[str] = set()
        # Obsolete owned payloads queued for deletion after the manifest is
        # actually published (used during a batch, where the publish is deferred
        # to the end so nothing referenced by the committed manifest is removed).
        self._pending_obsolete_payloads: list[Dict[str, Any]] = []

        # Manifest revision last loaded/written (stale-writer detection).
        # None until a manifest is loaded or first written.
        self._manifest_revision: Optional[int] = None

        # Snapshot-related state (Phase 1)
        self._snapshots: Dict[str, Any] = {}  # Snapshot name → metadata

        # Check if path is an existing bundle (has metadata.json or items.json)
        path_str = str(path)
        metadata_path = self._storage.join_paths(path_str, METADATA_FILE)
        items_path = self._storage.join_paths(path_str, ITEMS_FILE)

        if is_http_path(path_str):
            # HTTP/HTTPS: skip directory listing (not supported); check manifest files directly
            is_existing_bundle = self._storage.exists(
                metadata_path
            ) or self._storage.exists(items_path)
            if not is_existing_bundle:
                raise FileNotFoundError(
                    f"No datafolio bundle found at '{path_str}'. "
                    "HTTP/HTTPS paths must point to an existing bundle."
                )
        else:
            is_existing_bundle = self._storage.exists(path_str) and (
                self._storage.exists(metadata_path) or self._storage.exists(items_path)
            )

        if is_existing_bundle:
            # Open existing bundle
            self._bundle_dir = path_str
            self._bundle_path = Path(self._bundle_dir)
            self._is_new = False
            self._load_manifests()
            # Initialize metadata from file
            self.metadata = MetadataDict(self, **self._metadata_raw)
        else:
            # Create new bundle
            if random_suffix:
                # Append random suffix to the last component of the path
                # For 'experiments/my-exp' → 'experiments/my-exp-blue-happy-falcon'
                if is_cloud_path(path_str):
                    # Cloud path: split on '/'
                    parts = path_str.rstrip("/").split("/")
                    last_component = parts[-1]
                    parent = "/".join(parts[:-1])
                    bundle_name = make_bundle_name(last_component)
                    self._bundle_dir = (
                        self._storage.join_paths(parent, bundle_name)
                        if parent
                        else bundle_name
                    )
                else:
                    # Local path
                    path_obj = Path(path_str)
                    parent = path_obj.parent
                    last_component = path_obj.name
                    bundle_name = make_bundle_name(last_component)
                    self._bundle_dir = str(parent / bundle_name)
            else:
                # Use exact path as provided
                self._bundle_dir = path_str
            self._bundle_path = Path(self._bundle_dir)
            self._is_new = True

            # Initialize metadata with timestamps and version info
            self._metadata_raw = metadata or {}
            now = datetime.now(timezone.utc).isoformat()
            if "created_at" not in self._metadata_raw:
                self._metadata_raw["created_at"] = now
            if "updated_at" not in self._metadata_raw:
                self._metadata_raw["updated_at"] = now

            # Add datafolio version info (don't overwrite if already present)
            if "_datafolio" not in self._metadata_raw:
                from datafolio import __version__

                self._metadata_raw["_datafolio"] = {
                    "version": __version__,
                    "created_by": "datafolio",
                }

            self.metadata = MetadataDict(self, **self._metadata_raw)

            # Create directory structure with retries on collision
            self._initialize_bundle()

        self._cf = cloudfiles.CloudFiles(resolve_path(self._bundle_dir))

        # Initialize data accessor for autocomplete support
        # Create it here (not lazily) so autocomplete is immediately available
        self._data_accessor = DataAccessor(self)

    # ==================== Well-factored I/O Helper Functions ====================

    def _sync_data_accessor(self) -> None:
        """Sync data accessor after items have changed.

        This ensures autocomplete is updated immediately when items are added,
        removed, or modified, rather than waiting for the next access to .data.
        """
        if hasattr(self, "_data_accessor"):
            self._data_accessor._sync_items()

    def _check_read_only(self) -> None:
        """Raise error if folio is in read-only mode.

        Raises:
            RuntimeError: If folio is in read-only mode with helpful message
        """
        if self._read_only:
            msg = "Cannot modify a read-only DataFolio"
            if self._in_snapshot_mode and self._loaded_snapshot:
                msg += f" (loaded from snapshot '{self._loaded_snapshot}')"
            msg += ". Open without read_only=True to make changes."
            raise RuntimeError(msg)

    def _exists(self, path: str) -> bool:
        """Check if a path exists (local or cloud).

        Args:
            path: Path to check

        Returns:
            True if path exists
        """
        if is_cloud_path(path):
            # For cloud, we'll try to list contents
            # This is a placeholder - you'll refine for cloudfiles
            try:
                from cloudfiles import CloudFiles

                cf = CloudFiles(path)
                # Try to list - if it works, directory exists
                list(cf.list())
                return True
            except:
                return False
        else:
            return Path(path).exists()

    def _mkdir(self, path: str, parents: bool = True, exist_ok: bool = True) -> None:
        """Create a directory (local or cloud).

        Args:
            path: Directory path to create
            parents: Create parent directories if needed
            exist_ok: Don't error if directory exists
        """
        if is_cloud_path(path):
            # Cloud storage is object-based, no need to create directories
            # They're created implicitly when you write files
            pass
        else:
            Path(path).mkdir(parents=parents, exist_ok=exist_ok)

    def _join_paths(self, *parts: str) -> str:
        """Join path components (local or cloud).

        Args:
            *parts: Path components to join

        Returns:
            Joined path string
        """
        if any(is_cloud_path(str(p)) for p in parts):
            # Cloud path - use forward slashes
            return "/".join(str(p).rstrip("/") for p in parts)
        else:
            # Local path
            return str(Path(*parts))

    def _write_json(self, path: str, data: Any) -> None:
        """Write JSON data to file (local or cloud).

        Args:
            path: File path
            data: Data to serialize
        """
        import orjson

        content = orjson.dumps(
            data, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
        )

        if is_cloud_path(path):
            from cloudfiles import CloudFiles

            # Extract directory and filename
            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
            cf.put(filename, content)
        else:
            with open(path, "wb") as f:
                f.write(content)

    def _read_json(self, path: str) -> Any:
        """Read JSON data from file (local or cloud).

        Args:
            path: File path

        Returns:
            Deserialized data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file content is empty or invalid
        """
        import orjson

        if is_cloud_path(path):
            from cloudfiles import CloudFiles

            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
            content = cf.get(filename)

            # Handle case where file doesn't exist or is empty
            if content is None:
                raise FileNotFoundError(f"File not found in cloud storage: {path}")
            if not content:
                raise ValueError(f"Empty file in cloud storage: {path}")

            return orjson.loads(content)
        else:
            with open(path, "rb") as f:
                return orjson.loads(f.read())

    def _write_parquet(self, path: str, df: Any) -> None:
        """Write DataFrame to parquet (local or cloud).

        Args:
            path: File path
            df: pandas DataFrame
        """
        # pandas.to_parquet handles cloud paths if fsspec/cloud libs installed
        df.to_parquet(path, index=False)

    def _read_parquet(self, path: str) -> Any:
        """Read parquet file to DataFrame (local or cloud).

        Args:
            path: File path

        Returns:
            pandas DataFrame
        """
        import pandas as pd

        return pd.read_parquet(path)

    def _write_joblib(self, path: str, obj: Any) -> None:
        """Write object with joblib (local or cloud).

        Args:
            path: File path
            obj: Object to serialize
        """
        import joblib

        if is_cloud_path(path):
            # Write to temp file, then upload
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                joblib.dump(obj, tmp.name)
                with open(tmp.name, "rb") as f:
                    content = f.read()
                # Upload
                from cloudfiles import CloudFiles

                parts = path.rsplit("/", 1)
                if len(parts) == 2:
                    dir_path, filename = parts
                else:
                    dir_path = ""
                    filename = parts[0]
                cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
                cf.put(filename, content)
                # Cleanup
                Path(tmp.name).unlink()
        else:
            joblib.dump(obj, path)

    def _read_joblib(self, path: str) -> Any:
        """Read object with joblib (local or cloud).

        Args:
            path: File path

        Returns:
            Deserialized object
        """
        import joblib

        if is_cloud_path(path):
            # Download to temp file, then load
            import tempfile

            from cloudfiles import CloudFiles

            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
            content = cf.get(filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                tmp.write(content)
                tmp.flush()
                obj = joblib.load(tmp.name)
                Path(tmp.name).unlink()
                return obj
        else:
            return joblib.load(path)

    def _write_numpy(self, path: str, array: Any) -> None:
        """Write numpy array to file (local or cloud).

        Args:
            path: File path
            array: numpy array to save
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "NumPy is required to save numpy arrays. "
                "Install with: pip install numpy"
            )

        if is_cloud_path(path):
            # Write to temp file, then upload
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                np.save(tmp.name, array)
                with open(tmp.name, "rb") as f:
                    content = f.read()
                # Upload
                from cloudfiles import CloudFiles

                parts = path.rsplit("/", 1)
                if len(parts) == 2:
                    dir_path, filename = parts
                else:
                    dir_path = ""
                    filename = parts[0]
                cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
                cf.put(filename, content)
                # Cleanup
                Path(tmp.name).unlink()
        else:
            np.save(path, array)

    def _read_numpy(self, path: str) -> Any:
        """Read numpy array from file (local or cloud).

        Args:
            path: File path

        Returns:
            numpy array
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "NumPy is required to load numpy arrays. "
                "Install with: pip install numpy"
            )

        if is_cloud_path(path):
            # Download to temp file, then load
            import tempfile

            from cloudfiles import CloudFiles

            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
            content = cf.get(filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                tmp.write(content)
                tmp.flush()
                array = np.load(tmp.name)
                Path(tmp.name).unlink()
                return array
        else:
            return np.load(path)

    def _write_timestamp(self, path: str, timestamp: datetime) -> None:
        """Write timestamp to JSON file (local or cloud).

        Args:
            path: File path
            timestamp: datetime object (must be timezone-aware, will be converted to UTC)
        """
        # Convert to UTC and create ISO 8601 string
        utc_timestamp = timestamp.astimezone(timezone.utc)
        iso_string = utc_timestamp.isoformat()

        # Write as JSON using existing method
        self._storage.write_json(path, {"iso_string": iso_string})

    def _read_timestamp(self, path: str) -> datetime:
        """Read timestamp from JSON file (local or cloud).

        Args:
            path: File path

        Returns:
            UTC-aware datetime object
        """
        # Read JSON using existing method
        data = self._storage.read_json(path)
        iso_string = data["iso_string"]

        # Parse ISO 8601 string to datetime
        return datetime.fromisoformat(iso_string)

    def _copy_file(self, src: Union[str, Path], dst: str) -> None:
        """Copy a file (local to local/cloud).

        Args:
            src: Source file path (local)
            dst: Destination path (local or cloud)
        """
        if is_cloud_path(dst):
            from cloudfiles import CloudFiles

            with open(src, "rb") as f:
                content = f.read()
            parts = dst.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(dst)
            cf.put(filename, content)
        else:
            import shutil

            shutil.copy2(src, dst)

    # ==================== Bundle Initialization ====================

    def _write_readme(self) -> None:
        """Write a README.md documenting the on-disk format as a public surface.

        The on-disk layout is a compatibility surface: a folio must stay usable
        without the datafolio package, degrading into ordinary files plus a
        readable manifest. This README explains how to do exactly that.
        """
        from datafolio import __version__

        readme_content = f"""# DataFolio Bundle

This directory was created by [datafolio](https://github.com/ceesem/datafolio) version {__version__}.

It is designed to remain useful **without** the datafolio package: it is an
ordinary directory of standard files plus a readable JSON manifest.

## Structure

- `items.json` - **The authoritative catalog** of every data item. Read this to
  learn what the folio contains and where each item lives.
- `metadata.json` - User metadata and timestamps.
- `snapshots.json` - Named snapshots (if any), pinning item versions.
- `CONTENTS.md` - A **derived**, human-readable inventory (regenerated from
  `items.json`; never authoritative — do not parse it, read `items.json`).
- `tables/` - Parquet files for included (owned) tables.
- `models/` - Serialized ML models (joblib/skops).
- `artifacts/` - Numpy arrays (`.npy`), JSON data (`.json`), images, text, and
  other file artifacts.

## Reading a folio without datafolio

`items.json` is a JSON object: `{{"schema_version", "revision", "items": [...]}}`
(very old folios may be a bare `[...]` list). Each entry in `items` describes
one item version.

1. **Find the current items.** Use only entries where `is_current` is `true`
   (or absent, in old folios). Entries with `is_current: false` are prior
   versions preserved for a snapshot.
2. **Owned items** carry a `filename` relative to their type's subdirectory,
   derived from `item_type`: `included_table` → `tables/`, `model` →
   `models/`, everything else → `artifacts/`. Open the file at
   `<subdir>/<filename>` with the standard library for its format:
   - Parquet (`included_table`): `pandas.read_parquet` / `polars.read_parquet` /
     `pyarrow.parquet`.
   - JSON (`json_data`, `timestamp`): any JSON reader.
   - NumPy (`numpy_array`): `numpy.load`.
   - Text / images / other (`artifact`): open by extension.
3. **External references** (`referenced_table`) carry an absolute `path`/URI
   under `path` instead of a `filename`. datafolio does **not** copy or own that
   data — it is not stored in the bundle, and its bytes are not guaranteed or
   frozen (the reference records a link, which may have changed since). Read it
   directly from that path.

See the "Using a folio without Datafolio" page in the documentation for the full
manifest field reference and path-resolution rules.

## Usage with datafolio

```python
from datafolio import DataFolio

folio = DataFolio('{self.path}')
folio.describe()                       # view contents

df = folio.get_table('table_name')     # owned or referenced tables
model = folio.get_model('model_name')
array = folio.get_numpy('array_name')

lf = folio.scan_table('big_reference') # lazy scan of a large external table
folio.inspect_table('big_reference')   # record its schema/size on demand
```

## Documentation

For more information, see the [datafolio documentation](https://github.com/ceesem/datafolio).
"""

        readme_path = self._storage.join_paths(self._bundle_dir, "README.md")
        self._write_text_file(readme_path, readme_content)

    def _write_text_file(self, path: str, content: str) -> None:
        """Write a UTF-8 text file to local or cloud storage."""
        if is_cloud_path(path):
            from cloudfiles import CloudFiles

            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
            cf.put(filename, content.encode("utf-8"))
        else:
            with open(path, "w") as f:
                f.write(content)

    def _write_contents(self) -> None:
        """Regenerate the derived, human-facing ``CONTENTS.md`` inventory.

        This is a convenience view, **never** an authoritative catalog:
        ``items.json`` remains the source of truth. The file is clearly labelled
        as derived, records the manifest ``revision`` it represents, and lists
        each current item's logical name, type, description, and path. It is
        never required to read the folio. Best-effort: failures are swallowed so
        a documentation write can't fail a data mutation.
        """
        try:
            revision = self._manifest_revision or 0
            lines = [
                "# Contents",
                "",
                "> **Derived file — do not edit.** Regenerated from `items.json`, "
                "which is the authoritative catalog. This reflects manifest "
                f"revision **{revision}**.",
                "",
                "| Name | Type | Location | Description |",
                "| --- | --- | --- | --- |",
            ]
            from datafolio.storage import get_storage_directory

            def _cell(value: str) -> str:
                return str(value).replace("|", "\\|").replace("\n", " ")

            for name in sorted(self._items):
                item = self._items[name]
                item_type = item.get("item_type", "unknown")
                description = item.get("description", "") or ""
                if item.get("filename"):
                    try:
                        subdir = get_storage_directory(item_type)
                        location = f"{subdir}/{item['filename']}"
                    except Exception:
                        location = item["filename"]
                elif item.get("path"):
                    location = f"{item['path']} (external reference)"
                else:
                    location = ""
                lines.append(
                    f"| {_cell(name)} | {_cell(item_type)} | {_cell(location)} "
                    f"| {_cell(description)} |"
                )

            lines.append("")
            contents_path = self._storage.join_paths(self._bundle_dir, "CONTENTS.md")
            self._write_text_file(contents_path, "\n".join(lines))
        except Exception:
            # A derived convenience file must never break a real mutation.
            pass

    def _initialize_bundle(self, max_retries: int = 10) -> None:
        """Initialize new bundle directory structure with collision retry.

        Args:
            max_retries: Maximum number of retries on name collision
        """
        # Check if directory already exists
        if self._storage.exists(self._bundle_dir):
            if self._use_random_suffix:
                # Try to create directory, retry with new random name on collision
                for _ in range(max_retries):
                    # Collision! Generate new random name
                    # Extract the base name and regenerate with new suffix
                    if is_cloud_path(self._bundle_dir):
                        parts = self._bundle_dir.rstrip("/").split("/")
                        parent = "/".join(parts[:-1])
                        # Extract base name (before random suffix)
                        last_component = parts[-1]
                        base_parts = last_component.split("-")
                        if len(base_parts) >= 4:  # Has random suffix
                            base_name = "-".join(base_parts[:-3])
                        else:
                            base_name = last_component
                        new_name = make_bundle_name(base_name)
                        self._bundle_dir = (
                            self._storage.join_paths(parent, new_name)
                            if parent
                            else new_name
                        )
                    else:
                        parent = Path(self._bundle_dir).parent
                        last_component = Path(self._bundle_dir).name
                        base_parts = last_component.split("-")
                        if len(base_parts) >= 4:  # Has random suffix
                            base_name = "-".join(base_parts[:-3])
                        else:
                            base_name = last_component
                        new_name = make_bundle_name(base_name)
                        self._bundle_dir = str(parent / new_name)

                    # Check if new name is available
                    if not self._storage.exists(self._bundle_dir):
                        break
                else:
                    raise RuntimeError(
                        f"Failed to create unique bundle name after {max_retries} attempts"
                    )
            else:
                # No random suffix - fail immediately on collision
                raise FileExistsError(
                    f"Bundle directory already exists: {self._bundle_dir}. "
                    "Use use_random_suffix=True to generate unique names automatically."
                )

        # Create directory structure
        self._storage.mkdir(self._bundle_dir)
        self._storage.mkdir(self._storage.join_paths(self._bundle_dir, TABLES_DIR))
        self._storage.mkdir(self._storage.join_paths(self._bundle_dir, MODELS_DIR))
        self._storage.mkdir(self._storage.join_paths(self._bundle_dir, ARTIFACTS_DIR))

        # Write initial manifests and README
        self._save_metadata()
        self._save_items()
        self._write_readme()

    def _load_manifests(self) -> None:
        """Load all manifest files from existing bundle."""

        # Read metadata.json
        metadata_path = self._storage.join_paths(self._bundle_dir, METADATA_FILE)
        if self._storage.exists(metadata_path):
            self._metadata_raw = self._storage.read_json(metadata_path)
        else:
            self._metadata_raw = {}

        # Read unified items.json
        items_path = self._storage.join_paths(self._bundle_dir, ITEMS_FILE)
        if self._storage.exists(items_path):
            items_data = self._storage.read_json(items_path)

            # Handle both old format (list) and new format (dict with items).
            # The manifest's ``schema_version`` gates compatibility: known
            # versions (including the pre-versioning formats, treated as 0) are
            # read and migrated forward on the next write; an unknown *newer*
            # version is refused rather than silently reinterpreted.
            if isinstance(items_data, list):
                # Oldest format: a bare list of items (backward compatibility).
                items_list = items_data
                self._manifest_revision = 0
            else:
                schema_version = int(items_data.get("schema_version", 0) or 0)
                if schema_version not in SUPPORTED_MANIFEST_VERSIONS:
                    raise UnsupportedManifestVersionError(
                        f"items.json at {items_path} declares schema_version "
                        f"{schema_version}, which this build of datafolio does "
                        f"not understand (it supports up to "
                        f"{MANIFEST_SCHEMA_VERSION}). Upgrade the datafolio "
                        f"package to open this folio."
                    )
                # Dict format: {schema_version?, revision?, items}. Missing
                # revision (pre-versioning manifests) is treated as 0 and
                # migrated forward on the next write.
                items_list = items_data.get("items", [])
                self._manifest_revision = int(items_data.get("revision", 0) or 0)

            # Separate current versions from snapshot versions
            self._items = {}
            self._snapshot_versions = []

            for item in items_list:
                # Initialize snapshot fields for backward compatibility
                if "in_snapshots" not in item:
                    item["in_snapshots"] = []
                if "is_current" not in item:
                    item["is_current"] = True

                # Separate based on is_current flag
                if item.get("is_current", True):
                    self._items[item["name"]] = item
                else:
                    self._snapshot_versions.append(item)
        else:
            self._items = {}
            self._snapshot_versions = []

        # Read snapshots.json (if it exists)
        snapshots_path = self._storage.join_paths(self._bundle_dir, SNAPSHOTS_FILE)
        if self._storage.exists(snapshots_path):
            snapshots_data = self._storage.read_json(snapshots_path)
            self._snapshots = snapshots_data.get("snapshots", {})
        else:
            self._snapshots = {}

    # ==================== Manifest Save Methods ====================

    def _save_metadata(self) -> None:
        """Save metadata.json."""
        path = self._storage.join_paths(self._bundle_dir, METADATA_FILE)
        # Convert MetadataDict to regular dict for serialization
        data = (
            dict(self.metadata)
            if isinstance(self.metadata, MetadataDict)
            else self.metadata
        )
        self._storage.write_json(path, data)

    def _local_lock(self) -> Any:
        """Return this instance's reentrant local write lock (lazy).

        A single :class:`filelock.FileLock` object is reused for the folio's
        lifetime. filelock is reentrant *per object* (an internal counter), so
        nested acquisitions inside one process/instance — e.g. a public method
        entering the mutation guard and then calling ``_save_items`` — do not
        self-deadlock. Constructing a fresh FileLock per call would instead
        block on a second file descriptor.
        """
        if self._file_lock is None:
            from filelock import FileLock

            path = self._storage.join_paths(self._bundle_dir, ITEMS_FILE)
            self._file_lock = FileLock(path + ".lock", timeout=self._lock_timeout)
        return self._file_lock

    @contextlib.contextmanager
    def _mutation_guard(self):
        """Serialize a complete local mutation and reject a stale writer early.

        Wrapping an add/overwrite in this guard enforces the required ordering:

            acquire local folio lock
            → check the manifest revision (reject a stale writer)
            → [caller writes the new payload]
            → [caller atomically publishes the manifest via _save_items()]
            → release the lock

        Readers never enter this guard. The guard is reentrant (a public method
        may delegate to another guarded method, or to ``_save_items`` which
        re-acquires the same lock) and always releases on exception or normal
        exit. Cloud object stores have no local lock; the stale check still runs
        (best-effort — object stores lack conditional writes).
        """
        # Every mutation flows through this guard, so read-only enforcement
        # lives here as well as in the public methods (defense in depth).
        self._check_read_only()

        # Reentrant: an outer guard (or batch) already holds the lock and has
        # done the stale check. Just nest.
        if self._mutation_depth > 0:
            self._mutation_depth += 1
            try:
                yield
            finally:
                self._mutation_depth -= 1
            return

        path = self._storage.join_paths(self._bundle_dir, ITEMS_FILE)
        lock = None
        if not is_cloud_path(path):
            from filelock import Timeout

            lock = self._local_lock()
            try:
                lock.acquire(timeout=self._lock_timeout)
            except Timeout as exc:
                raise ConcurrentWriteError(
                    f"Could not acquire the folio write lock within "
                    f"{self._lock_timeout:g}s — another writer is active on "
                    f"{self._bundle_dir}. datafolio supports many readers but "
                    f"one writer per bundle."
                ) from exc
        self._mutation_depth = 1
        try:
            # Reject a stale writer BEFORE any payload is written or replaced.
            self._raise_if_manifest_stale(path)
            yield
        finally:
            self._mutation_depth = 0
            if lock is not None:
                lock.release()

    def _save_items(self) -> None:
        """Save unified items.json manifest (versioned, atomic, stale-checked).

        Writes ``{schema_version, revision, items}``. Local writes are atomic
        (temp + os.replace) and serialized by the per-folio reentrant lockfile;
        before overwriting, the on-disk revision is checked so a stale writer
        can't silently clobber a newer manifest (raises
        :class:`ConcurrentWriteError`). This supports the "many readers, one
        writer" model — see the class docstring. Cloud object stores lack
        conditional writes, so cross-writer safety there is best-effort (the
        revision advances but two simultaneous cloud writers can still race).

        Normally this runs inside a :meth:`_mutation_guard` (which already holds
        the lock and did the stale check); the reentrant lock makes the
        re-acquisition here a no-op counter bump. When called on its own (e.g.
        manifest-only operations like snapshotting or archiving) it still
        acquires the lock and checks staleness itself.
        """
        if self._batch_mode:
            return

        # Set flag to prevent auto-refresh during save
        self._in_save_operation = True
        try:
            path = self._storage.join_paths(self._bundle_dir, ITEMS_FILE)

            # Serialize local writers with the reentrant per-folio lock so the
            # read-check-write below is atomic on a single machine (no lock for
            # cloud). Reusing one FileLock object keeps nested acquisition safe.
            lock_ctx: Any = contextlib.nullcontext()
            if not is_cloud_path(path):
                lock_ctx = self._local_lock()

            with lock_ctx:
                self._raise_if_manifest_stale(path)

                all_items = list(self._items.values()) + self._snapshot_versions
                next_revision = (self._manifest_revision or 0) + 1
                items_data = {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "revision": next_revision,
                    "items": all_items,
                }
                self._storage.write_json(path, items_data)
                self._manifest_revision = next_revision

            # Update metadata timestamp when items change
            # This allows other instances to detect staleness
            if hasattr(self, "metadata"):
                from datetime import datetime, timezone

                # Use super() to update without triggering another save
                super(MetadataDict, self.metadata).__setitem__(
                    "updated_at", datetime.now(timezone.utc).isoformat()
                )
                self._save_metadata()

            # Refresh the derived, human-facing inventory (best-effort; never
            # authoritative — items.json remains the catalog).
            self._write_contents()

            # Sync data accessor to update autocomplete immediately
            self._sync_data_accessor()
        finally:
            # Always clear the flag
            self._in_save_operation = False

    def _raise_if_manifest_stale(self, path: str) -> None:
        """Guard against overwriting a manifest advanced by another writer.

        Compares the on-disk revision with the revision this instance last
        loaded/wrote. If the on-disk one is newer, another writer changed the
        bundle and proceeding would lose their work.

        Args:
            path: Path to items.json

        Raises:
            ConcurrentWriteError: If the on-disk manifest is newer than ours.
        """
        if self._manifest_revision is None:
            return  # nothing loaded/written yet (fresh bundle)
        if not self._storage.exists(path):
            return
        try:
            on_disk = self._storage.read_json(path)
        except Exception:
            return  # unreadable -> let the write proceed/fail normally
        disk_revision = (
            int(on_disk.get("revision", 0) or 0) if isinstance(on_disk, dict) else 0
        )
        if disk_revision > self._manifest_revision:
            raise ConcurrentWriteError(
                f"items.json was modified by another writer (on-disk revision "
                f"{disk_revision} > loaded {self._manifest_revision}). Call "
                f"refresh() and re-apply your change (datafolio supports many "
                f"readers but one writer per bundle)."
            )

    # ==================== Item version / payload naming ====================

    def _next_version_id(self, name: str) -> str:
        """Allocate a stable, collision-safe version id for an item version.

        Every persisted item version gets its own id (recognizable, not an
        opaque UUID) so snapshots can pin the exact version and payload files
        never clobber a version referenced by a committed manifest. The id
        embeds the manifest revision the write is heading toward, e.g.
        ``features--r17``. Repeated replacements of the same logical name before
        the revision advances (inside one batch) are disambiguated with a
        trailing counter so ids stay unique. Owned payload filenames are derived
        from this id (``<version_id><ext>``).

        Args:
            name: Logical item name.

        Returns:
            A version id string unique within this instance's lifetime.
        """
        next_rev = (self._manifest_revision or 0) + 1
        base = f"{name}--r{next_rev}"
        candidate = base
        counter = 1
        while candidate in self._reserved_version_ids:
            candidate = f"{base}-{counter}"
            counter += 1
        self._reserved_version_ids.add(candidate)
        return candidate

    def _reserve_payload_filename(
        self, name: str, extension: str, subdir: str
    ) -> tuple[str, str]:
        """Reserve a versioned payload filename and its version id.

        Guarantees the returned filename does not already exist on disk (so a
        stale or interrupted writer never overwrites a payload the committed
        manifest still references) and is unique among names reserved this
        session (safe across repeated replacements within one batch).

        Args:
            name: Logical item name.
            extension: File extension including the dot (e.g. ``.parquet``).
            subdir: Storage subdirectory the payload lives in.

        Returns:
            ``(version_id, filename)``.
        """
        while True:
            version_id = self._next_version_id(name)
            # Namespaced names ('a/b') intentionally map to subdirectories of
            # the category dir; validate_item_name guarantees no segment can
            # escape it ('..', absolute paths, and empty segments are rejected).
            filename = f"{version_id}{extension}"
            full = self._storage.join_paths(self._bundle_dir, subdir, filename)
            if not self._storage.exists(full):
                return version_id, filename
            # Name taken on disk (unlikely) — reserve the next and retry.

    def _apply_description(
        self, name: str, metadata: Dict[str, Any], description: Optional[str]
    ) -> None:
        """Apply the shared description semantics when (over)writing an item.

        - ``description=None`` on **create**: omit the description.
        - ``description=None`` on **overwrite**: preserve the existing one.
        - ``description=""``: explicitly remove any description.
        - non-empty ``description``: set/replace it.

        ``metadata`` is the freshly built entry (handlers only set
        ``description`` when it is non-empty); the prior entry, if any, is still
        in ``self._items`` at call time. A description is never discarded
        implicitly.

        Args:
            name: Logical item name.
            metadata: Freshly built metadata dict for the new version (mutated).
            description: The description argument as passed by the caller.
        """
        if description is None:
            prior = self._items.get(name)
            if prior is not None and prior.get("description") is not None:
                metadata["description"] = prior["description"]
            else:
                metadata.pop("description", None)
        elif description == "":
            metadata.pop("description", None)
        else:
            metadata["description"] = description

    def _obsolete_payload_after_commit(
        self, old_item: Optional[Dict[str, Any]]
    ) -> None:
        """Delete an owned payload made obsolete by a successful overwrite.

        Called only after the manifest has been published pointing at the new
        payload, and only for a prior version that is *not* preserved by any
        snapshot. A failed/interrupted operation may instead leave an
        unreferenced orphan — that is acceptable (no GC machinery).

        Args:
            old_item: The replaced item's metadata, or ``None`` if there was no
                prior version.
        """
        if not old_item:
            return
        if old_item.get("in_snapshots"):
            return  # preserved by a snapshot — never delete
        filename = old_item.get("filename")
        if not filename:
            return  # references own no payload
        if self._batch_mode:
            # The manifest publish is deferred to batch exit; defer the deletion
            # too, so an interrupted/stale batch can't remove a payload the
            # committed manifest still references.
            self._pending_obsolete_payloads.append(old_item)
            return
        self._delete_payload_if_unshared(old_item)

    def _payload_is_shared(self, item: Dict[str, Any]) -> bool:
        """Check whether another manifest descriptor references item's payload.

        A metadata-only copy-on-write (see :meth:`update_item`) produces two
        descriptors pointing at the same payload file. Any code path that
        deletes a payload must first confirm no *other* descriptor (current or
        snapshot version) still references it.

        Args:
            item: The descriptor about to lose its payload.

        Returns:
            True if some other descriptor references the same file.
        """
        filename = item.get("filename")
        if not filename:
            return False
        item_type = item.get("item_type")
        for other in list(self._items.values()) + self._snapshot_versions:
            if other is item:
                continue
            if (
                other.get("filename") == filename
                and other.get("item_type") == item_type
            ):
                return True
        return False

    def _delete_payload_if_unshared(self, item: Dict[str, Any]) -> None:
        """Best-effort deletion of a descriptor's payload file.

        The file is left alone when another descriptor still references it
        (shared payload after a metadata-only copy-on-write) or when the
        storage directory can't be resolved. Failures never propagate — an
        orphaned file is acceptable, a broken mutation is not.

        Args:
            item: The descriptor whose payload should be removed.
        """
        filename = item.get("filename")
        if not filename:
            return  # references own no payload
        if self._payload_is_shared(item):
            return
        item_type = item.get("item_type", "")
        try:
            from datafolio.storage import get_storage_directory

            subdir = get_storage_directory(item_type)
        except Exception:
            return
        old_path = self._storage.join_paths(self._bundle_dir, subdir, filename)
        try:
            if self._storage.exists(old_path):
                self._storage.delete_file(old_path)
        except Exception:
            # Best-effort cleanup; an orphan is acceptable.
            pass

    def _copy_payload_file(self, src_path: str, dst_path: str) -> None:
        """Copy one payload file between bundles (local or cloud on each side).

        Reads the whole object via cloudfiles (the same credential chain used
        for all bundle I/O; ``use_https`` honored on the read side) and writes
        it to the destination. Local paths are converted to absolute
        ``file://`` URIs for a uniform code path; parent directories of a
        local destination are created as needed (namespaced item names map to
        subdirectories).

        Args:
            src_path: Full path of the source payload file.
            dst_path: Full path of the destination payload file.

        Raises:
            FileNotFoundError: If the source payload does not exist.
        """
        src_cf_path = (
            src_path
            if is_cloud_path(src_path)
            else f"file://{Path(src_path).resolve()}"
        )
        src_dir, _, src_filename = src_cf_path.rpartition("/")
        src_cf = cloudfiles.CloudFiles(src_dir, use_https=self._use_https)
        content = src_cf.get(src_filename)
        if content is None:
            raise FileNotFoundError(f"Payload file not found: {src_path}")

        if is_cloud_path(dst_path) and not dst_path.startswith("file://"):
            dst_dir, _, dst_filename = dst_path.rpartition("/")
            cloudfiles.CloudFiles(dst_dir).put(dst_filename, content)
        else:
            dst = Path(dst_path.removeprefix("file://"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(content)

    def _commit_owned_item(
        self,
        name: str,
        handler_key: str,
        extension: str,
        description: Optional[str],
        build_metadata,
    ) -> None:
        """Add/overwrite an owned (payload-backed) item under shared invariants.

        Runs the full guarded mutation for every owned item type so the
        invariants live in one place:

        1. acquire the mutation guard (lock + stale-writer check *before* any
           payload is written);
        2. copy-on-write a snapshotted prior version out of the way;
        3. reserve a collision-safe versioned filename + ``version_id`` and let
           ``build_metadata`` write the payload there;
        4. apply the description semantics (never discard one implicitly);
        5. publish the manifest atomically;
        6. delete the now-obsolete prior payload (only if unsnapshotted).

        Args:
            name: Logical item name.
            handler_key: Registered handler item_type (for the storage subdir).
            extension: Payload file extension including the dot.
            description: Description argument as passed by the caller.
            build_metadata: ``callable(filename, version_id) -> metadata`` that
                writes the payload to ``filename`` and returns its metadata dict.
        """
        handler = get_registry().get(handler_key)
        subdir = handler.get_storage_subdir()
        with self._mutation_guard():
            prior = self._items.get(name)
            if self._is_in_snapshots(name):
                self._handle_copy_on_write(name)
            version_id, filename = self._reserve_payload_filename(
                name, extension, subdir
            )
            metadata = build_metadata(filename, version_id)
            metadata["version_id"] = version_id
            self._apply_description(name, metadata, description)
            metadata.setdefault("in_snapshots", [])
            metadata.setdefault("is_current", True)
            self._items[name] = metadata
            self._save_items()
            self._obsolete_payload_after_commit(prior)

    def _save_snapshots(self) -> None:
        """Save snapshots.json manifest."""

        if self._batch_mode:
            return

        path = self._storage.join_paths(self._bundle_dir, SNAPSHOTS_FILE)
        snapshots_data = {"snapshots": self._snapshots}
        self._storage.write_json(path, snapshots_data)

    # ==================== Snapshot Context Capture ====================

    def _sanitize_git_remote_url(self, url: str) -> Optional[str]:
        """Remove credentials from git remote URLs.

        Handles various git URL formats and removes embedded credentials
        (tokens, username:password) from HTTP(S) URLs while preserving the
        repository information.

        Args:
            url: Git remote URL (potentially with embedded credentials)

        Returns:
            Sanitized URL with credentials removed, or None if sanitization fails

        Examples:
            >>> _sanitize_git_remote_url('https://token@github.com/user/repo.git')
            'https://github.com/user/repo.git'

            >>> _sanitize_git_remote_url('https://user:pass@gitlab.com/repo.git')
            'https://gitlab.com/repo.git'

            >>> _sanitize_git_remote_url('git@github.com:user/repo.git')
            'git@github.com:user/repo.git'  # SSH format preserved (no credentials)

        Security:
            This prevents credential leakage when snapshots containing git
            information are shared with collaborators or made public.
        """
        from urllib.parse import urlparse, urlunparse

        if not url:
            return None

        # Handle SSH format (git@host:path) - safe to keep as-is
        # SSH URLs don't contain credentials, they use SSH keys
        if url.startswith("git@") or url.startswith("ssh://"):
            return url

        # Handle git:// protocol - no credentials possible
        if url.startswith("git://"):
            return url

        # Handle file paths - local repositories
        if url.startswith("/") or url.startswith("file://"):
            return url

        # Handle HTTP(S) URLs - need to strip credentials if present
        try:
            parsed = urlparse(url)

            # If it's http/https and has userinfo (credentials before @)
            if parsed.scheme in ("http", "https"):
                # Check if there's an @ in the netloc (indicates userinfo)
                if "@" in parsed.netloc:
                    # Extract just the host:port part (everything after @)
                    host_with_port = parsed.netloc.split("@")[-1]

                    # Rebuild URL without credentials
                    clean_url = urlunparse(
                        (
                            parsed.scheme,
                            host_with_port,  # Just host:port, no userinfo
                            parsed.path,
                            parsed.params,
                            parsed.query,
                            parsed.fragment,
                        )
                    )
                    return clean_url

                # No credentials present - return as-is
                return url

            # Other schemes - return as-is
            return url

        except Exception:
            # If parsing fails, safer to return None than risk leaking
            return None

    def _capture_git_info(self) -> Optional[Dict[str, Any]]:
        """Capture current git repository state.

        Captures commit hash, branch name, dirty status, and remote URL.
        Remote URLs are automatically sanitized to remove embedded credentials
        (tokens, passwords) for security.

        Returns:
            Git info dict with:
            - commit: Full commit hash
            - commit_short: Short (7-char) commit hash
            - branch: Current branch name
            - dirty: Whether there are uncommitted changes (bool)
            - remote: Repository URL (sanitized, credentials removed)
            Or None if not a git repository

        The state captured is that of the repository containing the *current
        working directory* — i.e. the code that is running — not the bundle
        directory (which is often outside the code repo, or in the cloud).

        Security:
            - Git remote URLs like "https://token@github.com/repo.git" are
              automatically cleaned to "https://github.com/repo.git"
            - Uncommitted file list is NOT captured to avoid exposing sensitive
              filenames (.env, secrets.yaml, etc.)
        """
        import subprocess
        from pathlib import Path

        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None

            # Get commit hash
            commit_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            commit = (
                commit_result.stdout.strip() if commit_result.returncode == 0 else ""
            )
            commit_short = commit[:7] if commit else ""

            # Get branch name
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            branch = (
                branch_result.stdout.strip() if branch_result.returncode == 0 else ""
            )

            # Get remote URL
            remote_result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            remote = (
                remote_result.stdout.strip() if remote_result.returncode == 0 else None
            )

            # Check for uncommitted changes (dirty flag only, no file list for security)
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            dirty = (
                bool(status_result.stdout.strip())
                if status_result.returncode == 0
                else False
            )

            git_info: Dict[str, Any] = {
                "commit": commit,
                "commit_short": commit_short,
                "branch": branch,
                "dirty": dirty,
            }

            # Sanitize remote URL to remove any embedded credentials
            if remote:
                sanitized_remote = self._sanitize_git_remote_url(remote)
                if sanitized_remote:  # Only include if sanitization succeeded
                    git_info["remote"] = sanitized_remote

            return git_info

        except Exception:
            # Git not available or error occurred
            return None

    def _capture_environment_info(self) -> Dict[str, Any]:
        """Capture Python environment information.

        Returns:
            Environment info dict with Python version, platform, packages
        """
        import platform
        import sys
        from pathlib import Path

        env_info: Dict[str, Any] = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }

        # Try to capture uv.lock hash if it exists
        try:
            import hashlib

            uv_lock = Path.cwd() / "uv.lock"
            if uv_lock.exists():
                with open(uv_lock, "rb") as f:
                    lock_hash = hashlib.md5(f.read()).hexdigest()
                env_info["uv_lock_hash"] = lock_hash
        except Exception:
            pass

        # Try to capture requirements
        try:
            requirements_file = Path.cwd() / "requirements.txt"
            if requirements_file.exists():
                env_info["requirements"] = requirements_file.read_text()
        except Exception:
            pass

        return env_info

    def _capture_execution_info(self) -> Dict[str, Any]:
        """Capture execution context for reproducibility.

        Returns:
            Execution info dict with entry point and working directory
        """
        import sys
        from pathlib import Path

        exec_info: Dict[str, Any] = {
            "working_dir": str(Path.cwd()),
        }

        # Try to capture command line that was run
        if sys.argv:
            exec_info["entry_point"] = " ".join(sys.argv)

        return exec_info

    # ==================== Snapshot Creation ====================

    def create_snapshot(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        capture_git: bool = True,
        capture_environment: bool = False,
        capture_execution: bool = False,
    ) -> Self:
        """Create a named snapshot of the current bundle state.

        A snapshot captures:
        - Current versions of all items (via item_versions dict)
        - Current metadata state (via metadata_snapshot dict)
        - Git repository state (commit, branch, dirty status) [optional]
        - Python environment (version, packages) [optional, off by default]
        - Execution context (entry point, working directory) [optional, off by default]

        After creating a snapshot, all current items are marked as being
        in that snapshot. Future overwrites will trigger copy-on-write
        to preserve the snapshot state.

        IMMUTABILITY: Snapshots freeze *owned* items — included tables, models,
        and artifacts stored in the bundle — via copy-on-write. Referenced
        (external) tables are NOT owned: a snapshot preserves the reference link,
        not the bytes, so the external content can still change. See
        :meth:`mutable_references` and :meth:`inspect_table` (which records a
        point-in-time ``source_identity``). ``get_snapshot_info`` flags any
        mutable references a snapshot contains.

        SECURITY NOTE: Environment variables (API keys, tokens, etc.) are NEVER
        captured. The capture_environment flag only captures Python version,
        platform, and package versions from uv.lock or requirements.txt.

        Args:
            name: Snapshot name (filesystem-safe, no @ symbol)
            description: Optional human-readable description
            tags: Optional list of tags for organization
            capture_git: Whether to capture git state (default: True)
            capture_environment: Whether to capture Python environment info
                like version and packages (default: False for security)
            capture_execution: Whether to capture execution context like
                entry point and working directory (default: False for security)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If snapshot name is invalid or already exists

        Examples:
            >>> folio = DataFolio('experiments/my-exp')
            >>> folio.add_table('results', df)
            >>> folio.create_snapshot('v1.0-baseline', description='Initial results')
            >>>
            >>> # Later, overwriting will preserve the snapshot
            >>> folio.add_table('results', new_df, overwrite=True)  # Creates v2
        """
        self._check_read_only()

        if self._batch_mode:
            # Inside batch() the manifest publish is deferred to batch exit,
            # but snapshots.json is not — a snapshot taken here would either
            # be silently lost or pin uncommitted state. Refuse loudly.
            raise RuntimeError(
                "create_snapshot() cannot be called inside a batch() block: "
                "the batch's items are not committed yet. Exit the batch "
                "first, then create the snapshot."
            )

        from datetime import datetime, timezone

        # Validate snapshot name
        validate_snapshot_name(name)

        # Check if snapshot already exists
        if name in self._snapshots:
            raise ValueError(f"Snapshot '{name}' already exists")

        # Capture current item versions by their stable ``version_id``. Every
        # version persisted by this build has one; it works uniformly for owned
        # items (checksummed) and external references (no checksum), so the
        # snapshot pins the exact descriptor and reopening resolves it again.
        # Legacy items lacking a version_id get one assigned now (from their
        # checksum where available) and it is persisted with this snapshot.
        item_versions: Dict[str, str] = {}
        for item_name, item_meta in self._items.items():
            version_id = item_meta.get("version_id")
            if not version_id:
                version_id = item_meta.get("checksum") or self._next_version_id(
                    item_name
                )
                item_meta["version_id"] = version_id
            item_versions[item_name] = version_id

        # Capture current metadata state
        metadata_snapshot = dict(self.metadata) if hasattr(self, "metadata") else {}

        # Build snapshot metadata
        snapshot_meta: Dict[str, Any] = {
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "item_versions": item_versions,
            "metadata_snapshot": metadata_snapshot,
        }

        # Add optional fields
        if description:
            snapshot_meta["description"] = description
        if tags:
            snapshot_meta["tags"] = tags
        else:
            snapshot_meta["tags"] = []

        # Capture context
        if capture_git:
            git_info = self._capture_git_info()
            if git_info:
                snapshot_meta["git"] = git_info

        if capture_environment:
            snapshot_meta["environment"] = self._capture_environment_info()

        if capture_execution:
            snapshot_meta["execution"] = self._capture_execution_info()

        # Update all current items to mark them as in this snapshot
        for item_name in self._items:
            item = self._items[item_name]
            if "in_snapshots" not in item:
                item["in_snapshots"] = []
            item["in_snapshots"].append(name)

        # Store snapshot
        self._snapshots[name] = snapshot_meta

        # Save snapshots.json and items.json
        self._save_snapshots()
        self._save_items()

        return self

    @property
    def snapshots(self) -> SnapshotAccessor:
        """Access snapshots in dict-like manner.

        Returns:
            SnapshotAccessor for accessing snapshots

        Examples:
            >>> # List all snapshots
            >>> for name in folio.snapshots:
            ...     print(name)
            >>>
            >>> # Access specific snapshot
            >>> snapshot = folio.snapshots['v1.0']
            >>> df = snapshot.get_table('results')
            >>> print(snapshot.metadata)
            >>>
            >>> # Check if snapshot exists
            >>> if 'v1.0' in folio.snapshots:
            ...     print("Snapshot exists")
        """
        return SnapshotAccessor(self)

    def list_snapshots(self) -> list[Dict[str, Any]]:
        """List all snapshots with their metadata.

        Returns:
            List of snapshot metadata dicts with name, timestamp, description, tags

        Examples:
            >>> snapshots = folio.list_snapshots()
            >>> for snap in snapshots:
            ...     print(f"{snap['name']}: {snap['description']}")
        """
        result = []
        for name, meta in self._snapshots.items():
            snapshot_info = {
                "name": name,
                "timestamp": meta.get("timestamp", ""),
                "description": meta.get("description"),
                "tags": meta.get("tags", []),
                "num_items": len(meta.get("item_versions", {})),
            }
            result.append(snapshot_info)

        # Sort by timestamp (newest first)
        result.sort(key=lambda x: x["timestamp"], reverse=True)
        return result

    def delete_snapshot(self, name: str, cleanup_orphans: bool = False) -> Self:
        """Delete a snapshot.

        Removes the snapshot from the registry and updates items' in_snapshots lists.
        Optionally cleans up orphaned item versions that are no longer referenced.

        Args:
            name: Snapshot name to delete
            cleanup_orphans: If True, delete item versions no longer in any snapshot

        Returns:
            Self for method chaining

        Raises:
            KeyError: If snapshot doesn't exist

        Examples:
            >>> folio.delete_snapshot('experimental-v5')
            >>> folio.delete_snapshot('old-snapshot', cleanup_orphans=True)
        """
        self._check_read_only()

        if name not in self._snapshots:
            raise KeyError(f"Snapshot '{name}' not found")

        # Remove from snapshots registry
        del self._snapshots[name]

        # Remove snapshot from all items' in_snapshots lists
        for item in self._items.values():
            if "in_snapshots" in item and name in item["in_snapshots"]:
                item["in_snapshots"].remove(name)

        # Also check snapshot versions
        for item in self._snapshot_versions:
            if "in_snapshots" in item and name in item["in_snapshots"]:
                item["in_snapshots"].remove(name)

        # Save manifests
        self._save_snapshots()
        self._save_items()

        # Optionally cleanup orphaned versions
        if cleanup_orphans:
            self.cleanup_orphaned_versions()

        return self

    def compare_snapshots(self, snapshot1: str, snapshot2: str) -> Dict[str, Any]:
        """Compare two snapshots.

        Returns a dictionary showing differences between the two snapshots including:
        - added_items: Items in snapshot2 but not snapshot1
        - removed_items: Items in snapshot1 but not snapshot2
        - modified_items: Items in both but with different versions
        - shared_items: Items in both with same version
        - metadata_changes: Metadata fields that changed (old_value, new_value)

        Args:
            snapshot1: First snapshot name
            snapshot2: Second snapshot name

        Returns:
            Dictionary with comparison results

        Raises:
            KeyError: If either snapshot doesn't exist

        Examples:
            >>> diff = folio.compare_snapshots('v1.0', 'v2.0')
            >>> print(diff['modified_items'])
            ['classifier', 'config']
            >>> print(diff['metadata_changes']['accuracy'])
            (0.89, 0.91)
        """
        if snapshot1 not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot1}' not found")
        if snapshot2 not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot2}' not found")

        snap1_meta = self._snapshots[snapshot1]
        snap2_meta = self._snapshots[snapshot2]

        snap1_items = snap1_meta.get("item_versions", {})
        snap2_items = snap2_meta.get("item_versions", {})

        # Find item differences
        snap1_names = set(snap1_items.keys())
        snap2_names = set(snap2_items.keys())

        added_items = sorted(snap2_names - snap1_names)
        removed_items = sorted(snap1_names - snap2_names)

        # Check for modified items (different versions)
        shared_names = snap1_names & snap2_names
        modified_items = []
        unchanged_items = []

        for item_name in shared_names:
            if snap1_items[item_name] != snap2_items[item_name]:
                modified_items.append(item_name)
            else:
                unchanged_items.append(item_name)

        # Compare metadata
        snap1_metadata = snap1_meta.get("metadata_snapshot", {})
        snap2_metadata = snap2_meta.get("metadata_snapshot", {})

        metadata_changes = {}
        all_metadata_keys = set(snap1_metadata.keys()) | set(snap2_metadata.keys())

        for key in all_metadata_keys:
            val1 = snap1_metadata.get(key)
            val2 = snap2_metadata.get(key)
            if val1 != val2:
                metadata_changes[key] = (val1, val2)

        return {
            "added_items": added_items,
            "removed_items": removed_items,
            "modified_items": sorted(modified_items),
            "shared_items": sorted(unchanged_items),
            "metadata_changes": metadata_changes,
        }

    def diff_from_snapshot(self, snapshot: Optional[str] = None) -> Dict[str, Any]:
        """Compare current state to a snapshot.

        This is useful for seeing what has changed since a snapshot was created,
        similar to 'git status' showing changes since last commit.

        Args:
            snapshot: Snapshot name to compare to. If None, uses most recent snapshot.

        Returns:
            Dictionary with comparison results including:
            - snapshot_name: The snapshot being compared to
            - added_items: Items in current state but not in snapshot
            - removed_items: Items in snapshot but not in current state
            - modified_items: Items in both but with different checksums/versions
            - unchanged_items: Items in both with same checksum/version
            - metadata_changes: Metadata fields that changed

        Raises:
            KeyError: If snapshot doesn't exist
            ValueError: If no snapshots exist and snapshot=None

        Examples:
            >>> # Compare to last snapshot
            >>> diff = folio.diff_from_snapshot()
            >>> print(f"Modified: {diff['modified_items']}")
            ['classifier', 'config']

            >>> # Compare to specific snapshot
            >>> diff = folio.diff_from_snapshot('v1.0')
            >>> print(f"Added since v1.0: {diff['added_items']}")
            ['new_feature']
        """
        # Get snapshot to compare to
        if snapshot is None:
            # Use most recent snapshot (list_snapshots() sorts newest-first)
            if not self._snapshots:
                raise ValueError("No snapshots exist. Create a snapshot first.")
            snapshots_list = self.list_snapshots()
            if not snapshots_list:
                raise ValueError("No snapshots exist. Create a snapshot first.")
            snapshot = snapshots_list[0]["name"]

        if snapshot not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot}' not found")

        snapshot_meta = self._snapshots[snapshot]
        snapshot_items = snapshot_meta.get("item_versions", {})

        # Map current items by name for version comparison.
        current_by_name = {item["name"]: item for item in self._items.values()}

        # Find item differences
        snapshot_names = set(snapshot_items.keys())
        current_names = set(current_by_name.keys())

        added_items = sorted(current_names - snapshot_names)
        removed_items = sorted(snapshot_names - current_names)

        # Check for modified items by version token. Snapshots record a stable
        # ``version_id`` (older snapshots recorded a checksum); an item is
        # unchanged if the recorded token still matches the current item's
        # version_id or checksum.
        shared_names = snapshot_names & current_names
        modified_items = []
        unchanged_items = []

        for item_name in shared_names:
            snapshot_token = snapshot_items[item_name]
            current = current_by_name[item_name]
            if snapshot_token in (
                current.get("version_id"),
                current.get("checksum"),
            ):
                unchanged_items.append(item_name)
            else:
                modified_items.append(item_name)

        # Compare metadata
        snapshot_metadata = snapshot_meta.get("metadata_snapshot", {})
        current_metadata = dict(self.metadata)

        metadata_changes = {}
        all_metadata_keys = set(snapshot_metadata.keys()) | set(current_metadata.keys())

        for key in all_metadata_keys:
            # Skip internal metadata fields
            if key in ("created_at", "updated_at", "_datafolio"):
                continue

            val_snapshot = snapshot_metadata.get(key)
            val_current = current_metadata.get(key)
            if val_snapshot != val_current:
                metadata_changes[key] = (val_snapshot, val_current)

        return {
            "snapshot_name": snapshot,
            "added_items": added_items,
            "removed_items": removed_items,
            "modified_items": sorted(modified_items),
            "unchanged_items": sorted(unchanged_items),
            "metadata_changes": metadata_changes,
        }

    def cleanup_orphaned_versions(self, dry_run: bool = False) -> list[str]:
        """Delete item versions not in any snapshot and not current.

        An item version is orphaned if:
        - It's not the current version of any item
        - It's not referenced by any snapshot

        Args:
            dry_run: If True, return what would be deleted without deleting

        Returns:
            List of deleted filenames (or would-be deleted if dry_run=True)

        Examples:
            >>> # See what would be deleted
            >>> orphans = folio.cleanup_orphaned_versions(dry_run=True)
            >>> print(f"Would delete {len(orphans)} files")
            >>>
            >>> # Actually delete
            >>> deleted = folio.cleanup_orphaned_versions()
            >>> print(f"Deleted {len(deleted)} orphaned versions")
        """
        from datafolio.base.registry import get_handler

        deleted_files = []

        # Find orphaned snapshot versions
        orphaned_versions = []
        for item in self._snapshot_versions:
            in_snapshots = item.get("in_snapshots", [])
            # If not in any snapshot, it's orphaned
            if not in_snapshots:
                orphaned_versions.append(item)

        # Delete orphaned versions
        for item in orphaned_versions:
            filename = item.get("filename")

            if not dry_run and filename:
                # Delete the physical file (unless another descriptor still
                # shares it — a metadata-only copy-on-write can leave the
                # current item pointing at this same payload). Don't use
                # handler.delete() as that would delete from _items.
                self._delete_payload_if_unshared(item)

                # Remove from snapshot_versions list
                self._snapshot_versions.remove(item)

            if filename:
                deleted_files.append(filename)

        # Save updated manifest if we deleted anything
        if not dry_run and deleted_files:
            self._save_items()

        return deleted_files

    def restore_snapshot(self, snapshot: str, confirm: bool = False) -> Self:
        """Restore working state to snapshot (DESTRUCTIVE).

        This operation:
        - Replaces current metadata with snapshot metadata
        - Sets current item versions to match snapshot
        - Restores items deleted after the snapshot was taken
        - Removes items added after snapshot
        - Does NOT delete the snapshot itself

        WARNING: This is a destructive operation that overwrites current state.

        Args:
            snapshot: Snapshot name to restore
            confirm: Must be True to proceed (safety check)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If confirm=False
            KeyError: If snapshot doesn't exist

        Examples:
            >>> folio.restore_snapshot('v1.0', confirm=True)
            >>> # Working state now matches v1.0 snapshot
        """
        self._check_read_only()

        if not confirm:
            raise ValueError(
                "restore_snapshot requires confirm=True as this is a destructive operation"
            )

        if snapshot not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot}' not found")

        snap_meta = self._snapshots[snapshot]
        snap_items = snap_meta.get("item_versions", {})
        snap_metadata = snap_meta.get("metadata_snapshot", {})

        # Every version already lives in its own payload file, so restoring is
        # pure manifest surgery: repoint each logical name at the descriptor
        # the snapshot pinned. No payload is ever copied or moved, so an
        # interrupted restore can't lose data.
        with self._mutation_guard():
            # Resolve every pinned descriptor up front — fail loudly BEFORE
            # touching any state if the snapshot is unrestorable.
            pinned_by_name: Dict[str, Dict[str, Any]] = {}
            for item_name, version_token in snap_items.items():
                pinned = self._find_item_by_checksum(item_name, version_token)
                if pinned is None:
                    raise KeyError(
                        f"Cannot restore snapshot '{snapshot}': it pins version "
                        f"'{version_token}' of item '{item_name}', but that "
                        f"version is no longer in the manifest (was it removed "
                        f"by cleanup_orphaned_versions?)."
                    )
                pinned_by_name[item_name] = pinned

            # Restore metadata
            self.metadata.clear()
            self.metadata.update(snap_metadata)

            # Remove items not in the snapshot (added after it was taken). A
            # version pinned by some OTHER snapshot is preserved as a snapshot
            # version; an unpinned one is gone for good (this is the
            # documented destructive part).
            for item_name in set(self._items) - set(snap_items):
                item = self._items[item_name]
                if item.get("in_snapshots"):
                    self._handle_copy_on_write(item_name)
                else:
                    self._delete_payload_if_unshared(item)
                del self._items[item_name]

            # Repoint every snapshot item at its pinned descriptor.
            for item_name, pinned in pinned_by_name.items():
                current = self._items.get(item_name)
                if current is pinned:
                    continue  # already the working version

                # Displace the current version (if any): preserve it when a
                # snapshot pins it, otherwise drop it and its payload.
                if current is not None:
                    if current.get("in_snapshots"):
                        self._handle_copy_on_write(item_name)
                    else:
                        self._delete_payload_if_unshared(current)
                    del self._items[item_name]

                # Promote the pinned descriptor back to current. It stays
                # listed in the snapshots that pin it.
                if pinned in self._snapshot_versions:
                    self._snapshot_versions.remove(pinned)
                pinned["is_current"] = True
                self._items[item_name] = pinned

            # Save updated state
            self._save_items()

        return self

    @classmethod
    def load_snapshot(
        cls,
        bundle_dir: Union[str, Path],
        snapshot: str,
    ) -> "DataFolio":
        """Load a DataFolio in snapshot state.

        Creates a DataFolio instance configured to access items and metadata
        as they existed at snapshot time. Snapshots are always read-only
        to preserve snapshot immutability.

        Args:
            bundle_dir: Path to bundle directory
            snapshot: Snapshot name to load

        Returns:
            Read-only DataFolio instance in snapshot state

        Raises:
            KeyError: If snapshot doesn't exist

        Examples:
            Load snapshot for inspection:
            >>> paper = DataFolio.load_snapshot('research/exp', 'paper-v1')
            >>> model = paper.get_model('classifier')
            >>> print(paper.metadata['accuracy'])
            >>> paper.add_table('new', df)  # Error: snapshots are always read-only

            Compare multiple snapshots:
            >>> v1 = DataFolio.load_snapshot('path', 'v1.0')
            >>> v2 = DataFolio.load_snapshot('path', 'v2.0')
            >>> print(f"v1: {v1.metadata['accuracy']}, v2: {v2.metadata['accuracy']}")
        """
        # Load folio as read-only (snapshots are always immutable)
        folio = cls(bundle_dir, read_only=True)

        # Verify snapshot exists
        if snapshot not in folio._snapshots:
            raise KeyError(f"Snapshot '{snapshot}' not found in bundle")

        # Set snapshot mode
        folio._in_snapshot_mode = True
        folio._loaded_snapshot = snapshot

        # Get snapshot metadata
        snapshot_meta = folio._snapshots[snapshot]

        # Replace current metadata with snapshot metadata
        # Use dict methods directly to bypass read-only checks during setup
        snapshot_metadata = snapshot_meta.get("metadata_snapshot", {})
        dict.clear(folio.metadata)
        dict.update(folio.metadata, snapshot_metadata)

        # Get snapshot item versions (using checksums as version identifiers)
        snapshot_versions = snapshot_meta.get("item_versions", {})

        # For each item in snapshot, point to that version
        for item_name, checksum in snapshot_versions.items():
            # Find the item with this name and checksum
            item = folio._find_item_by_checksum(item_name, checksum)
            if item:
                folio._items[item_name] = item

        # Remove items not in snapshot
        current_items = list(folio._items.keys())
        for item_name in current_items:
            if item_name not in snapshot_versions:
                del folio._items[item_name]

        return folio

    def get_snapshot(self, snapshot: str) -> "DataFolio":
        """Get a snapshot from this folio as a new DataFolio instance.

        Convenience method for loading a snapshot when you already have a folio.
        Equivalent to DataFolio.load_snapshot(self._bundle_dir, snapshot).
        Snapshots are always read-only to preserve immutability.

        Args:
            snapshot: Snapshot name to load

        Returns:
            Read-only DataFolio instance in snapshot state

        Raises:
            KeyError: If snapshot doesn't exist

        Examples:
            >>> folio = DataFolio('experiments/classifier')
            >>> baseline = folio.get_snapshot('v1.0-baseline')
            >>> assert baseline.metadata['accuracy'] == 0.89
            >>> assert baseline.read_only  # Snapshots are always read-only
            >>>
            >>> # Compare current state to snapshot
            >>> current_acc = folio.metadata['accuracy']
            >>> baseline_acc = baseline.metadata['accuracy']
            >>> print(f"Improvement: {current_acc - baseline_acc:.2%}")
        """
        return self.__class__.load_snapshot(self._bundle_dir, snapshot)

    def export_snapshot(
        self,
        snapshot: str,
        target_path: Union[str, Path],
        *,
        include_snapshot_metadata: bool = True,
    ) -> "DataFolio":
        """Export a snapshot to a clean, standalone bundle.

        Creates a new DataFolio bundle containing only the items and metadata
        from the specified snapshot. This is useful for:
        - Sharing a specific snapshot with collaborators
        - Creating a clean bundle for deployment
        - Starting fresh without version history

        Args:
            snapshot: Name of snapshot to export
            target_path: Path for new bundle (must not exist)
            include_snapshot_metadata: If True, adds snapshot info to new bundle's
                metadata under '_source_snapshot' key (default: True)

        Returns:
            New DataFolio instance at target_path

        Raises:
            KeyError: If snapshot doesn't exist
            ValueError: If target_path already exists

        Examples:
            Export a baseline snapshot for sharing:
            >>> folio = DataFolio('experiments/classifier')
            >>> baseline = folio.export_snapshot('v1.0-baseline', 'shared/baseline')
            >>> # New bundle contains only v1.0-baseline state, no history

            Export for deployment:
            >>> production = folio.export_snapshot('production-v2', 'deploy/v2')
            >>> # Clean bundle ready for deployment

            Export without metadata reference:
            >>> clean = folio.export_snapshot(
            ...     'v1.0',
            ...     'clean-export',
            ...     include_snapshot_metadata=False
            ... )
        """
        from pathlib import Path

        target_path = Path(target_path)

        # Check target doesn't exist
        if target_path.exists():
            raise ValueError(f"Target path already exists: {target_path}")

        # Verify snapshot exists
        if snapshot not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot}' not found")

        # Load the snapshot
        snapshot_folio = self.get_snapshot(snapshot)

        # Create new empty bundle
        new_folio = self.__class__(target_path)

        # Copy metadata from snapshot
        snapshot_metadata = dict(snapshot_folio.metadata)

        # Remove internal metadata
        for key in ["created_at", "updated_at", "_datafolio"]:
            snapshot_metadata.pop(key, None)

        # Add snapshot metadata to new bundle
        if include_snapshot_metadata:
            snapshot_info = self.get_snapshot_info(snapshot)
            snapshot_metadata["_source_snapshot"] = {
                "name": snapshot,
                "source_bundle": str(Path(self._bundle_dir).resolve()),
                "timestamp": snapshot_info["timestamp"],
                "description": snapshot_info.get("description"),
                "tags": snapshot_info.get("tags"),
            }

        # Update new folio's metadata
        new_folio.metadata.update(snapshot_metadata)

        # Copy all items from the snapshot by copying their payload FILES
        # directly (never a deserialize/re-serialize round-trip): descriptors
        # — including descriptions, lineage, and type-specific metadata — are
        # carried over verbatim, external references stay references (their
        # data is not owned and is never copied), and bytes are preserved
        # exactly.
        import copy as _copy

        from datafolio.storage import get_storage_directory

        for item_name, item_meta in snapshot_folio._items.items():
            new_item = _copy.deepcopy(dict(item_meta))
            new_item["in_snapshots"] = []
            new_item["is_current"] = True

            filename = item_meta.get("filename")
            if filename:
                subdir = get_storage_directory(item_meta["item_type"])
                src = self._storage.join_paths(self._bundle_dir, subdir, filename)
                dst = self._storage.join_paths(new_folio._bundle_dir, subdir, filename)
                self._copy_payload_file(src, dst)

            new_folio._items[item_name] = new_item

        # Save items manifest
        new_folio._save_items()

        return new_folio

    def _find_item_by_checksum(
        self, name: str, version_token: str
    ) -> Optional[Dict[str, Any]]:
        """Find the item version a snapshot pinned for ``name``.

        Snapshots record a stable ``version_id`` per item. Older snapshots
        recorded a checksum instead, so this matches either — the exact version
        recorded at snapshot time is returned, whether it is still the current
        item or has since been superseded and moved to ``_snapshot_versions``.

        Args:
            name: Item name.
            version_token: The ``version_id`` (or legacy checksum) recorded in
                the snapshot's ``item_versions``.

        Returns:
            Item metadata dict or None if not found.
        """

        def _matches(item: Dict[str, Any]) -> bool:
            return (
                item.get("version_id") == version_token
                or item.get("checksum") == version_token
            )

        if name in self._items and _matches(self._items[name]):
            return self._items[name]

        for item in self._snapshot_versions:
            if item.get("name") == name and _matches(item):
                return item

        return None

    def get_snapshot_info(self, snapshot: str) -> Dict[str, Any]:
        """Get detailed information about a snapshot.

        Returns the full snapshot metadata including item versions, metadata state,
        git info, environment info, and execution context.

        Args:
            snapshot: Snapshot name

        Returns:
            Dictionary containing all snapshot metadata

        Raises:
            KeyError: If snapshot doesn't exist

        Examples:
            >>> info = folio.get_snapshot_info('v1.0')
            >>> print(info['description'])
            'Baseline model'
            >>> print(info['git']['commit'])
            'a3f2b8c'
            >>> print(info['metadata_snapshot']['accuracy'])
            0.89
        """
        if snapshot not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot}' not found")

        # Return a copy of the snapshot metadata, surfacing any mutable external
        # references it contains (their bytes are not owned/frozen — see
        # mutable_references()).
        info = dict(self._snapshots[snapshot])
        versions = info.get("item_versions", {}) or {}
        type_map = {
            it.get("name"): it.get("item_type")
            for it in list(self._items.values()) + self._snapshot_versions
        }
        mutable = [
            name for name in versions if type_map.get(name) == "referenced_table"
        ]
        if mutable:
            info["mutable_references"] = mutable
            info["mutable_reference_warning"] = (
                "This snapshot references external tables whose content is not "
                "owned by the bundle and may have changed since the snapshot was "
                "taken. Snapshots preserve the reference link, not the bytes."
            )
        return info

    def mutable_references(self) -> list[str]:
        """Names of referenced tables whose external content is not owned.

        Referenced tables link to external data that datafolio never copies, so
        the bytes at the referenced path can change over time. Snapshots
        preserve the *link*, not the content — only owned items (included
        tables, models, artifacts) are truly immutable in a snapshot. Use
        :meth:`inspect_table` to record a point-in-time ``source_identity`` for
        detecting drift.

        Returns:
            List of referenced-table names in the current working set.
        """
        self._refresh_if_needed()
        return [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "referenced_table"
        ]

    def reproduce_instructions(self, snapshot: Optional[str] = None) -> str:
        """Generate human-readable instructions to reproduce a snapshot.

        Creates a formatted guide with steps to:
        1. Restore code (git checkout)
        2. Restore environment (Python version, dependencies)
        3. Run execution command
        4. Verify expected results

        Args:
            snapshot: Snapshot name. If None and no snapshots exist, raises error.

        Returns:
            Formatted string with reproduction steps

        Raises:
            KeyError: If snapshot doesn't exist
            ValueError: If snapshot is None and no snapshots exist

        Examples:
            >>> instructions = folio.reproduce_instructions('v1.0')
            >>> print(instructions)
            To reproduce snapshot 'v1.0':

            1. Restore code:
               git checkout a3f2b8c

            2. Restore environment:
               python --version  # Should be 3.11.5
               uv sync

            3. Run training:
               python train.py --config config.json

            4. Expected results:
               - accuracy: 0.89
               - f1_score: 0.87
        """
        if snapshot is None:
            if not self._snapshots:
                raise ValueError(
                    "No snapshot specified and no snapshots exist. "
                    "Create a snapshot first or specify a snapshot name."
                )
            # Use the most recent snapshot
            snapshots = self.list_snapshots()
            if snapshots:
                snapshot = snapshots[0]["name"]
            else:
                raise ValueError("No snapshots available")

        if snapshot not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot}' not found")

        snap_meta = self._snapshots[snapshot]
        lines = []

        # Header
        lines.append(f"To reproduce snapshot '{snapshot}':")
        if snap_meta.get("description"):
            lines.append(f"Description: {snap_meta['description']}")
        lines.append("")

        step = 1

        # Git restore
        if "git" in snap_meta:
            git_info = snap_meta["git"]
            lines.append(f"{step}. Restore code:")
            if git_info.get("remote"):
                lines.append(f"   git clone {git_info['remote']}")
                lines.append(f"   cd <repository>")
            commit = git_info.get("commit_short") or git_info.get("commit", "")[:7]
            lines.append(f"   git checkout {commit}")
            if git_info.get("dirty"):
                lines.append("   Note: Original snapshot had uncommitted changes")
            lines.append("")
            step += 1

        # Environment restore
        if "environment" in snap_meta:
            env_info = snap_meta["environment"]
            lines.append(f"{step}. Restore environment:")
            if "python_version" in env_info:
                py_ver = env_info["python_version"]
                lines.append(f"   python --version  # Should be {py_ver}")
            lines.append("   uv sync  # Or: pip install -r requirements.txt")
            lines.append("")
            step += 1

        # Execution
        if "execution" in snap_meta:
            exec_info = snap_meta["execution"]
            lines.append(f"{step}. Run execution:")
            if "entry_point" in exec_info:
                lines.append(f"   {exec_info['entry_point']}")
            if "working_dir" in exec_info:
                lines.append(f"   # Working directory: {exec_info['working_dir']}")
            lines.append("")
            step += 1

        # Expected results
        if "metadata_snapshot" in snap_meta:
            metadata = snap_meta["metadata_snapshot"]
            if metadata:
                lines.append(f"{step}. Expected results:")
                # Show up to 5 metadata fields
                for i, (key, value) in enumerate(list(metadata.items())[:5]):
                    # Skip internal fields
                    if key in ("created_at", "updated_at"):
                        continue
                    lines.append(f"   - {key}: {value}")
                if len(metadata) > 5:
                    lines.append(f"   ... and {len(metadata) - 5} more fields")
                lines.append("")

        return "\n".join(lines)

    # ==================== Snapshot Version Management ====================

    def _is_in_snapshots(self, name: str) -> bool:
        """Check if an item is referenced by any snapshots.

        Args:
            name: Item name to check

        Returns:
            True if item exists and is referenced by at least one snapshot
        """
        if name not in self._items:
            return False

        item = self._items[name]
        in_snapshots = item.get("in_snapshots", [])
        return len(in_snapshots) > 0

    def _handle_copy_on_write(self, name: str) -> None:
        """Preserve a snapshotted item's version before it is overwritten.

        With versioned payload filenames (``<name>--r<rev><ext>``) each version
        already lives in its own file, so overwriting never touches the bytes a
        snapshot depends on — there is nothing to rename. This method simply
        marks the outgoing version non-current and moves its descriptor to
        ``_snapshot_versions`` (uniformly for every item type, including
        external references, whose descriptor must be preserved exactly as
        recorded at snapshot time). The caller then installs the new version as
        current under the same logical name.

        Args:
            name: Item name being overwritten.
        """
        if not self._is_in_snapshots(name):
            return
        old_item = self._items[name]
        old_item["is_current"] = False
        # Ensure the preserved version is addressable by a stable id so
        # snapshots can pin it unambiguously across reopen.
        if not old_item.get("version_id"):
            old_item["version_id"] = old_item.get("checksum") or self._next_version_id(
                name
            )
        self._snapshot_versions.append(old_item)
        # Caller installs the replacement in _items[name].

    # ==================== Auto-Refresh Methods ====================

    def _check_if_stale(self) -> bool:
        """Check if the in-memory state is stale compared to disk/cloud.

        Returns:
            True if manifests should be reloaded, False otherwise
        """
        if not self._auto_refresh_enabled:
            return False

        # Read the remote metadata.json to get its updated_at timestamp
        metadata_path = self._storage.join_paths(self._bundle_dir, METADATA_FILE)
        if not self._storage.exists(metadata_path):
            # Metadata file doesn't exist - nothing to refresh
            return False

        try:
            remote_metadata = self._storage.read_json(metadata_path)
            remote_updated_at = remote_metadata.get("updated_at")
            local_updated_at = self.metadata.get("updated_at")

            # If either is missing, can't compare - assume fresh
            if remote_updated_at is None or local_updated_at is None:
                return False

            # Compare timestamps - if different, we're stale
            return remote_updated_at != local_updated_at

        except Exception:
            # If we can't read/parse metadata, assume fresh to avoid errors
            return False

    def _refresh_if_needed(self) -> None:
        """Refresh manifests from disk/cloud if they've been updated externally."""
        # Skip refresh if we're in the middle of a save operation
        if self._in_save_operation:
            return
        # A snapshot-mode folio is pinned to the versions recorded at snapshot
        # time; auto-refreshing would reload the *current* items and silently
        # drop the snapshot view. Never refresh in snapshot mode.
        if self._in_snapshot_mode:
            return
        if self._check_if_stale():
            self.refresh()

    def refresh(self) -> Self:
        """Explicitly refresh manifests from disk/cloud.

        This reloads items.json and metadata.json from the bundle directory,
        syncing the in-memory state with any external updates.

        Useful when working with multiple DataFolio instances pointing to
        the same bundle, or when the bundle is updated by another process.

        Returns:
            Self for method chaining

        Examples:
            Explicit refresh after external update:
            >>> folio1 = DataFolio('experiments/shared')
            >>> folio2 = DataFolio('experiments/shared')
            >>> folio1.add_table('results', df)
            >>> folio2.refresh()  # Manually sync
            >>> assert 'results' in folio2.list_contents()['included_tables']

            Auto-refresh (happens automatically):
            >>> folio1.add_table('results', df)
            >>> # folio2 auto-refreshes on next read operation
            >>> assert 'results' in folio2.list_contents()['included_tables']
        """
        # Reload manifests from disk/cloud
        self._load_manifests()

        # Sync the MetadataDict with new values
        if hasattr(self, "metadata") and isinstance(self.metadata, MetadataDict):
            # Update existing MetadataDict without triggering saves
            # Use super() to bypass auto-save behavior
            super(MetadataDict, self.metadata).clear()
            super(MetadataDict, self.metadata).update(self._metadata_raw)
        else:
            # Initial creation (shouldn't happen in refresh, but defensive)
            self.metadata = MetadataDict(self, **self._metadata_raw)

        return self

    @contextlib.contextmanager
    def batch(self):
        """Context manager for batch operations.

        Delays saving items.json until the context exits. This is useful
        when adding many items at once to avoid repeated disk I/O.

        The mutation guard (local write lock + stale-writer check) is held for
        the *entire* batch, so the whole batch either commits or is rejected as
        a unit — a stale writer is caught before any payload is written, and no
        other writer can interleave. Obsolete payloads replaced during the batch
        are deleted only after the single final manifest publish.

        Examples:
            >>> with folio.batch():
            ...     for i in range(100):
            ...         folio.add_numpy(f'array_{i}', arr)
            # items.json saved once at end of block
        """
        with self._mutation_guard():
            self._batch_mode = True
            try:
                yield
            finally:
                self._batch_mode = False
                self._save_items()
                # Flush payload deletions deferred during the batch (now that
                # the manifest pointing at the new payloads is published).
                pending, self._pending_obsolete_payloads = (
                    self._pending_obsolete_payloads,
                    [],
                )
                for old_item in pending:
                    self._obsolete_payload_after_commit(old_item)

    def validate(self) -> Dict[str, bool]:
        """Validate existence and integrity of all items.

        Checks if:
        1. Included items exist in the bundle
        2. Referenced items exist at their external path
        3. Checksums match (for included single files)

        Returns:
            Dict mapping item names to validation status (True if valid)

        Examples:
            >>> status = folio.validate()
            >>> if not all(status.values()):
            ...     print("Bundle corrupted!")
        """
        results = {}
        for name, item in self._items.items():
            item_type = item.get("item_type")
            is_valid = False

            if item_type == "referenced_table":
                # For references, just check existence (resolve relative paths
                # against the bundle so portable references validate correctly).
                path = item.get("path")
                if path:
                    is_valid = self._storage.exists(self._resolve_reference_path(path))
            elif "filename" in item:
                # For included items, check existence in bundle
                # Get handler to find subdir
                handler = get_registry().get(item_type)
                subdir = handler.get_storage_subdir()
                filepath = self._storage.join_paths(
                    self._bundle_dir, subdir, item["filename"]
                )
                is_valid = self._storage.exists(filepath)

                # Check checksum if available and file exists
                if is_valid and "checksum" in item:
                    current_checksum = self._storage.calculate_checksum(filepath)
                    if current_checksum != item["checksum"]:
                        is_valid = False

            results[name] = is_valid

        return results

    def is_valid(self) -> bool:
        """Check if the entire bundle is valid.

        Convenience method that runs validate() and returns True only if
        all items pass validation.

        Returns:
            True if all items are valid, False otherwise

        Examples:
            >>> if not folio.is_valid():
            ...     print("Bundle corrupted!")
        """
        return all(self.validate().values())

    # ==================== Public API Methods ====================

    def list_contents(self, include_archived: bool = False) -> Dict[str, list[str]]:
        """List all contents in the DataFolio.

        Args:
            include_archived: If True, include archived (hidden) items in the results.
                Defaults to False so archived items are hidden from normal views.

        Returns:
            Dictionary with keys 'referenced_tables', 'included_tables', 'numpy_arrays',
            'json_data', 'timestamps', 'models', and 'artifacts',
            each containing a list of names

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.reference_table('data1', path='s3://bucket/data.parquet')
            >>> folio.add_numpy('embeddings', np.array([1, 2, 3]))
            >>> folio.list_contents()
            {'referenced_tables': ['data1'], 'included_tables': [], 'numpy_arrays': ['embeddings'],
             'json_data': [], 'timestamps': [], 'models': [], 'artifacts': []}
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        def _visible(item: Dict[str, Any]) -> bool:
            return include_archived or not item.get("archived", False)

        # Filter items by type
        referenced_tables = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "referenced_table" and _visible(item)
        ]
        included_tables = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "included_table" and _visible(item)
        ]
        numpy_arrays = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "numpy_array" and _visible(item)
        ]
        json_data = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "json_data" and _visible(item)
        ]
        timestamps = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "timestamp" and _visible(item)
        ]
        models = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "model" and _visible(item)
        ]
        artifacts = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "artifact" and _visible(item)
        ]

        return {
            "referenced_tables": referenced_tables,
            "included_tables": included_tables,
            "numpy_arrays": numpy_arrays,
            "json_data": json_data,
            "timestamps": timestamps,
            "models": models,
            "artifacts": artifacts,
        }

    @property
    def tables(self) -> list[str]:
        """Get list of all table names (both referenced and included).

        Returns:
            List of all table names (strings)

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.reference_table('training', path='s3://bucket/data.parquet')
            >>> folio.add_table('results', df)
            >>> folio.tables
            ['training', 'results']
        """
        return [
            name
            for name, item in self._items.items()
            if item.get("item_type") in ("referenced_table", "included_table")
        ]

    @property
    def models(self) -> list[str]:
        """Get list of model names.

        Returns:
            List of model names (strings)

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.add_model('classifier', model)
            >>> folio.models
            ['classifier']
        """
        return [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "model"
        ]

    @property
    def artifacts(self) -> list[str]:
        """Get list of artifact names.

        Returns:
            List of artifact names (strings)

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.add_artifact('plot', 'plot.png')
            >>> folio.artifacts
            ['plot']
        """
        return [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "artifact"
        ]

    @property
    def data(self) -> DataAccessor:
        """Access items with autocomplete support.

        Returns:
            DataAccessor that provides attribute-style and dictionary-style
            access to all items with autocomplete in IPython/Jupyter.

        Note:
            Autocomplete suggestions are automatically updated when items are
            added or deleted. However, JupyterLab's autocomplete may cache
            results, so if a newly added item doesn't appear, try re-evaluating
            the cell or use dictionary-style access: folio.data['item_name'].

        Examples:
            Attribute-style access:
            >>> df = folio.data.results.content
            >>> desc = folio.data.results.description
            >>> inputs = folio.data.results.inputs

            Dictionary-style access:
            >>> df = folio.data['results'].content
            >>> model = folio.data['classifier'].content

            Artifacts return file path:
            >>> with open(folio.data.plot.content, 'rb') as f:
            ...     img = f.read()

            Autocomplete (in IPython/Jupyter):
            >>> folio.data.<TAB>  # Shows: results, classifier, embeddings, ...

            Autocomplete updates automatically:
            >>> folio.add_table('new_data', df)
            >>> folio.data.new_data.content  # Autocompletes immediately
        """
        # Sync items in case they changed since initialization
        self._data_accessor._sync_items()
        return self._data_accessor

    @property
    def read_only(self) -> bool:
        """Check if folio is in read-only mode.

        Returns:
            True if folio is read-only, False otherwise

        Examples:
            >>> folio = DataFolio('path', read_only=True)
            >>> folio.read_only
            True
        """
        return self._read_only

    @property
    def in_snapshot_mode(self) -> bool:
        """Check if folio was loaded from a snapshot.

        Returns:
            True if loaded via load_snapshot(), False otherwise

        Examples:
            >>> snapshot = DataFolio.load_snapshot('path', 'v1.0')
            >>> snapshot.in_snapshot_mode
            True
        """
        return self._in_snapshot_mode

    @property
    def loaded_snapshot(self) -> Optional[str]:
        """Get name of loaded snapshot, or None.

        Returns:
            Snapshot name if loaded via load_snapshot(), None otherwise

        Examples:
            >>> snapshot = DataFolio.load_snapshot('path', 'v1.0')
            >>> snapshot.loaded_snapshot
            'v1.0'
        """
        return self._loaded_snapshot

    @property
    def path(self) -> str:
        """Get the absolute path to the bundle directory.

        Returns absolute local path for local bundles, or the full cloud path
        for cloud bundles (e.g., s3://bucket/path).

        Returns:
            Absolute path string

        Examples:
            >>> folio = DataFolio('experiments/test')
            >>> print(folio.path)
            '/absolute/path/to/experiments/test'

            >>> folio = DataFolio('s3://bucket/experiments/test')
            >>> print(folio.path)
            's3://bucket/experiments/test'
        """

        # Cloud paths are returned as-is
        if is_cloud_path(self._bundle_dir):
            return self._bundle_dir

        # For local paths, return absolute filesystem path (not file:// URL)
        return str(Path(self._bundle_dir).resolve())

    def __repr__(self) -> str:
        """Return string representation of DataFolio."""
        contents = self.list_contents()
        total_items = sum(len(v) for v in contents.values())
        snapshot_count = len(self._snapshots)

        repr_str = f"DataFolio(bundle_dir='{self._bundle_dir}', items={total_items}, snapshots={snapshot_count})"

        if self._read_only:
            repr_str += " [READ-ONLY]"

        if self._in_snapshot_mode and self._loaded_snapshot:
            repr_str += f" [snapshot: {self._loaded_snapshot}]"

        return repr_str

    def describe(
        self,
        pattern: Optional[str] = None,
        return_string: bool = False,
        show_empty: bool = False,
        max_metadata_fields: int = 10,
        snapshot: Optional[str] = None,
        include_archived: bool = False,
        show_paths: bool = False,
    ) -> Optional[str]:
        """Generate a human-readable description of all items in the bundle.

        Includes lineage information showing inputs and dependencies.

        Args:
            pattern: Optional glob pattern to filter items by name (e.g. 'examples/*',
                '*/weights'). Uses fnmatch rules — '*' matches any characters including '/'.
            return_string: If True, return as string instead of printing
            show_empty: If True, show empty sections
            max_metadata_fields: Maximum metadata fields to show
            snapshot: Optional snapshot name to describe instead of the full bundle
            include_archived: If True, show archived (hidden) items. Defaults to False.
            show_paths: If True, show the file path for each item. Especially useful
                for cloud-hosted folios where paths can be shared with collaborators
                who don't use datafolio.

        Returns:
            None if return_string=False, otherwise the description string

        Examples:
            >>> folio.describe()  # Show full bundle
            >>> folio.describe('examples/*')  # Show only items under 'examples/'
            >>> folio.describe(snapshot='v1.0')  # Show specific snapshot
            >>> folio.describe(include_archived=True)  # Show archived items too
            >>> folio.describe(show_paths=True)  # Show file paths for sharing

        See DisplayFormatter.describe() for full documentation.
        """
        # Build header for snapshot mode
        header_lines = []

        if self._in_snapshot_mode and self._loaded_snapshot:
            snapshot_meta = self._snapshots.get(self._loaded_snapshot, {})
            header_lines.append(f"Snapshot: {self._loaded_snapshot}")
            timestamp = snapshot_meta.get("timestamp", "")
            if timestamp:
                header_lines.append(f"Created: {timestamp}")
            desc = snapshot_meta.get("description", "")
            if desc:
                header_lines.append(f"Description: {desc}")
            tags = snapshot_meta.get("tags", [])
            if tags:
                header_lines.append(f"Tags: {', '.join(tags)}")
            header_lines.append("")  # Blank line

        if self._read_only:
            header_lines.append("[READ-ONLY MODE]")
            header_lines.append("")  # Blank line

        # Get main description from formatter
        formatter = DisplayFormatter(self)
        main_description = formatter.describe(
            return_string=True,  # Always get as string so we can prepend header
            show_empty=show_empty,
            max_metadata_fields=max_metadata_fields,
            snapshot=snapshot,
            pattern=pattern,
            include_archived=include_archived,
            show_paths=show_paths,
        )

        # Combine header and main description
        if header_lines:
            full_description = "\n".join(header_lines) + main_description
        else:
            full_description = main_description

        if return_string:
            return full_description
        else:
            print(full_description)
            return None

    def reference_table(
        self,
        name: str,
        path: Union[str, Path],
        table_format: str = "parquet",
        num_rows: Optional[int] = None,
        version: Optional[int] = None,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        code: Optional[str] = None,
        overwrite: bool = False,
        allow_full_load: bool = False,
        polars_only: Optional[bool] = None,
    ) -> Self:
        """Add a reference to an external table (not copied to bundle).

        This is a cheap, offline manifest operation and performs **no remote
        I/O**: it does not stat the object, read its schema, count rows, or
        verify existence. Linking a private or currently-unreachable URI is
        therefore fast and never blocks on network/credentials. Use
        :meth:`inspect_table` to enrich the entry with schema/size/identity, and
        :meth:`validate` to check existence.

        Writes immediately to items.json. Behaves like :meth:`add_table` with
        respect to existing names: overwriting requires ``overwrite=True``, and
        replacing a snapshotted name triggers copy-on-write.

        Args:
            name: Unique name for this table
            path: Path to the table (local or cloud)
            table_format: Format of the table ('parquet' (canonical) or 'csv')
            num_rows: Optional number of rows (overrides inferred value)
            version: Optional source version number recorded in the manifest
            description: Optional description
            inputs: Optional list of items this was derived from
            code: Optional code snippet that created this
            overwrite: If True, allow replacing an existing table (default: False)
            allow_full_load: If True, this reference bypasses the folio's
                ``max_eager_bytes`` guard on eager ``get_table`` reads.
            polars_only: If True, this reference is readable only lazily / via
                polars (``get_lazy`` / ``frame='polars'``); eager pandas reads
                raise a clear error. If None (default), inferred from the
                layout: a sharded/partitioned directory dataset is polars-only.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists (and overwrite=False) or format is invalid

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.reference_table(
            ...     'raw_data',
            ...     path='s3://bucket/data.parquet',
            ...     table_format='parquet',
            ...     num_rows=1_000_000
            ... )
        """
        self._check_read_only()

        # Validate item name
        validate_item_name(name)

        validate_table_format(table_format)

        # Overwriting an existing, non-snapshotted item requires overwrite=True.
        if name in self._items and not self._is_in_snapshots(name) and not overwrite:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. "
                f"Use overwrite=True to replace it."
            )

        # A reference owns no payload, so there is no versioned file to write —
        # but it still needs the full guarded, copy-on-write, description-
        # preserving lifecycle (and a stable version_id so snapshots can pin the
        # exact descriptor recorded at snapshot time).
        registry = get_registry()
        handler = registry.get("referenced_table")

        with self._mutation_guard():
            if self._is_in_snapshots(name):
                self._handle_copy_on_write(name)

            metadata = handler.add(
                self,
                name,
                str(path),
                description=description,
                inputs=inputs,
                table_format=table_format,
                allow_full_load=allow_full_load,
                polars_only=polars_only,
            )

            # Add extra fields not handled by base handler
            if num_rows is not None:
                metadata["num_rows"] = num_rows
            if version is not None:
                metadata["version"] = version
            if code is not None:
                metadata["code"] = code

            metadata["version_id"] = self._next_version_id(name)
            self._apply_description(name, metadata, description)
            metadata.setdefault("in_snapshots", [])
            metadata.setdefault("is_current", True)

            self._items[name] = metadata
            self._save_items()

        return self

    def add_table(
        self,
        name: str,
        data: Any,  # pandas or Polars DataFrame
        description: Optional[str] = None,
        overwrite: bool = False,
        inputs: Optional[list[str]] = None,
        models: Optional[list[str]] = None,
        code: Optional[str] = None,
    ) -> Self:
        """Add a table to be included in the bundle.

        Writes immediately to tables/ directory and updates items.json.

        Args:
            name: Unique name for this table
            data: pandas or Polars DataFrame to include
            description: Optional description
            overwrite: If True, allow overwriting existing table (default: False)
            inputs: Optional list of table names used to create this table
            models: Optional list of model names used to create this table
            code: Optional code snippet that created this table

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists and overwrite=False
            TypeError: If data is not a DataFrame

        Examples:
            >>> import pandas as pd
            >>> folio = DataFolio('experiments', prefix='test')
            >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> folio.add_table('summary', df)
            >>> # With lineage
            >>> pred_df = pd.DataFrame({'pred': [0, 1, 0]})
            >>> folio.add_table('predictions', pred_df,
            ...     inputs=['test_data'],
            ...     models=['classifier'],
            ...     code='pred = model.predict(X_test)')
        """
        self._check_read_only()

        # Validate item name

        validate_item_name(name)

        # Overwriting an existing, non-snapshotted item requires overwrite=True.
        # A snapshotted item is always preserved via copy-on-write inside the
        # guarded commit below.
        if name in self._items and not self._is_in_snapshots(name) and not overwrite:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. "
                f"Use overwrite=True to replace it."
            )

        from datafolio.utils import get_file_extension

        def _build(filename: str, version_id: str) -> Dict[str, Any]:
            metadata = (
                get_registry()
                .get("included_table")
                .add(
                    self,
                    name,
                    data,
                    description=description,
                    inputs=inputs,
                    _filename=filename,
                )
            )
            if models is not None:
                metadata["models"] = models
            if code is not None:
                metadata["code"] = code
            return metadata

        self._commit_owned_item(
            name, "included_table", get_file_extension("parquet"), description, _build
        )
        return self

    def get_table(
        self,
        name: str,
        frame: str = "pandas",
        allow_full_load: bool = False,
        **kwargs: Any,
    ) -> Any:  # Returns pandas.DataFrame or polars.DataFrame
        """Get a table by name (works for both included and referenced).

        For included tables, reads from bundle directory.
        For referenced tables, reads from the specified external path.

        The ``frame`` argument selects the returned DataFrame flavor. With
        ``frame='pandas'`` (default), supports all pandas.read_parquet() arguments
        for filtering and optimization:
        - `columns`: List of column names to read (column pruning)
        - `filters`: Row filtering predicates (row filtering)
        - `engine`: Parquet engine ('pyarrow' or 'fastparquet')

        This is an *eager* read: it materializes the whole table. For large
        (especially referenced) tables, use :meth:`get_lazy` instead. The eager
        read is subject to the folio's ``max_eager_bytes`` guard.

        Args:
            name: Name of the table
            frame: Output flavor — ``'pandas'`` (default) or ``'polars'`` for an
                eager polars DataFrame.
            allow_full_load: Bypass the ``max_eager_bytes`` guard for this call.
            **kwargs: Additional arguments passed to the reader
                     (pandas: columns, filters, engine)

        Returns:
            pandas DataFrame (``frame='pandas'``) or polars DataFrame (``frame='polars'``)

        Raises:
            KeyError: If table name doesn't exist
            ValueError: If the table exceeds ``max_eager_bytes`` (and isn't
                flagged ``allow_full_load``), or if ``frame`` is invalid
            ImportError: If reading from cloud requires missing dependencies
            FileNotFoundError: If referenced file doesn't exist

        Examples:
            Basic usage:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> import pandas as pd
            >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> folio.add_table('test', df)
            >>> retrieved = folio.get_table('test')
            >>> assert len(retrieved) == 3

            Eager polars DataFrame:
            >>> pdf = folio.get_table('test', frame='polars')

            Column selection (read only specific columns):
            >>> df_subset = folio.get_table('test', columns=['a'])
            >>> assert list(df_subset.columns) == ['a']

            Row filtering (requires pyarrow engine):
            >>> df_filtered = folio.get_table('test',
            ...     filters=[('a', '>', 1)],
            ...     engine='pyarrow')
            >>> assert len(df_filtered) == 2
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        # Check if item exists
        if name not in self._items:
            raise KeyError(f"Table '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        # Validate it's a table
        if item_type not in ("included_table", "referenced_table"):
            raise ValueError(f"Item '{name}' is not a table (type: {item_type})")

        # Guard against accidentally materializing a huge table
        self._check_eager_size(name, item, allow_full_load)

        # Get handler and delegate to it
        registry = get_registry()
        handler = registry.get(item_type)

        if frame == "polars":
            # Eager polars: scan lazily then collect.
            return handler.get_lazy(self, name, **kwargs).collect()
        elif frame == "pandas":
            # Sharded/partitioned (polars-only) tables can't be materialized as
            # pandas — raise a clear, actionable error instead of a cryptic one.
            if item.get("polars_only"):
                raise _polars_only_error(name)
            return handler.get(self, name, **kwargs)
        else:
            raise ValueError(f"Unknown frame '{frame}'. Use 'pandas' or 'polars'.")

    def _check_eager_size(
        self, name: str, item: Dict[str, Any], allow_full_load: bool
    ) -> None:
        """Enforce the eager-load size guard for a table.

        Raises if the table's recorded ``size_bytes`` exceeds
        ``max_eager_bytes`` and neither the call nor the item opts out via
        ``allow_full_load``. Unknown size (no ``size_bytes``) is allowed through.

        Args:
            name: Table name (for the error message)
            item: The item's metadata dict
            allow_full_load: Per-call override

        Raises:
            ValueError: If the eager load exceeds the configured ceiling
        """
        if allow_full_load or item.get("allow_full_load"):
            return
        limit = self._max_eager_bytes
        if limit is None:
            return
        size = item.get("size_bytes")
        if size is None and item.get("item_type") == "referenced_table":
            # Reference sizes aren't recorded at (offline) creation. Stat the
            # object now — a cheap metadata lookup (HEAD), far cheaper than the
            # full read this guard protects. Operate on the effective resolved
            # path so legacy relative references stat correctly. Unknown size is
            # allowed through.
            try:
                size = self._storage.file_size(
                    self._resolve_reference_path(item["path"])
                )
            except Exception:
                size = None
        if size is not None and size > limit:
            size_mb = size / (1024 * 1024)
            limit_mb = limit / (1024 * 1024)
            raise ValueError(
                f"Table '{name}' is ~{size_mb:.0f} MB, above the "
                f"{limit_mb:.0f} MB eager-load limit. Use get_lazy('{name}') "
                f"for predicate/projection pushdown, or pass allow_full_load=True "
                f"(or set max_eager_bytes=None to disable this guard)."
            )

    def scan_table(self, name: str, **kwargs: Any) -> Any:  # Returns polars.LazyFrame
        """Scan a table as a **genuinely lazy** polars LazyFrame.

        Returns a lazy scan with predicate/projection pushdown that does not
        download or materialize the whole table up front — for referenced
        tables this reads from the external path without copying it. Not subject
        to the ``max_eager_bytes`` guard (the whole point is to avoid a full
        read).

        This is guaranteed lazy: if the table's location cannot be scanned
        lazily (an unsupported scheme, or a non-scannable format), it raises a
        clear error rather than silently downloading the object. For an eager
        read that does download, use ``get_table(name, frame='polars')``.

        Args:
            name: Name of the table
            **kwargs: Additional arguments passed to the polars scanner
                (e.g. ``storage_options`` for cloud credentials)

        Returns:
            polars LazyFrame

        Raises:
            KeyError: If table name doesn't exist
            ValueError: If the named item is not a table, or its location/format
                cannot be scanned lazily.
            ImportError: If polars is not installed
            NotImplementedError: If the table's format has no lazy scanner

        Examples:
            >>> import polars as pl
            >>> folio.reference_table('big', path='s3://bucket/huge.parquet')
            >>> lf = folio.scan_table('big')
            >>> lf.filter(pl.col('x') > 0).select('y').collect()  # pushdown
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Table '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        if item_type not in ("included_table", "referenced_table"):
            raise ValueError(f"Item '{name}' is not a table (type: {item_type})")

        registry = get_registry()
        handler = registry.get(item_type)
        return handler.get_lazy(self, name, **kwargs)

    def get_lazy(self, name: str, **kwargs: Any) -> Any:  # Returns polars.LazyFrame
        """Alias for :meth:`scan_table` (a genuinely lazy polars scan).

        Kept for backward compatibility; prefer :meth:`scan_table`, whose name
        makes the lazy contract explicit.
        """
        return self.scan_table(name, **kwargs)

    def get_data_path(self, name: str) -> str:
        """Get the path to any stored item, delegating to the appropriate type-specific method.

        Automatically detects the item type and calls the appropriate path getter:
        - Tables (included or referenced): delegates to get_table_path()
        - Artifacts: delegates to get_artifact_path()
        - All other bundled items (numpy arrays, JSON, timestamps): returns the bundle file path

        Args:
            name: Name of the item

        Returns:
            Path to the item file

        Raises:
            KeyError: If item name doesn't exist

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.add_table('results', df)
            >>> folio.get_data_path('results')  # returns path to parquet file
            >>> folio.reference_table('data', path='s3://bucket/file.parquet')
            >>> folio.get_data_path('data')  # returns 's3://bucket/file.parquet'
        """
        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in DataFolio")

        item_type = self._items[name].get("item_type")

        dispatch = {
            "included_table": self.get_table_path,
            "referenced_table": self.get_table_path,
            "artifact": self.get_artifact_path,
            "model": self.get_model_path,
            "numpy_array": self.get_numpy_path,
            "json_data": self.get_json_path,
            "timestamp": self.get_timestamp_path,
        }

        if item_type not in dispatch:
            raise ValueError(
                f"Item '{name}' has unknown type '{item_type}' with no path method."
            )

        return dispatch[item_type](name)

    def get_table_path(self, name: str) -> str:
        """Get the path to a table file, whether included in the bundle or referenced externally.

        For included tables, returns the full path to the parquet file inside the bundle.
        For referenced tables, returns the external path recorded at reference time.

        Args:
            name: Name of the table

        Returns:
            Path to the table file

        Raises:
            KeyError: If table name doesn't exist
            ValueError: If named item is not a table

        Examples:
            >>> folio = DataFolio('experiments/my-run')
            >>> folio.add_table('results', df)
            >>> path = folio.get_table_path('results')
            >>> print(path)
            'experiments/my-run/tables/results.parquet'

            >>> folio.reference_table('raw', path='s3://data-lake/raw.parquet')
            >>> folio.get_table_path('raw')
            's3://data-lake/raw.parquet'
        """
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Table '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        if item_type == "referenced_table":
            return self._resolve_reference_path(item["path"])
        elif item_type == "included_table":
            registry = get_registry()
            handler = registry.get(item_type)
            subdir = handler.get_storage_subdir()
            return self._storage.join_paths(self._bundle_dir, subdir, item["filename"])
        else:
            raise ValueError(f"Item '{name}' is not a table (type: {item_type})")

    def _resolve_reference_path(self, path: str) -> str:
        """Resolve a stored reference path to an accessible location.

        Cloud URIs, ``file://`` URIs, and absolute local paths are returned
        unchanged. A relative path is resolved against the bundle directory so
        that references stored relative to the bundle stay portable when the
        bundle (and its adjacent data) is moved or copied.

        Args:
            path: The path string stored in the manifest.

        Returns:
            A path usable by the storage backend / readers.
        """
        import os

        if is_cloud_path(path) or path.startswith("file://") or os.path.isabs(path):
            return path
        return self._storage.join_paths(self._bundle_dir, path)

    def get_table_info(self, name: str) -> Union[TableReference, IncludedTable]:
        """Get metadata about a table (referenced or included).

        Returns the manifest entry containing information like:
        - For referenced tables: path, table_format, is_directory, num_rows, version, description
        - For included tables: filename, table_format, is_directory, num_rows, num_cols, columns, dtypes, description

        Args:
            name: Name of the table

        Returns:
            Dictionary with table metadata

        Raises:
            KeyError: If table name doesn't exist

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.reference_table('data', path='s3://bucket/data.parquet', num_rows=1000000)
            >>> info = folio.get_table_info('data')
            >>> info['num_rows']
            1000000
            >>> info['table_format']
            'parquet'
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Table '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        if item_type in ("referenced_table", "included_table"):
            return item
        else:
            raise ValueError(f"Item '{name}' is not a table (type: {item_type})")

    def inspect_table(self, name: str) -> Union[TableReference, IncludedTable]:
        """Read a table's source and enrich its manifest entry.

        Unlike :meth:`reference_table` (a cheap, offline manifest write), this
        performs I/O against the table's location to record schema, size, row
        count, and (for external references) available source identity. The
        enriched metadata is persisted to the manifest and returned.

        This is the explicit, opt-in counterpart to offline reference creation:
        linking a table never touches the network, but inspecting it does.

        Args:
            name: Name of the table (included or referenced)

        Returns:
            The updated manifest entry.

        Raises:
            KeyError: If the table name doesn't exist.
            ValueError: If the named item is not a table.
            FileNotFoundError: If a referenced object does not exist.
            RuntimeError: If the object exists but its schema can't be read.

        Examples:
            >>> folio.reference_table('big', path='s3://bucket/data.parquet')
            >>> info = folio.inspect_table('big')  # reads schema/size now
            >>> info['columns'], info['num_rows']
        """
        self._check_read_only()
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Table '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")
        if item_type not in ("referenced_table", "included_table"):
            raise ValueError(f"Item '{name}' is not a table (type: {item_type})")

        registry = get_registry()
        handler = registry.get(item_type)
        enrichment = handler.inspect(self, name)

        if enrichment:
            self._items[name].update(enrichment)
            self._save_items()

        return self._items[name]

    def get_model_info(self, name: str) -> IncludedItem:
        """Get metadata about a model.

        Returns the manifest entry containing information like:
        - filename, item_type, description

        Args:
            name: Name of the model

        Returns:
            Dictionary with model metadata

        Raises:
            KeyError: If model name doesn't exist
            ValueError: If named item is not a model

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.add_model('classifier', model, description='Random forest classifier')
            >>> info = folio.get_model_info('classifier')
            >>> info['description']
            'Random forest classifier'
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Model '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "model":
            raise ValueError(
                f"Item '{name}' is not a model (type: {item.get('item_type')})"
            )

        return item

    def get_artifact_info(self, name: str) -> IncludedItem:
        """Get metadata about an artifact.

        Returns the manifest entry containing information like:
        - filename, item_type, category, description

        Args:
            name: Name of the artifact

        Returns:
            Dictionary with artifact metadata

        Raises:
            KeyError: If artifact name doesn't exist
            ValueError: If named item is not an artifact

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.add_artifact('plot', 'plot.png', category='plots', description='Loss curve')
            >>> info = folio.get_artifact_info('plot')
            >>> info['category']
            'plots'
            >>> info['description']
            'Loss curve'
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Artifact '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "artifact":
            raise ValueError(
                f"Item '{name}' is not an artifact (type: {item.get('item_type')})"
            )

        return item

    def add_sklearn(
        self,
        name: str,
        model: Any,
        description: Optional[str] = None,
        overwrite: bool = False,
        inputs: Optional[list[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None,
        custom: bool = False,
    ) -> Self:
        """Add a scikit-learn style model to the bundle.

        Writes immediately to models/ directory and updates items.json.

        Args:
            name: Unique name for this model
            model: Trained model to include
            description: Optional description
            overwrite: If True, allow overwriting existing model (default: False)
            inputs: Optional list of table names used for training
            hyperparameters: Optional dict of hyperparameters
            code: Optional code snippet that trained this model
            custom: If True, use skops format for portable pipelines with custom
                transformers. If False (default), use joblib format.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists and overwrite=False

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> folio = DataFolio('experiments', prefix='test')
            >>> model = RandomForestClassifier(n_estimators=100, max_depth=10)
            >>> # ... train model ...
            >>> folio.add_sklearn('classifier', model,
            ...     description='Random forest classifier',
            ...     inputs=['training_data', 'validation_data'],
            ...     hyperparameters={'n_estimators': 100, 'max_depth': 10},
            ...     code='model.fit(X_train, y_train)')
            >>>
            >>> # Portable pipeline with custom transformer (skops)
            >>> folio.add_sklearn('pipeline', custom_pipeline, custom=True)
        """
        self._check_read_only()

        validate_item_name(name)

        if name in self._items and not overwrite:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. "
                f"Use overwrite=True to replace it."
            )

        extension = ".skops" if custom else ".joblib"

        def _build(filename: str, version_id: str) -> Dict[str, Any]:
            metadata = (
                get_registry()
                .get("model")
                .add(
                    self,
                    name,
                    model,
                    description=description,
                    inputs=inputs,
                    custom=custom,
                    _filename=filename,
                )
            )
            if hyperparameters is not None:
                metadata["hyperparameters"] = hyperparameters
            if code is not None:
                metadata["code"] = code
            return metadata

        self._commit_owned_item(name, "model", extension, description, _build)
        return self

    def add_model(
        self,
        name: str,
        model: Any,
        description: Optional[str] = None,
        overwrite: bool = False,
        custom: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Add a scikit-learn style model to the bundle.

        This is a convenience method that delegates to add_sklearn().

        Args:
            name: Unique name for this model
            model: Trained sklearn-style model
            description: Optional description
            overwrite: If True, allow overwriting existing model (default: False)
            custom: If True use skops format for portability (required for custom transformers)
            **kwargs: Additional arguments passed to add_sklearn()
                (e.g., hyperparameters, inputs, code)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists and overwrite=False

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> model = RandomForestClassifier()
            >>> folio.add_model('clf', model, hyperparameters={'n_estimators': 100})

            With custom transformer (portable):
            >>> folio.add_model('pipeline', custom_pipeline, custom=True)
        """
        return self.add_sklearn(
            name,
            model,
            description=description,
            overwrite=overwrite,
            custom=custom,
            **kwargs,
        )

    def get_sklearn(self, name: str) -> Any:
        """Get a scikit-learn style model by name.


        Args:
            name: Name of the model

        Returns:
            The model object

        Raises:
            KeyError: If model name doesn't exist
            ValueError: If named item is not a sklearn model

        Examples:
            >>> folio = DataFolio('experiments/test')
            >>> model = folio.get_sklearn('classifier')
        """
        if name not in self._items:
            raise KeyError(f"Model '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "model":
            raise ValueError(
                f"Item '{name}' is not a sklearn model (type: {item.get('item_type')})"
            )

        # Delegate to handler

        registry = get_registry()
        handler = registry.get("model")
        return handler.get(self, name)

    def get_model(self, name: str, **kwargs: Any) -> Any:
        """Get a scikit-learn style model by name.

        This is a convenience method that delegates to get_sklearn().


        Args:
            name: Name of the model
            **kwargs: Additional arguments (currently unused, kept for backward compatibility)

        Returns:
            The model object

        Raises:
            KeyError: If model name doesn't exist
            ValueError: If named item is not a model

        Examples:
            >>> folio = DataFolio('experiments/test')
            >>> model = folio.get_model('classifier')
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Model '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        # Dispatch based on item type
        if item_type == "model":
            return self.get_sklearn(name)
        else:
            raise ValueError(f"Item '{name}' is not a model (type: {item_type})")

    def get_model_path(self, name: str) -> str:
        """Get the path to a model file stored in the bundle.

        Args:
            name: Name of the model

        Returns:
            Path to the model file

        Raises:
            KeyError: If model name doesn't exist
            ValueError: If named item is not a model

        Examples:
            >>> folio = DataFolio('experiments/my-run')
            >>> folio.add_sklearn('classifier', model)
            >>> path = folio.get_model_path('classifier')
            >>> print(path)
            'experiments/my-run/models/classifier.joblib'
        """
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Model '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "model":
            raise ValueError(
                f"Item '{name}' is not a model (type: {item.get('item_type')})"
            )

        handler = get_registry().get("model")
        subdir = handler.get_storage_subdir()
        return self._storage.join_paths(self._bundle_dir, subdir, item["filename"])

    def add_artifact(
        self,
        name: str,
        path: Union[str, Path],
        category: Optional[str] = None,
        description: Optional[str] = None,
        overwrite: bool = False,
    ) -> Self:
        """Add an artifact file to the bundle.

        Copies file immediately to artifacts/ directory and updates included_items.json.

        Args:
            name: Unique name for this artifact
            path: Path to the file to include
            category: Optional category ('plots', 'configs', etc.)
            description: Optional description
            overwrite: If True, allow overwriting existing artifact (default: False)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists and overwrite=False
            FileNotFoundError: If file doesn't exist

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.add_artifact('loss_curve', 'plots/training_loss.png', category='plots')
            >>> # Update with overwrite
            >>> folio.add_artifact('loss_curve', 'plots/updated_loss.png', category='plots', overwrite=True)
        """
        self._check_read_only()

        validate_item_name(name)

        if name in self._items and not overwrite:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. "
                f"Use overwrite=True to replace it."
            )

        # Owned payload filename preserves the source file's extension.
        extension = Path(str(path)).suffix

        def _build(filename: str, version_id: str) -> Dict[str, Any]:
            metadata = (
                get_registry()
                .get("artifact")
                .add(self, name, str(path), description=description, _filename=filename)
            )
            if category is not None:
                metadata["category"] = category
            return metadata

        self._commit_owned_item(name, "artifact", extension, description, _build)
        return self

    def add_file(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
        overwrite: bool = False,
    ) -> Self:
        """Add a file to the bundle (convenience wrapper for add_artifact).

        This is a file-centric interface that makes it easy to include
        arbitrary files like code, documentation, configs, etc.

        Args:
            path: Path to the file to include
            name: Optional name for this file (default: uses filename from path)
            category: Optional category ('code', 'docs', 'configs', etc.)
            description: Optional description
            overwrite: If True, allow overwriting existing file (default: False)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists and overwrite=False
            FileNotFoundError: If file doesn't exist

        Examples:
            Add files using their filenames as names:
            >>> folio = DataFolio('experiments/test')
            >>> folio.add_file('path/to/readme.md', description='Project README')
            >>> folio.add_file('src/train.py', category='code')

            Add with custom name:
            >>> folio.add_file('path/to/config.yaml', name='model_config')

            Add and overwrite:
            >>> folio.add_file('update.md', overwrite=True)

            Method chaining:
            >>> folio.add_file('readme.md').add_file('train.py').add_file('config.yaml')
        """
        # Use filename as name if not provided
        if name is None:
            # Use filename without extension as the name
            # (artifact handler will add the extension back)
            name = Path(path).stem

        # Delegate to add_artifact
        return self.add_artifact(
            name=name,
            path=path,
            category=category,
            description=description,
            overwrite=overwrite,
        )

    def get_artifact_path(self, name: str) -> str:
        """Get the path to an artifact file.

        Args:
            name: Name of the artifact

        Returns:
            Path to the artifact file

        Raises:
            KeyError: If artifact name doesn't exist
            ValueError: If named item is not an artifact

        Examples:
            >>> folio = DataFolio('experiments/test-blue-happy-falcon')
            >>> path = folio.get_artifact_path('plot')
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Artifact '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "artifact":
            raise ValueError(
                f"Item '{name}' is not an artifact (type: {item.get('item_type')})"
            )

        # Delegate to handler

        registry = get_registry()
        handler = registry.get("artifact")
        return handler.get(self, name)

    def get_item_path(self, name: str) -> str:
        """Get the path to any item stored in the folio.

        For items stored within the bundle (included tables, models, artifacts,
        arrays, JSON data, timestamps), returns the full path to the data file.
        For referenced tables, returns the external path recorded at reference time.

        This is especially useful for cloud-hosted folios where collaborators can
        directly access or download underlying files without using datafolio.

        Args:
            name: Name of the item

        Returns:
            Full path to the item's data file. For cloud folios this will be a
            cloud URI (e.g. ``s3://bucket/.../results.parquet``). For local folios
            this will be an absolute file-system path.

        Raises:
            KeyError: If item name doesn't exist
            ValueError: If item has no associated file path

        Examples:
            >>> folio = DataFolio('s3://bucket/experiments/my-run')
            >>> path = folio.get_item_path('results')
            >>> print(path)
            's3://bucket/experiments/my-run/tables/results.parquet'

            >>> path = folio.get_item_path('classifier')
            >>> print(path)
            's3://bucket/experiments/my-run/models/classifier.joblib'

            >>> # For referenced tables the external path is returned
            >>> folio.reference_table('raw', path='s3://data-lake/raw.parquet')
            >>> folio.get_item_path('raw')
            's3://data-lake/raw.parquet'
        """
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        # Referenced tables point to external data — return that path directly
        if item_type == "referenced_table":
            return item["path"]

        # All bundled items store their filename in metadata
        if "filename" not in item:
            raise ValueError(
                f"Item '{name}' (type: {item_type}) has no associated file path"
            )

        registry = get_registry()
        handler = registry.get(item_type)
        subdir = handler.get_storage_subdir()
        return self._storage.join_paths(self._bundle_dir, subdir, item["filename"])

    def add_numpy(
        self,
        name: str,
        array: Any,
        description: Optional[str] = None,
        overwrite: bool = False,
        inputs: Optional[list[str]] = None,
        code: Optional[str] = None,
    ) -> Self:
        """Add a numpy array to the bundle.

        Saves array to artifacts/ directory as .npy file and updates items.json.

        Args:
            name: Unique name for this array
            array: numpy array to save
            description: Optional description
            overwrite: If True, allow overwriting existing array (default: False)
            inputs: Optional list of items this was derived from
            code: Optional code snippet that created this array

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists and overwrite=False
            ImportError: If numpy is not installed
            TypeError: If data is not a numpy array

        Examples:
            >>> import numpy as np
            >>> folio = DataFolio('experiments/test')
            >>> embeddings = np.random.randn(100, 128)
            >>> folio.add_numpy('embeddings', embeddings, description='Model embeddings')
            >>> # With lineage
            >>> predictions = np.array([0, 1, 0, 1])
            >>> folio.add_numpy('predictions', predictions,
            ...     inputs=['test_data'],
            ...     code='predictions = model.predict(X)')
        """
        self._check_read_only()

        validate_item_name(name)

        # Overwriting any existing item requires overwrite=True; a snapshotted
        # prior version is then preserved via copy-on-write in the commit.
        if name in self._items and not overwrite:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. "
                f"Use overwrite=True to replace it."
            )

        def _build(filename: str, version_id: str) -> Dict[str, Any]:
            metadata = (
                get_registry()
                .get("numpy_array")
                .add(
                    self,
                    name,
                    array,
                    description=description,
                    inputs=inputs,
                    _filename=filename,
                )
            )
            if code is not None:
                metadata["code"] = code
            return metadata

        self._commit_owned_item(name, "numpy_array", ".npy", description, _build)
        return self

    def get_numpy(self, name: str) -> Any:
        """Get a numpy array by name.


        Args:
            name: Name of the array

        Returns:
            numpy array

        Raises:
            KeyError: If array name doesn't exist
            ValueError: If named item is not a numpy array
            ImportError: If numpy is not installed

        Examples:
            >>> folio = DataFolio('experiments/test')
            >>> embeddings = folio.get_numpy('embeddings')
            >>> print(embeddings.shape)
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Array '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "numpy_array":
            raise ValueError(
                f"Item '{name}' is not a numpy array (type: {item.get('item_type')})"
            )

        # Get handler and delegate to it
        registry = get_registry()
        handler = registry.get("numpy_array")
        return handler.get(self, name)

    def get_numpy_path(self, name: str) -> str:
        """Get the path to a numpy array file stored in the bundle.

        Args:
            name: Name of the array

        Returns:
            Path to the .npy file

        Raises:
            KeyError: If array name doesn't exist
            ValueError: If named item is not a numpy array

        Examples:
            >>> folio = DataFolio('experiments/my-run')
            >>> folio.add_numpy('embeddings', arr)
            >>> path = folio.get_numpy_path('embeddings')
            >>> print(path)
            'experiments/my-run/artifacts/embeddings.npy'
        """
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Array '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "numpy_array":
            raise ValueError(
                f"Item '{name}' is not a numpy array (type: {item.get('item_type')})"
            )

        handler = get_registry().get("numpy_array")
        subdir = handler.get_storage_subdir()
        return self._storage.join_paths(self._bundle_dir, subdir, item["filename"])

    def add_json(
        self,
        name: str,
        data: Union[dict, list, int, float, str, bool, None],
        description: Optional[str] = None,
        overwrite: bool = False,
        inputs: Optional[list[str]] = None,
        code: Optional[str] = None,
    ) -> Self:
        """Add JSON-serializable data to the bundle.

        Saves data to artifacts/ directory as .json file and updates items.json.
        Supports dicts, lists, scalars, and other JSON-serializable types.

        Args:
            name: Unique name for this data
            data: JSON-serializable data (dict, list, scalar, etc.)
            description: Optional description
            overwrite: If True, allow overwriting existing data (default: False)
            inputs: Optional list of items this was derived from
            code: Optional code snippet that created this data

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists and overwrite=False, or data not JSON-serializable
            TypeError: If data cannot be serialized to JSON

        Examples:
            >>> folio = DataFolio('experiments/test')
            >>> config = {'learning_rate': 0.01, 'batch_size': 32}
            >>> folio.add_json('config', config, description='Model config')
            >>> # With list data
            >>> class_names = ['cat', 'dog', 'bird']
            >>> folio.add_json('classes', class_names)
            >>> # With scalar
            >>> folio.add_json('best_accuracy', 0.95)
        """
        self._check_read_only()

        validate_item_name(name)

        if name in self._items and not overwrite:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. "
                f"Use overwrite=True to replace it."
            )

        def _build(filename: str, version_id: str) -> Dict[str, Any]:
            metadata = (
                get_registry()
                .get("json_data")
                .add(
                    self,
                    name,
                    data,
                    description=description,
                    inputs=inputs,
                    _filename=filename,
                )
            )
            if code is not None:
                metadata["code"] = code
            return metadata

        self._commit_owned_item(name, "json_data", ".json", description, _build)
        return self

    def get_json(self, name: str) -> Any:
        """Get JSON data by name.


        Args:
            name: Name of the JSON data

        Returns:
            Deserialized JSON data (dict, list, scalar, etc.)

        Raises:
            KeyError: If data name doesn't exist
            ValueError: If named item is not JSON data

        Examples:
            >>> folio = DataFolio('experiments/test')
            >>> config = folio.get_json('config')
            >>> print(config['learning_rate'])
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"JSON data '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "json_data":
            raise ValueError(
                f"Item '{name}' is not JSON data (type: {item.get('item_type')})"
            )

        # Delegate to handler
        registry = get_registry()
        handler = registry.get("json_data")
        return handler.get(self, name)

    def get_json_path(self, name: str) -> str:
        """Get the path to a JSON data file stored in the bundle.

        Args:
            name: Name of the JSON data

        Returns:
            Path to the .json file

        Raises:
            KeyError: If data name doesn't exist
            ValueError: If named item is not JSON data

        Examples:
            >>> folio = DataFolio('experiments/my-run')
            >>> folio.add_json('config', {'lr': 0.01})
            >>> path = folio.get_json_path('config')
            >>> print(path)
            'experiments/my-run/artifacts/config.json'
        """
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"JSON data '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "json_data":
            raise ValueError(
                f"Item '{name}' is not JSON data (type: {item.get('item_type')})"
            )

        handler = get_registry().get("json_data")
        subdir = handler.get_storage_subdir()
        return self._storage.join_paths(self._bundle_dir, subdir, item["filename"])

    def add_timestamp(
        self,
        name: str,
        timestamp: Union[datetime, int, float],
        description: Optional[str] = None,
        overwrite: bool = False,
        inputs: Optional[list[str]] = None,
        code: Optional[str] = None,
    ) -> Self:
        """Add a timestamp to the bundle.

        Saves timestamp to artifacts/ directory as .json file and updates items.json.
        Accepts timezone-aware datetime objects or Unix timestamps (int/float).
        All timestamps are stored in UTC as ISO 8601 strings.

        Args:
            name: Unique name for this timestamp
            timestamp: Timezone-aware datetime object or Unix timestamp (int/float).
                      Naive datetimes will raise ValueError.
            description: Optional description
            overwrite: If True, allow overwriting existing timestamp (default: False)
            inputs: Optional list of items this was derived from
            code: Optional code snippet that created this timestamp

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists and overwrite=False, or if datetime is naive
            TypeError: If timestamp is not a datetime or numeric type

        Examples:
            >>> from datetime import datetime, timezone
            >>> folio = DataFolio('experiments/test')
            >>>
            >>> # Add timezone-aware datetime
            >>> event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
            >>> folio.add_timestamp('event_time', event_time, description='Event occurred')
            >>>
            >>> # Add Unix timestamp
            >>> folio.add_timestamp('start_time', 1705318200, description='Start time')
            >>>
            >>> # With lineage
            >>> from datetime import datetime, timezone
            >>> import pytz
            >>> eastern = pytz.timezone('US/Eastern')
            >>> local_time = eastern.localize(datetime(2024, 1, 15, 10, 30, 0))
            >>> folio.add_timestamp('local_event', local_time,
            ...     inputs=['event_log'],
            ...     code='timestamp = event_log.iloc[0]["timestamp"]')
        """
        self._check_read_only()

        validate_item_name(name)

        if name in self._items and not overwrite:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. "
                f"Use overwrite=True to replace it."
            )

        def _build(filename: str, version_id: str) -> Dict[str, Any]:
            metadata = (
                get_registry()
                .get("timestamp")
                .add(
                    self,
                    name,
                    timestamp,
                    description=description,
                    inputs=inputs,
                    _filename=filename,
                )
            )
            if code is not None:
                metadata["code"] = code
            return metadata

        self._commit_owned_item(name, "timestamp", ".json", description, _build)
        return self

    def get_timestamp(self, name: str, as_unix: bool = False) -> Union[datetime, float]:
        """Get a timestamp by name.


        Args:
            name: Name of the timestamp
            as_unix: If True, return Unix timestamp (float); if False, return datetime (default)

        Returns:
            UTC-aware datetime object (default) or Unix timestamp (if as_unix=True)

        Raises:
            KeyError: If timestamp name doesn't exist
            ValueError: If named item is not a timestamp

        Examples:
            >>> folio = DataFolio('experiments/test')
            >>>
            >>> # Get as datetime (default)
            >>> event_time = folio.get_timestamp('event_time')
            >>> print(event_time.isoformat())
            '2024-01-15T10:30:00+00:00'
            >>>
            >>> # Get as Unix timestamp
            >>> unix_time = folio.get_timestamp('event_time', as_unix=True)
            >>> print(unix_time)
            1705318200.0
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Timestamp '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "timestamp":
            raise ValueError(
                f"Item '{name}' is not a timestamp (type: {item.get('item_type')})"
            )

        # Delegate to handler
        registry = get_registry()
        handler = registry.get("timestamp")
        dt = handler.get(self, name, as_unix=False)
        return dt.timestamp() if as_unix else dt

    def get_timestamp_path(self, name: str) -> str:
        """Get the path to a timestamp file stored in the bundle.

        Args:
            name: Name of the timestamp

        Returns:
            Path to the timestamp file

        Raises:
            KeyError: If timestamp name doesn't exist
            ValueError: If named item is not a timestamp

        Examples:
            >>> folio = DataFolio('experiments/my-run')
            >>> folio.add_timestamp('event_time', dt)
            >>> path = folio.get_timestamp_path('event_time')
            >>> print(path)
            'experiments/my-run/artifacts/event_time.json'
        """
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Timestamp '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "timestamp":
            raise ValueError(
                f"Item '{name}' is not a timestamp (type: {item.get('item_type')})"
            )

        handler = get_registry().get("timestamp")
        subdir = handler.get_storage_subdir()
        return self._storage.join_paths(self._bundle_dir, subdir, item["filename"])

    def add_data(
        self,
        name: str,
        data: Any = None,
        reference: Optional[Union[str, Path]] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> Self:
        """Generic data addition with automatic type detection.

        Convenience method that dispatches to the appropriate specific method
        based on data type. For fine-grained control, use the specific methods:
        add_table(), add_numpy(), add_json(), or reference_table().

        Args:
            name: Unique name for this data
            data: Data to save (DataFrame, numpy array, dict, list, scalar)
            reference: If provided, creates a reference to external data instead
            description: Optional description
            **kwargs: Additional arguments passed to the specific method

        Returns:
            Self for method chaining

        Raises:
            ValueError: If neither data nor reference is provided, or both are provided
            TypeError: If data type is not supported

        Examples:
            DataFrame (saves as parquet):
            >>> folio.add_data('results', df)

            Numpy array (saves as .npy):
            >>> folio.add_data('embeddings', np.array([1, 2, 3]))

            JSON data (saves as .json):
            >>> folio.add_data('config', {'lr': 0.01})
            >>> folio.add_data('classes', ['cat', 'dog'])
            >>> folio.add_data('accuracy', 0.95)

            External reference:
            >>> folio.add_data('raw', reference='s3://bucket/data.parquet')
        """
        self._check_read_only()

        # Validate inputs
        if data is None and reference is None:
            raise ValueError("Must provide either 'data' or 'reference' parameter")
        if data is not None and reference is not None:
            raise ValueError("Cannot provide both 'data' and 'reference' parameters")

        # Handle reference
        if reference is not None:
            return self.reference_table(
                name, reference, description=description, **kwargs
            )

        # Type detection only. We then delegate to the type-specific public
        # method so that duplicate-name checks, overwrite semantics, snapshot
        # copy-on-write, and type-specific metadata are enforced identically to
        # the explicit APIs (add_data must not bypass those invariants).
        from datafolio.base.registry import detect_handler

        handler = detect_handler(data)
        item_type = handler.item_type if handler is not None else None

        # Fallback: primitives (int, float, str, bool, None) -> JSON.
        # These don't auto-detect (to avoid conflicts), but add_data accepts them.
        if item_type is None and isinstance(data, (int, float, str, bool, type(None))):
            item_type = "json_data"

        if item_type is None:
            raise TypeError(
                f"Unsupported data type: {type(data).__name__}. "
                f"No handler found for this type. "
                f"Supported types: pandas.DataFrame, Polars DataFrame, numpy.ndarray, "
                f"sklearn models, dict, list, datetime, file paths, or JSON scalars. "
                f"Use explicit add_*() methods for more control."
            )

        # Delegate to the corresponding public method (value arg name varies).
        if item_type == "included_table":
            return self.add_table(name, data, description=description, **kwargs)
        elif item_type == "numpy_array":
            return self.add_numpy(name, data, description=description, **kwargs)
        elif item_type == "json_data":
            return self.add_json(name, data, description=description, **kwargs)
        elif item_type == "timestamp":
            return self.add_timestamp(name, data, description=description, **kwargs)
        elif item_type == "model":
            return self.add_model(name, data, description=description, **kwargs)
        elif item_type == "artifact":
            return self.add_artifact(name, data, description=description, **kwargs)
        else:
            raise TypeError(
                f"Detected item type '{item_type}' has no generic add path. "
                f"Use the corresponding explicit add_*() method."
            )

    def get_data(self, name: str) -> Any:
        """Generic data getter that returns any data type.

        Automatically detects the item type and calls the appropriate getter.
        For fine-grained control, use the specific methods: get_table(),
        get_numpy(), or get_json().

        Args:
            name: Name of the data item

        Returns:
            The data (DataFrame, numpy array, dict, list, or scalar)

        Raises:
            KeyError: If item name doesn't exist
            ValueError: If item is not a data type (e.g., is a model or artifact)

        Examples:
            >>> folio.add_data('results', df)
            >>> folio.add_data('embeddings', np_array)
            >>> folio.add_data('config', {'lr': 0.01})
            >>> # Later, retrieve without knowing the type
            >>> results = folio.get_data('results')  # Returns DataFrame
            >>> embeddings = folio.get_data('embeddings')  # Returns numpy array
            >>> config = folio.get_data('config')  # Returns dict
        """
        # Auto-refresh so a freshly-written item from another writer is visible.
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        # Delegate to the type-specific public getter so eager-size guards,
        # polars_only handling, and refresh behave identically to the
        # explicit APIs (get_data must not bypass those).
        if item_type in ("referenced_table", "included_table"):
            return self.get_table(name)
        elif item_type == "numpy_array":
            return self.get_numpy(name)
        elif item_type == "json_data":
            return self.get_json(name)
        elif item_type == "timestamp":
            return self.get_timestamp(name)
        else:
            # Not a "data" type (e.g. model or artifact), or unknown.
            raise ValueError(
                f"Item '{name}' is not a data item (type: {item_type}). "
                f"Use get_model() for models or get_artifact_path() for artifacts."
            )

    def update_item(
        self,
        name: str,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        code: Optional[str] = None,
    ) -> Self:
        """Update metadata for an existing item.

        Allows you to modify the description, inputs, or code fields
        of an item after it's been added to the bundle.

        If the item's current version is pinned by a snapshot, the update is
        applied copy-on-write: the snapshot keeps the metadata exactly as
        recorded, and the working item gets a new version (sharing the same
        payload file) carrying the edit.

        Passing ``None`` for a field leaves it unchanged; passing an empty
        value (``""`` for description/code, ``[]`` for inputs) removes it.

        Args:
            name: Name of the item to update
            description: New description ("" removes it; None leaves unchanged)
            inputs: New list of input items ([] removes it; None leaves unchanged)
            code: New code snippet ("" removes it; None leaves unchanged)

        Returns:
            Self for method chaining

        Raises:
            KeyError: If item doesn't exist

        Examples:
            Update description:
            >>> folio = DataFolio('experiments/test')
            >>> folio.update_item('predictions', description='Model predictions on test set')

            Update inputs:
            >>> folio.update_item('final_model', inputs=['train_data', 'val_data'])

            Update multiple fields:
            >>> folio.update_item(
            ...     'feature_matrix',
            ...     description='Normalized feature matrix',
            ...     inputs=['raw_data'],
            ...     code='features = normalize(raw_data)'
            ... )

            Clear a field by passing an empty string:
            >>> folio.update_item('temp', code='')  # Removes code field
        """
        self._check_read_only()

        # Validate item exists
        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in DataFolio")

        with self._mutation_guard():
            item = self._items[name]

            if self._is_in_snapshots(name):
                # Copy-on-write: the snapshot must keep seeing the metadata it
                # pinned. The new working version shares the payload file (the
                # bytes didn't change); every payload-deletion path checks
                # _payload_is_shared before removing a file.
                import copy

                new_item = copy.deepcopy(dict(item))
                self._handle_copy_on_write(name)
                new_item["in_snapshots"] = []
                new_item["is_current"] = True
                new_item["version_id"] = self._next_version_id(name)
                self._items[name] = new_item
                item = new_item

            # Update fields if provided
            if description is not None:
                if description == "":
                    # Empty string removes the field
                    item.pop("description", None)
                else:
                    item["description"] = description

            if inputs is not None:
                if not inputs:
                    # Empty list removes the field
                    item.pop("inputs", None)
                else:
                    item["inputs"] = inputs

            if code is not None:
                if code == "":
                    # Empty string removes the field
                    item.pop("code", None)
                else:
                    item["code"] = code

            # Save updated manifest
            self._save_items()

        return self

    def delete(self, name: Union[str, list[str]], warn_dependents: bool = True) -> Self:
        """Delete one or more items from the DataFolio.

        Removes items from the manifest and deletes associated files.
        Does not enforce lineage - can delete items that other items depend on.

        Args:
            name: Name(s) of item(s) to delete (string or list of strings)
            warn_dependents: If True, print warning if deleted items have dependents

        Returns:
            Self for method chaining

        Raises:
            KeyError: If any item name doesn't exist

        Examples:
            Delete single item:
            >>> folio = DataFolio('experiments/test')
            >>> folio.delete('old_model')

            Delete multiple items:
            >>> folio.delete(['temp_data', 'debug_plot', 'old_model'])

            Delete without warnings:
            >>> folio.delete('item', warn_dependents=False)
        """
        self._check_read_only()

        import warnings

        # Convert single name to list for uniform processing
        names_to_delete = [name] if isinstance(name, str) else name

        # Validate all items exist before deleting any
        for item_name in names_to_delete:
            if item_name not in self._items:
                raise KeyError(f"Item '{item_name}' not found in DataFolio")

        with self._mutation_guard():
            for item_name in names_to_delete:
                item = self._items[item_name]

                # Check for dependents and warn if requested
                if warn_dependents:
                    dependents = self.get_dependents(item_name)
                    if dependents:
                        warnings.warn(
                            f"Deleting '{item_name}' which is used by: "
                            f"{', '.join(dependents)}. "
                            f"Those items may have broken lineage.",
                            UserWarning,
                            stacklevel=2,
                        )

                if self._is_in_snapshots(item_name):
                    # A snapshot pins this version: keep the payload and move
                    # the descriptor to the snapshot versions — deleting the
                    # bytes would silently corrupt every snapshot containing
                    # it. The logical name disappears from the working set.
                    self._handle_copy_on_write(item_name)
                else:
                    # No snapshot references this version — remove the payload
                    # (unless a snapshotted descriptor still shares the file
                    # after a metadata-only copy-on-write).
                    self._delete_payload_if_unshared(item)

                # Remove from items manifest
                del self._items[item_name]

            # Save updated manifest
            self._save_items()

        return self

    # ==================== Archive Methods ====================

    def archive(self, name: Union[str, list[str]]) -> Self:
        """Mark item(s) as archived (hidden from default views, not deleted).

        Archived items remain on disk and are still accessible via get_data() /
        get_table() etc., but are excluded from list_contents(), describe(), and
        copy() by default.  Pass include_archived=True to those methods to reveal
        them again, or call unarchive() to restore them permanently.

        Accepts a single name, a list of names, or a glob pattern (fnmatch rules,
        e.g. ``'intermediate/*'``).

        Args:
            name: Item name, list of names, or glob pattern to archive.

        Returns:
            Self for method chaining

        Raises:
            KeyError: If a specific name (non-glob) is not found

        Examples:
            Archive a single item:
            >>> folio.archive('debug_output')

            Archive multiple items:
            >>> folio.archive(['debug_output', 'temp_features'])

            Archive by glob pattern:
            >>> folio.archive('intermediate/*')
        """
        self._check_read_only()

        names_to_archive: list[str]
        if isinstance(name, str):
            # Check if it looks like a glob pattern
            if any(c in name for c in ("*", "?", "[")):
                names_to_archive = fnmatch.filter(list(self._items.keys()), name)
            else:
                if name not in self._items:
                    raise KeyError(f"Item '{name}' not found in DataFolio")
                names_to_archive = [name]
        else:
            # List of explicit names — validate all exist first
            for n in name:
                if n not in self._items:
                    raise KeyError(f"Item '{n}' not found in DataFolio")
            names_to_archive = list(name)

        for n in names_to_archive:
            self._items[n]["archived"] = True

        self._save_items()
        return self

    def unarchive(self, name: Union[str, list[str]]) -> Self:
        """Restore archived item(s) to active status.

        Removes the ``archived`` flag so the items appear again in
        list_contents(), describe(), and copy() by default.

        Accepts a single name, a list of names, or a glob pattern (fnmatch rules).

        Args:
            name: Item name, list of names, or glob pattern to unarchive.

        Returns:
            Self for method chaining

        Raises:
            KeyError: If a specific name (non-glob) is not found

        Examples:
            Unarchive a single item:
            >>> folio.unarchive('debug_output')

            Unarchive multiple items:
            >>> folio.unarchive(['debug_output', 'temp_features'])

            Unarchive by glob pattern:
            >>> folio.unarchive('intermediate/*')
        """
        self._check_read_only()

        names_to_unarchive: list[str]
        if isinstance(name, str):
            if any(c in name for c in ("*", "?", "[")):
                names_to_unarchive = fnmatch.filter(list(self._items.keys()), name)
            else:
                if name not in self._items:
                    raise KeyError(f"Item '{name}' not found in DataFolio")
                names_to_unarchive = [name]
        else:
            for n in name:
                if n not in self._items:
                    raise KeyError(f"Item '{n}' not found in DataFolio")
            names_to_unarchive = list(name)

        for n in names_to_unarchive:
            self._items[n].pop("archived", None)

        self._save_items()
        return self

    # ==================== Lineage Methods ====================

    def get_inputs(self, item_name: str) -> list[str]:
        """Get list of items that were inputs to this item.

        Args:
            item_name: Name of the item

        Returns:
            List of item names that were inputs

        Raises:
            KeyError: If item doesn't exist

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> # After adding items with lineage...
            >>> inputs = folio.get_inputs('predictions')
            >>> # Returns: ['test_data', 'classifier']
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if item_name not in self._items:
            raise KeyError(f"Item '{item_name}' not found in DataFolio")

        item = self._items[item_name]
        inputs = item.get("inputs", [])

        # For tables, also include models
        if item.get("item_type") == "included_table" and "models" in item:
            inputs = inputs + item.get("models", [])

        return inputs

    def get_dependents(self, item_name: str) -> list[str]:
        """Get list of items that depend on this item.

        Args:
            item_name: Name of the item

        Returns:
            List of item names that use this as input

        Raises:
            KeyError: If item doesn't exist

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> # After adding items with lineage...
            >>> dependents = folio.get_dependents('classifier')
            >>> # Returns items that used 'classifier' as input
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        if item_name not in self._items:
            raise KeyError(f"Item '{item_name}' not found in DataFolio")

        dependents = []
        for name, item in self._items.items():
            # Check inputs
            if item_name in item.get("inputs", []):
                dependents.append(name)
            # Check models (for tables)
            if item.get("item_type") == "included_table":
                if item_name in item.get("models", []):
                    dependents.append(name)

        return dependents

    def get_lineage_graph(self) -> Dict[str, list[str]]:
        """Get full dependency graph for all items in bundle.

        Returns:
            Dictionary mapping item names to their input item names

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> graph = folio.get_lineage_graph()
            >>> # Returns: {'predictions': ['test_data', 'classifier'], ...}
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        graph = {}
        for name in self._items.keys():
            graph[name] = self.get_inputs(name)
        return graph

    # ==================== Copy Method ====================

    def _resolve_lineage_closure(self, seeds: list[str]) -> set[str]:
        """Return the full set of items transitively required by the seed items.

        Uses ``get_lineage_graph()`` to walk upstream dependencies (BFS) from
        each seed.  Items referenced in ``inputs`` metadata that are not present
        in this folio are silently skipped.

        Args:
            seeds: Names of the root items to start from.

        Returns:
            Set of item names (seeds + all transitive upstream deps that exist
            in this folio).
        """
        graph = self.get_lineage_graph()  # {item: [inputs]}
        closure: set[str] = set()
        queue: list[str] = list(seeds)
        while queue:
            current = queue.pop()
            if current in closure:
                continue
            if current not in self._items:
                # External reference — skip (don't add to closure)
                continue
            closure.add(current)
            for dep in graph.get(current, []):
                if dep not in closure:
                    queue.append(dep)
        return closure

    def copy(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        include_items: Optional[list[str]] = None,
        exclude_items: Optional[list[str]] = None,
        random_suffix: bool = False,
        follow_lineage: bool = False,
        include_archived: bool = False,
    ) -> "DataFolio":
        """Create a copy of this bundle at a new location.

        Useful for creating derived experiments or checkpoints.

        Args:
            path: Destination path for the new bundle. Used as the exact bundle
                location (e.g., 'gs://bucket/experiments/my-copy').
            name: If provided, appended to path as a subdirectory
                (e.g., path='experiments', name='exp-v2' → 'experiments/exp-v2').
                If None, path is used as-is.
            metadata_updates: Metadata fields to update/add in the copy
            include_items: If specified, only copy these items (by name)
            exclude_items: Items to exclude from copy (by name)
            random_suffix: If True, append random suffix to new bundle name (default: False)
            follow_lineage: If True and include_items is provided, automatically
                include all transitive upstream dependencies of the named items.
                Items referenced in lineage that are not present in this folio
                (e.g. external tables) are silently skipped.
            include_archived: If True, archived items are included in the copy.
                Defaults to False so archived items are excluded.

        Returns:
            New DataFolio instance

        Raises:
            ValueError: If include_items and exclude_items are both specified

        Examples:
            >>> # Copy to exact destination path
            >>> folio2 = folio.copy('gs://bucket/experiments/my-copy')

            >>> # Copy to base directory with explicit name subdirectory
            >>> folio2 = folio.copy('experiments', name='exp-v2')

            >>> # Copy with random suffix
            >>> folio2 = folio.copy('experiments/exp-v2', random_suffix=True)

            >>> # Copy with metadata updates to track parent
            >>> folio2 = folio.copy(
            ...     'experiments/exp-v2',
            ...     metadata_updates={
            ...         'parent_bundle': folio._bundle_dir,
            ...         'changes': 'Increased max_depth to 15'
            ...     }
            ... )

            >>> # Copy only specific items (e.g., for derived experiment)
            >>> folio2 = folio.copy(
            ...     'experiments/exp-v2-tuned',
            ...     include_items=['training_data', 'validation_data'],
            ...     metadata_updates={'status': 'in_progress'}
            ... )

            >>> # Copy only final outputs, auto-resolving all upstream deps
            >>> folio2 = folio.copy(
            ...     'results',
            ...     include_items=['final_model', 'test_results'],
            ...     follow_lineage=True,
            ... )

            >>> # Include archived items in the copy
            >>> folio2 = folio.copy('archive_backup', include_archived=True)
        """
        import shutil

        if include_items is not None and exclude_items is not None:
            raise ValueError("Cannot specify both include_items and exclude_items")

        # Construct full path for new bundle
        if name is not None:
            new_path = self._storage.join_paths(str(path), name)
        else:
            new_path = str(path)

        # Create new bundle
        new_metadata = dict(self.metadata)
        if metadata_updates:
            new_metadata.update(metadata_updates)

        new_folio = DataFolio(
            path=new_path, metadata=new_metadata, random_suffix=random_suffix
        )

        # Wrap copy operation in try/except to cleanup on failure
        try:
            # Determine which items to copy
            # Start from only non-archived items unless include_archived is True
            if include_archived:
                items_to_copy = set(self._items.keys())
            else:
                items_to_copy = {
                    n
                    for n, item in self._items.items()
                    if not item.get("archived", False)
                }

            if include_items is not None:
                if follow_lineage:
                    # Expand seeds to the full transitive closure, then intersect
                    # with the visible set so we don't pull in archived deps
                    # unless the caller opted in via include_archived.
                    closure = self._resolve_lineage_closure(list(include_items))
                    items_to_copy = items_to_copy.intersection(closure)
                else:
                    items_to_copy = items_to_copy.intersection(include_items)
            if exclude_items is not None:
                items_to_copy = items_to_copy.difference(exclude_items)

            # Copy items
            for item_name in items_to_copy:
                item = self._items[item_name]
                item_type = item.get("item_type")

                if item_type == "referenced_table":
                    # Just copy the reference (no data to copy)
                    new_folio._items[item_name] = dict(item)

                elif "filename" in item:
                    # Copy file-based items (tables, models, arrays, JSON, etc.)
                    # Dynamically get storage directory for this item type
                    from datafolio.storage import get_storage_directory

                    storage_dir = get_storage_directory(item_type)

                    src_path = self._storage.join_paths(
                        self._bundle_dir, storage_dir, item["filename"]
                    )
                    dst_path = self._storage.join_paths(
                        new_folio._bundle_dir, storage_dir, item["filename"]
                    )
                    self._copy_payload_file(src_path, dst_path)

                    new_folio._items[item_name] = dict(item)

                else:
                    # Handle items without files (shouldn't happen, but be defensive)
                    new_folio._items[item_name] = dict(item)

            # Save the new manifest
            new_folio._save_items()

        except Exception:
            # Cleanup on failure: remove the partially created bundle
            import shutil as shutil_module

            if self._storage.exists(new_folio._bundle_dir):
                if is_cloud_path(new_folio._bundle_dir):
                    # Use cloudfiles to delete cloud directory
                    cf = cloudfiles.CloudFiles(new_folio._bundle_dir)
                    cf.delete(cf.list())
                else:
                    # Use shutil to delete local directory
                    shutil_module.rmtree(new_folio._bundle_dir)
            raise

        return new_folio
