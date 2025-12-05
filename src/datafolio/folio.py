"""Main DataFolio class for bundling analysis artifacts."""

import contextlib
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
    TABLES_DIR,
    IncludedItem,
    IncludedTable,
    TableReference,
    TimestampItem,
    is_cloud_path,
    make_bundle_name,
)

from .utils import resolve_path


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

    def get_table(self, name: str) -> Any:
        """Get a table as it existed in this snapshot.

        Args:
            name: Table name

        Returns:
            Table data (pandas DataFrame)

        Raises:
            KeyError: If table not in snapshot
        """
        if name not in self.item_versions:
            raise KeyError(f"Table '{name}' not found in snapshot '{self._name}'")

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
            return handler.get(self._folio, name)
        finally:
            # Restore original item
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
        cache_enabled: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_ttl: Optional[int] = None,
    ):
        """Initialize a new or open an existing DataFolio.

        If the directory doesn't exist, creates a new bundle.
        If it exists, opens the existing bundle and reads manifests.

        Args:
            path: Full path to bundle directory (local or cloud)
            metadata: Optional dictionary of analysis metadata (for new bundles)
            random_suffix: If True, append random suffix to bundle name (default: False)
            read_only: If True, prevent all write operations (default: False)
            cache_enabled: If True, enable local caching for remote data (default: False)
            cache_dir: Optional cache directory (default: ~/.datafolio_cache)
            cache_ttl: Optional TTL override in seconds (default: 1800 = 30 minutes)

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

            Enable caching for cloud bundles (faster repeated access):
            >>> folio = DataFolio('gs://bucket/experiment', cache_enabled=True)
            >>> df = folio.get_table('data')  # Downloads and caches
            >>> df = folio.get_table('data')  # Loads from cache (instant)

            Custom cache configuration:
            >>> folio = DataFolio(
            ...     'gs://bucket/experiment',
            ...     cache_enabled=True,
            ...     cache_dir='/mnt/shared/cache',
            ...     cache_ttl=3600  # 1 hour
            ... )
        """
        # Read-only mode flag
        self._read_only = read_only

        # Snapshot mode flags (set by load_snapshot())
        self._in_snapshot_mode = False
        self._loaded_snapshot: Optional[str] = None

        # Storage backend for all I/O operations
        self._storage = StorageBackend()

        # Cache manager (initialized after bundle_dir is set)
        self._cache_enabled = cache_enabled
        self._cache_manager: Optional["CacheManager"] = None
        self._cache_dir = cache_dir
        self._cache_ttl = cache_ttl

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

        # Batch mode flag
        self._batch_mode = False

        # Snapshot-related state (Phase 1)
        self._snapshots: Dict[str, Any] = {}  # Snapshot name → metadata

        # Check if path is an existing bundle (has metadata.json or items.json)
        path_str = str(path)
        metadata_path = self._storage.join_paths(path_str, METADATA_FILE)
        items_path = self._storage.join_paths(path_str, ITEMS_FILE)

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

        # Initialize cache manager if caching is enabled and bundle is remote
        if self._cache_enabled and is_cloud_path(self._bundle_dir):
            from datafolio.cache import CacheConfig, CacheManager

            # Build config kwargs, only include cache_dir if explicitly provided
            config_kwargs = {"enabled": True}
            if self._cache_dir:
                config_kwargs["cache_dir"] = Path(self._cache_dir)

            config = CacheConfig(**config_kwargs)
            self._cache_manager = CacheManager(
                bundle_path=self._bundle_dir,
                config=config,
                ttl_override=self._cache_ttl,
            )

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

    def _get_file_path_with_cache(self, name: str, remote_path: str) -> str:
        """Get file path, using cache if available and enabled.

        Args:
            name: Item name
            remote_path: Path to remote file

        Returns:
            Path to file (either cached local path or remote path)
        """
        # If caching not enabled, return remote path
        if not self._cache_enabled or self._cache_manager is None:
            return remote_path

        # Get item metadata
        item = self._items[name]
        item_type = item["item_type"]
        filename = item.get("filename", name)
        checksum = item.get("checksum")

        # Define fetch function for cache manager
        def fetch_from_remote():
            """Fetch file bytes from remote storage."""
            import os

            from datafolio.utils import resolve_path

            bundle_dir = resolve_path(self._bundle_dir)
            remote_full_path = resolve_path(remote_path)

            # Check if path is within bundle or external
            if remote_full_path.startswith(bundle_dir):
                # Path is within bundle - use relative path with self._cf
                relative_path = remote_full_path[len(bundle_dir) :].lstrip("/")
                data_bytes = self._cf.get(relative_path)
            else:
                # External reference - create new CloudFiles instance
                # Split into directory and filename
                dir_path = os.path.dirname(remote_full_path)
                file_name = os.path.basename(remote_full_path)
                cf_external = cloudfiles.CloudFiles(dir_path)
                data_bytes = cf_external.get(file_name)

            return (data_bytes, item_type, filename)

        # Try to get from cache (will fetch if needed)
        try:
            cached_path = self._cache_manager.get(
                item_name=name,
                remote_fetch_fn=fetch_from_remote,
                remote_checksum=checksum,
            )

            if cached_path:
                # Return local cached path as string
                return str(cached_path)
        except Exception as e:
            # If caching fails, fall back to remote path
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Cache error for {name}, falling back to remote: {e}")

        # Fallback to remote path
        return remote_path

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

    def _write_pytorch(
        self,
        path: str,
        model: Any,
        init_args: Optional[Dict[str, Any]] = None,
        save_class: bool = False,
    ) -> None:
        """Write PyTorch model state dict with metadata (local or cloud).

        Saves a dictionary containing:
        - state_dict: Model weights
        - metadata: Class name, module, and optional init args
        - serialized_class: Optional dill-serialized class (if save_class=True)

        Args:
            path: File path
            model: PyTorch model (will save state_dict)
            init_args: Optional dict of args needed to instantiate the model
            save_class: If True, use dill to serialize the model class
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required to save PyTorch models. "
                "Install with: pip install torch"
            )

        # Prepare save bundle
        save_bundle = {
            "state_dict": model.state_dict(),
            "metadata": {
                "model_class": model.__class__.__name__,
                "model_module": model.__class__.__module__,
            },
        }

        if init_args is not None:
            save_bundle["metadata"]["init_args"] = init_args

        # Optionally serialize class with dill
        if save_class:
            try:
                import dill

                save_bundle["serialized_class"] = dill.dumps(model.__class__)
            except ImportError:
                raise ImportError(
                    "dill is required when save_class=True. "
                    "Install with: pip install dill"
                )

        if is_cloud_path(path):
            # Write to temp file, then upload
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                torch.save(save_bundle, tmp.name)
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
            torch.save(save_bundle, path)

    def _read_pytorch(self, path: str) -> Dict[str, Any]:
        """Read PyTorch model bundle (local or cloud).

        Args:
            path: File path

        Returns:
            Dictionary containing state_dict, metadata, and optionally serialized_class
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required to load PyTorch models. "
                "Install with: pip install torch"
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                tmp.write(content)
                tmp.flush()
                bundle = torch.load(tmp.name, weights_only=False)
                Path(tmp.name).unlink()
                return bundle
        else:
            return torch.load(path, weights_only=False)

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
        """Write a README.md file to document the bundle structure."""
        from datafolio import __version__

        readme_content = f"""# DataFolio Bundle

This directory was created by [datafolio](https://github.com/ceesem/datafolio) version {__version__}.

## Structure

- `metadata.json` - User metadata and timestamps
- `items.json` - Manifest of all data items (tables, models, artifacts)
- `tables/` - Parquet files for included tables
- `models/` - Serialized ML models (sklearn, PyTorch)
- `artifacts/` - Plots, configs, numpy arrays, JSON data, and other files

## Usage

Load this bundle in Python:

```python
from datafolio import DataFolio

# Open the bundle
folio = DataFolio('{self.path}')

# View contents
folio.describe()

# Access data
df = folio.get_table('table_name')
model = folio.get_model('model_name')
array = folio.get_numpy('array_name')
```

## Documentation

For more information, see the [datafolio documentation](https://github.com/ceesem/datafolio).
"""

        readme_path = self._storage.join_paths(self._bundle_dir, "README.md")

        # Write README based on storage type
        if is_cloud_path(self._bundle_dir):
            # For cloud storage, use cloudfiles
            from cloudfiles import CloudFiles

            parts = readme_path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(readme_path)
            cf.put(filename, readme_content.encode("utf-8"))
        else:
            # For local storage, write directly
            with open(readme_path, "w") as f:
                f.write(readme_content)

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
        from datafolio.utils import SNAPSHOTS_FILE

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

            # Handle both old format (list) and new format (dict with items)
            if isinstance(items_data, list):
                # Old format: just a list of items (backward compatibility)
                items_list = items_data
            else:
                # New format: dict with items list
                items_list = items_data.get("items", [])

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

    def _save_items(self) -> None:
        """Save unified items.json manifest in simplified snapshot format."""
        if self._batch_mode:
            return

        path = self._storage.join_paths(self._bundle_dir, ITEMS_FILE)

        # Combine current versions and snapshot versions
        all_items = list(self._items.values()) + self._snapshot_versions

        # Save in simplified format: dict with items list only
        items_data = {
            "items": all_items,
        }
        self._storage.write_json(path, items_data)

        # Update metadata timestamp when items change
        # This allows other instances to detect staleness
        if hasattr(self, "metadata"):
            from datetime import datetime, timezone

            # Use super() to update without triggering another save
            super(MetadataDict, self.metadata).__setitem__(
                "updated_at", datetime.now(timezone.utc).isoformat()
            )
            self._save_metadata()

        # Sync data accessor to update autocomplete immediately
        self._sync_data_accessor()

    def _save_snapshots(self) -> None:
        """Save snapshots.json manifest."""
        from datafolio.utils import SNAPSHOTS_FILE

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
                cwd=self._bundle_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None

            # Get commit hash
            commit_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._bundle_dir,
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
                cwd=self._bundle_dir,
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
                cwd=self._bundle_dir,
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
                cwd=self._bundle_dir,
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

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
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

        from datetime import datetime, timezone

        from datafolio.utils import validate_snapshot_name

        # Validate snapshot name
        validate_snapshot_name(name)

        # Check if snapshot already exists
        if name in self._snapshots:
            raise ValueError(f"Snapshot '{name}' already exists")

        # Capture current item versions (using checksum as version identifier)
        item_versions: Dict[str, str] = {}
        for item_name, item_meta in self._items.items():
            # Use checksum as version identifier to detect actual content changes
            # This is more reliable than filename since filenames can be reused
            checksum = item_meta.get("checksum", "")

            # For cloud files or items without checksums, generate a version ID
            if not checksum:
                import uuid

                # Generate a consistent version ID based on item metadata
                # Use created_at timestamp if available, otherwise generate new UUID
                if "created_at" in item_meta:
                    version_id = f"v_{item_meta['created_at']}"
                else:
                    version_id = f"v_{uuid.uuid4().hex[:8]}"
                item_versions[item_name] = version_id
            else:
                item_versions[item_name] = checksum

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
            # Use most recent snapshot
            if not self._snapshots:
                raise ValueError("No snapshots exist. Create a snapshot first.")
            snapshots_list = self.list_snapshots()
            if not snapshots_list:
                raise ValueError("No snapshots exist. Create a snapshot first.")
            snapshot = snapshots_list[-1]["name"]

        if snapshot not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot}' not found")

        snapshot_meta = self._snapshots[snapshot]
        snapshot_items = snapshot_meta.get("item_versions", {})

        # Get current item versions (name -> checksum mapping)
        current_items = {
            item["name"]: item["checksum"] for item in self._items.values()
        }

        # Find item differences
        snapshot_names = set(snapshot_items.keys())
        current_names = set(current_items.keys())

        added_items = sorted(current_names - snapshot_names)
        removed_items = sorted(snapshot_names - current_names)

        # Check for modified items (different checksums)
        shared_names = snapshot_names & current_names
        modified_items = []
        unchanged_items = []

        for item_name in shared_names:
            # Compare checksums
            snapshot_checksum = snapshot_items[item_name]
            current_checksum = current_items[item_name]

            if snapshot_checksum != current_checksum:
                modified_items.append(item_name)
            else:
                unchanged_items.append(item_name)

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
            item_name = item.get("name")
            item_type = item.get("item_type")
            filename = item.get("filename")

            if not dry_run and filename:
                # Delete the physical file directly by filename
                # Don't use handler.delete() as that would delete from _items
                try:
                    # Determine storage directory
                    if item_type == "included_table":
                        subdir = TABLES_DIR
                    elif item_type in ("model", "sklearn_model", "pytorch_model"):
                        subdir = MODELS_DIR
                    else:
                        subdir = ARTIFACTS_DIR

                    # Delete the snapshot version file
                    file_path = self._storage.join_paths(
                        self._bundle_dir, subdir, filename
                    )
                    if self._storage.exists(file_path):
                        self._storage.delete_file(file_path)
                except Exception:
                    # Deletion failed - still remove from manifest
                    pass

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

        # Restore metadata
        self.metadata.clear()
        self.metadata.update(snap_metadata)

        # Find items to remove (items not in snapshot)
        current_item_names = set(self._items.keys())
        snapshot_item_names = set(snap_items.keys())
        items_to_remove = current_item_names - snapshot_item_names

        # Remove items not in snapshot
        for item_name in items_to_remove:
            item = self._items[item_name]
            item_type = item.get("item_type")

            # Delete physical file
            from datafolio.base.registry import get_handler

            try:
                handler = get_handler(item_type)
                handler.delete(folio=self, name=item_name)
            except (KeyError, Exception):
                pass

            del self._items[item_name]

        # For items in snapshot, restore them from snapshot version if needed
        for item_name, snapshot_checksum in snap_items.items():
            if item_name in self._items:
                current_item = self._items[item_name]
                current_checksum = current_item.get("checksum", "")

                # If different version, need to restore from snapshot version
                if current_checksum != snapshot_checksum:
                    # Find the snapshot version item
                    snapshot_item = None
                    for item in self._snapshot_versions:
                        if item.get("name") == item_name and snapshot in item.get(
                            "in_snapshots", []
                        ):
                            snapshot_item = item
                            break

                    if snapshot_item:
                        # Copy snapshot version file to current location
                        from datafolio.base.registry import get_handler

                        try:
                            handler = get_handler(snapshot_item["item_type"])

                            # Get paths
                            snapshot_filename = snapshot_item.get("filename")
                            current_filename = current_item.get("filename")

                            if snapshot_filename and current_filename:
                                # Determine storage directory
                                item_type = snapshot_item.get("item_type", "")
                                if item_type == "included_table":
                                    subdir = TABLES_DIR
                                elif item_type in (
                                    "model",
                                    "sklearn_model",
                                    "pytorch_model",
                                ):
                                    subdir = MODELS_DIR
                                else:
                                    subdir = ARTIFACTS_DIR

                                # Copy snapshot file to current file
                                snapshot_path = self._storage.join_paths(
                                    self._bundle_dir, subdir, snapshot_filename
                                )
                                current_path = self._storage.join_paths(
                                    self._bundle_dir, subdir, current_filename
                                )

                                if self._storage.exists(snapshot_path):
                                    # Delete current file and copy snapshot file
                                    if self._storage.exists(current_path):
                                        self._storage.delete_file(current_path)
                                    self._storage.copy_file(snapshot_path, current_path)

                                    # Update item metadata to match snapshot
                                    current_item.update(
                                        {
                                            "checksum": snapshot_item.get("checksum"),
                                            "num_rows": snapshot_item.get("num_rows"),
                                            "num_cols": snapshot_item.get("num_cols"),
                                            "dtypes": snapshot_item.get("dtypes"),
                                            "columns": snapshot_item.get("columns"),
                                        }
                                    )
                        except (KeyError, Exception):
                            # Handler not found or restore failed
                            pass

        # Save updated state
        self._save_items()
        self._save_metadata()

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

        # Copy all items from snapshot
        for item_name in snapshot_folio._items.keys():
            item_meta = snapshot_folio._items[item_name]
            item_type = item_meta["item_type"]

            # Get the handler for this item type
            from datafolio.base.registry import get_handler

            handler = get_handler(item_type)

            if handler:
                # Load the item from snapshot
                item_data = handler.get(snapshot_folio, item_name)

                # Add to new folio - handler writes data and returns metadata
                new_metadata = handler.add(new_folio, item_name, item_data)

                # Initialize snapshot fields for new items
                if "in_snapshots" not in new_metadata:
                    new_metadata["in_snapshots"] = []
                if "is_current" not in new_metadata:
                    new_metadata["is_current"] = True

                # Store metadata
                new_folio._items[item_name] = new_metadata

        # Save items manifest
        new_folio._save_items()

        return new_folio

    def _find_item_by_checksum(
        self, name: str, checksum: str
    ) -> Optional[Dict[str, Any]]:
        """Find item by name and checksum.

        Args:
            name: Item name
            checksum: Checksum to match

        Returns:
            Item metadata dict or None if not found
        """
        # Check current items
        if name in self._items:
            item = self._items[name]
            if item.get("checksum") == checksum:
                return item

        # Check snapshot versions
        for item in self._snapshot_versions:
            if item.get("name") == name and item.get("checksum") == checksum:
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

        # Return a copy of the snapshot metadata
        return dict(self._snapshots[snapshot])

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

    def _rename_to_snapshot_version(self, name: str) -> None:
        """Rename current item file to snapshot version format.

        When an item is in snapshots and needs to be overwritten, this method:
        1. Renames the current file from <name>.<ext> to <name>@<snapshot>.<ext>
        2. Uses the first snapshot name from in_snapshots list
        3. Updates the item metadata to mark it as not current

        Args:
            name: Item name to rename

        Raises:
            ValueError: If item not found or not in any snapshots
        """
        if name not in self._items:
            raise ValueError(f"Item '{name}' not found")

        item = self._items[name]
        in_snapshots = item.get("in_snapshots", [])

        if not in_snapshots:
            raise ValueError(f"Item '{name}' is not in any snapshots")

        # Get the first snapshot name to use in filename
        first_snapshot = in_snapshots[0]

        # Get current filename and create snapshot version filename
        current_filename = item.get("filename")
        if not current_filename:
            # For referenced tables, no file to rename
            return

        # Create snapshot version filename: <name>@<snapshot>.<ext>
        # Extract extension from current filename
        from pathlib import Path

        file_path = Path(current_filename)
        name_part = file_path.stem
        ext_part = file_path.suffix

        # Create new filename with @snapshot
        snapshot_filename = f"{name_part}@{first_snapshot}{ext_part}"

        # Get storage subdir based on item type
        item_type = item.get("item_type", "")
        if item_type == "included_table":
            subdir = TABLES_DIR
        elif item_type in ("model", "sklearn_model", "pytorch_model"):
            subdir = MODELS_DIR
        elif item_type in ("artifact", "numpy_array", "json_data", "timestamp"):
            subdir = ARTIFACTS_DIR
        else:
            # Referenced tables don't have files to rename
            return

        # Build full paths
        old_path = self._storage.join_paths(self._bundle_dir, subdir, current_filename)
        new_path = self._storage.join_paths(self._bundle_dir, subdir, snapshot_filename)

        # Rename the file
        if self._storage.exists(old_path):
            # For cloud storage, this is copy + delete
            # For local storage, this is os.rename
            self._storage.copy_file(old_path, new_path)
            self._storage.delete_file(old_path)

        # Update item metadata
        item["filename"] = snapshot_filename
        item["is_current"] = False

    def _handle_copy_on_write(self, name: str) -> None:
        """Handle copy-on-write logic when overwriting an item.

        If the item exists and is in snapshots:
        1. Rename current file to @snapshot version
        2. Move old metadata to _snapshot_versions list
        3. Caller will create new current entry in _items

        Args:
            name: Item name being added/overwritten
        """
        if self._is_in_snapshots(name):
            # Item is in snapshots - need to preserve it
            self._rename_to_snapshot_version(name)

            # Move old item from _items to _snapshot_versions
            old_item = self._items[name]
            self._snapshot_versions.append(old_item)
            # Note: caller will add new item to _items with same name

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
        if self._check_if_stale():
            self.refresh()

    def _get_with_cache(self, name: str, handler_get_fn: callable) -> Any:
        """Get an item with caching if enabled.

        Args:
            name: Item name
            handler_get_fn: Function to call if cache miss (handler.get)

        Returns:
            Item data (from cache or remote)
        """
        # If caching not enabled or not a cloud bundle, use handler directly
        if not self._cache_manager:
            return handler_get_fn()

        # Get item metadata
        item = self._items[name]
        item_type = item.get("item_type")
        filename = item.get("filename")
        remote_checksum = item.get("checksum")

        # Define fetch function for cache manager
        def fetch_from_remote():
            # Read data from remote using handler
            data = handler_get_fn()

            # Serialize to bytes for caching
            import tempfile
            from pathlib import Path

            with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
                tmp_path = Path(tmp.name)

            try:
                # Write data to temp file using storage backend
                if item_type == "included_table":
                    self._storage.write_parquet(str(tmp_path), data)
                elif item_type == "model":
                    import joblib

                    joblib.dump(data, tmp_path)
                elif item_type in ("pytorch_model", "pytorch_state_dict"):
                    import torch

                    torch.save(data, tmp_path)
                else:
                    # For other types, try generic serialization
                    import joblib

                    joblib.dump(data, tmp_path)

                # Read bytes
                data_bytes = tmp_path.read_bytes()
                return (data_bytes, item_type, filename)
            finally:
                tmp_path.unlink()

        # Try to get from cache
        cache_path = self._cache_manager.get(
            name, remote_fetch_fn=fetch_from_remote, remote_checksum=remote_checksum
        )

        if cache_path is None:
            # No cache, use handler directly
            return handler_get_fn()

        # Load from cache file
        if item_type == "included_table":
            return self._storage.read_parquet(str(cache_path))
        elif item_type == "model":
            import joblib

            return joblib.load(cache_path)
        elif item_type in ("pytorch_model", "pytorch_state_dict"):
            import torch

            return torch.load(cache_path, weights_only=False)
        else:
            # Generic deserialization
            import joblib

            return joblib.load(cache_path)

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

        Examples:
            >>> with folio.batch():
            ...     for i in range(100):
            ...         folio.add_numpy(f'array_{i}', arr)
            # items.json saved once at end of block
        """
        self._batch_mode = True
        try:
            yield
        finally:
            self._batch_mode = False
            self._save_items()

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
                # For references, just check existence
                path = item.get("path")
                if path:
                    is_valid = self._storage.exists(path)
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

    def list_contents(self) -> Dict[str, list[str]]:
        """List all contents in the DataFolio.

        Returns:
            Dictionary with keys 'referenced_tables', 'included_tables', 'numpy_arrays',
            'json_data', 'timestamps', 'models', 'pytorch_models', and 'artifacts',
            each containing a list of names

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.reference_table('data1', path='s3://bucket/data.parquet')
            >>> folio.add_numpy('embeddings', np.array([1, 2, 3]))
            >>> folio.list_contents()
            {'referenced_tables': ['data1'], 'included_tables': [], 'numpy_arrays': ['embeddings'],
             'json_data': [], 'timestamps': [], 'models': [], 'pytorch_models': [], 'artifacts': []}
        """
        # Auto-refresh if bundle was updated externally
        self._refresh_if_needed()

        # Filter items by type
        referenced_tables = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "referenced_table"
        ]
        included_tables = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "included_table"
        ]
        numpy_arrays = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "numpy_array"
        ]
        json_data = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "json_data"
        ]
        timestamps = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "timestamp"
        ]
        models = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "model"
        ]
        pytorch_models = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "pytorch_model"
        ]
        artifacts = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "artifact"
        ]

        return {
            "referenced_tables": referenced_tables,
            "included_tables": included_tables,
            "numpy_arrays": numpy_arrays,
            "json_data": json_data,
            "timestamps": timestamps,
            "models": models,
            "pytorch_models": pytorch_models,
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
    def pytorch_models(self) -> list[str]:
        """Get list of PyTorch model names.

        Returns:
            List of PyTorch model names (strings)

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.add_pytorch('neural_net', model)
            >>> folio.pytorch_models
            ['neural_net']
        """
        return [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "pytorch_model"
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
        from datafolio.utils import is_cloud_path

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
        return_string: bool = False,
        show_empty: bool = False,
        max_metadata_fields: int = 10,
        snapshot: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a human-readable description of all items in the bundle.

        Includes lineage information showing inputs and dependencies.

        Args:
            return_string: If True, return as string instead of printing
            show_empty: If True, show empty sections
            max_metadata_fields: Maximum metadata fields to show
            snapshot: Optional snapshot name to describe instead of the full bundle

        Returns:
            None if return_string=False, otherwise the description string

        Examples:
            >>> folio.describe()  # Show full bundle
            >>> folio.describe(snapshot='v1.0')  # Show specific snapshot

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
    ) -> Self:
        """Add a reference to an external table (not copied to bundle).

        Writes immediately to items.json.

        Args:
            name: Unique name for this table
            path: Path to the table (local or cloud)
            table_format: Format of the table ('parquet', 'delta', 'csv')
            num_rows: Optional number of rows
            version: Optional version number (for Delta tables)
            description: Optional description
            inputs: Optional list of items this was derived from
            code: Optional code snippet that created this

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists or format is invalid

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.reference_table(
            ...     'raw_data',
            ...     path='s3://bucket/data.parquet',
            ...     table_format='parquet',
            ...     num_rows=1_000_000
            ... )
        """
        from datafolio.utils import validate_table_format

        # Validate inputs
        if name in self._items:
            raise ValueError(f"Item '{name}' already exists in this DataFolio")

        validate_table_format(table_format)

        # Get handler and delegate metadata creation
        registry = get_registry()
        handler = registry.get("referenced_table")

        # Handler builds metadata (path resolution happens in handler)
        metadata = handler.add(
            self,
            name,
            str(path),
            description=description,
            inputs=inputs,
            table_format=table_format,
        )

        # Validate item name
        from datafolio.utils import validate_item_name

        validate_item_name(name)

        # Add extra fields not handled by base handler
        if num_rows is not None:
            metadata["num_rows"] = num_rows
        if version is not None:
            metadata["version"] = version
        if code is not None:
            metadata["code"] = code

        # Initialize snapshot fields for new items
        if "in_snapshots" not in metadata:
            metadata["in_snapshots"] = []
        if "is_current" not in metadata:
            metadata["is_current"] = True

        self._items[name] = metadata

        # Write immediately
        self._save_items()

        return self

    def add_table(
        self,
        name: str,
        data: Any,  # pandas.DataFrame
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
            data: DataFrame to include
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
        from datafolio.utils import validate_item_name

        validate_item_name(name)

        # Handle overwriting logic
        if name in self._items:
            if self._is_in_snapshots(name):
                # Item is in snapshots - must preserve it via copy-on-write
                self._handle_copy_on_write(name)
            elif not overwrite:
                # Item exists but not in snapshots - respect overwrite flag
                raise ValueError(
                    f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
                )
            # else: overwrite=True and not in snapshots, so just replace it

        # Get handler and delegate storage + metadata creation
        registry = get_registry()
        handler = registry.get("included_table")

        # Handler builds metadata and writes data
        metadata = handler.add(
            self,
            name,
            data,
            description=description,
            inputs=inputs,
        )

        # Add extra fields not handled by base handler
        if models is not None:
            metadata["models"] = models
        if code is not None:
            metadata["code"] = code

        # Initialize snapshot fields for new items
        if "in_snapshots" not in metadata:
            metadata["in_snapshots"] = []
        if "is_current" not in metadata:
            metadata["is_current"] = True

        self._items[name] = metadata

        # Update manifest
        self._save_items()

        return self

    def get_table(self, name: str) -> Any:  # Returns pandas.DataFrame
        """Get a table by name (works for both included and referenced).

        For included tables, reads from bundle directory.
        For referenced tables, reads from the specified external path.
        If caching is enabled (cache_enabled=True), cloud-based tables are cached locally
        for faster repeated access.

        Args:
            name: Name of the table

        Returns:
            pandas DataFrame

        Raises:
            KeyError: If table name doesn't exist
            ImportError: If reading from cloud requires missing dependencies
            FileNotFoundError: If referenced file doesn't exist

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> import pandas as pd
            >>> df = pd.DataFrame({'a': [1, 2, 3]})
            >>> folio.add_table('test', df)
            >>> retrieved = folio.get_table('test')
            >>> assert len(retrieved) == 3
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

        # Get handler and delegate to it (with caching if enabled)
        registry = get_registry()
        handler = registry.get(item_type)
        return self._get_with_cache(name, lambda: handler.get(self, name))

    def get_data_path(self, name: str) -> str:
        """Get the path to a referenced table.

        Args:
            name: Name of the referenced table

        Returns:
            Path to the table

        Raises:
            KeyError: If table name doesn't exist
            ValueError: If table is included (not a reference)

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.reference_table('data', path='s3://bucket/file.parquet')
            >>> path = folio.get_data_path('data')
            >>> assert 's3://bucket/file.parquet' in path
        """
        if name not in self._items:
            raise KeyError(f"Table '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        if item_type == "included_table":
            raise ValueError(
                f"Table '{name}' is included in bundle, not a reference. "
                "Use get_table() instead."
            )
        elif item_type == "referenced_table":
            return item["path"]
        else:
            raise ValueError(f"Item '{name}' is not a table (type: {item_type})")

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
    ) -> Self:
        """Add a scikit-learn style model to the bundle.

        Writes immediately to models/ directory and updates items.json.
        Uses joblib for serialization.

        Args:
            name: Unique name for this model
            model: Trained model to include
            description: Optional description
            overwrite: If True, allow overwriting existing model (default: False)
            inputs: Optional list of table names used for training
            hyperparameters: Optional dict of hyperparameters
            code: Optional code snippet that trained this model

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
        """
        # Validate inputs
        if not overwrite and name in self._items:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
            )

        # Get handler and delegate storage + metadata creation
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("model")

        # Handler builds metadata and writes model
        metadata = handler.add(
            self, name, model, description=description, inputs=inputs
        )

        # Validate item name
        from datafolio.utils import validate_item_name

        validate_item_name(name)

        # Add extra fields not handled by base handler
        if hyperparameters is not None:
            metadata["hyperparameters"] = hyperparameters
        if code is not None:
            metadata["code"] = code

        # Initialize snapshot fields for new items
        if "in_snapshots" not in metadata:
            metadata["in_snapshots"] = []
        if "is_current" not in metadata:
            metadata["is_current"] = True

        self._items[name] = metadata

        self._save_items()

        return self

    def add_model(
        self,
        name: str,
        model: Any,
        description: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> Self:
        """Add a model with automatic framework detection.

        Automatically detects PyTorch vs sklearn-style models and uses the
        appropriate serialization method. For fine-grained control, use
        add_sklearn() or add_pytorch().

        Args:
            name: Unique name for this model
            model: Trained model (PyTorch or sklearn-style)
            description: Optional description
            overwrite: If True, allow overwriting existing model (default: False)
            **kwargs: Additional arguments passed to the specific method
                (e.g., init_args for PyTorch, hyperparameters for sklearn)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists and overwrite=False

        Examples:
            Sklearn model:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> model = RandomForestClassifier()
            >>> folio.add_model('clf', model, hyperparameters={'n_estimators': 100})

            PyTorch model:
            >>> import torch.nn as nn
            >>> model = MyNeuralNet(input_dim=10, hidden_dim=50)
            >>> folio.add_model('nn', model, init_args={'input_dim': 10, 'hidden_dim': 50})
        """
        # Check if PyTorch model
        if hasattr(model, "state_dict") and hasattr(model, "load_state_dict"):
            try:
                import torch.nn as nn

                if isinstance(model, nn.Module):
                    return self.add_pytorch(
                        name,
                        model,
                        description=description,
                        overwrite=overwrite,
                        **kwargs,
                    )
            except ImportError:
                pass

        # Otherwise, assume sklearn-style (joblib-serializable)
        return self.add_sklearn(
            name, model, description=description, overwrite=overwrite, **kwargs
        )

    def get_sklearn(self, name: str) -> Any:
        """Get a scikit-learn style model by name.

        If caching is enabled (cache_enabled=True), cloud-based models are cached locally
        for faster repeated access.

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

        # Delegate to handler (with caching if enabled)
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("model")
        return self._get_with_cache(name, lambda: handler.get(self, name))

    def get_model(self, name: str, **kwargs) -> Any:
        """Get a model by name with automatic type detection.

        Automatically detects whether the model is PyTorch or sklearn-style
        and uses the appropriate loader. For fine-grained control, use
        get_sklearn() or get_pytorch().

        If caching is enabled (cache_enabled=True), cloud-based models are cached locally
        for faster repeated access.

        Args:
            name: Name of the model
            **kwargs: Additional arguments passed to get_pytorch() if it's a PyTorch model
                (e.g., model_class, reconstruct)

        Returns:
            The model object

        Raises:
            KeyError: If model name doesn't exist
            ValueError: If named item is not a model

        Examples:
            >>> folio = DataFolio('experiments/test')
            >>> # Works for both sklearn and PyTorch models
            >>> sklearn_model = folio.get_model('classifier')
            >>> pytorch_model = folio.get_model('neural_net', model_class=MyModel)
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
        elif item_type == "pytorch_model":
            return self.get_pytorch(name, **kwargs)
        else:
            raise ValueError(f"Item '{name}' is not a model (type: {item_type})")

    def add_pytorch(
        self,
        name: str,
        model: Any,
        description: Optional[str] = None,
        overwrite: bool = False,
        inputs: Optional[list[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        init_args: Optional[Dict[str, Any]] = None,
        save_class: bool = False,
        code: Optional[str] = None,
    ) -> Self:
        """Add a PyTorch model to the bundle.

        Saves the model's state_dict along with metadata about the model class.
        Follows PyTorch best practices by saving state_dict rather than full model.

        Args:
            name: Unique name for this model
            model: PyTorch model to save (state_dict will be extracted)
            description: Optional description
            overwrite: If True, allow overwriting existing model (default: False)
            inputs: Optional list of table names used for training
            hyperparameters: Optional dict of hyperparameters
            init_args: Optional dict of arguments needed to instantiate the model class
            save_class: If True, use dill to serialize the model class for reconstruction
            code: Optional code snippet that trained this model

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists and overwrite=False
            ImportError: If torch (or dill when save_class=True) is not installed

        Examples:
            Basic usage:
            >>> import torch.nn as nn
            >>> class MyModel(nn.Module):
            ...     def __init__(self, input_dim, hidden_dim):
            ...         super().__init__()
            ...         self.fc1 = nn.Linear(input_dim, hidden_dim)
            ...         self.fc2 = nn.Linear(hidden_dim, 1)
            ...     def forward(self, x):
            ...         return self.fc2(torch.relu(self.fc1(x)))
            >>> model = MyModel(10, 50)
            >>> # ... train model ...
            >>> folio.add_pytorch('my_model', model,
            ...     description='Simple feedforward network',
            ...     inputs=['training_data'],
            ...     hyperparameters={'input_dim': 10, 'hidden_dim': 50},
            ...     init_args={'input_dim': 10, 'hidden_dim': 50})

            With class serialization:
            >>> folio.add_pytorch('my_model', model,
            ...     save_class=True,  # Saves the class definition with dill
            ...     init_args={'input_dim': 10, 'hidden_dim': 50})
        """
        # Validate inputs
        if not overwrite and name in self._items:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
            )

        # Get handler and delegate storage + metadata creation
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("pytorch_model")

        # Handler builds metadata and writes model
        metadata = handler.add(
            self,
            name,
            model,
            description=description,
            inputs=inputs,
            init_args=init_args,
            save_class=save_class,
        )

        # Validate item name
        from datafolio.utils import validate_item_name

        validate_item_name(name)

        # Add extra fields not handled by base handler
        if hyperparameters is not None:
            metadata["hyperparameters"] = hyperparameters
        if code is not None:
            metadata["code"] = code
        if save_class:
            metadata["has_serialized_class"] = True

        # Initialize snapshot fields for new items
        if "in_snapshots" not in metadata:
            metadata["in_snapshots"] = []
        if "is_current" not in metadata:
            metadata["is_current"] = True

        self._items[name] = metadata

        self._save_items()

        return self

    def get_pytorch(
        self,
        name: str,
        model_class: Optional[type] = None,
        reconstruct: bool = True,
    ) -> Any:
        """Get a PyTorch model by name.

        Can return either the state_dict only, or a reconstructed model.
        Supports caching for cloud bundles when caching is enabled.

        Args:
            name: Name of the model
            model_class: Optional model class to use for reconstruction.
                If provided, must accept **init_args from manifest.
            reconstruct: If True, attempt to reconstruct the model (default: True).
                If False, returns just the state_dict.

        Returns:
            If reconstruct=False: Returns state_dict dictionary
            If reconstruct=True: Returns reconstructed model with weights loaded

        Raises:
            KeyError: If model name doesn't exist
            ValueError: If named item is not a pytorch_model
            ImportError: If torch is not installed
            RuntimeError: If reconstruction fails

        Examples:
            Get state_dict only:
            >>> folio = DataFolio('experiments/test')
            >>> state_dict = folio.get_pytorch('my_model', reconstruct=False)
            >>> # Manually load into your model
            >>> model = MyModel(10, 50)
            >>> model.load_state_dict(state_dict)

            Reconstruct with provided class:
            >>> model = folio.get_pytorch('my_model', model_class=MyModel)

            Auto-reconstruct (requires model class to be importable):
            >>> model = folio.get_pytorch('my_model')  # Tries to find class automatically
        """
        if name not in self._items:
            raise KeyError(f"Model '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "pytorch_model":
            raise ValueError(
                f"Item '{name}' is not a PyTorch model (type: {item.get('item_type')})"
            )

        # Delegate to handler (with caching if enabled)
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("pytorch_model")

        # Use caching if available
        return self._get_with_cache(
            name,
            lambda: handler.get(
                self, name, model_class=model_class, reconstruct=reconstruct
            ),
        )

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
        # Validate inputs
        if not overwrite and name in self._items:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
            )

        # Get handler and delegate storage + metadata creation
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("artifact")

        # Handler builds metadata and copies file
        metadata = handler.add(self, name, str(path), description=description)

        # Validate item name
        from datafolio.utils import validate_item_name

        validate_item_name(name)

        # Add extra fields not handled by base handler
        if category is not None:
            metadata["category"] = category

        # Initialize snapshot fields for new items
        if "in_snapshots" not in metadata:
            metadata["in_snapshots"] = []
        if "is_current" not in metadata:
            metadata["is_current"] = True

        self._items[name] = metadata

        self._save_items()

        return self

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
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("artifact")
        return handler.get(self, name)

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
        # Validate inputs
        if not overwrite and name in self._items:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
            )

        # Get handler and delegate storage + metadata creation
        registry = get_registry()
        handler = registry.get("numpy_array")

        # Handler builds metadata and writes data
        metadata = handler.add(
            self,
            name,
            array,
            description=description,
            inputs=inputs,
        )

        # Validate item name
        from datafolio.utils import validate_item_name

        validate_item_name(name)

        # Add extra fields not handled by base handler
        if code is not None:
            metadata["code"] = code

        # Initialize snapshot fields for new items
        if "in_snapshots" not in metadata:
            metadata["in_snapshots"] = []
        if "is_current" not in metadata:
            metadata["is_current"] = True

        self._items[name] = metadata

        # Update manifest
        self._save_items()

        return self

    def get_numpy(self, name: str) -> Any:
        """Get a numpy array by name.

        Arrays are NOT cached - always read fresh from disk.

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
        # Validate inputs
        if not overwrite and name in self._items:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
            )

        # Get handler and delegate storage + metadata creation
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("json_data")

        # Handler builds metadata and writes data
        metadata = handler.add(self, name, data, description=description, inputs=inputs)

        # Validate item name
        from datafolio.utils import validate_item_name

        validate_item_name(name)

        # Add extra fields not handled by base handler
        if code is not None:
            metadata["code"] = code

        # Initialize snapshot fields for new items
        if "in_snapshots" not in metadata:
            metadata["in_snapshots"] = []
        if "is_current" not in metadata:
            metadata["is_current"] = True

        self._items[name] = metadata

        self._save_items()

        return self

    def get_json(self, name: str) -> Any:
        """Get JSON data by name.

        Data is NOT cached - always read fresh from disk.

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
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("json_data")
        return handler.get(self, name)

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
        # Validate inputs
        if not overwrite and name in self._items:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
            )

        # Get handler and delegate storage + metadata creation
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("timestamp")

        # Handler builds metadata and writes data
        metadata = handler.add(
            self, name, timestamp, description=description, inputs=inputs
        )

        # Validate item name
        from datafolio.utils import validate_item_name

        validate_item_name(name)

        # Add extra fields not handled by base handler
        if code is not None:
            metadata["code"] = code

        # Initialize snapshot fields for new items
        if "in_snapshots" not in metadata:
            metadata["in_snapshots"] = []
        if "is_current" not in metadata:
            metadata["is_current"] = True

        self._items[name] = metadata

        self._save_items()

        return self

    def get_timestamp(self, name: str, as_unix: bool = False) -> Union[datetime, float]:
        """Get a timestamp by name.

        Timestamps are NOT cached - always read fresh from disk.

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
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("timestamp")
        return handler.get(self, name, as_unix=as_unix)

    def add_data(
        self,
        name: str,
        data: Any = None,
        reference: Optional[Union[str, Path]] = None,
        description: Optional[str] = None,
        **kwargs,
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

        # Auto-detect handler using registry
        from datafolio.base.registry import detect_handler, get_handler

        handler = detect_handler(data)

        # Fallback: primitives (int, float, str, bool, None) -> JSON handler
        # These don't auto-detect to avoid conflicts, but add_data() accepts them
        if handler is None and isinstance(data, (int, float, str, bool, type(None))):
            handler = get_handler("json_data")

        if handler is None:
            raise TypeError(
                f"Unsupported data type: {type(data).__name__}. "
                f"No handler found for this type. "
                f"Supported types: pandas.DataFrame, numpy.ndarray, torch.nn.Module, "
                f"sklearn models, dict, list, datetime, file paths, or JSON scalars. "
                f"Use explicit add_*() methods for more control."
            )

        # Validate item name
        from datafolio.utils import validate_item_name

        validate_item_name(name)

        # Use handler to add data
        metadata = handler.add(
            folio=self,
            name=name,
            data=data,
            description=description,
            inputs=kwargs.get("inputs"),
            **kwargs,
        )

        # Initialize snapshot fields for new items
        if "in_snapshots" not in metadata:
            metadata["in_snapshots"] = []
        if "is_current" not in metadata:
            metadata["is_current"] = True

        self._items[name] = metadata

        self._save_items()

        return self

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
        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        # Only allow "data" types - not models or artifacts
        data_types = (
            "referenced_table",
            "included_table",
            "numpy_array",
            "json_data",
            "timestamp",
        )
        if item_type not in data_types:
            raise ValueError(
                f"Item '{name}' is not a data item (type: {item_type}). "
                f"Use get_model() for models or get_artifact_path() for artifacts."
            )

        # Get handler from registry and use it to retrieve data
        from datafolio.base.registry import get_handler

        try:
            handler = get_handler(item_type)
            return handler.get(folio=self, name=name)
        except KeyError:
            # No handler registered for this type
            raise ValueError(
                f"Item '{name}' has unknown type '{item_type}'. "
                f"No handler registered for this type. "
                f"Use type-specific get methods if available."
            )

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

        # Delete each item
        for item_name in names_to_delete:
            item = self._items[item_name]
            item_type = item.get("item_type")

            # Check for dependents and warn if requested
            if warn_dependents:
                dependents = self.get_dependents(item_name)
                if dependents:
                    warnings.warn(
                        f"Deleting '{item_name}' which is used by: {', '.join(dependents)}. "
                        f"Those items may have broken lineage.",
                        UserWarning,
                        stacklevel=2,
                    )

            # Delete physical file using handler
            from datafolio.base.registry import get_handler

            try:
                handler = get_handler(item_type)
                handler.delete(folio=self, name=item_name)
            except KeyError:
                # No handler for this type - skip file deletion
                # (e.g., unknown/legacy item types)
                pass

            # Remove from items manifest
            del self._items[item_name]

        # Save updated manifest
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

    def copy(
        self,
        new_path: Union[str, Path],
        metadata_updates: Optional[Dict[str, Any]] = None,
        include_items: Optional[list[str]] = None,
        exclude_items: Optional[list[str]] = None,
        random_suffix: bool = False,
    ) -> "DataFolio":
        """Create a copy of this bundle with a new name.

        Useful for creating derived experiments or checkpoints.

        Args:
            new_path: Full path for new bundle directory
            metadata_updates: Metadata fields to update/add in the copy
            include_items: If specified, only copy these items (by name)
            exclude_items: Items to exclude from copy (by name)
            random_suffix: If True, append random suffix to new bundle name (default: False)

        Returns:
            New DataFolio instance

        Raises:
            ValueError: If include_items and exclude_items are both specified

        Examples:
            >>> # Simple copy
            >>> folio2 = folio.copy('experiments/exp-v2')

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
        """
        import shutil

        if include_items is not None and exclude_items is not None:
            raise ValueError("Cannot specify both include_items and exclude_items")

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
            items_to_copy = set(self._items.keys())
            if include_items is not None:
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

                elif item_type == "included_table":
                    # Copy the parquet file
                    src_path = self._storage.join_paths(
                        self._bundle_dir, TABLES_DIR, item["filename"]
                    )
                    dst_path = self._storage.join_paths(
                        new_folio._bundle_dir, TABLES_DIR, item["filename"]
                    )
                    if not is_cloud_path(src_path):
                        shutil.copy2(src_path, dst_path)
                    else:
                        # For cloud paths, use cloudfiles
                        # Extract directory and filename from paths
                        src_parts = src_path.rsplit("/", 1)
                        src_dir = src_parts[0] if len(src_parts) == 2 else ""
                        src_filename = src_parts[-1]

                        dst_parts = dst_path.rsplit("/", 1)
                        dst_dir = dst_parts[0] if len(dst_parts) == 2 else ""
                        dst_filename = dst_parts[-1]

                        src_cf = (
                            cloudfiles.CloudFiles(src_dir)
                            if src_dir
                            else cloudfiles.CloudFiles(src_path)
                        )
                        content = src_cf.get(src_filename)
                        dst_cf = (
                            cloudfiles.CloudFiles(dst_dir)
                            if dst_dir
                            else cloudfiles.CloudFiles(dst_path)
                        )
                        dst_cf.put(dst_filename, content)

                    new_folio._items[item_name] = dict(item)

                elif item_type == "model":
                    # Copy the model file
                    src_path = self._storage.join_paths(
                        self._bundle_dir, MODELS_DIR, item["filename"]
                    )
                    dst_path = self._storage.join_paths(
                        new_folio._bundle_dir, MODELS_DIR, item["filename"]
                    )
                    if not is_cloud_path(src_path):
                        shutil.copy2(src_path, dst_path)
                    else:
                        # For cloud paths, use cloudfiles
                        # Extract directory and filename from paths
                        src_parts = src_path.rsplit("/", 1)
                        src_dir = src_parts[0] if len(src_parts) == 2 else ""
                        src_filename = src_parts[-1]

                        dst_parts = dst_path.rsplit("/", 1)
                        dst_dir = dst_parts[0] if len(dst_parts) == 2 else ""
                        dst_filename = dst_parts[-1]

                        src_cf = (
                            cloudfiles.CloudFiles(src_dir)
                            if src_dir
                            else cloudfiles.CloudFiles(src_path)
                        )
                        content = src_cf.get(src_filename)
                        dst_cf = (
                            cloudfiles.CloudFiles(dst_dir)
                            if dst_dir
                            else cloudfiles.CloudFiles(dst_path)
                        )
                        dst_cf.put(dst_filename, content)

                    new_folio._items[item_name] = dict(item)

                elif item_type == "artifact":
                    # Copy the artifact file
                    src_path = self._storage.join_paths(
                        self._bundle_dir, ARTIFACTS_DIR, item["filename"]
                    )
                    dst_path = self._storage.join_paths(
                        new_folio._bundle_dir, ARTIFACTS_DIR, item["filename"]
                    )
                    if not is_cloud_path(src_path):
                        shutil.copy2(src_path, dst_path)
                    else:
                        # For cloud paths, use cloudfiles
                        # Extract directory and filename from paths
                        src_parts = src_path.rsplit("/", 1)
                        src_dir = src_parts[0] if len(src_parts) == 2 else ""
                        src_filename = src_parts[-1]

                        dst_parts = dst_path.rsplit("/", 1)
                        dst_dir = dst_parts[0] if len(dst_parts) == 2 else ""
                        dst_filename = dst_parts[-1]

                        src_cf = (
                            cloudfiles.CloudFiles(src_dir)
                            if src_dir
                            else cloudfiles.CloudFiles(src_path)
                        )
                        content = src_cf.get(src_filename)
                        dst_cf = (
                            cloudfiles.CloudFiles(dst_dir)
                            if dst_dir
                            else cloudfiles.CloudFiles(dst_path)
                        )
                        dst_cf.put(dst_filename, content)

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

    # Cache Management Methods

    def cache_status(self, item_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get cache status for an item or entire bundle.

        Args:
            item_name: Name of item to check. If None, returns overall cache stats.

        Returns:
            Dict with cache status information, or None if caching not enabled or item not found.
            For specific items:
                - cached: Whether item is cached
                - cache_path: Path to cached file
                - size_bytes: Size of cached file
                - cached_at: Timestamp when cached
                - last_accessed: Last access timestamp
                - access_count: Number of times accessed
                - ttl_remaining: Seconds until cache expires (None if no TTL)
            For bundle-level (item_name=None):
                - bundle_path: Original bundle path
                - cache_dir: Cache directory path
                - ttl_seconds: TTL in seconds
                - cache_hits: Number of cache hits
                - cache_misses: Number of cache misses
                - cache_hit_rate: Hit rate (0.0-1.0)

        Examples:
            Check if a specific item is cached:
            >>> status = folio.cache_status('my_table')
            >>> if status and status['cached']:
            ...     print(f"Cache expires in {status['ttl_remaining']} seconds")

            Get overall cache statistics:
            >>> stats = folio.cache_status()
            >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        """
        if not self._cache_manager:
            return None

        if item_name is None:
            # Return bundle-level stats
            return self._cache_manager.get_stats()
        else:
            # Return item-level status
            return self._cache_manager.get_status(item_name)

    def clear_cache(self, item_name: Optional[str] = None) -> None:
        """Clear cached items.

        Args:
            item_name: Name of specific item to clear. If None, clears all cached items for this bundle.

        Examples:
            Clear a specific item:
            >>> folio.clear_cache('my_table')

            Clear entire cache:
            >>> folio.clear_cache()
        """
        if not self._cache_manager:
            return

        if item_name is None:
            self._cache_manager.clear_all()
        else:
            self._cache_manager.clear_item(item_name)

    def invalidate_cache(self, item_name: str) -> None:
        """Invalidate cache for an item without deleting the file.

        This marks the cached item as invalid, forcing a re-fetch on next access,
        but keeps the file on disk (useful for stale cache fallback).

        Args:
            item_name: Name of item to invalidate

        Examples:
            Force re-download on next access:
            >>> folio.invalidate_cache('my_table')
            >>> table = folio.get_table('my_table')  # Will re-download
        """
        if not self._cache_manager:
            return

        self._cache_manager.invalidate(item_name)

    def refresh_cache(self, item_name: str) -> None:
        """Refresh cache for an item by re-downloading from remote.

        This is equivalent to invalidating and then fetching the item.

        Args:
            item_name: Name of item to refresh

        Raises:
            ValueError: If item doesn't exist in bundle
            RuntimeError: If caching is not enabled

        Examples:
            >>> folio.refresh_cache('my_table')
        """
        if not self._cache_manager:
            raise RuntimeError("Caching is not enabled for this DataFolio")

        if item_name not in self._items:
            raise ValueError(f"Item '{item_name}' not found in bundle")

        # Invalidate cache
        self._cache_manager.invalidate(item_name)

        # Re-fetch based on item type
        item_meta = self._items[item_name]
        item_type = item_meta["item_type"]

        if item_type in ["parquet_table", "csv_table"]:
            self.get_table(item_name)
        elif item_type == "sklearn_model":
            self.get_sklearn(item_name)
        elif item_type == "pytorch_model":
            self.get_pytorch(item_name)
        else:
            # For other types, just invalidate (don't auto-fetch)
            pass
