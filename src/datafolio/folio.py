"""Main DataFolio class for bundling analysis artifacts."""

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
    ):
        """Initialize a new or open an existing DataFolio.

        If the directory doesn't exist, creates a new bundle.
        If it exists, opens the existing bundle and reads manifests.

        Args:
            path: Full path to bundle directory (local or cloud)
            metadata: Optional dictionary of analysis metadata (for new bundles)
            random_suffix: If True, append random suffix to bundle name (default: False)

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
        """
        # Storage backend for all I/O operations
        self._storage = StorageBackend()

        # Unified items dictionary - all items stored here with item_type
        self._items: Dict[str, Union[TableReference, IncludedTable, IncludedItem]] = {}

        # For storing models/artifacts before writing
        self._models_data: Dict[str, Any] = {}  # Model storage
        self._artifacts_paths: Dict[str, Union[str, Path]] = {}  # Artifact file paths

        # Store random suffix setting for collision retry
        self._use_random_suffix = random_suffix

        # Auto-refresh tracking for multi-instance consistency
        self._auto_refresh_enabled: bool = True  # Can be disabled if needed

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

    # ==================== Well-factored I/O Helper Functions ====================

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
        # Read metadata.json
        metadata_path = self._storage.join_paths(self._bundle_dir, METADATA_FILE)
        if self._storage.exists(metadata_path):
            self._metadata_raw = self._storage.read_json(metadata_path)
        else:
            self._metadata_raw = {}

        # Read unified items.json
        items_path = self._storage.join_paths(self._bundle_dir, ITEMS_FILE)
        if self._storage.exists(items_path):
            items_list = self._storage.read_json(items_path)
            self._items = {item["name"]: item for item in items_list}
        else:
            self._items = {}

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
        """Save unified items.json manifest."""
        path = self._storage.join_paths(self._bundle_dir, ITEMS_FILE)
        self._storage.write_json(path, list(self._items.values()))

        # Update metadata timestamp when items change
        # This allows other instances to detect staleness
        if hasattr(self, "metadata"):
            from datetime import datetime, timezone

            # Use super() to update without triggering another save
            super(MetadataDict, self.metadata).__setitem__(
                "updated_at", datetime.now(timezone.utc).isoformat()
            )
            self._save_metadata()

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
        """
        return DataAccessor(self)

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
        return f"DataFolio(bundle_dir='{self._bundle_dir}', items={total_items})"

    def describe(
        self,
        return_string: bool = False,
        show_empty: bool = False,
        max_metadata_fields: int = 10,
    ) -> Optional[str]:
        """Generate a human-readable description of all items in the bundle.

        Includes lineage information showing inputs and dependencies.

        See DisplayFormatter.describe() for full documentation.
        """
        formatter = DisplayFormatter(self)
        return formatter.describe(
            return_string=return_string,
            show_empty=show_empty,
            max_metadata_fields=max_metadata_fields,
        )

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

        # Add extra fields not handled by base handler
        if num_rows is not None:
            metadata["num_rows"] = num_rows
        if version is not None:
            metadata["version"] = version
        if code is not None:
            metadata["code"] = code

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
        # Validate inputs
        if not overwrite and name in self._items:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
            )

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

        self._items[name] = metadata

        # Update manifest
        self._save_items()

        return self

    def get_table(self, name: str) -> Any:  # Returns pandas.DataFrame
        """Get a table by name (works for both included and referenced).

        For included tables, reads from bundle directory.
        For referenced tables, reads from the specified external path.
        Tables are NOT cached - always read fresh from disk.

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

        # Get handler and delegate to it
        registry = get_registry()
        handler = registry.get(item_type)
        return handler.get(self, name)

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
        - For referenced tables: path, table_format, num_rows, version, description
        - For included tables: filename, num_rows, num_cols, columns, dtypes, description

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

        # Add extra fields not handled by base handler
        if hyperparameters is not None:
            metadata["hyperparameters"] = hyperparameters
        if code is not None:
            metadata["code"] = code

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

        Models are NOT cached - always read fresh from disk.

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
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("model")
        return handler.get(self, name)

    def get_model(self, name: str, **kwargs) -> Any:
        """Get a model by name with automatic type detection.

        Automatically detects whether the model is PyTorch or sklearn-style
        and uses the appropriate loader. For fine-grained control, use
        get_sklearn() or get_pytorch().

        Models are NOT cached - always read fresh from disk.

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
            self, name, model, description=description, inputs=inputs
        )

        # Add extra fields not handled by base handler
        if hyperparameters is not None:
            metadata["hyperparameters"] = hyperparameters
        if init_args is not None:
            metadata["init_args"] = init_args
        if code is not None:
            metadata["code"] = code
        if save_class:
            metadata["has_serialized_class"] = save_class

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

        # Delegate to handler
        from datafolio.base.registry import get_registry

        registry = get_registry()
        handler = registry.get("pytorch_model")
        return handler.get(self, name, model_class=model_class)

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

        # Add extra fields not handled by base handler
        if category is not None:
            metadata["category"] = category

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

        # Add extra fields not handled by base handler
        if code is not None:
            metadata["code"] = code

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

        # Add extra fields not handled by base handler
        if code is not None:
            metadata["code"] = code

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

        # Add extra fields not handled by base handler
        if code is not None:
            metadata["code"] = code

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

        # Use handler to add data
        metadata = handler.add(
            folio=self,
            name=name,
            data=data,
            description=description,
            inputs=kwargs.get("inputs"),
            **kwargs,
        )
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
                    src_cf = cloudfiles.CloudFiles(src_path)
                    content = src_cf.get(item["filename"])
                    dst_cf = cloudfiles.CloudFiles(dst_path)
                    dst_cf.put(item["filename"], content)

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
                    src_cf = cloudfiles.CloudFiles(src_path)
                    content = src_cf.get(item["filename"])
                    dst_cf = cloudfiles.CloudFiles(dst_path)
                    dst_cf.put(item["filename"], content)

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
                    src_cf = cloudfiles.CloudFiles(src_path)
                    content = src_cf.get(item["filename"])
                    dst_cf = cloudfiles.CloudFiles(dst_path)
                    dst_cf.put(item["filename"], content)

                new_folio._items[item_name] = dict(item)

        # Save the new manifest
        new_folio._save_items()

        return new_folio
