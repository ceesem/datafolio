"""Main DataFolio class for bundling analysis artifacts."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cloudfiles
from typing_extensions import Self

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
    is_cloud_path,
    make_bundle_name,
)

from .utils import resolve_path


class MetadataDict(dict):
    """Dictionary that auto-saves to file on any modification."""

    def __init__(self, parent: "DataFolio", *args, **kwargs):
        """Initialize MetadataDict with parent reference.

        Args:
            parent: Parent DataFolio instance for callbacks
            *args: Positional arguments for dict
            **kwargs: Keyword arguments for dict
        """
        # Initialize parent AFTER super().__init__() to avoid triggering saves during initialization
        super().__init__(*args, **kwargs)
        self._parent = parent

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item and trigger save."""
        super().__setitem__(key, value)
        if hasattr(self, "_parent"):  # Skip during initialization
            # Update timestamp (avoid infinite loop by not triggering for 'updated_at')
            if key != "updated_at":
                super().__setitem__(
                    "updated_at", datetime.now(timezone.utc).isoformat()
                )
            self._parent._save_metadata()

    def __delitem__(self, key: str) -> None:
        """Delete item and trigger save."""
        super().__delitem__(key)
        super().__setitem__("updated_at", datetime.now(timezone.utc).isoformat())
        self._parent._save_metadata()

    def update(self, *args, **kwargs) -> None:
        """Update dict and trigger save."""
        super().update(*args, **kwargs)
        super().__setitem__("updated_at", datetime.now(timezone.utc).isoformat())
        self._parent._save_metadata()

    def clear(self) -> None:
        """Clear dict and trigger save."""
        super().clear()
        super().__setitem__("updated_at", datetime.now(timezone.utc).isoformat())
        self._parent._save_metadata()

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default and trigger save if key was added."""
        had_key = key in self
        result = super().setdefault(key, default)
        if not had_key:
            super().__setitem__("updated_at", datetime.now(timezone.utc).isoformat())
            self._parent._save_metadata()
        return result


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
        # Unified items dictionary - all items stored here with item_type
        self._items: Dict[str, Union[TableReference, IncludedTable, IncludedItem]] = {}

        # For storing models/artifacts before writing
        self._models_data: Dict[str, Any] = {}  # Model storage
        self._artifacts_paths: Dict[str, Union[str, Path]] = {}  # Artifact file paths

        # Store random suffix setting for collision retry
        self._use_random_suffix = random_suffix

        # Check if path is an existing bundle (has metadata.json or items.json)
        path_str = str(path)
        metadata_path = self._join_paths(path_str, METADATA_FILE)
        items_path = self._join_paths(path_str, ITEMS_FILE)

        is_existing_bundle = self._exists(path_str) and (
            self._exists(metadata_path) or self._exists(items_path)
        )

        if is_existing_bundle:
            # Open existing bundle
            self._bundle_dir = path_str
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
                        self._join_paths(parent, bundle_name) if parent else bundle_name
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

            self._is_new = True

            # Initialize metadata with timestamps
            self._metadata_raw = metadata or {}
            now = datetime.now(timezone.utc).isoformat()
            if "created_at" not in self._metadata_raw:
                self._metadata_raw["created_at"] = now
            if "updated_at" not in self._metadata_raw:
                self._metadata_raw["updated_at"] = now
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

    def _initialize_bundle(self, max_retries: int = 10) -> None:
        """Initialize new bundle directory structure with collision retry.

        Args:
            max_retries: Maximum number of retries on name collision
        """
        # Check if directory already exists
        if self._exists(self._bundle_dir):
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
                            self._join_paths(parent, new_name) if parent else new_name
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
                    if not self._exists(self._bundle_dir):
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
        self._mkdir(self._bundle_dir)
        self._mkdir(self._join_paths(self._bundle_dir, TABLES_DIR))
        self._mkdir(self._join_paths(self._bundle_dir, MODELS_DIR))
        self._mkdir(self._join_paths(self._bundle_dir, ARTIFACTS_DIR))

        # Write initial manifests
        self._save_metadata()
        self._save_items()

    def _load_manifests(self) -> None:
        """Load all manifest files from existing bundle."""
        # Read metadata.json
        metadata_path = self._join_paths(self._bundle_dir, METADATA_FILE)
        if self._exists(metadata_path):
            self._metadata_raw = self._read_json(metadata_path)
        else:
            self._metadata_raw = {}

        # Read unified items.json
        items_path = self._join_paths(self._bundle_dir, ITEMS_FILE)
        if self._exists(items_path):
            items_list = self._read_json(items_path)
            self._items = {item["name"]: item for item in items_list}
        else:
            self._items = {}

    # ==================== Manifest Save Methods ====================

    def _save_metadata(self) -> None:
        """Save metadata.json."""
        path = self._join_paths(self._bundle_dir, METADATA_FILE)
        # Convert MetadataDict to regular dict for serialization
        data = (
            dict(self.metadata)
            if isinstance(self.metadata, MetadataDict)
            else self.metadata
        )
        self._write_json(path, data)

    def _save_items(self) -> None:
        """Save unified items.json manifest."""
        path = self._join_paths(self._bundle_dir, ITEMS_FILE)
        self._write_json(path, list(self._items.values()))

    # ==================== Public API Methods ====================

    def list_contents(self) -> Dict[str, list[str]]:
        """List all contents in the DataFolio.

        Returns:
            Dictionary with keys 'referenced_tables', 'included_tables', 'numpy_arrays',
            'json_data', 'models', 'pytorch_models', and 'artifacts', each containing a list of names

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.reference_table('data1', path='s3://bucket/data.parquet')
            >>> folio.add_numpy('embeddings', np.array([1, 2, 3]))
            >>> folio.list_contents()
            {'referenced_tables': ['data1'], 'included_tables': [], 'numpy_arrays': ['embeddings'],
             'json_data': [], 'models': [], 'pytorch_models': [], 'artifacts': []}
        """
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

    def __repr__(self) -> str:
        """Return string representation of DataFolio."""
        contents = self.list_contents()
        total_items = sum(len(v) for v in contents.values())
        return f"DataFolio(bundle_dir='{self._bundle_dir}', items={total_items})"

    def describe(
        self, return_string: bool = False, show_empty: bool = False
    ) -> Optional[str]:
        """Generate a human-readable description of all items in the bundle.

        Includes lineage information showing inputs and dependencies.

        Args:
            return_string: If True, return the description as a string instead of printing
            show_empty: If True, show empty sections (e.g., "Models (0): (none)")

        Returns:
            None if return_string=False (prints to stdout), otherwise returns the description string

        Examples:
            Print description (default):
            >>> folio = DataFolio('experiments/test')
            >>> folio.describe()
            DataFolio: experiments/test
            ===========================
            Tables (2):
              • raw_data (reference): Training data
              • results: Model results
                ↳ inputs: raw_data

            Get description as string:
            >>> folio = DataFolio('experiments/test')
            >>> text = folio.describe(return_string=True)

            Show empty sections:
            >>> folio.describe(show_empty=True)
            Tables (2):
              • raw_data: Training data
              • results: Model results
            Models (0):
              (none)
        """
        lines = []
        lines.append(f"DataFolio: {self._bundle_dir}")
        lines.append("=" * len(lines[0]))
        lines.append("")

        # Add timestamps if available
        if "created_at" in self.metadata:
            lines.append(f"Created: {self.metadata['created_at']}")
        if "updated_at" in self.metadata:
            lines.append(f"Updated: {self.metadata['updated_at']}")
        if "parent_bundle" in self.metadata:
            lines.append(f"Parent: {self.metadata['parent_bundle']}")
        if any(
            k in self.metadata for k in ["created_at", "updated_at", "parent_bundle"]
        ):
            lines.append("")

        contents = self.list_contents()

        # Combine referenced and included tables
        ref_tables = contents["referenced_tables"]
        inc_tables = contents["included_tables"]
        all_tables = ref_tables + inc_tables

        if all_tables or show_empty:
            lines.append(f"Tables ({len(all_tables)}):")
            if all_tables:
                # Show referenced tables first
                for name in ref_tables:
                    item = self._items[name]
                    desc = item.get("description", "(no description)")
                    lines.append(f"  • {name} (reference): {desc}")
                    # Show path for referenced tables
                    if "path" in item:
                        lines.append(f"    ↳ path: {item['path']}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")

                # Then included tables
                for name in inc_tables:
                    item = self._items[name]
                    desc = item.get("description", "(no description)")
                    lines.append(f"  • {name}: {desc}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
                    if "models" in item and item["models"]:
                        lines.append(f"    ↳ models: {', '.join(item['models'])}")
            else:
                lines.append("  (none)")
            lines.append("")

        # Numpy arrays
        numpy_arrays = contents["numpy_arrays"]
        if numpy_arrays or show_empty:
            lines.append(f"Numpy Arrays ({len(numpy_arrays)}):")
            if numpy_arrays:
                for name in numpy_arrays:
                    item = self._items[name]
                    desc = item.get("description", "(no description)")
                    shape = item.get("shape", "unknown")
                    dtype = item.get("dtype", "unknown")
                    lines.append(f"  • {name}: {desc}")
                    lines.append(f"    ↳ shape: {shape}, dtype: {dtype}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
            else:
                lines.append("  (none)")
            lines.append("")

        # JSON data
        json_data = contents["json_data"]
        if json_data or show_empty:
            lines.append(f"JSON Data ({len(json_data)}):")
            if json_data:
                for name in json_data:
                    item = self._items[name]
                    desc = item.get("description", "(no description)")
                    data_type = item.get("data_type", "unknown")
                    lines.append(f"  • {name}: {desc}")
                    lines.append(f"    ↳ type: {data_type}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
            else:
                lines.append("  (none)")
            lines.append("")

        # Models
        models = contents["models"]
        if models or show_empty:
            lines.append(f"Models ({len(models)}):")
            if models:
                for name in models:
                    item = self._items[name]
                    desc = item.get("description", "(no description)")
                    lines.append(f"  • {name}: {desc}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
                    if "hyperparameters" in item and item["hyperparameters"]:
                        # Show a few key hyperparameters
                        hps = item["hyperparameters"]
                        hp_str = ", ".join(f"{k}={v}" for k, v in list(hps.items())[:3])
                        if len(hps) > 3:
                            hp_str += f", ... ({len(hps) - 3} more)"
                        lines.append(f"    ↳ hyperparameters: {hp_str}")
            else:
                lines.append("  (none)")
            lines.append("")

        # PyTorch Models
        pytorch_models = contents["pytorch_models"]
        if pytorch_models or show_empty:
            lines.append(f"PyTorch Models ({len(pytorch_models)}):")
            if pytorch_models:
                for name in pytorch_models:
                    item = self._items[name]
                    desc = item.get("description", "(no description)")
                    lines.append(f"  • {name}: {desc}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
                    if "hyperparameters" in item and item["hyperparameters"]:
                        # Show a few key hyperparameters
                        hps = item["hyperparameters"]
                        hp_str = ", ".join(f"{k}={v}" for k, v in list(hps.items())[:3])
                        if len(hps) > 3:
                            hp_str += f", ... ({len(hps) - 3} more)"
                        lines.append(f"    ↳ hyperparameters: {hp_str}")
                    if "init_args" in item and item["init_args"]:
                        # Show init args
                        init_args = item["init_args"]
                        init_str = ", ".join(
                            f"{k}={v}" for k, v in list(init_args.items())[:3]
                        )
                        if len(init_args) > 3:
                            init_str += f", ... ({len(init_args) - 3} more)"
                        lines.append(f"    ↳ init_args: {init_str}")
            else:
                lines.append("  (none)")
            lines.append("")

        # Artifacts
        artifacts = contents["artifacts"]
        if artifacts or show_empty:
            lines.append(f"Artifacts ({len(artifacts)}):")
            if artifacts:
                for name in artifacts:
                    item = self._items[name]
                    desc = item.get("description", "(no description)")
                    category = item.get("category", "")
                    category_str = f" ({category})" if category else ""
                    lines.append(f"  • {name}{category_str}: {desc}")
            else:
                lines.append("  (none)")

        # Build final output
        output = "\n".join(lines)

        # Print or return based on parameter
        if return_string:
            return output
        else:
            print(output)
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
        from datafolio.utils import resolve_path, validate_table_format

        # Validate inputs
        if name in self._items:
            raise ValueError(f"Item '{name}' already exists in this DataFolio")

        validate_table_format(table_format)

        # Resolve path (cloud paths remain as-is)
        resolved_path = resolve_path(path, make_absolute=True)

        # Create reference metadata
        ref: TableReference = {
            "name": name,
            "item_type": "referenced_table",
            "path": resolved_path,
            "table_format": table_format,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if num_rows is not None:
            ref["num_rows"] = num_rows
        if version is not None:
            ref["version"] = version
        if description is not None:
            ref["description"] = description
        if inputs is not None:
            ref["inputs"] = inputs
        if code is not None:
            ref["code"] = code

        self._items[name] = ref

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
        import pandas as pd

        # Validate inputs
        if not overwrite and name in self._items:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
            )

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data).__name__}")

        # Create metadata
        filename = f"{name}.parquet"
        included: IncludedTable = {
            "name": name,
            "item_type": "included_table",
            "filename": filename,
            "num_rows": len(data),
            "num_cols": len(data.columns),
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if description is not None:
            included["description"] = description
        if inputs is not None:
            included["inputs"] = inputs
        if models is not None:
            included["models"] = models
        if code is not None:
            included["code"] = code

        self._items[name] = included

        # Write parquet file immediately
        table_path = self._join_paths(self._bundle_dir, TABLES_DIR, filename)
        self._write_parquet(table_path, data)

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
        # Check if item exists
        if name not in self._items:
            raise KeyError(f"Table '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        # Handle included table
        if item_type == "included_table":
            table_path = self._join_paths(
                self._bundle_dir, TABLES_DIR, item["filename"]
            )
            return self._read_parquet(table_path)

        # Handle referenced table
        elif item_type == "referenced_table":
            from datafolio.readers import read_table

            table_format = item["table_format"]
            path = item["path"]

            # Read the table (not cached)
            return read_table(path, table_format=table_format)

        else:
            raise ValueError(f"Item '{name}' is not a table (type: {item_type})")

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

        # Create metadata
        filename = f"{name}.joblib"
        item: IncludedItem = {
            "name": name,
            "filename": filename,
            "item_type": "model",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if description is not None:
            item["description"] = description
        if inputs is not None:
            item["inputs"] = inputs
        if hyperparameters is not None:
            item["hyperparameters"] = hyperparameters
        if code is not None:
            item["code"] = code

        self._items[name] = item

        # Write model immediately
        model_path = self._join_paths(self._bundle_dir, MODELS_DIR, filename)
        self._write_joblib(model_path, model)

        # Update manifest
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

        # Read model (not cached)
        model_path = self._join_paths(self._bundle_dir, MODELS_DIR, item["filename"])
        return self._read_joblib(model_path)

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

        # Create metadata
        filename = f"{name}.pt"
        item: IncludedItem = {
            "name": name,
            "filename": filename,
            "item_type": "pytorch_model",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if description is not None:
            item["description"] = description
        if inputs is not None:
            item["inputs"] = inputs
        if hyperparameters is not None:
            item["hyperparameters"] = hyperparameters
        if init_args is not None:
            item["init_args"] = init_args
        if code is not None:
            item["code"] = code

        # Store metadata about whether class was serialized
        item["has_serialized_class"] = save_class

        self._items[name] = item

        # Write model immediately
        model_path = self._join_paths(self._bundle_dir, MODELS_DIR, filename)
        self._write_pytorch(
            model_path, model, init_args=init_args, save_class=save_class
        )

        # Update manifest
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

        # Read model bundle (not cached)
        model_path = self._join_paths(self._bundle_dir, MODELS_DIR, item["filename"])
        bundle = self._read_pytorch(model_path)

        # Extract state_dict
        state_dict = bundle["state_dict"]

        # If not reconstructing, just return state_dict
        if not reconstruct:
            return state_dict

        # Attempt reconstruction
        metadata = bundle.get("metadata", {})
        init_args = item.get("init_args", {})

        # Option 1: User provided model_class
        if model_class is not None:
            try:
                model = model_class(**init_args)
                model.load_state_dict(state_dict)
                return model
            except Exception as e:
                raise RuntimeError(
                    f"Failed to reconstruct model with provided model_class: {e}"
                )

        # Option 2: Try to import from module.class metadata
        model_module = metadata.get("model_module")
        model_class_name = metadata.get("model_class")

        if model_module and model_class_name:
            try:
                import importlib

                module = importlib.import_module(model_module)
                model_cls = getattr(module, model_class_name)
                model = model_cls(**init_args)
                model.load_state_dict(state_dict)
                return model
            except (ImportError, AttributeError) as e:
                # Fall through to try dill
                pass

        # Option 3: Try to use dill-serialized class
        if "serialized_class" in bundle and bundle["serialized_class"] is not None:
            try:
                import dill

                model_cls = dill.loads(bundle["serialized_class"])
                model = model_cls(**init_args)
                model.load_state_dict(state_dict)
                return model
            except Exception as e:
                raise RuntimeError(
                    f"Failed to reconstruct model from serialized class: {e}"
                )

        # No reconstruction method worked
        raise RuntimeError(
            f"Cannot auto-reconstruct model '{name}'. Try one of:\n"
            f"1. Provide model_class: folio.get_pytorch('{name}', model_class=YourModelClass)\n"
            f"2. Get state_dict only: folio.get_pytorch('{name}', reconstruct=False)\n"
            f"3. Ensure model class '{model_class_name}' from '{model_module}' is importable\n"
            f"4. Re-save model with save_class=True to enable automatic reconstruction"
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
        from pathlib import Path

        # Validate inputs
        if not overwrite and name in self._items:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
            )

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Artifact file not found: {path}")

        # Create metadata
        filename = f"{name}{path_obj.suffix}"
        item: IncludedItem = {
            "name": name,
            "filename": filename,
            "item_type": "artifact",
        }

        if category is not None:
            item["category"] = category
        if description is not None:
            item["description"] = description

        self._items[name] = item

        # Copy artifact immediately
        artifact_path = self._join_paths(self._bundle_dir, ARTIFACTS_DIR, filename)
        self._copy_file(path_obj, artifact_path)

        # Update manifest
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
        if name not in self._items:
            raise KeyError(f"Artifact '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "artifact":
            raise ValueError(
                f"Item '{name}' is not an artifact (type: {item.get('item_type')})"
            )

        # Return path to artifact in bundle
        return self._join_paths(self._bundle_dir, ARTIFACTS_DIR, item["filename"])

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
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "NumPy is required to save numpy arrays. "
                "Install with: pip install numpy"
            )

        # Validate inputs
        if not overwrite and name in self._items:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. Use overwrite=True to replace it."
            )

        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(array).__name__}")

        # Create metadata
        filename = f"{name}.npy"
        item: IncludedItem = {
            "name": name,
            "filename": filename,
            "item_type": "numpy_array",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if description is not None:
            item["description"] = description
        if inputs is not None:
            item["inputs"] = inputs
        if code is not None:
            item["code"] = code

        self._items[name] = item

        # Write array immediately
        array_path = self._join_paths(self._bundle_dir, ARTIFACTS_DIR, filename)
        self._write_numpy(array_path, array)

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
        if name not in self._items:
            raise KeyError(f"Array '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "numpy_array":
            raise ValueError(
                f"Item '{name}' is not a numpy array (type: {item.get('item_type')})"
            )

        # Read array (not cached)
        array_path = self._join_paths(self._bundle_dir, ARTIFACTS_DIR, item["filename"])
        return self._read_numpy(array_path)

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

        # Check if data is JSON-serializable
        import orjson

        try:
            # Test serialization
            orjson.dumps(data)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Data is not JSON-serializable: {e}")

        # Create metadata
        filename = f"{name}.json"
        item: IncludedItem = {
            "name": name,
            "filename": filename,
            "item_type": "json_data",
            "data_type": type(data).__name__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if description is not None:
            item["description"] = description
        if inputs is not None:
            item["inputs"] = inputs
        if code is not None:
            item["code"] = code

        self._items[name] = item

        # Write JSON immediately
        json_path = self._join_paths(self._bundle_dir, ARTIFACTS_DIR, filename)
        self._write_json(json_path, data)

        # Update manifest
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
        if name not in self._items:
            raise KeyError(f"JSON data '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "json_data":
            raise ValueError(
                f"Item '{name}' is not JSON data (type: {item.get('item_type')})"
            )

        # Read JSON (not cached)
        json_path = self._join_paths(self._bundle_dir, ARTIFACTS_DIR, item["filename"])
        return self._read_json(json_path)

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

        # Type detection and dispatch
        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                return self.add_table(name, data, description=description, **kwargs)
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(data, np.ndarray):
                return self.add_numpy(name, data, description=description, **kwargs)
        except ImportError:
            pass

        # Check if JSON-serializable (dict, list, scalar)
        if isinstance(data, (dict, list, int, float, str, bool, type(None))):
            return self.add_json(name, data, description=description, **kwargs)

        # Unsupported type
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            f"Supported types: pandas.DataFrame, numpy.ndarray, dict, list, or JSON scalars. "
            f"For other file types, use add_artifact()."
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
        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        # Dispatch based on item type
        if item_type in ("referenced_table", "included_table"):
            return self.get_table(name)
        elif item_type == "numpy_array":
            return self.get_numpy(name)
        elif item_type == "json_data":
            return self.get_json(name)
        else:
            raise ValueError(
                f"Item '{name}' is not a data item (type: {item_type}). "
                f"Use get_model() for models or get_artifact_path() for artifacts."
            )

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
                src_path = self._join_paths(
                    self._bundle_dir, TABLES_DIR, item["filename"]
                )
                dst_path = self._join_paths(
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
                src_path = self._join_paths(
                    self._bundle_dir, MODELS_DIR, item["filename"]
                )
                dst_path = self._join_paths(
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
                src_path = self._join_paths(
                    self._bundle_dir, ARTIFACTS_DIR, item["filename"]
                )
                dst_path = self._join_paths(
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
