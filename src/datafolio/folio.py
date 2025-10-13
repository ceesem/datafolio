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

    def pop(self, *args, **kwargs) -> Any:
        """Pop item and trigger save."""
        result = super().pop(*args, **kwargs)
        super().__setitem__("updated_at", datetime.now(timezone.utc).isoformat())
        self._parent._save_metadata()
        return result

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
        use_random_suffix: bool = False,
    ):
        """Initialize a new or open an existing DataFolio.

        If the directory doesn't exist, creates a new bundle.
        If it exists, opens the existing bundle and reads manifests.

        Args:
            path: Full path to bundle directory (local or cloud)
            metadata: Optional dictionary of analysis metadata (for new bundles)
            use_random_suffix: If True, append random suffix to bundle name (default: False)

        Examples:
            Create new bundle with exact name:
            >>> folio = DataFolio('experiments/protein-analysis')
            # Creates: experiments/protein-analysis/

            Create new bundle with random suffix:
            >>> folio = DataFolio(
            ...     'experiments/protein-analysis',
            ...     use_random_suffix=True
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
        self._use_random_suffix = use_random_suffix

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
            if use_random_suffix:
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
            Dictionary with keys 'referenced_tables', 'included_tables',
            'models', and 'artifacts', each containing a list of names

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.reference_table('data1', path='s3://bucket/data.parquet')
            >>> folio.list_contents()
            {'referenced_tables': ['data1'], 'included_tables': [], 'models': [], 'artifacts': []}
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
        models = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "model"
        ]
        artifacts = [
            name
            for name, item in self._items.items()
            if item.get("item_type") == "artifact"
        ]

        return {
            "referenced_tables": referenced_tables,
            "included_tables": included_tables,
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

    def __repr__(self) -> str:
        """Return string representation of DataFolio."""
        contents = self.list_contents()
        total_items = sum(len(v) for v in contents.values())
        return f"DataFolio(bundle_dir='{self._bundle_dir}', items={total_items})"

    def describe(self) -> str:
        """Generate a human-readable description of all items in the bundle.

        Includes lineage information showing inputs and dependencies.

        Returns:
            Formatted string showing name, type, description, and lineage for all items

        Examples:
            >>> folio = DataFolio('experiments', prefix='test')
            >>> folio.reference_table('raw_data', path='s3://bucket/data.parquet', description='Training data')
            >>> folio.add_table('results', df, description='Model results', inputs=['raw_data'], models=['classifier'])
            >>> folio.add_model('classifier', model, description='Random forest', inputs=['raw_data'])
            >>> print(folio.describe())
            DataFolio: experiments/test-blue-happy-falcon
            ================================================
            Created: 2025-10-12T14:30:00Z
            Updated: 2025-10-12T16:45:00Z

            Referenced Tables (1):
              • raw_data [referenced_table]: Training data

            Included Tables (1):
              • results [included_table]: Model results
                ↳ inputs: raw_data
                ↳ models: classifier

            Models (1):
              • classifier [model]: Random forest
                ↳ inputs: raw_data

            Artifacts (0):
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

        # Referenced tables
        ref_tables = contents["referenced_tables"]
        lines.append(f"Referenced Tables ({len(ref_tables)}):")
        if ref_tables:
            for name in ref_tables:
                item = self._items[name]
                desc = item.get("description", "(no description)")
                lines.append(f"  • {name} [{item['item_type']}]: {desc}")
                # Show lineage if present
                if "inputs" in item and item["inputs"]:
                    lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
        else:
            lines.append("  (none)")
        lines.append("")

        # Included tables
        inc_tables = contents["included_tables"]
        lines.append(f"Included Tables ({len(inc_tables)}):")
        if inc_tables:
            for name in inc_tables:
                item = self._items[name]
                desc = item.get("description", "(no description)")
                lines.append(f"  • {name} [{item['item_type']}]: {desc}")
                # Show lineage if present
                if "inputs" in item and item["inputs"]:
                    lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
                if "models" in item and item["models"]:
                    lines.append(f"    ↳ models: {', '.join(item['models'])}")
        else:
            lines.append("  (none)")
        lines.append("")

        # Models
        models = contents["models"]
        lines.append(f"Models ({len(models)}):")
        if models:
            for name in models:
                item = self._items[name]
                desc = item.get("description", "(no description)")
                lines.append(f"  • {name} [{item['item_type']}]: {desc}")
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

        # Artifacts
        artifacts = contents["artifacts"]
        lines.append(f"Artifacts ({len(artifacts)}):")
        if artifacts:
            for name in artifacts:
                item = self._items[name]
                desc = item.get("description", "(no description)")
                category = item.get("category", "")
                category_str = f" ({category})" if category else ""
                lines.append(f"  • {name} [{item['item_type']}]{category_str}: {desc}")
        else:
            lines.append("  (none)")

        return "\n".join(lines)

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

    def add_model(
        self,
        name: str,
        model: Any,
        description: Optional[str] = None,
        overwrite: bool = False,
        inputs: Optional[list[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None,
    ) -> Self:
        """Add a trained model to the bundle.

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
            >>> folio.add_model('classifier', model,
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

    def get_model(self, name: str) -> Any:
        """Get a model by name.

        Models are NOT cached - always read fresh from disk.

        Args:
            name: Name of the model

        Returns:
            The model object

        Raises:
            KeyError: If model name doesn't exist
            ValueError: If named item is not a model

        Examples:
            >>> folio = DataFolio('experiments/test-blue-happy-falcon')
            >>> model = folio.get_model('classifier')
        """
        if name not in self._items:
            raise KeyError(f"Model '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "model":
            raise ValueError(
                f"Item '{name}' is not a model (type: {item.get('item_type')})"
            )

        # Read model (not cached)
        model_path = self._join_paths(self._bundle_dir, MODELS_DIR, item["filename"])
        return self._read_joblib(model_path)

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
        use_random_suffix: bool = False,
    ) -> "DataFolio":
        """Create a copy of this bundle with a new name.

        Useful for creating derived experiments or checkpoints.

        Args:
            new_path: Full path for new bundle directory
            metadata_updates: Metadata fields to update/add in the copy
            include_items: If specified, only copy these items (by name)
            exclude_items: Items to exclude from copy (by name)
            use_random_suffix: If True, append random suffix to new bundle name (default: False)

        Returns:
            New DataFolio instance

        Raises:
            ValueError: If include_items and exclude_items are both specified

        Examples:
            >>> # Simple copy
            >>> folio2 = folio.copy('experiments/exp-v2')

            >>> # Copy with random suffix
            >>> folio2 = folio.copy('experiments/exp-v2', use_random_suffix=True)

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
            path=new_path, metadata=new_metadata, use_random_suffix=use_random_suffix
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
