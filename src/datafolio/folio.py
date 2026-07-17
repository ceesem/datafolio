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
from datafolio.context import ContextCaptureMixin
from datafolio.display import DisplayFormatter
from datafolio.metadata import MetadataDict
from datafolio.snapshots import (  # noqa: F401
    SnapshotAccessor,
    SnapshotMixin,
    SnapshotView,
)
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
    _polars_only_error,
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


class DataFolio(SnapshotMixin, ContextCaptureMixin):
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

df = folio.get('table_name')           # any item: tables, arrays, JSON, ...
model = folio.get_model('model_name')
path = folio.item_path('table_name')   # direct path to the payload file

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

    # ==================== Core item API ====================

    def add(
        self,
        name: str,
        obj: Any,
        *,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        overwrite: bool = False,
        code: Optional[str] = None,
        **type_opts: Any,
    ) -> Self:
        """Add an object to the folio (the single write entry point).

        The object's type selects the storage format automatically:

        - pandas / polars DataFrame, polars LazyFrame → Parquet table
        - numpy array → ``.npy``
        - dict / list / scalar (int, float, str, bool, None) → JSON
        - timezone-aware datetime (or Unix timestamp via add()'s datetime
          detection is not applied to bare numbers — those store as JSON)
          → timestamp
        - scikit-learn estimator → model (joblib)

        For anything else, use the explicit verbs: :meth:`add_model` for
        arbitrary picklable model-like objects, :meth:`add_file` to copy a
        file into the folio, or :meth:`reference_table` to link external
        table data without copying it.

        Strings are always stored as JSON data — a string that happens to be
        a path is never interpreted as a file (use :meth:`add_file`).

        JSON caveat: non-finite floats (``nan``/``inf``) have no JSON
        representation and are serialized as ``null``; they come back as
        ``None``. Store numeric arrays with NaNs as numpy arrays or tables
        instead.

        Args:
            name: Unique item name. May be namespaced with '/'
                (e.g. ``'examples/weights'``).
            obj: The object to store.
            description: Optional description. On overwrite, ``None``
                preserves the existing description and ``""`` clears it.
            inputs: Optional lineage — names of items this was derived from.
            overwrite: Must be True to replace an existing item. A prior
                version pinned by a snapshot is preserved via copy-on-write.
            code: Optional code snippet that created this item.
            **type_opts: Type-specific options. Tables: ``models`` (model
                lineage), ``preserve_index`` (keep a non-default pandas
                index). Models: ``custom`` (skops format), ``hyperparameters``.
                Files: ``category``. Unknown options raise ``TypeError``.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the name already exists (and ``overwrite=False``)
                or violates the name grammar.
            TypeError: If no handler supports the object's type, or an
                unknown type option was passed.

        Examples:
            >>> folio.add('results', df)                     # table
            >>> folio.add('embeddings', np.zeros((10, 8)))   # numpy
            >>> folio.add('config', {'lr': 0.01})            # JSON
            >>> folio.add('accuracy', 0.95)                  # JSON scalar
            >>> folio.add('trained', clf, inputs=['results'])  # sklearn model
            >>> folio.add('results', df2, overwrite=True)    # replace
        """
        self._check_read_only()
        validate_item_name(name)

        from datafolio.base.registry import detect_handler

        handler = detect_handler(obj)
        item_type = handler.item_type if handler is not None else None

        # Primitives don't auto-detect (keeps handler detection unambiguous)
        # but are valid JSON payloads.
        if item_type is None and isinstance(obj, (int, float, str, bool, type(None))):
            item_type = "json_data"

        if item_type is None:
            raise TypeError(
                f"Unsupported data type: {type(obj).__name__}. add() accepts "
                f"DataFrames/LazyFrames, numpy arrays, dicts/lists/scalars, "
                f"timezone-aware datetimes, and scikit-learn estimators. For "
                f"arbitrary picklable objects use add_model(); for files use "
                f"add_file(); for external tables use reference_table()."
            )

        return self._add_item(
            name,
            obj,
            item_type,
            description=description,
            inputs=inputs,
            overwrite=overwrite,
            code=code,
            **type_opts,
        )

    def _add_item(
        self,
        name: str,
        obj: Any,
        item_type: str,
        *,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        overwrite: bool = False,
        code: Optional[str] = None,
        **type_opts: Any,
    ) -> Self:
        """Shared guarded commit for every owned item type.

        All write invariants live here exactly once: read-only check, name
        validation, the uniform overwrite rule, snapshot copy-on-write,
        the mutation guard, description preservation, and the atomic
        manifest publish (via :meth:`_commit_owned_item`).
        """
        self._check_read_only()
        validate_item_name(name)

        if name in self._items and not overwrite:
            raise ValueError(
                f"Item '{name}' already exists in this DataFolio. "
                f"Use overwrite=True to replace it."
            )

        handler_kwargs: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}

        if item_type == "included_table":
            from datafolio.utils import get_file_extension

            extension = get_file_extension("parquet")
            models = type_opts.pop("models", None)
            if models is not None:
                extras["models"] = models
            if type_opts.pop("preserve_index", False):
                handler_kwargs["preserve_index"] = True
        elif item_type == "numpy_array":
            extension = ".npy"
        elif item_type == "json_data":
            extension = ".json"
        elif item_type == "timestamp":
            extension = ".json"
        elif item_type == "model":
            handler_kwargs["custom"] = bool(type_opts.pop("custom", False))
            hyperparameters = type_opts.pop("hyperparameters", None)
            if hyperparameters is not None:
                extras["hyperparameters"] = hyperparameters
            extension = ".skops" if handler_kwargs["custom"] else ".joblib"
        elif item_type == "artifact":
            extension = Path(str(obj)).suffix
            category = type_opts.pop("category", None)
            if category is not None:
                extras["category"] = category
            obj = str(obj)
        else:
            raise TypeError(f"No add() path for item type '{item_type}'.")

        if type_opts:
            raise TypeError(
                f"Unknown option(s) for item type '{item_type}': {sorted(type_opts)}"
            )

        def _build(filename: str, version_id: str) -> Dict[str, Any]:
            metadata = (
                get_registry()
                .get(item_type)
                .add(
                    self,
                    name,
                    obj,
                    description=description,
                    inputs=inputs,
                    _filename=filename,
                    **handler_kwargs,
                )
            )
            metadata.update(extras)
            if code is not None:
                metadata["code"] = code
            return metadata

        self._commit_owned_item(name, item_type, extension, description, _build)
        return self

    def get(self, name: str, **type_opts: Any) -> Any:
        """Get any item by name (the single read entry point).

        Returns the natural object for the item's type: tables come back as
        pandas DataFrames by default (``frame='polars'`` for polars), numpy
        arrays as arrays, JSON data as dicts/lists/scalars, timestamps as
        UTC-aware datetimes, models as loaded model objects, and files
        (artifacts) as the path to the payload — the only type whose "value"
        is a file.

        Eager table reads are subject to the folio's ``max_eager_bytes``
        guard; use :meth:`scan_table` for a lazy scan of large tables.

        Args:
            name: Item name.
            **type_opts: Type-specific options. Tables: ``frame``
                ('pandas'/'polars'), ``allow_full_load``, plus reader options
                (``columns``, ``filters``, ...). Timestamps: ``as_unix``.
                Models: ``trusted`` (skops-format models only). Unknown
                options raise ``TypeError``.

        Returns:
            The item's content (or path, for files).

        Raises:
            KeyError: If the item doesn't exist.
            TypeError: If an unknown type option was passed.
            ValueError: If a table exceeds the eager-load guard.

        Examples:
            >>> df = folio.get('results')
            >>> pl_df = folio.get('results', frame='polars')
            >>> cfg = folio.get('config')
            >>> path = folio.get('plot')  # artifact -> file path
        """
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")
        registry = get_registry()

        if item_type in ("included_table", "referenced_table"):
            frame = type_opts.pop("frame", "pandas")
            allow_full_load = type_opts.pop("allow_full_load", False)
            return self._get_table(
                name, frame=frame, allow_full_load=allow_full_load, **type_opts
            )
        if item_type == "timestamp":
            as_unix = type_opts.pop("as_unix", False)
            self._reject_unknown_opts(item_type, type_opts)
            dt = registry.get("timestamp").get(self, name, as_unix=False)
            return dt.timestamp() if as_unix else dt
        if item_type == "model":
            trusted = type_opts.pop("trusted", False)
            self._reject_unknown_opts(item_type, type_opts)
            return registry.get("model").get(self, name, trusted=trusted)
        if item_type == "artifact":
            self._reject_unknown_opts(item_type, type_opts)
            return self.item_path(name)
        if item_type in ("numpy_array", "json_data"):
            self._reject_unknown_opts(item_type, type_opts)
            return registry.get(item_type).get(self, name)

        raise ValueError(f"Item '{name}' has unknown type '{item_type}'.")

    @staticmethod
    def _reject_unknown_opts(item_type: str, type_opts: Dict[str, Any]) -> None:
        """Raise TypeError for leftover get()/add() options (never swallow)."""
        if type_opts:
            raise TypeError(
                f"Unknown option(s) for item type '{item_type}': {sorted(type_opts)}"
            )

    def _get_table(
        self,
        name: str,
        frame: str = "pandas",
        allow_full_load: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Eager table read (pandas or polars) with the shared guards."""
        item = self._items[name]
        item_type = item.get("item_type")

        if item_type not in ("included_table", "referenced_table"):
            raise ValueError(f"Item '{name}' is not a table (type: {item_type})")

        # Guard against accidentally materializing a huge table
        self._check_eager_size(name, item, allow_full_load)

        handler = get_registry().get(item_type)

        if frame == "polars":
            # Eager polars: scan lazily then collect.
            return handler.get_lazy(self, name, **kwargs).collect()
        elif frame == "pandas":
            # Sharded/partitioned (polars-only) tables can't be materialized
            # as pandas — raise a clear, actionable error.
            if item.get("polars_only"):
                raise _polars_only_error(name)
            return handler.get(self, name, **kwargs)
        else:
            raise ValueError(f"Unknown frame '{frame}'. Use 'pandas' or 'polars'.")

    def get_model(self, name: str, trusted: bool = False) -> Any:
        """Get a model by name (the one explicit typed getter).

        SECURITY: loading a joblib-format model executes pickle — never load
        models from folios you don't trust. Models saved with ``custom=True``
        use the skops format, which refuses unknown types unless you pass
        ``trusted=True`` after reviewing the error's type list.

        Args:
            name: Model name.
            trusted: For skops-format models, trust the non-standard types
                found in the file (required to load custom pipelines).

        Returns:
            The loaded model object.

        Raises:
            KeyError: If no item has this name.
            ValueError: If the named item is not a model.

        Examples:
            >>> clf = folio.get_model('classifier')
            >>> pipe = folio.get_model('custom_pipeline', trusted=True)
        """
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Model '{name}' not found in DataFolio")

        item = self._items[name]
        if item.get("item_type") != "model":
            raise ValueError(
                f"Item '{name}' is not a model (type: {item.get('item_type')})"
            )

        return get_registry().get("model").get(self, name, trusted=trusted)

    def add_model(
        self,
        name: str,
        model: Any,
        *,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        overwrite: bool = False,
        code: Optional[str] = None,
        custom: bool = False,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """Add a model-like object to the bundle (explicit verb).

        Unlike :meth:`add`, which only auto-detects scikit-learn estimators,
        this stores *any* picklable object as a model — storing arbitrary
        objects with pickle is an explicit act.

        Args:
            name: Unique name for this model.
            model: The model object to serialize.
            description: Optional description.
            inputs: Optional lineage (e.g. training data item names).
            overwrite: Must be True to replace an existing item.
            code: Optional code snippet that trained this model.
            custom: If True, use the skops format (portable pipelines with
                custom transformers; safer loading). Default joblib.
            hyperparameters: Optional dict of hyperparameters to record.

        Returns:
            Self for method chaining.

        Examples:
            >>> folio.add_model('classifier', clf,
            ...     hyperparameters={'n_estimators': 100})
            >>> folio.add_model('pipeline', custom_pipeline, custom=True)
        """
        return self._add_item(
            name,
            model,
            "model",
            description=description,
            inputs=inputs,
            overwrite=overwrite,
            code=code,
            custom=custom,
            hyperparameters=hyperparameters,
        )

    def add_file(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        *,
        category: Optional[str] = None,
        description: Optional[str] = None,
        overwrite: bool = False,
    ) -> Self:
        """Copy a file into the bundle (explicit verb for file payloads).

        The file's extension is preserved. Retrieve the stored file's path
        with ``get(name)`` or :meth:`item_path`.

        Args:
            path: Path to the file to copy in.
            name: Optional item name (default: the filename without extension).
            category: Optional grouping label ('plots', 'configs', ...).
            description: Optional description.
            overwrite: Must be True to replace an existing item.

        Returns:
            Self for method chaining.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the name already exists (and overwrite=False).

        Examples:
            >>> folio.add_file('plots/loss.png', category='plots')
            >>> folio.add_file('config.yaml', name='model_config')
            >>> open(folio.get('loss'), 'rb')  # stored file path
        """
        if name is None:
            name = Path(path).stem
        return self._add_item(
            name,
            str(path),
            "artifact",
            description=description,
            overwrite=overwrite,
            category=category,
        )

    def item_path(self, name: str) -> str:
        """Get the path to an item's payload file.

        For items stored in the bundle, returns the full path to the payload
        (a cloud URI for cloud folios — shareable with collaborators who
        don't use datafolio). For external references, returns the external
        path recorded at reference time.

        Args:
            name: Item name.

        Returns:
            Full path to the item's data file.

        Raises:
            KeyError: If the item doesn't exist.
            ValueError: If the item has no associated file.

        Examples:
            >>> folio.item_path('results')
            '/abs/path/bundle/tables/results--r2.parquet'
            >>> folio.item_path('raw')  # external reference
            's3://data-lake/raw.parquet'
        """
        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in DataFolio")

        item = self._items[name]
        item_type = item.get("item_type")

        if item_type == "referenced_table":
            return self._resolve_reference_path(item["path"])

        if "filename" not in item:
            raise ValueError(
                f"Item '{name}' (type: {item_type}) has no associated file path"
            )

        handler = get_registry().get(item_type)
        subdir = handler.get_storage_subdir()
        return self._storage.join_paths(self._bundle_dir, subdir, item["filename"])

    def item_info(self, name: str) -> Dict[str, Any]:
        """Get an item's manifest entry (a defensive copy).

        Contains the item's type, payload location, creation time, lineage,
        and type-specific fields (columns/dtypes/num_rows for tables, etc.).
        Mutating the returned dict does not change the manifest — use
        :meth:`update_item` for that.

        Args:
            name: Item name.

        Returns:
            A copy of the manifest entry.

        Raises:
            KeyError: If the item doesn't exist.

        Examples:
            >>> info = folio.item_info('results')
            >>> info['num_rows'], info['columns']
        """
        import copy

        self._refresh_if_needed()

        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in DataFolio")

        return copy.deepcopy(dict(self._items[name]))

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
