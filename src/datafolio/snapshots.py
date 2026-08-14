"""Snapshot machinery for DataFolio.

``SnapshotView``/``SnapshotAccessor`` provide read access to pinned snapshot
state; ``SnapshotMixin`` carries every snapshot-related method of
:class:`~datafolio.folio.DataFolio`. All of it operates on the folio's
manifest state — versioned payload filenames mean snapshot operations are
manifest surgery, never byte copies.
"""

import contextlib  # noqa: F401
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from typing_extensions import Self

from datafolio.base.registry import get_registry
from datafolio.metadata import MetadataDict
from datafolio.utils import (
    SNAPSHOTS_FILE,
    _polars_only_error,
    validate_snapshot_name,
)

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


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
        """Get metadata as it existed at snapshot time (defensive copy)."""
        import copy

        return copy.deepcopy(self._snapshot_meta.get("metadata_snapshot", {}))

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
        """Get snapshot tags (defensive copy)."""
        return list(self._snapshot_meta.get("tags", []))

    @property
    def item_versions(self) -> Dict[str, str]:
        """Get pinned item version tokens in this snapshot (defensive copy)."""
        return dict(self._snapshot_meta.get("item_versions", {}))

    def get(self, name: str, frame: str = "pandas") -> Any:
        """Get a table from this snapshot (alias of :meth:`get_table`).

        Mirrors :meth:`DataFolio.get` for the snapshot view. Only tables are
        stored per-version today, so this delegates to :meth:`get_table`.

        Args:
            name: Table name
            frame: ``'pandas'`` (default) or ``'polars'``

        Returns:
            Table data as it existed in this snapshot.
        """
        return self.get_table(name, frame=frame)

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
        """Resolve the EXACT item version this snapshot pinned.

        Resolution goes through the recorded ``item_versions`` token (a
        ``version_id``, or a checksum in legacy snapshots) — never through
        ``in_snapshots`` membership alone, which could hand back a newer
        current version wearing a stale marker.

        Args:
            name: Item name

        Returns:
            Item metadata dict for the pinned version

        Raises:
            KeyError: If the name isn't in the snapshot, or the pinned
                version is no longer present in the manifest (fail closed
                rather than serving different data under the snapshot name).
        """
        token = self._snapshot_meta.get("item_versions", {}).get(name)
        if token is None:
            raise KeyError(f"Item '{name}' not found in snapshot '{self._name}'")

        item = self._folio._find_item_by_checksum(name, token)
        if item is None:
            raise KeyError(
                f"Snapshot '{self._name}' pins version '{token}' of item "
                f"'{name}', but that version is no longer in the manifest "
                f"(was it removed by cleanup_orphaned_versions?)."
            )
        return item


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
        self._folio._refresh_if_needed()
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
        self._folio._refresh_if_needed()
        return name in self._folio._snapshots

    def __iter__(self):
        """Iterate over snapshot names."""
        self._folio._refresh_if_needed()
        return iter(list(self._folio._snapshots.keys()))

    def __len__(self) -> int:
        """Get number of snapshots."""
        self._folio._refresh_if_needed()
        return len(self._folio._snapshots)

    def keys(self):
        """Get snapshot names."""
        self._folio._refresh_if_needed()
        return self._folio._snapshots.keys()

    def values(self):
        """Get SnapshotView objects for all snapshots."""
        self._folio._refresh_if_needed()
        return [SnapshotView(self._folio, name) for name in self._folio._snapshots]

    def items(self):
        """Get (name, SnapshotView) pairs."""
        self._folio._refresh_if_needed()
        return [
            (name, SnapshotView(self._folio, name)) for name in self._folio._snapshots
        ]


class SnapshotMixin:
    """Snapshot API of :class:`~datafolio.folio.DataFolio` (mixin).

    Split out of ``folio.py`` purely for maintainability — every method
    operates on the folio instance (``self``) exactly as before.
    """

    def _save_snapshots(self) -> None:
        """Publish the manifest (snapshots are embedded in it since v2).

        Kept as a thin alias so call sites and tests that mutate
        ``_snapshots`` directly still commit through the single authoritative
        manifest.
        """
        self._save_items()

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
            >>> folio.add('results', df)
            >>> folio.create_snapshot('v1.0-baseline', description='Initial results')
            >>>
            >>> # Later, overwriting will preserve the snapshot
            >>> folio.add('results', new_df, overwrite=True)  # Creates v2
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

        # Sync with other writers BEFORE the guard (refresh is suppressed
        # inside it): an up-to-date instance snapshots the current committed
        # state; a stale one (auto-refresh disabled) fails in the guard.
        self._refresh_if_needed()

        # Slow external context capture happens OUTSIDE the write lock (git
        # subprocesses, filesystem probes). It reads no folio state.
        context: Dict[str, Any] = {}
        if capture_git:
            git_info = self._capture_git_info()
            if git_info:
                context["git"] = git_info
        if capture_environment:
            context["environment"] = self._capture_environment_info()
        if capture_execution:
            context["execution"] = self._capture_execution_info()

        # EVERYTHING that reads or mutates folio state happens inside the
        # guard: the duplicate-name check, the item-version capture (with its
        # legacy version_id backfill), and the metadata copy. Capturing any
        # of it before the guard would let auto-refresh replace the state
        # mid-capture — a stale notebook could launder its staleness through
        # a metadata read and commit a snapshot pinning versions that no
        # longer exist. Inside the guard, refresh is suppressed and the
        # stale-writer check has already run, so the capture is one
        # consistent revision.
        with self._mutation_guard():
            if name in self._snapshots:
                raise ValueError(f"Snapshot '{name}' already exists")

            # Capture current item versions by their stable ``version_id``.
            # Legacy items lacking one get it assigned now (from their
            # checksum where available); the guard's spine discards the
            # backfill on a failed publication.
            item_versions: Dict[str, str] = {}
            for item_name, item_meta in self._items.items():
                version_id = item_meta.get("version_id")
                if not version_id:
                    version_id = item_meta.get("checksum") or self._next_version_id(
                        item_name
                    )
                    item_meta["version_id"] = version_id
                item_versions[item_name] = version_id

            # Capture current metadata state (DEEP copy: later nested edits
            # of the live metadata must not rewrite the snapshot's record)
            import copy as _copy

            md = getattr(self, "_metadata_dict", None)
            metadata_snapshot = _copy.deepcopy(dict(md)) if md is not None else {}

            snapshot_meta: Dict[str, Any] = {
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "item_versions": item_versions,
                "metadata_snapshot": metadata_snapshot,
                # Defensive copy: the caller's list must not stay live-wired
                # into the registry
                "tags": list(tags) if tags else [],
            }
            if description:
                snapshot_meta["description"] = description
            snapshot_meta.update(context)

            # The registry entry and any backfilled version_ids commit in ONE
            # atomic manifest write (membership derived from the registry).
            self._snapshots[name] = snapshot_meta
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
        self._refresh_if_needed()
        result = []
        for name, meta in self._snapshots.items():
            snapshot_info = {
                "name": name,
                "timestamp": meta.get("timestamp", ""),
                "description": meta.get("description"),
                "tags": list(meta.get("tags", [])),
                "num_items": len(meta.get("item_versions", {})),
            }
            result.append(snapshot_info)

        # Sort by timestamp (newest first)
        result.sort(key=lambda x: x["timestamp"], reverse=True)
        return result

    def delete_snapshot(self, name: str, cleanup_orphans: bool = False) -> Self:
        """Delete a snapshot.

        Removes the snapshot from the registry (membership is derived, so
        that single removal is the whole deletion).
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

        if self._batch_mode:
            raise RuntimeError(
                "delete_snapshot() cannot be called inside a batch() block: "
                "the batch's items are not committed yet. Exit the batch "
                "first."
            )

        if name not in self._snapshots:
            raise KeyError(f"Snapshot '{name}' not found")

        # Mutate and publish under the guard: a stale notebook fails BEFORE
        # any in-memory or on-disk change. Removing the registry entry IS the
        # deletion (membership is derived from the registry) — one atomic
        # manifest write.
        with self._mutation_guard():
            del self._snapshots[name]
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
        self._refresh_if_needed()
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
        self._refresh_if_needed()

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

        # Find orphaned snapshot versions: no LIVE marker (ghost markers —
        # naming snapshots absent from the registry — don't count).
        orphaned_versions = []
        for item in self._snapshot_versions:
            if not self._snapshot_pins(item):
                orphaned_versions.append(item)

        for item in orphaned_versions:
            # Report only files that will actually be deletable (a payload
            # shared with a live descriptor after a metadata-only
            # copy-on-write is kept).
            if item.get("filename") and not self._payload_is_shared(item):
                deleted_files.append(item["filename"])

        if not dry_run and self._batch_mode:
            raise RuntimeError(
                "cleanup_orphaned_versions() cannot run inside a batch() "
                "block: its payload deletion must follow a real manifest "
                "publish. Exit the batch first."
            )

        if dry_run or not orphaned_versions:
            return deleted_files

        # Publish the manifest without the orphans FIRST, then best-effort
        # delete their payload files (never the reverse: a failed manifest
        # write must not leave committed entries pointing at deleted bytes).
        # The whole operation is guarded like any other mutation.
        with self._mutation_guard():
            # Remove every orphaned descriptor (including payload-less
            # reference versions, which would otherwise linger forever).
            removed = list(orphaned_versions)
            for item in removed:
                self._snapshot_versions.remove(item)
            self._save_items()
            for item in removed:
                self._delete_payload_if_unshared(item)

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

        if self._batch_mode:
            raise RuntimeError(
                "restore_snapshot() cannot run inside a batch() block: its "
                "payload cleanup must follow a real manifest publish. Exit "
                "the batch first."
            )

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

            # Restore metadata (bulk internal update — the single manifest
            # publish at the end of the restore commits it; per-key saves here
            # would spray intermediate revisions)
            dict.clear(self.metadata)
            dict.update(self.metadata, snap_metadata)

            # Remove items not in the snapshot (added after it was taken). A
            # version pinned by some OTHER snapshot is preserved as a snapshot
            # version; an unpinned one is gone for good (this is the
            # documented destructive part). Payload deletion is DEFERRED
            # until after the manifest publish succeeds — the committed
            # manifest must never point at deleted bytes.
            newly_unreferenced: list = []
            for item_name in set(self._items) - set(snap_items):
                item = self._items[item_name]
                if self._snapshot_pins(item):
                    self._handle_copy_on_write(item_name)
                else:
                    newly_unreferenced.append(item)
                del self._items[item_name]

            # Repoint every snapshot item at its pinned descriptor.
            for item_name, pinned in pinned_by_name.items():
                current = self._items.get(item_name)
                if current is pinned:
                    continue  # already the working version

                # Displace the current version (if any): preserve it when a
                # snapshot pins it, otherwise drop it (payload deleted only
                # after the publish).
                if current is not None:
                    if self._snapshot_pins(current):
                        self._handle_copy_on_write(item_name)
                    else:
                        newly_unreferenced.append(current)
                    del self._items[item_name]

                # Promote the pinned descriptor back to current. It stays
                # listed in the snapshots that pin it.
                if pinned in self._snapshot_versions:
                    self._snapshot_versions.remove(pinned)
                pinned["is_current"] = True
                self._items[item_name] = pinned

            # Publish the manifest; only then best-effort delete the payloads
            # it no longer references.
            self._save_items()
            for item in newly_unreferenced:
                self._delete_payload_if_unshared(item)

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
            >>> paper.add('new', df)  # Error: snapshots are always read-only

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

        # Get snapshot metadata
        snapshot_meta = folio._snapshots[snapshot]
        snapshot_versions = snapshot_meta.get("item_versions", {})

        # FAIL CLOSED: resolve every pinned version BEFORE changing any
        # state. A pinned descriptor that can't be found must be an error —
        # keeping the current item under the snapshot name would silently
        # serve newer data as if it were the snapshot.
        pinned: Dict[str, Dict[str, Any]] = {}
        for item_name, version_token in snapshot_versions.items():
            item = folio._find_item_by_checksum(item_name, version_token)
            if item is None:
                raise KeyError(
                    f"Cannot load snapshot '{snapshot}': it pins version "
                    f"'{version_token}' of item '{item_name}', but that "
                    f"version is no longer in the manifest (was it removed "
                    f"by cleanup_orphaned_versions?)."
                )
            pinned[item_name] = item

        # Set snapshot mode
        folio._in_snapshot_mode = True
        folio._loaded_snapshot = snapshot

        # Replace current metadata with snapshot metadata
        # Use dict methods directly to bypass read-only checks during setup
        snapshot_metadata = snapshot_meta.get("metadata_snapshot", {})
        dict.clear(folio.metadata)
        dict.update(folio.metadata, snapshot_metadata)

        # Build the item mapping from scratch: exactly the pinned versions,
        # nothing else. A current item is never retained merely because its
        # logical name appears in the snapshot.
        folio._items = pinned

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
        from datafolio.utils import is_cloud_path

        # Cloud URIs must stay strings: Path() would mangle 'gs://bucket/x'
        # into a local-looking 'gs:/bucket/x'. Filesystem Path checks apply
        # to local targets only.
        target_str = str(target_path)
        if is_cloud_path(target_str):
            target: Union[str, Path] = target_str.rstrip("/")
            if self._storage.exists(target):
                raise ValueError(f"Target path already exists: {target}")
        else:
            target = Path(target_str)
            if target.exists():
                raise ValueError(f"Target path already exists: {target}")
        target_path = target

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
            # Cloud source URIs are preserved verbatim; only local sources
            # are resolved to absolute filesystem paths.
            source_bundle = (
                self._bundle_dir
                if is_cloud_path(self._bundle_dir)
                else str(Path(self._bundle_dir).resolve())
            )
            snapshot_metadata["_source_snapshot"] = {
                "name": snapshot,
                "source_bundle": source_bundle,
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
            new_item.pop("in_snapshots", None)  # legacy field, not persisted
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
        self._refresh_if_needed()
        if snapshot not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot}' not found")

        # Return a DEEP copy of the snapshot metadata (mutating the result
        # must never corrupt the live registry), surfacing any mutable
        # external references it contains (their bytes are not owned/frozen
        # — see mutable_references()).
        import copy

        info = copy.deepcopy(dict(self._snapshots[snapshot]))
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

    def _snapshot_pins(self, item: Dict[str, Any]) -> list[str]:
        """Snapshot names that pin this descriptor's exact version.

        Membership is DERIVED from the snapshot registry's ``item_versions``
        (matching the descriptor's ``version_id``, or its checksum for legacy
        snapshots) — never from persisted markers. Denormalized
        ``in_snapshots`` markers were dropped in manifest v2; they were a
        cache of exactly this computation and a standing source of
        consistency bugs (ghost markers, duplicates, partial removal).
        """
        name = item.get("name")
        tokens = {item.get("version_id"), item.get("checksum")} - {None}
        if not name or not tokens:
            return []
        return [
            snap_name
            for snap_name, meta in self._snapshots.items()
            if meta.get("item_versions", {}).get(name) in tokens
        ]

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
        return bool(self._snapshot_pins(item))

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
