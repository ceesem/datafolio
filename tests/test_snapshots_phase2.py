"""Tests for Snapshot Phase 2: Copy-on-Write Version Management.

These tests verify copy-on-write when overwriting items referenced by
snapshots. With versioned payload filenames (``<name>--r<rev><ext>``) each
version already lives in its own file, so overwriting never touches the bytes a
snapshot depends on — there is nothing to rename. The outgoing version is simply
marked non-current and preserved; the replacement becomes the current version.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from datafolio import DataFolio


class TestCopyOnWrite:
    """Tests for copy-on-write when overwriting items in snapshots."""

    def test_overwrite_without_snapshots(self, tmp_path):
        """Overwriting works normally when the item is not in snapshots."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add("data", df1)
        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 0

        # Overwrite - should work since not in snapshots
        folio.add("data", df2, overwrite=True)

        # Should still have only 1 current item, no preserved versions
        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 0

        loaded_df = folio.get("data")
        pd.testing.assert_frame_equal(loaded_df, df2)

    def test_copy_on_write_when_in_snapshots(self, tmp_path):
        """Copy-on-write preserves the old version when overwriting."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add("data", df1)

        # Manually mark as in snapshot (Phase 3 does this via create_snapshot()).
        folio.create_snapshot("v1.0")

        # Overwrite - should trigger copy-on-write
        folio.add("data", df2, overwrite=True)

        # 1 current item + 1 preserved snapshot version
        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 1

        # Current item should be the new data
        assert folio._items["data"]["is_current"] is True
        assert folio._items["data"]["in_snapshots"] == []

        # Snapshot version should be the old data, non-current, in its own file.
        snapshot_item = folio._snapshot_versions[0]
        assert snapshot_item["name"] == "data"
        assert snapshot_item["is_current"] is False
        assert snapshot_item["in_snapshots"] == ["v1.0"]
        # No rename: the preserved version keeps its own distinct payload file.
        assert snapshot_item["filename"] != folio._items["data"]["filename"]
        assert snapshot_item.get("version_id")

        loaded_df = folio.get("data")
        pd.testing.assert_frame_equal(loaded_df, df2)

    def test_preserved_version_file_is_not_renamed(self, tmp_path):
        """The old version's payload file is preserved in place, not renamed."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add("data", df1)
        original_filename = folio._items["data"]["filename"]
        tables_dir = Path(folio._bundle_dir) / "tables"
        assert (tables_dir / original_filename).exists()

        folio.create_snapshot("v1.0")
        folio.add("data", df2, overwrite=True)

        # The old file still exists under its ORIGINAL name (never renamed).
        preserved = folio._snapshot_versions[0]
        assert preserved["filename"] == original_filename
        assert (tables_dir / original_filename).exists()

        # A new, distinct current file holds the new data.
        current_filename = folio._items["data"]["filename"]
        assert current_filename != original_filename
        assert (tables_dir / current_filename).exists()

        pd.testing.assert_frame_equal(
            pd.read_parquet(tables_dir / original_filename), df1
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(tables_dir / current_filename), df2
        )

    def test_items_json_contains_both_versions(self, tmp_path):
        """items.json persists both the current and the preserved version."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add("data", df1)
        folio.create_snapshot("v1.0")
        folio.add("data", df2, overwrite=True)

        items_path = Path(folio._bundle_dir) / "items.json"
        with open(items_path) as f:
            items_data = json.load(f)

        items = items_data["items"]
        assert len(items) == 2

        current_items = [item for item in items if item.get("is_current")]
        assert len(current_items) == 1
        # Current payload is versioned and distinct from the preserved one.
        assert current_items[0]["filename"] == folio._items["data"]["filename"]

        snapshot_items = [item for item in items if not item.get("is_current")]
        assert len(snapshot_items) == 1
        assert "v1.0" in snapshot_items[0]["in_snapshots"]
        assert snapshot_items[0]["filename"] != current_items[0]["filename"]

    def test_reload_after_copy_on_write(self, tmp_path):
        """The bundle reloads correctly after copy-on-write."""
        folio1 = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio1.add("data", df1)
        folio1.create_snapshot("v1.0")
        folio1.add("data", df2, overwrite=True)
        preserved_filename = folio1._snapshot_versions[0]["filename"]

        folio2 = DataFolio(folio1._bundle_dir)

        assert len(folio2._items) == 1
        assert len(folio2._snapshot_versions) == 1

        loaded_df = folio2.get("data")
        pd.testing.assert_frame_equal(loaded_df, df2)

        assert folio2._snapshot_versions[0]["in_snapshots"] == ["v1.0"]
        assert folio2._snapshot_versions[0]["filename"] == preserved_filename

    def test_multiple_overwrites(self, tmp_path):
        """Multiple overwrites create multiple preserved versions."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        df3 = pd.DataFrame({"a": [7, 8, 9]})

        folio.add("data", df1)
        folio.create_snapshot("v1.0")

        folio.add("data", df2, overwrite=True)
        folio.create_snapshot("v2.0")

        folio.add("data", df3, overwrite=True)

        # 1 current + 2 preserved versions
        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 2

        loaded_df = folio.get("data")
        pd.testing.assert_frame_equal(loaded_df, df3)

        # Each preserved version has its own distinct payload file and version id.
        filenames = [item["filename"] for item in folio._snapshot_versions]
        version_ids = [item.get("version_id") for item in folio._snapshot_versions]
        assert len(set(filenames)) == 2
        assert all(version_ids)
        assert len(set(version_ids)) == 2
        # v1.0 and v2.0 are each represented among the preserved versions.
        snaps = [set(item["in_snapshots"]) for item in folio._snapshot_versions]
        assert {"v1.0"} in snaps
        assert {"v2.0"} in snaps


class TestCopyOnWriteErrors:
    """Tests for error handling in copy-on-write logic."""

    def test_cannot_overwrite_without_flag(self, tmp_path):
        """Overwriting without the flag raises (backward compat)."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add("data", df1)

        with pytest.raises(ValueError, match="already exists"):
            folio.add("data", df2)

    def test_can_overwrite_with_flag_when_not_in_snapshots(self, tmp_path):
        """overwrite=True works when the item is not in snapshots."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add("data", df1)
        folio.add("data", df2, overwrite=True)

        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 0

    def test_automatic_copy_on_write_when_in_snapshots(self, tmp_path):
        """Copy-on-write happens automatically for items in snapshots."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add("data", df1)
        folio.create_snapshot("v1.0")

        folio.add("data", df2, overwrite=True)

        assert len(folio._snapshot_versions) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
