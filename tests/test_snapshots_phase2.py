"""Tests for Snapshot Phase 2: Copy-on-Write Version Management.

These tests verify that copy-on-write logic works correctly when overwriting
items that are referenced by snapshots.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from datafolio import DataFolio


class TestCopyOnWrite:
    """Tests for copy-on-write when overwriting items in snapshots."""

    def test_overwrite_without_snapshots(self, tmp_path):
        """Test that overwriting works normally when item not in snapshots."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        # Add item
        folio.add_table("data", df1)
        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 0

        # Overwrite - should work since not in snapshots
        folio.add_table("data", df2, overwrite=True)

        # Should still have only 1 current item
        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 0

        # Should have new data
        loaded_df = folio.get_table("data")
        pd.testing.assert_frame_equal(loaded_df, df2)

    def test_copy_on_write_when_in_snapshots(self, tmp_path):
        """Test that copy-on-write preserves old version when overwriting."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        # Add item
        folio.add_table("data", df1)

        # Manually mark as in snapshot (Phase 3 will do this via create_snapshot())
        folio._items["data"]["in_snapshots"] = ["v1.0"]

        # Overwrite - should trigger copy-on-write
        folio.add_table("data", df2, overwrite=True)

        # Should have 1 current item + 1 snapshot version
        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 1

        # Current item should be new data
        assert folio._items["data"]["is_current"] is True
        assert folio._items["data"]["in_snapshots"] == []

        # Snapshot version should be old data
        snapshot_item = folio._snapshot_versions[0]
        assert snapshot_item["name"] == "data"
        assert snapshot_item["is_current"] is False
        assert snapshot_item["in_snapshots"] == ["v1.0"]
        assert "@v1.0" in snapshot_item["filename"]  # Renamed with @ syntax

        # Current data should be new
        loaded_df = folio.get_table("data")
        pd.testing.assert_frame_equal(loaded_df, df2)

    def test_snapshot_filename_uses_first_snapshot_name(self, tmp_path):
        """Test that renamed file uses first snapshot name."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        # Add item
        folio.add_table("data", df1)

        # Mark as in multiple snapshots
        folio._items["data"]["in_snapshots"] = ["v1.0", "v1.1", "v1.2"]

        # Overwrite
        folio.add_table("data", df2, overwrite=True)

        # Snapshot version should use first snapshot name (v1.0)
        snapshot_item = folio._snapshot_versions[0]
        assert "@v1.0" in snapshot_item["filename"]
        assert snapshot_item["in_snapshots"] == ["v1.0", "v1.1", "v1.2"]

    def test_file_actually_renamed(self, tmp_path):
        """Test that physical file is renamed with @snapshot suffix."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        # Add item
        folio.add_table("data", df1)

        # Check original file exists
        tables_dir = Path(folio._bundle_dir) / "tables"
        original_file = tables_dir / "data.parquet"
        assert original_file.exists()

        # Mark as in snapshot
        folio._items["data"]["in_snapshots"] = ["v1.0"]

        # Overwrite - triggers copy-on-write
        folio.add_table("data", df2, overwrite=True)

        # Old file should be renamed to @v1.0
        snapshot_file = tables_dir / "data@v1.0.parquet"
        assert snapshot_file.exists()

        # New current file should exist
        assert original_file.exists()

        # Old file should have old data
        df_snapshot = pd.read_parquet(snapshot_file)
        pd.testing.assert_frame_equal(df_snapshot, df1)

        # New file should have new data
        df_current = pd.read_parquet(original_file)
        pd.testing.assert_frame_equal(df_current, df2)

    def test_items_json_contains_both_versions(self, tmp_path):
        """Test that items.json saves both current and snapshot versions."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        # Add item
        folio.add_table("data", df1)

        # Mark as in snapshot
        folio._items["data"]["in_snapshots"] = ["v1.0"]

        # Overwrite
        folio.add_table("data", df2, overwrite=True)

        # Read items.json directly
        items_path = Path(folio._bundle_dir) / "items.json"
        with open(items_path) as f:
            items_data = json.load(f)

        # Should have 2 items with same name
        items = items_data["items"]
        assert len(items) == 2

        # One should be current
        current_items = [item for item in items if item.get("is_current")]
        assert len(current_items) == 1
        assert current_items[0]["filename"] == "data.parquet"

        # One should be snapshot version
        snapshot_items = [item for item in items if not item.get("is_current")]
        assert len(snapshot_items) == 1
        assert "@v1.0" in snapshot_items[0]["filename"]
        assert "v1.0" in snapshot_items[0]["in_snapshots"]

    def test_reload_after_copy_on_write(self, tmp_path):
        """Test that bundle reloads correctly after copy-on-write."""
        # Create bundle and trigger copy-on-write
        folio1 = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio1.add_table("data", df1)
        folio1._items["data"]["in_snapshots"] = ["v1.0"]
        folio1.add_table("data", df2, overwrite=True)

        # Reload bundle
        folio2 = DataFolio(folio1._bundle_dir)

        # Should have 1 current + 1 snapshot version
        assert len(folio2._items) == 1
        assert len(folio2._snapshot_versions) == 1

        # Current should be new data
        loaded_df = folio2.get_table("data")
        pd.testing.assert_frame_equal(loaded_df, df2)

        # Snapshot version should be preserved
        assert folio2._snapshot_versions[0]["in_snapshots"] == ["v1.0"]
        assert "@v1.0" in folio2._snapshot_versions[0]["filename"]

    def test_multiple_overwrites(self, tmp_path):
        """Test multiple overwrites creating multiple snapshot versions."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        df3 = pd.DataFrame({"a": [7, 8, 9]})

        # First version
        folio.add_table("data", df1)
        folio._items["data"]["in_snapshots"] = ["v1.0"]

        # Second version (first overwrite)
        folio.add_table("data", df2, overwrite=True)
        folio._items["data"]["in_snapshots"] = ["v2.0"]

        # Third version (second overwrite)
        folio.add_table("data", df3, overwrite=True)

        # Should have 1 current + 2 snapshot versions
        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 2

        # Current should be newest data
        loaded_df = folio.get_table("data")
        pd.testing.assert_frame_equal(loaded_df, df3)

        # Should have v1.0 and v2.0 snapshot versions
        snapshot_files = [item["filename"] for item in folio._snapshot_versions]
        assert any("@v1.0" in f for f in snapshot_files)
        assert any("@v2.0" in f for f in snapshot_files)


class TestCopyOnWriteErrors:
    """Tests for error handling in copy-on-write logic."""

    def test_cannot_overwrite_without_flag(self, tmp_path):
        """Test that overwriting without flag raises error (backward compat)."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)

        # Should raise error without overwrite=True
        with pytest.raises(ValueError, match="already exists"):
            folio.add_table("data", df2)

    def test_can_overwrite_with_flag_when_not_in_snapshots(self, tmp_path):
        """Test that overwrite=True works when item not in snapshots."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)
        folio.add_table("data", df2, overwrite=True)  # Should work

        # Should only have 1 version
        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 0

    def test_automatic_copy_on_write_when_in_snapshots(self, tmp_path):
        """Test that copy-on-write happens automatically for items in snapshots."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)
        folio._items["data"]["in_snapshots"] = ["v1.0"]

        # Should work even without overwrite=True (snapshots force preservation)
        # Actually, it still requires overwrite=True for now
        folio.add_table("data", df2, overwrite=True)

        # Should have created snapshot version
        assert len(folio._snapshot_versions) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
