"""Tests for Snapshot Phase 1: Core Data Structures.

These tests verify that the extended items.json and snapshots.json structures
work correctly, including backward compatibility with existing bundles.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from datafolio import DataFolio


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing bundles."""

    def test_load_old_items_format(self, tmp_path):
        """Test loading bundle with old items.json format (list)."""
        # Create a bundle directory manually with old format
        bundle_dir = tmp_path / "old-bundle"
        bundle_dir.mkdir()
        (bundle_dir / "tables").mkdir()
        (bundle_dir / "models").mkdir()
        (bundle_dir / "artifacts").mkdir()

        # Create old-format items.json (just a list)
        old_items = [
            {
                "name": "test_table",
                "item_type": "included_table",
                "filename": "test_table.parquet",
                "table_format": "parquet",
                "is_directory": False,
                "checksum": "abc123",
                "num_rows": 100,
                "num_cols": 5,
                "columns": ["a", "b", "c", "d", "e"],
                "dtypes": {
                    "a": "int64",
                    "b": "float64",
                    "c": "object",
                    "d": "bool",
                    "e": "datetime64[ns]",
                },
            }
        ]

        items_path = bundle_dir / "items.json"
        with open(items_path, "w") as f:
            json.dump(old_items, f)

        # Create minimal metadata.json
        metadata_path = bundle_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({"created_at": "2024-01-15T10:00:00Z"}, f)

        # Load the bundle - should auto-upgrade
        folio = DataFolio(bundle_dir)

        # Verify items loaded correctly
        assert "test_table" in folio._items
        assert folio._items["test_table"]["name"] == "test_table"

        # Verify snapshot fields were added
        assert folio._items["test_table"]["is_current"] is True
        assert folio._items["test_table"]["in_snapshots"] == []

        # Verify snapshots initialized as empty
        assert folio._snapshots == {}

    def test_load_old_bundle_without_snapshots(self, tmp_path):
        """Test loading bundle that doesn't have snapshots.json."""
        # Create a normal bundle using DataFolio
        folio1 = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        folio1.add_table("data", df)

        # Verify snapshots.json doesn't exist
        snapshots_path = Path(folio1._bundle_dir) / "snapshots.json"
        assert not snapshots_path.exists()

        # Reload the bundle
        folio2 = DataFolio(folio1._bundle_dir)

        # Should have empty snapshots
        assert folio2._snapshots == {}

        # But should have items loaded correctly
        assert "data" in folio2._items


class TestNewItemsFormat:
    """Tests for new items.json format with versioning."""

    def test_new_bundle_has_snapshot_fields(self, tmp_path):
        """Test that new bundles save items with snapshot fields."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        folio.add_table("data", df)

        # Read items.json directly
        items_path = Path(folio._bundle_dir) / "items.json"
        with open(items_path) as f:
            items_data = json.load(f)

        # Should be new format: dict with items list
        assert "items" in items_data

        # Check snapshot fields are present
        assert len(items_data["items"]) == 1
        item = items_data["items"][0]
        assert "is_current" in item
        assert "in_snapshots" in item
        assert item["is_current"] is True
        assert item["in_snapshots"] == []

    def test_reload_new_format_bundle(self, tmp_path):
        """Test that bundles with new format reload correctly."""
        # Create bundle with new format
        folio1 = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        folio1.add_table("data", df)

        # Reload
        folio2 = DataFolio(folio1._bundle_dir)

        # Verify snapshot fields loaded
        assert folio2._items["data"]["is_current"] is True
        assert folio2._items["data"]["in_snapshots"] == []


class TestSnapshotsManifest:
    """Tests for snapshots.json loading and saving."""

    def test_empty_snapshots_on_new_bundle(self, tmp_path):
        """Test that new bundles have empty snapshots."""
        folio = DataFolio(tmp_path / "test-bundle")

        assert folio._snapshots == {}

    def test_snapshots_not_created_automatically(self, tmp_path):
        """Test that snapshots.json is not created until needed."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # snapshots.json should NOT exist yet (we haven't created any snapshots)
        snapshots_path = Path(folio._bundle_dir) / "snapshots.json"
        assert not snapshots_path.exists()

    def test_manual_snapshots_json_loading(self, tmp_path):
        """Test loading a bundle with manually created snapshots.json."""
        # Create bundle
        bundle_dir = tmp_path / "test-bundle"
        bundle_dir.mkdir()
        (bundle_dir / "tables").mkdir()
        (bundle_dir / "models").mkdir()
        (bundle_dir / "artifacts").mkdir()

        # Create items.json in new format
        items_data = {
            "items": [
                {
                    "name": "data",
                    "item_type": "included_table",
                    "filename": "data.parquet",
                    "table_format": "parquet",
                    "is_directory": False,
                    "checksum": "abc123",
                    "num_rows": 100,
                    "num_cols": 2,
                    "columns": ["a", "b"],
                    "dtypes": {"a": "int64", "b": "int64"},
                    "is_current": False,
                    "in_snapshots": ["v1.0"],
                }
            ],
        }

        with open(bundle_dir / "items.json", "w") as f:
            json.dump(items_data, f)

        # Create snapshots.json
        snapshots_data = {
            "snapshots": {
                "v1.0": {
                    "name": "v1.0",
                    "timestamp": "2025-11-20T15:00:00Z",
                    "description": "First snapshot",
                    "tags": ["baseline"],
                    "item_versions": {"data": 1},
                    "metadata_snapshot": {"experiment": "test"},
                }
            }
        }

        with open(bundle_dir / "snapshots.json", "w") as f:
            json.dump(snapshots_data, f)

        # Create metadata.json
        with open(bundle_dir / "metadata.json", "w") as f:
            json.dump({"created_at": "2025-11-20T10:00:00Z"}, f)

        # Load bundle
        folio = DataFolio(bundle_dir)

        # Verify snapshots loaded
        assert "v1.0" in folio._snapshots
        assert folio._snapshots["v1.0"]["name"] == "v1.0"
        assert folio._snapshots["v1.0"]["description"] == "First snapshot"
        assert folio._snapshots["v1.0"]["tags"] == ["baseline"]
        assert folio._snapshots["v1.0"]["item_versions"]["data"] == 1


class TestItemNameValidation:
    """Tests for item name validation."""

    def test_reject_at_symbol_in_name(self, tmp_path):
        """Test that @ symbol is rejected in item names."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})

        # Should raise ValueError for @ in name
        with pytest.raises(ValueError, match="cannot contain '@' symbol"):
            folio.add_table("my@table", df)

    def test_valid_item_names(self, tmp_path):
        """Test that valid item names are accepted."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})

        # These should all be valid
        folio.add_table("my_table", df)
        folio.add_table("my-table-2", df.copy())
        folio.add_table("MyTable3", df.copy())
        folio.add_table("table.v1", df.copy())

        assert len(folio._items) == 4


class TestMultipleItems:
    """Tests for bundles with multiple items and version tracking."""

    def test_multiple_items_snapshot_tracking(self, tmp_path):
        """Test that multiple items are tracked correctly."""
        folio = DataFolio(tmp_path / "test-bundle")

        # Add multiple items
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        folio.add_table("table1", df1)
        folio.add_table("table2", df2)

        # Both should be current versions
        assert folio._items["table1"]["is_current"] is True
        assert folio._items["table2"]["is_current"] is True

        # Both should have empty in_snapshots
        assert folio._items["table1"]["in_snapshots"] == []
        assert folio._items["table2"]["in_snapshots"] == []

    def test_reload_multiple_items(self, tmp_path):
        """Test reloading bundle with multiple items."""
        # Create bundle
        folio1 = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        folio1.add_table("table1", df1)
        folio1.add_table("table2", df2)

        # Reload
        folio2 = DataFolio(folio1._bundle_dir)

        # Verify both items loaded with snapshot tracking
        assert folio2._items["table1"]["is_current"] is True
        assert folio2._items["table2"]["is_current"] is True
        assert len(folio2._items) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
