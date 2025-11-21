"""Tests for Snapshot Phase 4: Snapshot Loading.

These tests verify that snapshots can be accessed and read via the
SnapshotView and SnapshotAccessor classes.
"""

import pandas as pd
import pytest

from datafolio import DataFolio


class TestSnapshotAccessor:
    """Tests for dict-like snapshot accessor."""

    def test_access_snapshot_by_name(self, tmp_path):
        """Test accessing snapshot using dict-like syntax."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        # Access snapshot
        snapshot = folio.snapshots["v1.0"]
        assert snapshot.name == "v1.0"

    def test_snapshot_in_operator(self, tmp_path):
        """Test 'in' operator for checking snapshot existence."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        assert "v1.0" in folio.snapshots
        assert "v2.0" not in folio.snapshots

    def test_iterate_snapshots(self, tmp_path):
        """Test iterating over snapshot names."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")
        folio.add_table("data2", df.copy())
        folio.create_snapshot("v2.0")

        snapshot_names = list(folio.snapshots)
        assert "v1.0" in snapshot_names
        assert "v2.0" in snapshot_names
        assert len(snapshot_names) == 2

    def test_snapshot_len(self, tmp_path):
        """Test len() on snapshots accessor."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        assert len(folio.snapshots) == 0

        folio.create_snapshot("v1.0")
        assert len(folio.snapshots) == 1

        folio.add_table("data2", df.copy())
        folio.create_snapshot("v2.0")
        assert len(folio.snapshots) == 2

    def test_snapshot_keys_values_items(self, tmp_path):
        """Test keys(), values(), items() methods."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")
        folio.add_table("data2", df.copy())
        folio.create_snapshot("v2.0")

        # Keys
        assert set(folio.snapshots.keys()) == {"v1.0", "v2.0"}

        # Values
        snapshot_views = list(folio.snapshots.values())
        assert len(snapshot_views) == 2
        assert all(hasattr(s, "get_table") for s in snapshot_views)

        # Items
        items = list(folio.snapshots.items())
        assert len(items) == 2
        names = [name for name, _ in items]
        assert set(names) == {"v1.0", "v2.0"}

    def test_access_nonexistent_snapshot(self, tmp_path):
        """Test accessing nonexistent snapshot raises KeyError."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        with pytest.raises(KeyError, match="not found"):
            _ = folio.snapshots["nonexistent"]


class TestSnapshotView:
    """Tests for SnapshotView read-only access."""

    def test_snapshot_properties(self, tmp_path):
        """Test accessing snapshot metadata properties."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0", description="Test snapshot", tags=["test"])

        snapshot = folio.snapshots["v1.0"]
        assert snapshot.name == "v1.0"
        assert snapshot.description == "Test snapshot"
        assert snapshot.tags == ["test"]
        assert "data" in snapshot.item_versions
        assert snapshot.timestamp  # Should have timestamp

    def test_snapshot_metadata_property(self, tmp_path):
        """Test accessing snapshot metadata."""
        folio = DataFolio(
            tmp_path / "test-bundle", metadata={"experiment": "test", "value": 42}
        )
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        snapshot = folio.snapshots["v1.0"]
        assert snapshot.metadata["experiment"] == "test"
        assert snapshot.metadata["value"] == 42

    def test_get_table_from_current_snapshot(self, tmp_path):
        """Test reading table from snapshot that's still current."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        # Read from snapshot
        snapshot = folio.snapshots["v1.0"]
        loaded_df = snapshot.get_table("data")

        pd.testing.assert_frame_equal(loaded_df, df)

    def test_get_table_from_old_snapshot(self, tmp_path):
        """Test reading table from old snapshot after overwrite."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        # Create v1.0 snapshot
        folio.add_table("data", df1)
        folio.create_snapshot("v1.0")

        # Overwrite and create v2.0
        folio.add_table("data", df2, overwrite=True)
        folio.create_snapshot("v2.0")

        # Read from v1.0 snapshot - should get old data
        snapshot_v1 = folio.snapshots["v1.0"]
        loaded_df1 = snapshot_v1.get_table("data")
        pd.testing.assert_frame_equal(loaded_df1, df1)

        # Read from v2.0 snapshot - should get new data
        snapshot_v2 = folio.snapshots["v2.0"]
        loaded_df2 = snapshot_v2.get_table("data")
        pd.testing.assert_frame_equal(loaded_df2, df2)

        # Current version should also be new data
        current_df = folio.get_table("data")
        pd.testing.assert_frame_equal(current_df, df2)

    def test_get_nonexistent_table_from_snapshot(self, tmp_path):
        """Test getting nonexistent table from snapshot raises KeyError."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        snapshot = folio.snapshots["v1.0"]

        with pytest.raises(KeyError, match="not found"):
            snapshot.get_table("nonexistent")

    def test_snapshot_item_versions(self, tmp_path):
        """Test that item_versions dict shows correct items."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio.add_table("table1", df1)
        folio.add_table("table2", df2)
        folio.create_snapshot("v1.0")

        snapshot = folio.snapshots["v1.0"]
        assert "table1" in snapshot.item_versions
        assert "table2" in snapshot.item_versions


class TestListSnapshots:
    """Tests for list_snapshots() method."""

    def test_list_empty_snapshots(self, tmp_path):
        """Test listing snapshots when none exist."""
        folio = DataFolio(tmp_path / "test-bundle")
        snapshots = folio.list_snapshots()
        assert snapshots == []

    def test_list_single_snapshot(self, tmp_path):
        """Test listing single snapshot."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0", description="First snapshot", tags=["baseline"])

        snapshots = folio.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["name"] == "v1.0"
        assert snapshots[0]["description"] == "First snapshot"
        assert snapshots[0]["tags"] == ["baseline"]
        assert snapshots[0]["num_items"] == 1

    def test_list_multiple_snapshots_sorted(self, tmp_path):
        """Test that snapshots are sorted by timestamp (newest first)."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})

        # Create snapshots in order
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        folio.add_table("data2", df.copy())
        folio.create_snapshot("v2.0")

        folio.add_table("data3", df.copy())
        folio.create_snapshot("v3.0")

        snapshots = folio.list_snapshots()
        assert len(snapshots) == 3

        # Should be sorted newest first
        assert snapshots[0]["name"] == "v3.0"
        assert snapshots[1]["name"] == "v2.0"
        assert snapshots[2]["name"] == "v1.0"

        # Timestamps should be in descending order
        assert snapshots[0]["timestamp"] >= snapshots[1]["timestamp"]
        assert snapshots[1]["timestamp"] >= snapshots[2]["timestamp"]


class TestSnapshotReloadingPhase4:
    """Tests for loading bundles and accessing snapshots."""

    def test_reload_and_access_snapshots(self, tmp_path):
        """Test that snapshots are accessible after reloading bundle."""
        # Create bundle with snapshots
        folio1 = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio1.add_table("data", df1)
        folio1.create_snapshot("v1.0")

        folio1.add_table("data", df2, overwrite=True)
        folio1.create_snapshot("v2.0")

        # Reload bundle
        folio2 = DataFolio(folio1._bundle_dir)

        # Should be able to access both snapshots
        assert "v1.0" in folio2.snapshots
        assert "v2.0" in folio2.snapshots

        # Should be able to read old data from v1.0
        snapshot_v1 = folio2.snapshots["v1.0"]
        loaded_df1 = snapshot_v1.get_table("data")
        pd.testing.assert_frame_equal(loaded_df1, df1)

        # Should be able to read new data from v2.0
        snapshot_v2 = folio2.snapshots["v2.0"]
        loaded_df2 = snapshot_v2.get_table("data")
        pd.testing.assert_frame_equal(loaded_df2, df2)


class TestCompleteSnapshotWorkflow:
    """End-to-end tests for complete snapshot workflow."""

    def test_complete_versioning_workflow(self, tmp_path):
        """Test complete workflow: create, snapshot, overwrite, access old."""
        folio = DataFolio(tmp_path / "experiment")

        # Initial state
        train_v1 = pd.DataFrame({"features": [1, 2, 3], "labels": [0, 1, 0]})
        folio.add_table("train_data", train_v1)
        folio.metadata["accuracy"] = 0.85
        folio.create_snapshot("v1.0-baseline", description="Initial model")

        # Improved state
        train_v2 = pd.DataFrame(
            {"features": [1, 2, 3, 4, 5], "labels": [0, 1, 0, 1, 1]}
        )
        folio.add_table("train_data", train_v2, overwrite=True)
        folio.metadata["accuracy"] = 0.92
        folio.create_snapshot("v2.0-optimized", description="Tuned model")

        # Verify we have both snapshots
        snapshots = folio.list_snapshots()
        assert len(snapshots) == 2

        # Access v1.0 snapshot
        v1_snapshot = folio.snapshots["v1.0-baseline"]
        v1_data = v1_snapshot.get_table("train_data")
        assert len(v1_data) == 3  # Original had 3 rows
        assert v1_snapshot.metadata["accuracy"] == 0.85

        # Access v2.0 snapshot
        v2_snapshot = folio.snapshots["v2.0-optimized"]
        v2_data = v2_snapshot.get_table("train_data")
        assert len(v2_data) == 5  # New version has 5 rows
        assert v2_snapshot.metadata["accuracy"] == 0.92

        # Current state should match v2.0
        current_data = folio.get_table("train_data")
        pd.testing.assert_frame_equal(current_data, train_v2)
        assert folio.metadata["accuracy"] == 0.92


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
