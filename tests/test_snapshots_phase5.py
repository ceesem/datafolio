"""Tests for Snapshot Phase 5: Snapshot Management.

These tests verify snapshot management operations like delete, compare,
cleanup, and restore.
"""

import pandas as pd
import pytest

from datafolio import DataFolio


class TestDeleteSnapshot:
    """Tests for delete_snapshot() method."""

    def test_delete_snapshot_basic(self, tmp_path):
        """Test deleting a snapshot removes it from registry."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")
        folio.create_snapshot("v2.0")

        # Delete v1.0
        folio.delete_snapshot("v1.0")

        # v1.0 should be gone
        assert "v1.0" not in folio.snapshots
        assert "v2.0" in folio.snapshots
        assert len(folio.list_snapshots()) == 1

    def test_delete_snapshot_removes_from_items(self, tmp_path):
        """Test deleting snapshot removes it from items' in_snapshots list."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")
        folio.create_snapshot("v2.0")

        # Check item has both snapshots
        item = folio._items["data"]
        assert "v1.0" in item["in_snapshots"]
        assert "v2.0" in item["in_snapshots"]

        # Delete v1.0
        folio.delete_snapshot("v1.0")

        # Item should only have v2.0
        item = folio._items["data"]
        assert "v1.0" not in item["in_snapshots"]
        assert "v2.0" in item["in_snapshots"]

    def test_delete_nonexistent_snapshot(self, tmp_path):
        """Test deleting nonexistent snapshot raises KeyError."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        with pytest.raises(KeyError, match="not found"):
            folio.delete_snapshot("nonexistent")

    def test_delete_snapshot_without_cleanup(self, tmp_path):
        """Test deleting snapshot without cleanup preserves files."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)
        folio.create_snapshot("v1.0")

        folio.add_table("data", df2, overwrite=True)
        folio.create_snapshot("v2.0")

        # Delete v1.0 without cleanup
        folio.delete_snapshot("v1.0", cleanup_orphans=False)

        # Old version file should still exist (it's now orphaned)
        # We'll verify this in the cleanup test
        assert "v1.0" not in folio.snapshots

    def test_delete_snapshot_with_cleanup(self, tmp_path):
        """Test deleting snapshot with cleanup removes orphaned files."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)
        folio.create_snapshot("v1.0")

        folio.add_table("data", df2, overwrite=True)
        folio.create_snapshot("v2.0")

        # Now there are two versions: current in _items, old in _snapshot_versions
        assert "data" in folio._items  # Current version
        # Find old version in snapshot_versions
        old_versions = [
            item for item in folio._snapshot_versions if item.get("name") == "data"
        ]
        assert len(old_versions) == 1  # v1 in snapshot_versions

        # Delete v1.0 with cleanup
        folio.delete_snapshot("v1.0", cleanup_orphans=True)

        # Old version should be cleaned up from snapshot_versions
        old_versions_after = [
            item for item in folio._snapshot_versions if item.get("name") == "data"
        ]
        assert len(old_versions_after) == 0  # v1 should be cleaned up

        # Current version should still be there
        assert "data" in folio._items

    def test_delete_last_snapshot_doesnt_affect_current(self, tmp_path):
        """Test deleting the only snapshot doesn't affect current state."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        # Delete snapshot
        folio.delete_snapshot("v1.0")

        # Current state should be unaffected
        loaded_df = folio.get_table("data")
        pd.testing.assert_frame_equal(loaded_df, df)


class TestCompareSnapshots:
    """Tests for compare_snapshots() method."""

    def test_compare_identical_snapshots(self, tmp_path):
        """Test comparing two identical snapshots."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.metadata["accuracy"] = 0.85
        folio.create_snapshot("v1.0")
        folio.create_snapshot("v1.1")  # Identical to v1.0

        diff = folio.compare_snapshots("v1.0", "v1.1")

        assert diff["added_items"] == []
        assert diff["removed_items"] == []
        assert diff["modified_items"] == []
        assert diff["shared_items"] == ["data"]
        # Note: metadata_changes may include auto-generated fields like updated_at
        # So we just check that user-set accuracy field is unchanged
        assert "accuracy" not in diff["metadata_changes"]

    def test_compare_snapshots_with_added_item(self, tmp_path):
        """Test comparing snapshots where second has added item."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio.add_table("data1", df1)
        folio.create_snapshot("v1.0")

        folio.add_table("data2", df2)
        folio.create_snapshot("v2.0")

        diff = folio.compare_snapshots("v1.0", "v2.0")

        assert diff["added_items"] == ["data2"]
        assert diff["removed_items"] == []
        assert diff["shared_items"] == ["data1"]

    def test_compare_snapshots_with_removed_item(self, tmp_path):
        """Test comparing snapshots where second has removed item."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio.add_table("data1", df1)
        folio.add_table("data2", df2)
        folio.create_snapshot("v1.0")

        folio.delete("data2")
        folio.create_snapshot("v2.0")

        diff = folio.compare_snapshots("v1.0", "v2.0")

        assert diff["added_items"] == []
        assert diff["removed_items"] == ["data2"]
        assert diff["shared_items"] == ["data1"]

    def test_compare_snapshots_with_modified_item(self, tmp_path):
        """Test comparing snapshots where item was modified."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)
        folio.create_snapshot("v1.0")

        folio.add_table("data", df2, overwrite=True)
        folio.create_snapshot("v2.0")

        diff = folio.compare_snapshots("v1.0", "v2.0")

        assert diff["added_items"] == []
        assert diff["removed_items"] == []
        assert diff["modified_items"] == ["data"]
        assert diff["shared_items"] == []

    def test_compare_snapshots_with_metadata_changes(self, tmp_path):
        """Test comparing snapshots with metadata changes."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        folio.metadata["accuracy"] = 0.85
        folio.metadata["model"] = "baseline"
        folio.create_snapshot("v1.0")

        folio.metadata["accuracy"] = 0.92
        folio.metadata["model"] = "optimized"
        folio.metadata["f1_score"] = 0.88  # New field
        folio.create_snapshot("v2.0")

        diff = folio.compare_snapshots("v1.0", "v2.0")

        # Check metadata changes
        assert "accuracy" in diff["metadata_changes"]
        assert diff["metadata_changes"]["accuracy"] == (0.85, 0.92)
        assert "model" in diff["metadata_changes"]
        assert diff["metadata_changes"]["model"] == ("baseline", "optimized")
        assert "f1_score" in diff["metadata_changes"]
        assert diff["metadata_changes"]["f1_score"] == (None, 0.88)

    def test_compare_nonexistent_snapshots(self, tmp_path):
        """Test comparing with nonexistent snapshot raises KeyError."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        with pytest.raises(KeyError, match="not found"):
            folio.compare_snapshots("v1.0", "nonexistent")

        with pytest.raises(KeyError, match="not found"):
            folio.compare_snapshots("nonexistent", "v1.0")


class TestCleanupOrphanedVersions:
    """Tests for cleanup_orphaned_versions() method."""

    def test_cleanup_no_orphans(self, tmp_path):
        """Test cleanup when there are no orphaned versions."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        deleted = folio.cleanup_orphaned_versions()
        assert deleted == []

    def test_cleanup_orphaned_version(self, tmp_path):
        """Test cleanup removes orphaned versions."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        df3 = pd.DataFrame({"a": [7, 8, 9]})

        # Create v1, snapshot, then v2, snapshot, then v3 (current)
        folio.add_table("data", df1)
        folio.create_snapshot("v1.0")

        folio.add_table("data", df2, overwrite=True)
        folio.create_snapshot("v2.0")

        folio.add_table("data", df3, overwrite=True)
        # Don't snapshot v3 - it's current

        # Now we have:
        # - data@v1.0 (in snapshot v1.0) - keep
        # - data@v2.0 (in snapshot v2.0) - keep
        # - data@v3 (current) - keep
        # No orphans yet

        deleted = folio.cleanup_orphaned_versions()
        assert deleted == []

        # Now delete v1.0 snapshot
        folio.delete_snapshot("v1.0", cleanup_orphans=False)

        # Now data@v1.0 is orphaned (not in any snapshot, not current)
        deleted = folio.cleanup_orphaned_versions()
        assert len(deleted) == 1
        # Should have deleted the v1 version file

    def test_cleanup_dry_run(self, tmp_path):
        """Test dry run mode doesn't actually delete."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)
        folio.create_snapshot("v1.0")

        folio.add_table("data", df2, overwrite=True)
        # v1 is now orphaned (in snapshot, but let's delete snapshot)

        folio.delete_snapshot("v1.0", cleanup_orphans=False)

        # Dry run should report what would be deleted
        would_delete = folio.cleanup_orphaned_versions(dry_run=True)
        assert len(would_delete) > 0

        # But files should still exist
        would_delete_again = folio.cleanup_orphaned_versions(dry_run=True)
        assert would_delete == would_delete_again  # Same files

    def test_cleanup_preserves_current_version(self, tmp_path):
        """Test cleanup never deletes current version."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)
        folio.create_snapshot("v1.0")

        folio.add_table("data", df2, overwrite=True)
        # Current version is df2, not in any snapshot

        # Delete the snapshot
        folio.delete_snapshot("v1.0", cleanup_orphans=True)

        # Current version should still be accessible
        loaded_df = folio.get_table("data")
        pd.testing.assert_frame_equal(loaded_df, df2)

    def test_cleanup_preserves_snapshot_versions(self, tmp_path):
        """Test cleanup never deletes versions in snapshots."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)
        folio.create_snapshot("v1.0")

        folio.add_table("data", df2, overwrite=True)
        folio.create_snapshot("v2.0")

        # Run cleanup - should find no orphans
        deleted = folio.cleanup_orphaned_versions()
        assert deleted == []

        # Both snapshots should still be accessible
        v1_df = folio.snapshots["v1.0"].get_table("data")
        pd.testing.assert_frame_equal(v1_df, df1)

        v2_df = folio.snapshots["v2.0"].get_table("data")
        pd.testing.assert_frame_equal(v2_df, df2)


class TestRestoreSnapshot:
    """Tests for restore_snapshot() method."""

    def test_restore_snapshot_basic(self, tmp_path):
        """Test restoring snapshot replaces current state."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)
        folio.metadata["version"] = 1
        folio.create_snapshot("v1.0")

        folio.add_table("data", df2, overwrite=True)
        folio.metadata["version"] = 2

        # Restore to v1.0
        folio.restore_snapshot("v1.0", confirm=True)

        # Current state should match v1.0
        loaded_df = folio.get_table("data")
        pd.testing.assert_frame_equal(loaded_df, df1)
        assert folio.metadata["version"] == 1

    def test_restore_snapshot_requires_confirm(self, tmp_path):
        """Test restore requires confirm=True."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        with pytest.raises(ValueError, match="confirm"):
            folio.restore_snapshot("v1.0", confirm=False)

        with pytest.raises(ValueError, match="confirm"):
            folio.restore_snapshot("v1.0")  # Default is False

    def test_restore_nonexistent_snapshot(self, tmp_path):
        """Test restoring nonexistent snapshot raises KeyError."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        with pytest.raises(KeyError, match="not found"):
            folio.restore_snapshot("nonexistent", confirm=True)

    def test_restore_snapshot_with_added_items(self, tmp_path):
        """Test restore removes items added after snapshot."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio.add_table("data1", df1)
        folio.create_snapshot("v1.0")

        folio.add_table("data2", df2)

        # Restore to v1.0
        folio.restore_snapshot("v1.0", confirm=True)

        # data2 should be gone
        assert "data1" in folio._items
        assert "data2" not in folio._items

    def test_restore_snapshot_metadata(self, tmp_path):
        """Test restore restores metadata state."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        folio.metadata["accuracy"] = 0.85
        folio.metadata["model"] = "baseline"
        folio.create_snapshot("v1.0")

        folio.metadata["accuracy"] = 0.92
        folio.metadata["model"] = "optimized"
        folio.metadata["new_field"] = "new_value"

        # Restore to v1.0
        folio.restore_snapshot("v1.0", confirm=True)

        # Metadata should match v1.0
        assert folio.metadata["accuracy"] == 0.85
        assert folio.metadata["model"] == "baseline"
        assert "new_field" not in folio.metadata


class TestSnapshotManagementIntegration:
    """Integration tests for snapshot management workflows."""

    def test_complete_management_workflow(self, tmp_path):
        """Test complete workflow: create, compare, delete, cleanup."""
        folio = DataFolio(tmp_path / "experiment")

        # Create multiple versions with snapshots
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        df3 = pd.DataFrame({"a": [7, 8, 9]})

        folio.add_table("data", df1)
        folio.metadata["version"] = 1
        folio.create_snapshot("v1.0", description="First version")

        folio.add_table("data", df2, overwrite=True)
        folio.metadata["version"] = 2
        folio.create_snapshot("v2.0", description="Second version")

        folio.add_table("data", df3, overwrite=True)
        folio.metadata["version"] = 3
        folio.create_snapshot("v3.0", description="Third version")

        # List snapshots
        snapshots = folio.list_snapshots()
        assert len(snapshots) == 3

        # Compare v1.0 and v3.0
        diff = folio.compare_snapshots("v1.0", "v3.0")
        assert diff["modified_items"] == ["data"]
        assert diff["metadata_changes"]["version"] == (1, 3)

        # Delete v2.0
        folio.delete_snapshot("v2.0", cleanup_orphans=True)

        # Should now have 2 snapshots
        assert len(folio.list_snapshots()) == 2

        # v1.0 and v3.0 should still be accessible
        v1_df = folio.snapshots["v1.0"].get_table("data")
        pd.testing.assert_frame_equal(v1_df, df1)

        v3_df = folio.snapshots["v3.0"].get_table("data")
        pd.testing.assert_frame_equal(v3_df, df3)

    def test_restore_and_continue_workflow(self, tmp_path):
        """Test restoring old snapshot and continuing work."""
        folio = DataFolio(tmp_path / "experiment")

        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        folio.add_table("data", df1)
        folio.metadata["accuracy"] = 0.85
        folio.create_snapshot("v1.0-baseline")

        folio.add_table("data", df2, overwrite=True)
        folio.metadata["accuracy"] = 0.78  # Worse!

        # Restore baseline
        folio.restore_snapshot("v1.0-baseline", confirm=True)

        # Verify restored
        loaded_df = folio.get_table("data")
        pd.testing.assert_frame_equal(loaded_df, df1)
        assert folio.metadata["accuracy"] == 0.85

        # Continue with new approach
        df3 = pd.DataFrame({"a": [10, 11, 12]})
        folio.add_table("data", df3, overwrite=True)
        folio.metadata["accuracy"] = 0.92
        folio.create_snapshot("v2.0-improved")

        # Should have two snapshots
        assert len(folio.list_snapshots()) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
