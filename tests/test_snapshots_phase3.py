"""Tests for Snapshot Phase 3: Snapshot Creation.

These tests verify that create_snapshot() works correctly and captures
all necessary context (git, environment, execution).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from datafolio import DataFolio


class TestSnapshotCreation:
    """Tests for basic snapshot creation functionality."""

    def test_create_simple_snapshot(self, tmp_path):
        """Test creating a basic snapshot."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Create snapshot
        folio.create_snapshot("v1.0", description="First snapshot")

        # Snapshot should exist
        assert "v1.0" in folio._snapshots

        # Should have captured metadata
        snapshot = folio._snapshots["v1.0"]
        assert snapshot["name"] == "v1.0"
        assert snapshot["description"] == "First snapshot"
        assert "timestamp" in snapshot
        assert "item_versions" in snapshot
        assert "metadata_snapshot" in snapshot

    def test_snapshot_marks_items(self, tmp_path):
        """Test that snapshot marks all current items."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio.add_table("data1", df1)
        folio.add_table("data2", df2)

        # Create snapshot
        folio.create_snapshot("v1.0")

        # Both items should be marked as in snapshot
        assert "v1.0" in folio._items["data1"]["in_snapshots"]
        assert "v1.0" in folio._items["data2"]["in_snapshots"]

    def test_snapshot_with_tags(self, tmp_path):
        """Test creating snapshot with tags."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Create snapshot with tags
        folio.create_snapshot("v1.0", tags=["baseline", "paper"])

        snapshot = folio._snapshots["v1.0"]
        assert snapshot["tags"] == ["baseline", "paper"]

    def test_snapshot_captures_item_versions(self, tmp_path):
        """Test that snapshot captures current item versions."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio.add_table("data1", df1)
        folio.add_table("data2", df2)

        # Create snapshot
        folio.create_snapshot("v1.0")

        # Should capture both items with their checksums
        snapshot = folio._snapshots["v1.0"]
        assert "data1" in snapshot["item_versions"]
        assert "data2" in snapshot["item_versions"]
        # item_versions now stores checksums (strings) not version numbers
        assert isinstance(snapshot["item_versions"]["data1"], str)
        assert isinstance(snapshot["item_versions"]["data2"], str)
        assert len(snapshot["item_versions"]["data1"]) > 0  # Non-empty checksum
        assert len(snapshot["item_versions"]["data2"]) > 0  # Non-empty checksum

    def test_snapshot_captures_metadata_state(self, tmp_path):
        """Test that snapshot captures current metadata."""
        folio = DataFolio(
            tmp_path / "test-bundle", metadata={"experiment": "test1", "value": 42}
        )
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Create snapshot
        folio.create_snapshot("v1.0")

        # Should capture metadata
        snapshot = folio._snapshots["v1.0"]
        assert "metadata_snapshot" in snapshot
        assert snapshot["metadata_snapshot"]["experiment"] == "test1"
        assert snapshot["metadata_snapshot"]["value"] == 42

    def test_snapshots_saved_to_file(self, tmp_path):
        """Test that snapshots are persisted to snapshots.json."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Create snapshot
        folio.create_snapshot("v1.0", description="Test snapshot")

        # Check snapshots.json exists
        snapshots_path = Path(folio._bundle_dir) / "snapshots.json"
        assert snapshots_path.exists()

        # Check contents
        with open(snapshots_path) as f:
            snapshots_data = json.load(f)

        assert "snapshots" in snapshots_data
        assert "v1.0" in snapshots_data["snapshots"]
        assert snapshots_data["snapshots"]["v1.0"]["description"] == "Test snapshot"

    def test_snapshot_updates_items_json(self, tmp_path):
        """Test that creating snapshot updates items.json with in_snapshots."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Create snapshot
        folio.create_snapshot("v1.0")

        # Read items.json
        items_path = Path(folio._bundle_dir) / "items.json"
        with open(items_path) as f:
            items_data = json.load(f)

        # Item should have v1.0 in in_snapshots
        item = items_data["items"][0]
        assert "v1.0" in item["in_snapshots"]


class TestSnapshotValidation:
    """Tests for snapshot name validation and error handling."""

    def test_reject_invalid_snapshot_names(self, tmp_path):
        """Test that invalid snapshot names are rejected."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Should reject @ symbol
        with pytest.raises(ValueError, match="can only contain"):
            folio.create_snapshot("v1@test")

        # Should reject invalid characters
        with pytest.raises(ValueError, match="can only contain"):
            folio.create_snapshot("v1/test")

    def test_reject_duplicate_snapshot_names(self, tmp_path):
        """Test that duplicate snapshot names are rejected."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Create first snapshot
        folio.create_snapshot("v1.0")

        # Should reject duplicate
        with pytest.raises(ValueError, match="already exists"):
            folio.create_snapshot("v1.0")

    def test_valid_snapshot_names(self, tmp_path):
        """Test that valid snapshot names are accepted."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # These should all work
        folio.create_snapshot("v1.0")
        folio.add_table("data2", df.copy())
        folio.create_snapshot("baseline-2024")
        folio.add_table("data3", df.copy())
        folio.create_snapshot("experiment_1")
        folio.add_table("data4", df.copy())
        folio.create_snapshot("v2.0.1")

        assert len(folio._snapshots) == 4


class TestSnapshotContextCapture:
    """Tests for environment/git/execution context capture."""

    def test_snapshot_captures_environment(self, tmp_path):
        """Test that environment info is captured when explicitly enabled."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Create snapshot with environment capture enabled
        folio.create_snapshot("v1.0", capture_environment=True)

        snapshot = folio._snapshots["v1.0"]
        assert "environment" in snapshot
        assert "python_version" in snapshot["environment"]
        assert "platform" in snapshot["environment"]

    def test_snapshot_can_disable_context_capture(self, tmp_path):
        """Test that context capture can be disabled."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Create snapshot without context
        folio.create_snapshot(
            "v1.0",
            capture_git=False,
            capture_environment=False,
            capture_execution=False,
        )

        snapshot = folio._snapshots["v1.0"]
        assert "git" not in snapshot
        assert "environment" not in snapshot
        assert "execution" not in snapshot

    def test_git_info_captured_when_in_repo(self, tmp_path):
        """Test that git info is captured when in a git repo."""
        # This test will only pass if run in a git repo
        # It's okay if it's skipped in non-git environments
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Create snapshot
        folio.create_snapshot("v1.0")

        snapshot = folio._snapshots["v1.0"]
        # Git info may or may not be present depending on test environment
        # Just check it doesn't crash
        if "git" in snapshot:
            assert "commit" in snapshot["git"]
            assert "branch" in snapshot["git"]


class TestSnapshotReloading:
    """Tests for loading bundles with snapshots."""

    def test_reload_bundle_with_snapshots(self, tmp_path):
        """Test that snapshots are loaded when reopening a bundle."""
        # Create bundle and snapshot
        folio1 = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio1.add_table("data", df)
        folio1.create_snapshot("v1.0", description="Test snapshot")

        # Reload bundle
        folio2 = DataFolio(folio1._bundle_dir)

        # Snapshot should be loaded
        assert "v1.0" in folio2._snapshots
        assert folio2._snapshots["v1.0"]["description"] == "Test snapshot"

        # Item should still be marked as in snapshot
        assert "v1.0" in folio2._items["data"]["in_snapshots"]


class TestSnapshotIntegrationWithCopyOnWrite:
    """Tests for integration between snapshots and copy-on-write."""

    def test_snapshot_then_overwrite_triggers_copy_on_write(self, tmp_path):
        """Test that overwriting after snapshot creates new version."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        # Add data and create snapshot
        folio.add_table("data", df1)
        folio.create_snapshot("v1.0")

        # Overwrite - should trigger copy-on-write
        folio.add_table("data", df2, overwrite=True)

        # Should have 1 current + 1 snapshot version
        assert len(folio._items) == 1
        assert len(folio._snapshot_versions) == 1

        # Snapshot version should be preserved
        snapshot_item = folio._snapshot_versions[0]
        assert "@v1.0" in snapshot_item["filename"]
        assert "v1.0" in snapshot_item["in_snapshots"]

        # Current version should have new data
        loaded_df = folio.get_table("data")
        pd.testing.assert_frame_equal(loaded_df, df2)

    def test_multiple_snapshots_same_item(self, tmp_path):
        """Test creating multiple snapshots of same item before overwriting."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        # Add data
        folio.add_table("data", df1)

        # Create multiple snapshots
        folio.create_snapshot("v1.0")
        folio.create_snapshot("v1.1")  # Same data, different snapshot

        # Item should be in both snapshots
        assert "v1.0" in folio._items["data"]["in_snapshots"]
        assert "v1.1" in folio._items["data"]["in_snapshots"]

        # Overwrite - should preserve for both snapshots
        folio.add_table("data", df2, overwrite=True)

        # Should have snapshot version with both snapshots referenced
        snapshot_item = folio._snapshot_versions[0]
        assert "v1.0" in snapshot_item["in_snapshots"]
        assert "v1.1" in snapshot_item["in_snapshots"]

        # Filename should use first snapshot name
        assert "@v1.0" in snapshot_item["filename"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
