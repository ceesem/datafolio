"""Tests for snapshot export functionality.

These tests verify:
- Export creates clean, standalone bundles
- Exported bundles contain correct items and metadata
- Source snapshot information is preserved
- Export handles various data types
"""

import pandas as pd
import pytest

from datafolio import DataFolio


class TestExportSnapshot:
    """Test snapshot export functionality."""

    def test_export_basic(self, tmp_path):
        """Test basic snapshot export."""
        # Create source bundle with data
        source = DataFolio(tmp_path / "source")
        df = pd.DataFrame({"a": [1, 2, 3]})
        source.add_table("data", df)
        source.metadata["accuracy"] = 0.89
        source.create_snapshot("v1.0")

        # Export snapshot
        exported = source.export_snapshot("v1.0", tmp_path / "exported")

        # Verify exported bundle
        assert not exported.read_only
        assert exported.metadata["accuracy"] == 0.89
        assert "data" in exported._items

        # Verify data is intact
        exported_df = exported.get_table("data")
        pd.testing.assert_frame_equal(exported_df, df)

    def test_export_includes_source_metadata(self, tmp_path):
        """Test that export includes source snapshot metadata."""
        source = DataFolio(tmp_path / "source")
        df = pd.DataFrame({"a": [1, 2, 3]})
        source.add_table("data", df)
        source.create_snapshot("v1.0", description="Test snapshot", tags=["baseline"])

        # Export with metadata
        exported = source.export_snapshot("v1.0", tmp_path / "exported")

        # Check source snapshot metadata
        assert "_source_snapshot" in exported.metadata
        source_info = exported.metadata["_source_snapshot"]
        assert source_info["name"] == "v1.0"
        assert source_info["description"] == "Test snapshot"
        assert "baseline" in source_info["tags"]
        assert "source_bundle" in source_info

    def test_export_without_source_metadata(self, tmp_path):
        """Test export without source snapshot metadata."""
        source = DataFolio(tmp_path / "source")
        df = pd.DataFrame({"a": [1, 2, 3]})
        source.add_table("data", df)
        source.create_snapshot("v1.0")

        # Export without metadata
        exported = source.export_snapshot(
            "v1.0", tmp_path / "exported", include_snapshot_metadata=False
        )

        # Should not have source snapshot metadata
        assert "_source_snapshot" not in exported.metadata

    def test_export_creates_clean_bundle(self, tmp_path):
        """Test that export creates bundle without version history."""
        source = DataFolio(tmp_path / "source")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        source.add_table("data", df1)
        source.create_snapshot("v1.0")

        # Add more data and create another snapshot
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        source.add_table("data2", df2)
        source.create_snapshot("v2.0")

        # Export v1.0
        exported = source.export_snapshot("v1.0", tmp_path / "exported")

        # Exported bundle should have no snapshots
        assert len(exported._snapshots) == 0

        # Should only have data from v1.0
        assert "data" in exported._items
        assert "data2" not in exported._items

    def test_export_multiple_items(self, tmp_path):
        """Test exporting snapshot with multiple items."""
        source = DataFolio(tmp_path / "source")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        source.add_table("table1", df1)
        source.add_table("table2", df2)
        source.add_json("config", {"key": "value"})
        source.metadata["experiment"] = "test"
        source.create_snapshot("v1.0")

        # Export
        exported = source.export_snapshot("v1.0", tmp_path / "exported")

        # Verify all items present
        assert "table1" in exported._items
        assert "table2" in exported._items
        assert "config" in exported._items

        # Verify data integrity
        pd.testing.assert_frame_equal(exported.get_table("table1"), df1)
        pd.testing.assert_frame_equal(exported.get_table("table2"), df2)
        assert exported.get_json("config") == {"key": "value"}
        assert exported.metadata["experiment"] == "test"

    def test_export_preserves_metadata(self, tmp_path):
        """Test that export preserves custom metadata."""
        source = DataFolio(tmp_path / "source")
        df = pd.DataFrame({"a": [1, 2, 3]})
        source.add_table("data", df)
        source.metadata["accuracy"] = 0.89
        source.metadata["model_type"] = "random_forest"
        source.metadata["dataset"] = "iris"
        source.create_snapshot("v1.0")

        # Export
        exported = source.export_snapshot("v1.0", tmp_path / "exported")

        # All custom metadata should be preserved
        assert exported.metadata["accuracy"] == 0.89
        assert exported.metadata["model_type"] == "random_forest"
        assert exported.metadata["dataset"] == "iris"

    def test_export_nonexistent_snapshot(self, tmp_path):
        """Test exporting nonexistent snapshot raises error."""
        source = DataFolio(tmp_path / "source")

        with pytest.raises(KeyError, match="not found"):
            source.export_snapshot("nonexistent", tmp_path / "exported")

    def test_export_to_existing_path(self, tmp_path):
        """Test that export fails if target exists."""
        source = DataFolio(tmp_path / "source")
        df = pd.DataFrame({"a": [1, 2, 3]})
        source.add_table("data", df)
        source.create_snapshot("v1.0")

        # Create target directory
        target = tmp_path / "exported"
        target.mkdir()

        # Should fail
        with pytest.raises(ValueError, match="already exists"):
            source.export_snapshot("v1.0", target)

    def test_export_is_mutable(self, tmp_path):
        """Test that exported bundle is mutable."""
        source = DataFolio(tmp_path / "source")
        df = pd.DataFrame({"a": [1, 2, 3]})
        source.add_table("data", df)
        source.create_snapshot("v1.0")

        # Export
        exported = source.export_snapshot("v1.0", tmp_path / "exported")

        # Should be mutable
        assert not exported.read_only

        # Should be able to add data
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        exported.add_table("new_data", df2)
        assert "new_data" in exported._items

    def test_export_snapshot_from_history(self, tmp_path):
        """Test exporting old snapshot after multiple changes."""
        source = DataFolio(tmp_path / "source")

        # v1.0: Initial state
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        source.add_table("data", df1)
        source.metadata["version"] = "1.0"
        source.create_snapshot("v1.0")

        # v2.0: Modified state
        source.add_table("data", pd.DataFrame({"a": [4, 5, 6]}), overwrite=True)
        source.metadata["version"] = "2.0"
        source.create_snapshot("v2.0")

        # v3.0: Further modified
        source.add_table("data", pd.DataFrame({"a": [7, 8, 9]}), overwrite=True)
        source.metadata["version"] = "3.0"
        source.create_snapshot("v3.0")

        # Export v1.0
        exported = source.export_snapshot("v1.0", tmp_path / "exported")

        # Should have v1.0 state
        assert exported.metadata["version"] == "1.0"
        exported_df = exported.get_table("data")
        pd.testing.assert_frame_equal(exported_df, df1)


class TestExportWorkflows:
    """Integration tests for export workflows."""

    def test_share_baseline_workflow(self, tmp_path):
        """Test workflow: export baseline for sharing."""
        # Research bundle with multiple iterations
        research = DataFolio(tmp_path / "research")
        df = pd.DataFrame({"data": [1, 2, 3]})
        research.add_table("data", df)
        research.metadata["accuracy"] = 0.89
        research.create_snapshot("baseline")

        # Continue research
        research.metadata["accuracy"] = 0.91
        research.create_snapshot("improved")

        # Export baseline for collaborator
        shared = research.export_snapshot("baseline", tmp_path / "shared")

        # Collaborator gets clean bundle with baseline state
        assert shared.metadata["accuracy"] == 0.89
        assert len(shared._snapshots) == 0  # No history
        assert "_source_snapshot" in shared.metadata

    def test_deployment_workflow(self, tmp_path):
        """Test workflow: export for production deployment."""
        # Development bundle
        dev = DataFolio(tmp_path / "dev")
        df = pd.DataFrame({"features": [1, 2, 3]})
        dev.add_table("training_data", df)
        dev.metadata["model_version"] = "2.1"
        dev.metadata["environment"] = "development"
        dev.create_snapshot("production-v2.1", tags=["production"])

        # Export for deployment
        prod = dev.export_snapshot("production-v2.1", tmp_path / "production")

        # Production bundle is clean
        assert prod.metadata["model_version"] == "2.1"
        assert "training_data" in prod._items
        assert len(prod._snapshots) == 0

    def test_archive_workflow(self, tmp_path):
        """Test workflow: archive old snapshot."""
        # Active research bundle
        active = DataFolio(tmp_path / "active")
        df = pd.DataFrame({"data": [1, 2, 3]})
        active.add_table("data", df)
        active.metadata["experiment"] = "paper-2024"
        active.create_snapshot("paper-submission")

        # Many more iterations...
        for i in range(5):
            active.metadata["iteration"] = i
            active.create_snapshot(f"iter-{i}")

        # Archive paper submission before cleaning up
        archive = active.export_snapshot(
            "paper-submission", tmp_path / "archive/paper-2024"
        )

        # Archive is standalone
        assert archive.metadata["experiment"] == "paper-2024"
        assert len(archive._snapshots) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
