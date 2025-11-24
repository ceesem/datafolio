"""Tests for read-only mode and load_snapshot functionality.

These tests verify:
- read_only parameter works on any DataFolio
- load_snapshot() classmethod creates read-only folios by default
- All write operations are blocked in read-only mode
- Read operations work normally in read-only mode
- Multiple snapshots can be loaded simultaneously
"""

import pandas as pd
import pytest

from datafolio import DataFolio


class TestReadOnlyMode:
    """Test read-only mode on any folio."""

    def test_read_only_flag_default_false(self, tmp_path):
        """Test read_only defaults to False."""
        folio = DataFolio(tmp_path / "test")
        assert folio.read_only is False
        assert folio.in_snapshot_mode is False
        assert folio.loaded_snapshot is None

    def test_read_only_prevents_add_data(self, tmp_path):
        """Test read-only prevents add_data()."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("existing", df)

        # Open as read-only
        folio_ro = DataFolio(tmp_path / "test", read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            folio_ro.add_data("new", df)

    def test_read_only_prevents_delete(self, tmp_path):
        """Test read-only prevents delete()."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Open as read-only
        folio_ro = DataFolio(tmp_path / "test", read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            folio_ro.delete("data")

    def test_read_only_prevents_metadata_write(self, tmp_path):
        """Test read-only prevents metadata modifications."""
        folio = DataFolio(tmp_path / "test")

        # Open as read-only
        folio_ro = DataFolio(tmp_path / "test", read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            folio_ro.metadata["test"] = 123

    def test_read_only_prevents_metadata_update(self, tmp_path):
        """Test read-only prevents metadata.update()."""
        folio = DataFolio(tmp_path / "test")

        # Open as read-only
        folio_ro = DataFolio(tmp_path / "test", read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            folio_ro.metadata.update({"test": 123})

    def test_read_only_prevents_metadata_delete(self, tmp_path):
        """Test read-only prevents metadata deletion."""
        folio = DataFolio(tmp_path / "test")
        folio.metadata["test"] = 123

        # Open as read-only
        folio_ro = DataFolio(tmp_path / "test", read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            del folio_ro.metadata["test"]

    def test_read_only_prevents_create_snapshot(self, tmp_path):
        """Test read-only prevents creating snapshots."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Open as read-only
        folio_ro = DataFolio(tmp_path / "test", read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            folio_ro.create_snapshot("v1.0")

    def test_read_only_prevents_delete_snapshot(self, tmp_path):
        """Test read-only prevents deleting snapshots."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        # Open as read-only
        folio_ro = DataFolio(tmp_path / "test", read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            folio_ro.delete_snapshot("v1.0")

    def test_read_only_prevents_restore_snapshot(self, tmp_path):
        """Test read-only prevents restoring snapshots."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        # Open as read-only
        folio_ro = DataFolio(tmp_path / "test", read_only=True)

        with pytest.raises(RuntimeError, match="read-only"):
            folio_ro.restore_snapshot("v1.0", confirm=True)

    def test_read_only_allows_read_operations(self, tmp_path):
        """Test read-only allows all read operations."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.metadata["accuracy"] = 0.89

        # Open as read-only
        folio_ro = DataFolio(tmp_path / "test", read_only=True)

        # All read operations should work
        assert folio_ro.get_table("data") is not None
        assert folio_ro.metadata["accuracy"] == 0.89
        folio_ro.describe()  # Should not error
        assert len(folio_ro.list_contents()) > 0

    def test_repr_shows_read_only(self, tmp_path):
        """Test __repr__ shows READ-ONLY status."""
        folio = DataFolio(tmp_path / "test", read_only=True)
        repr_str = repr(folio)
        assert "[READ-ONLY]" in repr_str


class TestLoadSnapshot:
    """Test loading snapshots as DataFolio instances."""

    def test_load_snapshot_basic(self, tmp_path):
        """Test basic snapshot loading."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.metadata["accuracy"] = 0.89
        folio.create_snapshot("v1.0")

        # Load snapshot
        snapshot = DataFolio.load_snapshot(tmp_path / "test", "v1.0")

        assert snapshot.in_snapshot_mode
        assert snapshot.loaded_snapshot == "v1.0"
        assert snapshot.read_only  # Default is True
        assert snapshot.metadata["accuracy"] == 0.89

    def test_load_snapshot_read_only_by_default(self, tmp_path):
        """Test snapshots load as read-only by default."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        snapshot = DataFolio.load_snapshot(tmp_path / "test", "v1.0")

        # Should be read-only by default
        assert snapshot.read_only

        # Cannot modify
        with pytest.raises(RuntimeError, match="read-only.*snapshot"):
            snapshot.add_table("new", df)

    def test_load_snapshot_has_correct_items(self, tmp_path):
        """Test loaded snapshot has correct items."""
        folio = DataFolio(tmp_path / "test")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio.add_table("data1", df1)
        folio.create_snapshot("v1.0")

        folio.add_table("data2", df2)
        folio.create_snapshot("v2.0")

        # Load v1.0 - should only have data1
        v1 = DataFolio.load_snapshot(tmp_path / "test", "v1.0")
        assert "data1" in v1._items
        assert "data2" not in v1._items

        # Load v2.0 - should have both
        v2 = DataFolio.load_snapshot(tmp_path / "test", "v2.0")
        assert "data1" in v2._items
        assert "data2" in v2._items

    def test_load_snapshot_has_correct_metadata(self, tmp_path):
        """Test loaded snapshot has correct metadata."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        folio.metadata["accuracy"] = 0.85
        folio.create_snapshot("v1.0")

        folio.metadata["accuracy"] = 0.90
        folio.create_snapshot("v2.0")

        # Load snapshots
        v1 = DataFolio.load_snapshot(tmp_path / "test", "v1.0")
        v2 = DataFolio.load_snapshot(tmp_path / "test", "v2.0")

        assert v1.metadata["accuracy"] == 0.85
        assert v2.metadata["accuracy"] == 0.90

    def test_load_multiple_snapshots_simultaneously(self, tmp_path):
        """Test loading multiple snapshots at once."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        for i, acc in enumerate([0.85, 0.87, 0.90], 1):
            folio.metadata["accuracy"] = acc
            folio.create_snapshot(f"v{i}.0")

        # Load all simultaneously
        v1 = DataFolio.load_snapshot(tmp_path / "test", "v1.0")
        v2 = DataFolio.load_snapshot(tmp_path / "test", "v2.0")
        v3 = DataFolio.load_snapshot(tmp_path / "test", "v3.0")

        assert v1.metadata["accuracy"] == 0.85
        assert v2.metadata["accuracy"] == 0.87
        assert v3.metadata["accuracy"] == 0.90

    def test_load_nonexistent_snapshot(self, tmp_path):
        """Test loading nonexistent snapshot raises error."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            DataFolio.load_snapshot(tmp_path / "test", "nonexistent")

    def test_get_snapshot_instance_method(self, tmp_path):
        """Test get_snapshot() instance method."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.metadata["accuracy"] = 0.89
        folio.create_snapshot("v1.0")

        # Modify current state
        folio.metadata["accuracy"] = 0.91

        # Get snapshot using instance method
        snapshot = folio.get_snapshot("v1.0")

        assert snapshot.in_snapshot_mode
        assert snapshot.loaded_snapshot == "v1.0"
        assert snapshot.metadata["accuracy"] == 0.89  # Snapshot value
        assert folio.metadata["accuracy"] == 0.91  # Current value unchanged

    def test_get_snapshot_equivalent_to_load_snapshot(self, tmp_path):
        """Test get_snapshot() is equivalent to load_snapshot()."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.metadata["accuracy"] = 0.89
        folio.create_snapshot("v1.0")

        # Get snapshot both ways
        snapshot1 = folio.get_snapshot("v1.0")
        snapshot2 = DataFolio.load_snapshot(tmp_path / "test", "v1.0")

        # Should be equivalent
        assert snapshot1.in_snapshot_mode == snapshot2.in_snapshot_mode
        assert snapshot1.loaded_snapshot == snapshot2.loaded_snapshot
        assert snapshot1.metadata["accuracy"] == snapshot2.metadata["accuracy"]
        assert list(snapshot1._items.keys()) == list(snapshot2._items.keys())

    def test_repr_shows_snapshot_name(self, tmp_path):
        """Test __repr__ shows snapshot name."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        snapshot = DataFolio.load_snapshot(tmp_path / "test", "v1.0")
        repr_str = repr(snapshot)

        assert "[READ-ONLY]" in repr_str
        assert "[snapshot: v1.0]" in repr_str

    def test_describe_shows_snapshot_info(self, tmp_path):
        """Test describe() shows snapshot information."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot(
            "v1.0", description="Test snapshot", tags=["test", "baseline"]
        )

        snapshot = DataFolio.load_snapshot(tmp_path / "test", "v1.0")
        desc = snapshot.describe(return_string=True)

        assert "Snapshot: v1.0" in desc
        assert "Test snapshot" in desc
        assert "test" in desc
        assert "[READ-ONLY MODE]" in desc


class TestSnapshotWorkflows:
    """Integration tests for common snapshot workflows."""

    def test_paper_submission_workflow(self, tmp_path):
        """Test paper submission workflow."""
        # Create and snapshot for paper
        folio = DataFolio(tmp_path / "research")
        df = pd.DataFrame({"data": [1, 2, 3]})
        folio.add_table("data", df)
        folio.metadata["accuracy"] = 0.89
        folio.create_snapshot("neurips-2025", description="Paper submission")

        # Continue research
        folio.metadata["accuracy"] = 0.91
        folio.create_snapshot("v2.0")

        # Later: Load paper version
        paper = DataFolio.load_snapshot(tmp_path / "research", "neurips-2025")

        assert paper.metadata["accuracy"] == 0.89
        assert paper.read_only

        # Cannot modify
        with pytest.raises(RuntimeError):
            paper.add_table("new", df)

    def test_ab_testing_workflow(self, tmp_path):
        """Test A/B testing workflow."""
        # Create baseline
        folio = DataFolio(tmp_path / "models")
        df = pd.DataFrame({"data": [1, 2, 3]})
        folio.add_table("data", df)
        folio.metadata["latency"] = 50
        folio.create_snapshot("baseline")

        # Create experimental version
        folio.metadata["latency"] = 45
        folio.create_snapshot("experimental")

        # Deploy both
        baseline = DataFolio.load_snapshot(tmp_path / "models", "baseline")
        experimental = DataFolio.load_snapshot(tmp_path / "models", "experimental")

        # Compare
        assert baseline.metadata["latency"] == 50
        assert experimental.metadata["latency"] == 45
        assert experimental.metadata["latency"] < baseline.metadata["latency"]

    def test_compare_multiple_versions(self, tmp_path):
        """Test comparing multiple snapshot versions."""
        folio = DataFolio(tmp_path / "tuning")
        df = pd.DataFrame({"data": [1, 2, 3]})
        folio.add_table("data", df)

        # Create multiple versions
        for lr, acc in [(0.001, 0.85), (0.01, 0.90), (0.1, 0.87)]:
            folio.metadata["lr"] = lr
            folio.metadata["accuracy"] = acc
            folio.create_snapshot(f"lr{lr}")

        # Load and compare
        v1 = DataFolio.load_snapshot(tmp_path / "tuning", "lr0.001")
        v2 = DataFolio.load_snapshot(tmp_path / "tuning", "lr0.01")
        v3 = DataFolio.load_snapshot(tmp_path / "tuning", "lr0.1")

        results = [
            (v1.metadata["lr"], v1.metadata["accuracy"]),
            (v2.metadata["lr"], v2.metadata["accuracy"]),
            (v3.metadata["lr"], v3.metadata["accuracy"]),
        ]

        # Find best
        best_lr, best_acc = max(results, key=lambda x: x[1])
        assert best_lr == 0.01
        assert best_acc == 0.90


class TestErrorMessages:
    """Test that error messages are clear and actionable."""

    def test_read_only_error_message(self, tmp_path):
        """Test read-only error message is clear."""
        folio = DataFolio(tmp_path / "test", read_only=True)
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(RuntimeError) as exc_info:
            folio.add_table("data", df)

        error_msg = str(exc_info.value)
        assert "read-only" in error_msg.lower()
        assert "read_only=True" in error_msg

    def test_snapshot_read_only_error_message(self, tmp_path):
        """Test snapshot read-only error message mentions snapshot."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        snapshot = DataFolio.load_snapshot(tmp_path / "test", "v1.0")

        with pytest.raises(RuntimeError) as exc_info:
            snapshot.add_table("new", df)

        error_msg = str(exc_info.value)
        assert "read-only" in error_msg.lower()
        assert "v1.0" in error_msg

    def test_metadata_read_only_error_message(self, tmp_path):
        """Test metadata read-only error message is clear."""
        folio = DataFolio(tmp_path / "test", read_only=True)

        with pytest.raises(RuntimeError) as exc_info:
            folio.metadata["test"] = 123

        error_msg = str(exc_info.value)
        assert "metadata" in error_msg.lower()
        assert "read-only" in error_msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
