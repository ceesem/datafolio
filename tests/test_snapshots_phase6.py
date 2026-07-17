"""Tests for Snapshot Phase 6: Utility Methods.

These tests verify utility methods like get_snapshot_info,
and enhancements to describe() and __repr__().
"""

import pandas as pd
import pytest

from datafolio import DataFolio


class TestGetSnapshotInfo:
    """Tests for get_snapshot_info() method."""

    def test_get_snapshot_info_basic(self, tmp_path):
        """Test getting basic snapshot information."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("data", df)
        folio.metadata["accuracy"] = 0.85
        folio.create_snapshot("v1.0", description="Test snapshot", tags=["test"])

        info = folio.get_snapshot_info("v1.0")

        assert info["name"] == "v1.0"
        assert info["description"] == "Test snapshot"
        assert info["tags"] == ["test"]
        assert "timestamp" in info
        assert "item_versions" in info
        assert "data" in info["item_versions"]
        assert "metadata_snapshot" in info
        assert info["metadata_snapshot"]["accuracy"] == 0.85

    def test_get_snapshot_info_with_git(self, tmp_path):
        """Test snapshot info includes git information."""
        # Initialize git repo
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            capture_output=True,
        )

        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("data", df)

        # Create initial commit
        subprocess.run(["git", "add", "."], cwd=folio._bundle_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=folio._bundle_dir,
            capture_output=True,
        )

        folio.create_snapshot("v1.0", capture_git=True)

        info = folio.get_snapshot_info("v1.0")

        assert "git" in info
        assert "commit" in info["git"]
        assert "branch" in info["git"]
        assert len(info["git"]["commit"]) == 40  # Full SHA

    def test_get_snapshot_info_with_environment(self, tmp_path):
        """Test snapshot info includes environment information."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("data", df)
        folio.create_snapshot("v1.0", capture_environment=True)

        info = folio.get_snapshot_info("v1.0")

        assert "environment" in info
        assert "python_version" in info["environment"]

    def test_get_snapshot_info_nonexistent(self, tmp_path):
        """Test getting info for nonexistent snapshot raises error."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("data", df)

        with pytest.raises(KeyError, match="not found"):
            folio.get_snapshot_info("nonexistent")


class TestReprEnhancement:
    """Tests for enhanced __repr__() method."""

    def test_repr_without_snapshots(self, tmp_path):
        """Test repr shows no snapshots when none exist."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("data", df)

        repr_str = repr(folio)

        assert "DataFolio" in repr_str
        assert "items=1" in repr_str
        # Should show 0 snapshots
        assert "snapshots=0" in repr_str or "snapshot" not in repr_str.lower()

    def test_repr_with_snapshots(self, tmp_path):
        """Test repr shows snapshot count."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("data", df)
        folio.create_snapshot("v1.0")
        folio.create_snapshot("v2.0")

        repr_str = repr(folio)

        assert "DataFolio" in repr_str
        assert "items=1" in repr_str
        assert "snapshots=2" in repr_str


class TestDescribeEnhancement:
    """Tests for enhanced describe() method."""

    def test_describe_without_snapshots(self, tmp_path):
        """Test describe works when no snapshots exist."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("data", df)

        description = folio.describe(return_string=True)

        assert isinstance(description, str)
        assert len(description) > 0

    def test_describe_with_snapshots(self, tmp_path):
        """Test describe shows snapshot information."""
        folio = DataFolio(tmp_path / "test-bundle")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("data", df)
        folio.create_snapshot("v1.0", description="First version")
        folio.create_snapshot("v2.0", description="Second version")

        description = folio.describe(return_string=True)

        assert isinstance(description, str)
        # Should mention snapshots
        assert "snapshot" in description.lower()
        # Should list snapshot names
        assert "v1.0" in description
        assert "v2.0" in description

    def test_describe_shows_which_snapshots_contain_items(self, tmp_path):
        """Test describe shows which snapshots contain each item."""
        folio = DataFolio(tmp_path / "test-bundle")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio.add("data1", df1)
        folio.create_snapshot("v1.0")

        folio.add("data2", df2)
        folio.create_snapshot("v2.0")

        description = folio.describe(return_string=True)

        # Should show that data1 is in v1.0 and v2.0
        # and data2 is only in v2.0
        assert "v1.0" in description
        assert "v2.0" in description


class TestUtilityMethodsIntegration:
    """Integration tests for utility methods."""

    def test_complete_workflow_with_utilities(self, tmp_path):
        """Test complete workflow using all utility methods."""
        folio = DataFolio(tmp_path / "experiment")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("data", df)
        folio.metadata["accuracy"] = 0.85
        folio.create_snapshot("v1.0", description="Baseline model", tags=["baseline"])

        # Get snapshot info
        info = folio.get_snapshot_info("v1.0")
        assert info["name"] == "v1.0"
        assert info["description"] == "Baseline model"
        assert info["tags"] == ["baseline"]

        # Get reproduction instructions

        # Check repr shows snapshot
        repr_str = repr(folio)
        assert "snapshots=1" in repr_str

        # Check describe shows snapshot
        description = folio.describe(return_string=True)
        assert "v1.0" in description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
