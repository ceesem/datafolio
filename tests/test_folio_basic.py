"""Tests for basic DataFolio functionality."""

from pathlib import Path

import pytest

from datafolio import DataFolio


class TestDataFolioInit:
    """Tests for DataFolio initialization."""

    def test_init_empty(self, tmp_path):
        """Test creating empty DataFolio."""
        folio = DataFolio(tmp_path / "test")
        # Should have auto-populated timestamps
        assert "created_at" in folio.metadata
        assert "updated_at" in folio.metadata
        assert len(folio._items) == 0

    def test_init_with_metadata(self, tmp_path):
        """Test creating DataFolio with metadata."""
        metadata = {
            "experiment": "test_001",
            "timestamp": "2024-01-15",
            "parameters": {"lr": 0.01},
        }
        folio = DataFolio(tmp_path / "test", metadata=metadata)
        # Should have user metadata plus auto-populated timestamps
        assert folio.metadata["experiment"] == "test_001"
        assert folio.metadata["timestamp"] == "2024-01-15"
        assert folio.metadata["parameters"] == {"lr": 0.01}
        assert "created_at" in folio.metadata
        assert "updated_at" in folio.metadata

    def test_metadata_is_mutable(self, tmp_path):
        """Test that metadata can be modified after initialization."""
        folio = DataFolio(tmp_path / "test", metadata={"key": "value"})
        folio.metadata["new_key"] = "new_value"
        assert folio.metadata["new_key"] == "new_value"

    def test_exact_bundle_name_without_random_suffix(self, tmp_path):
        """Test creating bundle with exact name (no random suffix)."""
        folio = DataFolio(tmp_path / "my-experiment", use_random_suffix=False)

        # Bundle directory should be exactly the prefix
        assert folio._bundle_dir.endswith("my-experiment")
        assert "my-experiment" in str(folio._bundle_dir)

    def test_random_suffix_when_enabled(self, tmp_path):
        """Test creating bundle with random suffix."""
        folio = DataFolio(tmp_path / "my-experiment", use_random_suffix=True)

        # Bundle directory should have prefix plus random suffix
        assert "my-experiment" in str(folio._bundle_dir)
        # Should have more than just the prefix
        bundle_name = Path(folio._bundle_dir).name
        assert bundle_name != "my-experiment"
        assert bundle_name.startswith("my-experiment-")

    def test_reopen_existing_bundle_without_random_suffix(self, tmp_path):
        """Test that reopening an existing bundle loads it correctly."""
        # Create first bundle
        folio1 = DataFolio(tmp_path / "test-experiment", use_random_suffix=False)
        original_created_at = folio1.metadata["created_at"]

        # Opening the same path loads the existing bundle
        folio2 = DataFolio(tmp_path / "test-experiment", use_random_suffix=False)

        # Should load the same bundle, not create a new one
        assert folio2._bundle_dir == folio1._bundle_dir
        assert folio2.metadata["created_at"] == original_created_at

    def test_collision_retry_with_random_suffix(self, tmp_path):
        """Test that collision is handled with retry when random suffix enabled."""
        # Create first bundle
        folio1 = DataFolio(tmp_path / "test", use_random_suffix=True)

        # Create another with same prefix - should get different random suffix
        folio2 = DataFolio(tmp_path / "test", use_random_suffix=True)

        # Should have different bundle directories
        assert folio1._bundle_dir != folio2._bundle_dir


class TestListContents:
    """Tests for list_contents method."""

    def test_empty_contents(self, tmp_path):
        """Test listing contents of empty DataFolio."""
        folio = DataFolio(tmp_path / "test")
        contents = folio.list_contents()

        assert contents == {
            "referenced_tables": [],
            "included_tables": [],
            "numpy_arrays": [],
            "json_data": [],
            "models": [],
            "pytorch_models": [],
            "artifacts": [],
        }

    def test_list_contents_structure(self, tmp_path):
        """Test that list_contents returns correct structure."""
        folio = DataFolio(tmp_path / "test")
        contents = folio.list_contents()

        # Verify all expected keys are present
        expected_keys = {
            "referenced_tables",
            "included_tables",
            "numpy_arrays",
            "json_data",
            "models",
            "pytorch_models",
            "artifacts",
        }
        assert set(contents.keys()) == expected_keys

        # Verify all values are lists
        for value in contents.values():
            assert isinstance(value, list)


class TestRepr:
    """Tests for string representation."""

    def test_repr_empty(self, tmp_path):
        """Test repr of empty DataFolio."""
        folio = DataFolio(tmp_path / "test")
        repr_str = repr(folio)

        assert "DataFolio" in repr_str
        assert "items=0" in repr_str

    def test_repr_with_items(self, tmp_path):
        """Test repr with items."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test", metadata={"key1": "val1"})
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("test", df)
        repr_str = repr(folio)

        assert "items=1" in repr_str


class TestLoad:
    """Tests for loading existing bundles."""

    def test_load_nonexistent_directory(self):
        """Test that loading a nonexistent directory creates new bundle."""
        # With the new API, passing a nonexistent path creates a new bundle
        # This is expected behavior
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()
        try:
            folio = DataFolio(Path(temp_dir) / "test")
            assert folio._bundle_dir is not None
        finally:
            shutil.rmtree(temp_dir)
