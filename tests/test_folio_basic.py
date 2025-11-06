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
        folio = DataFolio(tmp_path / "my-experiment", random_suffix=False)

        # Bundle directory should be exactly the prefix
        assert folio._bundle_dir.endswith("my-experiment")
        assert "my-experiment" in str(folio._bundle_dir)

    def test_random_suffix_when_enabled(self, tmp_path):
        """Test creating bundle with random suffix."""
        folio = DataFolio(tmp_path / "my-experiment", random_suffix=True)

        # Bundle directory should have prefix plus random suffix
        assert "my-experiment" in str(folio._bundle_dir)
        # Should have more than just the prefix
        bundle_name = Path(folio._bundle_dir).name
        assert bundle_name != "my-experiment"
        assert bundle_name.startswith("my-experiment-")

    def test_reopen_existing_bundle_without_random_suffix(self, tmp_path):
        """Test that reopening an existing bundle loads it correctly."""
        # Create first bundle
        folio1 = DataFolio(tmp_path / "test-experiment", random_suffix=False)
        original_created_at = folio1.metadata["created_at"]

        # Opening the same path loads the existing bundle
        folio2 = DataFolio(tmp_path / "test-experiment", random_suffix=False)

        # Should load the same bundle, not create a new one
        assert folio2._bundle_dir == folio1._bundle_dir
        assert folio2.metadata["created_at"] == original_created_at

    def test_collision_retry_with_random_suffix(self, tmp_path):
        """Test that collision is handled with retry when random suffix enabled."""
        # Create first bundle
        folio1 = DataFolio(tmp_path / "test", random_suffix=True)

        # Create another with same prefix - should get different random suffix
        folio2 = DataFolio(tmp_path / "test", random_suffix=True)

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
            "timestamps": [],
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
            "timestamps",
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


class TestPathProperty:
    """Tests for the path property."""

    def test_path_returns_absolute_local_path(self, tmp_path):
        """Test that path property returns absolute path for local directories."""
        folio = DataFolio(tmp_path / "test")

        # Path should be absolute
        path = folio.path
        assert Path(path).is_absolute()
        assert "test" in path

    def test_path_with_relative_input(self, tmp_path):
        """Test that path property makes relative paths absolute."""
        import os

        # Change to tmp_path and create with relative path
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            folio = DataFolio("relative-test")

            # Path should be absolute even though we passed relative
            path = folio.path
            assert Path(path).is_absolute()
            assert "relative-test" in path
        finally:
            os.chdir(original_cwd)

    def test_path_with_absolute_input(self, tmp_path):
        """Test that path property works with absolute input paths."""
        abs_path = tmp_path / "absolute-test"
        folio = DataFolio(abs_path)

        # Path should remain absolute
        path = folio.path
        assert Path(path).is_absolute()
        assert path == str(abs_path.resolve())

    def test_path_consistent_across_reopen(self, tmp_path):
        """Test that path property returns same value after reopening bundle."""
        # Create bundle
        folio1 = DataFolio(tmp_path / "test")
        path1 = folio1.path

        # Reopen bundle
        folio2 = DataFolio(tmp_path / "test")
        path2 = folio2.path

        # Paths should be identical
        assert path1 == path2


class TestVersionAndReadme:
    """Tests for version info and README creation."""

    def test_version_info_in_metadata(self, tmp_path):
        """Test that version info is added to metadata.json."""
        folio = DataFolio(tmp_path / "test")

        # Check that _datafolio key exists
        assert "_datafolio" in folio.metadata
        assert "version" in folio.metadata["_datafolio"]
        assert "created_by" in folio.metadata["_datafolio"]
        assert folio.metadata["_datafolio"]["created_by"] == "datafolio"

    def test_readme_created(self, tmp_path):
        """Test that README.md is created."""
        folio = DataFolio(tmp_path / "test")

        # Check that README exists
        readme_path = tmp_path / "test" / "README.md"
        assert readme_path.exists()

    def test_readme_contains_version(self, tmp_path):
        """Test that README contains version information."""
        folio = DataFolio(tmp_path / "test")

        readme_path = tmp_path / "test" / "README.md"
        readme_content = readme_path.read_text()

        # Check for key content
        assert "DataFolio Bundle" in readme_content
        assert "datafolio" in readme_content
        assert "version" in readme_content
        assert "## Structure" in readme_content
        assert "## Usage" in readme_content

    def test_readme_contains_usage_example(self, tmp_path):
        """Test that README contains usage example with correct path."""
        folio = DataFolio(tmp_path / "test")

        readme_path = tmp_path / "test" / "README.md"
        readme_content = readme_path.read_text()

        # Check that the path is included in usage example
        assert "from datafolio import DataFolio" in readme_content
        assert "folio.describe()" in readme_content
        assert "folio.get_table" in readme_content

    def test_existing_datafolio_metadata_not_overwritten(self, tmp_path):
        """Test that existing _datafolio metadata is not overwritten."""
        custom_version = {"version": "custom", "created_by": "test"}
        folio = DataFolio(tmp_path / "test", metadata={"_datafolio": custom_version})

        # Should preserve custom version info
        assert folio.metadata["_datafolio"] == custom_version


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


class TestDelete:
    """Tests for deleting items from DataFolio."""

    def test_delete_single_table(self, tmp_path):
        """Test deleting a single table."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data1", df)
        folio.add_table("data2", df)

        # Delete one table
        folio.delete("data1")

        # Verify it's removed from manifest
        assert "data1" not in folio._items
        assert "data2" in folio._items

        # Verify file is deleted
        table_path = tmp_path / "test" / "tables" / "data1.parquet"
        assert not table_path.exists()

    def test_delete_multiple_items(self, tmp_path):
        """Test deleting multiple items at once."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data1", df)
        folio.add_table("data2", df)
        folio.add_table("data3", df)

        # Delete multiple items
        folio.delete(["data1", "data2"])

        # Verify both are removed
        assert "data1" not in folio._items
        assert "data2" not in folio._items
        assert "data3" in folio._items

        # Verify files are deleted
        assert not (tmp_path / "test" / "tables" / "data1.parquet").exists()
        assert not (tmp_path / "test" / "tables" / "data2.parquet").exists()
        assert (tmp_path / "test" / "tables" / "data3.parquet").exists()

    def test_delete_model(self, tmp_path):
        """Test deleting a model."""

        class DummyModel:
            def __init__(self):
                self.param = 42

        folio = DataFolio(tmp_path / "test")
        model = DummyModel()
        folio.add_model("clf", model)

        # Delete model
        folio.delete("clf")

        # Verify it's removed
        assert "clf" not in folio._items
        assert not (tmp_path / "test" / "models" / "clf.joblib").exists()

    def test_delete_numpy_array(self, tmp_path):
        """Test deleting a numpy array."""
        import numpy as np

        folio = DataFolio(tmp_path / "test")
        arr = np.array([1, 2, 3])
        folio.add_numpy("embeddings", arr)

        # Delete array
        folio.delete("embeddings")

        # Verify it's removed
        assert "embeddings" not in folio._items
        assert not (tmp_path / "test" / "artifacts" / "embeddings.npy").exists()

    def test_delete_json_data(self, tmp_path):
        """Test deleting JSON data."""
        folio = DataFolio(tmp_path / "test")
        folio.add_json("config", {"lr": 0.01, "batch_size": 32})

        # Delete JSON
        folio.delete("config")

        # Verify it's removed
        assert "config" not in folio._items
        assert not (tmp_path / "test" / "artifacts" / "config.json").exists()

    def test_delete_artifact(self, tmp_path):
        """Test deleting an artifact file."""
        # Create artifact file
        artifact_file = tmp_path / "plot.png"
        artifact_file.write_text("fake image")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("plot", artifact_file)

        # Delete artifact
        folio.delete("plot")

        # Verify it's removed
        assert "plot" not in folio._items
        assert not (tmp_path / "test" / "artifacts" / "plot.png").exists()

    def test_delete_referenced_table(self, tmp_path):
        """Test deleting a referenced table (metadata only, no file)."""
        import pandas as pd

        # Create external parquet file
        df = pd.DataFrame({"a": [1, 2, 3]})
        external_file = tmp_path / "external.parquet"
        df.to_parquet(external_file)

        folio = DataFolio(tmp_path / "test")
        folio.reference_table("external_data", external_file, table_format="parquet")

        # Delete reference
        folio.delete("external_data")

        # Verify reference is removed from manifest
        assert "external_data" not in folio._items

        # Verify external file still exists (shouldn't be deleted)
        assert external_file.exists()

    def test_delete_nonexistent_item_raises_error(self, tmp_path):
        """Test that deleting nonexistent item raises KeyError."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.delete("nonexistent")

    def test_delete_validates_all_before_deleting(self, tmp_path):
        """Test that delete validates all items exist before deleting any."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data1", df)
        folio.add_table("data2", df)

        # Try to delete with one nonexistent item
        with pytest.raises(KeyError, match="nonexistent"):
            folio.delete(["data1", "nonexistent", "data2"])

        # Both items should still exist (transaction-like behavior)
        assert "data1" in folio._items
        assert "data2" in folio._items

    def test_delete_with_dependents_warns(self, tmp_path):
        """Test that deleting item with dependents shows warning."""
        import warnings

        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("raw", df)
        folio.add_table("processed", df, inputs=["raw"])

        # Delete item with dependent - should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            folio.delete("raw")

            # Check warning was issued
            assert len(w) == 1
            assert "used by" in str(w[0].message)
            assert "processed" in str(w[0].message)

        # Item should still be deleted despite warning
        assert "raw" not in folio._items

    def test_delete_without_warning(self, tmp_path):
        """Test deleting with warn_dependents=False."""
        import warnings

        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("raw", df)
        folio.add_table("processed", df, inputs=["raw"])

        # Delete without warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            folio.delete("raw", warn_dependents=False)

            # No warning should be issued
            assert len(w) == 0

        # Item should be deleted
        assert "raw" not in folio._items

    def test_delete_returns_self_for_chaining(self, tmp_path):
        """Test that delete returns self for method chaining."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data1", df)
        folio.add_table("data2", df)

        # Chain delete operations
        result = folio.delete("data1").delete("data2")

        assert result is folio
        assert "data1" not in folio._items
        assert "data2" not in folio._items

    def test_delete_updates_manifest(self, tmp_path):
        """Test that delete saves updated manifest."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Delete and reload
        folio.delete("data")

        # Reload bundle and verify item is gone
        folio2 = DataFolio(tmp_path / "test")
        assert "data" not in folio2._items
