"""Tests for generic data addition (numpy, json, and add_data/get_data)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from datafolio import DataFolio


class TestAddNumpy:
    """Tests for add_numpy method."""

    def test_add_numpy_basic(self, tmp_path):
        """Test basic numpy array addition."""
        folio = DataFolio(tmp_path / "test")
        array = np.array([1, 2, 3, 4, 5])

        folio.add("test_array", array)

        assert "test_array" in folio._items
        assert folio._items["test_array"]["item_type"] == "numpy_array"
        assert folio._items["test_array"]["shape"] == [5]
        assert "int" in folio._items["test_array"]["dtype"]

    def test_add_numpy_multidimensional(self, tmp_path):
        """Test multidimensional numpy array."""
        folio = DataFolio(tmp_path / "test")
        array = np.random.randn(10, 128)

        folio.add("embeddings", array, description="Model embeddings")

        metadata = folio._items["embeddings"]
        assert metadata["item_type"] == "numpy_array"
        assert metadata["shape"] == [10, 128]
        assert metadata["description"] == "Model embeddings"

    def test_add_numpy_with_lineage(self, tmp_path):
        """Test numpy array with lineage."""
        folio = DataFolio(tmp_path / "test")
        array = np.array([0, 1, 0, 1])

        folio.add(
            "predictions",
            array,
            inputs=["test_data"],
        )

        metadata = folio._items["predictions"]
        assert metadata["inputs"] == ["test_data"]

    def test_add_numpy_appears_in_list_contents(self, tmp_path):
        """Test numpy array appears in list_contents."""
        folio = DataFolio(tmp_path / "test")
        array = np.array([1, 2, 3])
        folio.add("test", array)

        contents = folio.list_contents()
        assert "test" in contents["numpy_arrays"]
        assert len(contents["numpy_arrays"]) == 1

    def test_add_numpy_duplicate_name(self, tmp_path):
        """Test error when adding duplicate name."""
        folio = DataFolio(tmp_path / "test")
        array1 = np.array([1, 2, 3])
        array2 = np.array([4, 5, 6])

        folio.add("test", array1)

        with pytest.raises(ValueError, match="already exists"):
            folio.add("test", array2)

    def test_add_list_stores_json(self, tmp_path):
        """A plain list routes to JSON, not the numpy handler."""
        folio = DataFolio(tmp_path / "test")
        folio.add("test", [1, 2, 3])
        assert folio._items["test"]["item_type"] == "json_data"
        assert folio.get("test") == [1, 2, 3]


class TestGetNumpy:
    """Tests for get_numpy method."""

    def test_get_numpy_from_disk(self, tmp_path):
        """Test getting numpy array (read from disk)."""
        folio = DataFolio(tmp_path / "test")
        original_array = np.array([1.5, 2.5, 3.5])
        folio.add("test", original_array)

        retrieved = folio.get("test")

        np.testing.assert_array_equal(retrieved, original_array)

    def test_get_numpy_multidimensional(self, tmp_path):
        """Test getting multidimensional array."""
        folio = DataFolio(tmp_path / "test")
        original = np.random.randn(5, 10, 3)
        folio.add("test", original)

        retrieved = folio.get("test")

        assert retrieved.shape == (5, 10, 3)
        np.testing.assert_array_equal(retrieved, original)

    def test_get_numpy_not_found(self, tmp_path):
        """Test error when array doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get("nonexistent")

    def test_get_returns_stored_type(self, tmp_path):
        """get() returns whatever type the item actually is."""
        folio = DataFolio(tmp_path / "test")
        folio.add("config", {"lr": 0.01})
        assert folio.get("config") == {"lr": 0.01}


class TestAddJson:
    """Tests for add_json method."""

    def test_add_json_dict(self, tmp_path):
        """Test adding dict as JSON."""
        folio = DataFolio(tmp_path / "test")
        config = {"learning_rate": 0.01, "batch_size": 32}

        folio.add("config", config, description="Model config")

        assert "config" in folio._items
        assert folio._items["config"]["item_type"] == "json_data"
        assert folio._items["config"]["data_type"] == "dict"
        assert folio._items["config"]["description"] == "Model config"

    def test_add_json_list(self, tmp_path):
        """Test adding list as JSON."""
        folio = DataFolio(tmp_path / "test")
        classes = ["cat", "dog", "bird"]

        folio.add("classes", classes)

        metadata = folio._items["classes"]
        assert metadata["item_type"] == "json_data"
        assert metadata["data_type"] == "list"

    def test_add_json_scalar(self, tmp_path):
        """Test adding scalar as JSON."""
        folio = DataFolio(tmp_path / "test")

        folio.add("accuracy", 0.95)
        folio.add("count", 100)
        folio.add("name", "experiment1")

        assert folio._items["accuracy"]["data_type"] == "float"
        assert folio._items["count"]["data_type"] == "int"
        assert folio._items["name"]["data_type"] == "str"

    def test_add_json_with_lineage(self, tmp_path):
        """Test JSON data with lineage."""
        folio = DataFolio(tmp_path / "test")
        metrics = {"accuracy": 0.95, "f1": 0.92}

        folio.add(
            "metrics",
            metrics,
            inputs=["test_data"],
        )

        metadata = folio._items["metrics"]
        assert metadata["inputs"] == ["test_data"]

    def test_add_json_appears_in_list_contents(self, tmp_path):
        """Test JSON data appears in list_contents."""
        folio = DataFolio(tmp_path / "test")
        folio.add("config", {"lr": 0.01})

        contents = folio.list_contents()
        assert "config" in contents["json_data"]
        assert len(contents["json_data"]) == 1

    def test_add_json_duplicate_name(self, tmp_path):
        """Test error when adding duplicate name."""
        folio = DataFolio(tmp_path / "test")
        folio.add("config", {"lr": 0.01})

        with pytest.raises(ValueError, match="already exists"):
            folio.add("config", {"lr": 0.02})

    def test_add_numpy_routes_to_numpy_handler(self, tmp_path):
        """numpy arrays route to the numpy handler, never JSON."""
        folio = DataFolio(tmp_path / "test")
        folio.add("test", np.array([1, 2, 3]))
        assert folio._items["test"]["item_type"] == "numpy_array"


class TestGetJson:
    """Tests for get_json method."""

    def test_get_json_dict(self, tmp_path):
        """Test getting dict from JSON."""
        folio = DataFolio(tmp_path / "test")
        original = {"learning_rate": 0.01, "batch_size": 32}
        folio.add("config", original)

        retrieved = folio.get("config")

        assert retrieved == original

    def test_get_json_list(self, tmp_path):
        """Test getting list from JSON."""
        folio = DataFolio(tmp_path / "test")
        original = ["cat", "dog", "bird"]
        folio.add("classes", original)

        retrieved = folio.get("classes")

        assert retrieved == original

    def test_get_json_scalar(self, tmp_path):
        """Test getting scalar from JSON."""
        folio = DataFolio(tmp_path / "test")
        folio.add("accuracy", 0.95)

        retrieved = folio.get("accuracy")

        assert retrieved == 0.95

    def test_get_json_not_found(self, tmp_path):
        """Test error when JSON data doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get("nonexistent")

    def test_get_numpy_item_returns_array(self, tmp_path):
        """get() on a numpy item returns the array."""
        folio = DataFolio(tmp_path / "test")
        folio.add("embeddings", np.array([1, 2, 3]))
        assert list(folio.get("embeddings")) == [1, 2, 3]


class TestAddData:
    """Tests for generic add_data dispatcher."""

    def test_add_data_dataframe(self, tmp_path):
        """Test add_data with DataFrame."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        folio.add("results", df, description="Test results")

        assert "results" in folio._items
        assert folio._items["results"]["item_type"] == "included_table"
        assert folio._items["results"]["description"] == "Test results"

    def test_add_data_numpy(self, tmp_path):
        """Test add_data with numpy array."""
        folio = DataFolio(tmp_path / "test")
        array = np.array([1, 2, 3, 4, 5])

        folio.add("embeddings", array, description="Model embeddings")

        assert "embeddings" in folio._items
        assert folio._items["embeddings"]["item_type"] == "numpy_array"

    def test_add_data_dict(self, tmp_path):
        """Test add_data with dict."""
        folio = DataFolio(tmp_path / "test")
        config = {"lr": 0.01, "batch_size": 32}

        folio.add("config", config)

        assert "config" in folio._items
        assert folio._items["config"]["item_type"] == "json_data"

    def test_add_data_list(self, tmp_path):
        """Test add_data with list."""
        folio = DataFolio(tmp_path / "test")
        classes = ["cat", "dog", "bird"]

        folio.add("classes", classes)

        assert "classes" in folio._items
        assert folio._items["classes"]["item_type"] == "json_data"

    def test_add_data_scalar(self, tmp_path):
        """Test add_data with scalar values."""
        folio = DataFolio(tmp_path / "test")

        folio.add("accuracy", 0.95)
        folio.add("epoch", 100)
        folio.add("model_name", "resnet50")

        assert folio._items["accuracy"]["item_type"] == "json_data"
        assert folio._items["epoch"]["item_type"] == "json_data"
        assert folio._items["model_name"]["item_type"] == "json_data"

    def test_reference_table_is_the_reference_verb(self, tmp_path):
        """External references use reference_table(), not add()."""
        folio = DataFolio(tmp_path / "test")

        folio.reference_table("raw", path="s3://bucket/data.parquet")

        assert "raw" in folio._items
        assert folio._items["raw"]["item_type"] == "referenced_table"

    def test_add_requires_object(self, tmp_path):
        """add() requires the object positionally; None is a valid JSON value."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(TypeError):
            folio.add("test")

        folio.add("nothing", None)
        assert folio.get("nothing") is None

    def test_add_rejects_reference_kwarg(self, tmp_path):
        """The old add_data(reference=...) form is gone — clear error."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(TypeError):
            folio.add("test", [1, 2, 3], reference="s3://bucket/file.parquet")

    def test_add_data_unsupported_type(self, tmp_path):
        """Test error with unsupported data type."""
        folio = DataFolio(tmp_path / "test")

        class CustomClass:
            pass

        with pytest.raises(TypeError, match="Unsupported data type"):
            folio.add("test", CustomClass())


class TestGetData:
    """Tests for generic get_data dispatcher."""

    def test_get_data_table(self, tmp_path):
        """Test get_data with table."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("results", df)

        retrieved = folio.get("results")

        assert isinstance(retrieved, pd.DataFrame)
        assert len(retrieved) == 3

    def test_get_data_numpy(self, tmp_path):
        """Test get_data with numpy array."""
        folio = DataFolio(tmp_path / "test")
        array = np.array([1, 2, 3, 4, 5])
        folio.add("embeddings", array)

        retrieved = folio.get("embeddings")

        assert isinstance(retrieved, np.ndarray)
        np.testing.assert_array_equal(retrieved, array)

    def test_get_data_json(self, tmp_path):
        """Test get_data with JSON data."""
        folio = DataFolio(tmp_path / "test")
        config = {"lr": 0.01}
        folio.add("config", config)

        retrieved = folio.get("config")

        assert retrieved == config

    def test_get_data_not_found(self, tmp_path):
        """Test error when item doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get("nonexistent")

    def test_get_data_not_data_item(self, tmp_path):
        """Test error when item is not a data item."""
        # Create a test artifact file
        artifact_file = tmp_path / "test.txt"
        artifact_file.write_text("test content")

        folio = DataFolio(tmp_path / "test")
        folio.add_file(artifact_file, name="artifact")

        # Files come back as the stored payload path
        path = folio.get("artifact")
        assert path.endswith(".txt")


class TestDataIntegration:
    """Integration tests for data methods."""

    def test_mixed_data_types(self, tmp_path):
        """Test using multiple data types together."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")

        # Add different types of data
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add("table", df)

        array = np.array([1, 2, 3, 4, 5])
        folio.add("array", array)

        config = {"lr": 0.01, "batch_size": 32}
        folio.add("config", config)

        # Verify all are tracked
        contents = folio.list_contents()
        assert "table" in contents["included_tables"]
        assert "array" in contents["numpy_arrays"]
        assert "config" in contents["json_data"]

    def test_method_chaining_with_data(self, tmp_path):
        """Test method chaining with data methods."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")

        folio.add("table", pd.DataFrame({"a": [1, 2, 3]})).add(
            "array", np.array([1, 2, 3])
        ).add("config", {"lr": 0.01})

        contents = folio.list_contents()
        assert len(contents["included_tables"]) == 1
        assert len(contents["numpy_arrays"]) == 1
        assert len(contents["json_data"]) == 1

    def test_reopen_bundle_with_new_data_types(self, tmp_path):
        """Test reopening a bundle with numpy and json data."""
        # Create bundle with data
        folio1 = DataFolio(tmp_path / "test")
        folio1.add("embeddings", np.array([1, 2, 3]))
        folio1.add("config", {"lr": 0.01})

        # Reopen bundle
        folio2 = DataFolio(tmp_path / "test")

        # Verify data is loaded correctly
        assert "embeddings" in folio2._items
        assert "config" in folio2._items
        assert folio2._items["embeddings"]["item_type"] == "numpy_array"
        assert folio2._items["config"]["item_type"] == "json_data"

        # Verify data can be retrieved
        embeddings = folio2.get("embeddings")
        config = folio2.get("config")
        np.testing.assert_array_equal(embeddings, np.array([1, 2, 3]))
        assert config == {"lr": 0.01}


class TestSubdirectoryNames:
    """Tests for using path-like names (e.g. 'group/item') to organize data."""

    def test_add_table_with_subdir_name(self, tmp_path):
        """Test that a DataFrame can be added with a path-like name."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"x": [1, 2, 3]})
        folio.add("examples/data", df)

        assert "examples/data" in folio._items
        # Payload filename is versioned but keeps the subdir + extension.
        filename = folio._items["examples/data"]["filename"]
        assert filename.startswith("examples/data--r") and filename.endswith(".parquet")
        # Verify file exists on disk
        expected_path = tmp_path / "test" / "tables" / filename
        assert expected_path.exists()

    def test_roundtrip_table_with_subdir_name(self, tmp_path):
        """Test that a DataFrame added with a path-like name can be retrieved."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        folio.add("group/results", df)

        result = folio.get("group/results")
        pd.testing.assert_frame_equal(result, df)

    def test_roundtrip_numpy_with_subdir_name(self, tmp_path):
        """Test that a numpy array added with a path-like name can be retrieved."""
        folio = DataFolio(tmp_path / "test")
        arr = np.array([1.0, 2.0, 3.0])
        folio.add("run1/embeddings", arr)

        result = folio.get("run1/embeddings")
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_json_with_subdir_name(self, tmp_path):
        """Test that JSON data added with a path-like name can be retrieved."""
        folio = DataFolio(tmp_path / "test")
        config = {"lr": 0.01, "epochs": 10}
        folio.add("run1/config", config)

        result = folio.get("run1/config")
        assert result == config

    def test_add_data_generic_with_subdir_name(self, tmp_path):
        """Test add_data / get_data round-trip with path-like names."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"v": [10, 20]})
        folio.add("experiment/output", df)

        result = folio.get("experiment/output")
        pd.testing.assert_frame_equal(result, df)

    def test_persist_and_reload_with_subdir_names(self, tmp_path):
        """Test that bundles with path-like names survive save/reload."""
        import pandas as pd

        folio1 = DataFolio(tmp_path / "test")
        folio1.add("phase1/results", pd.DataFrame({"n": [1, 2]}))
        folio1.add("phase1/weights", np.array([0.5, 0.5]))

        folio2 = DataFolio(tmp_path / "test")
        assert "phase1/results" in folio2._items
        assert "phase1/weights" in folio2._items

        pd.testing.assert_frame_equal(
            folio2.get("phase1/results"), pd.DataFrame({"n": [1, 2]})
        )

    def test_delete_with_subdir_name(self, tmp_path):
        """Test that items with path-like names can be deleted."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        folio.add("group/data", pd.DataFrame({"x": [1]}))

        assert "group/data" in folio._items
        folio.delete("group/data")
        assert "group/data" not in folio._items

    def test_multiple_subdir_groups(self, tmp_path):
        """Test that multiple groups can coexist in the same bundle."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        folio.add("train/data", pd.DataFrame({"x": [1, 2, 3]}))
        folio.add("val/data", pd.DataFrame({"x": [4, 5]}))
        folio.add("test/data", pd.DataFrame({"x": [6]}))

        assert len(folio._items) == 3
        assert folio.get("train/data").shape == (3, 1)
        assert folio.get("val/data").shape == (2, 1)
        assert folio.get("test/data").shape == (1, 1)
