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

        folio.add_numpy("test_array", array)

        assert "test_array" in folio._items
        assert folio._items["test_array"]["item_type"] == "numpy_array"
        assert folio._items["test_array"]["shape"] == [5]
        assert "int" in folio._items["test_array"]["dtype"]

    def test_add_numpy_multidimensional(self, tmp_path):
        """Test multidimensional numpy array."""
        folio = DataFolio(tmp_path / "test")
        array = np.random.randn(10, 128)

        folio.add_numpy("embeddings", array, description="Model embeddings")

        metadata = folio._items["embeddings"]
        assert metadata["item_type"] == "numpy_array"
        assert metadata["shape"] == [10, 128]
        assert metadata["description"] == "Model embeddings"

    def test_add_numpy_with_lineage(self, tmp_path):
        """Test numpy array with lineage."""
        folio = DataFolio(tmp_path / "test")
        array = np.array([0, 1, 0, 1])

        folio.add_numpy(
            "predictions",
            array,
            inputs=["test_data"],
            code="predictions = model.predict(X)",
        )

        metadata = folio._items["predictions"]
        assert metadata["inputs"] == ["test_data"]
        assert metadata["code"] == "predictions = model.predict(X)"

    def test_add_numpy_appears_in_list_contents(self, tmp_path):
        """Test numpy array appears in list_contents."""
        folio = DataFolio(tmp_path / "test")
        array = np.array([1, 2, 3])
        folio.add_numpy("test", array)

        contents = folio.list_contents()
        assert "test" in contents["numpy_arrays"]
        assert len(contents["numpy_arrays"]) == 1

    def test_add_numpy_duplicate_name(self, tmp_path):
        """Test error when adding duplicate name."""
        folio = DataFolio(tmp_path / "test")
        array1 = np.array([1, 2, 3])
        array2 = np.array([4, 5, 6])

        folio.add_numpy("test", array1)

        with pytest.raises(ValueError, match="already exists"):
            folio.add_numpy("test", array2)

    def test_add_numpy_not_array(self, tmp_path):
        """Test error when data is not a numpy array."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(TypeError, match="Expected numpy array"):
            folio.add_numpy("test", [1, 2, 3])


class TestGetNumpy:
    """Tests for get_numpy method."""

    def test_get_numpy_from_disk(self, tmp_path):
        """Test getting numpy array (read from disk)."""
        folio = DataFolio(tmp_path / "test")
        original_array = np.array([1.5, 2.5, 3.5])
        folio.add_numpy("test", original_array)

        retrieved = folio.get_numpy("test")

        np.testing.assert_array_equal(retrieved, original_array)

    def test_get_numpy_multidimensional(self, tmp_path):
        """Test getting multidimensional array."""
        folio = DataFolio(tmp_path / "test")
        original = np.random.randn(5, 10, 3)
        folio.add_numpy("test", original)

        retrieved = folio.get_numpy("test")

        assert retrieved.shape == (5, 10, 3)
        np.testing.assert_array_equal(retrieved, original)

    def test_get_numpy_not_found(self, tmp_path):
        """Test error when array doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_numpy("nonexistent")

    def test_get_numpy_wrong_type(self, tmp_path):
        """Test error when item is not a numpy array."""
        folio = DataFolio(tmp_path / "test")
        folio.add_json("config", {"lr": 0.01})

        with pytest.raises(ValueError, match="not a numpy array"):
            folio.get_numpy("config")


class TestAddJson:
    """Tests for add_json method."""

    def test_add_json_dict(self, tmp_path):
        """Test adding dict as JSON."""
        folio = DataFolio(tmp_path / "test")
        config = {"learning_rate": 0.01, "batch_size": 32}

        folio.add_json("config", config, description="Model config")

        assert "config" in folio._items
        assert folio._items["config"]["item_type"] == "json_data"
        assert folio._items["config"]["data_type"] == "dict"
        assert folio._items["config"]["description"] == "Model config"

    def test_add_json_list(self, tmp_path):
        """Test adding list as JSON."""
        folio = DataFolio(tmp_path / "test")
        classes = ["cat", "dog", "bird"]

        folio.add_json("classes", classes)

        metadata = folio._items["classes"]
        assert metadata["item_type"] == "json_data"
        assert metadata["data_type"] == "list"

    def test_add_json_scalar(self, tmp_path):
        """Test adding scalar as JSON."""
        folio = DataFolio(tmp_path / "test")

        folio.add_json("accuracy", 0.95)
        folio.add_json("count", 100)
        folio.add_json("name", "experiment1")

        assert folio._items["accuracy"]["data_type"] == "float"
        assert folio._items["count"]["data_type"] == "int"
        assert folio._items["name"]["data_type"] == "str"

    def test_add_json_with_lineage(self, tmp_path):
        """Test JSON data with lineage."""
        folio = DataFolio(tmp_path / "test")
        metrics = {"accuracy": 0.95, "f1": 0.92}

        folio.add_json(
            "metrics",
            metrics,
            inputs=["test_data"],
            code="metrics = evaluate(model, X_test)",
        )

        metadata = folio._items["metrics"]
        assert metadata["inputs"] == ["test_data"]
        assert metadata["code"] == "metrics = evaluate(model, X_test)"

    def test_add_json_appears_in_list_contents(self, tmp_path):
        """Test JSON data appears in list_contents."""
        folio = DataFolio(tmp_path / "test")
        folio.add_json("config", {"lr": 0.01})

        contents = folio.list_contents()
        assert "config" in contents["json_data"]
        assert len(contents["json_data"]) == 1

    def test_add_json_duplicate_name(self, tmp_path):
        """Test error when adding duplicate name."""
        folio = DataFolio(tmp_path / "test")
        folio.add_json("config", {"lr": 0.01})

        with pytest.raises(ValueError, match="already exists"):
            folio.add_json("config", {"lr": 0.02})

    def test_add_json_not_serializable(self, tmp_path):
        """Test error when data is not JSON-serializable."""
        folio = DataFolio(tmp_path / "test")

        # numpy arrays are not directly JSON serializable (need to use add_numpy)
        with pytest.raises(TypeError, match="not JSON-serializable"):
            folio.add_json("test", np.array([1, 2, 3]))


class TestGetJson:
    """Tests for get_json method."""

    def test_get_json_dict(self, tmp_path):
        """Test getting dict from JSON."""
        folio = DataFolio(tmp_path / "test")
        original = {"learning_rate": 0.01, "batch_size": 32}
        folio.add_json("config", original)

        retrieved = folio.get_json("config")

        assert retrieved == original

    def test_get_json_list(self, tmp_path):
        """Test getting list from JSON."""
        folio = DataFolio(tmp_path / "test")
        original = ["cat", "dog", "bird"]
        folio.add_json("classes", original)

        retrieved = folio.get_json("classes")

        assert retrieved == original

    def test_get_json_scalar(self, tmp_path):
        """Test getting scalar from JSON."""
        folio = DataFolio(tmp_path / "test")
        folio.add_json("accuracy", 0.95)

        retrieved = folio.get_json("accuracy")

        assert retrieved == 0.95

    def test_get_json_not_found(self, tmp_path):
        """Test error when JSON data doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_json("nonexistent")

    def test_get_json_wrong_type(self, tmp_path):
        """Test error when item is not JSON data."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("embeddings", np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="not JSON data"):
            folio.get_json("embeddings")


class TestAddData:
    """Tests for generic add_data dispatcher."""

    def test_add_data_dataframe(self, tmp_path):
        """Test add_data with DataFrame."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        folio.add_data("results", df, description="Test results")

        assert "results" in folio._items
        assert folio._items["results"]["item_type"] == "included_table"
        assert folio._items["results"]["description"] == "Test results"

    def test_add_data_numpy(self, tmp_path):
        """Test add_data with numpy array."""
        folio = DataFolio(tmp_path / "test")
        array = np.array([1, 2, 3, 4, 5])

        folio.add_data("embeddings", array, description="Model embeddings")

        assert "embeddings" in folio._items
        assert folio._items["embeddings"]["item_type"] == "numpy_array"

    def test_add_data_dict(self, tmp_path):
        """Test add_data with dict."""
        folio = DataFolio(tmp_path / "test")
        config = {"lr": 0.01, "batch_size": 32}

        folio.add_data("config", config)

        assert "config" in folio._items
        assert folio._items["config"]["item_type"] == "json_data"

    def test_add_data_list(self, tmp_path):
        """Test add_data with list."""
        folio = DataFolio(tmp_path / "test")
        classes = ["cat", "dog", "bird"]

        folio.add_data("classes", classes)

        assert "classes" in folio._items
        assert folio._items["classes"]["item_type"] == "json_data"

    def test_add_data_scalar(self, tmp_path):
        """Test add_data with scalar values."""
        folio = DataFolio(tmp_path / "test")

        folio.add_data("accuracy", 0.95)
        folio.add_data("epoch", 100)
        folio.add_data("model_name", "resnet50")

        assert folio._items["accuracy"]["item_type"] == "json_data"
        assert folio._items["epoch"]["item_type"] == "json_data"
        assert folio._items["model_name"]["item_type"] == "json_data"

    def test_add_data_reference(self, tmp_path):
        """Test add_data with reference parameter."""
        folio = DataFolio(tmp_path / "test")

        folio.add_data("raw", reference="s3://bucket/data.parquet")

        assert "raw" in folio._items
        assert folio._items["raw"]["item_type"] == "referenced_table"

    def test_add_data_no_data_or_reference(self, tmp_path):
        """Test error when neither data nor reference provided."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(ValueError, match="Must provide either"):
            folio.add_data("test")

    def test_add_data_both_data_and_reference(self, tmp_path):
        """Test error when both data and reference provided."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(ValueError, match="Cannot provide both"):
            folio.add_data("test", data=[1, 2, 3], reference="s3://bucket/file.parquet")

    def test_add_data_unsupported_type(self, tmp_path):
        """Test error with unsupported data type."""
        folio = DataFolio(tmp_path / "test")

        class CustomClass:
            pass

        with pytest.raises(TypeError, match="Unsupported data type"):
            folio.add_data("test", CustomClass())


class TestGetData:
    """Tests for generic get_data dispatcher."""

    def test_get_data_table(self, tmp_path):
        """Test get_data with table."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df)

        retrieved = folio.get_data("results")

        assert isinstance(retrieved, pd.DataFrame)
        assert len(retrieved) == 3

    def test_get_data_numpy(self, tmp_path):
        """Test get_data with numpy array."""
        folio = DataFolio(tmp_path / "test")
        array = np.array([1, 2, 3, 4, 5])
        folio.add_numpy("embeddings", array)

        retrieved = folio.get_data("embeddings")

        assert isinstance(retrieved, np.ndarray)
        np.testing.assert_array_equal(retrieved, array)

    def test_get_data_json(self, tmp_path):
        """Test get_data with JSON data."""
        folio = DataFolio(tmp_path / "test")
        config = {"lr": 0.01}
        folio.add_json("config", config)

        retrieved = folio.get_data("config")

        assert retrieved == config

    def test_get_data_not_found(self, tmp_path):
        """Test error when item doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_data("nonexistent")

    def test_get_data_not_data_item(self, tmp_path):
        """Test error when item is not a data item."""
        # Create a test artifact file
        artifact_file = tmp_path / "test.txt"
        artifact_file.write_text("test content")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("artifact", artifact_file)

        with pytest.raises(ValueError, match="not a data item"):
            folio.get_data("artifact")


class TestDataIntegration:
    """Integration tests for data methods."""

    def test_mixed_data_types(self, tmp_path):
        """Test using multiple data types together."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")

        # Add different types of data
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_data("table", df)

        array = np.array([1, 2, 3, 4, 5])
        folio.add_data("array", array)

        config = {"lr": 0.01, "batch_size": 32}
        folio.add_data("config", config)

        # Verify all are tracked
        contents = folio.list_contents()
        assert "table" in contents["included_tables"]
        assert "array" in contents["numpy_arrays"]
        assert "config" in contents["json_data"]

    def test_method_chaining_with_data(self, tmp_path):
        """Test method chaining with data methods."""
        import pandas as pd

        folio = DataFolio(tmp_path / "test")

        folio.add_data("table", pd.DataFrame({"a": [1, 2, 3]})).add_data(
            "array", np.array([1, 2, 3])
        ).add_data("config", {"lr": 0.01})

        contents = folio.list_contents()
        assert len(contents["included_tables"]) == 1
        assert len(contents["numpy_arrays"]) == 1
        assert len(contents["json_data"]) == 1

    def test_reopen_bundle_with_new_data_types(self, tmp_path):
        """Test reopening a bundle with numpy and json data."""
        # Create bundle with data
        folio1 = DataFolio(tmp_path / "test")
        folio1.add_numpy("embeddings", np.array([1, 2, 3]))
        folio1.add_json("config", {"lr": 0.01})

        # Reopen bundle
        folio2 = DataFolio(tmp_path / "test")

        # Verify data is loaded correctly
        assert "embeddings" in folio2._items
        assert "config" in folio2._items
        assert folio2._items["embeddings"]["item_type"] == "numpy_array"
        assert folio2._items["config"]["item_type"] == "json_data"

        # Verify data can be retrieved
        embeddings = folio2.get_numpy("embeddings")
        config = folio2.get_json("config")
        np.testing.assert_array_equal(embeddings, np.array([1, 2, 3]))
        assert config == {"lr": 0.01}
