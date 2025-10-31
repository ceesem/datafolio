"""Tests for DataAccessor and ItemProxy (autocomplete-friendly access)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datafolio import DataFolio


class DummyModel:
    """Dummy model for testing."""

    def __init__(self, param=42):
        self.param = param


class TestDataAccessor:
    """Tests for DataAccessor class."""

    def test_data_accessor_attribute_access(self, tmp_path):
        """Test accessing items via attribute syntax."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df)

        # Access via attribute
        item = folio.data.results
        assert item is not None
        assert item._name == "results"

    def test_data_accessor_dict_access(self, tmp_path):
        """Test accessing items via dictionary syntax."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df)

        # Access via dictionary
        item = folio.data["results"]
        assert item is not None
        assert item._name == "results"

    def test_data_accessor_both_access_methods_equivalent(self, tmp_path):
        """Test that both access methods return equivalent proxies."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df)

        # Both should work the same
        attr_item = folio.data.results
        dict_item = folio.data["results"]

        assert attr_item._name == dict_item._name
        assert attr_item._folio is dict_item._folio

    def test_data_accessor_nonexistent_attribute_error(self, tmp_path):
        """Test that accessing nonexistent item raises AttributeError."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(AttributeError, match="not found"):
            _ = folio.data.nonexistent

    def test_data_accessor_nonexistent_dict_key_error(self, tmp_path):
        """Test that accessing nonexistent item via dict raises KeyError."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            _ = folio.data["nonexistent"]

    def test_data_accessor_dir_returns_item_names(self, tmp_path):
        """Test that __dir__ returns all item names for autocomplete."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("table1", df)
        folio.add_table("table2", df)
        folio.add_numpy("array1", np.array([1, 2, 3]))

        # __dir__ should include all item names
        items = dir(folio.data)
        assert "table1" in items
        assert "table2" in items
        assert "array1" in items

    def test_data_accessor_repr(self, tmp_path):
        """Test DataAccessor repr shows organized summary."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df)
        model = DummyModel()
        folio.add_model("classifier", model)

        repr_str = repr(folio.data)
        assert "DataAccessor" in repr_str
        assert "results" in repr_str
        assert "classifier" in repr_str

    def test_data_accessor_empty_folio(self, tmp_path):
        """Test DataAccessor with empty folio."""
        folio = DataFolio(tmp_path / "test")

        repr_str = repr(folio.data)
        assert "no items" in repr_str


class TestItemProxy:
    """Tests for ItemProxy class."""

    def test_item_proxy_content_table(self, tmp_path):
        """Test .content returns DataFrame for tables."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df)

        # Get content via proxy
        result_df = folio.data.results.content
        pd.testing.assert_frame_equal(result_df, df)

    def test_item_proxy_content_numpy(self, tmp_path):
        """Test .content returns numpy array for numpy items."""
        folio = DataFolio(tmp_path / "test")
        arr = np.array([1, 2, 3, 4, 5])
        folio.add_numpy("embeddings", arr)

        # Get content via proxy
        result_arr = folio.data.embeddings.content
        np.testing.assert_array_equal(result_arr, arr)

    def test_item_proxy_content_json(self, tmp_path):
        """Test .content returns dict/list for JSON items."""
        folio = DataFolio(tmp_path / "test")
        config = {"lr": 0.01, "batch_size": 32}
        folio.add_json("config", config)

        # Get content via proxy
        result_config = folio.data.config.content
        assert result_config == config

    def test_item_proxy_content_model(self, tmp_path):
        """Test .content returns model object for models."""
        folio = DataFolio(tmp_path / "test")
        model = DummyModel(param=99)
        folio.add_model("classifier", model)

        # Get content via proxy
        result_model = folio.data.classifier.content
        assert result_model.param == 99

    def test_item_proxy_content_artifact(self, tmp_path):
        """Test .content returns file path for artifacts."""
        # Create artifact file
        artifact_file = tmp_path / "plot.png"
        artifact_file.write_bytes(b"fake image data")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("plot", artifact_file)

        # Get content (should be file path)
        result_path = folio.data.plot.content
        assert isinstance(result_path, str)

        # Should be able to open the file
        with open(result_path, "rb") as f:
            content = f.read()
        assert content == b"fake image data"

    def test_item_proxy_description(self, tmp_path):
        """Test .description property."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df, description="Test results")

        assert folio.data.results.description == "Test results"

    def test_item_proxy_description_none(self, tmp_path):
        """Test .description returns None when not set."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df)

        assert folio.data.results.description is None

    def test_item_proxy_type(self, tmp_path):
        """Test .type property returns item type."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df)
        folio.add_numpy("embeddings", np.array([1, 2, 3]))
        model = DummyModel()
        folio.add_model("classifier", model)

        assert folio.data.results.type == "included_table"
        assert folio.data.embeddings.type == "numpy_array"
        assert folio.data.classifier.type == "model"

    def test_item_proxy_path_referenced_table(self, tmp_path):
        """Test .path returns external path for referenced tables."""
        # Create external parquet file
        df = pd.DataFrame({"a": [1, 2, 3]})
        external_file = tmp_path / "external.parquet"
        df.to_parquet(external_file)

        folio = DataFolio(tmp_path / "test")
        folio.reference_table("external_data", external_file, table_format="parquet")

        path = folio.data.external_data.path
        assert path is not None
        assert "external.parquet" in path

    def test_item_proxy_path_artifact(self, tmp_path):
        """Test .path returns artifact path for artifacts."""
        artifact_file = tmp_path / "plot.png"
        artifact_file.write_text("fake image")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("plot", artifact_file)

        path = folio.data.plot.path
        assert path is not None
        assert "plot.png" in path

    def test_item_proxy_path_none_for_tables(self, tmp_path):
        """Test .path returns None for included tables."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df)

        path = folio.data.results.path
        assert path is None

    def test_item_proxy_inputs(self, tmp_path):
        """Test .inputs property returns lineage inputs."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("raw", df)
        folio.add_table("processed", df, inputs=["raw"])

        inputs = folio.data.processed.inputs
        assert inputs == ["raw"]

    def test_item_proxy_inputs_empty(self, tmp_path):
        """Test .inputs returns empty list when no inputs."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("raw", df)

        inputs = folio.data.raw.inputs
        assert inputs == []

    def test_item_proxy_dependents(self, tmp_path):
        """Test .dependents property returns lineage dependents."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("raw", df)
        folio.add_table("processed", df, inputs=["raw"])

        dependents = folio.data.raw.dependents
        assert "processed" in dependents

    def test_item_proxy_dependents_empty(self, tmp_path):
        """Test .dependents returns empty list when no dependents."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("raw", df)

        dependents = folio.data.raw.dependents
        assert dependents == []

    def test_item_proxy_metadata(self, tmp_path):
        """Test .metadata property returns full metadata dict."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df, description="Test results", inputs=["raw"])

        metadata = folio.data.results.metadata
        assert isinstance(metadata, dict)
        assert metadata["name"] == "results"
        assert metadata["description"] == "Test results"
        assert metadata["inputs"] == ["raw"]
        assert metadata["item_type"] == "included_table"

    def test_item_proxy_repr(self, tmp_path):
        """Test ItemProxy repr."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df, description="Test results")

        repr_str = repr(folio.data.results)
        assert "ItemProxy" in repr_str
        assert "results" in repr_str
        assert "included_table" in repr_str
        assert "Test results" in repr_str


class TestDataAccessorWorkflows:
    """Integration tests for common workflows using data accessor."""

    def test_workflow_explore_and_load(self, tmp_path):
        """Test workflow: explore available data then load it."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("results", df, description="Analysis results")

        # Explore what's available
        items = dir(folio.data)
        assert "results" in items

        # Check description
        desc = folio.data.results.description
        assert desc == "Analysis results"

        # Load the data
        loaded_df = folio.data.results.content
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_workflow_check_lineage(self, tmp_path):
        """Test workflow: check dependencies before loading."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("raw", df, description="Raw data")
        folio.add_table("processed", df, description="Processed", inputs=["raw"])

        # Check what processed depends on
        inputs = folio.data.processed.inputs
        assert inputs == ["raw"]

        # Check what depends on raw
        dependents = folio.data.raw.dependents
        assert "processed" in dependents

    def test_workflow_mixed_item_types(self, tmp_path):
        """Test accessing different item types in same workflow."""
        folio = DataFolio(tmp_path / "test")

        # Add different types
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.add_numpy("embeddings", np.array([1, 2, 3]))
        folio.add_json("config", {"lr": 0.01})
        model = DummyModel()
        folio.add_model("clf", model)

        # Access all via data accessor
        data_df = folio.data.data.content
        embeddings = folio.data.embeddings.content
        config = folio.data.config.content
        clf = folio.data.clf.content

        assert isinstance(data_df, pd.DataFrame)
        assert isinstance(embeddings, np.ndarray)
        assert isinstance(config, dict)
        assert isinstance(clf, DummyModel)

    def test_workflow_artifact_file_reading(self, tmp_path):
        """Test workflow: use artifact path with open()."""
        # Create artifact
        artifact_file = tmp_path / "data.txt"
        artifact_file.write_text("Hello, DataFolio!")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("textfile", artifact_file)

        # Use the path with open()
        with open(folio.data.textfile.content, "r") as f:
            content = f.read()

        assert content == "Hello, DataFolio!"
