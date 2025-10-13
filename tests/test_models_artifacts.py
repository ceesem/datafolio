"""Tests for model and artifact handling functionality."""

import tempfile
from pathlib import Path

import pytest

from datafolio import DataFolio


class DummyModel:
    """Dummy model for testing."""

    def __init__(self, param=1):
        self.param = param

    def predict(self, x):
        return x * self.param


class TestAddModel:
    """Tests for add_model method."""

    def test_add_model_basic(self, tmp_path):
        """Test basic model addition."""
        folio = DataFolio(tmp_path / "test")
        model = DummyModel(param=5)

        folio.add_model("classifier", model)

        assert "classifier" in folio._items
        assert folio._items["classifier"]["item_type"] == "model"
        # Verify model can be retrieved
        retrieved = folio.get_model("classifier")
        assert retrieved.param == 5

    def test_add_model_metadata(self, tmp_path):
        """Test model metadata is captured correctly."""
        folio = DataFolio(tmp_path / "test")
        model = DummyModel()

        folio.add_model("clf", model, description="Test classifier")

        metadata = folio._items["clf"]
        assert metadata["item_type"] == "model"
        assert metadata["filename"] == "clf.joblib"
        assert metadata["description"] == "Test classifier"

    def test_add_model_method_chaining(self, tmp_path):
        """Test method chaining works."""
        folio = DataFolio(tmp_path / "test")
        model = DummyModel()

        result = folio.add_model("clf", model)
        assert result is folio

    def test_add_model_duplicate_name(self, tmp_path):
        """Test error when adding duplicate name."""
        folio = DataFolio(tmp_path / "test")
        model1 = DummyModel(param=1)
        model2 = DummyModel(param=2)

        folio.add_model("clf", model1)

        with pytest.raises(ValueError, match="already exists"):
            folio.add_model("clf", model2)

    def test_add_model_appears_in_list_contents(self, tmp_path):
        """Test model appears in list_contents."""
        folio = DataFolio(tmp_path / "test")
        model = DummyModel()
        folio.add_model("clf", model)

        contents = folio.list_contents()
        assert "clf" in contents["models"]
        assert len(contents["models"]) == 1


class TestGetModel:
    """Tests for get_model method."""

    def test_get_model_from_disk(self, tmp_path):
        """Test getting a model (read from disk)."""
        folio = DataFolio(tmp_path / "test")
        model = DummyModel(param=42)
        folio.add_model("clf", model)

        retrieved = folio.get_model("clf")

        # Models are read fresh from disk, not cached
        assert retrieved.param == 42

    def test_get_model_not_found(self, tmp_path):
        """Test error when model doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_model("nonexistent")


class TestAddArtifact:
    """Tests for add_artifact method."""

    def test_add_artifact_basic(self, tmp_path):
        """Test basic artifact addition."""
        # Create a temporary file
        artifact_file = tmp_path / "test_source.txt"
        artifact_file.write_text("test content")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("test_file", artifact_file)

        assert "test_file" in folio._items
        assert folio._items["test_file"]["item_type"] == "artifact"

    def test_add_artifact_with_category(self, tmp_path):
        """Test artifact with category."""
        artifact_file = tmp_path / "plot.png"
        artifact_file.write_text("fake image")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("loss_curve", artifact_file, category="plots")

        metadata = folio._items["loss_curve"]
        assert metadata["item_type"] == "artifact"
        assert metadata["category"] == "plots"
        assert metadata["filename"] == "loss_curve.png"

    def test_add_artifact_preserves_extension(self, tmp_path):
        """Test that file extension is preserved."""
        artifact_file = tmp_path / "config.json"
        artifact_file.write_text('{"key": "value"}')

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("config", artifact_file)

        metadata = folio._items["config"]
        assert metadata["filename"] == "config.json"

    def test_add_artifact_with_description(self, tmp_path):
        """Test artifact with description."""
        artifact_file = tmp_path / "data.csv"
        artifact_file.write_text("col1,col2\n1,2")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact(
            "summary", artifact_file, category="data", description="Summary statistics"
        )

        metadata = folio._items["summary"]
        assert metadata["description"] == "Summary statistics"

    def test_add_artifact_method_chaining(self, tmp_path):
        """Test method chaining works."""
        artifact_file = tmp_path / "test.txt"
        artifact_file.write_text("test")

        folio = DataFolio(tmp_path / "test")
        result = folio.add_artifact("test", artifact_file)

        assert result is folio

    def test_add_artifact_duplicate_name(self, tmp_path):
        """Test error when adding duplicate name."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("test", file1)

        with pytest.raises(ValueError, match="already exists"):
            folio.add_artifact("test", file2)

    def test_add_artifact_file_not_found(self, tmp_path):
        """Test error when file doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(FileNotFoundError, match="not found"):
            folio.add_artifact("test", "/nonexistent/file.txt")

    def test_add_artifact_appears_in_list_contents(self, tmp_path):
        """Test artifact appears in list_contents."""
        artifact_file = tmp_path / "test.txt"
        artifact_file.write_text("test")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("test", artifact_file)

        contents = folio.list_contents()
        assert "test" in contents["artifacts"]
        assert len(contents["artifacts"]) == 1


class TestGetArtifactPath:
    """Tests for get_artifact_path method."""

    def test_get_artifact_path_in_bundle(self, tmp_path):
        """Test getting artifact path (stored in bundle)."""
        artifact_file = tmp_path / "source.txt"
        artifact_file.write_text("test content")

        folio = DataFolio(tmp_path / "test")
        folio.add_artifact("test", artifact_file)

        retrieved_path = folio.get_artifact_path("test")

        # Path should point to file in bundle
        assert "artifacts" in retrieved_path
        assert Path(retrieved_path).exists()

    def test_get_artifact_path_not_found(self, tmp_path):
        """Test error when artifact doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_artifact_path("nonexistent")


class TestModelsArtifactsIntegration:
    """Integration tests for models and artifacts."""

    def test_mixed_items(self, tmp_path):
        """Test using both models and artifacts."""
        # Create artifacts
        plot_file = tmp_path / "plot.png"
        plot_file.write_text("fake plot")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        # Create folio with both models and artifacts
        folio = DataFolio(tmp_path / "test")
        model1 = DummyModel(param=1)
        model2 = DummyModel(param=2)

        folio.add_model("clf1", model1)
        folio.add_model("clf2", model2)
        folio.add_artifact("plot", plot_file, category="plots")
        folio.add_artifact("config", config_file, category="configs")

        # Verify tracking
        contents = folio.list_contents()
        assert len(contents["models"]) == 2
        assert len(contents["artifacts"]) == 2
        assert "clf1" in contents["models"]
        assert "plot" in contents["artifacts"]

    def test_chaining_with_models_and_artifacts(self, tmp_path):
        """Test method chaining with models and artifacts."""
        artifact_file = tmp_path / "test.txt"
        artifact_file.write_text("test")

        folio = DataFolio(tmp_path / "test")
        model = DummyModel()

        folio.add_model("clf", model).add_artifact("artifact", artifact_file)

        contents = folio.list_contents()
        assert "clf" in contents["models"]
        assert "artifact" in contents["artifacts"]

    def test_models_and_artifacts_share_namespace(self, tmp_path):
        """Test that models and artifacts share the same namespace."""
        # They use the same _items dict, so names must be unique
        artifact_file = tmp_path / "test.txt"
        artifact_file.write_text("test")

        folio = DataFolio(tmp_path / "test")
        model = DummyModel()

        # This should fail because they share the same _items dict
        folio.add_model("item1", model)

        with pytest.raises(ValueError, match="already exists"):
            folio.add_artifact("item1", artifact_file)
