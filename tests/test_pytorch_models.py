"""Tests for PyTorch model handling functionality."""

import tempfile
from pathlib import Path

import pytest

from datafolio import DataFolio

# Skip all tests if PyTorch is not installed
pytest.importorskip("torch")

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple PyTorch model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestAddPyTorch:
    """Tests for add_pytorch method."""

    def test_add_pytorch_basic(self, tmp_path):
        """Test basic PyTorch model addition."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel(input_dim=5, hidden_dim=10)

        folio.add_pytorch(
            "neural_net", model, init_args={"input_dim": 5, "hidden_dim": 10}
        )

        assert "neural_net" in folio._items
        assert folio._items["neural_net"]["item_type"] == "pytorch_model"
        assert folio._items["neural_net"]["filename"] == "neural_net.pt"

    def test_add_pytorch_metadata(self, tmp_path):
        """Test PyTorch model metadata is captured correctly."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel()

        folio.add_pytorch(
            "net",
            model,
            description="Test neural network",
            hyperparameters={"learning_rate": 0.001},
            init_args={"input_dim": 10, "hidden_dim": 20},
        )

        metadata = folio._items["net"]
        assert metadata["item_type"] == "pytorch_model"
        assert metadata["filename"] == "net.pt"
        assert metadata["description"] == "Test neural network"
        assert metadata["hyperparameters"]["learning_rate"] == 0.001
        assert metadata["init_args"]["input_dim"] == 10

    def test_add_pytorch_with_save_class(self, tmp_path):
        """Test saving PyTorch model with class serialization."""
        pytest.importorskip("dill")

        folio = DataFolio(tmp_path / "test")
        model = SimpleModel()

        folio.add_pytorch(
            "net",
            model,
            init_args={"input_dim": 10, "hidden_dim": 20},
            save_class=True,
        )

        metadata = folio._items["net"]
        assert metadata["has_serialized_class"] is True

    def test_add_pytorch_method_chaining(self, tmp_path):
        """Test method chaining works."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel()

        result = folio.add_pytorch(
            "net", model, init_args={"input_dim": 10, "hidden_dim": 20}
        )
        assert result is folio

    def test_add_pytorch_duplicate_name(self, tmp_path):
        """Test error when adding duplicate name."""
        folio = DataFolio(tmp_path / "test")
        model1 = SimpleModel(input_dim=5)
        model2 = SimpleModel(input_dim=10)

        folio.add_pytorch("net", model1, init_args={"input_dim": 5, "hidden_dim": 20})

        with pytest.raises(ValueError, match="already exists"):
            folio.add_pytorch(
                "net", model2, init_args={"input_dim": 10, "hidden_dim": 20}
            )

    def test_add_pytorch_appears_in_list_contents(self, tmp_path):
        """Test PyTorch model appears in list_contents."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel()
        folio.add_pytorch("net", model, init_args={"input_dim": 10, "hidden_dim": 20})

        contents = folio.list_contents()
        assert "net" in contents["pytorch_models"]
        assert len(contents["pytorch_models"]) == 1

    def test_add_pytorch_with_lineage(self, tmp_path):
        """Test PyTorch model with lineage tracking."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel()

        folio.add_pytorch(
            "net",
            model,
            inputs=["training_data", "validation_data"],
            init_args={"input_dim": 10, "hidden_dim": 20},
            code="model.fit(X_train, y_train)",
        )

        metadata = folio._items["net"]
        assert metadata["inputs"] == ["training_data", "validation_data"]
        assert metadata["code"] == "model.fit(X_train, y_train)"


class TestGetPyTorch:
    """Tests for get_pytorch method."""

    def test_get_pytorch_state_dict_only(self, tmp_path):
        """Test getting state_dict only (no reconstruction)."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel(input_dim=5, hidden_dim=10)

        # Train slightly to give it distinct weights
        x = torch.randn(10, 5)
        y = model(x)

        folio.add_pytorch("net", model, init_args={"input_dim": 5, "hidden_dim": 10})

        # Get state dict only
        state_dict = folio.get_pytorch("net", reconstruct=False)

        assert isinstance(state_dict, dict)
        assert "fc1.weight" in state_dict
        assert "fc2.bias" in state_dict

    def test_get_pytorch_with_model_class(self, tmp_path):
        """Test getting PyTorch model with provided model class."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel(input_dim=5, hidden_dim=10)

        # Set some weights
        with torch.no_grad():
            model.fc1.weight.fill_(1.0)

        folio.add_pytorch("net", model, init_args={"input_dim": 5, "hidden_dim": 10})

        # Reconstruct with provided class
        loaded_model = folio.get_pytorch("net", model_class=SimpleModel)

        assert isinstance(loaded_model, SimpleModel)
        # Check weights were loaded
        assert torch.allclose(
            loaded_model.fc1.weight, torch.ones_like(loaded_model.fc1.weight)
        )

    def test_get_pytorch_auto_reconstruct(self, tmp_path):
        """Test automatic reconstruction from metadata."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel(input_dim=5, hidden_dim=10)

        folio.add_pytorch("net", model, init_args={"input_dim": 5, "hidden_dim": 10})

        # Auto-reconstruct (should work if SimpleModel is importable)
        try:
            loaded_model = folio.get_pytorch("net")
            assert isinstance(loaded_model, SimpleModel)
        except RuntimeError:
            # May fail if module path isn't importable, which is okay for this test
            pytest.skip("Auto-reconstruction requires importable module")

    def test_get_pytorch_with_dill(self, tmp_path):
        """Test reconstruction using dill-serialized class."""
        pytest.importorskip("dill")

        folio = DataFolio(tmp_path / "test")
        model = SimpleModel(input_dim=5, hidden_dim=10)

        folio.add_pytorch(
            "net",
            model,
            init_args={"input_dim": 5, "hidden_dim": 10},
            save_class=True,
        )

        # Should reconstruct from dill
        loaded_model = folio.get_pytorch("net")
        assert isinstance(loaded_model, nn.Module)

    def test_get_pytorch_not_found(self, tmp_path):
        """Test error when model doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_pytorch("nonexistent")

    def test_get_pytorch_wrong_type(self, tmp_path):
        """Test error when item is not a PyTorch model."""
        from sklearn.ensemble import RandomForestClassifier

        folio = DataFolio(tmp_path / "test")
        sklearn_model = RandomForestClassifier()
        folio.add_model("sklearn_model", sklearn_model)

        with pytest.raises(ValueError, match="not a PyTorch model"):
            folio.get_pytorch("sklearn_model")


class TestPyTorchModelsIntegration:
    """Integration tests for PyTorch models."""

    def test_pytorch_model_persistence(self, tmp_path):
        """Test that PyTorch models persist across folio instances."""
        # Create and save model
        folio1 = DataFolio(tmp_path / "test")
        model = SimpleModel(input_dim=5, hidden_dim=10)

        # Set specific weights
        with torch.no_grad():
            model.fc1.weight.fill_(2.0)
            model.fc2.bias.fill_(3.0)

        folio1.add_pytorch("net", model, init_args={"input_dim": 5, "hidden_dim": 10})

        # Load in new folio instance
        folio2 = DataFolio(tmp_path / "test")
        loaded_model = folio2.get_pytorch("net", model_class=SimpleModel)

        # Verify weights match
        assert torch.allclose(
            loaded_model.fc1.weight, torch.full_like(loaded_model.fc1.weight, 2.0)
        )
        assert torch.allclose(
            loaded_model.fc2.bias, torch.full_like(loaded_model.fc2.bias, 3.0)
        )

    def test_mixed_models(self, tmp_path):
        """Test using both sklearn and PyTorch models."""
        from sklearn.ensemble import RandomForestClassifier

        folio = DataFolio(tmp_path / "test")
        sklearn_model = RandomForestClassifier(n_estimators=10)
        pytorch_model = SimpleModel()

        folio.add_model("sklearn_clf", sklearn_model)
        folio.add_pytorch(
            "pytorch_net", pytorch_model, init_args={"input_dim": 10, "hidden_dim": 20}
        )

        # Verify tracking
        contents = folio.list_contents()
        assert len(contents["models"]) == 1
        assert len(contents["pytorch_models"]) == 1
        assert "sklearn_clf" in contents["models"]
        assert "pytorch_net" in contents["pytorch_models"]

    def test_pytorch_model_in_describe(self, tmp_path):
        """Test that PyTorch models appear in describe output."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel()

        folio.add_pytorch(
            "net",
            model,
            description="Simple feedforward network",
            init_args={"input_dim": 10, "hidden_dim": 20},
        )

        description = folio.describe(return_string=True)
        assert "PyTorch Models" in description
        assert "net" in description
        assert "Simple feedforward network" in description

    def test_pytorch_models_property(self, tmp_path):
        """Test pytorch_models property."""
        folio = DataFolio(tmp_path / "test")
        model1 = SimpleModel(input_dim=5)
        model2 = SimpleModel(input_dim=10)

        folio.add_pytorch("net1", model1, init_args={"input_dim": 5, "hidden_dim": 20})
        folio.add_pytorch("net2", model2, init_args={"input_dim": 10, "hidden_dim": 20})

        assert len(folio.pytorch_models) == 2
        assert "net1" in folio.pytorch_models
        assert "net2" in folio.pytorch_models


class TestSmartAddModel:
    """Tests for smart add_model with type detection."""

    def test_add_model_detects_pytorch(self, tmp_path):
        """Test add_model automatically detects PyTorch models."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel(input_dim=5, hidden_dim=10)

        # Use generic add_model, should detect it's PyTorch
        folio.add_model("net", model, init_args={"input_dim": 5, "hidden_dim": 10})

        assert "net" in folio._items
        assert folio._items["net"]["item_type"] == "pytorch_model"

    def test_add_model_detects_sklearn(self, tmp_path):
        """Test add_model automatically detects sklearn models."""
        from sklearn.ensemble import RandomForestClassifier

        folio = DataFolio(tmp_path / "test")
        model = RandomForestClassifier(n_estimators=10)

        # Use generic add_model, should detect it's sklearn
        folio.add_model("clf", model, hyperparameters={"n_estimators": 10})

        assert "clf" in folio._items
        assert folio._items["clf"]["item_type"] == "model"

    def test_add_model_pytorch_with_description(self, tmp_path):
        """Test add_model with PyTorch and description."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel()

        folio.add_model(
            "net",
            model,
            description="Neural network",
            init_args={"input_dim": 10, "hidden_dim": 20},
        )

        assert folio._items["net"]["description"] == "Neural network"
        assert folio._items["net"]["item_type"] == "pytorch_model"

    def test_add_model_sklearn_with_description(self, tmp_path):
        """Test add_model with sklearn and description."""
        from sklearn.ensemble import RandomForestClassifier

        folio = DataFolio(tmp_path / "test")
        model = RandomForestClassifier()

        folio.add_model("clf", model, description="Random forest")

        assert folio._items["clf"]["description"] == "Random forest"
        assert folio._items["clf"]["item_type"] == "model"


class TestSmartGetModel:
    """Tests for smart get_model with type detection."""

    def test_get_model_detects_sklearn(self, tmp_path):
        """Test get_model automatically loads sklearn models."""
        from sklearn.ensemble import RandomForestClassifier

        folio = DataFolio(tmp_path / "test")
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Fit a simple model
        import numpy as np

        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model.fit(X, y)

        folio.add_sklearn("clf", model)

        # Use generic get_model
        loaded_model = folio.get_model("clf")

        assert isinstance(loaded_model, RandomForestClassifier)
        assert loaded_model.n_estimators == 10

    def test_get_model_detects_pytorch(self, tmp_path):
        """Test get_model automatically loads PyTorch models."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel(input_dim=5, hidden_dim=10)

        # Set specific weights
        with torch.no_grad():
            model.fc1.weight.fill_(2.0)

        folio.add_pytorch("net", model, init_args={"input_dim": 5, "hidden_dim": 10})

        # Use generic get_model with model_class
        loaded_model = folio.get_model("net", model_class=SimpleModel)

        assert isinstance(loaded_model, SimpleModel)
        assert torch.allclose(
            loaded_model.fc1.weight, torch.full_like(loaded_model.fc1.weight, 2.0)
        )

    def test_get_model_pytorch_state_dict(self, tmp_path):
        """Test get_model with PyTorch can return state_dict."""
        folio = DataFolio(tmp_path / "test")
        model = SimpleModel()

        folio.add_pytorch("net", model, init_args={"input_dim": 10, "hidden_dim": 20})

        # Get state_dict only
        state_dict = folio.get_model("net", reconstruct=False)

        assert isinstance(state_dict, dict)
        assert "fc1.weight" in state_dict

    def test_get_model_not_found(self, tmp_path):
        """Test error when model doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_model("nonexistent")

    def test_get_model_mixed_types(self, tmp_path):
        """Test get_model works for both model types in same folio."""
        from sklearn.ensemble import RandomForestClassifier

        folio = DataFolio(tmp_path / "test")

        # Add both types
        sklearn_model = RandomForestClassifier(n_estimators=10)
        pytorch_model = SimpleModel(input_dim=5, hidden_dim=10)

        folio.add_sklearn("clf", sklearn_model)
        folio.add_pytorch(
            "net", pytorch_model, init_args={"input_dim": 5, "hidden_dim": 10}
        )

        # Load both with generic get_model
        loaded_sklearn = folio.get_model("clf")
        loaded_pytorch = folio.get_model("net", model_class=SimpleModel)

        assert isinstance(loaded_sklearn, RandomForestClassifier)
        assert isinstance(loaded_pytorch, SimpleModel)
