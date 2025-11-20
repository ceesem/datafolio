"""Unit tests for PyTorchHandler.

These tests directly test the PyTorchHandler class methods to ensure
full coverage of the handler implementation.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from datafolio.handlers.pytorch_models import PyTorchHandler

# Try to import PyTorch but don't skip all tests if not available
try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@pytest.fixture
def handler():
    """Create a PyTorchHandler instance."""
    return PyTorchHandler()


@pytest.fixture
def mock_folio(tmp_path):
    """Create a mock DataFolio instance."""
    folio = Mock()
    folio._bundle_dir = str(tmp_path / "bundle")
    folio._storage = Mock()
    folio._items = {}
    return folio


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")

    class SimpleNet(nn.Module):
        def __init__(self, input_dim=10, output_dim=2):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)

    return SimpleNet(input_dim=10, output_dim=2)


# ============================================================================
# Test item_type property
# ============================================================================


def test_item_type(handler):
    """Test that item_type returns correct identifier."""
    assert handler.item_type == "pytorch_model"


# ============================================================================
# Test can_handle
# ============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_can_handle_with_pytorch_model(handler, simple_model):
    """Test can_handle returns True for PyTorch models."""
    assert handler.can_handle(simple_model) is True


def test_can_handle_with_non_model(handler):
    """Test can_handle returns False for non-PyTorch objects."""
    assert handler.can_handle("not a model") is False
    assert handler.can_handle(123) is False
    assert handler.can_handle([1, 2, 3]) is False
    assert handler.can_handle({"key": "value"}) is False


def test_can_handle_without_torch_installed(handler):
    """Test can_handle returns False when PyTorch is not installed."""
    # Mock ImportError when trying to import torch.nn
    import sys

    # Save the original modules
    original_torch = sys.modules.get("torch")
    original_torch_nn = sys.modules.get("torch.nn")

    try:
        # Remove torch modules to simulate import failure
        sys.modules["torch"] = None
        sys.modules["torch.nn"] = None

        # Clear the module so reimport is attempted
        if "torch" in sys.modules:
            del sys.modules["torch"]
        if "torch.nn" in sys.modules:
            del sys.modules["torch.nn"]

        # Patch __import__ to raise ImportError for torch
        def mock_import(name, *args, **kwargs):
            if name.startswith("torch"):
                raise ImportError(f"No module named '{name}'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = handler.can_handle("anything")
            # When import fails, can_handle should return False
            # This exercises the except ImportError path
            assert result is False
    finally:
        # Restore original modules
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        if original_torch_nn is not None:
            sys.modules["torch.nn"] = original_torch_nn


# ============================================================================
# Test add method
# ============================================================================


def test_add_basic(handler, mock_folio, simple_model):
    """Test basic add operation."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_pytorch = Mock()

    metadata = handler.add(
        folio=mock_folio,
        name="test_model",
        model=simple_model,
        description=None,
        inputs=None,
    )

    # Verify write_pytorch was called
    assert mock_folio._storage.write_pytorch.called

    # Verify metadata structure
    assert metadata["name"] == "test_model"
    assert metadata["item_type"] == "pytorch_model"
    assert metadata["filename"] == "test_model.pt"
    assert metadata["model_type"] == "SimpleNet"
    assert "torch_version" in metadata
    assert "created_at" in metadata


def test_add_with_description(handler, mock_folio, simple_model):
    """Test add with description."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_pytorch = Mock()

    metadata = handler.add(
        folio=mock_folio,
        name="test_model",
        model=simple_model,
        description="Test description",
        inputs=None,
    )

    assert metadata["description"] == "Test description"


def test_add_with_inputs(handler, mock_folio, simple_model):
    """Test add with lineage inputs."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_pytorch = Mock()

    metadata = handler.add(
        folio=mock_folio,
        name="test_model",
        model=simple_model,
        description=None,
        inputs=["input1", "input2"],
    )

    assert metadata["inputs"] == ["input1", "input2"]


def test_add_with_both_description_and_inputs(handler, mock_folio, simple_model):
    """Test add with both description and inputs."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_pytorch = Mock()

    metadata = handler.add(
        folio=mock_folio,
        name="model",
        model=simple_model,
        description="My model",
        inputs=["data1", "data2"],
    )

    assert metadata["description"] == "My model"
    assert metadata["inputs"] == ["data1", "data2"]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_add_non_module_raises_error(handler, mock_folio):
    """Test add raises TypeError for non-Module objects."""
    not_a_model = "not a PyTorch model"

    with pytest.raises(TypeError, match="Expected torch.nn.Module"):
        handler.add(
            folio=mock_folio,
            name="test",
            model=not_a_model,
            description=None,
            inputs=None,
        )


def test_add_without_torch_raises_error(handler, mock_folio):
    """Test add raises ImportError when PyTorch is not available."""
    # Mock ImportError when trying to import torch in add()
    import sys

    # Save the original modules
    original_torch = sys.modules.get("torch")
    original_torch_nn = sys.modules.get("torch.nn")

    try:
        # Remove torch modules to simulate import failure
        if "torch" in sys.modules:
            del sys.modules["torch"]
        if "torch.nn" in sys.modules:
            del sys.modules["torch.nn"]

        # Patch __import__ to raise ImportError for torch
        def mock_import(name, *args, **kwargs):
            if name.startswith("torch"):
                raise ImportError(f"No module named '{name}'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="PyTorch is required"):
                handler.add(
                    folio=mock_folio,
                    name="test",
                    model="fake_model",
                    description=None,
                    inputs=None,
                )
    finally:
        # Restore original modules
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        if original_torch_nn is not None:
            sys.modules["torch.nn"] = original_torch_nn


def test_add_creates_correct_filepath(handler, mock_folio, simple_model):
    """Test add creates filepath in correct subdirectory."""
    mock_folio._bundle_dir = "/path/to/bundle"
    mock_folio._storage.join_paths = Mock(return_value="/path/to/bundle/models/test.pt")
    mock_folio._storage.write_pytorch = Mock()

    handler.add(
        folio=mock_folio, name="test", model=simple_model, description=None, inputs=None
    )

    # Verify join_paths was called with correct arguments
    mock_folio._storage.join_paths.assert_called_once_with(
        "/path/to/bundle", "models", "test.pt"
    )


def test_add_timestamp_is_utc(handler, mock_folio, simple_model):
    """Test add sets created_at timestamp in UTC."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_pytorch = Mock()

    metadata = handler.add(
        folio=mock_folio, name="test", model=simple_model, description=None, inputs=None
    )

    # Parse the ISO timestamp
    created_at_str = metadata["created_at"]
    created_at = datetime.fromisoformat(created_at_str)

    # Should end with +00:00 or Z for UTC
    assert created_at_str.endswith("+00:00") or created_at_str.endswith("Z")


# ============================================================================
# Test get method
# ============================================================================


def test_get_without_model_class_raises_error(handler, mock_folio):
    """Test get raises ValueError when model_class is not provided and no auto-reconstruction."""
    mock_folio._items = {"test": {"filename": "test.pt"}}
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    # Return a bundle without serialized_class or metadata for auto-reconstruction
    mock_folio._storage.read_pytorch = Mock(
        return_value={"state_dict": {}, "metadata": {}}
    )

    with pytest.raises(ValueError, match="model_class is required"):
        handler.get(folio=mock_folio, name="test", model_class=None)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_get_with_model_class(handler, mock_folio, simple_model):
    """Test get with model_class parameter."""
    import torch.nn as nn

    # Setup mock
    mock_folio._items = {"test": {"filename": "test.pt"}}
    mock_folio._storage.join_paths = lambda *args: "/".join(args)

    # Mock read_pytorch to return a bundle dict
    mock_bundle = {
        "state_dict": simple_model.state_dict(),
        "metadata": {"model_class": "SimpleNet"},
    }
    mock_folio._storage.read_pytorch = Mock(return_value=mock_bundle)

    # Get should reconstruct the model using the provided model_class
    result = handler.get(
        folio=mock_folio, name="test", model_class=simple_model.__class__
    )

    # Verify storage.read_pytorch was called
    assert mock_folio._storage.read_pytorch.called
    # Verify result is a model instance
    assert isinstance(result, nn.Module)


def test_get_nonexistent_raises_error(handler, mock_folio):
    """Test get raises KeyError for non-existent item."""
    mock_folio._items = {}

    with pytest.raises(KeyError):
        handler.get(folio=mock_folio, name="nonexistent")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_get_reconstruct_false(handler, mock_folio, simple_model):
    """Test get with reconstruct=False returns state_dict only."""
    mock_folio._items = {"test": {"filename": "test.pt"}}
    mock_folio._storage.join_paths = lambda *args: "/".join(args)

    state_dict = simple_model.state_dict()
    mock_bundle = {"state_dict": state_dict, "metadata": {}}
    mock_folio._storage.read_pytorch = Mock(return_value=mock_bundle)

    result = handler.get(folio=mock_folio, name="test", reconstruct=False)

    # Should return just the state_dict
    assert result == state_dict
    assert isinstance(result, dict)


def test_get_creates_correct_filepath(handler, mock_folio):
    """Test get creates filepath from metadata."""
    import torch.nn as nn

    mock_folio._items = {"test": {"filename": "test_model.pt"}}
    mock_folio._bundle_dir = "/bundle"
    mock_folio._storage.join_paths = Mock(return_value="/bundle/models/test_model.pt")
    mock_folio._storage.read_pytorch = Mock(
        return_value={"state_dict": {}, "metadata": {}}
    )

    # Use reconstruct=False to avoid ValueError
    handler.get(folio=mock_folio, name="test", reconstruct=False)

    # Verify join_paths was called correctly
    mock_folio._storage.join_paths.assert_called_once_with(
        "/bundle", "models", "test_model.pt"
    )


# ============================================================================
# Test get_storage_subdir
# ============================================================================


def test_get_storage_subdir(handler):
    """Test that get_storage_subdir returns correct subdirectory."""
    subdir = handler.get_storage_subdir()

    # PyTorch models should be in "models" subdirectory
    # (item_type='pytorch_model' maps to MODELS category)
    assert subdir == "models"


# ============================================================================
# Test delete method (inherited from BaseHandler)
# ============================================================================


def test_delete_removes_file(handler, mock_folio):
    """Test delete method removes file."""
    mock_folio._items = {"test": {"filename": "test.pt"}}
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.delete_file = Mock()

    handler.delete(folio=mock_folio, name="test")

    # Verify delete_file was called
    assert mock_folio._storage.delete_file.called


# ============================================================================
# Edge cases and error conditions
# ============================================================================


def test_add_with_extra_kwargs(handler, mock_folio, simple_model):
    """Test add ignores extra kwargs."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_pytorch = Mock()

    # Should not raise error with extra kwargs
    metadata = handler.add(
        folio=mock_folio,
        name="test",
        model=simple_model,
        description=None,
        inputs=None,
        extra_param="ignored",
        another_param=123,
    )

    assert metadata["name"] == "test"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_add_model_type_captured(handler, mock_folio):
    """Test that model type name is captured correctly."""

    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(5, 3)

        def forward(self, x):
            return self.layer(x)

    model = CustomModel()

    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_pytorch = Mock()

    metadata = handler.add(
        folio=mock_folio, name="custom", model=model, description=None, inputs=None
    )

    assert metadata["model_type"] == "CustomModel"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_handler_instance_reuse(handler, mock_folio, simple_model):
    """Test that handler instance can be reused for multiple operations."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_pytorch = Mock()

    # Add multiple models with same handler instance
    metadata1 = handler.add(
        folio=mock_folio,
        name="model1",
        model=simple_model,
        description=None,
        inputs=None,
    )
    metadata2 = handler.add(
        folio=mock_folio,
        name="model2",
        model=simple_model,
        description=None,
        inputs=None,
    )

    assert metadata1["name"] == "model1"
    assert metadata2["name"] == "model2"
    assert mock_folio._storage.write_pytorch.call_count == 2
