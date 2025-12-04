"""Test PyTorch model caching integration."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from datafolio import DataFolio

# Skip entire module if torch not available
torch = pytest.importorskip("torch")


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""

    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim: int = 10, hidden_dim: int = 5):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    return SimpleModel(input_dim=10, hidden_dim=5)


@pytest.fixture
def temp_bundle_dir():
    """Create temporary directory for bundle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@patch("datafolio.folio.is_cloud_path")
def test_pytorch_model_caching(
    mock_is_cloud, temp_bundle_dir, temp_cache_dir, simple_model
):
    """Test that PyTorch models can be cached."""
    mock_is_cloud.return_value = True

    folio = DataFolio(
        temp_bundle_dir / "test",
        cache_enabled=True,
        cache_dir=temp_cache_dir,
    )

    # Skip if cache manager not initialized
    if folio._cache_manager is None:
        pytest.skip("Cache manager not initialized")

    # Add model
    folio.add_pytorch(
        "test_model", simple_model, init_args={"input_dim": 10, "hidden_dim": 5}
    )

    # Get model - should cache
    loaded_model = folio.get_pytorch("test_model", reconstruct=False)
    assert loaded_model is not None

    # Verify it's a state dict
    assert isinstance(loaded_model, dict)
    assert "fc1.weight" in loaded_model
    assert "fc2.weight" in loaded_model

    # Check cache status
    status = folio.cache_status("test_model")
    if status is not None:
        assert status["cached"] is True

    # Get again - should hit cache
    loaded_model2 = folio.get_pytorch("test_model", reconstruct=False)
    assert loaded_model2 is not None

    # Verify state dicts match
    assert set(loaded_model.keys()) == set(loaded_model2.keys())
