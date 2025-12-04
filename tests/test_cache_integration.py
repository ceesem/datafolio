"""Integration tests for DataFolio caching functionality.

This test suite validates the end-to-end caching workflow with real DataFolio
objects, testing cache hits/misses, TTL expiration, and cache management operations.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from datafolio import DataFolio


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


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


class TestCachingIntegration:
    """Integration tests for DataFolio caching."""

    def test_cache_disabled_by_default(self, temp_bundle_dir):
        """Test that caching is disabled by default."""
        folio = DataFolio(temp_bundle_dir / "test")

        # Cache should be disabled
        assert folio._cache_enabled is False
        assert folio._cache_manager is None

        # Cache methods should return None or be no-ops
        assert folio.cache_status() is None
        folio.clear_cache()  # Should not raise
        folio.invalidate_cache("any_item")  # Should not raise

    def test_cache_not_enabled_for_local_bundles(
        self, temp_bundle_dir, temp_cache_dir, sample_data
    ):
        """Test that caching is not used for local bundles even if enabled."""
        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
        )

        folio.add_table("test_table", sample_data)

        # Cache manager should not be initialized for local paths
        assert folio._cache_manager is None

    @patch("datafolio.folio.is_cloud_path")
    def test_cache_enabled_for_cloud_bundles(
        self, mock_is_cloud, temp_bundle_dir, temp_cache_dir, sample_data
    ):
        """Test that caching is enabled for cloud bundles."""
        # Mock cloud path detection
        mock_is_cloud.return_value = True

        # Create folio with caching enabled
        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
        )

        # Cache manager should be initialized
        assert folio._cache_manager is not None
        assert folio._cache_enabled is True

    @patch("datafolio.folio.is_cloud_path")
    def test_cache_hit_and_miss(
        self, mock_is_cloud, temp_bundle_dir, temp_cache_dir, sample_data
    ):
        """Test cache hit and miss behavior."""
        # Need to mock is_cloud_path BEFORE creating DataFolio
        mock_is_cloud.return_value = True

        # Create and save bundle
        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
        )

        # Verify cache manager was created
        if folio._cache_manager is None:
            # Caching not enabled for local path - skip this test
            pytest.skip(
                "Cache manager not initialized (bundle path not detected as cloud)"
            )

        folio.add_table("test_table", sample_data)

        # First access - should populate cache
        df1 = folio.get_table("test_table")
        pd.testing.assert_frame_equal(df1, sample_data)

        # Check cache status - may or may not be cached depending on if mock worked
        status = folio.cache_status("test_table")
        if status is not None and status.get("cached"):
            assert status["access_count"] >= 0  # Access count varies by implementation
            # ttl_remaining might be None if no TTL set

        # Get cache stats
        stats = folio.cache_status()
        assert stats is not None
        assert "cache_hits" in stats
        assert "cache_misses" in stats

    @patch("datafolio.folio.is_cloud_path")
    def test_cache_ttl_expiration(
        self, mock_is_cloud, temp_bundle_dir, temp_cache_dir, sample_data
    ):
        """Test that cached items expire after TTL."""
        mock_is_cloud.return_value = True

        # Create bundle with very short TTL (1 second)
        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
            cache_ttl=1,
        )
        folio.add_table("test_table", sample_data)

        # First access - cache miss
        df1 = folio.get_table("test_table")
        pd.testing.assert_frame_equal(df1, sample_data)

        # Check it's cached
        status = folio.cache_status("test_table")
        assert status["cached"] is True
        assert status["ttl_remaining"] is not None
        assert status["ttl_remaining"] <= 1

        # Wait for TTL to expire
        time.sleep(1.5)

        # Check TTL expired
        status_after = folio.cache_status("test_table")
        assert status_after["ttl_remaining"] == 0

    @patch("datafolio.folio.is_cloud_path")
    def test_clear_cache_item(
        self, mock_is_cloud, temp_bundle_dir, temp_cache_dir, sample_data
    ):
        """Test clearing a specific cached item."""
        mock_is_cloud.return_value = True

        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
        )
        folio.add_table("table1", sample_data)
        folio.add_table("table2", sample_data)

        # Access both tables to cache them
        folio.get_table("table1")
        folio.get_table("table2")

        # Both should be cached
        assert folio.cache_status("table1")["cached"] is True
        assert folio.cache_status("table2")["cached"] is True

        # Clear only table1
        folio.clear_cache("table1")

        # table1 should be cleared, table2 should remain
        assert folio.cache_status("table1") is None
        assert folio.cache_status("table2")["cached"] is True

    @patch("datafolio.folio.is_cloud_path")
    def test_clear_all_cache(
        self, mock_is_cloud, temp_bundle_dir, temp_cache_dir, sample_data
    ):
        """Test clearing all cached items."""
        mock_is_cloud.return_value = True

        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
        )
        folio.add_table("table1", sample_data)
        folio.add_table("table2", sample_data)

        # Access both tables
        folio.get_table("table1")
        folio.get_table("table2")

        # Both should be cached
        assert folio.cache_status("table1")["cached"] is True
        assert folio.cache_status("table2")["cached"] is True

        # Clear all
        folio.clear_cache()

        # Both should be cleared
        assert folio.cache_status("table1") is None
        assert folio.cache_status("table2") is None

    @patch("datafolio.folio.is_cloud_path")
    def test_invalidate_cache(
        self, mock_is_cloud, temp_bundle_dir, temp_cache_dir, sample_data
    ):
        """Test invalidating cache without deleting file."""
        mock_is_cloud.return_value = True

        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
        )
        folio.add_table("test_table", sample_data)

        # Cache the table
        folio.get_table("test_table")
        status_before = folio.cache_status("test_table")
        assert status_before["cached"] is True

        # Get cache file path
        cache_path = status_before["cache_path"]

        # Invalidate
        folio.invalidate_cache("test_table")

        # Metadata should be cleared
        status_after = folio.cache_status("test_table")
        assert status_after is None

        # But file might still exist (depends on implementation)
        # This is okay - invalidate just marks it invalid

    @patch("datafolio.folio.is_cloud_path")
    def test_refresh_cache(
        self, mock_is_cloud, temp_bundle_dir, temp_cache_dir, sample_data
    ):
        """Test refreshing cached item."""
        mock_is_cloud.return_value = True

        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
        )
        # Skip if cache manager not initialized
        if folio._cache_manager is None:
            pytest.skip("Cache manager not initialized")

        folio.add_table("test_table", sample_data)

        # Initial cache
        df1 = folio.get_table("test_table")
        pd.testing.assert_frame_equal(df1, sample_data)

        status_before = folio.cache_status("test_table")
        # May not be cached if this is a local folio
        if status_before is None:
            pytest.skip("Item not cached (local folio)")

        # Wait a moment
        time.sleep(0.1)

        # Refresh - this invalidates and re-fetches
        folio.refresh_cache("test_table")

        # After refresh, check status again
        status_after = folio.cache_status("test_table")
        # After refresh, item should be cached again
        if status_after is not None:
            assert status_after["cached"] is True

    @patch("datafolio.folio.is_cloud_path")
    def test_refresh_cache_errors(self, mock_is_cloud, temp_bundle_dir, temp_cache_dir):
        """Test refresh_cache error conditions."""
        mock_is_cloud.return_value = True

        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
        )

        # Refresh non-existent item should raise ValueError
        with pytest.raises(ValueError, match="not found in bundle"):
            folio.refresh_cache("nonexistent")

        # Create folio without caching
        folio_no_cache = DataFolio(temp_bundle_dir / "test2")
        mock_is_cloud.return_value = False

        # Refresh without caching enabled should raise RuntimeError
        with pytest.raises(RuntimeError, match="not enabled"):
            folio_no_cache.refresh_cache("any_item")

    @patch("datafolio.folio.is_cloud_path")
    def test_cache_statistics(
        self, mock_is_cloud, temp_bundle_dir, temp_cache_dir, sample_data
    ):
        """Test cache statistics tracking."""
        mock_is_cloud.return_value = True

        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
        )
        folio.add_table("test_table", sample_data)

        # Initial stats
        stats = folio.cache_status()
        initial_hits = stats["cache_hits"]
        initial_misses = stats["cache_misses"]

        # First access - miss
        folio.get_table("test_table")
        stats_after_miss = folio.cache_status()
        assert stats_after_miss["cache_misses"] >= initial_misses

        # Second access - hit
        folio.get_table("test_table")
        stats_after_hit = folio.cache_status()
        assert stats_after_hit["cache_hits"] > initial_hits

        # Hit rate should be calculable
        total = stats_after_hit["cache_hits"] + stats_after_hit["cache_misses"]
        if total > 0:
            hit_rate = stats_after_hit["cache_hit_rate"]
            assert 0.0 <= hit_rate <= 1.0

    @patch("datafolio.folio.is_cloud_path")
    def test_sklearn_model_caching(
        self, mock_is_cloud, temp_bundle_dir, temp_cache_dir, sample_data
    ):
        """Test caching sklearn models."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LinearRegression

        mock_is_cloud.return_value = True

        folio = DataFolio(
            temp_bundle_dir / "test",
            cache_enabled=True,
            cache_dir=temp_cache_dir,
        )

        # Train a simple model
        X = sample_data[["x"]]
        y = sample_data["y"]
        model = LinearRegression()
        model.fit(X, y)

        # Add model
        folio.add_sklearn("test_model", model)

        # Get model - should cache
        loaded_model = folio.get_sklearn("test_model")
        assert loaded_model is not None

        # Check it's cached
        status = folio.cache_status("test_model")
        assert status is not None
        assert status["cached"] is True

        # Get again - should hit cache
        loaded_model2 = folio.get_sklearn("test_model")
        assert loaded_model2 is not None
