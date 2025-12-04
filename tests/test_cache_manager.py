"""Unit tests for cache manager."""

import tempfile
from pathlib import Path

import pytest

from datafolio.cache.config import CacheConfig
from datafolio.cache.manager import CacheManager
from datafolio.cache.validation import ChecksumMismatchError


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache_config(temp_cache_dir):
    """Create test cache configuration."""
    return CacheConfig(
        enabled=True,
        cache_dir=temp_cache_dir,
        default_ttl=1800,
        checksum_algorithm="md5",
        strict_mode=False,
    )


@pytest.fixture
def cache_manager(cache_config):
    """Create test cache manager."""
    return CacheManager(
        bundle_path="gs://test-bucket/test-bundle",
        config=cache_config,
    )


class TestCacheManagerInitialization:
    """Test cache manager initialization."""

    def test_init_creates_lock_directory(self, cache_manager):
        """Test that initialization creates lock directory."""
        # Lock dir should be created during init
        assert cache_manager.lock_dir.exists()

        # Cache dir itself is created lazily when items are cached
        # (not during init to avoid creating empty directories)

    def test_bundle_path_normalization(self):
        """Test bundle path normalization."""
        # Test cloud path
        assert (
            CacheManager._normalize_bundle_path("gs://bucket/path/bundle")
            == "gs_bucket_path_bundle"
        )

        # Test local path
        assert (
            CacheManager._normalize_bundle_path("/local/path/bundle")
            == "local_path_bundle"
        )

        # Test S3 path
        assert (
            CacheManager._normalize_bundle_path("s3://my-bucket/data")
            == "s3_my-bucket_data"
        )

    def test_custom_ttl_override(self, cache_config):
        """Test TTL can be overridden per bundle."""
        manager = CacheManager(
            bundle_path="gs://test/bundle",
            config=cache_config,
            ttl_override=3600,
        )
        assert manager.ttl == 3600


class TestCacheOperations:
    """Test basic cache operations."""

    def test_is_cached_returns_false_initially(self, cache_manager):
        """Test that items are not cached initially."""
        assert cache_manager.is_cached("test_item") is False

    def test_get_cache_path(self, cache_manager):
        """Test cache path generation."""
        path = cache_manager._get_item_cache_path(
            "test_data", "included_table", "test_data.parquet"
        )

        assert "tables" in str(path)
        assert "test_data.parquet" in str(path)

    def test_get_cache_path_models(self, cache_manager):
        """Test cache path for models."""
        path = cache_manager._get_item_cache_path(
            "classifier", "model", "classifier.joblib"
        )

        assert "models" in str(path)
        assert "classifier.joblib" in str(path)

    def test_get_returns_none_without_fetch_fn(self, cache_manager):
        """Test that get returns None when item not cached and no fetch function."""
        result = cache_manager.get("nonexistent_item")
        assert result is None

    def test_fetch_and_cache(self, cache_manager):
        """Test fetching and caching an item."""
        test_data = b"Test data content"

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        # Fetch and cache
        cache_path = cache_manager.get(
            "test_item",
            remote_fetch_fn=mock_fetch,
        )

        # Verify item was cached
        assert cache_path is not None
        assert cache_path.exists()
        assert cache_path.read_bytes() == test_data

        # Verify metadata was updated
        item_meta = cache_manager.metadata.get_item("test_item")
        assert item_meta is not None
        assert item_meta["item_type"] == "included_table"
        assert item_meta["filename"] == "test.parquet"

    def test_cache_hit_uses_cached_file(self, cache_manager):
        """Test that subsequent gets use cached file."""
        test_data = b"Test data"

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        # First fetch
        path1 = cache_manager.get("test_item", remote_fetch_fn=mock_fetch)

        # Second fetch (should use cache)
        path2 = cache_manager.get("test_item")

        # Should return same path
        assert path1 == path2
        assert path2 is not None

        # Check stats
        stats = cache_manager.get_stats()
        assert stats["cache_hits"] >= 1

    def test_invalidate_item(self, cache_manager):
        """Test invalidating cached item."""
        test_data = b"Test data"

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        # Cache an item
        cache_manager.get("test_item", remote_fetch_fn=mock_fetch)

        # Invalidate it
        cache_manager.invalidate("test_item")

        # Should not be cached anymore
        assert cache_manager.metadata.get_item("test_item") is None

    def test_clear_item_removes_file(self, cache_manager):
        """Test that clear_item removes file from disk."""
        test_data = b"Test data"

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        # Cache an item
        cache_path = cache_manager.get("test_item", remote_fetch_fn=mock_fetch)

        # Clear it
        cache_manager.clear_item("test_item")

        # File should be gone
        assert not cache_path.exists()
        assert cache_manager.metadata.get_item("test_item") is None

    def test_clear_all_removes_everything(self, cache_manager):
        """Test that clear_all removes all cached items."""
        test_data = b"Test data"

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        # Cache multiple items
        cache_manager.get("item1", remote_fetch_fn=mock_fetch)
        cache_manager.get("item2", remote_fetch_fn=mock_fetch)

        # Clear all
        cache_manager.clear_all()

        # Should not be cached
        assert cache_manager.metadata.get_item("item1") is None
        assert cache_manager.metadata.get_item("item2") is None


class TestChecksumValidation:
    """Test checksum validation during caching."""

    def test_checksum_verification_success(self, cache_manager):
        """Test successful checksum verification."""
        test_data = b"Test data"
        # Compute expected checksum
        from datafolio.cache.validation import compute_checksum_from_bytes

        expected_checksum = compute_checksum_from_bytes(test_data, "md5")

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        # Should succeed
        cache_path = cache_manager.get(
            "test_item",
            remote_fetch_fn=mock_fetch,
            remote_checksum=expected_checksum,
        )

        assert cache_path is not None
        assert cache_path.exists()

    def test_checksum_mismatch_non_strict(self, cache_manager):
        """Test checksum mismatch in non-strict mode (warning only)."""
        test_data = b"Test data"
        wrong_checksum = "0" * 32  # Wrong checksum

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        # Should warn but succeed in non-strict mode
        with pytest.warns(UserWarning, match="Checksum mismatch"):
            cache_path = cache_manager.get(
                "test_item",
                remote_fetch_fn=mock_fetch,
                remote_checksum=wrong_checksum,
            )

        assert cache_path is not None

    def test_checksum_mismatch_strict(self, cache_config, temp_cache_dir):
        """Test checksum mismatch in strict mode raises exception."""
        # Create strict mode manager
        strict_config = CacheConfig(
            enabled=True,
            cache_dir=temp_cache_dir,
            checksum_algorithm="md5",
            strict_mode=True,
        )
        manager = CacheManager(
            bundle_path="gs://test/bundle",
            config=strict_config,
        )

        test_data = b"Test data"
        wrong_checksum = "0" * 32

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        # Should raise in strict mode
        with pytest.raises(ChecksumMismatchError):
            manager.get(
                "test_item",
                remote_fetch_fn=mock_fetch,
                remote_checksum=wrong_checksum,
            )


class TestCacheStatus:
    """Test cache status and statistics."""

    def test_get_status_uncached_item(self, cache_manager):
        """Test getting status of uncached item."""
        status = cache_manager.get_status("nonexistent")
        assert status is None

    def test_get_status_cached_item(self, cache_manager):
        """Test getting status of cached item."""
        test_data = b"Test data"

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        cache_manager.get("test_item", remote_fetch_fn=mock_fetch)

        status = cache_manager.get_status("test_item")
        assert status is not None
        assert status["cached"] is True
        assert "cache_path" in status
        assert "size_bytes" in status
        assert "cached_at" in status
        assert "access_count" in status
        assert status["ttl_remaining"] is not None

    def test_cache_stats(self, cache_manager):
        """Test getting cache statistics."""
        stats = cache_manager.get_stats()

        assert "bundle_path" in stats
        assert "cache_dir" in stats
        assert "ttl_seconds" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats

    def test_cache_hit_rate_calculation(self, cache_manager):
        """Test cache hit rate calculation."""
        test_data = b"Test data"

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        # First access - miss
        cache_manager.get("test_item", remote_fetch_fn=mock_fetch)

        # Second access - hit
        cache_manager.get("test_item")

        stats = cache_manager.get_stats()
        # 1 hit, 1 miss = 50% hit rate
        assert stats["cache_hit_rate"] == 0.5


class TestConcurrency:
    """Test concurrent cache access."""

    def test_lock_file_creation(self, cache_manager):
        """Test that lock files are created."""
        lock_path = cache_manager._get_lock_path("test_item")

        assert "test_item.lock" in str(lock_path)
        assert lock_path.parent == cache_manager.lock_dir

    def test_atomic_write_via_temp_file(self, cache_manager):
        """Test that writes use temporary files for atomicity."""
        test_data = b"Test data"

        def mock_fetch():
            return (test_data, "included_table", "test.parquet")

        cache_path = cache_manager.get("test_item", remote_fetch_fn=mock_fetch)

        # Final file should exist
        assert cache_path.exists()

        # Temp file should not exist
        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        assert not temp_path.exists()


class TestFileSizeLimit:
    """Test file size limit enforcement."""

    def test_file_size_limit_enforced(self, cache_config, temp_cache_dir):
        """Test that files exceeding max_item_size are rejected."""
        # Create config with small size limit (1KB)
        small_config = CacheConfig(
            enabled=True,
            cache_dir=temp_cache_dir,
            max_item_size=1024,  # 1KB limit
        )
        manager = CacheManager(
            bundle_path="gs://test/bundle",
            config=small_config,
        )

        # Create data larger than limit (2KB)
        large_data = b"X" * 2048

        def mock_fetch():
            return (large_data, "included_table", "test.parquet")

        # Should raise ValueError for oversized item
        with pytest.raises(ValueError, match="exceeds maximum cache size"):
            manager.get("large_item", remote_fetch_fn=mock_fetch)

    def test_file_within_size_limit_cached(self, cache_config, temp_cache_dir):
        """Test that files within max_item_size are cached normally."""
        # Create config with size limit
        config = CacheConfig(
            enabled=True,
            cache_dir=temp_cache_dir,
            max_item_size=10 * 1024,  # 10KB limit
        )
        manager = CacheManager(
            bundle_path="gs://test/bundle",
            config=config,
        )

        # Create data smaller than limit (1KB)
        small_data = b"X" * 1024

        def mock_fetch():
            return (small_data, "included_table", "test.parquet")

        # Should cache successfully
        cache_path = manager.get("small_item", remote_fetch_fn=mock_fetch)
        assert cache_path is not None
        assert cache_path.exists()
        assert cache_path.read_bytes() == small_data
