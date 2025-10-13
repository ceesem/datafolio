"""Tests for utility functions."""

from pathlib import Path

import pytest

from datafolio.utils import (
    get_file_extension,
    is_cloud_path,
    resolve_path,
    validate_table_format,
)


class TestCloudPathDetection:
    """Tests for cloud path detection."""

    def test_s3_path(self):
        """Test S3 path detection."""
        assert is_cloud_path("s3://bucket/file.parquet") is True

    def test_gs_path(self):
        """Test Google Cloud Storage path detection."""
        assert is_cloud_path("gs://bucket/file.parquet") is True

    def test_gcs_path(self):
        """Test GCS path detection."""
        assert is_cloud_path("gcs://bucket/file.parquet") is True

    def test_azure_path(self):
        """Test Azure path detection."""
        assert is_cloud_path("az://container/file.parquet") is True
        assert is_cloud_path("azure://container/file.parquet") is True

    def test_https_path(self):
        """Test HTTPS path detection."""
        assert is_cloud_path("https://example.com/file.parquet") is True

    def test_local_absolute_path(self):
        """Test local absolute path is not cloud."""
        assert is_cloud_path("/local/path/file.parquet") is False

    def test_local_relative_path(self):
        """Test local relative path is not cloud."""
        assert is_cloud_path("data/file.parquet") is False

    def test_path_object(self):
        """Test Path object detection."""
        assert is_cloud_path(Path("/local/path")) is False


class TestPathResolution:
    """Tests for path resolution."""

    def test_cloud_path_unchanged(self):
        """Test cloud paths are returned as-is."""
        cloud_path = "s3://bucket/file.parquet"
        assert resolve_path(cloud_path) == cloud_path

    def test_absolute_local_path(self):
        """Test absolute local paths get file:// prefix."""
        abs_path = "/absolute/path/file.csv"
        result = resolve_path(abs_path)
        assert result.startswith("file://")
        assert "/absolute/path/file.csv" in result

    def test_relative_path_to_absolute(self):
        """Test converting relative to absolute path."""
        result = resolve_path("data/file.csv", make_absolute=True)
        assert result.startswith("file://")
        assert "data/file.csv" in result

    def test_path_with_base_dir(self):
        """Test path resolution with base directory."""
        result = resolve_path("file.csv", base_dir="/base/dir", make_absolute=True)
        assert "/base/dir" in result
        assert result.endswith("file.csv")


class TestTableFormatValidation:
    """Tests for table format validation."""

    def test_valid_parquet(self):
        """Test parquet format is valid."""
        validate_table_format("parquet")  # Should not raise

    def test_valid_delta(self):
        """Test delta format is valid."""
        validate_table_format("delta")  # Should not raise

    def test_valid_csv(self):
        """Test CSV format is valid."""
        validate_table_format("csv")  # Should not raise

    def test_invalid_format(self):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported table format"):
            validate_table_format("invalid_format")


class TestFileExtension:
    """Tests for file extension mapping."""

    def test_parquet_extension(self):
        """Test parquet extension."""
        assert get_file_extension("parquet") == ".parquet"

    def test_csv_extension(self):
        """Test CSV extension."""
        assert get_file_extension("csv") == ".csv"

    def test_delta_extension(self):
        """Test Delta Lake extension (none, it's a directory)."""
        assert get_file_extension("delta") == ""
