"""Comprehensive tests for StorageBackend.

This test suite covers all StorageBackend methods with:
- Local file system operations
- Cloud storage operations (mocked)
- Error handling
- Edge cases
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from datafolio.storage import StorageBackend


@pytest.fixture
def storage():
    """Create a StorageBackend instance."""
    return StorageBackend()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def sample_array():
    """Create a sample numpy array for testing."""
    return np.array([[1, 2], [3, 4]])


@pytest.fixture
def sample_dict():
    """Create a sample dict for JSON testing."""
    return {"key": "value", "number": 42, "list": [1, 2, 3]}


# ============================================================================
# File System Operations - Local
# ============================================================================


class TestFileSystemLocal:
    """Test file system operations with local paths."""

    def test_exists_true(self, storage, temp_dir):
        """Test exists() returns True for existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        assert storage.exists(str(test_file)) is True

    def test_exists_false(self, storage, temp_dir):
        """Test exists() returns False for non-existing file."""
        test_file = temp_dir / "nonexistent.txt"

        assert storage.exists(str(test_file)) is False

    def test_mkdir_creates_directory(self, storage, temp_dir):
        """Test mkdir() creates a new directory."""
        new_dir = temp_dir / "subdir"

        storage.mkdir(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_mkdir_with_parents(self, storage, temp_dir):
        """Test mkdir() creates parent directories."""
        nested_dir = temp_dir / "a" / "b" / "c"

        storage.mkdir(str(nested_dir), parents=True)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_mkdir_exist_ok(self, storage, temp_dir):
        """Test mkdir() doesn't error when directory exists."""
        new_dir = temp_dir / "subdir"
        new_dir.mkdir()

        # Should not raise error
        storage.mkdir(str(new_dir), exist_ok=True)

        assert new_dir.exists()

    def test_join_paths_local(self, storage):
        """Test join_paths() with local paths."""
        result = storage.join_paths("path", "to", "file.txt")

        # Should use Path join logic
        expected = str(Path("path") / "to" / "file.txt")
        assert result == expected

    def test_delete_file_existing(self, storage, temp_dir):
        """Test delete_file() removes existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        storage.delete_file(str(test_file))

        assert not test_file.exists()

    def test_delete_file_nonexistent(self, storage, temp_dir):
        """Test delete_file() handles non-existing file gracefully."""
        test_file = temp_dir / "nonexistent.txt"

        # Should not raise error
        storage.delete_file(str(test_file))

    def test_copy_file_local_to_local(self, storage, temp_dir):
        """Test copy_file() from local to local."""
        src = temp_dir / "source.txt"
        dst = temp_dir / "dest.txt"
        src.write_text("test content")

        storage.copy_file(str(src), str(dst))

        assert dst.exists()
        assert dst.read_text() == "test content"


# ============================================================================
# File System Operations - Cloud (Mocked)
# ============================================================================


class TestFileSystemCloud:
    """Test file system operations with cloud paths."""

    def test_exists_cloud_path_true(self, storage):
        """Test exists() with cloud path that exists."""
        mock_cf = Mock()
        mock_cf.list.return_value = ["file1", "file2"]

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            result = storage.exists("s3://bucket/path")

        assert result is True

    def test_exists_cloud_path_false(self, storage):
        """Test exists() with cloud path that doesn't exist."""
        mock_cf = Mock()
        mock_cf.list.side_effect = Exception("Not found")

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            result = storage.exists("s3://bucket/path")

        assert result is False

    def test_exists_cloud_path_empty(self, storage):
        """Test exists() with cloud path that has no files (empty prefix)."""
        mock_cf = Mock()
        mock_cf.list.return_value = []

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            result = storage.exists("s3://bucket/path")

        assert result is False

    def test_mkdir_cloud_path(self, storage):
        """Test mkdir() with cloud path (should be no-op)."""
        # Should not raise error
        storage.mkdir("s3://bucket/path")

    def test_join_paths_cloud(self, storage):
        """Test join_paths() with cloud paths."""
        result = storage.join_paths("s3://bucket", "path", "file.txt")

        assert result == "s3://bucket/path/file.txt"

    def test_join_paths_cloud_strips_trailing_slashes(self, storage):
        """Test join_paths() strips trailing slashes for cloud paths."""
        result = storage.join_paths("s3://bucket/", "path/", "file.txt")

        assert result == "s3://bucket/path/file.txt"

    def test_delete_file_cloud(self, storage):
        """Test delete_file() with cloud path."""
        mock_cf = Mock()

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            storage.delete_file("s3://bucket/path/file.txt")

        mock_cf.delete.assert_called_once_with("file.txt")

    def test_copy_file_local_to_cloud(self, storage, temp_dir):
        """Test copy_file() from local to cloud."""
        src = temp_dir / "source.txt"
        src.write_text("test content")

        mock_cf = Mock()

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            storage.copy_file(str(src), "s3://bucket/dest.txt")

        # Verify CloudFiles.put was called with correct content
        assert mock_cf.put.called
        call_args = mock_cf.put.call_args
        assert call_args[0][0] == "dest.txt"
        assert call_args[0][1] == b"test content"


# ============================================================================
# JSON I/O
# ============================================================================


class TestJsonIO:
    """Test JSON read/write operations."""

    def test_write_read_json_local(self, storage, temp_dir, sample_dict):
        """Test write_json() and read_json() with local path."""
        json_file = temp_dir / "data.json"

        storage.write_json(str(json_file), sample_dict)

        assert json_file.exists()

        result = storage.read_json(str(json_file))
        assert result == sample_dict

    def test_write_json_with_numpy_values(self, storage, temp_dir):
        """Test write_json() handles numpy values."""
        data = {"array": np.array([1, 2, 3]), "value": np.int64(42)}
        json_file = temp_dir / "numpy_data.json"

        storage.write_json(str(json_file), data)

        result = storage.read_json(str(json_file))
        assert result["array"] == [1, 2, 3]
        assert result["value"] == 42

    def test_write_read_json_cloud(self, storage, sample_dict):
        """Test write_json() and read_json() with cloud path."""
        mock_cf = Mock()
        written_content = None

        def capture_put(filename, content, cache_control=None):
            nonlocal written_content
            written_content = content

        def return_content(filename):
            return written_content

        mock_cf.put = capture_put
        mock_cf.get = return_content

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            storage.write_json("s3://bucket/data.json", sample_dict)
            result = storage.read_json("s3://bucket/data.json")

        assert result == sample_dict

    def test_read_json_nonexistent(self, storage, temp_dir):
        """Test read_json() raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            storage.read_json(str(temp_dir / "nonexistent.json"))


# ============================================================================
# Parquet I/O
# ============================================================================


class TestParquetIO:
    """Test Parquet read/write operations."""

    def test_write_read_parquet_local(self, storage, temp_dir, sample_dataframe):
        """Test write_parquet() and read_parquet() with local path."""
        parquet_file = temp_dir / "data.parquet"

        storage.write_parquet(str(parquet_file), sample_dataframe)

        assert parquet_file.exists()

        result = storage.read_parquet(str(parquet_file))
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_read_parquet_with_columns(self, storage, temp_dir, sample_dataframe):
        """Test read_parquet() with columns parameter."""
        parquet_file = temp_dir / "data.parquet"
        storage.write_parquet(str(parquet_file), sample_dataframe)

        result = storage.read_parquet(str(parquet_file), columns=["a"])

        assert list(result.columns) == ["a"]
        assert len(result) == 3

    def test_read_parquet_nonexistent(self, storage, temp_dir):
        """Test read_parquet() raises error for non-existent file."""
        with pytest.raises(Exception):  # pandas raises various exceptions
            storage.read_parquet(str(temp_dir / "nonexistent.parquet"))


# ============================================================================
# Joblib I/O
# ============================================================================


class TestJoblibIO:
    """Test Joblib read/write operations."""

    def test_write_read_joblib_local(self, storage, temp_dir, sample_dict):
        """Test write_joblib() and read_joblib() with local path."""
        joblib_file = temp_dir / "data.joblib"

        storage.write_joblib(str(joblib_file), sample_dict)

        assert joblib_file.exists()

        result = storage.read_joblib(str(joblib_file))
        assert result == sample_dict

    def test_write_read_joblib_complex_object(self, storage, temp_dir):
        """Test write_joblib() with numpy array."""
        # Use numpy array as a "complex object" since joblib handles it well
        obj = np.array([1, 2, 3, 4, 5])
        joblib_file = temp_dir / "object.joblib"

        storage.write_joblib(str(joblib_file), obj)
        result = storage.read_joblib(str(joblib_file))

        np.testing.assert_array_equal(result, obj)

    def test_write_read_joblib_cloud(self, storage, temp_dir):
        """Test write_joblib() and read_joblib() with cloud path."""
        mock_cf = Mock()
        written_content = None

        def capture_put(filename, content, **kwargs):
            nonlocal written_content
            written_content = content

        def return_content(filename):
            return written_content

        mock_cf.put = capture_put
        mock_cf.get = return_content

        data = {"test": "value"}

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            storage.write_joblib("s3://bucket/data.joblib", data)
            result = storage.read_joblib("s3://bucket/data.joblib")

        assert result == data


# ============================================================================
# PyTorch I/O
# ============================================================================


class TestPyTorchIO:
    """Test PyTorch read/write operations."""

    @pytest.fixture
    def sample_model(self):
        """Create a simple PyTorch model."""
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        return SimpleModel()

    def test_write_read_pytorch_local(self, storage, temp_dir, sample_model):
        """Test write_pytorch() and read_pytorch() with local path."""
        pytest.importorskip("torch")

        pt_file = temp_dir / "model.pt"

        storage.write_pytorch(str(pt_file), sample_model)

        assert pt_file.exists()

        bundle = storage.read_pytorch(str(pt_file))

        assert "state_dict" in bundle
        assert "metadata" in bundle
        assert bundle["metadata"]["model_class"] == "SimpleModel"

    def test_write_pytorch_with_init_args(self, storage, temp_dir, sample_model):
        """Test write_pytorch() with init_args."""
        pytest.importorskip("torch")

        pt_file = temp_dir / "model.pt"
        init_args = {"input_dim": 10, "output_dim": 5}

        storage.write_pytorch(str(pt_file), sample_model, init_args=init_args)

        bundle = storage.read_pytorch(str(pt_file))

        assert bundle["metadata"]["init_args"] == init_args

    def test_write_pytorch_with_optimizer_state(self, storage, temp_dir, sample_model):
        """Test write_pytorch() with optimizer state."""
        pytest.importorskip("torch")
        import torch

        pt_file = temp_dir / "model.pt"
        optimizer = torch.optim.Adam(sample_model.parameters())
        optimizer_state = optimizer.state_dict()

        storage.write_pytorch(
            str(pt_file), sample_model, optimizer_state=optimizer_state
        )

        bundle = storage.read_pytorch(str(pt_file))

        assert "optimizer_state" in bundle

    def test_write_pytorch_with_save_class(self, storage, temp_dir, sample_model):
        """Test write_pytorch() with save_class=True."""
        pytest.importorskip("torch")
        pytest.importorskip("dill")

        pt_file = temp_dir / "model.pt"

        storage.write_pytorch(str(pt_file), sample_model, save_class=True)

        bundle = storage.read_pytorch(str(pt_file))

        assert "serialized_class" in bundle

    def test_write_pytorch_save_class_without_dill(
        self, storage, temp_dir, sample_model
    ):
        """Test write_pytorch() save_class=True raises error without dill."""
        pytest.importorskip("torch")

        pt_file = temp_dir / "model.pt"

        with patch.dict("sys.modules", {"dill": None}):
            with pytest.raises(ImportError, match="dill is required"):
                storage.write_pytorch(str(pt_file), sample_model, save_class=True)

    def test_write_pytorch_without_torch(self, storage, temp_dir):
        """Test write_pytorch() raises error without PyTorch."""
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="PyTorch is required"):
                storage.write_pytorch(str(temp_dir / "model.pt"), None)

    def test_read_pytorch_without_torch(self, storage, temp_dir):
        """Test read_pytorch() raises error without PyTorch."""
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="PyTorch is required"):
                storage.read_pytorch(str(temp_dir / "model.pt"))

    def test_write_read_pytorch_cloud(self, storage, temp_dir, sample_model):
        """Test write_pytorch() and read_pytorch() with cloud path."""
        pytest.importorskip("torch")

        mock_cf = Mock()
        written_content = None

        def capture_put(filename, content, **kwargs):
            nonlocal written_content
            written_content = content

        def return_content(filename):
            return written_content

        mock_cf.put = capture_put
        mock_cf.get = return_content

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            storage.write_pytorch("s3://bucket/model.pt", sample_model)
            bundle = storage.read_pytorch("s3://bucket/model.pt")

        assert "state_dict" in bundle
        assert "metadata" in bundle


# ============================================================================
# Numpy I/O
# ============================================================================


class TestNumpyIO:
    """Test Numpy read/write operations."""

    def test_write_read_numpy_local(self, storage, temp_dir, sample_array):
        """Test write_numpy() and read_numpy() with local path."""
        npy_file = temp_dir / "array.npy"

        storage.write_numpy(str(npy_file), sample_array)

        assert npy_file.exists()

        result = storage.read_numpy(str(npy_file))
        np.testing.assert_array_equal(result, sample_array)

    def test_write_read_numpy_1d(self, storage, temp_dir):
        """Test write_numpy() and read_numpy() with 1D array."""
        arr = np.array([1, 2, 3, 4, 5])
        npy_file = temp_dir / "array_1d.npy"

        storage.write_numpy(str(npy_file), arr)
        result = storage.read_numpy(str(npy_file))

        np.testing.assert_array_equal(result, arr)

    def test_write_read_numpy_3d(self, storage, temp_dir):
        """Test write_numpy() and read_numpy() with 3D array."""
        arr = np.random.rand(2, 3, 4)
        npy_file = temp_dir / "array_3d.npy"

        storage.write_numpy(str(npy_file), arr)
        result = storage.read_numpy(str(npy_file))

        np.testing.assert_array_almost_equal(result, arr)

    def test_write_numpy_without_numpy(self, storage, temp_dir):
        """Test write_numpy() raises error without numpy."""
        with patch.dict("sys.modules", {"numpy": None}):
            with pytest.raises(ImportError, match="NumPy is required"):
                storage.write_numpy(str(temp_dir / "array.npy"), None)

    def test_read_numpy_without_numpy(self, storage, temp_dir):
        """Test read_numpy() raises error without numpy."""
        with patch.dict("sys.modules", {"numpy": None}):
            with pytest.raises(ImportError, match="NumPy is required"):
                storage.read_numpy(str(temp_dir / "array.npy"))

    def test_write_read_numpy_cloud(self, storage, sample_array):
        """Test write_numpy() and read_numpy() with cloud path."""
        mock_cf = Mock()
        written_content = None

        def capture_put(filename, content, **kwargs):
            nonlocal written_content
            written_content = content

        def return_content(filename):
            return written_content

        mock_cf.put = capture_put
        mock_cf.get = return_content

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            storage.write_numpy("s3://bucket/array.npy", sample_array)
            result = storage.read_numpy("s3://bucket/array.npy")

        np.testing.assert_array_equal(result, sample_array)


# ============================================================================
# Timestamp I/O
# ============================================================================


class TestTimestampIO:
    """Test Timestamp read/write operations."""

    def test_write_read_timestamp_local(self, storage, temp_dir):
        """Test write_timestamp() and read_timestamp() with local path."""
        ts_file = temp_dir / "timestamp.json"
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)

        storage.write_timestamp(str(ts_file), timestamp)

        assert ts_file.exists()

        result = storage.read_timestamp(str(ts_file))

        assert result == timestamp
        assert result.tzinfo == timezone.utc

    def test_write_timestamp_converts_to_utc(self, storage, temp_dir):
        """Test write_timestamp() converts timezone-aware timestamps to UTC."""
        from datetime import timedelta

        ts_file = temp_dir / "timestamp.json"

        # Create timestamp in a different timezone (UTC+5)
        import datetime as dt

        custom_tz = dt.timezone(timedelta(hours=5))
        timestamp = datetime(2024, 1, 15, 15, 30, 45, tzinfo=custom_tz)

        storage.write_timestamp(str(ts_file), timestamp)
        result = storage.read_timestamp(str(ts_file))

        # Result should be in UTC
        expected_utc = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        assert result == expected_utc

    def test_write_read_timestamp_cloud(self, storage):
        """Test write_timestamp() and read_timestamp() with cloud path."""
        mock_cf = Mock()
        written_content = None

        def capture_put(filename, content, cache_control=None):
            nonlocal written_content
            written_content = content

        def return_content(filename):
            return written_content

        mock_cf.put = capture_put
        mock_cf.get = return_content

        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            storage.write_timestamp("s3://bucket/timestamp.json", timestamp)
            result = storage.read_timestamp("s3://bucket/timestamp.json")

        assert result == timestamp


# ============================================================================
# High-level Methods
# ============================================================================


class TestHighLevelMethods:
    """Test high-level convenience methods."""

    def test_read_table_delegates_to_readers(self, storage, temp_dir, sample_dataframe):
        """Test read_table() delegates to readers.read_table()."""
        parquet_file = temp_dir / "data.parquet"
        sample_dataframe.to_parquet(parquet_file, index=False)

        result = storage.read_table(str(parquet_file), "parquet")

        pd.testing.assert_frame_equal(result, sample_dataframe)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_join_paths_empty_parts(self, storage):
        """Test join_paths() handles empty parts."""
        result = storage.join_paths("path", "", "file.txt")

        assert "file.txt" in result

    def test_join_paths_single_part(self, storage):
        """Test join_paths() with single part."""
        result = storage.join_paths("file.txt")

        assert result == "file.txt"

    def test_delete_file_cloud_no_directory(self, storage):
        """Test delete_file() cloud path without directory component."""
        mock_cf = Mock()

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            storage.delete_file("s3://bucket/file.txt")

        # Should still work
        assert mock_cf.delete.called

    def test_copy_file_cloud_no_directory(self, storage, temp_dir):
        """Test copy_file() to cloud path without directory component."""
        src = temp_dir / "source.txt"
        src.write_text("content")

        mock_cf = Mock()

        with patch("cloudfiles.CloudFiles", return_value=mock_cf):
            storage.copy_file(str(src), "s3://bucket/file.txt")

        assert mock_cf.put.called
