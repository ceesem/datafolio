"""Unit tests for PolarsHandler.

These tests directly test the PolarsHandler class methods to ensure
full coverage of the handler implementation.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from datafolio.handlers.tables import PolarsHandler

# Try to import Polars but don't skip all tests if not available
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None


@pytest.fixture
def handler():
    """Create a PolarsHandler instance."""
    return PolarsHandler()


@pytest.fixture
def mock_folio(tmp_path):
    """Create a mock DataFolio instance."""
    folio = Mock()
    folio._bundle_dir = str(tmp_path / "bundle")
    folio._storage = Mock()
    folio._items = {}
    return folio


@pytest.fixture
def simple_dataframe():
    """Create a simple Polars DataFrame."""
    if not HAS_POLARS:
        pytest.skip("Polars not installed")

    return pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})


# ============================================================================
# Test item_type property
# ============================================================================


def test_item_type(handler):
    """Test that item_type returns correct identifier."""
    assert handler.item_type == "polars_table"


# ============================================================================
# Test can_handle
# ============================================================================


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_can_handle_with_polars_dataframe(handler, simple_dataframe):
    """Test can_handle returns True for Polars DataFrames."""
    assert handler.can_handle(simple_dataframe) is True


def test_can_handle_with_non_dataframe(handler):
    """Test can_handle returns False for non-Polars objects."""
    assert handler.can_handle("not a dataframe") is False
    assert handler.can_handle(123) is False
    assert handler.can_handle([1, 2, 3]) is False
    assert handler.can_handle({"key": "value"}) is False


@pytest.mark.skipif(HAS_POLARS, reason="Skip when Polars is installed")
def test_can_handle_without_polars_installed(handler):
    """Test can_handle returns False when Polars is not installed."""
    # When polars is not installed, can_handle should return False
    result = handler.can_handle("anything")
    assert result is False


def test_can_handle_with_pandas_dataframe(handler):
    """Test can_handle returns False for pandas DataFrames."""
    # Import pandas if available
    try:
        import pandas as pd

        pandas_df = pd.DataFrame({"a": [1, 2, 3]})
        assert handler.can_handle(pandas_df) is False
    except ImportError:
        pytest.skip("pandas not installed")


# ============================================================================
# Test add method
# ============================================================================


def test_add_basic(handler, mock_folio, simple_dataframe):
    """Test basic add operation."""
    import pandas as pd

    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_parquet = Mock()
    mock_folio._storage.calculate_checksum = Mock(return_value="abc123")

    metadata = handler.add(
        folio=mock_folio,
        name="test_data",
        data=simple_dataframe,
        description=None,
        inputs=None,
    )

    # Verify write_parquet was called
    assert mock_folio._storage.write_parquet.called
    # Verify it was called with a pandas DataFrame (conversion)
    call_args = mock_folio._storage.write_parquet.call_args
    assert isinstance(call_args[0][1], pd.DataFrame)

    # Verify metadata structure
    assert metadata["name"] == "test_data"
    assert metadata["item_type"] == "polars_table"
    assert metadata["filename"] == "test_data.parquet"
    assert metadata["dataframe_library"] == "polars"
    assert metadata["num_rows"] == 3
    assert metadata["num_cols"] == 3
    assert metadata["columns"] == ["a", "b", "c"]
    assert "dtypes" in metadata
    assert "created_at" in metadata
    assert metadata["checksum"] == "abc123"


def test_add_with_description(handler, mock_folio, simple_dataframe):
    """Test add with description."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_parquet = Mock()
    mock_folio._storage.calculate_checksum = Mock(return_value="abc123")

    metadata = handler.add(
        folio=mock_folio,
        name="test_data",
        data=simple_dataframe,
        description="Test description",
        inputs=None,
    )

    assert metadata["description"] == "Test description"


def test_add_with_inputs(handler, mock_folio, simple_dataframe):
    """Test add with lineage inputs."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_parquet = Mock()
    mock_folio._storage.calculate_checksum = Mock(return_value="abc123")

    metadata = handler.add(
        folio=mock_folio,
        name="test_data",
        data=simple_dataframe,
        description=None,
        inputs=["input1", "input2"],
    )

    assert metadata["inputs"] == ["input1", "input2"]


def test_add_with_both_description_and_inputs(handler, mock_folio, simple_dataframe):
    """Test add with both description and inputs."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_parquet = Mock()
    mock_folio._storage.calculate_checksum = Mock(return_value="abc123")

    metadata = handler.add(
        folio=mock_folio,
        name="data",
        data=simple_dataframe,
        description="My data",
        inputs=["data1", "data2"],
    )

    assert metadata["description"] == "My data"
    assert metadata["inputs"] == ["data1", "data2"]


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_add_non_dataframe_raises_error(handler, mock_folio):
    """Test add raises TypeError for non-Polars objects."""
    not_a_dataframe = "not a Polars DataFrame"

    with pytest.raises(TypeError, match="Expected Polars DataFrame"):
        handler.add(
            folio=mock_folio,
            name="test",
            data=not_a_dataframe,
            description=None,
            inputs=None,
        )


def test_add_creates_correct_filepath(handler, mock_folio, simple_dataframe):
    """Test add creates filepath in correct subdirectory."""
    mock_folio._bundle_dir = "/path/to/bundle"
    mock_folio._storage.join_paths = Mock(
        return_value="/path/to/bundle/tables/test.parquet"
    )
    mock_folio._storage.write_parquet = Mock()
    mock_folio._storage.calculate_checksum = Mock(return_value="abc123")

    handler.add(
        folio=mock_folio,
        name="test",
        data=simple_dataframe,
        description=None,
        inputs=None,
    )

    # Verify join_paths was called with correct arguments
    mock_folio._storage.join_paths.assert_called_once_with(
        "/path/to/bundle", "tables", "test.parquet"
    )


def test_add_timestamp_is_utc(handler, mock_folio, simple_dataframe):
    """Test add sets created_at timestamp in UTC."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_parquet = Mock()
    mock_folio._storage.calculate_checksum = Mock(return_value="abc123")

    metadata = handler.add(
        folio=mock_folio,
        name="test",
        data=simple_dataframe,
        description=None,
        inputs=None,
    )

    # Parse the ISO timestamp
    created_at_str = metadata["created_at"]
    created_at = datetime.fromisoformat(created_at_str)

    # Should end with +00:00 or Z for UTC
    assert created_at_str.endswith("+00:00") or created_at_str.endswith("Z")


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_add_captures_dtypes(handler, mock_folio):
    """Test that dtypes are captured correctly."""
    df = pl.DataFrame(
        {"int_col": [1, 2, 3], "float_col": [1.0, 2.0, 3.0], "str_col": ["a", "b", "c"]}
    )

    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_parquet = Mock()
    mock_folio._storage.calculate_checksum = Mock(return_value="abc123")

    metadata = handler.add(
        folio=mock_folio, name="typed_data", data=df, description=None, inputs=None
    )

    # Check that dtypes are captured
    assert "int_col" in metadata["dtypes"]
    assert "float_col" in metadata["dtypes"]
    assert "str_col" in metadata["dtypes"]


# ============================================================================
# Test get method
# ============================================================================


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_get_basic(handler, mock_folio, simple_dataframe):
    """Test basic get operation."""
    import pandas as pd

    # Setup mock - simulate stored pandas dataframe
    mock_folio._items = {"test": {"filename": "test.parquet"}}
    mock_folio._storage.join_paths = lambda *args: "/".join(args)

    # Create a pandas version for the mock return
    pandas_df = simple_dataframe.to_pandas()
    mock_folio._storage.read_parquet = Mock(return_value=pandas_df)

    result = handler.get(folio=mock_folio, name="test")

    # Verify storage.read_parquet was called
    assert mock_folio._storage.read_parquet.called
    # Verify result is a Polars DataFrame
    assert isinstance(result, pl.DataFrame)
    # Verify data is correct (compare as pandas for simplicity)
    assert result.to_pandas().equals(pandas_df)


def test_get_nonexistent_raises_error(handler, mock_folio):
    """Test get raises KeyError for non-existent item."""
    mock_folio._items = {}

    with pytest.raises(KeyError):
        handler.get(folio=mock_folio, name="nonexistent")


def test_get_creates_correct_filepath(handler, mock_folio):
    """Test get creates filepath from metadata."""
    import pandas as pd

    mock_folio._items = {"test": {"filename": "test_data.parquet"}}
    mock_folio._bundle_dir = "/bundle"
    mock_folio._storage.join_paths = Mock(
        return_value="/bundle/tables/test_data.parquet"
    )
    mock_folio._storage.read_parquet = Mock(return_value=pd.DataFrame({"a": [1, 2, 3]}))

    handler.get(folio=mock_folio, name="test")

    # Verify join_paths was called correctly
    mock_folio._storage.join_paths.assert_called_once_with(
        "/bundle", "tables", "test_data.parquet"
    )


# ============================================================================
# Test get_storage_subdir
# ============================================================================


def test_get_storage_subdir(handler):
    """Test that get_storage_subdir returns correct subdirectory."""
    subdir = handler.get_storage_subdir()

    # Polars tables should be in "tables" subdirectory
    # (item_type='polars_table' maps to TABLES category)
    assert subdir == "tables"


# ============================================================================
# Test delete method (inherited from BaseHandler)
# ============================================================================


def test_delete_removes_file(handler, mock_folio):
    """Test delete method removes file."""
    mock_folio._items = {"test": {"filename": "test.parquet"}}
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.delete_file = Mock()

    handler.delete(folio=mock_folio, name="test")

    # Verify delete_file was called
    assert mock_folio._storage.delete_file.called


# ============================================================================
# Edge cases and error conditions
# ============================================================================


def test_add_with_extra_kwargs(handler, mock_folio, simple_dataframe):
    """Test add ignores extra kwargs."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_parquet = Mock()
    mock_folio._storage.calculate_checksum = Mock(return_value="abc123")

    # Should not raise error with extra kwargs
    metadata = handler.add(
        folio=mock_folio,
        name="test",
        data=simple_dataframe,
        description=None,
        inputs=None,
        extra_param="ignored",
        another_param=123,
    )

    assert metadata["name"] == "test"


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_handler_instance_reuse(handler, mock_folio, simple_dataframe):
    """Test that handler instance can be reused for multiple operations."""
    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_parquet = Mock()
    mock_folio._storage.calculate_checksum = Mock(return_value="abc123")

    # Add multiple dataframes with same handler instance
    metadata1 = handler.add(
        folio=mock_folio,
        name="data1",
        data=simple_dataframe,
        description=None,
        inputs=None,
    )
    metadata2 = handler.add(
        folio=mock_folio,
        name="data2",
        data=simple_dataframe,
        description=None,
        inputs=None,
    )

    assert metadata1["name"] == "data1"
    assert metadata2["name"] == "data2"
    assert mock_folio._storage.write_parquet.call_count == 2


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_add_empty_dataframe(handler, mock_folio):
    """Test add with empty Polars DataFrame."""
    empty_df = pl.DataFrame({"a": [], "b": []})

    mock_folio._storage.join_paths = lambda *args: "/".join(args)
    mock_folio._storage.write_parquet = Mock()
    mock_folio._storage.calculate_checksum = Mock(return_value="abc123")

    metadata = handler.add(
        folio=mock_folio,
        name="empty",
        data=empty_df,
        description=None,
        inputs=None,
    )

    assert metadata["num_rows"] == 0
    assert metadata["num_cols"] == 2
    assert metadata["columns"] == ["a", "b"]
