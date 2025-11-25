"""Tests for table utility methods: get_table_path, table_info, preview_table."""

import pytest

from datafolio import DataFolio

# Try to import dependencies
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pyarrow

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
class TestGetTablePath:
    """Tests for get_table_path() method."""

    def test_get_table_path_basic(self, tmp_path):
        """Test getting path to a table."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": [1, 2, 3]})
        folio.add_table("data", df)

        path = folio.get_table_path("data")

        assert isinstance(path, str)
        assert "tables" in path
        assert path.endswith(".parquet")

    def test_get_table_path_nonexistent(self, tmp_path):
        """Test error when table doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_table_path("nonexistent")

    def test_get_table_path_referenced_table(self, tmp_path):
        """Test error when trying to get path of referenced table."""
        folio = DataFolio(tmp_path / "test")

        # Create temp parquet file
        temp_file = tmp_path / "external.parquet"
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.to_parquet(temp_file)

        folio.reference_table("external", str(temp_file))

        # Should error - use get_data_path instead
        with pytest.raises(ValueError, match="referenced table"):
            folio.get_table_path("external")

    def test_get_table_path_non_table(self, tmp_path):
        """Test error when item is not a table."""
        folio = DataFolio(tmp_path / "test")

        folio.add_json("config", {"key": "value"})

        with pytest.raises(ValueError, match="not a table"):
            folio.get_table_path("config")

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_get_table_path_polars(self, tmp_path):
        """Test getting path to a Polars table."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"x": [1, 2, 3]})
        folio.add_table("data", df)

        path = folio.get_table_path("data")

        assert isinstance(path, str)
        assert "tables" in path
        assert path.endswith(".parquet")

        # Verify file exists
        import os

        assert os.path.exists(path)


@pytest.mark.skipif(
    not HAS_PANDAS or not HAS_PYARROW, reason="pandas and pyarrow required"
)
class TestTableInfo:
    """Tests for table_info() method."""

    def test_table_info_basic(self, tmp_path):
        """Test getting table info."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": range(100), "b": range(100, 200), "c": ["x"] * 100})
        folio.add_table("data", df)

        info = folio.table_info("data")

        assert info["num_rows"] == 100
        assert info["num_columns"] == 3
        assert "size_bytes" in info
        assert "size_mb" in info
        assert info["size_mb"] > 0
        assert info["columns"] == ["a", "b", "c"]
        assert "schema" in info

    def test_table_info_large_table(self, tmp_path):
        """Test table info for larger table."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(10000), "data": ["test" * 100] * 10000})
        folio.add_table("large", df)

        info = folio.table_info("large")

        assert info["num_rows"] == 10000
        assert info["num_columns"] == 2
        assert info["size_mb"] > 0

    def test_table_info_nonexistent(self, tmp_path):
        """Test error when table doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.table_info("nonexistent")

    def test_table_info_non_table(self, tmp_path):
        """Test error when item is not a table."""
        folio = DataFolio(tmp_path / "test")

        folio.add_json("config", {"key": "value"})

        with pytest.raises(ValueError, match="must be an included table"):
            folio.table_info("config")

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_table_info_polars(self, tmp_path):
        """Test table info for Polars table."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"x": range(50), "y": range(50, 100)})
        folio.add_table("data", df)

        info = folio.table_info("data")

        assert info["num_rows"] == 50
        assert info["num_columns"] == 2
        assert info["columns"] == ["x", "y"]


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
class TestPreviewTable:
    """Tests for preview_table() method."""

    def test_preview_table_basic(self, tmp_path):
        """Test basic table preview."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(100)})
        folio.add_table("data", df)

        preview = folio.preview_table("data", n=10)

        assert isinstance(preview, pd.DataFrame)
        assert len(preview) == 10
        assert preview["value"].tolist() == list(range(10))

    def test_preview_table_default_n(self, tmp_path):
        """Test preview with default n=10."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(100)})
        folio.add_table("data", df)

        preview = folio.preview_table("data")

        assert len(preview) == 10

    def test_preview_table_larger_than_table(self, tmp_path):
        """Test preview when n > table size."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(5)})
        folio.add_table("data", df)

        preview = folio.preview_table("data", n=100)

        # Should only get 5 rows
        assert len(preview) == 5

    def test_preview_table_nonexistent(self, tmp_path):
        """Test error when table doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.preview_table("nonexistent")

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_preview_table_polars(self, tmp_path):
        """Test preview of Polars table."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"x": range(50)})
        folio.add_table("data", df)

        preview = folio.preview_table("data", n=5)

        # Should return Polars DataFrame (type preservation)
        assert isinstance(preview, pl.DataFrame)
        assert len(preview) == 5

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_preview_table_with_return_type(self, tmp_path):
        """Test preview with explicit return type."""
        folio = DataFolio(tmp_path / "test")

        # Add pandas table
        df_pandas = pd.DataFrame({"x": range(50)})
        folio.add_table("pandas_data", df_pandas)

        # Preview as Polars
        preview = folio.preview_table("pandas_data", n=5, return_type="polars")

        assert isinstance(preview, pl.DataFrame)
        assert len(preview) == 5

        # Preview as pandas
        preview = folio.preview_table("pandas_data", n=5, return_type="pandas")

        assert isinstance(preview, pd.DataFrame)
        assert len(preview) == 5


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
class TestTableUtilsIntegration:
    """Integration tests for table utility methods."""

    def test_workflow_check_before_load(self, tmp_path):
        """Test workflow: check size, then decide how to load."""
        folio = DataFolio(tmp_path / "test")

        # Add moderate-sized table
        df = pd.DataFrame({"value": range(1000), "data": ["test"] * 1000})
        folio.add_table("data", df)

        # Check info first
        if HAS_PYARROW:
            info = folio.table_info("data")
            assert info["num_rows"] == 1000

        # Preview to check schema
        preview = folio.preview_table("data", n=5)
        assert len(preview) == 5
        assert list(preview.columns) == ["value", "data"]

        # Get full table
        full = folio.get_table("data")
        assert len(full) == 1000

    def test_workflow_external_tool_integration(self, tmp_path):
        """Test workflow: get path for external tools."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        folio.add_table("data", df)

        # Get path for external tool
        path = folio.get_table_path("data")

        # Simulate external tool reading directly
        df_external = pd.read_parquet(path)
        assert df_external.equals(df)

        if HAS_POLARS:
            # Polars can also read it
            df_polars = pl.read_parquet(path)
            assert df_polars.to_pandas().equals(df)

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow required")
    def test_workflow_decide_processing_strategy(self, tmp_path):
        """Test workflow: use table_info to decide processing strategy."""
        folio = DataFolio(tmp_path / "test")

        # Small table
        df_small = pd.DataFrame({"x": range(10)})
        folio.add_table("small", df_small)

        # Large table
        df_large = pd.DataFrame({"value": range(100000)})
        folio.add_table("large", df_large)

        # Check sizes
        info_small = folio.table_info("small")
        info_large = folio.table_info("large")

        # Small table: load directly
        if info_small["num_rows"] < 1000:
            data = folio.get_table("small")
            assert len(data) == 10

        # Large table: preview first, then decide
        preview = folio.preview_table("large", n=10)
        assert len(preview) == 10

        # For actual processing, would use chunked iteration
        # (not testing full iteration here to keep test fast)
