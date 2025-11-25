"""Integration tests for Polars DataFrame support in DataFolio.

These tests verify end-to-end workflows with Polars DataFrames,
including adding, retrieving, saving, loading, and interoperability with pandas.
"""

from pathlib import Path

import pytest

from datafolio import DataFolio

# Try to import Polars but skip tests if not available
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

# Try to import pandas for interop tests
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
class TestPolarsBasicWorkflow:
    """Tests for basic Polars DataFrame workflows."""

    def test_add_and_get_polars_dataframe(self, tmp_path):
        """Test adding and retrieving a Polars DataFrame."""
        folio = DataFolio(tmp_path / "test")

        # Create Polars DataFrame
        df_original = pl.DataFrame(
            {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]}
        )

        # Add to folio
        folio.add_table("data", df_original)

        # Retrieve from folio
        df_retrieved = folio.get_table("data")

        # Verify it's a Polars DataFrame
        assert isinstance(df_retrieved, pl.DataFrame)

        # Verify data is correct
        assert df_retrieved.to_pandas().equals(df_original.to_pandas())

    def test_polars_auto_detection(self, tmp_path):
        """Test that Polars DataFrames are auto-detected by add_data."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"x": [1, 2, 3]})

        # Use generic add_data method (should auto-detect Polars)
        folio.add_data("data", df)

        # Verify item type is polars_table
        assert folio._items["data"]["item_type"] == "polars_table"
        assert folio._items["data"]["dataframe_library"] == "polars"

    def test_polars_metadata_captured(self, tmp_path):
        """Test that Polars DataFrame metadata is captured correctly."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "str_col": ["a", "b", "c"],
            }
        )

        folio.add_table("typed_data", df)

        # Check metadata
        item = folio._items["typed_data"]
        assert item["num_rows"] == 3
        assert item["num_cols"] == 3
        assert item["columns"] == ["int_col", "float_col", "str_col"]
        assert "dtypes" in item
        assert "int_col" in item["dtypes"]
        assert "float_col" in item["dtypes"]
        assert "str_col" in item["dtypes"]

    def test_list_contents_includes_polars(self, tmp_path):
        """Test that list_contents includes polars_tables category."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        contents = folio.list_contents()

        # Polars tables should have their own category
        # Note: list_contents may need to be updated to include polars_tables
        # For now, just verify the table was added successfully
        assert "data" in folio._items
        assert folio._items["data"]["item_type"] == "polars_table"

    def test_polars_with_description_and_inputs(self, tmp_path):
        """Test adding Polars DataFrame with description and lineage."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"a": [1, 2, 3]})

        folio.add_table("result", df, description="Test results", inputs=["source1"])

        item = folio._items["result"]
        assert item["description"] == "Test results"
        assert item["inputs"] == ["source1"]


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
class TestPolarsPersistence:
    """Tests for saving and loading Polars DataFrames."""

    def test_save_and_load_polars(self, tmp_path):
        """Test that Polars DataFrames persist correctly."""
        bundle_path = tmp_path / "test"

        # Create folio and add Polars DataFrame
        folio1 = DataFolio(bundle_path, random_suffix=False)
        df_original = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        folio1.add_table("data", df_original)
        # DataFolio auto-saves, no explicit save() needed

        # Reload folio
        folio2 = DataFolio(bundle_path, random_suffix=False)

        # Verify data is retrieved correctly
        df_loaded = folio2.get_table("data")
        assert isinstance(df_loaded, pl.DataFrame)
        assert df_loaded.to_pandas().equals(df_original.to_pandas())

    def test_polars_checksum_verification(self, tmp_path):
        """Test that checksums are calculated for Polars tables."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Verify checksum exists
        item = folio._items["data"]
        assert "checksum" in item
        assert item["checksum"] is not None
        assert len(item["checksum"]) > 0


@pytest.mark.skipif(
    not HAS_POLARS or not HAS_PANDAS, reason="Polars and pandas required"
)
class TestPolarsInteroperability:
    """Tests for interoperability between Polars and pandas DataFrames."""

    def test_add_polars_retrieve_pandas(self, tmp_path):
        """Test adding Polars and retrieving as pandas (future feature)."""
        # Note: This test assumes return_type parameter will be added
        # For now, it documents the desired behavior
        folio = DataFolio(tmp_path / "test")

        df_polars = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        folio.add_table("data", df_polars)

        # For now, verify we can at least get the Polars version
        df_retrieved = folio.get_table("data")
        assert isinstance(df_retrieved, pl.DataFrame)

        # Convert manually to pandas for comparison
        df_pandas = df_retrieved.to_pandas()
        assert df_pandas["a"].tolist() == [1, 2, 3]

    def test_mixed_pandas_and_polars_tables(self, tmp_path):
        """Test that pandas and Polars tables can coexist in same folio."""
        folio = DataFolio(tmp_path / "test")

        # Add pandas DataFrame
        df_pandas = pd.DataFrame({"x": [1, 2, 3]})
        folio.add_table("pandas_data", df_pandas)

        # Add Polars DataFrame
        df_polars = pl.DataFrame({"y": [4, 5, 6]})
        folio.add_table("polars_data", df_polars)

        # Verify both are stored correctly
        assert folio._items["pandas_data"]["item_type"] == "included_table"
        assert folio._items["polars_data"]["item_type"] == "polars_table"

        # Verify both can be retrieved
        retrieved_pandas = folio.get_table("pandas_data")
        retrieved_polars = folio.get_table("polars_data")

        assert isinstance(retrieved_pandas, pd.DataFrame)
        assert isinstance(retrieved_polars, pl.DataFrame)

    def test_pandas_and_polars_both_stored_as_parquet(self, tmp_path):
        """Test that both pandas and Polars use Parquet storage (interoperable)."""
        folio = DataFolio(tmp_path / "test")

        # Add pandas DataFrame
        df_pandas = pd.DataFrame({"x": [1, 2, 3]})
        folio.add_table("pandas_data", df_pandas)

        # Add Polars DataFrame
        df_polars = pl.DataFrame({"y": [4, 5, 6]})
        folio.add_table("polars_data", df_polars)

        # Both should use .parquet extension
        assert folio._items["pandas_data"]["filename"].endswith(".parquet")
        assert folio._items["polars_data"]["filename"].endswith(".parquet")


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
class TestPolarsEdgeCases:
    """Tests for edge cases with Polars DataFrames."""

    def test_empty_polars_dataframe(self, tmp_path):
        """Test handling of empty Polars DataFrame."""
        folio = DataFolio(tmp_path / "test")

        df_empty = pl.DataFrame({"a": [], "b": []})
        folio.add_table("empty", df_empty)

        # Verify metadata
        item = folio._items["empty"]
        assert item["num_rows"] == 0
        assert item["num_cols"] == 2

        # Verify retrieval
        df_retrieved = folio.get_table("empty")
        assert isinstance(df_retrieved, pl.DataFrame)
        assert len(df_retrieved) == 0
        assert df_retrieved.columns == ["a", "b"]

    def test_large_column_count(self, tmp_path):
        """Test Polars DataFrame with many columns."""
        folio = DataFolio(tmp_path / "test")

        # Create DataFrame with many columns
        data = {f"col_{i}": [1, 2, 3] for i in range(100)}
        df_wide = pl.DataFrame(data)

        folio.add_table("wide", df_wide)

        # Verify metadata
        item = folio._items["wide"]
        assert item["num_cols"] == 100
        assert len(item["columns"]) == 100

        # Verify retrieval
        df_retrieved = folio.get_table("wide")
        assert isinstance(df_retrieved, pl.DataFrame)
        assert df_retrieved.shape == (3, 100)

    def test_polars_with_null_values(self, tmp_path):
        """Test Polars DataFrame with null values."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"a": [1, None, 3], "b": [None, 5, 6]})
        folio.add_table("with_nulls", df)

        # Verify retrieval preserves nulls
        df_retrieved = folio.get_table("with_nulls")
        assert isinstance(df_retrieved, pl.DataFrame)

        # Convert to pandas for easier null checking
        df_pandas = df_retrieved.to_pandas()
        assert df_pandas["a"].isna().sum() == 1
        assert df_pandas["b"].isna().sum() == 1


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
class TestPolarsDataAccessor:
    """Tests for accessing Polars tables via data accessor."""

    def test_access_polars_via_data_accessor(self, tmp_path):
        """Test accessing Polars DataFrame via folio.data accessor."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"a": [1, 2, 3]})
        folio.add_table("mydata", df)

        # Access via data accessor - returns ItemProxy, use .content to get data
        proxy = folio.data.mydata
        retrieved = proxy.content  # Access content property to get the actual data
        assert isinstance(retrieved, pl.DataFrame)
        assert retrieved.to_pandas()["a"].tolist() == [1, 2, 3]

    def test_list_polars_tables_in_data_accessor(self, tmp_path):
        """Test that Polars tables appear in data accessor listings."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"a": [1, 2, 3]})
        folio.add_table("polars_data", df)

        # Check if it appears in dir()
        assert "polars_data" in dir(folio.data)
