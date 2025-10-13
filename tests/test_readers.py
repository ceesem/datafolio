"""Tests for data readers."""

from pathlib import Path

import pandas as pd
import pytest

from datafolio.readers import read_csv, read_parquet, read_table


class TestReadParquet:
    """Tests for read_parquet function."""

    def test_read_local_parquet(self, tmp_path):
        """Test reading a local Parquet file."""
        # Create test data
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        parquet_file = tmp_path / "test.parquet"
        df.to_parquet(parquet_file, index=False)

        # Read it back
        loaded_df = read_parquet(parquet_file)

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_read_parquet_with_kwargs(self, tmp_path):
        """Test reading Parquet with additional kwargs."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        parquet_file = tmp_path / "test.parquet"
        df.to_parquet(parquet_file, index=False)

        # Read specific columns
        loaded_df = read_parquet(parquet_file, columns=["a"])

        assert list(loaded_df.columns) == ["a"]
        assert len(loaded_df) == 3

    def test_read_parquet_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            read_parquet("/nonexistent/file.parquet")

    def test_read_parquet_from_path_object(self, tmp_path):
        """Test reading from Path object."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        parquet_file = tmp_path / "test.parquet"
        df.to_parquet(parquet_file, index=False)

        # Read using Path object
        loaded_df = read_parquet(Path(parquet_file))

        pd.testing.assert_frame_equal(df, loaded_df)


class TestReadCSV:
    """Tests for read_csv function."""

    def test_read_local_csv(self, tmp_path):
        """Test reading a local CSV file."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        csv_file = tmp_path / "test.csv"
        df.to_csv(csv_file, index=False)

        # Read it back
        loaded_df = read_csv(csv_file)

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_read_csv_with_kwargs(self, tmp_path):
        """Test reading CSV with additional kwargs."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        csv_file = tmp_path / "test.csv"
        df.to_csv(csv_file, index=False)

        # Read with specific dtype
        loaded_df = read_csv(csv_file, dtype={"a": float})

        assert loaded_df["a"].dtype == float

    def test_read_csv_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            read_csv("/nonexistent/file.csv")


# Delta table support is not implemented yet
# class TestReadDelta:
#     """Tests for read_delta function."""
#
#     def test_read_delta_missing_dependency(self):
#         """Test error when deltalake is not installed."""
#         # This test assumes deltalake might not be installed
#         # We'll check if the ImportError is raised with the right message
#         try:
#             read_delta('/fake/path')
#         except ImportError as e:
#             assert 'deltalake is required' in str(e)
#             assert 'datafolio[delta]' in str(e)
#         except (FileNotFoundError, IOError):
#             # If deltalake IS installed, we expect file not found
#             pass


class TestReadTable:
    """Tests for read_table dispatcher function."""

    def test_read_table_parquet(self, tmp_path):
        """Test reading Parquet via read_table."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        parquet_file = tmp_path / "test.parquet"
        df.to_parquet(parquet_file, index=False)

        loaded_df = read_table(parquet_file, "parquet")

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_read_table_csv(self, tmp_path):
        """Test reading CSV via read_table."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        csv_file = tmp_path / "test.csv"
        df.to_csv(csv_file, index=False)

        loaded_df = read_table(csv_file, "csv")

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_read_table_invalid_format(self, tmp_path):
        """Test error with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported table format"):
            read_table("/path/file.txt", "txt")

    def test_read_table_with_kwargs(self, tmp_path):
        """Test passing kwargs through read_table."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        parquet_file = tmp_path / "test.parquet"
        df.to_parquet(parquet_file, index=False)

        # Read specific columns
        loaded_df = read_table(parquet_file, "parquet", columns=["a"])

        assert list(loaded_df.columns) == ["a"]


class TestReadReferenced:
    """Integration tests for reading referenced data."""

    def test_read_referenced_parquet_with_folio(self, tmp_path):
        """Test reading referenced Parquet through DataFolio."""
        from datafolio import DataFolio

        # Create test data
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})
        parquet_file = tmp_path / "data.parquet"
        df.to_parquet(parquet_file, index=False)

        # Create folio with reference
        folio = DataFolio(tmp_path / "test")
        folio.reference_table(
            "test_data", path=str(parquet_file), table_format="parquet", num_rows=5
        )

        # Read the referenced data
        loaded_df = folio.get_table("test_data")

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_read_referenced_csv_with_folio(self, tmp_path):
        """Test reading referenced CSV through DataFolio."""
        from datafolio import DataFolio

        # Create test data
        df = pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})
        csv_file = tmp_path / "data.csv"
        df.to_csv(csv_file, index=False)

        # Create folio with reference
        folio = DataFolio(tmp_path / "test")
        folio.reference_table("test_csv", path=str(csv_file), table_format="csv")

        # Read the referenced data
        loaded_df = folio.get_table("test_csv")

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_read_referenced_data_after_save_load(self, tmp_path):
        """Test reading referenced data through create/load cycle."""
        from datafolio import DataFolio

        # Create test data
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data_file = tmp_path / "data.parquet"
        df.to_parquet(data_file, index=False)

        # Create folio
        folio = DataFolio(tmp_path / "test", metadata={"test": "reference"})
        bundle_path = folio._bundle_dir
        folio.reference_table(
            "external_data", path=str(data_file), table_format="parquet"
        )

        # Load folio from directory
        loaded_folio = DataFolio(path=bundle_path)
        loaded_df = loaded_folio.get_table("external_data")

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_referenced_data_not_cached(self, tmp_path):
        """Test that referenced data is read fresh each time (not cached)."""
        from datafolio import DataFolio

        df = pd.DataFrame({"a": [1, 2, 3]})
        data_file = tmp_path / "data.parquet"
        df.to_parquet(data_file, index=False)

        folio = DataFolio(tmp_path / "test")
        folio.reference_table("data", path=str(data_file), table_format="parquet")

        # Read the data twice - should work without caching
        df1 = folio.get_table("data")
        df2 = folio.get_table("data")

        # Both reads should succeed and return the same data
        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_frame_equal(df1, df)
