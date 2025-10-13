"""Tests for data handling (tables) functionality."""

import pandas as pd
import pytest

from datafolio import DataFolio


class TestReferenceTable:
    """Tests for reference_table method."""

    def test_reference_table_basic(self, tmp_path):
        """Test basic table referencing."""
        folio = DataFolio(tmp_path / "test")
        folio.reference_table("data1", path="s3://bucket/file.parquet")

        assert "data1" in folio._items
        assert folio._items["data1"]["path"] == "s3://bucket/file.parquet"
        assert folio._items["data1"]["table_format"] == "parquet"
        assert folio._items["data1"]["item_type"] == "referenced_table"

    def test_reference_table_with_metadata(self, tmp_path):
        """Test referencing with full metadata."""
        folio = DataFolio(tmp_path / "test")
        folio.reference_table(
            "data1",
            path="s3://bucket/file.parquet",
            table_format="parquet",
            num_rows=1_000_000,
            description="Raw training data",
        )

        ref = folio._items["data1"]
        assert ref["num_rows"] == 1_000_000
        assert ref["description"] == "Raw training data"
        assert ref["item_type"] == "referenced_table"

    def test_reference_table_delta_with_version(self, tmp_path):
        """Test referencing Delta table with version."""
        folio = DataFolio(tmp_path / "test")
        folio.reference_table(
            "features", path="s3://bucket/features/", table_format="delta", version=3
        )

        ref = folio._items["features"]
        assert ref["table_format"] == "delta"
        assert ref["version"] == 3
        assert ref["item_type"] == "referenced_table"

    def test_reference_table_method_chaining(self, tmp_path):
        """Test method chaining works."""
        folio = DataFolio(tmp_path / "test")
        result = folio.reference_table("data1", path="s3://bucket/file.parquet")

        assert result is folio  # Returns self

    def test_reference_table_duplicate_name(self, tmp_path):
        """Test error when referencing duplicate name."""
        folio = DataFolio(tmp_path / "test")
        folio.reference_table("data1", path="s3://bucket/file.parquet")

        with pytest.raises(ValueError, match="already exists"):
            folio.reference_table("data1", path="s3://bucket/other.parquet")

    def test_reference_table_invalid_format(self, tmp_path):
        """Test error with invalid table format."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(ValueError, match="Unsupported table format"):
            folio.reference_table(
                "data1", path="s3://bucket/file.txt", table_format="txt"
            )

    def test_reference_table_appears_in_list_contents(self, tmp_path):
        """Test referenced table appears in list_contents."""
        folio = DataFolio(tmp_path / "test")
        folio.reference_table("data1", path="s3://bucket/file.parquet")

        contents = folio.list_contents()
        assert "data1" in contents["referenced_tables"]
        assert "data1" not in contents["included_tables"]


class TestAddTable:
    """Tests for add_table method."""

    def test_add_table_basic(self, tmp_path):
        """Test basic table addition."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        folio.add_table("test", df)

        assert "test" in folio._items
        assert folio._items["test"]["item_type"] == "included_table"
        # Verify table can be retrieved
        retrieved = folio.get_table("test")
        pd.testing.assert_frame_equal(df, retrieved)

    def test_add_table_metadata(self, tmp_path):
        """Test table metadata is captured correctly."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        folio.add_table("test", df, description="Test data")

        metadata = folio._items["test"]
        assert metadata["num_rows"] == 3
        assert metadata["num_cols"] == 2
        assert metadata["columns"] == ["a", "b"]
        assert metadata["description"] == "Test data"
        assert metadata["item_type"] == "included_table"
        assert "a" in metadata["dtypes"]

    def test_add_table_method_chaining(self, tmp_path):
        """Test method chaining works."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        result = folio.add_table("test", df)
        assert result is folio

    def test_add_table_duplicate_name(self, tmp_path):
        """Test error when adding duplicate name."""
        folio = DataFolio(tmp_path / "test")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio.add_table("test", df1)

        with pytest.raises(ValueError, match="already exists"):
            folio.add_table("test", df2)

    def test_add_table_conflicts_with_reference(self, tmp_path):
        """Test error when name conflicts with referenced table."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.reference_table("test", path="s3://bucket/file.parquet")

        with pytest.raises(ValueError, match="already exists"):
            folio.add_table("test", df)

    def test_add_table_non_dataframe(self, tmp_path):
        """Test error with non-DataFrame input."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            folio.add_table("test", [1, 2, 3])

    def test_add_table_appears_in_list_contents(self, tmp_path):
        """Test included table appears in list_contents."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("test", df)

        contents = folio.list_contents()
        assert "test" in contents["included_tables"]
        assert "test" not in contents["referenced_tables"]


class TestGetTable:
    """Tests for get_table method."""

    def test_get_table_included(self, tmp_path):
        """Test getting an included table."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        folio.add_table("test", df)

        retrieved = folio.get_table("test")

        # Tables are read fresh from disk, not cached
        pd.testing.assert_frame_equal(retrieved, df)

    def test_get_table_not_found(self, tmp_path):
        """Test error when table doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_table("nonexistent")


class TestGetDataPath:
    """Tests for get_data_path method."""

    def test_get_data_path_basic(self, tmp_path):
        """Test getting path to referenced data."""
        folio = DataFolio(tmp_path / "test")
        folio.reference_table("test", path="s3://bucket/file.parquet")

        path = folio.get_data_path("test")
        assert path == "s3://bucket/file.parquet"

    def test_get_data_path_cloud_path(self, tmp_path):
        """Test cloud path is returned as-is."""
        folio = DataFolio(tmp_path / "test")
        cloud_path = "gs://my-bucket/data.parquet"
        folio.reference_table("test", path=cloud_path)

        path = folio.get_data_path("test")
        assert path == cloud_path

    def test_get_data_path_included_table(self, tmp_path):
        """Test error when trying to get path of included table."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("test", df)

        with pytest.raises(ValueError, match="included in bundle"):
            folio.get_data_path("test")

    def test_get_data_path_not_found(self, tmp_path):
        """Test error when referenced table doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_data_path("nonexistent")


class TestIntegration:
    """Integration tests for data handling."""

    def test_mixed_tables(self, tmp_path):
        """Test using both referenced and included tables."""
        folio = DataFolio(tmp_path / "test")

        # Add referenced table
        folio.reference_table(
            "big_data", path="s3://bucket/big.parquet", num_rows=1_000_000
        )

        # Add included table
        df = pd.DataFrame({"result": [1, 2, 3]})
        folio.add_table("results", df)

        # Verify both are tracked
        contents = folio.list_contents()
        assert "big_data" in contents["referenced_tables"]
        assert "results" in contents["included_tables"]

        # Verify retrieval works
        retrieved_df = folio.get_table("results")
        pd.testing.assert_frame_equal(retrieved_df, df)
        assert folio.get_data_path("big_data") == "s3://bucket/big.parquet"

    def test_chaining_multiple_operations(self, tmp_path):
        """Test method chaining with multiple operations."""
        folio = DataFolio(tmp_path / "test", metadata={"experiment": "test"})
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})

        folio.reference_table("ref1", path="s3://bucket/file1.parquet").add_table(
            "inc1", df1
        ).reference_table("ref2", path="s3://bucket/file2.parquet").add_table(
            "inc2", df2
        )

        contents = folio.list_contents()
        assert len(contents["referenced_tables"]) == 2
        assert len(contents["included_tables"]) == 2
