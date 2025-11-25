"""Tests for iter_table functionality - chunked iteration over tables.

These tests verify memory-efficient iteration using DuckDB.
"""

import pytest

from datafolio import DataFolio

# Try to import dependencies
try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

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


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestIterTableBasics:
    """Basic iter_table functionality tests."""

    def test_iter_table_simple(self, tmp_path):
        """Test basic iteration over table."""
        folio = DataFolio(tmp_path / "test")

        # Add table with 25 rows
        df = pd.DataFrame({"a": range(25), "b": range(25, 50)})
        folio.add_table("data", df)

        # Iterate with chunk_size=10
        chunks = list(folio.iter_table("data", chunk_size=10))

        # Should have 3 chunks (10, 10, 5)
        assert len(chunks) == 3
        assert len(chunks[0]) == 10
        assert len(chunks[1]) == 10
        assert len(chunks[2]) == 5

        # Verify all data is present
        all_data = pd.concat(chunks, ignore_index=True)
        assert len(all_data) == 25
        assert all_data["a"].tolist() == list(range(25))

    def test_iter_table_exact_chunks(self, tmp_path):
        """Test iteration when rows divide evenly by chunk_size."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(100)})
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=25))

        # Should have exactly 4 chunks of 25
        assert len(chunks) == 4
        assert all(len(chunk) == 25 for chunk in chunks)

    def test_iter_table_large_chunk_size(self, tmp_path):
        """Test iteration when chunk_size is larger than table."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": range(10)})
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=1000))

        # Should have 1 chunk with all data
        assert len(chunks) == 1
        assert len(chunks[0]) == 10

    def test_iter_table_chunk_size_one(self, tmp_path):
        """Test iteration with chunk_size=1."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": range(5)})
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=1))

        # Should have 5 chunks of 1 row each
        assert len(chunks) == 5
        assert all(len(chunk) == 1 for chunk in chunks)


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestIterTableColumns:
    """Tests for column selection during iteration."""

    def test_iter_table_select_columns(self, tmp_path):
        """Test selecting specific columns."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": range(20), "b": range(20, 40), "c": range(40, 60)})
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=10, columns=["a", "c"]))

        # Verify only selected columns
        for chunk in chunks:
            assert list(chunk.columns) == ["a", "c"]
            assert "b" not in chunk.columns

    def test_iter_table_single_column(self, tmp_path):
        """Test selecting a single column."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(15), "y": range(15, 30)})
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=5, columns=["x"]))

        assert len(chunks) == 3
        for chunk in chunks:
            assert list(chunk.columns) == ["x"]
            assert len(chunk) == 5


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestIterTableFiltering:
    """Tests for WHERE clause filtering during iteration."""

    def test_iter_table_with_where(self, tmp_path):
        """Test iteration with WHERE filter."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(100), "category": ["A", "B"] * 50})
        folio.add_table("data", df)

        # Filter to only values > 80
        chunks = list(folio.iter_table("data", chunk_size=5, where="value > 80"))

        # Should get 19 rows (81-99)
        all_data = pd.concat(chunks, ignore_index=True)
        assert len(all_data) == 19
        assert all_data["value"].min() == 81
        assert all_data["value"].max() == 99

    def test_iter_table_where_with_string(self, tmp_path):
        """Test WHERE with string comparison."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame(
            {"name": ["Alice", "Bob", "Charlie", "Alice", "Bob"], "value": range(5)}
        )
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=2, where="name = 'Alice'"))

        all_data = pd.concat(chunks, ignore_index=True)
        assert len(all_data) == 2
        assert all(all_data["name"] == "Alice")

    def test_iter_table_where_multiple_conditions(self, tmp_path):
        """Test WHERE with multiple conditions."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame(
            {"category": ["A", "B", "A", "B", "A"] * 10, "value": range(50)}
        )
        folio.add_table("data", df)

        chunks = list(
            folio.iter_table(
                "data", chunk_size=5, where="category = 'A' AND value > 20"
            )
        )

        all_data = pd.concat(chunks, ignore_index=True)
        assert all(all_data["category"] == "A")
        assert all(all_data["value"] > 20)

    def test_iter_table_columns_and_where(self, tmp_path):
        """Test combining column selection and WHERE filter."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": range(30), "b": range(30, 60), "c": range(60, 90)})
        folio.add_table("data", df)

        chunks = list(
            folio.iter_table("data", chunk_size=5, columns=["a", "b"], where="a > 20")
        )

        all_data = pd.concat(chunks, ignore_index=True)
        assert list(all_data.columns) == ["a", "b"]
        assert len(all_data) == 9  # Rows where a > 20 (21-29)
        assert all(all_data["a"] > 20)


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS or not HAS_POLARS, reason="All deps required"
)
class TestIterTableReturnTypes:
    """Tests for different return types."""

    def test_iter_table_return_pandas(self, tmp_path):
        """Test returning pandas DataFrames (default)."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(20)})
        folio.add_table("data", df)

        for chunk in folio.iter_table("data", chunk_size=10, return_type="pandas"):
            assert isinstance(chunk, pd.DataFrame)

    def test_iter_table_return_polars(self, tmp_path):
        """Test returning Polars DataFrames."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(20)})
        folio.add_table("data", df)

        for chunk in folio.iter_table("data", chunk_size=10, return_type="polars"):
            assert isinstance(chunk, pl.DataFrame)

    def test_iter_polars_table_return_polars(self, tmp_path):
        """Test iterating Polars table and returning Polars."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"x": range(30)})
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=10, return_type="polars"))

        assert len(chunks) == 3
        for chunk in chunks:
            assert isinstance(chunk, pl.DataFrame)


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestIterTableErrors:
    """Tests for error handling."""

    def test_iter_nonexistent_table(self, tmp_path):
        """Test error when table doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            list(folio.iter_table("nonexistent", chunk_size=10))

    def test_iter_non_table_item(self, tmp_path):
        """Test error when item is not a table."""
        folio = DataFolio(tmp_path / "test")

        folio.add_json("config", {"key": "value"})

        with pytest.raises(ValueError, match="must be an included table"):
            list(folio.iter_table("config", chunk_size=10))

    def test_iter_referenced_table(self, tmp_path):
        """Test error when iterating referenced table."""
        folio = DataFolio(tmp_path / "test")

        temp_file = tmp_path / "external.parquet"
        df = pd.DataFrame({"x": range(10)})
        df.to_parquet(temp_file)

        folio.reference_table("external", str(temp_file))

        with pytest.raises(ValueError, match="must be an included table"):
            list(folio.iter_table("external", chunk_size=5))

    def test_iter_invalid_chunk_size(self, tmp_path):
        """Test error on invalid chunk_size."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(10)})
        folio.add_table("data", df)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(folio.iter_table("data", chunk_size=0))

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(folio.iter_table("data", chunk_size=-10))

    def test_iter_invalid_return_type(self, tmp_path):
        """Test error on invalid return type."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(10)})
        folio.add_table("data", df)

        with pytest.raises(ValueError, match="Invalid return_type"):
            list(folio.iter_table("data", chunk_size=5, return_type="invalid"))


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestIterTableMemoryEfficiency:
    """Tests to verify memory-efficient iteration."""

    def test_iter_large_table_memory_efficient(self, tmp_path):
        """Test that iteration doesn't load full table."""
        folio = DataFolio(tmp_path / "test")

        # Create moderately large table
        df = pd.DataFrame({"value": range(10000), "data": ["x" * 100] * 10000})
        folio.add_table("data", df)

        # Iterate with small chunks
        chunk_count = 0
        total_rows = 0

        for chunk in folio.iter_table("data", chunk_size=100):
            chunk_count += 1
            total_rows += len(chunk)
            # Each chunk should be small
            assert len(chunk) <= 100

        assert chunk_count == 100
        assert total_rows == 10000

    def test_iter_with_filter_reduces_data(self, tmp_path):
        """Test that WHERE filter reduces amount of data returned."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(10000)})
        folio.add_table("data", df)

        # Filter to small subset
        chunks = list(folio.iter_table("data", chunk_size=10, where="value < 50"))

        all_data = pd.concat(chunks, ignore_index=True)
        assert len(all_data) == 50  # Only 50 rows instead of 10000

    def test_iter_processes_data_incrementally(self, tmp_path):
        """Test that data is processed chunk by chunk."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(100)})
        folio.add_table("data", df)

        # Process data incrementally
        results = []
        for chunk in folio.iter_table("data", chunk_size=10):
            # Simulate processing
            chunk_mean = chunk["value"].mean()
            results.append(chunk_mean)

        # Should have 10 results (one per chunk)
        assert len(results) == 10


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestIterTableEdgeCases:
    """Tests for edge cases."""

    def test_iter_empty_table(self, tmp_path):
        """Test iteration over empty table."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": [], "b": []})
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=10))

        # Should have no chunks
        assert len(chunks) == 0

    def test_iter_table_where_no_matches(self, tmp_path):
        """Test iteration when WHERE filter matches no rows."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(20)})
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=5, where="value > 100"))

        # Should have no chunks
        assert len(chunks) == 0

    def test_iter_table_single_row(self, tmp_path):
        """Test iteration over single-row table."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": [42]})
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=10))

        assert len(chunks) == 1
        assert len(chunks[0]) == 1
        assert chunks[0]["x"].values[0] == 42

    def test_iter_table_with_nulls(self, tmp_path):
        """Test iteration over table with NULL values."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": [1, None, 3, None, 5] * 4, "b": range(20)})
        folio.add_table("data", df)

        chunks = list(folio.iter_table("data", chunk_size=5))

        all_data = pd.concat(chunks, ignore_index=True)
        assert len(all_data) == 20
        assert all_data["a"].isna().sum() == 8  # 8 NULLs
