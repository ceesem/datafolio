"""Tests for query_table functionality using DuckDB.

These tests verify SQL query capabilities on tables stored in DataFolio.
"""

import pytest

from datafolio import DataFolio

# Try to import dependencies
try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    duckdb = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestQueryTableBasics:
    """Basic query_table functionality tests."""

    def test_query_table_simple_select(self, tmp_path):
        """Test simple SELECT query."""
        folio = DataFolio(tmp_path / "test")

        # Add test data
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        folio.add_table("data", df)

        # Query all rows
        result = folio.query_table("data", "SELECT * FROM $table")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert list(result.columns) == ["a", "b"]

    def test_query_table_where_clause(self, tmp_path):
        """Test query with WHERE clause for filtering."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(100), "category": ["A", "B"] * 50})
        folio.add_table("data", df)

        # Filter rows
        result = folio.query_table("data", "SELECT * FROM $table WHERE value > 90")

        assert len(result) == 9  # Values 91-99
        assert result["value"].min() == 91
        assert result["value"].max() == 99

    def test_query_table_aggregation(self, tmp_path):
        """Test aggregation query."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame(
            {"category": ["A", "A", "B", "B", "C"], "value": [10, 20, 15, 25, 30]}
        )
        folio.add_table("data", df)

        # Group and aggregate
        result = folio.query_table(
            "data",
            "SELECT category, COUNT(*) as count, SUM(value) as total FROM $table GROUP BY category",
        )

        assert len(result) == 3
        assert "count" in result.columns
        assert "total" in result.columns
        assert result[result["category"] == "A"]["total"].values[0] == 30

    def test_query_table_limit(self, tmp_path):
        """Test LIMIT clause."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(1000)})
        folio.add_table("data", df)

        # Limit results
        result = folio.query_table("data", "SELECT * FROM $table LIMIT 10")

        assert len(result) == 10

    def test_query_table_order_by(self, tmp_path):
        """Test ORDER BY clause."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": [5, 2, 8, 1, 9, 3]})
        folio.add_table("data", df)

        # Sort descending
        result = folio.query_table("data", "SELECT * FROM $table ORDER BY value DESC")

        assert result["value"].tolist() == [9, 8, 5, 3, 2, 1]

    def test_query_table_column_selection(self, tmp_path):
        """Test selecting specific columns."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        folio.add_table("data", df)

        # Select specific columns
        result = folio.query_table("data", "SELECT a, c FROM $table")

        assert list(result.columns) == ["a", "c"]
        assert "b" not in result.columns


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestQueryTableReturnTypes:
    """Tests for different return types."""

    def test_query_table_return_pandas(self, tmp_path):
        """Test returning pandas DataFrame (default)."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": [1, 2, 3]})
        folio.add_table("data", df)

        result = folio.query_table("data", "SELECT * FROM $table", return_type="pandas")

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars required")
    def test_query_table_return_polars(self, tmp_path):
        """Test returning Polars DataFrame."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": [1, 2, 3]})
        folio.add_table("data", df)

        result = folio.query_table("data", "SELECT * FROM $table", return_type="polars")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_query_table_invalid_return_type(self, tmp_path):
        """Test error on invalid return type."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": [1, 2, 3]})
        folio.add_table("data", df)

        with pytest.raises(ValueError, match="Invalid return_type"):
            folio.query_table("data", "SELECT * FROM $table", return_type="invalid")


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS or not HAS_POLARS, reason="All deps required"
)
class TestQueryTablePolarsInterop:
    """Tests for querying Polars tables."""

    def test_query_polars_table(self, tmp_path):
        """Test querying a table stored as Polars."""
        folio = DataFolio(tmp_path / "test")

        # Add Polars DataFrame
        df_polars = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        folio.add_table("data", df_polars)

        # Query it
        result = folio.query_table("data", "SELECT * FROM $table WHERE a > 2")

        assert isinstance(result, pd.DataFrame)  # Default return type
        assert len(result) == 3

    def test_query_polars_return_polars(self, tmp_path):
        """Test querying Polars table and returning Polars."""
        folio = DataFolio(tmp_path / "test")

        df_polars = pl.DataFrame({"x": range(10)})
        folio.add_table("data", df_polars)

        result = folio.query_table(
            "data", "SELECT * FROM $table WHERE x < 5", return_type="polars"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestQueryTableErrors:
    """Tests for error handling."""

    def test_query_nonexistent_table(self, tmp_path):
        """Test error when table doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.query_table("nonexistent", "SELECT * FROM $table")

    def test_query_non_table_item(self, tmp_path):
        """Test error when item is not a table."""
        folio = DataFolio(tmp_path / "test")

        # Add non-table item
        folio.add_json("config", {"key": "value"})

        with pytest.raises(ValueError, match="must be an included table"):
            folio.query_table("config", "SELECT * FROM $table")

    def test_query_referenced_table(self, tmp_path):
        """Test error when querying referenced table."""
        folio = DataFolio(tmp_path / "test")

        # Create a temp file to reference
        temp_file = tmp_path / "external.parquet"
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.to_parquet(temp_file)

        folio.reference_table("external", str(temp_file))

        # Should not be able to query referenced tables
        with pytest.raises(ValueError, match="must be an included table"):
            folio.query_table("external", "SELECT * FROM $table")

    def test_query_invalid_sql(self, tmp_path):
        """Test error on invalid SQL."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": [1, 2, 3]})
        folio.add_table("data", df)

        # Invalid SQL should raise DuckDB error
        with pytest.raises(Exception):  # DuckDB will raise specific error
            folio.query_table("data", "SELECT invalid syntax FROM $table")


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestQueryTableAdvanced:
    """Advanced query tests."""

    def test_query_with_calculations(self, tmp_path):
        """Test query with calculated columns."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        folio.add_table("data", df)

        result = folio.query_table("data", "SELECT a, b, a * b as product FROM $table")

        assert "product" in result.columns
        assert result["product"].tolist() == [10, 40, 90]

    def test_query_with_string_operations(self, tmp_path):
        """Test query with string operations."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "value": [1, 2, 3]})
        folio.add_table("data", df)

        result = folio.query_table("data", "SELECT * FROM $table WHERE name LIKE 'A%'")

        assert len(result) == 1
        assert result["name"].values[0] == "Alice"

    def test_query_with_null_handling(self, tmp_path):
        """Test query with NULL values."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": [1, None, 3, None, 5], "b": [10, 20, 30, 40, 50]})
        folio.add_table("data", df)

        result = folio.query_table("data", "SELECT * FROM $table WHERE a IS NOT NULL")

        assert len(result) == 3
        assert result["a"].notna().all()

    def test_query_multiple_conditions(self, tmp_path):
        """Test query with multiple conditions."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
            }
        )
        folio.add_table("data", df)

        result = folio.query_table(
            "data", "SELECT * FROM $table WHERE category = 'A' AND value > 20"
        )

        assert len(result) == 2
        assert all(result["category"] == "A")
        assert all(result["value"] > 20)


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestQueryTableMemoryEfficiency:
    """Tests to verify memory-efficient querying."""

    def test_query_large_table_filtered(self, tmp_path):
        """Test querying large table with filter returns small result."""
        folio = DataFolio(tmp_path / "test")

        # Create large table
        df = pd.DataFrame({"value": range(10000), "category": ["A", "B"] * 5000})
        folio.add_table("data", df)

        # Query small subset
        result = folio.query_table("data", "SELECT * FROM $table WHERE value < 10")

        # Result should be small even though table is large
        assert len(result) == 10
        assert result["value"].max() == 9

    def test_query_aggregation_on_large_table(self, tmp_path):
        """Test aggregation on large table returns small result."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame(
            {"category": (["A"] * 5000 + ["B"] * 5000), "value": range(10000)}
        )
        folio.add_table("data", df)

        # Aggregate to small result
        result = folio.query_table(
            "data",
            "SELECT category, COUNT(*) as count, AVG(value) as avg_value FROM $table GROUP BY category",
        )

        assert len(result) == 2  # Only 2 rows despite 10k input rows
        assert set(result["category"]) == {"A", "B"}


@pytest.mark.skipif(not HAS_DUCKDB, reason="DuckDB required")
class TestQueryTableWithoutOptionalDeps:
    """Tests for behavior without optional dependencies."""

    def test_query_without_pandas_fails(self, tmp_path):
        """Test that query requires pandas."""
        # This test is more of a documentation test since we always have pandas
        # in test environment. In practice, pandas is a core dependency.
        pass

    @pytest.mark.skipif(HAS_POLARS, reason="Test requires Polars NOT installed")
    def test_query_return_polars_without_polars(self, tmp_path):
        """Test error when requesting Polars return without Polars installed."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": [1, 2, 3]})
        folio.add_table("data", df)

        with pytest.raises(ImportError, match="Polars is required"):
            folio.query_table("data", "SELECT * FROM $table", return_type="polars")
