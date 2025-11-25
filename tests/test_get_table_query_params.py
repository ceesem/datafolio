"""Tests for get_table() convenience query parameters.

These tests verify limit, offset, and where parameters in get_table().
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
class TestGetTableLimit:
    """Tests for limit parameter."""

    def test_get_table_with_limit(self, tmp_path):
        """Test getting table with limit."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(100)})
        folio.add_table("data", df)

        result = folio.get_table("data", limit=10)

        assert len(result) == 10
        assert isinstance(result, pd.DataFrame)

    def test_get_table_limit_larger_than_table(self, tmp_path):
        """Test limit larger than table size."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(10)})
        folio.add_table("data", df)

        result = folio.get_table("data", limit=100)

        assert len(result) == 10  # Only 10 rows available

    def test_get_table_limit_zero(self, tmp_path):
        """Test limit=0 returns empty result."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(10)})
        folio.add_table("data", df)

        result = folio.get_table("data", limit=0)

        assert len(result) == 0


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestGetTableOffset:
    """Tests for offset parameter."""

    def test_get_table_with_offset(self, tmp_path):
        """Test getting table with offset."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(100)})
        folio.add_table("data", df)

        result = folio.get_table("data", offset=90)

        assert len(result) == 10  # Rows 90-99
        assert result["value"].min() == 90
        assert result["value"].max() == 99

    def test_get_table_offset_larger_than_table(self, tmp_path):
        """Test offset larger than table size."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"x": range(10)})
        folio.add_table("data", df)

        result = folio.get_table("data", offset=20)

        assert len(result) == 0  # No rows after offset


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestGetTableLimitAndOffset:
    """Tests for combined limit and offset (pagination)."""

    def test_get_table_pagination(self, tmp_path):
        """Test pagination with limit and offset."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(100)})
        folio.add_table("data", df)

        # Page 1
        page1 = folio.get_table("data", limit=10, offset=0)
        assert len(page1) == 10
        assert page1["value"].tolist() == list(range(0, 10))

        # Page 2
        page2 = folio.get_table("data", limit=10, offset=10)
        assert len(page2) == 10
        assert page2["value"].tolist() == list(range(10, 20))

        # Page 3
        page3 = folio.get_table("data", limit=10, offset=20)
        assert len(page3) == 10
        assert page3["value"].tolist() == list(range(20, 30))

    def test_get_table_last_page_partial(self, tmp_path):
        """Test last page with fewer rows than limit."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(25)})
        folio.add_table("data", df)

        # Last page should have only 5 rows
        last_page = folio.get_table("data", limit=10, offset=20)
        assert len(last_page) == 5
        assert last_page["value"].tolist() == list(range(20, 25))


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestGetTableWhere:
    """Tests for where parameter."""

    def test_get_table_with_where(self, tmp_path):
        """Test getting table with WHERE filter."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(100), "category": ["A", "B"] * 50})
        folio.add_table("data", df)

        result = folio.get_table("data", where="value > 90")

        assert len(result) == 9  # Values 91-99
        assert result["value"].min() == 91

    def test_get_table_where_with_string(self, tmp_path):
        """Test WHERE with string comparison."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame(
            {"name": ["Alice", "Bob", "Charlie", "Alice", "Bob"], "value": range(5)}
        )
        folio.add_table("data", df)

        result = folio.get_table("data", where="name = 'Alice'")

        assert len(result) == 2
        assert all(result["name"] == "Alice")

    def test_get_table_where_multiple_conditions(self, tmp_path):
        """Test WHERE with multiple conditions."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame(
            {"category": ["A", "B", "A", "B", "A"] * 10, "value": range(50)}
        )
        folio.add_table("data", df)

        result = folio.get_table("data", where="category = 'A' AND value > 20")

        assert all(result["category"] == "A")
        assert all(result["value"] > 20)

    def test_get_table_where_no_matches(self, tmp_path):
        """Test WHERE that matches no rows."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(10)})
        folio.add_table("data", df)

        result = folio.get_table("data", where="value > 100")

        assert len(result) == 0


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestGetTableCombined:
    """Tests for combining multiple parameters."""

    def test_get_table_where_and_limit(self, tmp_path):
        """Test combining WHERE and LIMIT."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(100)})
        folio.add_table("data", df)

        result = folio.get_table("data", where="value < 50", limit=10)

        assert len(result) == 10  # Limited to 10 even though 50 match
        assert all(result["value"] < 50)

    def test_get_table_all_parameters(self, tmp_path):
        """Test using where, limit, and offset together."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(100)})
        folio.add_table("data", df)

        # Get values 20-39, but only return 5 starting from offset 10
        result = folio.get_table(
            "data", where="value >= 20 AND value < 40", limit=5, offset=10
        )

        assert len(result) == 5
        # Should be values 30-34 (offset 10 from the 20 matching rows)
        assert result["value"].tolist() == [30, 31, 32, 33, 34]


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS or not HAS_POLARS, reason="All deps required"
)
class TestGetTableWithPolars:
    """Tests for query parameters with Polars tables."""

    def test_get_polars_table_with_limit(self, tmp_path):
        """Test getting Polars table with limit returns Polars."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"value": range(100)})
        folio.add_table("data", df)

        result = folio.get_table("data", limit=10)

        # Should return Polars DataFrame
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10

    def test_get_polars_table_with_where(self, tmp_path):
        """Test getting Polars table with WHERE returns Polars."""
        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"value": range(50)})
        folio.add_table("data", df)

        result = folio.get_table("data", where="value > 40")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 9


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestGetTableErrors:
    """Tests for error handling."""

    def test_get_table_query_params_on_referenced_table(self, tmp_path):
        """Test error when using query params on referenced table."""
        folio = DataFolio(tmp_path / "test")

        # Create temp file
        temp_file = tmp_path / "external.parquet"
        df = pd.DataFrame({"x": range(10)})
        df.to_parquet(temp_file)

        folio.reference_table("external", str(temp_file))

        # Should error on limit
        with pytest.raises(ValueError, match="not supported for referenced tables"):
            folio.get_table("external", limit=5)

        # Should error on offset
        with pytest.raises(ValueError, match="not supported for referenced tables"):
            folio.get_table("external", offset=5)

        # Should error on where
        with pytest.raises(ValueError, match="not supported for referenced tables"):
            folio.get_table("external", where="x > 5")


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestGetTableBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_get_table_without_params_unchanged(self, tmp_path):
        """Test that get_table() without params works as before."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        folio.add_table("data", df)

        result = folio.get_table("data")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]

    def test_get_polars_table_without_params_unchanged(self, tmp_path):
        """Test that Polars tables still return Polars without params."""
        if not HAS_POLARS:
            pytest.skip("Polars not installed")

        folio = DataFolio(tmp_path / "test")

        df = pl.DataFrame({"x": [1, 2, 3]})
        folio.add_table("data", df)

        result = folio.get_table("data")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


@pytest.mark.skipif(
    not HAS_DUCKDB or not HAS_PANDAS, reason="DuckDB and pandas required"
)
class TestGetTableUseCases:
    """Real-world use case tests."""

    def test_preview_table(self, tmp_path):
        """Test previewing first few rows of large table."""
        folio = DataFolio(tmp_path / "test")

        # Large table
        df = pd.DataFrame({"value": range(10000), "data": ["x" * 100] * 10000})
        folio.add_table("data", df)

        # Quick preview
        preview = folio.get_table("data", limit=5)

        assert len(preview) == 5
        assert preview["value"].tolist() == [0, 1, 2, 3, 4]

    def test_sampling_data(self, tmp_path):
        """Test sampling data with offset and limit."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame({"value": range(1000)})
        folio.add_table("data", df)

        # Sample middle section
        sample = folio.get_table("data", offset=450, limit=100)

        assert len(sample) == 100
        assert sample["value"].min() == 450
        assert sample["value"].max() == 549

    def test_filtering_recent_data(self, tmp_path):
        """Test filtering for recent/relevant data."""
        folio = DataFolio(tmp_path / "test")

        df = pd.DataFrame(
            {
                "timestamp": range(1000),
                "value": range(1000),
                "status": ["active"] * 500 + ["inactive"] * 500,
            }
        )
        folio.add_table("data", df)

        # Get recent active items
        recent = folio.get_table("data", where="timestamp > 900 AND status = 'active'")

        assert len(recent) == 0  # All > 900 are inactive

        # Get older active items
        active = folio.get_table("data", where="status = 'active'", limit=10)
        assert len(active) == 10
        assert all(active["status"] == "active")
