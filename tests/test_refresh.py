"""Tests for auto-refresh functionality with multiple DataFolio instances."""

import pandas as pd
import pytest

from datafolio import DataFolio


class TestMultiInstanceRefresh:
    """Tests for automatic refresh when bundles are updated externally."""

    def test_list_contents_sees_new_items(self, tmp_path):
        """Test that list_contents() auto-refreshes and sees new items."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle with one item
        folio1 = DataFolio(bundle_path)
        folio1.add_json("config", {"version": 1})

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)
        assert "config" in folio2.list_contents()["json_data"]
        assert "new_data" not in folio2.list_contents()["json_data"]

        # Instance 1: Add new item
        folio1.add_json("new_data", {"value": 42})

        # Instance 2: Should auto-refresh and see new item
        contents = folio2.list_contents()
        assert "new_data" in contents["json_data"]
        assert "config" in contents["json_data"]

    def test_describe_sees_new_items(self, tmp_path):
        """Test that describe() auto-refreshes and shows new items."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle
        folio1 = DataFolio(bundle_path)
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio1.add_table("data1", df)

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)
        desc = folio2.describe(return_string=True)
        assert "data1" in desc
        assert "data2" not in desc

        # Instance 1: Add another table
        folio1.add_table("data2", df)

        # Instance 2: Should auto-refresh in describe()
        desc = folio2.describe(return_string=True)
        assert "data1" in desc
        assert "data2" in desc

    def test_get_table_sees_new_table(self, tmp_path):
        """Test that get_table() can access newly added tables."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle
        folio1 = DataFolio(bundle_path)
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        folio1.add_table("initial", df1)

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)

        # Instance 1: Add new table
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        folio1.add_table("new_table", df2)

        # Instance 2: Should be able to get new table after auto-refresh
        retrieved = folio2.get_table("new_table")
        pd.testing.assert_frame_equal(retrieved, df2)

    def test_get_json_sees_new_data(self, tmp_path):
        """Test that get_json() can access newly added JSON data."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle
        folio1 = DataFolio(bundle_path)
        folio1.add_json("config1", {"key": "value1"})

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)

        # Instance 1: Add new JSON data
        folio1.add_json("config2", {"key": "value2"})

        # Instance 2: Should be able to get new data
        retrieved = folio2.get_json("config2")
        assert retrieved == {"key": "value2"}

    def test_metadata_updates_are_visible(self, tmp_path):
        """Test that metadata updates are visible across instances."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle with metadata
        folio1 = DataFolio(bundle_path, metadata={"experiment": "test1"})

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)
        assert folio2.metadata["experiment"] == "test1"

        # Instance 1: Update metadata
        folio1.metadata["experiment"] = "test2"
        folio1.metadata["new_field"] = "new_value"

        # Instance 2: Should see updated metadata after refresh
        # Trigger refresh via describe()
        folio2.describe(return_string=True)
        assert folio2.metadata["experiment"] == "test2"
        assert folio2.metadata["new_field"] == "new_value"

    def test_data_accessor_sees_new_items(self, tmp_path):
        """Test that folio.data accessor auto-refreshes."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle
        folio1 = DataFolio(bundle_path)
        folio1.add_json("initial", {"value": 1})

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)

        # Instance 1: Add new item
        folio1.add_json("new_item", {"value": 2})

        # Instance 2: Access via data accessor should work
        retrieved = folio2.data.new_item.content
        assert retrieved == {"value": 2}

    def test_item_proxy_properties_refresh(self, tmp_path):
        """Test that ItemProxy properties auto-refresh."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle
        folio1 = DataFolio(bundle_path)
        folio1.add_json("config", {"value": 1}, description="Initial")

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)
        proxy = folio2.data.config

        # Instance 1: Add another item
        folio1.add_json("new_config", {"value": 2})

        # Instance 2: ItemProxy should still work and refresh
        assert proxy.description == "Initial"
        assert proxy.type == "json_data"

    def test_explicit_refresh(self, tmp_path):
        """Test explicit refresh() method."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle
        folio1 = DataFolio(bundle_path)
        folio1.add_json("data1", {"value": 1})

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)

        # Instance 1: Add new item
        folio1.add_json("data2", {"value": 2})

        # Instance 2: Explicitly refresh
        folio2.refresh()

        # Should see new item
        assert "data2" in folio2.list_contents()["json_data"]

    def test_refresh_returns_self_for_chaining(self, tmp_path):
        """Test that refresh() returns self for method chaining."""
        bundle_path = tmp_path / "shared"
        folio = DataFolio(bundle_path)

        result = folio.refresh()
        assert result is folio

    def test_no_refresh_when_nothing_changed(self, tmp_path):
        """Test that refresh is skipped when nothing changed."""
        bundle_path = tmp_path / "shared"

        folio1 = DataFolio(bundle_path)
        folio1.add_json("data", {"value": 1})

        folio2 = DataFolio(bundle_path)

        # Call list_contents multiple times - should not refresh unnecessarily
        # (This test mainly ensures no errors occur)
        for _ in range(3):
            contents = folio2.list_contents()
            assert "data" in contents["json_data"]

    def test_multiple_item_types(self, tmp_path):
        """Test refresh works with multiple item types."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle
        folio1 = DataFolio(bundle_path)
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio1.add_table("table1", df)

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)

        # Instance 1: Add different types of items
        folio1.add_json("config", {"key": "value"})
        folio1.reference_table("external", path="s3://bucket/data.parquet")

        # Instance 2: Should see all new items
        contents = folio2.list_contents()
        assert "table1" in contents["included_tables"]
        assert "config" in contents["json_data"]
        assert "external" in contents["referenced_tables"]

    def test_deleted_items_disappear(self, tmp_path):
        """Test that deleted items are removed after refresh."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle with two items
        folio1 = DataFolio(bundle_path)
        folio1.add_json("data1", {"value": 1})
        folio1.add_json("data2", {"value": 2})

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)
        assert "data1" in folio2.list_contents()["json_data"]
        assert "data2" in folio2.list_contents()["json_data"]

        # Instance 1: Delete an item
        folio1.delete("data1")

        # Instance 2: Should see deletion after refresh
        contents = folio2.list_contents()
        assert "data1" not in contents["json_data"]
        assert "data2" in contents["json_data"]

    def test_lineage_methods_refresh(self, tmp_path):
        """Test that lineage methods auto-refresh."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle with lineage
        folio1 = DataFolio(bundle_path)
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio1.add_table("raw_data", df)

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)

        # Instance 1: Add derived item
        folio1.add_json("processed", {"value": 1}, inputs=["raw_data"])

        # Instance 2: Should see lineage after refresh
        graph = folio2.get_lineage_graph()
        assert "processed" in graph
        assert graph["processed"] == ["raw_data"]

        inputs = folio2.get_inputs("processed")
        assert inputs == ["raw_data"]

    def test_get_table_info_refreshes(self, tmp_path):
        """Test that get_table_info() auto-refreshes."""
        bundle_path = tmp_path / "shared"

        # Instance 1: Create bundle
        folio1 = DataFolio(bundle_path)
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio1.add_table("data", df, description="Initial description")

        # Instance 2: Open same bundle
        folio2 = DataFolio(bundle_path)

        # Instance 1: Add new table
        folio1.add_table("new_data", df, description="New description")

        # Instance 2: Should be able to get info after refresh
        info = folio2.get_table_info("new_data")
        assert info["description"] == "New description"


class TestRefreshEdgeCases:
    """Test edge cases and error conditions for refresh."""

    def test_refresh_with_missing_metadata_file(self, tmp_path):
        """Test refresh handles missing metadata gracefully."""
        bundle_path = tmp_path / "shared"

        folio = DataFolio(bundle_path)
        folio.add_json("data", {"value": 1})

        # Delete metadata file
        metadata_file = bundle_path / "metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()

        # Refresh should not crash
        folio.refresh()

    def test_refresh_with_corrupted_metadata(self, tmp_path):
        """Test refresh handles corrupted metadata gracefully."""
        bundle_path = tmp_path / "shared"

        folio = DataFolio(bundle_path)
        folio.add_json("data", {"value": 1})

        # Corrupt metadata file
        metadata_file = bundle_path / "metadata.json"
        metadata_file.write_text("not valid json {")

        # Refresh should handle error gracefully (not crash)
        # The _check_if_stale catches exceptions
        folio.list_contents()  # Should not crash

    def test_new_bundle_no_stale_check(self, tmp_path):
        """Test that new bundles don't trigger unnecessary refreshes."""
        bundle_path = tmp_path / "new"

        # Create new bundle
        folio = DataFolio(bundle_path)
        folio.add_json("data", {"value": 1})

        # Operations should work without refresh issues
        assert "data" in folio.list_contents()["json_data"]


class TestRefreshPerformance:
    """Test that refresh doesn't cause performance issues."""

    def test_repeated_access_same_instance(self, tmp_path):
        """Test repeated access doesn't cause excessive refreshes."""
        bundle_path = tmp_path / "shared"

        folio = DataFolio(bundle_path)
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Access multiple times - should be efficient
        for _ in range(10):
            folio.list_contents()
            folio.describe(return_string=True)
            folio.get_table("data")

        # No assertion needed - just ensure no crashes or errors
