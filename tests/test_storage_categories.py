"""Tests for storage category system."""

import pytest

from datafolio.base.handler import BaseHandler
from datafolio.storage.categories import (
    ITEM_TYPE_TO_CATEGORY,
    StorageCategory,
    get_storage_category,
    get_storage_directory,
)


class TestStorageCategory:
    """Tests for StorageCategory enum."""

    def test_category_values(self):
        """Test that categories have correct string values."""
        assert StorageCategory.TABLES.value == "tables"
        assert StorageCategory.MODELS.value == "models"
        assert StorageCategory.ARTIFACTS.value == "artifacts"

    def test_directory_property(self):
        """Test that directory property returns the value."""
        assert StorageCategory.TABLES.directory == "tables"
        assert StorageCategory.MODELS.directory == "models"
        assert StorageCategory.ARTIFACTS.directory == "artifacts"

    def test_all_categories_exist(self):
        """Test that all standard categories exist (including reserved VIEWS)."""
        categories = {cat.name for cat in StorageCategory}
        assert categories == {"TABLES", "MODELS", "ARTIFACTS", "VIEWS"}


class TestItemTypeMapping:
    """Tests for ITEM_TYPE_TO_CATEGORY mapping."""

    def test_all_standard_types_mapped(self):
        """Test that all standard item types have category mappings."""
        expected_types = {
            "included_table",
            "referenced_table",
            "numpy_array",
            "json_data",
            "model",
            "artifact",
            "timestamp",
        }
        assert set(ITEM_TYPE_TO_CATEGORY.keys()) == expected_types

    def test_table_types_mapped_correctly(self):
        """Test that table types map to TABLES category."""
        assert ITEM_TYPE_TO_CATEGORY["included_table"] == StorageCategory.TABLES
        assert ITEM_TYPE_TO_CATEGORY["referenced_table"] == StorageCategory.TABLES

    def test_model_types_mapped_correctly(self):
        """Test that model types map to MODELS category."""
        assert ITEM_TYPE_TO_CATEGORY["model"] == StorageCategory.MODELS

    def test_artifact_types_mapped_correctly(self):
        """Test that artifact types map to ARTIFACTS category."""
        assert ITEM_TYPE_TO_CATEGORY["numpy_array"] == StorageCategory.ARTIFACTS
        assert ITEM_TYPE_TO_CATEGORY["json_data"] == StorageCategory.ARTIFACTS
        assert ITEM_TYPE_TO_CATEGORY["artifact"] == StorageCategory.ARTIFACTS
        assert ITEM_TYPE_TO_CATEGORY["timestamp"] == StorageCategory.ARTIFACTS


class TestCategoryFunctions:
    """Tests for category lookup functions."""

    def test_get_storage_category_for_table(self):
        """Test getting category for table item type."""
        category = get_storage_category("included_table")
        assert category == StorageCategory.TABLES

    def test_get_storage_category_for_model(self):
        """Test getting category for model item type."""
        category = get_storage_category("model")
        assert category == StorageCategory.MODELS

    def test_get_storage_category_for_artifact(self):
        """Test getting category for artifact item type."""
        category = get_storage_category("numpy_array")
        assert category == StorageCategory.ARTIFACTS

    def test_get_storage_category_unknown_type(self):
        """Test that unknown item type raises KeyError."""
        with pytest.raises(KeyError):
            get_storage_category("unknown_type")

    def test_get_storage_directory_for_table(self):
        """Test getting directory for table item type."""
        directory = get_storage_directory("included_table")
        assert directory == "tables"

    def test_get_storage_directory_for_model(self):
        """Test getting directory for model item type."""
        directory = get_storage_directory("model")
        assert directory == "models"

    def test_get_storage_directory_for_artifact(self):
        """Test getting directory for artifact item type."""
        directory = get_storage_directory("json_data")
        assert directory == "artifacts"

    def test_get_storage_directory_unknown_type(self):
        """Test that unknown item type raises KeyError."""
        with pytest.raises(KeyError):
            get_storage_directory("unknown_type")


class TestHandlerIntegration:
    """Tests for handler integration with storage categories."""

    def test_handler_gets_category_from_item_type(self):
        """Test that handler can get its category from item_type."""

        class MockHandler(BaseHandler):
            @property
            def item_type(self) -> str:
                return "included_table"

            def can_handle(self, data):
                return False

            def add(self, folio, name, data, **kwargs):
                return {}

            def get(self, folio, name, **kwargs):
                return None

        handler = MockHandler()
        category = handler.get_storage_category()
        assert category == StorageCategory.TABLES

    def test_handler_gets_subdir_from_category(self):
        """Test that handler derives subdirectory from category."""

        class MockHandler(BaseHandler):
            @property
            def item_type(self) -> str:
                return "model"

            def can_handle(self, data):
                return False

            def add(self, folio, name, data, **kwargs):
                return {}

            def get(self, folio, name, **kwargs):
                return None

        handler = MockHandler()
        subdir = handler.get_storage_subdir()
        assert subdir == "models"

    def test_handler_category_lookup_chain(self):
        """Test the full lookup chain: item_type -> category -> directory."""

        class ArtifactHandler(BaseHandler):
            @property
            def item_type(self) -> str:
                return "numpy_array"

            def can_handle(self, data):
                return False

            def add(self, folio, name, data, **kwargs):
                return {}

            def get(self, folio, name, **kwargs):
                return None

        handler = ArtifactHandler()

        # Check full chain
        assert handler.item_type == "numpy_array"
        assert handler.get_storage_category() == StorageCategory.ARTIFACTS
        assert handler.get_storage_subdir() == "artifacts"
