"""Tests for lineage tracking features."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from datafolio import DataFolio


class DummyModel:
    """Dummy model for testing."""

    def __init__(self, param=1):
        self.param = param


class TestAutoTimestamps:
    """Tests for automatic timestamp tracking."""

    def test_created_at_on_init(self, tmp_path):
        """Test that created_at timestamp is set on bundle creation."""
        folio = DataFolio(tmp_path / "test")

        assert "created_at" in folio.metadata
        assert "updated_at" in folio.metadata

        # Verify it's a valid ISO 8601 timestamp
        created_at = datetime.fromisoformat(folio.metadata["created_at"])
        assert created_at is not None

    def test_updated_at_on_metadata_change(self, tmp_path):
        """Test that updated_at timestamp is updated when metadata changes."""
        folio = DataFolio(tmp_path / "test")

        original_updated = folio.metadata["updated_at"]

        # Make a change to metadata
        import time

        time.sleep(0.01)  # Ensure timestamp will be different
        folio.metadata["experiment"] = "test_001"

        # Verify updated_at changed
        assert folio.metadata["updated_at"] != original_updated

        # Verify created_at didn't change
        assert "created_at" in folio.metadata

    def test_timestamps_preserved_on_load(self, tmp_path):
        """Test that timestamps are preserved when loading a bundle."""
        # Create and save bundle
        folio1 = DataFolio(tmp_path / "test")
        original_created = folio1.metadata["created_at"]

        # Load the same bundle
        folio2 = DataFolio(path=folio1._bundle_dir)

        # Timestamps should be preserved
        assert folio2.metadata["created_at"] == original_created


class TestLineageFields:
    """Tests for lineage field storage in items."""

    def test_add_table_with_lineage(self, tmp_path):
        """Test adding table with lineage metadata."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table(
            "processed_data",
            df,
            inputs=["raw_data"],
            models=["preprocessor"],
            code="df.transform(preprocessor)",
        )

        item = folio._items["processed_data"]
        assert item["inputs"] == ["raw_data"]
        assert item["models"] == ["preprocessor"]
        assert item["code"] == "df.transform(preprocessor)"
        assert "created_at" in item

    def test_reference_table_with_lineage(self, tmp_path):
        """Test referencing table with lineage metadata."""
        # Create a dummy parquet file
        df = pd.DataFrame({"a": [1, 2, 3]})
        parquet_path = tmp_path / "data.parquet"
        df.to_parquet(parquet_path)

        folio = DataFolio(tmp_path / "test")
        folio.reference_table(
            "external_data",
            parquet_path,
            table_format="parquet",
            inputs=["upstream_table"],
            code="spark.read.parquet(...)",
        )

        item = folio._items["external_data"]
        assert item["inputs"] == ["upstream_table"]
        assert item["code"] == "spark.read.parquet(...)"
        assert "created_at" in item

    def test_add_model_with_lineage(self, tmp_path):
        """Test adding model with lineage metadata."""
        folio = DataFolio(tmp_path / "test")
        model = DummyModel(param=42)

        folio.add_model(
            "classifier",
            model,
            inputs=["training_data", "validation_data"],
            hyperparameters={"learning_rate": 0.01, "epochs": 100},
            code="model.fit(X_train, y_train)",
        )

        item = folio._items["classifier"]
        assert item["inputs"] == ["training_data", "validation_data"]
        assert item["hyperparameters"] == {"learning_rate": 0.01, "epochs": 100}
        assert item["code"] == "model.fit(X_train, y_train)"
        assert "created_at" in item

    def test_lineage_fields_optional(self, tmp_path):
        """Test that lineage fields are optional."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        # Should work without lineage fields
        folio.add_table("data", df)

        item = folio._items["data"]
        assert "created_at" in item  # Timestamp is always added
        # Other lineage fields may be missing or None


class TestLineageQueries:
    """Tests for lineage query methods."""

    def test_get_inputs_basic(self, tmp_path):
        """Test getting inputs for an item."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("raw", df)
        folio.add_table("processed", df, inputs=["raw"])

        inputs = folio.get_inputs("processed")
        assert inputs == ["raw"]

    def test_get_inputs_with_models(self, tmp_path):
        """Test that get_inputs includes models for tables."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("processed", df, inputs=["raw_data"], models=["transformer"])

        inputs = folio.get_inputs("processed")
        assert "raw_data" in inputs
        assert "transformer" in inputs

    def test_get_inputs_empty(self, tmp_path):
        """Test getting inputs for item with no dependencies."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("data", df)

        inputs = folio.get_inputs("data")
        assert inputs == []

    def test_get_inputs_not_found(self, tmp_path):
        """Test error when getting inputs for nonexistent item."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get_inputs("nonexistent")

    def test_get_dependents_basic(self, tmp_path):
        """Test getting items that depend on an item."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("raw", df)
        folio.add_table("processed", df, inputs=["raw"])
        folio.add_table("final", df, inputs=["processed"])

        dependents = folio.get_dependents("raw")
        assert "processed" in dependents

    def test_get_dependents_model_dependency(self, tmp_path):
        """Test getting dependents when used as a model."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        model = DummyModel()

        folio.add_model("transformer", model)
        folio.add_table("processed", df, models=["transformer"])

        dependents = folio.get_dependents("transformer")
        assert "processed" in dependents

    def test_get_dependents_empty(self, tmp_path):
        """Test getting dependents for item with no dependents."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("data", df)

        dependents = folio.get_dependents("data")
        assert dependents == []

    def test_get_lineage_graph(self, tmp_path):
        """Test getting full lineage graph."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("raw", df)
        folio.add_table("processed", df, inputs=["raw"])
        folio.add_table("final", df, inputs=["processed"])

        graph = folio.get_lineage_graph()

        assert graph["raw"] == []
        assert graph["processed"] == ["raw"]
        assert graph["final"] == ["processed"]


class TestCopyMethod:
    """Tests for the copy() method."""

    def test_copy_basic(self, tmp_path):
        """Test basic bundle copying."""
        folio1 = DataFolio(tmp_path / "original" / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio1.add_table("data", df)
        folio1.metadata["experiment"] = "exp001"

        # Copy to new location
        folio2 = folio1.copy(base_dir=tmp_path / "copy", name="copy")

        # Verify copy has the same data
        assert "data" in folio2._items
        df_copy = folio2.get_table("data")
        pd.testing.assert_frame_equal(df_copy, df)

        # Verify metadata was copied
        assert folio2.metadata["experiment"] == "exp001"

        # Verify it's a separate bundle
        assert folio2._bundle_dir != folio1._bundle_dir

    def test_copy_with_metadata_updates(self, tmp_path):
        """Test copying with metadata updates."""
        folio1 = DataFolio(tmp_path / "original" / "test")
        folio1.metadata["experiment"] = "exp001"

        folio2 = folio1.copy(
            base_dir=tmp_path / "copy",
            name="test",
            metadata_updates={"experiment": "exp002", "notes": "Modified version"},
        )

        assert folio2.metadata["experiment"] == "exp002"
        assert folio2.metadata["notes"] == "Modified version"

    def test_copy_with_include_items(self, tmp_path):
        """Test copying with selective item inclusion."""
        folio1 = DataFolio(tmp_path / "original" / "test")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio1.add_table("data1", df1)
        folio1.add_table("data2", df2)

        folio2 = folio1.copy(
            base_dir=tmp_path / "copy", name="test", include_items=["data1"]
        )

        assert "data1" in folio2._items
        assert "data2" not in folio2._items

    def test_copy_with_exclude_items(self, tmp_path):
        """Test copying with selective item exclusion."""
        folio1 = DataFolio(tmp_path / "original" / "test")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        folio1.add_table("data1", df1)
        folio1.add_table("data2", df2)

        folio2 = folio1.copy(
            base_dir=tmp_path / "copy", name="test", exclude_items=["data2"]
        )

        assert "data1" in folio2._items
        assert "data2" not in folio2._items

    def test_copy_cannot_specify_both_include_exclude(self, tmp_path):
        """Test that specifying both include and exclude raises error."""
        folio1 = DataFolio(tmp_path / "original" / "test")

        with pytest.raises(ValueError, match="Cannot specify both"):
            folio1.copy(
                base_dir=tmp_path / "copy",
                name="test",
                include_items=["data1"],
                exclude_items=["data2"],
            )

    def test_copy_preserves_models(self, tmp_path):
        """Test that models are copied correctly."""
        folio1 = DataFolio(tmp_path / "original" / "test")
        model = DummyModel(param=42)
        folio1.add_model("clf", model)

        folio2 = folio1.copy(base_dir=tmp_path / "copy", name="test")

        # Verify model was copied
        assert "clf" in folio2._items
        model_copy = folio2.get_model("clf")
        assert model_copy.param == 42

    def test_copy_preserves_artifacts(self, tmp_path):
        """Test that artifacts are copied correctly."""
        # Create artifact file
        artifact_file = tmp_path / "plot.png"
        artifact_file.write_text("fake image")

        folio1 = DataFolio(tmp_path / "original" / "test")
        folio1.add_artifact("plot", artifact_file)

        folio2 = folio1.copy(base_dir=tmp_path / "copy", name="test")

        # Verify artifact was copied
        assert "plot" in folio2._items
        artifact_path = folio2.get_artifact_path("plot")
        assert Path(artifact_path).exists()


class TestDescribeWithLineage:
    """Tests for describe() method showing lineage."""

    def test_describe_shows_timestamps(self, tmp_path):
        """Test that describe shows created_at and updated_at."""
        folio = DataFolio(tmp_path / "test")

        description = folio.describe(return_string=True)

        assert "Created:" in description
        assert "Updated:" in description

    def test_describe_shows_item_lineage(self, tmp_path):
        """Test that describe shows lineage for items."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        folio.add_table("raw", df)
        folio.add_table("processed", df, inputs=["raw"])

        description = folio.describe(return_string=True)

        # Should show lineage with arrow notation
        assert "↳" in description or "raw" in description

    def test_describe_shows_model_hyperparameters(self, tmp_path):
        """Test that describe shows hyperparameters for models."""
        folio = DataFolio(tmp_path / "test")
        model = DummyModel()

        folio.add_model("clf", model, hyperparameters={"lr": 0.01, "epochs": 100})

        description = folio.describe(return_string=True)

        # Should mention hyperparameters
        assert "hyperparameters" in description.lower() or "lr" in description


class TestDescribeMetadataDisplay:
    """Tests for describe() method showing custom metadata."""

    def test_describe_shows_custom_metadata(self, tmp_path):
        """Test that describe shows custom metadata fields."""
        folio = DataFolio(
            tmp_path / "test",
            metadata={"experiment": "test_001", "scientist": "Dr. Smith"},
        )

        description = folio.describe(return_string=True)

        assert "Metadata" in description
        assert "experiment" in description
        assert "test_001" in description
        assert "scientist" in description
        assert "Dr. Smith" in description

    def test_describe_filters_internal_metadata(self, tmp_path):
        """Test that describe filters out internal metadata fields."""
        folio = DataFolio(tmp_path / "test", metadata={"experiment": "test_001"})

        description = folio.describe(return_string=True)

        # Should NOT show internal fields in the Metadata section
        # (they're shown in the timestamp area already)
        metadata_section = (
            description.split("Metadata")[1] if "Metadata" in description else ""
        )
        assert "_datafolio" not in metadata_section

    def test_describe_truncates_long_strings(self, tmp_path):
        """Test that describe truncates long string values."""
        long_string = "a" * 100
        folio = DataFolio(tmp_path / "test", metadata={"notes": long_string})

        description = folio.describe(return_string=True)

        # Should show truncated version with indicator
        assert "..." in description
        assert "more chars" in description

    def test_describe_shows_list_preview(self, tmp_path):
        """Test that describe shows preview for long lists."""
        long_list = list(range(20))
        folio = DataFolio(tmp_path / "test", metadata={"batch_sizes": long_list})

        description = folio.describe(return_string=True)

        # Should show preview with count of remaining items
        assert "batch_sizes" in description
        assert "more items" in description

    def test_describe_shows_dict_preview(self, tmp_path):
        """Test that describe shows preview for large dicts."""
        large_dict = {f"key_{i}": i for i in range(10)}
        folio = DataFolio(tmp_path / "test", metadata={"config": large_dict})

        description = folio.describe(return_string=True)

        # Should show preview with count of remaining fields
        assert "config" in description
        assert "more fields" in description

    def test_describe_respects_max_metadata_fields(self, tmp_path):
        """Test that describe respects max_metadata_fields parameter."""
        # Create folio with 15 custom metadata fields
        metadata = {f"field_{i}": f"value_{i}" for i in range(15)}
        folio = DataFolio(tmp_path / "test", metadata=metadata)

        # Request only 5 fields
        description = folio.describe(return_string=True, max_metadata_fields=5)

        # Should show "... (10 more fields)" at the end
        assert "(10 more fields)" in description

    def test_describe_no_custom_metadata(self, tmp_path):
        """Test describe when there's no custom metadata."""
        folio = DataFolio(tmp_path / "test")

        description = folio.describe(return_string=True)

        # Should NOT show Metadata section if only internal fields exist
        # But timestamps should still be shown
        assert "Created:" in description
        # Metadata section should not appear (or be empty)

    def test_describe_short_values_not_truncated(self, tmp_path):
        """Test that short values are not truncated."""
        folio = DataFolio(
            tmp_path / "test",
            metadata={
                "short_string": "hello",
                "short_list": [1, 2, 3],
                "short_dict": {"a": 1, "b": 2},
            },
        )

        description = folio.describe(return_string=True)

        # Should show full values without truncation indicators
        assert "hello" in description
        assert "[1, 2, 3]" in description
        # Dict should be fully shown
        assert "short_dict" in description

    def test_format_metadata_value_strings(self, tmp_path):
        """Test _format_metadata_value for different string lengths."""
        from datafolio.display import DisplayFormatter

        folio = DataFolio(tmp_path / "test")
        formatter = DisplayFormatter(folio)

        # Short string - no truncation
        short = formatter._format_metadata_value("hello")
        assert short == "hello"
        assert "..." not in short

        # Long string - truncated
        long = formatter._format_metadata_value("a" * 100, max_length=60)
        assert "..." in long
        assert "40 more chars" in long

    def test_format_metadata_value_lists(self, tmp_path):
        """Test _format_metadata_value for lists."""
        from datafolio.display import DisplayFormatter

        folio = DataFolio(tmp_path / "test")
        formatter = DisplayFormatter(folio)

        # Short list - no truncation
        short = formatter._format_metadata_value([1, 2, 3])
        assert short == "[1, 2, 3]"

        # Long list - truncated
        long = formatter._format_metadata_value(list(range(20)))
        assert "..." in long
        assert "15 more items" in long

    def test_format_metadata_value_dicts(self, tmp_path):
        """Test _format_metadata_value for dicts."""
        from datafolio.display import DisplayFormatter

        folio = DataFolio(tmp_path / "test")
        formatter = DisplayFormatter(folio)

        # Small dict - no truncation
        small = formatter._format_metadata_value({"a": 1, "b": 2})
        assert "..." not in small

        # Large dict - truncated
        large = formatter._format_metadata_value({f"key_{i}": i for i in range(10)})
        assert "..." in large
        assert "7 more fields" in large

    def test_metadata_sorting_consistent(self, tmp_path):
        """Test that metadata fields are sorted consistently."""
        metadata = {"zebra": "z", "apple": "a", "banana": "b"}
        folio = DataFolio(tmp_path / "test", metadata=metadata)

        description = folio.describe(return_string=True)

        # Fields should appear in alphabetical order
        apple_pos = description.find("apple")
        banana_pos = description.find("banana")
        zebra_pos = description.find("zebra")

        assert apple_pos < banana_pos < zebra_pos
