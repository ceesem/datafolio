"""Tests for item curation: archive/unarchive and follow_lineage copy."""

import numpy as np
import pandas as pd
import pytest

from datafolio import DataFolio


class TestArchive:
    """Tests for archive() and unarchive() methods."""

    def test_archive_marks_item(self, tmp_path):
        """archive() sets the 'archived' flag to True in item metadata."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("arr", np.array([1, 2, 3]))

        folio.archive("arr")

        assert folio._items["arr"].get("archived") is True

    def test_unarchive_removes_flag(self, tmp_path):
        """unarchive() removes the 'archived' key entirely."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("arr", np.array([1, 2, 3]))
        folio.archive("arr")

        folio.unarchive("arr")

        assert "archived" not in folio._items["arr"]

    def test_archived_items_hidden_from_list_contents(self, tmp_path):
        """Archived items are excluded from list_contents() by default."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("visible", np.array([1]))
        folio.add_numpy("hidden", np.array([2]))
        folio.archive("hidden")

        contents = folio.list_contents()
        all_names = [n for lst in contents.values() for n in lst]

        assert "visible" in all_names
        assert "hidden" not in all_names

    def test_archived_items_visible_with_flag(self, tmp_path):
        """Archived items appear when include_archived=True is passed."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("visible", np.array([1]))
        folio.add_numpy("hidden", np.array([2]))
        folio.archive("hidden")

        contents = folio.list_contents(include_archived=True)
        all_names = [n for lst in contents.values() for n in lst]

        assert "visible" in all_names
        assert "hidden" in all_names

    def test_archive_glob_pattern(self, tmp_path):
        """archive() with a glob pattern archives all matching items."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("intermediate/step1", np.array([1]))
        folio.add_numpy("intermediate/step2", np.array([2]))
        folio.add_numpy("final/result", np.array([3]))

        folio.archive("intermediate/*")

        assert folio._items["intermediate/step1"].get("archived") is True
        assert folio._items["intermediate/step2"].get("archived") is True
        assert "archived" not in folio._items["final/result"]

    def test_unarchive_glob_pattern(self, tmp_path):
        """unarchive() with a glob pattern restores all matching items."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("intermediate/step1", np.array([1]))
        folio.add_numpy("intermediate/step2", np.array([2]))
        folio.archive("intermediate/*")

        folio.unarchive("intermediate/*")

        assert "archived" not in folio._items["intermediate/step1"]
        assert "archived" not in folio._items["intermediate/step2"]

    def test_archive_list_of_names(self, tmp_path):
        """archive() accepts a list of names."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("a", np.array([1]))
        folio.add_numpy("b", np.array([2]))
        folio.add_numpy("c", np.array([3]))

        folio.archive(["a", "b"])

        assert folio._items["a"].get("archived") is True
        assert folio._items["b"].get("archived") is True
        assert "archived" not in folio._items["c"]

    def test_archive_unknown_name_raises(self, tmp_path):
        """archive() raises KeyError for an unknown item name."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.archive("does_not_exist")

    def test_unarchive_unknown_name_raises(self, tmp_path):
        """unarchive() raises KeyError for an unknown item name."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.unarchive("does_not_exist")

    def test_get_data_still_works_on_archived(self, tmp_path):
        """get_data() can still read archived items."""
        folio = DataFolio(tmp_path / "test")
        arr = np.array([10, 20, 30])
        folio.add_numpy("arr", arr)
        folio.archive("arr")

        result = folio.get_data("arr")

        np.testing.assert_array_equal(result, arr)

    def test_delete_works_on_archived(self, tmp_path):
        """delete() removes archived items from the folio."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("arr", np.array([1]))
        folio.archive("arr")

        folio.delete("arr", warn_dependents=False)

        assert "arr" not in folio._items

    def test_archived_field_persists_on_reload(self, tmp_path):
        """The 'archived' flag survives a save/reload cycle."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("arr", np.array([1, 2]))
        folio.archive("arr")

        # Reload from disk
        folio2 = DataFolio(tmp_path / "test")

        assert folio2._items["arr"].get("archived") is True

    def test_archive_returns_self(self, tmp_path):
        """archive() returns self for method chaining."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("arr", np.array([1]))

        result = folio.archive("arr")

        assert result is folio

    def test_unarchive_returns_self(self, tmp_path):
        """unarchive() returns self for method chaining."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("arr", np.array([1]))
        folio.archive("arr")

        result = folio.unarchive("arr")

        assert result is folio


class TestCopyArchived:
    """Tests for archive interaction with copy()."""

    def test_copy_excludes_archived_by_default(self, tmp_path):
        """copy() does not include archived items by default."""
        src = DataFolio(tmp_path / "src")
        src.add_numpy("visible", np.array([1]))
        src.add_numpy("hidden", np.array([2]))
        src.archive("hidden")

        dst = src.copy(str(tmp_path / "dst"))

        assert "visible" in dst._items
        assert "hidden" not in dst._items

    def test_copy_includes_archived_with_flag(self, tmp_path):
        """copy(include_archived=True) includes archived items."""
        src = DataFolio(tmp_path / "src")
        src.add_numpy("visible", np.array([1]))
        src.add_numpy("hidden", np.array([2]))
        src.archive("hidden")

        dst = src.copy(str(tmp_path / "dst"), include_archived=True)

        assert "visible" in dst._items
        assert "hidden" in dst._items

    def test_archived_flag_preserved_in_copy(self, tmp_path):
        """When archived items are copied, the 'archived' flag is preserved."""
        src = DataFolio(tmp_path / "src")
        src.add_numpy("arr", np.array([1]))
        src.archive("arr")

        dst = src.copy(str(tmp_path / "dst"), include_archived=True)

        assert dst._items["arr"].get("archived") is True


class TestCreateSnapshotWithArchived:
    """Tests for create_snapshot() behaviour with archived items."""

    def test_snapshot_captures_archived_items(self, tmp_path):
        """create_snapshot() includes archived items — snapshots capture complete state."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("active", np.array([1]))
        folio.add_numpy("intermediate", np.array([2]))
        folio.archive("intermediate")

        folio.create_snapshot("v1", description="test snapshot")

        snap_versions = folio._snapshots["v1"]["item_versions"]
        assert "active" in snap_versions
        assert "intermediate" in snap_versions


class TestFollowLineage:
    """Tests for copy(follow_lineage=True)."""

    def _build_lineage_folio(self, path):
        """Helper that builds a folio with a simple lineage chain:
        raw_data → features → model, predictions
        """
        folio = DataFolio(path)
        folio.add_numpy("raw_data", np.array([1, 2, 3]))
        folio.add_numpy("features", np.array([4, 5, 6]), inputs=["raw_data"])
        folio.add_numpy("model", np.array([0.1, 0.2]), inputs=["features"])
        folio.add_numpy(
            "predictions", np.array([0, 1, 0]), inputs=["features", "model"]
        )
        folio.add_numpy("unrelated", np.array([99]))
        return folio

    def test_follow_lineage_includes_transitive_deps(self, tmp_path):
        """follow_lineage=True pulls in all upstream items of the seeds."""
        src = self._build_lineage_folio(tmp_path / "src")

        dst = src.copy(
            str(tmp_path / "dst"),
            include_items=["predictions"],
            follow_lineage=True,
        )

        assert "predictions" in dst._items
        assert "features" in dst._items
        assert "raw_data" in dst._items
        assert "model" in dst._items

    def test_follow_lineage_excludes_unrelated(self, tmp_path):
        """follow_lineage=True does not copy items outside the seed's lineage."""
        src = self._build_lineage_folio(tmp_path / "src")

        dst = src.copy(
            str(tmp_path / "dst"),
            include_items=["predictions"],
            follow_lineage=True,
        )

        assert "unrelated" not in dst._items

    def test_follow_lineage_multiple_seeds(self, tmp_path):
        """follow_lineage=True works with multiple seed items."""
        folio = DataFolio(tmp_path / "src")
        folio.add_numpy("a", np.array([1]))
        folio.add_numpy("b", np.array([2]), inputs=["a"])
        folio.add_numpy("c", np.array([3]))
        folio.add_numpy("d", np.array([4]), inputs=["c"])

        dst = folio.copy(
            str(tmp_path / "dst"),
            include_items=["b", "d"],
            follow_lineage=True,
        )

        assert "a" in dst._items
        assert "b" in dst._items
        assert "c" in dst._items
        assert "d" in dst._items

    def test_follow_lineage_skips_external_refs(self, tmp_path):
        """follow_lineage=True silently skips inputs not present in the folio."""
        folio = DataFolio(tmp_path / "src")
        # 'external_table' is referenced in lineage but not in this folio
        folio.add_numpy("result", np.array([1]), inputs=["external_table"])

        # Should not raise; external_table is simply omitted
        dst = folio.copy(
            str(tmp_path / "dst"),
            include_items=["result"],
            follow_lineage=True,
        )

        assert "result" in dst._items
        assert "external_table" not in dst._items

    def test_follow_lineage_without_include_items_is_noop(self, tmp_path):
        """follow_lineage=True without include_items copies everything (no change)."""
        src = self._build_lineage_folio(tmp_path / "src")

        dst = src.copy(
            str(tmp_path / "dst"),
            follow_lineage=True,
        )

        assert set(dst._items.keys()) == set(src._items.keys())

    def test_follow_lineage_respects_archived_exclusion(self, tmp_path):
        """follow_lineage=True excludes archived deps unless include_archived=True."""
        folio = DataFolio(tmp_path / "src")
        folio.add_numpy("raw", np.array([1]))
        folio.add_numpy("processed", np.array([2]), inputs=["raw"])
        folio.add_numpy("result", np.array([3]), inputs=["processed"])
        folio.archive("raw")  # raw is archived

        # Without include_archived — 'raw' should be excluded
        dst = folio.copy(
            str(tmp_path / "dst"),
            include_items=["result"],
            follow_lineage=True,
        )
        assert "result" in dst._items
        assert "processed" in dst._items
        assert "raw" not in dst._items

        # With include_archived — 'raw' should be included
        dst2 = folio.copy(
            str(tmp_path / "dst2"),
            include_items=["result"],
            follow_lineage=True,
            include_archived=True,
        )
        assert "raw" in dst2._items


class TestDescribeArchived:
    """Tests for describe() interaction with archived items."""

    def test_describe_hides_archived_by_default(self, tmp_path):
        """describe() does not show archived items by default."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("visible", np.array([1]))
        folio.add_numpy("hidden", np.array([2]))
        folio.archive("hidden")

        text = folio.describe(return_string=True)

        assert "visible" in text
        assert "hidden" not in text

    def test_describe_shows_archived_with_flag(self, tmp_path):
        """describe(include_archived=True) shows archived items."""
        folio = DataFolio(tmp_path / "test")
        folio.add_numpy("visible", np.array([1]))
        folio.add_numpy("hidden", np.array([2]))
        folio.archive("hidden")

        text = folio.describe(return_string=True, include_archived=True)

        assert "visible" in text
        assert "hidden" in text
