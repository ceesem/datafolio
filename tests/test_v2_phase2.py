"""Regression tests for the V2 phase-2 infrastructure fixes.

Covers: item-name grammar (path-traversal), accessor hardening,
MetadataDict pop/popitem/|= semantics, and diff_from_snapshot default.
"""

import time

import pandas as pd
import pytest

from datafolio import DataFolio
from datafolio.utils import validate_item_name


class TestItemNameGrammar:
    """validate_item_name enforces a strict, traversal-safe grammar."""

    @pytest.mark.parametrize(
        "name",
        [
            "simple",
            "model-v1.2",
            "with_underscore",
            "examples/weights",
            "a/b/c",
            "0numeric-start",
        ],
    )
    def test_valid_names(self, name):
        validate_item_name(name)  # Must not raise

    @pytest.mark.parametrize(
        "name",
        [
            "../escape",
            "../../escape",
            "a/../b",
            "/absolute",
            "a//b",
            "a/",
            ".",
            "..",
            "_leading_underscore",
            "ns/_leading_underscore",
            "has space",
            "-leading-dash",
            "my@model",
            "",
        ],
    )
    def test_invalid_names(self, name):
        with pytest.raises((ValueError, TypeError)):
            validate_item_name(name)

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_item_name(0)
        with pytest.raises(TypeError):
            validate_item_name(None)

    def test_traversal_name_cannot_write_outside_bundle(self, tmp_path):
        """A '..' name must be rejected before any payload is written."""
        folio = DataFolio(tmp_path / "inner" / "bundle")
        with pytest.raises(ValueError):
            folio.add("../../escape", {"a": 1})
        # Nothing escaped the bundle
        assert not (tmp_path / "escape--r2.json").exists()
        assert not list(tmp_path.glob("escape*"))

    def test_namespaced_name_stays_inside_category_dir(self, tmp_path):
        folio = DataFolio(tmp_path / "bundle")
        folio.add("examples/data", pd.DataFrame({"x": [1]}))
        filename = folio._items["examples/data"]["filename"]
        payload = (tmp_path / "bundle" / "tables" / filename).resolve()
        assert payload.is_file()
        assert (tmp_path / "bundle" / "tables") in payload.parents


class TestAccessorHardening:
    """Item names must never shadow DataAccessor internals."""

    def test_legacy_underscore_item_does_not_break_accessor(self, tmp_path):
        """Names like '_folio' are now rejected at add time, but a manifest
        written by an older version may still contain them — the accessor must
        skip them rather than shadow its own attributes."""
        folio = DataFolio(tmp_path / "bundle")
        folio.add("good", 1)
        # Simulate a legacy manifest entry with a reserved-looking name
        folio._items["_folio"] = {
            "name": "_folio",
            "item_type": "json_data",
            "filename": "x.json",
            "in_snapshots": [],
            "is_current": True,
        }
        # Must not RecursionError / TypeError
        accessor = folio.data
        assert accessor.good.content == 1
        # The unsafe name is not a class attribute, but stays reachable
        # via dictionary-style access.
        assert accessor["_folio"] is not None

    def test_method_shadowing_item_is_skipped(self, tmp_path):
        folio = DataFolio(tmp_path / "bundle")
        folio._items["_sync_items"] = {
            "name": "_sync_items",
            "item_type": "json_data",
            "filename": "y.json",
            "in_snapshots": [],
            "is_current": True,
        }
        accessor = folio.data
        # The real method survives
        assert callable(accessor._sync_items)

    def test_namespaced_item_reachable_by_key(self, tmp_path):
        folio = DataFolio(tmp_path / "bundle")
        folio.add("ns/item", {"v": 2})
        assert folio.data["ns/item"].content == {"v": 2}


class TestMetadataDictCompleteness:
    """pop/popitem/|= respect read-only mode and persist."""

    def test_pop_persists(self, tmp_path):
        folio = DataFolio(tmp_path / "bundle")
        folio.metadata["author"] = "alice"
        folio.metadata.pop("author")
        reloaded = DataFolio(tmp_path / "bundle")
        assert "author" not in reloaded.metadata

    def test_pop_read_only_raises(self, tmp_path):
        folio = DataFolio(tmp_path / "bundle")
        folio.metadata["author"] = "alice"
        ro = DataFolio(tmp_path / "bundle", read_only=True)
        with pytest.raises(RuntimeError):
            ro.metadata.pop("author")
        assert ro.metadata["author"] == "alice"

    def test_pop_missing_key_with_default(self, tmp_path):
        folio = DataFolio(tmp_path / "bundle")
        assert folio.metadata.pop("nope", "fallback") == "fallback"
        with pytest.raises(KeyError):
            folio.metadata.pop("nope")

    def test_popitem_read_only_raises(self, tmp_path):
        folio = DataFolio(tmp_path / "bundle")
        folio.metadata["k"] = "v"
        ro = DataFolio(tmp_path / "bundle", read_only=True)
        with pytest.raises(RuntimeError):
            ro.metadata.popitem()

    def test_ior_persists_and_respects_read_only(self, tmp_path):
        folio = DataFolio(tmp_path / "bundle")
        folio.metadata |= {"merged": True}
        assert DataFolio(tmp_path / "bundle").metadata["merged"] is True
        ro = DataFolio(tmp_path / "bundle", read_only=True)
        with pytest.raises(RuntimeError):
            ro.metadata |= {"nope": 1}


class TestDiffDefaultSnapshot:
    """diff_from_snapshot(None) compares against the MOST RECENT snapshot."""

    def test_default_uses_newest(self, tmp_path):
        folio = DataFolio(tmp_path / "bundle")
        folio.add("cfg", {"v": 1})
        folio.create_snapshot("older")
        time.sleep(0.01)
        folio.create_snapshot("newer")
        diff = folio.diff_from_snapshot()
        assert diff["snapshot_name"] == "newer"
