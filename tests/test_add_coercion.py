"""Tests for add() accepting near-miss types (tuples, sets, Series, Index...)."""

import dataclasses
import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from datafolio import DataFolio
from datafolio.handlers.json_data import to_json_compatible


def _roundtrip(tmp_path, obj):
    """Add obj, reopen the folio from disk, and return what get() gives back."""
    folio = DataFolio(tmp_path / "f")
    folio.add("item", obj)
    return DataFolio(tmp_path / "f").get("item")


class TestJsonContainers:
    @pytest.mark.parametrize(
        "obj, expected",
        [
            ((1, 2, 3), [1, 2, 3]),
            (range(3), [0, 1, 2]),
            ([(1, 2), (3, 4)], [[1, 2], [3, 4]]),
            ({"a": (1, (2, 3))}, {"a": [1, [2, 3]]}),
        ],
    )
    def test_tuples_and_ranges_become_lists_silently(self, tmp_path, obj, expected):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _roundtrip(tmp_path, obj) == expected

    @pytest.mark.parametrize(
        "obj, expected",
        [
            ({3, 1, 2}, [1, 2, 3]),
            (frozenset("cab"), ["a", "b", "c"]),
            ({"tags": {"z", "a"}}, {"tags": ["a", "z"]}),
            ([{2, 1}], [[1, 2]]),
        ],
    )
    def test_sets_become_sorted_lists_with_warning(self, tmp_path, obj, expected):
        with pytest.warns(UserWarning, match="sets are stored as lists"):
            assert _roundtrip(tmp_path, obj) == expected

    def test_unsortable_set_keeps_all_members(self, tmp_path):
        with pytest.warns(UserWarning):
            result = _roundtrip(tmp_path, {1, "a"})
        assert sorted(result, key=str) == [1, "a"]

    def test_non_str_keys_become_strings_with_warning(self, tmp_path):
        with pytest.warns(UserWarning, match="non-str dict keys"):
            result = _roundtrip(tmp_path, {1: "a", 2: {3: "b"}})
        assert result == {"1": "a", "2": {"3": "b"}}

    def test_colliding_keys_raise(self, tmp_path):
        folio = DataFolio(tmp_path / "f")
        with pytest.raises(ValueError, match="collide"):
            folio.add("item", {1: "a", "1": "b"})
        assert "item" not in folio

    def test_unserializable_value_still_raises(self, tmp_path):
        folio = DataFolio(tmp_path / "f")
        with pytest.raises(TypeError, match="not JSON-serializable"):
            folio.add("item", ({"a": object()},))


@dataclasses.dataclass
class _Inner:
    tags: Any = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class _Config:
    lr: float
    layers: tuple
    inner: _Inner = dataclasses.field(default_factory=_Inner)


class TestDataclass:
    def test_stored_as_field_dict(self, tmp_path):
        cfg = _Config(lr=0.01, layers=(64, 32))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _roundtrip(tmp_path, cfg)
        assert result == {"lr": 0.01, "layers": [64, 32], "inner": {"tags": []}}

    def test_flat_dataclass_rebuilds_with_kwargs(self, tmp_path):
        @dataclasses.dataclass
        class Flat:
            a: int
            b: str

        assert Flat(**_roundtrip(tmp_path, Flat(1, "x"))) == Flat(1, "x")

    def test_nested_set_still_warns(self, tmp_path):
        with pytest.warns(UserWarning, match="sets are stored as lists"):
            result = _roundtrip(tmp_path, _Config(0.1, (), _Inner({2, 1})))
        assert result["inner"] == {"tags": [1, 2]}

    def test_data_type_names_the_class(self, tmp_path):
        folio = DataFolio(tmp_path / "f")
        folio.add("cfg", _Config(lr=0.01, layers=()))
        assert folio._items["cfg"]["data_type"] == "_Config"
        assert "type: _Config" in folio.describe(return_string=True)

    def test_dataclass_class_itself_is_rejected(self, tmp_path):
        folio = DataFolio(tmp_path / "f")
        with pytest.raises(TypeError, match="Unsupported data type"):
            folio.add("cfg", _Config)


class TestSeries:
    def test_named_series_with_index_roundtrips(self, tmp_path):
        s = pd.Series([1, 2, 3], index=["x", "y", "z"], name="counts")
        pd.testing.assert_series_equal(_roundtrip(tmp_path, s), s)

    def test_unnamed_series_roundtrips(self, tmp_path):
        s = pd.Series([1.5, 2.5])
        pd.testing.assert_series_equal(_roundtrip(tmp_path, s), s)

    def test_value_counts_roundtrips(self, tmp_path):
        s = pd.Series(["a", "b", "a"]).value_counts()
        pd.testing.assert_series_equal(_roundtrip(tmp_path, s), s)

    def test_series_stored_as_table(self, tmp_path):
        folio = DataFolio(tmp_path / "f")
        folio.add("item", pd.Series([1, 2], name="v"))
        assert folio._items["item"]["item_type"] == "included_table"
        polars_frame = folio.get("item", frame="polars")
        assert isinstance(polars_frame, pl.DataFrame)
        assert polars_frame.columns == ["v"]

    def test_polars_series(self, tmp_path):
        result = _roundtrip(tmp_path, pl.Series("z", [1, 2]))
        pd.testing.assert_series_equal(result, pd.Series([1, 2], name="z"))

    def test_preserve_index_cannot_be_passed_for_series(self, tmp_path):
        folio = DataFolio(tmp_path / "f")
        with pytest.raises(TypeError, match="set automatically for Series"):
            folio.add("item", pd.Series([1]), preserve_index=True)


class TestIndex:
    def test_numeric_index_becomes_array(self, tmp_path):
        result = _roundtrip(tmp_path, pd.Index([1, 2, 3]))
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_string_index_becomes_list(self, tmp_path):
        assert _roundtrip(tmp_path, pd.Index(["a", "b"])) == ["a", "b"]


class TestToJsonCompatible:
    def test_no_notes_for_plain_data(self):
        assert to_json_compatible({"a": [1, (2,)]}) == ({"a": [1, [2]]}, [])

    def test_reports_each_lossy_conversion_once(self):
        converted, notes = to_json_compatible({1: {2, 1}, 2: [{3}]})
        assert converted == {"1": [1, 2], "2": [[3]]}
        assert notes == ["non-str dict keys", "set"]

    def test_bool_and_none_keys_use_json_spelling(self):
        converted, _ = to_json_compatible({True: 1, None: 2})
        assert converted == {"true": 1, "null": 2}

    def test_unencodable_key_raises(self):
        with pytest.raises(TypeError, match="cannot be stored as JSON"):
            to_json_compatible({(1, 2): "a"})
