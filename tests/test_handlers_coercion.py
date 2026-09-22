"""Unit tests for the handler hooks used by add()'s type coercion."""

import pandas as pd
import pytest

from datafolio import DataFolio
from datafolio.handlers.json_data import JsonHandler
from datafolio.handlers.tables import DataframeHandler


@pytest.fixture
def folio(tmp_path):
    return DataFolio(tmp_path / "f")


class TestDataframeHandlerSeries:
    def test_series_name_recorded_in_metadata(self, folio):
        df = pd.DataFrame({"n": [1, 2]})
        metadata = DataframeHandler().add(folio, "t", df, _series_name="n")
        assert metadata["series"] == {"name": "n"}

    def test_unnamed_series_recorded_as_none(self, folio):
        df = pd.DataFrame({"value": [1, 2]})
        metadata = DataframeHandler().add(folio, "t", df, _series_name=None)
        assert metadata["series"] == {"name": None}

    def test_plain_frame_has_no_series_field(self, folio):
        metadata = DataframeHandler().add(folio, "t", pd.DataFrame({"a": [1]}))
        assert "series" not in metadata

    def test_get_returns_series_when_marked(self, folio):
        handler = DataframeHandler()
        df = pd.DataFrame({"value": [1, 2]})
        folio._items["t"] = handler.add(folio, "t", df, _series_name=None)
        result = handler.get(folio, "t")
        pd.testing.assert_series_equal(result, pd.Series([1, 2], name=None))

    def test_get_returns_frame_if_marker_does_not_fit(self, folio):
        # A marker on a multi-column table (e.g. a hand-edited manifest) is
        # ignored rather than silently dropping columns.
        handler = DataframeHandler()
        df = pd.DataFrame({"a": [1], "b": [2]})
        folio._items["t"] = handler.add(folio, "t", df, _series_name="a")
        pd.testing.assert_frame_equal(handler.get(folio, "t"), df)


class TestJsonHandlerDataType:
    def test_data_type_defaults_to_python_type(self, folio):
        metadata = JsonHandler().add(folio, "j", {"a": 1})
        assert metadata["data_type"] == "dict"

    def test_data_type_override(self, folio):
        metadata = JsonHandler().add(folio, "j", {"a": 1}, _data_type="Config")
        assert metadata["data_type"] == "Config"
