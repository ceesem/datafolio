"""Tests for preserve_index: index stored as plain columns + manifest metadata.

The parquet payload keeps index values as ordinary columns (readable by any
tool); only the pandas read path re-applies them via the manifest's
``index_columns`` field.
"""

import warnings

import pandas as pd
import pytest

from datafolio import DataFolio


class TestPreserveIndex:
    def test_named_index_round_trip(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="ts"))
        folio.add("t", df, preserve_index=True)

        back = folio.get("t")
        assert back.index.tolist() == [10, 20]
        assert back.index.name == "ts"
        assert folio._items["t"]["index_columns"] == ["ts"]

    def test_unnamed_index_round_trip(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        df = pd.DataFrame({"a": [1, 2]}, index=[7, 8])
        folio.add("t", df, preserve_index=True)

        back = folio.get("t")
        assert back.index.tolist() == [7, 8]
        assert back.index.name is None

    def test_multiindex_round_trip(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        idx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["k1", "k2"])
        df = pd.DataFrame({"v": [1, 2]}, index=idx)
        folio.add("t", df, preserve_index=True)

        back = folio.get("t")
        assert list(back.index.names) == ["k1", "k2"]
        assert back["v"].tolist() == [1, 2]

    def test_payload_keeps_plain_columns(self, tmp_path):
        """The parquet file itself stays a plain-column file — the index is
        an ordinary column for polars or any other reader (design principle:
        offload-friendly, no bespoke formats)."""
        pl = pytest.importorskip("polars")
        folio = DataFolio(tmp_path / "b")
        df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="ts"))
        folio.add("t", df, preserve_index=True)

        raw = pl.read_parquet(folio.item_path("t"))
        assert "ts" in raw.columns
        assert folio.get("t", frame="polars").columns == ["ts", "a"]

    def test_default_index_records_nothing(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1, 2]}), preserve_index=True)
        assert "index_columns" not in folio._items["t"]

    def test_dropped_index_warns_without_flag(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="ts"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            folio.add("t", df)
        assert any("reset_index" in str(x.message) for x in w)
        assert "index_columns" not in folio._items["t"]
