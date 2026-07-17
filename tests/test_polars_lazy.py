"""Tests for polars lazy/eager access and reference-table parity.

Covers:
- readers.scan_parquet / scan_csv / scan_table (native + byte fallback)
- handler.get_lazy on both table kinds; reference schema enrichment
- folio.get_lazy, get_table(frame="polars"), the eager-load size guard
- snapshot + accessor parity, reference_table overwrite/copy-on-write
- read_arrow round-trip fix
"""

import io

import pandas as pd
import pytest

from datafolio import DataFolio

pl = pytest.importorskip("polars")


# =============================================================================
# readers: scan helpers
# =============================================================================


class TestScanReaders:
    def test_scan_parquet_local_native(self, tmp_path):
        from datafolio.readers import scan_parquet

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p = tmp_path / "t.parquet"
        df.to_parquet(p, index=False)

        lf = scan_parquet(str(p))
        assert isinstance(lf, pl.LazyFrame)
        out = lf.collect()
        assert out.shape == (3, 2)
        assert out["a"].to_list() == [1, 2, 3]

    def test_scan_parquet_pushdown(self, tmp_path):
        from datafolio.readers import scan_parquet

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p = tmp_path / "t.parquet"
        df.to_parquet(p, index=False)

        out = scan_parquet(str(p)).filter(pl.col("a") > 1).select("b").collect()
        assert out["b"].to_list() == [5, 6]

    def test_scan_parquet_strips_file_scheme(self, tmp_path):
        from datafolio.readers import scan_parquet

        df = pd.DataFrame({"a": [1, 2, 3]})
        p = tmp_path / "t.parquet"
        df.to_parquet(p, index=False)

        # resolve_path-style file:// URI should be handled
        lf = scan_parquet(f"file://{p}")
        assert lf.collect()["a"].to_list() == [1, 2, 3]

    def test_scan_parquet_non_scannable_raises(self, tmp_path, monkeypatch):
        """A non-scannable scheme must RAISE, not silently download (truthful)."""
        import datafolio.readers as readers

        df = pd.DataFrame({"a": [1, 2, 3]})
        p = tmp_path / "t.parquet"
        df.to_parquet(p, index=False)

        # Pretend nothing is natively scannable -> lazy API must refuse.
        monkeypatch.setattr(readers, "_natively_scannable", lambda path: False)

        with pytest.raises(ValueError, match="genuine lazy"):
            readers.scan_parquet(str(p))

    def test_scan_csv_local(self, tmp_path):
        from datafolio.readers import scan_csv

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p = tmp_path / "t.csv"
        df.to_csv(p, index=False)

        out = scan_csv(str(p)).collect()
        assert out["a"].to_list() == [1, 2, 3]

    def test_scan_table_dispatch(self, tmp_path):
        from datafolio.readers import scan_table

        df = pd.DataFrame({"a": [1, 2, 3]})
        p = tmp_path / "t.parquet"
        df.to_parquet(p, index=False)

        assert isinstance(scan_table(str(p), "parquet"), pl.LazyFrame)

    def test_scan_table_unsupported_format(self, tmp_path):
        from datafolio.readers import scan_table

        with pytest.raises(NotImplementedError, match="Lazy scan"):
            scan_table("s3://bucket/tbl", "delta")

    def test_natively_scannable_classifier(self):
        from datafolio.readers import _natively_scannable

        assert _natively_scannable("/local/path.parquet")
        assert _natively_scannable("file:///local/path.parquet")
        assert _natively_scannable("s3://bucket/x.parquet")
        assert _natively_scannable("gs://bucket/x.parquet")
        # http(s) is genuinely scannable by polars (range requests), not a
        # download-and-wrap fake-lazy path.
        assert _natively_scannable("https://host/x.parquet")
        assert _natively_scannable("http://host/x.parquet")

    def test_polars_scan_path_normalization(self):
        from datafolio.readers import _polars_scan_path

        assert _polars_scan_path("file:///a/b.parquet") == "/a/b.parquet"
        assert _polars_scan_path("gcs://bucket/x") == "gs://bucket/x"
        assert _polars_scan_path("s3://bucket/x") == "s3://bucket/x"


class TestReadArrowFix:
    def test_read_arrow_roundtrip(self, tmp_path):
        import pyarrow as pa
        import pyarrow.feather as feather

        from datafolio.readers import read_arrow

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p = tmp_path / "t.arrow"
        feather.write_feather(pa.Table.from_pandas(df, preserve_index=False), p)

        loaded = read_arrow(str(p))
        pd.testing.assert_frame_equal(df, loaded)


# =============================================================================
# handlers: get_lazy + reference enrichment
# =============================================================================


class TestHandlerLazy:
    def test_included_get_lazy(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        folio.add("t", df)

        lf = folio.scan_table("t")
        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect()["a"].to_list() == [1, 2, 3]

    def test_reference_get_lazy(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ext = tmp_path / "ext.parquet"
        df.to_parquet(ext, index=False)

        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", ext)

        lf = folio.scan_table("ref")
        assert isinstance(lf, pl.LazyFrame)
        assert lf.filter(pl.col("a") > 1).collect().shape == (2, 2)

    def test_reference_schema_via_inspect(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        ext = tmp_path / "ext.parquet"
        df.to_parquet(ext, index=False)

        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", ext)  # offline: no schema yet
        assert "columns" not in folio.item_info("ref")

        folio.inspect_table("ref")  # explicit enrichment
        info = folio.item_info("ref")
        assert info["columns"] == ["a", "b"]
        assert set(info["dtypes"]) == {"a", "b"}
        assert info["num_rows"] == 3

    def test_reference_creation_is_offline(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ext = tmp_path / "ext.parquet"
        df.to_parquet(ext, index=False)

        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", ext)

        info = folio.item_info("ref")
        assert "columns" not in info

    def test_reference_records_size_via_inspect(self, tmp_path):
        df = pd.DataFrame({"a": range(100)})
        ext = tmp_path / "ext.parquet"
        df.to_parquet(ext, index=False)

        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", ext)
        assert "size_bytes" not in folio.item_info("ref")

        folio.inspect_table("ref")
        info = folio.item_info("ref")
        assert info["size_bytes"] == ext.stat().st_size

    def test_included_records_size(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": range(100)}))
        info = folio.item_info("t")
        assert info["size_bytes"] > 0


# =============================================================================
# folio: frame= and pandas/polars parity
# =============================================================================


class TestFramePivot:
    def test_get_table_default_pandas(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1, 2, 3]}))
        out = folio.get("t")
        assert isinstance(out, pd.DataFrame)

    def test_get_table_polars(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1, 2, 3]}))
        out = folio.get("t", frame="polars")
        assert isinstance(out, pl.DataFrame)
        assert out["a"].to_list() == [1, 2, 3]

    def test_get_table_polars_reference(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ext = tmp_path / "ext.parquet"
        df.to_parquet(ext, index=False)
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", ext)
        out = folio.get("ref", frame="polars")
        assert isinstance(out, pl.DataFrame)

    def test_get_table_bad_frame(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="Unknown frame"):
            folio.get("t", frame="dask")


# =============================================================================
# folio: materialization guard
# =============================================================================


class TestEagerGuard:
    def _make_ref(self, tmp_path, name="ref", **kwargs):
        df = pd.DataFrame({"a": range(1000)})
        ext = tmp_path / f"{name}.parquet"
        df.to_parquet(ext, index=False)
        folio = DataFolio(tmp_path / "b", max_eager_bytes=10)
        folio.reference_table(name, ext, **kwargs)
        return folio

    def test_over_limit_blocks_eager(self, tmp_path):
        folio = self._make_ref(tmp_path)
        with pytest.raises(ValueError, match="eager-load limit"):
            folio.get("ref")

    def test_over_limit_blocks_polars_eager(self, tmp_path):
        folio = self._make_ref(tmp_path)
        with pytest.raises(ValueError, match="eager-load limit"):
            folio.get("ref", frame="polars")

    def test_lazy_never_blocked(self, tmp_path):
        folio = self._make_ref(tmp_path)
        lf = folio.scan_table("ref")  # must not raise
        assert lf.collect().shape[0] == 1000

    def test_allow_full_load_per_call(self, tmp_path):
        folio = self._make_ref(tmp_path)
        out = folio.get("ref", allow_full_load=True)
        assert len(out) == 1000

    def test_allow_full_load_per_table(self, tmp_path):
        folio = self._make_ref(tmp_path, allow_full_load=True)
        assert len(folio.get("ref")) == 1000

    def test_guard_disabled(self, tmp_path):
        df = pd.DataFrame({"a": range(1000)})
        ext = tmp_path / "ext.parquet"
        df.to_parquet(ext, index=False)
        folio = DataFolio(tmp_path / "b", max_eager_bytes=None)
        folio.reference_table("ref", ext)
        assert len(folio.get("ref")) == 1000

    def test_unknown_size_allowed(self, tmp_path):
        folio = DataFolio(tmp_path / "b", max_eager_bytes=1)
        folio.add("t", pd.DataFrame({"a": [1, 2, 3]}))
        # Simulate a table whose size wasn't recorded.
        folio._items["t"].pop("size_bytes", None)
        assert len(folio.get("t")) == 3


# =============================================================================
# parity: accessors, snapshots, lifecycle
# =============================================================================


class TestParity:
    def test_accessor_lazy_and_polars(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1, 2, 3]}))
        assert isinstance(folio.data.t.lazy, pl.LazyFrame)
        assert isinstance(folio.get("t", frame="polars"), pl.DataFrame)

    def test_accessor_lazy_reference(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ext = tmp_path / "ext.parquet"
        df.to_parquet(ext, index=False)
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", ext)
        assert folio.data.ref.lazy.collect().shape == (3, 1)

    def test_snapshot_get_table_polars(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1, 2, 3]}))
        folio.create_snapshot("snap")
        snap = folio.snapshots["snap"]
        out = snap.get("t", frame="polars")
        assert isinstance(out, pl.DataFrame)

    def test_reference_overwrite(self, tmp_path):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        e1 = tmp_path / "e1.parquet"
        e2 = tmp_path / "e2.parquet"
        df1.to_parquet(e1, index=False)
        df2.to_parquet(e2, index=False)

        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", e1)
        with pytest.raises(ValueError, match="already exists"):
            folio.reference_table("ref", e2)

        folio.reference_table("ref", e2, overwrite=True)
        assert folio.item_info("ref")["path"].endswith("e2.parquet")
        assert folio.get("ref")["a"].to_list() == [1, 2, 3, 4, 5]

    def test_snapshot_get_table_pandas_still_works(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1, 2, 3]}))
        folio.create_snapshot("snap")
        out = folio.snapshots["snap"].get("t")
        assert isinstance(out, pd.DataFrame)

    def test_reference_copy_on_write(self, tmp_path):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [9, 9]})
        e1 = tmp_path / "e1.parquet"
        e2 = tmp_path / "e2.parquet"
        df1.to_parquet(e1, index=False)
        df2.to_parquet(e2, index=False)

        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", e1)
        folio.create_snapshot("snap")

        # Overwriting a snapshotted reference should copy-on-write, preserving
        # the snapshot's view.
        folio.reference_table("ref", e2, overwrite=True)
        assert folio.get("ref")["a"].to_list() == [9, 9]
        assert folio.snapshots["snap"].get("ref")["a"].to_list() == [1, 2, 3]


# =============================================================================
# sharded / partitioned polars-only tables (references to directories)
# =============================================================================


class TestShardedPolarsOnly:
    def _hive_dir(self, tmp_path, name="hive"):
        d = tmp_path / name
        pl.DataFrame({"g": ["a", "a", "b", "c"], "x": [1, 2, 3, 4]}).write_parquet(
            d, partition_by="g"
        )
        return d

    def test_directory_reference_is_polars_only(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("big", path=self._hive_dir(tmp_path))
        info = folio.item_info("big")
        assert info["is_directory"] is True
        assert info["polars_only"] is True

    def test_directory_reference_schema_via_inspect(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("big", path=self._hive_dir(tmp_path))
        assert "columns" not in folio.item_info("big")  # offline creation

        folio.inspect_table("big")
        info = folio.item_info("big")
        # partition column 'g' is included in the scanned schema
        assert set(info["columns"]) == {"g", "x"}
        assert info["num_rows"] == 4
        assert info["num_cols"] == 2

    def test_directory_reference_lazy_and_polars_work(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("big", path=self._hive_dir(tmp_path))
        assert folio.scan_table("big").select(pl.len()).collect().item() == 4
        assert folio.get("big", frame="polars").height == 4

    def test_directory_reference_pandas_friendly_error(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("big", path=self._hive_dir(tmp_path))
        with pytest.raises(ValueError, match="polars-only"):
            folio.get("big")

    def test_accessor_content_polars_only_errors(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("big", path=self._hive_dir(tmp_path))
        with pytest.raises(ValueError, match="polars-only"):
            folio.data.big.content
        # but .lazy / .polars still work
        assert isinstance(folio.data.big.lazy, pl.LazyFrame)
        assert isinstance(folio.get("big", frame="polars"), pl.DataFrame)

    def test_explicit_polars_only_single_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        p = tmp_path / "single.parquet"
        df.to_parquet(p, index=False)
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("forced", path=p, polars_only=True)
        assert folio.item_info("forced")["polars_only"] is True
        with pytest.raises(ValueError, match="polars-only"):
            folio.get("forced")
        assert folio.scan_table("forced").collect().shape == (3, 1)

    def test_single_file_reference_not_polars_only(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        p = tmp_path / "single.parquet"
        df.to_parquet(p, index=False)
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ok", path=p)
        assert "polars_only" not in folio.item_info("ok")
        assert len(folio.get("ok")) == 3  # pandas works
