"""Regression tests for the audit fixes.

Organized by finding number. Written test-first per AGENTS.md.
"""

import numpy as np
import pandas as pd
import pytest

from datafolio import DataFolio

pl = pytest.importorskip("polars")


# =============================================================================
# Finding 1: generic add_data/get_data must enforce the same invariants as the
# type-specific public methods (dup-name, overwrite, snapshot COW, guards).
# =============================================================================


class TestGenericApiInvariants:
    def test_add_data_duplicate_name_raises(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_data("t", pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(ValueError, match="already exists"):
            folio.add_data("t", pd.DataFrame({"a": [9]}))

    def test_add_data_overwrite_allows_replace(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_data("t", pd.DataFrame({"a": [1, 2, 3]}))
        folio.add_data("t", pd.DataFrame({"a": [9, 9]}), overwrite=True)
        assert folio.get_table("t")["a"].to_list() == [9, 9]

    def test_add_data_snapshot_copy_on_write(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_data("t", pd.DataFrame({"a": [1, 2, 3]}))
        folio.create_snapshot("snap")
        folio.add_data("t", pd.DataFrame({"a": [9]}), overwrite=True)
        # snapshot must still see the original data
        assert folio.snapshots["snap"].get_table("t")["a"].to_list() == [1, 2, 3]
        assert folio.get_table("t")["a"].to_list() == [9]

    def test_add_data_numpy_json_timestamp_delegate(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_data("arr", np.array([1, 2, 3]))
        folio.add_data("cfg", {"lr": 0.01})
        folio.add_data("scalar", 0.95)
        assert folio.get_data("arr").tolist() == [1, 2, 3]
        assert folio.get_data("cfg") == {"lr": 0.01}
        assert folio.get_data("scalar") == 0.95

    def test_get_data_polars_only_raises_friendly(self, tmp_path):
        # Build a partitioned dataset and reference it (auto polars_only).
        d = tmp_path / "hive"
        pl.DataFrame({"g": ["a", "b"], "x": [1, 2]}).write_parquet(d, partition_by="g")
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("big", path=d)
        with pytest.raises(ValueError, match="polars-only"):
            folio.get_data("big")

    def test_get_data_respects_eager_guard(self, tmp_path):
        df = pd.DataFrame({"a": range(1000)})
        ext = tmp_path / "e.parquet"
        df.to_parquet(ext, index=False)
        folio = DataFolio(tmp_path / "b", max_eager_bytes=10)
        folio.reference_table("ref", ext)
        with pytest.raises(ValueError, match="eager-load limit"):
            folio.get_data("ref")


# =============================================================================
# Finding 2: creating a reference must perform NO remote I/O. Enrichment is an
# explicit, separate operation (inspect_table).
# =============================================================================


class TestOfflineReferenceCreation:
    def test_reference_creation_does_no_remote_io(self, tmp_path, monkeypatch):
        """Linking an S3/GCS URI must not stat, scan, or open a client."""
        import datafolio.readers as readers
        from datafolio.storage.backend import StorageBackend

        def boom(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError("reference creation performed remote I/O")

        # Build the folio first (its __init__ legitimately constructs a local
        # CloudFiles for the bundle dir), THEN forbid any remote-I/O entrypoints.
        folio = DataFolio(tmp_path / "b")

        # file_size (HEAD) and scan_parquet (schema/data) are the remote-I/O
        # entrypoints; constructing a cloud client for the reference is also a
        # failure. (exists() on a local manifest file is fine — not remote.)
        monkeypatch.setattr(StorageBackend, "file_size", boom)
        monkeypatch.setattr(readers, "scan_parquet", boom)
        import cloudfiles

        monkeypatch.setattr(cloudfiles, "CloudFiles", lambda *a, **k: boom())

        # Should complete purely as a manifest write.
        folio.reference_table("remote", path="s3://bucket/private/data.parquet")
        info = folio.get_table_info("remote")
        assert info["path"] == "s3://bucket/private/data.parquet"
        # No auto-enriched fields.
        assert "size_bytes" not in info
        assert "columns" not in info
        assert "num_rows" not in info

    def test_reference_creation_preserves_explicit_num_rows(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table(
            "remote", path="s3://bucket/data.parquet", num_rows=1_000_000
        )
        assert folio.get_table_info("remote")["num_rows"] == 1_000_000

    def test_inspect_table_enriches_local(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        ext = tmp_path / "e.parquet"
        df.to_parquet(ext, index=False)
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", ext)
        assert "columns" not in folio.get_table_info("ref")  # offline creation

        result = folio.inspect_table("ref")
        info = folio.get_table_info("ref")
        assert info["columns"] == ["a", "b"]
        assert info["num_rows"] == 3
        assert info["size_bytes"] == ext.stat().st_size
        # inspect returns the enriched info
        assert result["num_rows"] == 3

    def test_inspect_table_unreachable_raises_actionable(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("gone", path=str(tmp_path / "does_not_exist.parquet"))
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            folio.inspect_table("gone")


# =============================================================================
# Finding 6: only accept formats that work end-to-end (parquet, csv). Delta and
# Iceberg are rejected immediately with an actionable error.
# =============================================================================


class TestUnsupportedFormats:
    @pytest.mark.parametrize("fmt", ["delta", "iceberg"])
    def test_delta_iceberg_rejected(self, tmp_path, fmt):
        folio = DataFolio(tmp_path / "b")
        with pytest.raises(ValueError, match="not supported"):
            folio.reference_table("x", path="s3://bucket/data", table_format=fmt)
        # nothing was recorded
        assert "x" not in folio._items

    @pytest.mark.parametrize("fmt", ["parquet", "csv"])
    def test_supported_formats_accepted(self, tmp_path, fmt):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("x", path=f"s3://bucket/data.{fmt}", table_format=fmt)
        assert folio.get_table_info("x")["table_format"] == fmt
