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


# =============================================================================
# Finding 10: relative local references stay relative in the manifest and
# resolve against the bundle, so moving/copying a bundle keeps links working.
# =============================================================================


class TestRelativeReferences:
    def test_relative_reference_stored_verbatim(self, tmp_path):
        bundle = tmp_path / "proj"
        folio = DataFolio(bundle)
        # data lives alongside the bundle contents
        pd.DataFrame({"a": [1, 2, 3]}).to_parquet(bundle / "ext.parquet", index=False)
        folio.reference_table("ref", path="ext.parquet")  # relative
        # manifest keeps it relative (not absolute file://)
        assert folio.get_table_info("ref")["path"] == "ext.parquet"

    def test_relative_reference_reads_via_bundle(self, tmp_path):
        bundle = tmp_path / "proj"
        folio = DataFolio(bundle)
        pd.DataFrame({"a": [1, 2, 3]}).to_parquet(bundle / "ext.parquet", index=False)
        folio.reference_table("ref", path="ext.parquet")
        assert folio.get_table("ref")["a"].to_list() == [1, 2, 3]

    def test_relative_reference_survives_move(self, tmp_path):
        import shutil

        src = tmp_path / "proj"
        folio = DataFolio(src)
        pd.DataFrame({"a": [1, 2, 3]}).to_parquet(src / "ext.parquet", index=False)
        folio.reference_table("ref", path="ext.parquet")
        del folio

        # Move the whole bundle (with its adjacent data) elsewhere.
        dst = tmp_path / "moved"
        shutil.copytree(src, dst)
        shutil.rmtree(src)

        moved = DataFolio(dst)
        assert moved.get_table("ref")["a"].to_list() == [1, 2, 3]

    def test_absolute_reference_unchanged(self, tmp_path):
        data = tmp_path / "ext.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(data, index=False)
        folio = DataFolio(tmp_path / "proj")
        folio.reference_table("ref", path=str(data))  # absolute
        stored = folio.get_table_info("ref")["path"]
        assert stored.startswith("file://") or stored == str(data)
        assert folio.get_table("ref")["a"].to_list() == [1]

    def test_cloud_reference_unchanged(self, tmp_path):
        folio = DataFolio(tmp_path / "proj")
        folio.reference_table("ref", path="s3://bucket/data.parquet")
        assert folio.get_table_info("ref")["path"] == "s3://bucket/data.parquet"


# =============================================================================
# Finding 3: characterize lazy scan semantics and enforce a truthful contract.
# scan_table()/get_lazy() are genuinely lazy or raise; they never silently
# perform an unguarded full download.
# =============================================================================


class TestLazyScanContract:
    def test_local_parquet_is_lazy(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_table("t", pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        lf = folio.scan_table("t")
        assert isinstance(lf, pl.LazyFrame)
        # pushdown: only matching rows/cols materialize
        assert lf.filter(pl.col("a") > 1).select("b").collect()["b"].to_list() == [5, 6]

    def test_partitioned_parquet_is_lazy(self, tmp_path):
        d = tmp_path / "hive"
        pl.DataFrame({"g": ["a", "a", "b"], "x": [1, 2, 3]}).write_parquet(
            d, partition_by="g"
        )
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("big", path=d)
        lf = folio.scan_table("big")
        assert isinstance(lf, pl.LazyFrame)
        assert lf.select(pl.len()).collect().item() == 3

    def test_get_lazy_is_alias_for_scan_table(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_table("t", pd.DataFrame({"a": [1, 2, 3]}))
        assert isinstance(folio.get_lazy("t"), pl.LazyFrame)

    def test_native_scheme_classification(self):
        from datafolio.readers import _natively_scannable

        for p in [
            "/local/x.parquet",
            "file:///x.parquet",
            "s3://b/x.parquet",
            "gs://b/x.parquet",
            "az://b/x.parquet",
            "https://h/x.parquet",
            "http://h/x.parquet",
        ]:
            assert _natively_scannable(p), p

    def test_non_scannable_scheme_raises_not_downloads(self, tmp_path, monkeypatch):
        import datafolio.readers as readers

        df = pd.DataFrame({"a": [1, 2, 3]})
        p = tmp_path / "t.parquet"
        df.to_parquet(p, index=False)
        monkeypatch.setattr(readers, "_natively_scannable", lambda path: False)
        with pytest.raises(ValueError, match="genuine lazy"):
            readers.scan_parquet(str(p))

    def test_credential_forwarding(self, tmp_path, monkeypatch):
        """storage_options must be forwarded to the native polars scanner."""
        import datafolio.readers as readers

        captured = {}

        def fake_scan(path, storage_options=None, **kwargs):
            captured["storage_options"] = storage_options
            return "LF"

        monkeypatch.setattr(readers._require_polars(), "scan_parquet", fake_scan)
        readers.scan_parquet(
            "s3://bucket/x.parquet", storage_options={"aws_region": "us-east-1"}
        )
        assert captured["storage_options"] == {"aws_region": "us-east-1"}

    def test_unsupported_lazy_format_raises(self, tmp_path):
        from datafolio.readers import scan_table

        with pytest.raises(NotImplementedError, match="Lazy scan"):
            scan_table("s3://b/data", "delta")

    def test_eager_polars_download_path_still_works(self, tmp_path):
        """get_table(frame='polars') is the explicit eager op."""
        folio = DataFolio(tmp_path / "b")
        folio.add_table("t", pd.DataFrame({"a": [1, 2, 3]}))
        out = folio.get_table("t", frame="polars")
        assert isinstance(out, pl.DataFrame)
        assert out["a"].to_list() == [1, 2, 3]


# =============================================================================
# Finding 4: add_table accepts a polars LazyFrame, materialized via a streaming
# sink (bounded memory), with the same manifest/overwrite/snapshot behavior.
# =============================================================================


class TestLazyFrameInput:
    def test_add_table_lazyframe_roundtrip(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).filter(pl.col("a") > 1)
        folio.add_table("t", lf)
        out = folio.get_table("t")
        assert out["a"].to_list() == [2, 3]
        assert out["b"].to_list() == [5, 6]

    def test_add_table_lazyframe_metadata(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        folio.add_table("t", lf)
        info = folio.get_table_info("t")
        assert info["columns"] == ["a", "b"]
        assert info["num_rows"] == 3
        assert info["num_cols"] == 2
        assert "checksum" in info
        assert "size_bytes" in info

    def test_add_table_lazyframe_overwrite_and_snapshot(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_table("t", pl.LazyFrame({"a": [1, 2, 3]}))
        folio.create_snapshot("snap")
        folio.add_table("t", pl.LazyFrame({"a": [9]}), overwrite=True)
        assert folio.get_table("t")["a"].to_list() == [9]
        assert folio.snapshots["snap"].get_table("t")["a"].to_list() == [1, 2, 3]

    def test_add_data_accepts_lazyframe(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_data("t", pl.LazyFrame({"a": [1, 2, 3]}))
        assert folio.get_table("t")["a"].to_list() == [1, 2, 3]

    def test_add_table_lazyframe_scan_roundtrip(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_table("t", pl.LazyFrame({"a": [1, 2, 3]}))
        assert folio.scan_table("t").collect()["a"].to_list() == [1, 2, 3]


# =============================================================================
# Finding 5: cloud Parquet writes must not buffer the whole file in memory.
# =============================================================================


class TestBoundedCloudWrites:
    def test_cloud_write_uses_tempfile_not_bytesio(self, tmp_path, monkeypatch):
        """write_parquet(cloud) serializes to a temp file + streams it, and
        never builds a whole-file in-memory buffer via _cloud_write_bytes."""
        import os
        import shutil

        from datafolio.storage.backend import StorageBackend

        backend = StorageBackend()
        captured = {}

        def fake_upload(dst, local_src):
            # Proof it's a real on-disk file, not an in-memory buffer.
            assert os.path.exists(local_src)
            captured["dst"] = dst
            shutil.copy(local_src, tmp_path / "uploaded.parquet")

        def boom_bytes(*a, **k):  # pragma: no cover
            raise AssertionError("whole-file BytesIO path was used")

        monkeypatch.setattr(backend, "_upload_file", fake_upload)
        monkeypatch.setattr(backend, "_cloud_write_bytes", boom_bytes)

        backend.write_parquet("s3://bucket/x.parquet", pl.DataFrame({"a": [1, 2, 3]}))

        assert captured["dst"] == "s3://bucket/x.parquet"
        assert pl.read_parquet(tmp_path / "uploaded.parquet")["a"].to_list() == [
            1,
            2,
            3,
        ]

    def test_upload_file_streams_file_object(self, tmp_path, monkeypatch):
        """_upload_file passes a file OBJECT to CloudFiles.put, not bytes."""
        from datafolio.storage.backend import StorageBackend

        backend = StorageBackend()
        src = tmp_path / "x.bin"
        src.write_bytes(b"hello world")
        captured = {}

        class FakeCF:
            def put(self, filename, content, **k):
                captured["is_file_obj"] = hasattr(content, "read")

        monkeypatch.setattr(
            backend,
            "_get_cloud_client",
            lambda dst, use_https=False: (FakeCF(), "x.bin"),
        )
        backend._upload_file("s3://bucket/x.bin", str(src))
        assert captured["is_file_obj"] is True

    def test_cloud_write_tempfile_cleaned_up(self, tmp_path, monkeypatch):
        """The temp file is removed even if the upload fails."""
        import glob
        import os
        import tempfile

        from datafolio.storage.backend import StorageBackend

        backend = StorageBackend()
        before = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.parquet")))

        def failing_upload(dst, local_src):
            raise RuntimeError("upload failed")

        monkeypatch.setattr(backend, "_upload_file", failing_upload)
        with pytest.raises(RuntimeError):
            backend.write_parquet("s3://bucket/x.parquet", pl.DataFrame({"a": [1]}))

        after = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.parquet")))
        assert after == before  # no leaked temp files


# =============================================================================
# Finding 9: one canonical (Arrow-derived) schema representation for both
# included and inspected referenced tables.
# =============================================================================


class TestArrowSchemaNormalization:
    def test_included_uses_arrow_dtypes(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_table("t", pd.DataFrame({"a": [1, 2, 3]}))
        # Arrow logical type string, not a pandas/polars display string.
        assert folio.get_table_info("t")["dtypes"]["a"] == "int64"

    def test_inspected_reference_matches_included_convention(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
        folio = DataFolio(tmp_path / "b")
        folio.add_table("inc", df)
        ext = tmp_path / "e.parquet"
        df.to_parquet(ext, index=False)
        folio.reference_table("ref", ext)
        folio.inspect_table("ref")
        # same Arrow-derived dtype strings and column order for both kinds
        assert (
            folio.get_table_info("ref")["dtypes"]
            == folio.get_table_info("inc")["dtypes"]
        )
        assert (
            folio.get_table_info("ref")["columns"]
            == folio.get_table_info("inc")["columns"]
        )

    def test_lazyframe_add_uses_arrow_dtypes(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_table("t", pl.LazyFrame({"a": [1, 2, 3]}))
        assert folio.get_table_info("t")["dtypes"]["a"] == "int64"


# =============================================================================
# Finding 7: external references are mutable; inspect captures point-in-time
# source identity; snapshots surface mutable-reference warnings.
# =============================================================================


class TestReferenceIdentity:
    def test_reference_marked_mutable(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", path="s3://bucket/data.parquet")
        assert folio.get_table_info("ref")["mutable"] is True

    def test_inspect_captures_source_identity(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ext = tmp_path / "e.parquet"
        df.to_parquet(ext, index=False)
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", ext)
        folio.inspect_table("ref")
        ident = folio.get_table_info("ref")["source_identity"]
        assert ident["size"] == ext.stat().st_size
        assert "last_modified" in ident  # from head()

    def test_mutable_references_listing(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_table("owned", pd.DataFrame({"a": [1]}))
        folio.reference_table("ext", path="s3://bucket/x.parquet")
        assert folio.mutable_references() == ["ext"]

    def test_snapshot_info_flags_mutable_references(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_table("owned", pd.DataFrame({"a": [1]}))
        folio.reference_table("ext", path="s3://bucket/x.parquet")
        folio.create_snapshot("snap")
        info = folio.get_snapshot_info("snap")
        assert "ext" in info.get("mutable_references", [])
        assert "mutable_reference_warning" in info


# =============================================================================
# Finding 8: manifest schema_version + monotonic revision, atomic local writes,
# and stale-writer detection (many readers, one writer).
# =============================================================================


class TestManifestContract:
    def _read_items(self, bundle):
        import orjson

        return orjson.loads((bundle / "items.json").read_bytes())

    def test_manifest_has_schema_version_and_revision(self, tmp_path):
        from datafolio.folio import MANIFEST_SCHEMA_VERSION

        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add_table("x", pd.DataFrame({"a": [1]}))
        data = self._read_items(bundle)
        assert data["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert isinstance(data["revision"], int)

    def test_revision_increments_monotonically(self, tmp_path):
        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add_table("x", pd.DataFrame({"a": [1]}))
        r1 = self._read_items(bundle)["revision"]
        folio.add_table("y", pd.DataFrame({"a": [2]}))
        r2 = self._read_items(bundle)["revision"]
        assert r2 == r1 + 1

    def test_backward_compatible_bare_list_manifest(self, tmp_path):
        import orjson

        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add_table("x", pd.DataFrame({"a": [1, 2, 3]}))

        # Rewrite items.json in the oldest (bare list) format.
        items_path = bundle / "items.json"
        current = orjson.loads(items_path.read_bytes())
        items_path.write_bytes(orjson.dumps(current["items"]))

        # Opening still works, and the next write migrates to the new format.
        folio2 = DataFolio(bundle)
        assert folio2.get_table("x")["a"].to_list() == [1, 2, 3]
        folio2.add_table("y", pd.DataFrame({"a": [9]}))
        migrated = orjson.loads(items_path.read_bytes())
        assert migrated["schema_version"] >= 1
        assert migrated["revision"] >= 1

    def test_stale_writer_detected(self, tmp_path):
        from datafolio import ConcurrentWriteError

        bundle = tmp_path / "b"
        writer_a = DataFolio(bundle)
        writer_a.add_table("x", pd.DataFrame({"a": [1]}))

        # Second instance loads the current revision.
        writer_b = DataFolio(bundle)

        # A advances the manifest; B is now stale.
        writer_a.add_table("y", pd.DataFrame({"a": [2]}))

        with pytest.raises(ConcurrentWriteError):
            writer_b.add_table("z", pd.DataFrame({"a": [3]}))

    def test_no_leftover_tmp_files(self, tmp_path):
        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add_table("x", pd.DataFrame({"a": [1]}))
        leftovers = list(bundle.glob("*.tmp"))
        assert leftovers == []
