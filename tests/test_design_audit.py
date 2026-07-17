"""Tests for the design-audit changes (descriptions, versioned payloads,
mutation guard, reference snapshots, schema versioning, lazy access, and
format portability).

Organized by the audit's priority numbering.
"""

import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import pytest

from datafolio import (
    ConcurrentWriteError,
    DataFolio,
    UnsupportedManifestVersionError,
)

pl = pytest.importorskip("polars")


# =============================================================================
# P1: descriptions are preserved on overwrite unless explicitly changed/removed.
# =============================================================================


class TestDescriptionSemantics:
    def test_create_with_none_omits(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1]}))
        assert "description" not in folio.item_info("t")

    def test_overwrite_with_none_preserves(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1]}), description="keep me")
        folio.add("t", pd.DataFrame({"a": [2]}), overwrite=True)
        assert folio.item_info("t")["description"] == "keep me"

    def test_overwrite_with_nonempty_replaces(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1]}), description="old")
        folio.add("t", pd.DataFrame({"a": [2]}), overwrite=True, description="new")
        assert folio.item_info("t")["description"] == "new"

    def test_overwrite_with_empty_string_removes(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1]}), description="old")
        folio.add("t", pd.DataFrame({"a": [2]}), overwrite=True, description="")
        assert "description" not in folio.item_info("t")

    @pytest.mark.parametrize(
        "add,value",
        [
            ("add", np.array([1, 2, 3])),
            ("add", {"k": 1}),
        ],
    )
    def test_preserved_across_types(self, tmp_path, add, value):
        folio = DataFolio(tmp_path / "b")
        getattr(folio, add)("x", value, description="desc")
        getattr(folio, add)("x", value, overwrite=True)
        assert folio._items["x"]["description"] == "desc"

    def test_reference_description_preserved(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("r", path="s3://bucket/a.parquet", description="linked")
        folio.reference_table("r", path="s3://bucket/b.parquet", overwrite=True)
        assert folio.item_info("r")["description"] == "linked"

    def test_timestamp_description_preserved(self, tmp_path):
        from datetime import datetime, timezone

        folio = DataFolio(tmp_path / "b")
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        folio.add("ts", dt, description="event")
        folio.add("ts", dt, overwrite=True)
        assert folio._items["ts"]["description"] == "event"

    def test_persists_after_reopen(self, tmp_path):
        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add("t", pd.DataFrame({"a": [1]}), description="keep")
        folio.add("t", pd.DataFrame({"a": [2]}), overwrite=True)
        reopened = DataFolio(bundle)
        assert reopened.item_info("t")["description"] == "keep"

    def test_generic_add_data_preserves(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1]}), description="via generic")
        folio.add("t", pd.DataFrame({"a": [2]}), overwrite=True)
        assert folio.item_info("t")["description"] == "via generic"

    def test_snapshot_cow_preserves_description(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1]}), description="v1 desc")
        folio.create_snapshot("snap")
        folio.add("t", pd.DataFrame({"a": [2]}), overwrite=True)
        # current keeps the preserved description; snapshot version keeps its own
        assert folio.item_info("t")["description"] == "v1 desc"
        snap_item = folio._snapshot_versions[0]
        assert snap_item["description"] == "v1 desc"

    def test_batch_preserves_description(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("c", {"a": 1}, description="cfg")
        with folio.batch():
            folio.add("c", {"a": 2}, overwrite=True)
        assert folio._items["c"]["description"] == "cfg"


# =============================================================================
# P4: versioned payload filenames never clobber committed/ snapshotted bytes.
# =============================================================================


class TestVersionedPayloads:
    def test_filenames_are_versioned(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("features", pd.DataFrame({"a": [1]}))
        fn = folio._items["features"]["filename"]
        assert fn.startswith("features--r") and fn.endswith(".parquet")

    def test_overwrite_writes_new_file_not_clobber(self, tmp_path):
        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add("t", pd.DataFrame({"a": [1]}))
        first = folio._items["t"]["filename"]
        folio.add("t", pd.DataFrame({"a": [2]}), overwrite=True)
        second = folio._items["t"]["filename"]
        assert first != second

    def test_obsolete_unsnapshotted_payload_deleted(self, tmp_path):
        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add("t", pd.DataFrame({"a": [1]}))
        first = folio._items["t"]["filename"]
        folio.add("t", pd.DataFrame({"a": [2]}), overwrite=True)
        assert not (bundle / "tables" / first).exists()

    def test_snapshotted_payload_retained(self, tmp_path):
        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add("t", pd.DataFrame({"a": [1]}))
        first = folio._items["t"]["filename"]
        folio.create_snapshot("snap")
        folio.add("t", pd.DataFrame({"a": [2]}), overwrite=True)
        # the snapshotted version's file must survive
        assert (bundle / "tables" / first).exists()

    def test_batch_multiple_replacements_collision_safe(self, tmp_path):
        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        with folio.batch():
            folio.add("t", pd.DataFrame({"a": [1]}))
            folio.add("t", pd.DataFrame({"a": [2]}), overwrite=True)
            folio.add("t", pd.DataFrame({"a": [3]}), overwrite=True)
        assert folio.get("t")["a"].to_list() == [3]
        # only the final payload survives; no collisions occurred
        remaining = list((bundle / "tables").glob("*.parquet"))
        assert len(remaining) == 1
        assert remaining[0].name == folio._items["t"]["filename"]

    def test_legacy_stable_filename_readable(self, tmp_path):
        import orjson

        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add("t", pd.DataFrame({"a": [1, 2, 3]}))
        # Rewrite to a legacy stable filename on disk + in manifest.
        tables = bundle / "tables"
        versioned = folio._items["t"]["filename"]
        (tables / versioned).rename(tables / "t.parquet")
        items_path = bundle / "items.json"
        data = orjson.loads(items_path.read_bytes())
        for item in data["items"]:
            if item["name"] == "t":
                item["filename"] = "t.parquet"
        items_path.write_bytes(orjson.dumps(data))

        reopened = DataFolio(bundle)
        assert reopened.get("t")["a"].to_list() == [1, 2, 3]


# =============================================================================
# P3: mutation guard — full mutation serialized, stale writer rejected before
# any payload is written, lock released after exceptions, batch atomic.
# =============================================================================


class TestMutationGuard:
    def test_two_instances_same_revision_second_write_is_stale(self, tmp_path):
        bundle = tmp_path / "b"
        a = DataFolio(bundle)
        a.add("x", pd.DataFrame({"a": [1]}))
        b = DataFolio(bundle)  # loaded at current revision
        a.add("y", pd.DataFrame({"a": [2]}))  # advances revision
        with pytest.raises(ConcurrentWriteError):
            b.add("z", pd.DataFrame({"a": [3]}))

    def test_stale_overwrite_does_not_corrupt_committed_payload(self, tmp_path):
        bundle = tmp_path / "b"
        a = DataFolio(bundle)
        a.add("data", pd.DataFrame({"a": [1]}))
        b = DataFolio(bundle)
        a.add("data", pd.DataFrame({"a": [2]}), overwrite=True)
        committed = a._items["data"]["filename"]
        with pytest.raises(ConcurrentWriteError):
            b.add("data", pd.DataFrame({"a": [999]}), overwrite=True)
        # A fresh reader sees A's committed value, uncorrupted.
        fresh = DataFolio(bundle)
        assert fresh.get("data")["a"].to_list() == [2]
        assert fresh._items["data"]["filename"] == committed

    def test_stale_new_item_rejected(self, tmp_path):
        bundle = tmp_path / "b"
        a = DataFolio(bundle)
        a.add("x", pd.DataFrame({"a": [1]}))
        b = DataFolio(bundle)
        a.add("cfg", {"a": 1})
        with pytest.raises(ConcurrentWriteError):
            b.add("other", {"b": 2})

    def test_lock_released_after_exception(self, tmp_path):
        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        # Force an error inside a mutation (numpy handler rejects
        # object-dtype arrays).
        with pytest.raises((TypeError, ValueError)):
            folio.add("bad", np.array([object()], dtype=object))
        # The lock was released — a subsequent write succeeds.
        folio.add("t", pd.DataFrame({"a": [1]}))
        assert folio.get("t")["a"].to_list() == [1]

    def test_readers_do_not_block_on_lock(self, tmp_path):
        # A reader opening a second instance never needs the write lock.
        bundle = tmp_path / "b"
        w = DataFolio(bundle)
        w.add("t", pd.DataFrame({"a": [1, 2, 3]}))
        reader = DataFolio(bundle, read_only=True)
        assert reader.get("t")["a"].to_list() == [1, 2, 3]

    def test_batch_is_atomic_under_staleness(self, tmp_path):
        bundle = tmp_path / "b"
        a = DataFolio(bundle)
        a.add("seed", pd.DataFrame({"a": [1]}))
        b = DataFolio(bundle)
        a.add("advance", pd.DataFrame({"a": [2]}))  # b now stale
        # The whole batch is rejected as a unit (stale check at guard entry).
        with pytest.raises(ConcurrentWriteError):
            with b.batch():
                b.add("x", {"a": 1})
                b.add("y", {"a": 2})
        fresh = DataFolio(bundle)
        assert "x" not in fresh._items
        assert "y" not in fresh._items


def _hammer_worker(bundle, name, barrier=None):
    """Worker: open a folio and try to add a uniquely-named table."""
    try:
        folio = DataFolio(bundle)
        folio.add(name, pd.DataFrame({"a": [1, 2, 3]}))
        return "ok"
    except ConcurrentWriteError:
        return "stale"
    except Exception as exc:  # pragma: no cover - surfaced in assertion
        return f"error:{type(exc).__name__}:{exc}"


class TestMutationGuardMultiprocess:
    def test_concurrent_writers_do_not_corrupt_manifest(self, tmp_path):
        bundle = str(tmp_path / "b")
        # Seed the bundle so all workers load the same starting revision.
        DataFolio(bundle).add("seed", pd.DataFrame({"a": [0]}))

        ctx = mp.get_context("spawn")
        with ctx.Pool(4) as pool:
            results = pool.starmap(
                _hammer_worker,
                [(bundle, f"w{i}") for i in range(4)],
            )

        # At least one writer commits; the manifest is always readable and
        # never corrupted (this is the key safety property). Stale losers must
        # refresh and retry — that's the contract, not a corruption.
        assert any(r == "ok" for r in results)
        assert all(r in ("ok", "stale") for r in results), results
        reopened = DataFolio(bundle)
        # The committed manifest is valid JSON with a consistent revision.
        assert reopened._manifest_revision is not None
        assert "seed" in reopened._items


# =============================================================================
# P5: snapshotted reference version handling.
# =============================================================================


class TestReferenceSnapshotLifecycle:
    def _bundle_with_ref(self, tmp_path):
        d = tmp_path
        pd.DataFrame({"a": [1, 2, 3]}).to_parquet(d / "v1.parquet")
        pd.DataFrame({"a": [9, 9]}).to_parquet(d / "v2.parquet")
        folio = DataFolio(d / "b")
        folio.reference_table("ref", path=str(d / "v1.parquet"), description="first")
        folio.create_snapshot("snap")
        folio.reference_table("ref", path=str(d / "v2.parquet"), overwrite=True)
        return folio

    def test_old_reference_marked_noncurrent(self, tmp_path):
        folio = self._bundle_with_ref(tmp_path)
        assert folio._items["ref"]["is_current"] is True
        assert len(folio._snapshot_versions) == 1
        old = folio._snapshot_versions[0]
        assert old["is_current"] is False
        assert old["version_id"]

    def test_current_is_only_current_with_name(self, tmp_path):
        folio = self._bundle_with_ref(tmp_path)
        currents = [
            it
            for it in list(folio._items.values()) + folio._snapshot_versions
            if it["name"] == "ref" and it.get("is_current")
        ]
        assert len(currents) == 1

    def test_snapshot_view_returns_recorded_reference(self, tmp_path):
        folio = self._bundle_with_ref(tmp_path)
        assert folio.snapshots["snap"].get("ref")["a"].to_list() == [1, 2, 3]

    def test_reopen_preserves_current_reference(self, tmp_path):
        folio = self._bundle_with_ref(tmp_path)
        reopened = DataFolio(folio._bundle_dir)
        assert reopened.get("ref")["a"].to_list() == [9, 9]
        assert reopened.snapshots["snap"].get("ref")["a"].to_list() == [1, 2, 3]

    def test_load_snapshot_returns_exact_descriptor(self, tmp_path):
        folio = self._bundle_with_ref(tmp_path)
        paper = DataFolio.load_snapshot(folio._bundle_dir, "snap")
        info = paper.item_info("ref")
        assert info["path"].endswith("v1.parquet")
        assert info["description"] == "first"
        assert paper.get("ref")["a"].to_list() == [1, 2, 3]


# =============================================================================
# P6: manifest schema version validation.
# =============================================================================


class TestSchemaVersioning:
    def test_future_version_rejected(self, tmp_path):
        import orjson

        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add("t", pd.DataFrame({"a": [1]}))
        items_path = bundle / "items.json"
        data = orjson.loads(items_path.read_bytes())
        data["schema_version"] = 999
        items_path.write_bytes(orjson.dumps(data))
        with pytest.raises(UnsupportedManifestVersionError, match="[Uu]pgrade"):
            DataFolio(bundle)

    def test_current_version_accepted(self, tmp_path):
        bundle = tmp_path / "b"
        DataFolio(bundle).add("t", pd.DataFrame({"a": [1]}))
        reopened = DataFolio(bundle)
        assert reopened.get("t")["a"].to_list() == [1]

    def test_pre_versioning_dict_migrated(self, tmp_path):
        import orjson

        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add("t", pd.DataFrame({"a": [1]}))
        items_path = bundle / "items.json"
        data = orjson.loads(items_path.read_bytes())
        del data["schema_version"]  # pre-versioning dict manifest
        data.pop("revision", None)
        items_path.write_bytes(orjson.dumps(data))
        reopened = DataFolio(bundle)
        assert reopened.get("t")["a"].to_list() == [1]
        reopened.add("u", pd.DataFrame({"a": [2]}))
        migrated = orjson.loads(items_path.read_bytes())
        assert migrated["schema_version"] >= 1


# =============================================================================
# P7: lazy access for references (snapshot guards + scan; cloud metadata-only).
# =============================================================================


class TestLazySnapshotAccess:
    def test_snapshot_scan_table_is_lazy(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        folio.create_snapshot("snap")
        lf = folio.snapshots["snap"].scan_table("t")
        assert isinstance(lf, pl.LazyFrame)
        assert lf.filter(pl.col("a") > 1).select("b").collect()["b"].to_list() == [5, 6]

    def test_snapshot_get_table_polars_only_guard(self, tmp_path):
        d = tmp_path
        pl.DataFrame({"g": ["a", "b"], "x": [1, 2]}).write_parquet(
            d / "hive", partition_by="g"
        )
        folio = DataFolio(d / "b")
        folio.reference_table("big", path=str(d / "hive"))
        folio.create_snapshot("snap")
        with pytest.raises(ValueError, match="polars-only"):
            folio.snapshots["snap"].get("big")

    def test_snapshot_get_table_eager_size_guard(self, tmp_path):
        df = pd.DataFrame({"a": range(1000)})
        ext = tmp_path / "e.parquet"
        df.to_parquet(ext, index=False)
        folio = DataFolio(tmp_path / "b", max_eager_bytes=10)
        folio.reference_table("ref", path=str(ext))
        folio.create_snapshot("snap")
        with pytest.raises(ValueError, match="eager-load limit"):
            folio.snapshots["snap"].get("ref")

    def test_cloud_exists_is_metadata_only(self, tmp_path):
        """StorageBackend.exists must not download the object to test presence."""
        from datafolio.storage.backend import StorageBackend

        backend = StorageBackend()
        calls = {"get": 0, "exists": 0}

        class FakeCF:
            def __init__(self, *a, **k):
                pass

            def get(self, *a, **k):  # pragma: no cover - must not be called
                calls["get"] += 1
                return b"data"

            def exists(self, *a, **k):
                calls["exists"] += 1
                return True

        import cloudfiles

        # Patch CloudFiles used inside exists().
        orig = cloudfiles.CloudFiles
        cloudfiles.CloudFiles = FakeCF
        try:
            assert backend.exists("s3://bucket/data.parquet") is True
        finally:
            cloudfiles.CloudFiles = orig
        assert calls["exists"] >= 1
        assert calls["get"] == 0


# =============================================================================
# P2: the on-disk format is usable without datafolio.
# =============================================================================


class TestFormatPortability:
    def test_reconstruct_inventory_without_datafolio(self, tmp_path):
        """Reconstruct the inventory and read standard objects using only JSON +
        filesystem + standard format libraries — no datafolio reading APIs."""
        import json
        from datetime import datetime, timezone

        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add("features", pd.DataFrame({"a": [1, 2, 3]}))
        folio.add("emb", np.array([1.0, 2.0, 3.0]))
        folio.add("config", {"lr": 0.01})
        folio.add("run_at", datetime(2024, 1, 1, tzinfo=timezone.utc))
        txt = tmp_path / "notes.txt"
        txt.write_text("hello")
        folio.add_file(txt, name="notes")
        folio.reference_table("ext", path="s3://bucket/huge.parquet")
        del folio

        # --- Read ONLY items.json + files, no DataFolio methods. ---
        manifest = json.loads((bundle / "items.json").read_bytes())
        assert manifest["schema_version"] >= 1
        current = [it for it in manifest["items"] if it.get("is_current", True)]
        by_name = {it["name"]: it for it in current}
        assert set(by_name) == {"features", "emb", "config", "run_at", "notes", "ext"}

        subdir_for = {
            "included_table": "tables",
            "model": "models",
        }

        def resolve(item):
            sub = subdir_for.get(item["item_type"], "artifacts")
            return bundle / sub / item["filename"]

        # Parquet via pandas
        df = pd.read_parquet(resolve(by_name["features"]))
        assert df["a"].to_list() == [1, 2, 3]
        # NumPy via numpy
        arr = np.load(resolve(by_name["emb"]))
        assert arr.tolist() == [1.0, 2.0, 3.0]
        # JSON via json
        cfg = json.loads(resolve(by_name["config"]).read_bytes())
        assert cfg == {"lr": 0.01}
        # Text artifact
        assert resolve(by_name["notes"]).read_text() == "hello"
        # External reference: not owned, carries an absolute URI, no filename.
        assert "filename" not in by_name["ext"]
        assert by_name["ext"]["path"] == "s3://bucket/huge.parquet"

    def test_contents_md_is_derived_and_records_revision(self, tmp_path):
        bundle = tmp_path / "b"
        folio = DataFolio(bundle)
        folio.add("features", pd.DataFrame({"a": [1]}), description="the features")
        contents = (bundle / "CONTENTS.md").read_text()
        assert "Derived" in contents
        assert "do not edit" in contents.lower()
        assert "features" in contents
        assert "the features" in contents
        assert f"revision **{folio._manifest_revision}**" in contents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
