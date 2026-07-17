"""Regression tests for the V2 phase-3 snapshot-integrity fixes.

Each test class corresponds to a confirmed bug from the July 2026 audit:
delete() destroying snapshotted data, restore_snapshot not restoring
deleted items, export_snapshot crashing on references, create_snapshot
silently losing snapshots inside batch(), update_item leaking edits into
snapshot views, and read-only mode not being enforced on most add methods.
"""

import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from datafolio import DataFolio


class TestDeletePreservesSnapshots:
    def test_delete_snapshotted_item_keeps_snapshot_readable(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1, 2, 3]}))
        folio.create_snapshot("v1")

        folio.delete("t")

        assert "t" not in folio._items
        snap_df = folio.snapshots["v1"].get("t")
        assert snap_df["a"].tolist() == [1, 2, 3]

        # Survives reopen too
        reloaded = DataFolio(tmp_path / "b")
        assert reloaded.snapshots["v1"].get("t")["a"].tolist() == [1, 2, 3]

    def test_delete_unsnapshotted_item_removes_payload(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("cfg", {"x": 1})
        filename = folio._items["cfg"]["filename"]
        payload = tmp_path / "b" / "artifacts" / filename
        assert payload.exists()

        folio.delete("cfg")
        assert not payload.exists()

    def test_deleted_snapshotted_item_restorable(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("gone", {"b": 2})
        folio.create_snapshot("v1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            folio.delete("gone")

        folio.restore_snapshot("v1", confirm=True)
        assert folio.get("gone") == {"b": 2}


class TestRestoreSnapshot:
    def test_restore_after_overwrite_restores_content(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1]}))
        folio.create_snapshot("v1")
        folio.add("t", pd.DataFrame({"a": [99]}), overwrite=True)

        folio.restore_snapshot("v1", confirm=True)
        assert folio.get("t")["a"].tolist() == [1]

    def test_restore_removes_items_added_after_snapshot(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("original", 1)
        folio.create_snapshot("v1")
        folio.add("later", 2)

        folio.restore_snapshot("v1", confirm=True)
        assert "later" not in folio._items
        assert folio.get("original") == 1

    def test_restore_preserves_items_pinned_by_other_snapshots(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("original", 1)
        folio.create_snapshot("v1")
        folio.add("later", 2)
        folio.create_snapshot("v2")

        folio.restore_snapshot("v1", confirm=True)
        assert "later" not in folio._items
        # v2 still sees 'later'
        assert folio.snapshots["v2"] is not None
        assert "later" in folio.snapshots["v2"].item_versions
        # And restoring v2 brings it back
        folio.restore_snapshot("v2", confirm=True)
        assert folio.get("later") == 2

    def test_restore_restores_metadata(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.metadata["accuracy"] = 0.9
        folio.add("m", 1)
        folio.create_snapshot("v1")
        folio.metadata["accuracy"] = 0.5

        folio.restore_snapshot("v1", confirm=True)
        assert folio.metadata["accuracy"] == 0.9

    def test_restore_fails_loudly_when_pinned_version_missing(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("cfg", 1)
        folio.create_snapshot("v1")
        # Simulate a manifest that lost the pinned version
        folio._snapshots["v1"]["item_versions"]["cfg"] = "nonexistent-token"

        with pytest.raises(KeyError, match="nonexistent-token"):
            folio.restore_snapshot("v1", confirm=True)
        # Nothing was mutated
        assert folio.get("cfg") == 1

    def test_restore_requires_confirm(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("cfg", 1)
        folio.create_snapshot("v1")
        with pytest.raises(ValueError, match="confirm=True"):
            folio.restore_snapshot("v1")


class TestExportSnapshot:
    def test_export_with_external_reference(self, tmp_path):
        ext = tmp_path / "ext.parquet"
        pd.DataFrame({"z": [1, 2]}).to_parquet(ext, index=False)

        folio = DataFolio(tmp_path / "b")
        folio.reference_table("ref", path=str(ext))
        folio.add("cfg", {"a": 1})
        folio.create_snapshot("v1")

        out = folio.export_snapshot("v1", tmp_path / "export")

        # Reference stays a reference (external data not copied)
        ref_item = out._items["ref"]
        assert ref_item["item_type"] == "referenced_table"
        # Stored as a file:// URI (reference paths are normalized on add)
        assert ref_item["path"].endswith(str(ext))
        assert out.get("ref")["z"].tolist() == [1, 2]
        assert out.get("cfg") == {"a": 1}

    def test_export_preserves_descriptions_and_lineage(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("raw", pd.DataFrame({"a": [1]}), description="raw rows")
        folio.add(
            "clean",
            pd.DataFrame({"a": [1]}),
            description="cleaned",
            inputs=["raw"],
        )
        folio.create_snapshot("v1")

        out = folio.export_snapshot("v1", tmp_path / "export")
        assert out._items["raw"]["description"] == "raw rows"
        assert out._items["clean"]["description"] == "cleaned"
        assert out.get_inputs("clean") == ["raw"]

    def test_export_copies_bytes_exactly(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1, 2, 3]}))
        folio.create_snapshot("v1")

        out = folio.export_snapshot("v1", tmp_path / "export")
        src_file = (
            tmp_path / "b" / "tables" / folio._items["t"]["filename"]
        ).read_bytes()
        dst_file = (
            tmp_path / "export" / "tables" / out._items["t"]["filename"]
        ).read_bytes()
        assert src_file == dst_file

    def test_export_snapshotted_version_not_current(self, tmp_path):
        """Export must ship the PINNED version, not the current one."""
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"a": [1]}))
        folio.create_snapshot("v1")
        folio.add("t", pd.DataFrame({"a": [99]}), overwrite=True)

        out = folio.export_snapshot("v1", tmp_path / "export")
        assert out.get("t")["a"].tolist() == [1]


class TestSnapshotInBatch:
    def test_create_snapshot_inside_batch_raises(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        with folio.batch():
            folio.add("a", 1)
            with pytest.raises(RuntimeError, match="batch"):
                folio.create_snapshot("mid-batch")

        # Batch still committed; no dangling snapshot bookkeeping
        reloaded = DataFolio(tmp_path / "b")
        assert reloaded.get("a") == 1
        assert list(reloaded._snapshots) == []
        assert reloaded._items["a"].get("in_snapshots") == []

        # And snapshotting right after the batch works
        folio.create_snapshot("after-batch")
        assert "after-batch" in DataFolio(tmp_path / "b")._snapshots


class TestReadOnlyEnforcement:
    @pytest.fixture()
    def ro(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("existing", 1)
        return DataFolio(tmp_path / "b", read_only=True)

    def test_add_methods_blocked(self, ro, tmp_path):
        src = tmp_path / "f.txt"
        src.write_text("x")
        with pytest.raises(RuntimeError):
            ro.add("arr", np.array([1]))
        with pytest.raises(RuntimeError):
            ro.add("j", 1)
        with pytest.raises(RuntimeError):
            ro.add("ts", datetime.now(timezone.utc))
        with pytest.raises(RuntimeError):
            ro.add_model("m", object())
        with pytest.raises(RuntimeError):
            ro.add_file(src, name="f")
        with pytest.raises(RuntimeError):
            ro.add_file(src)

    def test_mutations_blocked(self, ro):
        with pytest.raises(RuntimeError):
            ro.delete("existing")
        with pytest.raises(RuntimeError):
            ro.update_item("existing", description="new")

    def test_nothing_leaked_to_disk(self, ro, tmp_path):
        for attempt in (
            lambda: ro.add("arr", np.array([1])),
            lambda: ro.add("j", 1),
        ):
            with pytest.raises(RuntimeError):
                attempt()
        reloaded = DataFolio(tmp_path / "b")
        assert sorted(reloaded._items) == ["existing"]


class TestUpdateItemCopyOnWrite:
    def test_snapshot_view_keeps_original_metadata(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("cfg", {"v": 1}, description="original")
        folio.create_snapshot("v1")

        folio.update_item("cfg", description="edited after snapshot")

        assert folio._items["cfg"]["description"] == "edited after snapshot"
        snap = DataFolio.load_snapshot(str(tmp_path / "b"), "v1")
        assert snap._items["cfg"]["description"] == "original"
        # Both versions read the same (shared) payload
        assert snap.get("cfg") == {"v": 1}
        assert folio.get("cfg") == {"v": 1}

    def test_shared_payload_survives_snapshot_cleanup(self, tmp_path):
        """delete_snapshot(cleanup_orphans=True) must not delete a payload
        the current item still shares after a metadata-only CoW."""
        folio = DataFolio(tmp_path / "b")
        folio.add("cfg", {"v": 1}, description="original")
        folio.create_snapshot("v1")
        folio.update_item("cfg", description="edited")

        folio.delete_snapshot("v1", cleanup_orphans=True)
        assert folio.get("cfg") == {"v": 1}

    def test_shared_payload_survives_current_delete(self, tmp_path):
        """Deleting the current item must not delete a payload a snapshot
        version still shares."""
        folio = DataFolio(tmp_path / "b")
        folio.add("cfg", {"v": 1}, description="original")
        folio.create_snapshot("v1")
        folio.update_item("cfg", description="edited")

        folio.delete("cfg")
        snap = DataFolio.load_snapshot(str(tmp_path / "b"), "v1")
        assert snap.get("cfg") == {"v": 1}

    def test_unsnapshotted_update_edits_in_place(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("cfg", 1, description="a")
        vid = folio._items["cfg"]["version_id"]
        folio.update_item("cfg", description="b")
        # No CoW needed — same version
        assert folio._items["cfg"]["version_id"] == vid
        assert folio._snapshot_versions == []

    def test_empty_string_clears_field(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("cfg", 1, description="a")
        folio.update_item("cfg", description="")
        assert "description" not in folio._items["cfg"]
        assert "code" not in folio._items["cfg"]
