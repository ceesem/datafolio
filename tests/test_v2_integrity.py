"""Regression tests for the external-audit integrity fixes (P1–P9).

Each class pins one confirmed failure mode:
- snapshot registry mutations racing the stale-writer check (P1)
- overwrite demoting a snapshotted item before the new payload exists (P2)
- metadata writes outside the single-writer protocol (P3)
- load_snapshot silently serving newer data under a snapshot name (P4)
- batch() publishing partial work on exception (P5)
- payload deletion before manifest publication (P6)
- copy() leaking source snapshot membership (P7)
- snapshotted references replaceable without overwrite=True (P8)
- unguarded manifest mutations (P9)
"""

import json

import pandas as pd
import pytest

from datafolio import ConcurrentWriteError, DataFolio


def _make_stale_pair(path):
    """Two notebooks on one folio; the second is left behind one revision."""
    owner = DataFolio(path)
    owner.add("seed", 1)
    stale = DataFolio(path)
    stale.get("seed")  # load state
    owner.add("advance", 2)  # owner moves the manifest forward
    stale._auto_refresh_enabled = False  # a genuinely stale notebook
    return owner, stale


class TestP1SnapshotRegistryGuarded:
    def test_stale_create_snapshot_fails_before_writing_snapshots_json(self, tmp_path):
        path = tmp_path / "b"
        owner, stale = _make_stale_pair(path)

        with pytest.raises(ConcurrentWriteError):
            stale.create_snapshot("stale-snap")

        # Nothing on disk mentions the failed snapshot
        snapshots_file = path / "snapshots.json"
        if snapshots_file.exists():
            on_disk = json.loads(snapshots_file.read_text())
            assert "stale-snap" not in on_disk.get("snapshots", {})
        items = json.loads((path / "items.json").read_text())
        for item in items["items"]:
            assert "stale-snap" not in item.get("in_snapshots", [])

        # In-memory state is restored: retry after refresh succeeds cleanly
        assert "stale-snap" not in stale._snapshots
        stale.refresh()
        stale.create_snapshot("stale-snap")
        assert "stale-snap" in DataFolio(path)._snapshots

    def test_stale_delete_snapshot_fails_cleanly(self, tmp_path):
        path = tmp_path / "b"
        owner = DataFolio(path)
        owner.add("x", 1)
        owner.create_snapshot("keep")
        stale = DataFolio(path)
        owner.add("y", 2)
        stale._auto_refresh_enabled = False

        with pytest.raises(ConcurrentWriteError):
            stale.delete_snapshot("keep")

        # Snapshot still fully present on disk and in a fresh view
        reopened = DataFolio(path)
        assert "keep" in reopened._snapshots
        assert reopened.snapshots["keep"].get("x") == 1
        # Stale notebook's memory was not left half-deleted
        assert "keep" in stale._snapshots

    def test_create_snapshot_survives_reopen(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        folio.create_snapshot("v1")
        reopened = DataFolio(tmp_path / "b")
        assert "v1" in reopened._snapshots
        assert reopened._items["x"]["in_snapshots"] == ["v1"]

    def test_delete_snapshot_in_batch_raises(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        folio.create_snapshot("v1")
        with folio.batch():
            with pytest.raises(RuntimeError, match="batch"):
                folio.delete_snapshot("v1")
        assert "v1" in DataFolio(tmp_path / "b")._snapshots

    def test_snapshot_view_resolves_exact_pinned_version(self, tmp_path):
        """The view must serve the version token recorded in item_versions,
        not whatever current item happens to carry the snapshot marker."""
        folio = DataFolio(tmp_path / "b")
        folio.add("t", pd.DataFrame({"v": [1]}))
        folio.create_snapshot("s")
        folio.add("t", pd.DataFrame({"v": [2]}), overwrite=True)
        # Corrupt a marker: pretend the current version claims membership
        folio._items["t"]["in_snapshots"] = ["s"]
        assert folio.snapshots["s"].get("t")["v"].tolist() == [1]


class TestP2ExceptionSafeOverwrite:
    def _boom_handler(self, monkeypatch):
        from datafolio.base.registry import get_registry

        handler = get_registry().get("included_table")
        monkeypatch.setattr(
            type(handler),
            "add",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")),
        )

    def test_failed_overwrite_leaves_current_intact(self, tmp_path, monkeypatch):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        folio.create_snapshot("s")

        self._boom_handler(monkeypatch)
        with pytest.raises(RuntimeError, match="disk full"):
            folio.add("x", pd.DataFrame({"v": [2]}), overwrite=True)
        monkeypatch.undo()

        # In-memory: x is still current and unchanged
        assert folio._items["x"]["is_current"] is True
        assert folio.get("x")["v"].tolist() == [1]
        # No half-finished copy-on-write state persists via a later mutation
        folio.add("y", 1)
        reopened = DataFolio(path)
        assert sorted(reopened._items) == ["x", "y"]
        assert reopened.get("x")["v"].tolist() == [1]
        assert reopened.snapshots["s"].get("x")["v"].tolist() == [1]

    def test_failed_overwrite_unsnapshotted_item(self, tmp_path, monkeypatch):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        self._boom_handler(monkeypatch)
        with pytest.raises(RuntimeError):
            folio.add("x", pd.DataFrame({"v": [2]}), overwrite=True)
        monkeypatch.undo()
        assert folio.get("x")["v"].tolist() == [1]
        assert DataFolio(path).get("x")["v"].tolist() == [1]

    def test_stale_reference_replacement_fails_cleanly(self, tmp_path):
        path = tmp_path / "b"
        owner, stale = _make_stale_pair(path)
        owner.reference_table("r", path="s3://bucket/a.parquet")
        with pytest.raises(ConcurrentWriteError):
            stale.reference_table("r2", path="s3://bucket/b.parquet")
        assert "r2" not in stale._items
        assert "r2" not in DataFolio(path)._items


class TestP3MetadataConcurrency:
    def test_two_notebooks_distinct_keys(self, tmp_path):
        path = tmp_path / "b"
        a = DataFolio(path)
        b = DataFolio(path)
        b._auto_refresh_enabled = False
        a.metadata["from_a"] = 1
        with pytest.raises(ConcurrentWriteError):
            b.metadata["from_b"] = 2
        # b's memory unchanged by the failed write
        assert "from_b" not in b.metadata
        # refresh-and-retry preserves both keys
        b._auto_refresh_enabled = True
        b.refresh()
        b.metadata["from_b"] = 2
        final = DataFolio(path).metadata
        assert final["from_a"] == 1 and final["from_b"] == 2

    def test_same_key_conflict_detected(self, tmp_path):
        path = tmp_path / "b"
        a = DataFolio(path)
        b = DataFolio(path)
        b._auto_refresh_enabled = False
        a.metadata["k"] = "from_a"
        with pytest.raises(ConcurrentWriteError):
            b.metadata["k"] = "from_b"
        assert DataFolio(path).metadata["k"] == "from_a"

    def test_metadata_write_advances_detectable_revision(self, tmp_path):
        path = tmp_path / "b"
        a = DataFolio(path)
        rev_before = a._manifest_revision or 0
        a.metadata["k"] = 1
        assert (a._manifest_revision or 0) > rev_before
        # Another instance's next write sees it as the base revision
        b = DataFolio(path)
        assert b._manifest_revision == a._manifest_revision

    def test_bulk_update_is_single_commit(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        rev0 = folio._manifest_revision
        folio.metadata.update({"a": 1, "b": 2, "c": 3})
        assert folio._manifest_revision == rev0 + 1

    def test_metadata_in_batch_commits_with_batch(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        with folio.batch():
            folio.metadata["inside"] = True
            folio.add("x", 1)
        assert DataFolio(path).metadata["inside"] is True

    def test_read_only_still_enforced(self, tmp_path):
        path = tmp_path / "b"
        DataFolio(path).metadata["k"] = 1
        ro = DataFolio(path, read_only=True)
        for attempt in (
            lambda: ro.metadata.__setitem__("x", 1),
            lambda: ro.metadata.pop("k"),
            lambda: ro.metadata.update({"x": 1}),
        ):
            with pytest.raises(RuntimeError):
                attempt()

    def test_snapshot_metadata_restore_single_commit(self, tmp_path):
        """restore_snapshot must not spray intermediate metadata saves."""
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.metadata["acc"] = 0.9
        folio.add("x", 1)
        folio.create_snapshot("v1")
        folio.metadata["acc"] = 0.5
        rev0 = folio._manifest_revision
        folio.restore_snapshot("v1", confirm=True)
        assert folio.metadata["acc"] == 0.9
        # one commit for the whole restore
        assert folio._manifest_revision == rev0 + 1


class TestP4LoadSnapshotFailClosed:
    def _folio_with_lost_pin(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", {"v": 1})
        folio.create_snapshot("s")
        folio.add("x", {"v": 2}, overwrite=True)
        # Simulate loss of the pinned v1 descriptor
        folio._snapshot_versions = [
            i for i in folio._snapshot_versions if i.get("name") != "x"
        ]
        folio._save_items()
        return path

    def test_missing_descriptor_raises(self, tmp_path):
        path = self._folio_with_lost_pin(tmp_path)
        with pytest.raises(KeyError, match="pins version"):
            DataFolio.load_snapshot(str(path), "s")

    def test_snapshot_view_missing_descriptor_raises(self, tmp_path):
        path = self._folio_with_lost_pin(tmp_path)
        folio = DataFolio(path)
        with pytest.raises(KeyError, match="pins version"):
            folio.snapshots["s"].get("x")

    def test_wrong_version_token_raises(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", 1)
        folio.create_snapshot("s")
        folio._snapshots["s"]["item_versions"]["x"] = "bogus-token"
        folio._save_snapshots()
        with pytest.raises(KeyError, match="pins version"):
            DataFolio.load_snapshot(str(path), "s")

    def test_references_resolve_in_snapshot(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.reference_table("r", path="s3://bucket/a.parquet")
        folio.create_snapshot("s")
        snap = DataFolio.load_snapshot(str(path), "s")
        assert snap._items["r"]["item_type"] == "referenced_table"

    def test_legacy_checksum_token_resolves(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        folio.create_snapshot("s")
        # Old snapshots recorded the checksum, not the version_id
        folio._snapshots["s"]["item_versions"]["x"] = folio._items["x"]["checksum"]
        folio._save_snapshots()
        snap = DataFolio.load_snapshot(str(path), "s")
        assert snap.get("x")["v"].tolist() == [1]


class TestP5BatchExceptionSafety:
    def test_exception_discards_staged_additions(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("committed", 1)
        with pytest.raises(ValueError, match="user error"):
            with folio.batch():
                folio.add("staged", 2)
                raise ValueError("user error")
        # In-memory and on-disk both show only committed state
        assert sorted(folio._items) == ["committed"]
        assert sorted(DataFolio(path)._items) == ["committed"]

    def test_exception_discards_staged_overwrite(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        with pytest.raises(RuntimeError):
            with folio.batch():
                folio.add("x", pd.DataFrame({"v": [2]}), overwrite=True)
                raise RuntimeError("abort")
        assert folio.get("x")["v"].tolist() == [1]
        assert DataFolio(path).get("x")["v"].tolist() == [1]

    def test_exception_discards_snapshotted_replacement(self, tmp_path):
        """Aborted batch must not persist partial copy-on-write state or
        delete the old committed payload."""
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        folio.create_snapshot("s")
        with pytest.raises(RuntimeError):
            with folio.batch():
                folio.add("x", pd.DataFrame({"v": [2]}), overwrite=True)
                raise RuntimeError("abort")
        assert folio.get("x")["v"].tolist() == [1]
        assert folio._items["x"]["is_current"] is True
        assert folio.snapshots["s"].get("x")["v"].tolist() == [1]
        reopened = DataFolio(path)
        assert reopened.get("x")["v"].tolist() == [1]

    def test_exception_discards_metadata_changes(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.metadata["k"] = "committed"
        with pytest.raises(RuntimeError):
            with folio.batch():
                folio.metadata["k"] = "staged"
                raise RuntimeError("abort")
        assert folio.metadata["k"] == "committed"
        assert DataFolio(path).metadata["k"] == "committed"

    def test_exception_discards_description_change(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", 1, description="original")
        with pytest.raises(RuntimeError):
            with folio.batch():
                folio.update_item("x", description="staged")
                raise RuntimeError("abort")
        assert folio._items["x"]["description"] == "original"
        assert DataFolio(path)._items["x"]["description"] == "original"

    def test_nested_batch_rejected(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        with folio.batch():
            folio.add("a", 1)
            with pytest.raises(RuntimeError, match="[Nn]ested"):
                with folio.batch():
                    pass
        # Outer batch still committed
        assert DataFolio(tmp_path / "b").get("a") == 1

    def test_normal_batch_commits_once(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        rev0 = folio._manifest_revision or 0
        with folio.batch():
            folio.add("a", 1)
            folio.add("b", 2)
        assert folio._manifest_revision == rev0 + 1
        assert sorted(DataFolio(path)._items) == ["a", "b"]


class TestP6ManifestFirstDeletion:
    def test_delete_keeps_payload_if_publish_fails(self, tmp_path, monkeypatch):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        payload = path / "tables" / folio._items["x"]["filename"]
        assert payload.exists()

        monkeypatch.setattr(
            type(folio._storage),
            "write_json",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk error")),
        )
        with pytest.raises(OSError):
            folio.delete("x")
        monkeypatch.undo()

        # The committed manifest still references the payload — it must exist
        assert payload.exists()
        reopened = DataFolio(path)
        assert reopened.get("x")["v"].tolist() == [1]

    def test_restore_keeps_payloads_if_publish_fails(self, tmp_path, monkeypatch):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        folio.create_snapshot("s")
        folio.add("later", pd.DataFrame({"v": [9]}))
        later_payload = path / "tables" / folio._items["later"]["filename"]

        monkeypatch.setattr(
            type(folio._storage),
            "write_json",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk error")),
        )
        with pytest.raises(OSError):
            folio.restore_snapshot("s", confirm=True)
        monkeypatch.undo()

        assert later_payload.exists()
        reopened = DataFolio(path)
        assert reopened.get("later")["v"].tolist() == [9]

    def test_cleanup_orphans_publishes_before_deleting(self, tmp_path, monkeypatch):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        folio.create_snapshot("s")
        folio.add("x", pd.DataFrame({"v": [2]}), overwrite=True)
        old_filename = folio._snapshot_versions[0]["filename"]
        folio.delete_snapshot("s")  # orphans the old version

        monkeypatch.setattr(
            type(folio._storage),
            "write_json",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk error")),
        )
        with pytest.raises(OSError):
            folio.cleanup_orphaned_versions()
        monkeypatch.undo()

        # Manifest write failed -> the still-referenced payload must survive
        assert (path / "tables" / old_filename).exists()


class TestP7CopyHygiene:
    def test_copy_clears_snapshot_membership(self, tmp_path):
        folio = DataFolio(tmp_path / "src")
        folio.add("x", 1, description="a thing")
        folio.create_snapshot("src-snap")
        copied = folio.copy(tmp_path / "dst")

        assert copied._items["x"]["in_snapshots"] == []
        assert copied._items["x"]["is_current"] is True
        assert copied._snapshots == {}
        assert copied._items["x"]["description"] == "a thing"
        # And a clean reopen agrees
        reopened = DataFolio(tmp_path / "dst")
        assert reopened._items["x"]["in_snapshots"] == []
        # Deleting in the copy must not think a snapshot pins it
        reopened.delete("x")
        assert reopened._snapshot_versions == []

    def test_copy_preserves_lineage_and_reference_path(self, tmp_path):
        folio = DataFolio(tmp_path / "src")
        folio.reference_table("raw", path="s3://bucket/raw.parquet")
        folio.add("clean", pd.DataFrame({"v": [1]}), inputs=["raw"])
        folio.create_snapshot("s")
        copied = folio.copy(tmp_path / "dst")
        assert copied._items["raw"]["path"] == "s3://bucket/raw.parquet"
        assert copied.get_inputs("clean") == ["raw"]

    def test_copy_contents_md_reflects_destination(self, tmp_path):
        folio = DataFolio(tmp_path / "src")
        folio.add("only_item", 1)
        copied = folio.copy(tmp_path / "dst")
        contents = (tmp_path / "dst" / "CONTENTS.md").read_text()
        assert "only_item" in contents


class TestP8UniformReferenceOverwrite:
    def test_snapshotted_reference_requires_overwrite(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("r", path="s3://bucket/a.parquet")
        folio.create_snapshot("s")
        with pytest.raises(ValueError, match="overwrite=True"):
            folio.reference_table("r", path="s3://bucket/OTHER.parquet")
        # Explicit overwrite works and preserves the pinned descriptor
        folio.reference_table("r", path="s3://bucket/OTHER.parquet", overwrite=True)
        assert folio._items["r"]["path"] == "s3://bucket/OTHER.parquet"
        assert (
            folio.snapshots["s"]._find_snapshot_item("r")["path"]
            == "s3://bucket/a.parquet"
        )


class TestP9RemainingMutationsGuarded:
    def test_stale_archive_fails_before_mutating(self, tmp_path):
        path = tmp_path / "b"
        owner, stale = _make_stale_pair(path)
        with pytest.raises(ConcurrentWriteError):
            stale.archive("seed")
        assert stale._items["seed"].get("archived") is not True
        assert DataFolio(path)._items["seed"].get("archived") is not True

    def test_stale_unarchive_fails_before_mutating(self, tmp_path):
        path = tmp_path / "b"
        owner = DataFolio(path)
        owner.add("seed", 1)
        owner.archive("seed")
        stale = DataFolio(path)
        owner.add("advance", 2)
        stale._auto_refresh_enabled = False
        with pytest.raises(ConcurrentWriteError):
            stale.unarchive("seed")
        assert DataFolio(path)._items["seed"].get("archived") is True

    def test_inspect_table_rejects_replaced_version(self, tmp_path, monkeypatch):
        """If the item is replaced while inspect's I/O runs, the stale
        enrichment must be rejected rather than applied to the new version."""
        path = tmp_path / "b"
        ext = tmp_path / "ext.parquet"
        pd.DataFrame({"z": [1]}).to_parquet(ext, index=False)
        folio = DataFolio(path)
        folio.reference_table("r", path=str(ext))

        from datafolio.handlers.tables import ReferenceTableHandler

        real_inspect = ReferenceTableHandler.inspect

        def slow_inspect(handler_self, f, name):
            result = real_inspect(handler_self, f, name)
            # Another writer replaces the reference mid-inspection
            other = DataFolio(path)
            other.reference_table("r", path=str(ext), overwrite=True)
            folio._auto_refresh_enabled = False
            return result

        monkeypatch.setattr(ReferenceTableHandler, "inspect", slow_inspect)
        with pytest.raises(ConcurrentWriteError):
            folio.inspect_table("r")
