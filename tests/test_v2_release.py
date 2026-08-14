"""Regression tests for the pre-release audit round (fresh cold audit +
codex round 4): transaction-boundary leaks, ownership edges, and behavioral
polish."""

import json

import numpy as np
import pandas as pd
import pytest

from datafolio import ConcurrentWriteError, DataFolio


def _fail_write_json(monkeypatch):
    import datafolio.storage.backend as B

    monkeypatch.setattr(
        B.StorageBackend,
        "write_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk error")),
    )


class TestH1SnapshotCaptureInsideGuard:
    def test_stale_writer_snapshot_rejected_not_laundered(self, tmp_path):
        """A stale notebook's create_snapshot must fail with
        ConcurrentWriteError — not launder its staleness through the
        metadata getter's auto-refresh and commit a mixed/unloadable state."""
        path = tmp_path / "b"
        a = DataFolio(path)
        a.add("a", 1)
        b = DataFolio(path)
        b.get("a")
        a.add("b", 2)  # b is now stale
        b._auto_refresh_enabled = False

        with pytest.raises(ConcurrentWriteError):
            b.create_snapshot("s")
        assert "s" not in DataFolio(path)._snapshots

    def test_fresh_snapshot_captures_consistent_state(self, tmp_path):
        """With auto-refresh on, a snapshot must pin the CURRENT manifest
        state — items and metadata from the same revision."""
        path = tmp_path / "b"
        a = DataFolio(path)
        a.add("a", 1)
        b = DataFolio(path)
        b.get("a")
        a.add("b", 2)  # b will refresh during create_snapshot

        b.create_snapshot("s")
        pinned = sorted(b._snapshots["s"]["item_versions"])
        assert pinned == ["a", "b"]
        # And it loads
        assert DataFolio.load_snapshot(str(path), "s").get("b") == 2

    def test_snapshot_pins_resolvable_versions_only(self, tmp_path):
        """Every committed snapshot must be loadable immediately after
        creation (no pinning of already-deleted versions)."""
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        folio.create_snapshot("s")
        snap = DataFolio.load_snapshot(str(path), "s")
        assert snap.get("x")["v"].tolist() == [1]


class TestH2MetadataSetterInsideGuard:
    def test_failed_wholesale_assignment_rolls_back(self, tmp_path, monkeypatch):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.metadata["old"] = 1

        _fail_write_json(monkeypatch)
        with pytest.raises(OSError):
            folio.metadata = {"rejected": 2}
        monkeypatch.undo()

        assert "rejected" not in folio.metadata
        assert folio.metadata["old"] == 1
        folio.add("later", 1)  # must not leak the rejected replacement
        final = DataFolio(path).metadata
        assert final.get("old") == 1
        assert "rejected" not in final


class TestH3PublicationBoundary:
    def test_post_publish_exception_keeps_committed_memory(self, tmp_path, monkeypatch):
        """An exception AFTER the manifest write must not roll memory back
        behind the committed disk state."""
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", 1, description="old")

        monkeypatch.setattr(
            type(folio),
            "_sync_data_accessor",
            lambda self: (_ for _ in ()).throw(RuntimeError("accessor boom")),
        )
        # Post-publish steps are best-effort: the operation SUCCEEDS
        folio.update_item("x", description="new")
        monkeypatch.undo()

        disk = json.loads((path / "items.json").read_text())
        assert folio._items["x"]["description"] == "new"
        assert folio._manifest_revision == disk["revision"]
        assert disk["items"][0]["description"] == "new"

    def test_pre_publish_exception_still_rolls_back(self, tmp_path, monkeypatch):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", 1, description="old")
        _fail_write_json(monkeypatch)
        with pytest.raises(OSError):
            folio.update_item("x", description="new")
        monkeypatch.undo()
        assert folio._items["x"]["description"] == "old"
        folio.add("y", 2)
        assert DataFolio(path)._items["x"]["description"] == "old"


class TestMetadataKeyCollision:
    def test_parent_and_self_keys_do_not_brick_the_folio(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.metadata["parent"] = "exp-001"
        folio.metadata["self"] = "circular"

        reopened = DataFolio(path)  # must not TypeError
        assert reopened.metadata["parent"] == "exp-001"
        assert reopened.metadata["self"] == "circular"

    def test_creation_with_reserved_keys(self, tmp_path):
        folio = DataFolio(tmp_path / "b", metadata={"parent": "x", "data": 1})
        assert folio.metadata["parent"] == "x"


class TestVersionIdCollisions:
    def test_copied_folio_reference_overwrite_keeps_snapshot_resolution(self, tmp_path):
        """version_ids carried into a copy must never collide with newly
        minted ones — a snapshot must keep resolving its own descriptor."""
        src = DataFolio(tmp_path / "src")
        # Push src's revision up so carried ids embed a high revision
        for i in range(5):
            src.add(f"pad{i}", i)
        src.reference_table("raw", path="s3://warehouse/v1.parquet")

        dst = src.copy(tmp_path / "dst", include_items=["raw"], follow_lineage=False)
        dst.create_snapshot("pin-v1")
        old_token = dst._snapshots["pin-v1"]["item_versions"]["raw"]

        # Drive the destination's revision toward the embedded number,
        # overwriting the reference each time
        for i in range(10):
            dst.reference_table(
                "raw", path=f"s3://warehouse/v2-DIFFERENT-{i}.parquet", overwrite=True
            )

        # All version ids in the manifest must be unique
        all_ids = [
            d.get("version_id")
            for d in list(dst._items.values()) + dst._snapshot_versions
        ]
        assert len(all_ids) == len(set(all_ids)), all_ids
        # And the snapshot still resolves the ORIGINAL reference
        pinned = dst._find_item_by_checksum("raw", old_token)
        assert pinned["path"] == "s3://warehouse/v1.parquet"


class TestFileUriBundles:
    def test_file_uri_folio_is_a_normal_local_folio(self, tmp_path):
        uri = f"file://{tmp_path}/b"
        folio = DataFolio(uri)
        folio.add("t", pd.DataFrame({"v": [1]}))  # tables must work
        assert folio.get("t")["v"].tolist() == [1]
        # Same physical directory opens as a plain path with shared state
        plain = DataFolio(tmp_path / "b")
        assert plain.get("t")["v"].tolist() == [1]

    def test_file_uri_shares_lock_and_revisions_with_plain_path(self, tmp_path):
        plain = DataFolio(tmp_path / "b")
        plain.add("seed", 1)
        via_uri = DataFolio(f"file://{tmp_path}/b")
        plain.add("more", 2)
        via_uri._auto_refresh_enabled = False
        with pytest.raises(ConcurrentWriteError):
            via_uri.add("stale_write", 3)


class TestExactRevisionEquality:
    def test_lower_disk_revision_rejected(self, tmp_path):
        from datafolio import ManifestReadError

        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", 1)
        manifest = json.loads((path / "items.json").read_text())
        manifest["revision"] = 0
        (path / "items.json").write_text(json.dumps(manifest))

        with pytest.raises((ConcurrentWriteError, ManifestReadError)):
            folio.add("y", 2)


class TestReaderRefreshUniform:
    def test_all_readers_see_other_writer(self, tmp_path):
        path = tmp_path / "b"
        a = DataFolio(path)
        a.add("t", pd.DataFrame({"v": [1]}))
        b = DataFolio(path)
        a.add("t2", pd.DataFrame({"v": [2]}))
        a.add_model("m", object())
        a.create_snapshot("s1")

        assert sorted(b.tables) == ["t", "t2"]
        assert b.models == ["m"]
        assert [s["name"] for s in b.list_snapshots()] == ["s1"]
        assert "s1" in dict(b.snapshots.items())
        assert b.get_snapshot_info("s1")["name"] == "s1"
        assert b.diff_from_snapshot("s1")["unchanged_items"]


class TestValidateReports:
    def test_unreachable_reference_reported_not_raised(self, tmp_path, monkeypatch):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("good", 1)
        folio.reference_table("bad", path="s3://no-such-bucket/x.parquet")

        import datafolio.storage.backend as B

        real = B.StorageBackend.exists

        def exploding_exists(self, p):
            if "no-such-bucket" in str(p):
                raise RuntimeError("NoSuchBucket")
            return real(self, p)

        monkeypatch.setattr(B.StorageBackend, "exists", exploding_exists)
        results = folio.validate()
        assert results["good"] is True
        assert results["bad"] is False
        assert folio.is_valid() is False


class TestCopyIncludeItemsTypo:
    def test_unknown_include_items_raise(self, tmp_path):
        folio = DataFolio(tmp_path / "src")
        folio.add("real_item", 1)
        with pytest.raises(KeyError, match="tpyo_item"):
            folio.copy(tmp_path / "dst", include_items=["tpyo_item"])
        assert not (tmp_path / "dst").exists()


class TestOwnershipBoundaries:
    def test_snapshot_tags_defensively_copied(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        my_tags = ["baseline"]
        folio.create_snapshot("s", tags=my_tags)
        my_tags.append("injected")
        assert folio._snapshots["s"]["tags"] == ["baseline"]
        listed = folio.list_snapshots()[0]["tags"]
        listed.append("injected2")
        assert folio._snapshots["s"]["tags"] == ["baseline"]

    def test_copy_metadata_deep_copied(self, tmp_path):
        src = DataFolio(tmp_path / "src")
        src.metadata["params"] = {"lr": 0.1}
        src.add("x", 1)
        dst = src.copy(tmp_path / "dst")
        # Mutating source nested metadata must not touch the copy
        src.metadata["params"]["lr"] = 999
        assert dst.metadata["params"]["lr"] == 0.1

    def test_metadata_update_poisoned_iterator_stages_first(self, tmp_path):
        """update() must materialize its input before mutating, so a caught
        exception inside a batch can't leave partial keys behind."""
        path = tmp_path / "b"
        folio = DataFolio(path)

        def poison():
            yield ("partial", 1)
            raise RuntimeError("boom")

        with folio.batch():
            try:
                folio.metadata.update(poison())
            except RuntimeError:
                pass  # caught INSIDE the batch
            folio.add("committed", 1)
        assert "partial" not in DataFolio(path).metadata


class TestBehavioralLows:
    def test_archive_of_pinned_item_does_not_edit_snapshot_view(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        folio.create_snapshot("s")
        folio.archive("x")
        pinned = folio.snapshots["s"]._find_snapshot_item("x")
        assert pinned.get("archived") is not True
        assert folio.list_contents()["json_data"] == []  # archived in working set

    def test_masked_array_rejected_clearly(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        with pytest.raises(ValueError, match="[Mm]asked"):
            folio.add("m", np.ma.array([1, 2, 3], mask=[0, 1, 0]))

    def test_preserve_index_with_lazyframe_raises(self, tmp_path):
        pl = pytest.importorskip("polars")
        folio = DataFolio(tmp_path / "b")
        lf = pl.LazyFrame({"a": [1, 2]})
        with pytest.raises(TypeError, match="index"):
            folio.add("t", lf, preserve_index=True)

    def test_empty_payload_error_names_the_real_problem(self, tmp_path):
        ext = tmp_path / "empty.parquet"
        ext.touch()  # exists, zero bytes
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("e", path=str(ext))
        with pytest.raises(ValueError, match="empty"):
            folio.get("e")

    def test_accessor_repr_includes_timestamps(self, tmp_path):
        from datetime import datetime, timezone

        folio = DataFolio(tmp_path / "b")
        folio.add("when", datetime.now(timezone.utc))
        assert "when" in repr(folio.data)

    def test_artifact_get_with_missing_payload_raises(self, tmp_path):
        src = tmp_path / "f.txt"
        src.write_text("x")
        folio = DataFolio(tmp_path / "b")
        folio.add_file(src, name="f")
        (tmp_path / "b" / "artifacts" / folio._items["f"]["filename"]).unlink()
        with pytest.raises(FileNotFoundError):
            folio.get("f")

    def test_delta_path_rejected_without_explicit_format(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        with pytest.raises(ValueError, match="[Dd]elta"):
            folio.reference_table("d", path="s3://bucket/table.delta")

    def test_init_crash_husk_reopens(self, tmp_path):
        """A directory containing only the empty standard subdirs (crash
        between mkdirs and first manifest write) must initialize cleanly."""
        husk = tmp_path / "husk"
        for sub in ("tables", "models", "artifacts"):
            (husk / sub).mkdir(parents=True)
        folio = DataFolio(husk)
        folio.add("x", 1)
        assert DataFolio(husk).get("x") == 1


# ============================================================================
# Audit round 5 (post-release-round review)
# ============================================================================


class TestRollbackNotLaundered:
    def test_read_does_not_adopt_lower_revision(self, tmp_path):
        """Auto-refresh must not adopt a LOWER on-disk revision: reading
        first would otherwise launder a rolled-back/replaced manifest past
        the exact-revision write check."""
        from datafolio import ManifestReadError

        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", 1)
        folio.add("z", 3)
        loaded_rev = folio._manifest_revision

        manifest = json.loads((path / "items.json").read_text())
        manifest["revision"] = 0
        (path / "items.json").write_text(json.dumps(manifest))

        # Reads keep serving committed memory, not the replaced manifest
        _ = folio.tables
        _ = folio.list_contents()
        assert folio._manifest_revision == loaded_rev

        # And the next write still fails closed
        with pytest.raises((ConcurrentWriteError, ManifestReadError)):
            folio.add("y", 2)

    def test_explicit_refresh_accepts_replacement(self, tmp_path):
        """refresh() remains the deliberate way to adopt a replaced manifest."""
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", 1)

        manifest = json.loads((path / "items.json").read_text())
        manifest["revision"] = 0
        (path / "items.json").write_text(json.dumps(manifest))

        folio.refresh()
        assert folio._manifest_revision == 0

    def test_higher_revision_still_auto_refreshes(self, tmp_path):
        path = tmp_path / "b"
        a = DataFolio(path)
        a.add("x", 1)
        b = DataFolio(path)
        a.add("y", 2)
        assert "y" in b.list_contents()["json_data"] or "y" in str(b.list_contents())


class TestConcurrentCreationRace:
    def test_second_creator_fails_instead_of_overwriting(self, tmp_path, monkeypatch):
        """Two constructors racing to create the same new folio: the loser
        must raise ConcurrentWriteError, not overwrite the winner."""
        target = tmp_path / "shared"
        state = {"raced": False}
        original_save = DataFolio._save_items

        def racing_save(self):
            # On the loser's initial publish, simulate another process
            # completing creation of the same folio between the loser's
            # existence check and its first manifest write.
            if not state["raced"] and self._manifest_revision is None:
                state["raced"] = True
                winner = DataFolio(target)
                winner.metadata["owner"] = "first"
            return original_save(self)

        monkeypatch.setattr(DataFolio, "_save_items", racing_save)
        with pytest.raises(ConcurrentWriteError):
            DataFolio(target)

        survivor = DataFolio(target)
        assert survivor.metadata["owner"] == "first"


class TestCallerInputOwnership:
    def test_constructor_metadata_not_mutated_or_aliased(self, tmp_path):
        meta = {"params": {"lr": 0.1}}
        folio = DataFolio(tmp_path / "b", metadata=meta)

        # The caller's dict must not grow timestamps/version stamps
        assert "created_at" not in meta
        assert "_datafolio" not in meta

        # ...and later caller mutations must not leak into a folio save
        meta["params"]["lr"] = 999
        folio.add("x", 1)
        again = DataFolio(tmp_path / "b")
        assert again.metadata["params"]["lr"] == 0.1

    def test_add_inputs_list_copied(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("a", 1)
        ins = ["a"]
        folio.add("b", pd.DataFrame({"v": [1]}), inputs=ins)
        ins.append("evil")
        folio.metadata["note"] = "unrelated save"
        again = DataFolio(tmp_path / "b")
        assert again.get_inputs("b") == ["a"]

    def test_update_item_inputs_list_copied(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("a", 1)
        folio.add("b", 2)
        ins = ["a"]
        folio.update_item("b", inputs=ins)
        ins.append("evil")
        folio.metadata["note"] = "unrelated save"
        again = DataFolio(tmp_path / "b")
        assert again.get_inputs("b") == ["a"]


class TestIorPoisonedIterator:
    def test_ior_partial_keys_not_committed(self, tmp_path):
        """metadata |= <iterator that raises midway>, caught inside a batch,
        must not commit the partially inserted keys (same staging contract
        as update())."""
        folio = DataFolio(tmp_path / "b")

        def pairs():
            yield ("good", 1)
            raise RuntimeError("boom")

        with folio.batch():
            try:
                folio.metadata |= pairs()
            except RuntimeError:
                pass
            folio.add("x", 1)

        again = DataFolio(tmp_path / "b")
        assert "good" not in again.metadata
