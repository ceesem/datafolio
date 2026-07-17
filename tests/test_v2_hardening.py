"""Regression tests for the wave-B hardening fixes: ghost snapshot markers,
deep snapshot metadata capture, revision-based reader staleness, the
folio.metadata property, numpy scalar dispatch, name-length headroom,
accessor iteration, and CLI validate/init."""

import numpy as np
import pandas as pd
import pytest

from datafolio import DataFolio


class TestGhostMarkers:
    """in_snapshots markers naming snapshots absent from the registry (a
    crash between the two snapshot-file writes) must not leak payloads or
    block deletion forever."""

    def _folio_with_ghost(self, tmp_path, monkeypatch):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        monkeypatch.setattr(
            type(folio),
            "_save_snapshots",
            lambda self: (_ for _ in ()).throw(OSError("crash")),
        )
        with pytest.raises(OSError):
            folio.create_snapshot("ghost")
        monkeypatch.undo()
        return path, folio

    def test_ghost_marker_does_not_trigger_cow_leak(self, tmp_path, monkeypatch):
        path, folio = self._folio_with_ghost(tmp_path, monkeypatch)
        folio = DataFolio(path)  # reopen: marker committed on disk
        assert folio._items["x"].get("in_snapshots") == ["ghost"]

        folio.add("x", pd.DataFrame({"v": [2]}), overwrite=True)
        # No version preserved for a snapshot that doesn't exist
        assert folio._snapshot_versions == []
        parquet_files = list((path / "tables").glob("*.parquet"))
        assert len(parquet_files) == 1  # old payload actually reclaimed

    def test_ghost_marked_item_is_deletable(self, tmp_path, monkeypatch):
        path, folio = self._folio_with_ghost(tmp_path, monkeypatch)
        folio = DataFolio(path)
        folio.delete("x")
        assert folio._snapshot_versions == []
        assert list((path / "tables").glob("*.parquet")) == []

    def test_snapshot_retry_does_not_duplicate_marker(self, tmp_path, monkeypatch):
        path, folio = self._folio_with_ghost(tmp_path, monkeypatch)
        folio = DataFolio(path)
        folio.create_snapshot("ghost")  # retry same name succeeds
        assert folio._items["x"]["in_snapshots"] == ["ghost"]

    def test_delete_snapshot_removes_all_marker_occurrences(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", 1)
        folio.create_snapshot("s")
        # Simulate a legacy manifest with a duplicated marker
        folio._items["x"]["in_snapshots"] = ["s", "s"]
        folio._save_items()
        folio.delete_snapshot("s")
        assert folio._items["x"]["in_snapshots"] == []

    def test_cleanup_reclaims_ghost_pinned_versions(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        folio.create_snapshot("s")
        folio.add("x", pd.DataFrame({"v": [2]}), overwrite=True)
        # Corrupt: the preserved version's snapshot vanishes from the registry
        del folio._snapshots["s"]
        folio._save_snapshots()
        deleted = folio.cleanup_orphaned_versions()
        assert deleted  # the ghost-pinned version was reclaimed
        assert folio._snapshot_versions == []


class TestDeepSnapshotMetadataCapture:
    def test_nested_metadata_edit_after_snapshot_does_not_leak(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.metadata["params"] = {"lr": 0.1}
        folio.add("x", 1)
        folio.create_snapshot("s")
        folio.metadata["params"]["lr"] = 999  # nested edit of LIVE metadata
        assert folio._snapshots["s"]["metadata_snapshot"]["params"]["lr"] == 0.1


class TestRevisionBasedStaleness:
    def test_reader_converges_even_if_metadata_write_crashed(self, tmp_path):
        """Staleness detection keys off the items.json revision, so a crash
        between the items and metadata writes can't blind other readers."""
        import json

        path = tmp_path / "b"
        writer = DataFolio(path)
        writer.add("a", 1)
        reader = DataFolio(path)

        # Simulate a committed item whose metadata bump never landed:
        # bump items.json revision directly with a new item entry.
        items = json.loads((path / "items.json").read_text())
        new_entry = dict(items["items"][0])
        new_entry.update(
            name="b", filename=items["items"][0]["filename"], version_id="b--r99"
        )
        items["items"].append(new_entry)
        items["revision"] += 1
        (path / "items.json").write_text(json.dumps(items))
        # metadata.json deliberately NOT touched

        assert "b" in reader.list_contents()["json_data"] or "b" in reader._items

    def test_metadata_only_change_still_detected(self, tmp_path):
        path = tmp_path / "b"
        a = DataFolio(path)
        a.add("seed", 1)
        b = DataFolio(path)
        a.metadata["k"] = "new"
        assert b.metadata.get("k") == "new"  # read auto-refreshes via revision


class TestMetadataProperty:
    def test_plain_dict_assignment_is_adopted_and_committed(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.metadata = {"replaced": True}
        assert DataFolio(path).metadata["replaced"] is True
        # Auto-save still works afterwards
        folio.metadata["after"] = 1
        assert DataFolio(path).metadata["after"] == 1

    def test_non_dict_assignment_raises(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        with pytest.raises(TypeError):
            folio.metadata = "not a mapping"

    def test_read_only_assignment_raises(self, tmp_path):
        path = tmp_path / "b"
        DataFolio(path).metadata["k"] = 1
        ro = DataFolio(path, read_only=True)
        with pytest.raises(RuntimeError):
            ro.metadata = {"nope": 1}


class TestNumpyScalars:
    def test_numpy_scalars_store_as_json(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("f", np.float64(0.5))
        folio.add("i", np.int64(7))
        folio.add("bl", np.bool_(True))
        assert folio.get("f") == 0.5
        assert folio.get("i") == 7
        assert folio.get("bl") is True
        for name in ("f", "i", "bl"):
            assert folio._items[name]["item_type"] == "json_data"

    def test_dict_with_numpy_values_stores(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("metrics", {"acc": np.float64(0.9), "arr": np.array([1, 2])})
        back = folio.get("metrics")
        assert back["acc"] == 0.9
        assert back["arr"] == [1, 2]

    def test_unserializable_dict_gets_clear_error(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        with pytest.raises(TypeError, match="JSON"):
            folio.add("big", {"n": 2**70})


class TestNameLengthHeadroom:
    def test_overlong_segment_rejected_with_clear_error(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        with pytest.raises(ValueError, match="segment"):
            folio.add("x" * 240, 1)

    def test_long_but_legal_name_works(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        name = "x" * 150
        folio.add(name, 1)
        assert folio.get(name) == 1


class TestAccessorIterationAndIsolation:
    def test_contains_and_iter(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("alpha", 1)
        folio.add("beta", 2)
        assert "alpha" in folio.data
        assert "nope" not in folio.data
        assert sorted(folio.data) == ["alpha", "beta"]

    def test_two_folios_do_not_clobber_each_other(self, tmp_path):
        f1 = DataFolio(tmp_path / "a")
        f1.add("only_in_a", 1)
        f2 = DataFolio(tmp_path / "b")
        f2.add("only_in_b", 2)
        acc1, acc2 = f1.data, f2.data
        # Both accessors expose exactly their own folio's items
        assert acc1.only_in_a.content == 1
        assert acc2.only_in_b.content == 2
        assert "only_in_b" not in acc1
        assert "only_in_a" not in acc2
        # And attribute namespaces are isolated (instance-level)
        assert "only_in_b" not in vars(acc1)
        assert "only_in_a" not in vars(acc2)


class TestCliValidate:
    def test_cli_validate_fails_on_missing_payload(self, tmp_path):
        from click.testing import CliRunner

        from datafolio.cli import cli

        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.add("x", pd.DataFrame({"v": [1]}))
        payload = path / "tables" / folio._items["x"]["filename"]
        payload.unlink()

        result = CliRunner().invoke(cli, ["validate", str(path)])
        assert result.exit_code == 1
        assert "x" in result.output

    def test_cli_validate_passes_on_healthy_folio(self, tmp_path):
        from click.testing import CliRunner

        from datafolio.cli import cli

        path = tmp_path / "b"
        DataFolio(path).add("x", 1)
        result = CliRunner().invoke(cli, ["validate", str(path)])
        assert result.exit_code == 0


class TestInitExistingDirectory:
    def test_library_allows_existing_empty_dir(self, tmp_path):
        target = tmp_path / "empty"
        target.mkdir()
        folio = DataFolio(target)
        folio.add("x", 1)
        assert DataFolio(target).get("x") == 1

    def test_library_refuses_nonempty_non_bundle_dir(self, tmp_path):
        target = tmp_path / "occupied"
        target.mkdir()
        (target / "file.txt").write_text("hi")
        with pytest.raises(FileExistsError, match="random_suffix"):
            DataFolio(target)

    def test_allow_existing_opts_in(self, tmp_path):
        target = tmp_path / "occupied"
        target.mkdir()
        (target / "file.txt").write_text("hi")
        folio = DataFolio(target, allow_existing=True)
        folio.add("x", 1)
        assert DataFolio(target).get("x") == 1
        assert (target / "file.txt").exists()  # existing files untouched

    def test_cli_init_existing_dir_with_confirmation(self, tmp_path):
        from click.testing import CliRunner

        from datafolio.cli import cli

        target = tmp_path / "workdir"
        target.mkdir()
        (target / "notes.txt").write_text("hi")
        result = CliRunner().invoke(cli, ["init", str(target)], input="y\n")
        assert result.exit_code == 0, result.output
        assert (target / "items.json").exists()


class TestWaveCLowSeverity:
    def test_item_proxy_errors_consistent_after_delete(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        proxy = folio.data.x
        folio.delete("x")
        for attr in ("description", "type", "metadata"):
            with pytest.raises(KeyError, match="not found in DataFolio"):
                getattr(proxy, attr)
        # repr must not raise (Jupyter renders it)
        assert "x" in repr(proxy)

    def test_preserve_index_name_collision_clear_error(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([9, 8], name="a"))
        with pytest.raises(ValueError, match="index.*column|column.*index"):
            folio.add("t", df, preserve_index=True)
        assert "t" not in folio._items  # nothing half-added

    def test_unnamed_multiindex_names_restore_as_none(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        idx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])  # unnamed levels
        df = pd.DataFrame({"v": [1, 2]}, index=idx)
        folio.add("t", df, preserve_index=True)
        back = folio.get("t")
        assert list(back.index.names) == [None, None]
        assert back["v"].tolist() == [1, 2]

    def test_lineage_deduped_with_legacy_models_field(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("m", 1)
        folio.add("t", pd.DataFrame({"v": [1]}), inputs=["m"])
        folio._items["t"]["models"] = ["m"]  # legacy manifest shape
        assert folio.get_inputs("t") == ["m"]
        assert folio.get_dependents("m") == ["t"]

    def test_archive_zero_match_glob_is_noop(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        rev = folio._manifest_revision
        folio.archive("no-such-prefix/*")
        assert folio._manifest_revision == rev  # no pointless commit

    def test_add_file_derived_name_hint(self, tmp_path):
        src = tmp_path / "my plot.png"
        src.write_bytes(b"x")
        folio = DataFolio(tmp_path / "b")
        with pytest.raises(ValueError, match="name="):
            folio.add_file(src)
        folio.add_file(src, name="my_plot")  # explicit name works
        assert "my_plot" in folio._items

    def test_metadata_update_midway_exception_rolls_back(self, tmp_path):
        path = tmp_path / "b"
        folio = DataFolio(path)
        folio.metadata["keep"] = 1

        def poison():
            yield ("ok", 2)
            raise RuntimeError("iterator exploded")

        with pytest.raises(RuntimeError):
            folio.metadata.update(poison())
        # Partial in-memory update discarded; later commit doesn't leak it
        assert "ok" not in folio.metadata
        folio.add("y", 1)
        assert "ok" not in DataFolio(path).metadata
