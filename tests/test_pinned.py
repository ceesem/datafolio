"""Tests for pinned(): suspending per-read staleness rechecks.

Against cloud storage each staleness recheck is two round trips (a manifest
read_json plus an exists), so a loop of small reads spends most of its time
re-verifying an unchanged manifest. pinned() pays for that check once on
entry and suspends it for the block.
"""

import pandas as pd
import pytest

from datafolio import DataFolio
from datafolio.utils import ITEMS_FILE


def _count_stale_checks(monkeypatch):
    """Spy on DataFolio._check_if_stale; returns a mutable [count] list."""
    calls = [0]
    original = DataFolio._check_if_stale

    def counting(self):
        calls[0] += 1
        return original(self)

    monkeypatch.setattr(DataFolio, "_check_if_stale", counting)
    return calls


def _count_manifest_round_trips(monkeypatch):
    """Count backend round trips that touch items.json.

    Returns a dict with 'read_json', 'exists' and 'total' keys, updated live.
    Payload reads (a JSON item's own file) are excluded so the count reflects
    manifest verification only.
    """
    import datafolio.storage.backend as B

    counts = {"read_json": 0, "exists": 0, "total": 0}
    read_json = B.StorageBackend.read_json
    exists = B.StorageBackend.exists

    def _bump(kind, path):
        if str(path).endswith(ITEMS_FILE):
            counts[kind] += 1
            counts["total"] += 1

    def counting_read_json(self, path, *a, **k):
        _bump("read_json", path)
        return read_json(self, path, *a, **k)

    def counting_exists(self, path, *a, **k):
        _bump("exists", path)
        return exists(self, path, *a, **k)

    monkeypatch.setattr(B.StorageBackend, "read_json", counting_read_json)
    monkeypatch.setattr(B.StorageBackend, "exists", counting_exists)
    return counts


@pytest.fixture
def folio(tmp_path):
    """A folio with a handful of items of mixed types."""
    f = DataFolio(tmp_path / "pinned")
    f.add("table", pd.DataFrame({"a": [1, 2, 3]}))
    for i in range(5):
        f.add(f"cfg_{i}", {"value": i})
    return f


class TestPinSuppressesStalenessChecks:
    """The point of the block: one check on entry, none per read."""

    def test_reads_inside_pin_do_not_recheck(self, folio, monkeypatch):
        calls = _count_stale_checks(monkeypatch)

        with folio.pinned():
            for i in range(5):
                folio.get(f"cfg_{i}")
            folio.get("table")
            folio.list_contents()

        assert calls[0] == 1  # the entry check, and nothing else

    def test_manifest_round_trips_drop_to_one_check(self, folio, monkeypatch):
        """Round trips, not just check count: one per read → one per block."""
        counts = _count_manifest_round_trips(monkeypatch)

        for i in range(5):
            folio.get(f"cfg_{i}")
        unpinned = counts["total"]

        counts["total"] = counts["read_json"] = counts["exists"] = 0
        with folio.pinned():
            for i in range(5):
                folio.get(f"cfg_{i}")
        pinned = counts["total"]

        assert unpinned == 5  # one manifest read per staleness check
        assert pinned == 1  # one entry check for the whole block
        assert counts["exists"] == 0  # the check never probes with exists()

    def test_entry_check_still_picks_up_external_writes(self, tmp_path, folio):
        """Entering a pin is as fresh as an unpinned read would have been."""
        other = DataFolio(tmp_path / "pinned")
        other.add("added_externally", {"v": 1})

        with folio.pinned():
            assert folio.get("added_externally") == {"v": 1}

    def test_external_write_is_invisible_until_exit(self, tmp_path, folio):
        """The documented trade: other writers are not seen inside the block."""
        with folio.pinned():
            folio.get("cfg_0")
            other = DataFolio(tmp_path / "pinned")
            other.add("late", {"v": 2})
            assert "late" not in folio.list_contents()["json_data"]

        assert "late" in folio.list_contents()["json_data"]


class TestPinNesting:
    """Re-entrancy: inner exits must not unpin an outer block."""

    def test_inner_exit_does_not_unpin_outer(self, folio, monkeypatch):
        calls = _count_stale_checks(monkeypatch)

        with folio.pinned():
            with folio.pinned():
                folio.get("cfg_0")
            # Still pinned: the inner exit dropped depth 2 -> 1, not to 0.
            assert folio._pin_depth == 1
            folio.get("cfg_1")
            folio.get("cfg_2")

        assert folio._pin_depth == 0
        assert calls[0] == 1  # only the outermost entry checked

    def test_deep_nesting_unwinds_exactly(self, folio):
        with folio.pinned():
            with folio.pinned():
                with folio.pinned():
                    assert folio._pin_depth == 3
                assert folio._pin_depth == 2
            assert folio._pin_depth == 1
        assert folio._pin_depth == 0

    def test_yields_self(self, folio):
        with folio.pinned() as f:
            assert f is folio


class TestPinExit:
    """After the block, normal behavior resumes."""

    def test_next_read_after_exit_rechecks(self, folio, monkeypatch):
        calls = _count_stale_checks(monkeypatch)

        with folio.pinned():
            folio.get("cfg_0")
        assert calls[0] == 1

        folio.get("cfg_1")
        assert calls[0] == 2  # rechecking again
        folio.get("cfg_2")
        assert calls[0] == 3

    def test_exception_inside_block_still_unpins(self, folio, monkeypatch):
        calls = _count_stale_checks(monkeypatch)

        with pytest.raises(ValueError):
            with folio.pinned():
                folio.get("cfg_0")
                raise ValueError("boom")

        assert folio._pin_depth == 0
        folio.get("cfg_1")
        assert calls[0] == 2  # entry check + the post-exit recheck

    def test_exception_unwinds_nested_pins(self, folio):
        with pytest.raises(ValueError):
            with folio.pinned():
                with folio.pinned():
                    raise ValueError("boom")
        assert folio._pin_depth == 0


class TestWritesInsidePin:
    """The pin suspends remote staleness rechecks, never local bookkeeping."""

    def test_add_round_trips_inside_and_after(self, tmp_path, folio):
        df = pd.DataFrame({"b": [4, 5, 6]})

        with folio.pinned():
            folio.add("written", {"v": 99})
            folio.add("written_table", df)
            # Readable inside the block (local state was updated).
            assert folio.get("written") == {"v": 99}
            pd.testing.assert_frame_equal(folio.get("written_table"), df)

        # Readable after the block...
        assert folio.get("written") == {"v": 99}
        # ...and committed to disk, visible to a fresh instance.
        reopened = DataFolio(tmp_path / "pinned")
        assert reopened.get("written") == {"v": 99}
        pd.testing.assert_frame_equal(reopened.get("written_table"), df)
        assert "written" in reopened.list_contents()["json_data"]

    def test_overwrite_and_delete_inside_pin(self, tmp_path, folio):
        with folio.pinned():
            folio.add("cfg_0", {"value": "replaced"}, overwrite=True)
            folio.delete("cfg_1")
            assert folio.get("cfg_0") == {"value": "replaced"}
            assert "cfg_1" not in folio

        reopened = DataFolio(tmp_path / "pinned")
        assert reopened.get("cfg_0") == {"value": "replaced"}
        assert "cfg_1" not in reopened

    def test_stale_writer_still_rejected_inside_pin(self, tmp_path, folio):
        """The write path's fail-closed check is independent of the pin: a
        pinned writer must not clobber another writer's committed manifest."""
        from datafolio import ConcurrentWriteError

        with folio.pinned():
            folio.get("cfg_0")  # pin established on the current revision
            other = DataFolio(tmp_path / "pinned")
            other.add("from_other", {"v": 1})  # advances the manifest

            with pytest.raises(ConcurrentWriteError):
                folio.add("mine", {"v": 2})

    def test_batch_inside_pin(self, tmp_path, folio):
        with folio.pinned():
            with folio.batch():
                for i in range(3):
                    folio.add(f"batched_{i}", {"i": i})
            assert folio.get("batched_2") == {"i": 2}

        reopened = DataFolio(tmp_path / "pinned")
        assert reopened.get("batched_0") == {"i": 0}


class TestPinLocalPaths:
    """On local paths the pin is a no-op cost-wise but must not error."""

    def test_local_folio_reads_identically(self, tmp_path):
        f = DataFolio(tmp_path / "local")
        df = pd.DataFrame({"a": [1, 2, 3]})
        f.add("table", df)
        f.add("cfg", {"k": "v"})

        unpinned = (f.get("cfg"), f.list_contents(), f.item_info("table"))
        with f.pinned():
            pinned = (f.get("cfg"), f.list_contents(), f.item_info("table"))
            pd.testing.assert_frame_equal(f.get("table"), df)

        assert pinned == unpinned

    def test_explicit_refresh_still_works_inside_pin(self, tmp_path):
        f1 = DataFolio(tmp_path / "shared")
        f1.add("a", {"v": 1})
        f2 = DataFolio(tmp_path / "shared")

        with f2.pinned():
            f1.add("b", {"v": 2})
            assert "b" not in f2.list_contents()["json_data"]
            f2.refresh()  # explicit refresh is never suppressed
            assert f2.get("b") == {"v": 2}

    def test_pin_on_snapshot_mode_folio(self, tmp_path):
        f = DataFolio(tmp_path / "snap")
        f.add("a", {"v": 1})
        f.create_snapshot("s1")

        snap = DataFolio.load_snapshot(tmp_path / "snap", "s1")
        with snap.pinned():
            assert snap.get("a") == {"v": 1}
        assert snap._pin_depth == 0


class TestPinBenchmark:
    """Acceptance: 60 reads, manifest round trips 60 -> 1."""

    def test_sixty_gets_round_trip_count(self, tmp_path, monkeypatch):
        f = DataFolio(tmp_path / "bench")
        for i in range(60):
            f.add(f"item_{i}", {"i": i})

        counts = _count_manifest_round_trips(monkeypatch)

        for i in range(60):
            f.get(f"item_{i}")
        unpinned = counts["total"]

        counts.update(read_json=0, exists=0, total=0)
        with f.pinned():
            for i in range(60):
                f.get(f"item_{i}")
        pinned = counts["total"]

        assert unpinned == 60
        assert pinned == 1
