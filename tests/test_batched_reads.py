"""Tests for the round-trip reductions on the read path.

Three related changes: the staleness check reads the manifest directly
instead of probing with exists() first, validate() resolves existence for
every item in one batched call, and get_many() overlaps reads concurrently.
"""

import threading

import numpy as np
import pandas as pd
import pytest

from datafolio import DataFolio
from datafolio.storage import StorageBackend


@pytest.fixture
def folio(tmp_path):
    """A folio with a handful of items of mixed types."""
    f = DataFolio(tmp_path / "batched")
    f.add("table", pd.DataFrame({"a": [1, 2, 3]}))
    f.add("arr", np.arange(10))
    for i in range(5):
        f.add(f"cfg_{i}", {"value": i})
    return f


class TestStalenessCheckIsOneRoundTrip:
    """The check reads the manifest; it no longer probes with exists()."""

    def test_no_exists_probe(self, folio, monkeypatch):
        calls = []
        original = StorageBackend.exists
        monkeypatch.setattr(
            StorageBackend,
            "exists",
            lambda self, path: (calls.append(path), original(self, path))[1],
        )

        folio.get("cfg_0")

        assert not [p for p in calls if p.endswith("items.json")]

    def test_missing_manifest_is_not_stale(self, tmp_path):
        """A missing manifest must read as 'not stale', as the probe did."""
        f = DataFolio(tmp_path / "gone")
        f.add("data", {"v": 1})
        (tmp_path / "gone" / "items.json").unlink()

        # Reads keep serving committed memory rather than crashing.
        assert f._check_if_stale() is False
        assert f.get("data") == {"v": 1}

    def test_corrupt_manifest_is_not_stale(self, tmp_path):
        f = DataFolio(tmp_path / "corrupt")
        f.add("data", {"v": 1})
        (tmp_path / "corrupt" / "items.json").write_text("not json {")

        assert f._check_if_stale() is False
        assert f.get("data") == {"v": 1}

    def test_still_detects_a_real_advance(self, tmp_path):
        f1 = DataFolio(tmp_path / "shared")
        f1.add("a", {"v": 1})
        f2 = DataFolio(tmp_path / "shared")
        f2.get("a")

        f1.add("b", {"v": 2})

        assert f2._check_if_stale() is True
        assert f2.get("b") == {"v": 2}


class TestExistsMany:
    """Batched existence, with single-path semantics preserved."""

    def test_mixed_present_and_missing(self, tmp_path):
        storage = StorageBackend()
        present = tmp_path / "here.txt"
        present.write_text("x")
        missing = tmp_path / "gone.txt"

        result = storage.exists_many([str(present), str(missing)])

        assert result == {str(present): True, str(missing): False}

    def test_empty_input(self, tmp_path):
        assert StorageBackend().exists_many([]) == {}

    def test_duplicates_collapse_but_all_keys_returned(self, tmp_path):
        storage = StorageBackend()
        p = tmp_path / "f.txt"
        p.write_text("x")

        result = storage.exists_many([str(p), str(p)])

        assert result == {str(p): True}

    def test_cloud_paths_group_into_one_call_per_directory(self, monkeypatch):
        """The actual batching: one threaded CloudFiles.exists() per dir."""
        import cloudfiles

        batches = []
        singles = []

        class FakeCF:
            def __init__(self, path, use_https=False):
                self.path = path

            def exists(self, paths):
                if isinstance(paths, str):
                    singles.append((self.path, paths))
                    return not paths.startswith("missing")
                batches.append((self.path, list(paths)))
                return {p: not p.startswith("missing") for p in paths}

            def list(self, prefix="", flat=False):
                return iter(())

        monkeypatch.setattr(cloudfiles, "CloudFiles", FakeCF)

        result = StorageBackend().exists_many(
            [
                "gs://bucket/folio/tables/a.parquet",
                "gs://bucket/folio/tables/b.parquet",
                "gs://bucket/folio/models/m.joblib",
                "gs://bucket/folio/tables/missing.parquet",
            ]
        )

        # Two directories -> two batched calls, not four single ones.
        assert len(batches) == 2
        assert sorted(len(files) for _, files in batches) == [1, 3]
        # Only the miss falls back to a single-path check (directory probe).
        assert len(singles) == 1
        assert result == {
            "gs://bucket/folio/tables/a.parquet": True,
            "gs://bucket/folio/tables/b.parquet": True,
            "gs://bucket/folio/models/m.joblib": True,
            "gs://bucket/folio/tables/missing.parquet": False,
        }

    def test_matches_exists_for_directories(self, tmp_path):
        """Directory payloads (sharded datasets) resolve as they do singly."""
        storage = StorageBackend()
        d = tmp_path / "shards"
        d.mkdir()
        (d / "part-0.parquet").write_text("x")

        assert storage.exists_many([str(d)]) == {str(d): storage.exists(str(d))}
        assert storage.exists_many([str(d)])[str(d)] is True


class TestValidateBatches:
    """validate() keeps its contract while checking existence in one call."""

    def test_all_valid(self, folio):
        status = folio.validate()

        assert set(status) == set(folio._items)
        assert all(status.values())
        assert folio.is_valid()

    def test_uses_one_batched_call(self, folio, monkeypatch):
        """Every item's existence is resolved in a single call.

        Locally that call still checks paths one by one — there is nothing to
        batch on a filesystem. The win is on cloud paths, which group into one
        threaded CloudFiles.exists() per directory (see TestExistsMany).
        """
        batched = []
        original = StorageBackend.exists_many
        monkeypatch.setattr(
            StorageBackend,
            "exists_many",
            lambda self, paths: (
                batched.append(list(paths)),
                original(self, paths),
            )[1],
        )

        folio.validate()

        assert len(batched) == 1
        assert len(batched[0]) == 7  # every item resolved in one call

    def test_missing_payload_reported_false(self, tmp_path):
        f = DataFolio(tmp_path / "b")
        f.add("good", {"v": 1})
        f.add("bad", {"v": 2})
        path = f.item_path("bad")
        __import__("pathlib").Path(path).unlink()

        status = f.validate()

        assert status["good"] is True
        assert status["bad"] is False
        assert not f.is_valid()

    def test_broken_reference_reported_false(self, tmp_path):
        f = DataFolio(tmp_path / "b")
        f.reference_table("ext", path=str(tmp_path / "nope.parquet"))

        assert f.validate()["ext"] is False

    def test_batch_failure_falls_back_per_path(self, folio, monkeypatch):
        """A failing batch must not sink the whole report."""
        monkeypatch.setattr(
            StorageBackend,
            "exists_many",
            lambda self, paths: (_ for _ in ()).throw(OSError("transient")),
        )

        status = folio.validate()

        assert all(status.values())  # per-path fallback found everything

    def test_per_path_errors_are_failures_not_crashes(self, folio, monkeypatch):
        monkeypatch.setattr(
            StorageBackend,
            "exists_many",
            lambda self, paths: (_ for _ in ()).throw(OSError("transient")),
        )
        monkeypatch.setattr(
            StorageBackend,
            "exists",
            lambda self, path: (_ for _ in ()).throw(OSError("still down")),
        )

        status = folio.validate()

        assert status  # a report, not an exception
        assert not any(status.values())


class TestGetMany:
    """Sugar over pinned() + a thread pool."""

    def test_returns_every_requested_item(self, folio):
        names = ["cfg_0", "cfg_1", "arr", "table"]

        got = folio.get_many(names)

        assert list(got) == names
        assert got["cfg_1"] == {"value": 1}
        np.testing.assert_array_equal(got["arr"], np.arange(10))
        pd.testing.assert_frame_equal(got["table"], pd.DataFrame({"a": [1, 2, 3]}))

    def test_matches_sequential_gets(self, folio):
        names = [f"cfg_{i}" for i in range(5)]

        assert folio.get_many(names) == {n: folio.get(n) for n in names}

    def test_empty_and_single(self, folio):
        assert folio.get_many([]) == {}
        assert folio.get_many(["cfg_0"]) == {"cfg_0": {"value": 0}}

    def test_duplicates_fetched_once(self, folio):
        assert folio.get_many(["cfg_0", "cfg_0", "cfg_1"]) == {
            "cfg_0": {"value": 0},
            "cfg_1": {"value": 1},
        }

    def test_threads_one_is_sequential(self, folio):
        names = [f"cfg_{i}" for i in range(5)]

        assert folio.get_many(names, threads=1) == folio.get_many(names)

    def test_missing_name_raises(self, folio):
        with pytest.raises(KeyError):
            folio.get_many(["cfg_0", "nope"])

    def test_type_opts_forwarded(self, folio):
        import polars as pl

        got = folio.get_many(["table"], frame="polars")

        assert isinstance(got["table"], pl.DataFrame)

    def test_type_opt_that_does_not_apply_raises(self, folio):
        with pytest.raises(TypeError):
            folio.get_many(["cfg_0"], frame="polars")

    def test_pins_for_the_duration(self, tmp_path, folio):
        """Another writer's changes are not adopted mid-call."""
        depths = []
        original = DataFolio.get

        def spy(self, name, **opts):
            depths.append(self._pin_depth)
            return original(self, name, **opts)

        folio.get_many(["cfg_0", "cfg_1"], threads=1)  # warm, unpatched
        DataFolio.get = spy
        try:
            folio.get_many(["cfg_0", "cfg_1"], threads=1)
        finally:
            DataFolio.get = original

        assert depths and all(d > 0 for d in depths)
        assert folio._pin_depth == 0  # unpinned on the way out

    def test_joins_an_outer_pin(self, folio):
        with folio.pinned():
            folio.get_many(["cfg_0", "cfg_1"])
            assert folio._pin_depth == 1  # still pinned by the outer block
        assert folio._pin_depth == 0

    def test_concurrent_reads_are_correct(self, tmp_path):
        """The real point: many threads, no torn reads."""
        f = DataFolio(tmp_path / "concurrent")
        expected = {}
        for i in range(40):
            df = pd.DataFrame({"x": range(5), "i": i})
            f.add(f"t_{i}", df)
            expected[f"t_{i}"] = df

        for _ in range(5):
            got = f.get_many(list(expected), threads=20)
            assert list(got) == list(expected)
            for name, df in expected.items():
                pd.testing.assert_frame_equal(got[name], df)

    def test_reads_overlap(self, tmp_path, monkeypatch):
        """Concurrency is real: reads are in flight simultaneously."""
        f = DataFolio(tmp_path / "overlap")
        for i in range(8):
            f.add(f"cfg_{i}", {"i": i})

        in_flight = 0
        peak = 0
        lock = threading.Lock()
        barrier = threading.Barrier(8, timeout=10)
        original = StorageBackend.read_json

        def tracked(self, path):
            nonlocal in_flight, peak
            if path.endswith("items.json"):
                return original(self, path)
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                barrier.wait()  # only passes if all 8 are concurrent
                return original(self, path)
            finally:
                with lock:
                    in_flight -= 1

        monkeypatch.setattr(StorageBackend, "read_json", tracked)
        f.get_many([f"cfg_{i}" for i in range(8)], threads=8)

        assert peak == 8
