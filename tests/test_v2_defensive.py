"""Regression tests for defensive snapshot metadata (P10) and cloud-safe
snapshot export (P11)."""

from unittest.mock import patch

import pytest

from datafolio import DataFolio


class TestP10DefensiveCopies:
    def test_get_snapshot_info_deep_copy(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        folio.create_snapshot("s", tags=["keep"])

        info = folio.get_snapshot_info("s")
        info["item_versions"]["x"] = "CORRUPTED"
        info["tags"].append("injected")
        info["metadata_snapshot"]["evil"] = True

        live = folio._snapshots["s"]
        assert live["item_versions"]["x"] != "CORRUPTED"
        assert "injected" not in live["tags"]
        assert "evil" not in live["metadata_snapshot"]
        # And the snapshot still loads correctly
        assert folio.snapshots["s"].get("x") == 1

    def test_snapshot_view_metadata_and_tags_defensive(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.metadata["acc"] = 0.9
        folio.add("x", 1)
        folio.create_snapshot("s", tags=["a"])

        view = folio.snapshots["s"]
        view.metadata["acc"] = "CORRUPTED"
        view.tags.append("injected")
        view.item_versions["x"] = "CORRUPTED"

        live = folio._snapshots["s"]
        assert live["metadata_snapshot"]["acc"] == 0.9
        assert live["tags"] == ["a"]
        assert live["item_versions"]["x"] != "CORRUPTED"


class TestP11CloudExport:
    def test_export_preserves_cloud_target_uri(self, tmp_path):
        """A gs:// target must reach DataFolio as an intact URI string,
        never mangled through Path ('gs://x' -> 'gs:/x'). Cloud construction
        is redirected to a local dir so no network I/O runs."""
        from datafolio.utils import is_cloud_path

        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        folio.create_snapshot("s")

        cloud_paths_seen = []
        original_init = DataFolio.__init__
        storage_cls = type(folio._storage)
        real_exists = storage_cls.exists

        def spy_init(self, path, *args, **kwargs):
            if is_cloud_path(str(path)):
                cloud_paths_seen.append(str(path))
                path = tmp_path / "export-dest"  # keep the test offline
            return original_init(self, path, *args, **kwargs)

        def offline_exists(self, p):
            if is_cloud_path(str(p)):
                return False  # pretend the cloud target is free, no network
            return real_exists(self, p)

        with (
            patch.object(storage_cls, "exists", offline_exists),
            patch.object(DataFolio, "__init__", spy_init),
        ):
            folio.export_snapshot("s", "gs://bucket/exports/my-exp")

        assert cloud_paths_seen == ["gs://bucket/exports/my-exp"]

    def test_export_source_bundle_cloud_uri_verbatim(self, tmp_path, monkeypatch):
        """_source_snapshot must record a cloud source URI verbatim, never
        Path.resolve()'d into a local-looking string."""
        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        folio.create_snapshot("s")

        # Pretend the source bundle is a cloud folio; stub out the snapshot
        # load (which would otherwise reopen the "cloud" source for real).
        monkeypatch.setattr(folio, "_bundle_dir", "gs://bucket/experiments/src")

        class _StubSnapshotFolio:
            _items: dict = {}
            metadata: dict = {}

        monkeypatch.setattr(folio, "get_snapshot", lambda name: _StubSnapshotFolio())
        out = folio.export_snapshot("s", tmp_path / "export")
        src_info = out.metadata["_source_snapshot"]
        assert src_info["source_bundle"] == "gs://bucket/experiments/src"

    def test_export_local_still_works_end_to_end(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("x", {"v": 1})
        folio.create_snapshot("s")
        out = folio.export_snapshot("s", tmp_path / "export")
        assert out.get("x") == {"v": 1}
        assert out.metadata["_source_snapshot"]["name"] == "s"

    def test_export_rejects_existing_local_target(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        folio.create_snapshot("s")
        (tmp_path / "occupied").mkdir()
        with pytest.raises(ValueError, match="already exists"):
            folio.export_snapshot("s", tmp_path / "occupied")


class TestItemVersionsTypeAnnotation:
    def test_snapshot_metadata_item_versions_are_strings(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add("x", 1)
        folio.create_snapshot("s")
        for token in folio._snapshots["s"]["item_versions"].values():
            assert isinstance(token, str)
