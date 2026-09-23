"""Unit tests for the per-user folio registry (datafolio.folio_registry)."""

from pathlib import Path

import orjson
import pytest

import datafolio
from datafolio.folio_registry import (
    MAX_RECENT,
    FolioRegistry,
    normalize_folio_path,
    registry_home,
    validate_alias,
)


@pytest.fixture
def reg(tmp_path):
    return FolioRegistry(tmp_path / "home")


class TestHome:
    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATAFOLIO_HOME", str(tmp_path / "custom"))
        assert registry_home() == tmp_path / "custom"
        assert FolioRegistry().path == tmp_path / "custom" / "registry.json"

    def test_default_home(self, monkeypatch):
        monkeypatch.delenv("DATAFOLIO_HOME", raising=False)
        assert registry_home() == Path.home() / ".datafolio"


class TestNormalize:
    def test_local_relative_becomes_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert normalize_folio_path("sub/folio") == str(
            (tmp_path / "sub" / "folio").resolve()
        )

    def test_tilde_expanded(self):
        assert normalize_folio_path("~/x") == str((Path.home() / "x").resolve())

    @pytest.mark.parametrize(
        "uri, expected",
        [
            ("gs://bucket/folio/", "gs://bucket/folio"),
            ("s3://bucket/a/b", "s3://bucket/a/b"),
        ],
    )
    def test_cloud_verbatim(self, uri, expected):
        assert normalize_folio_path(uri) == expected

    def test_file_uri_is_local(self, tmp_path):
        assert normalize_folio_path(f"file://{tmp_path}") == str(tmp_path.resolve())


class TestAliases:
    def test_set_and_get(self, reg, tmp_path):
        resolved = reg.set_alias("demo", tmp_path / "f")
        assert resolved == str((tmp_path / "f").resolve())
        assert reg.get_alias("demo") == resolved
        assert reg.aliases() == {"demo": resolved}

    def test_same_mapping_is_noop(self, reg, tmp_path):
        reg.set_alias("demo", tmp_path)
        before = reg.path.read_bytes()
        reg.set_alias("demo", tmp_path)
        assert reg.path.read_bytes() == before

    def test_conflict_raises_without_overwrite(self, reg, tmp_path):
        reg.set_alias("demo", tmp_path / "a")
        with pytest.raises(ValueError, match="already points to"):
            reg.set_alias("demo", tmp_path / "b")
        assert reg.get_alias("demo").endswith("a")

    def test_overwrite_rebinds(self, reg, tmp_path):
        reg.set_alias("demo", tmp_path / "a")
        reg.set_alias("demo", tmp_path / "b", overwrite=True)
        assert reg.get_alias("demo").endswith("b")

    @pytest.mark.parametrize("bad", ["", "has space", "a/b", "x@y", "../up"])
    def test_invalid_alias(self, reg, bad):
        with pytest.raises(ValueError, match="Alias"):
            reg.set_alias(bad, "/tmp/x")

    def test_validate_alias_ok(self):
        validate_alias("my-folio_v1.2")

    def test_unknown_alias_lists_known(self, reg, tmp_path):
        reg.set_alias("one", tmp_path)
        with pytest.raises(KeyError, match="Known aliases: one"):
            reg.get_alias("two")

    def test_remove(self, reg, tmp_path):
        reg.set_alias("demo", tmp_path)
        reg.remove_alias("demo")
        assert reg.aliases() == {}
        with pytest.raises(KeyError):
            reg.remove_alias("demo")

    def test_aliases_for(self, reg, tmp_path):
        reg.set_alias("b", tmp_path)
        reg.set_alias("a", tmp_path)
        reg.set_alias("other", tmp_path / "x")
        assert reg.aliases_for(tmp_path) == ["a", "b"]

    def test_cloud_alias(self, reg):
        reg.set_alias("shared", "gs://bucket/folio/")
        assert reg.get_alias("shared") == "gs://bucket/folio"


class TestRecents:
    def test_record_orders_most_recent_first(self, reg, tmp_path):
        reg.record_recent(tmp_path / "a")
        reg.record_recent(tmp_path / "b")
        reg.record_recent(tmp_path / "a")
        paths = [e["path"] for e in reg.recent()]
        assert paths == [str((tmp_path / p).resolve()) for p in ("a", "b")]

    def test_cap(self, reg, tmp_path):
        for i in range(MAX_RECENT + 5):
            reg.record_recent(tmp_path / f"f{i}")
        recent = reg.recent()
        assert len(recent) == MAX_RECENT
        assert recent[0]["path"].endswith(f"f{MAX_RECENT + 4}")

    def test_cap_never_evicts_aliases(self, reg, tmp_path):
        reg.set_alias("keep", tmp_path / "keep")
        for i in range(MAX_RECENT + 5):
            reg.record_recent(tmp_path / f"f{i}")
        assert "keep" in reg.aliases()

    def test_forget(self, reg, tmp_path):
        reg.record_recent(tmp_path / "a")
        assert reg.forget(tmp_path / "a") is True
        assert reg.forget(tmp_path / "a") is False
        assert reg.recent() == []

    def test_record_never_raises(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("not a directory")
        reg = FolioRegistry(blocker / "home")  # cannot be created
        reg.record_recent(tmp_path)  # must not raise
        assert reg.recent() == []


class TestRobustness:
    def test_missing_file(self, reg):
        assert reg.aliases() == {}
        assert reg.recent() == []

    @pytest.mark.parametrize("content", [b"{not json", b"[1, 2]", b'{"aliases": 3}'])
    def test_corrupt_file_tolerated(self, reg, tmp_path, content):
        reg.home.mkdir(parents=True)
        reg.path.write_bytes(content)
        assert reg.aliases() == {}
        reg.set_alias("demo", tmp_path)  # rewrites a valid file
        assert orjson.loads(reg.path.read_bytes())["aliases"]["demo"]


class TestPruneAndTargets:
    def test_prune_removes_missing_local_only(self, reg, tmp_path):
        live = tmp_path / "live"
        live.mkdir()
        reg.set_alias("live", live)
        reg.set_alias("gone", tmp_path / "gone")
        reg.set_alias("cloud", "gs://bucket/x")
        reg.record_recent(tmp_path / "gone-recent")
        removed = reg.prune()
        assert removed == sorted(
            [
                str((tmp_path / "gone").resolve()),
                str((tmp_path / "gone-recent").resolve()),
            ]
        )
        assert set(reg.aliases()) == {"live", "cloud"}
        assert reg.recent() == []

    def test_all_targets_dedupes(self, reg, tmp_path):
        reg.set_alias("b", tmp_path / "x")
        reg.set_alias("a", tmp_path / "x")
        reg.record_recent(tmp_path / "x")
        reg.record_recent(tmp_path / "y")
        targets = reg.all_targets()
        assert targets == [
            ("a", str((tmp_path / "x").resolve())),
            (None, str((tmp_path / "y").resolve())),
        ]
        assert reg.all_targets(aliases_only=True) == targets[:1]


class TestModuleHelpers:
    def test_set_list_remove(self, tmp_path):
        folio_dir = tmp_path / "f"
        datafolio.DataFolio(folio_dir)
        datafolio.set_alias("demo", folio_dir)
        FolioRegistry().record_recent("gs://bucket/remote")
        df = datafolio.list_folios()
        assert list(df.columns) == ["alias", "path", "last_accessed", "exists"]
        assert df.iloc[0]["alias"] == "demo"
        assert bool(df.iloc[0]["exists"]) is True
        assert df.iloc[1]["exists"] is None
        datafolio.remove_alias("demo")
        assert "demo" not in set(datafolio.list_folios()["alias"].dropna())

    def test_list_empty(self):
        df = datafolio.list_folios()
        assert df.empty
        assert list(df.columns) == ["alias", "path", "last_accessed", "exists"]
