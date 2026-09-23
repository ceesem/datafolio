"""Integration tests for `datafolio add`, `folios`, `init --alias`, and recents."""

import hashlib

import numpy as np
import orjson
import pandas as pd
import pytest
from click.testing import CliRunner

from datafolio import DataFolio
from datafolio.cli.main import cli
from datafolio.folio_registry import FolioRegistry


def _unwrapped(output: str) -> str:
    """Collapse Rich's line wrapping, which depends on the tmp path length."""
    return " ".join(output.split())


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def folio_dir(tmp_path):
    path = tmp_path / "folio"
    DataFolio(path, alias="demo")
    return path


@pytest.fixture
def downloads(tmp_path, df):
    d = tmp_path / "Downloads"
    d.mkdir()
    df.to_parquet(d / "blah.parquet", index=False)
    df.to_csv(d / "cells.csv", index=False)
    df.to_feather(d / "table.feather")
    np.save(d / "arr.npy", np.arange(5))
    (d / "config.json").write_bytes(orjson.dumps({"lr": 0.1}))
    (d / "notes.txt").write_text("hello")
    return d


def _invoke(runner, args):
    result = runner.invoke(cli, args, catch_exceptions=False)
    return result


class TestAddTables:
    @pytest.mark.parametrize("fname", ["blah.parquet", "cells.csv", "table.feather"])
    def test_tables_stored_as_parquet(self, runner, folio_dir, downloads, df, fname):
        result = _invoke(
            runner, ["add", "--to", str(folio_dir), str(downloads / fname)]
        )
        assert result.exit_code == 0, result.output
        name = fname.split(".")[0]
        folio = DataFolio(folio_dir)
        info = folio.item_info(name)
        assert info["item_type"] == "included_table"
        assert info["table_format"] == "parquet"
        pd.testing.assert_frame_equal(folio.get(name), df, check_dtype=False)
        assert "included_table" in result.output

    def test_reference(self, runner, folio_dir, downloads):
        result = _invoke(
            runner,
            [
                "add",
                "-a",
                "demo",
                str(downloads / "cells.csv"),
                "--reference",
            ],
        )
        assert result.exit_code == 0, result.output
        info = DataFolio(folio_dir).item_info("cells")
        assert info["item_type"] == "referenced_table"
        assert info["path"].endswith(str((downloads / "cells.csv").resolve()))

    def test_reference_rejects_feather(self, runner, folio_dir, downloads):
        result = runner.invoke(
            cli,
            [
                "add",
                "-a",
                "demo",
                str(downloads / "table.feather"),
                "--reference",
            ],
        )
        assert result.exit_code != 0
        assert "only supports parquet and csv" in result.output


class TestAddOtherTypes:
    def test_npy(self, runner, folio_dir, downloads):
        _invoke(runner, ["add", "-a", "demo", str(downloads / "arr.npy")])
        np.testing.assert_array_equal(DataFolio(folio_dir).get("arr"), np.arange(5))

    def test_json(self, runner, folio_dir, downloads):
        _invoke(runner, ["add", "-a", "demo", str(downloads / "config.json")])
        assert DataFolio(folio_dir).get("config") == {"lr": 0.1}

    def test_unknown_extension_is_artifact(self, runner, folio_dir, downloads):
        _invoke(runner, ["add", "-a", "demo", str(downloads / "notes.txt")])
        assert DataFolio(folio_dir).item_info("notes")["item_type"] == "artifact"

    @pytest.mark.parametrize("fname", ["cells.csv", "blah.parquet"])
    def test_as_file_keeps_bytes(self, runner, folio_dir, downloads, fname):
        src = downloads / fname
        result = _invoke(runner, ["add", "-a", "demo", str(src), "--as-file"])
        assert result.exit_code == 0, result.output
        folio = DataFolio(folio_dir)
        name = fname.split(".")[0]
        assert folio.item_info(name)["item_type"] == "artifact"
        stored = folio.item_path(name)
        assert stored.endswith(src.suffix)
        with open(stored, "rb") as f:
            assert (
                hashlib.md5(f.read()).hexdigest()
                == hashlib.md5(src.read_bytes()).hexdigest()
            )

    def test_reference_and_as_file_conflict(self, runner, folio_dir, downloads):
        result = runner.invoke(
            cli,
            [
                "add",
                "-a",
                "demo",
                str(downloads / "cells.csv"),
                "--reference",
                "--as-file",
            ],
        )
        assert result.exit_code == 2
        assert "cannot be combined" in result.output


class TestAddPositional:
    def test_readme_example(self, runner, folio_dir, downloads):
        """`datafolio add -a my-folio table.parquet -d '...'`, then find it."""
        result = _invoke(
            runner,
            [
                "add",
                "-a",
                "demo",
                str(downloads / "blah.parquet"),
                "-d",
                "this synapse table i thought was interesting",
            ],
        )
        assert result.exit_code == 0, result.output
        info = DataFolio(folio_dir).item_info("blah")
        assert info["description"] == "this synapse table i thought was interesting"
        found = _invoke(runner, ["find", "synapse", "--desc"])
        assert found.exit_code == 0
        assert "blah" in found.output

    def test_file_is_required(self, runner, folio_dir):
        result = runner.invoke(cli, ["add", "-a", "demo"])
        assert result.exit_code == 2
        assert "FILE" in result.output

    def test_old_file_flag_is_gone(self, runner, folio_dir, downloads):
        result = runner.invoke(
            cli, ["add", "-a", "demo", "--file", str(downloads / "cells.csv")]
        )
        assert result.exit_code == 2


class TestAddOptions:
    def test_name_and_description(self, runner, folio_dir, downloads):
        _invoke(
            runner,
            [
                "add",
                "-a",
                "demo",
                str(downloads / "cells.csv"),
                "--name",
                "my_cells",
                "-d",
                "Cell table",
            ],
        )
        info = DataFolio(folio_dir).item_info("my_cells")
        assert info["description"] == "Cell table"

    def test_invalid_stem_needs_name(self, runner, folio_dir, tmp_path, df):
        bad = tmp_path / "bad name!.csv"
        df.to_csv(bad, index=False)
        result = runner.invoke(cli, ["add", "-a", "demo", str(bad)])
        assert result.exit_code != 0
        assert "--name" in result.output

    def test_overwrite(self, runner, folio_dir, downloads):
        args = ["add", "-a", "demo", str(downloads / "cells.csv")]
        assert _invoke(runner, args).exit_code == 0
        result = runner.invoke(cli, args)
        assert result.exit_code == 1
        assert "already exists" in result.output
        assert _invoke(runner, args + ["--overwrite"]).exit_code == 0

    def test_missing_file(self, runner, folio_dir, tmp_path):
        result = runner.invoke(cli, ["add", "-a", "demo", str(tmp_path / "nope.csv")])
        assert result.exit_code == 2


class TestAddTargets:
    def test_global_alias(self, runner, folio_dir, downloads):
        result = _invoke(runner, ["-a", "demo", "add", str(downloads / "cells.csv")])
        assert result.exit_code == 0, result.output
        assert "cells" in DataFolio(folio_dir).list_contents()["included_tables"]

    def test_current_directory(self, runner, folio_dir, downloads, monkeypatch):
        monkeypatch.chdir(folio_dir)
        monkeypatch.delenv("DATAFOLIO_PATH", raising=False)
        result = _invoke(runner, ["add", str(downloads / "cells.csv")])
        assert result.exit_code == 0, result.output

    def test_tilde_in_to(self, runner, folio_dir, downloads, monkeypatch):
        monkeypatch.setenv("HOME", str(folio_dir.parent))
        result = _invoke(
            runner, ["add", "--to", "~/folio", str(downloads / "cells.csv")]
        )
        assert result.exit_code == 0, result.output

    def test_to_and_alias_conflict(self, runner, folio_dir, downloads):
        result = runner.invoke(
            cli,
            [
                "add",
                "--to",
                str(folio_dir),
                "-a",
                "demo",
                str(downloads / "cells.csv"),
            ],
        )
        assert result.exit_code == 2

    def test_global_folio_and_alias_conflict(self, runner, folio_dir):
        result = runner.invoke(cli, ["-f", str(folio_dir), "-a", "demo", "describe"])
        assert result.exit_code == 2

    def test_to_with_alias_name_hints_dash_a(self, runner, folio_dir, downloads):
        result = runner.invoke(
            cli, ["add", "--to", "demo", str(downloads / "cells.csv")]
        )
        assert result.exit_code == 1
        output = _unwrapped(result.output)
        assert "'demo' is a registered alias" in output
        assert "Use -a demo" in output

    def test_global_f_with_alias_name_hints_dash_a(self, runner, folio_dir):
        result = runner.invoke(cli, ["-f", "demo", "describe"])
        assert result.exit_code == 1
        assert "Use -a demo" in _unwrapped(result.output)

    def test_no_hint_for_unregistered_name(self, runner, downloads):
        result = runner.invoke(
            cli, ["add", "--to", "nothing", str(downloads / "cells.csv")]
        )
        assert result.exit_code == 1
        assert "registered alias" not in result.output

    def test_unknown_alias(self, runner, downloads):
        result = runner.invoke(cli, ["add", "-a", "nope", str(downloads / "cells.csv")])
        assert result.exit_code == 1
        assert "No folio registered as 'nope'" in result.output

    def test_never_creates_a_folio(self, runner, tmp_path, downloads):
        target = tmp_path / "not-a-folio"
        result = runner.invoke(
            cli, ["add", "--to", str(target), str(downloads / "cells.csv")]
        )
        assert result.exit_code == 1
        assert not target.exists()


class TestRecents:
    def test_successful_command_records_recent(self, runner, folio_dir, downloads):
        _invoke(runner, ["add", "-a", "demo", str(downloads / "cells.csv")])
        assert FolioRegistry().recent()[0]["path"] == str(folio_dir.resolve())

    def test_existing_commands_record_recent(self, runner, folio_dir):
        _invoke(runner, ["-f", str(folio_dir), "describe"])
        assert FolioRegistry().recent()[0]["path"] == str(folio_dir.resolve())

    def test_failed_command_does_not_record(self, runner, tmp_path, downloads):
        runner.invoke(cli, ["add", "--to", str(tmp_path), str(downloads / "x.csv")])
        assert FolioRegistry().recent() == []

    def test_existing_commands_accept_alias(self, runner, folio_dir):
        result = _invoke(runner, ["-a", "demo", "snapshot", "create", "v1"])
        assert result.exit_code == 0, result.output
        assert "v1" in DataFolio(folio_dir).list_snapshots()[0]["name"]


class TestFoliosGroup:
    def test_list(self, runner, folio_dir):
        result = _invoke(runner, ["folios", "list"])
        assert result.exit_code == 0
        assert "demo" in result.output

    def test_list_empty(self, runner):
        result = _invoke(runner, ["folios", "list"])
        assert "No folios registered" in result.output

    def test_alias_and_unalias(self, runner, folio_dir):
        result = _invoke(runner, ["folios", "alias", "second", str(folio_dir)])
        assert result.exit_code == 0, result.output
        assert FolioRegistry().get_alias("second") == str(folio_dir.resolve())
        assert _invoke(runner, ["folios", "unalias", "second"]).exit_code == 0
        assert "second" not in FolioRegistry().aliases()

    def test_alias_defaults_to_current_folio(self, runner, folio_dir, monkeypatch):
        monkeypatch.chdir(folio_dir)
        monkeypatch.delenv("DATAFOLIO_PATH", raising=False)
        assert _invoke(runner, ["folios", "alias", "here"]).exit_code == 0
        assert FolioRegistry().get_alias("here") == str(folio_dir.resolve())

    def test_alias_conflict_needs_overwrite(self, runner, folio_dir, tmp_path):
        other = tmp_path / "other"
        DataFolio(other)
        result = runner.invoke(cli, ["folios", "alias", "demo", str(other)])
        assert result.exit_code == 1
        assert "already points to" in result.output
        result = _invoke(runner, ["folios", "alias", "demo", str(other), "--overwrite"])
        assert result.exit_code == 0

    def test_alias_rejects_non_folio(self, runner, tmp_path):
        result = runner.invoke(cli, ["folios", "alias", "x", str(tmp_path)])
        assert result.exit_code == 1
        assert "Not a DataFolio bundle" in result.output

    def test_alias_cloud_path_accepted(self, runner):
        result = _invoke(runner, ["folios", "alias", "remote", "gs://bucket/folio"])
        assert result.exit_code == 0
        assert FolioRegistry().get_alias("remote") == "gs://bucket/folio"

    def test_unalias_unknown(self, runner):
        assert runner.invoke(cli, ["folios", "unalias", "nope"]).exit_code == 1

    def test_forget(self, runner, folio_dir):
        FolioRegistry().record_recent(folio_dir)
        assert _invoke(runner, ["folios", "forget", str(folio_dir)]).exit_code == 0
        assert FolioRegistry().recent() == []
        assert runner.invoke(cli, ["folios", "forget", str(folio_dir)]).exit_code == 1

    def test_prune(self, runner, tmp_path):
        FolioRegistry().set_alias("gone", tmp_path / "gone")
        result = _invoke(runner, ["folios", "prune"])
        assert "Pruned 1" in result.output
        assert "Nothing to prune" in _invoke(runner, ["folios", "prune"]).output


class TestInitAlias:
    def test_init_with_alias(self, runner, tmp_path):
        target = tmp_path / "new"
        result = _invoke(runner, ["init", str(target), "--alias", "fresh"])
        assert result.exit_code == 0, result.output
        assert FolioRegistry().get_alias("fresh") == str(target.resolve())
        assert "datafolio add -a fresh" in result.output
        assert FolioRegistry().recent()[0]["path"] == str(target.resolve())

    def test_init_alias_before_path_and_quoted_tilde(
        self, runner, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _invoke(
            runner, ["init", "--alias", "fly-synapses", "~/work/folios/synapse_data"]
        )
        assert result.exit_code == 0, result.output
        target = tmp_path / "work" / "folios" / "synapse_data"
        assert (target / "items.json").exists()
        assert not (tmp_path / "~").exists()
        assert FolioRegistry().get_alias("fly-synapses") == str(target.resolve())

    def test_init_alias_conflict(self, runner, folio_dir, tmp_path):
        result = runner.invoke(cli, ["init", str(tmp_path / "new"), "--alias", "demo"])
        assert result.exit_code == 1
        assert "already points to" in result.output
