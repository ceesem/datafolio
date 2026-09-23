"""CLI tests for `datafolio find` (search logic is covered in test_search.py)."""

import pandas as pd
import pytest
from click.testing import CliRunner

import datafolio
from datafolio import DataFolio
from datafolio.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def folios(tmp_path):
    a = DataFolio(tmp_path / "a", alias="alpha", metadata={"dataset": "minnie65"})
    a.add("cells", pd.DataFrame({"x": [1]}), description="cell table")
    a.add("config", {"k": 1})
    b = DataFolio(tmp_path / "b", alias="beta")
    b.add("synapses", pd.DataFrame({"y": [2]}))
    return a, b


def test_find_prints_matches(runner, folios):
    result = runner.invoke(cli, ["find", "cells"])
    assert result.exit_code == 0, result.output
    assert "alpha" in result.output
    assert "cell table" in result.output


def test_find_regex_and_type(runner, folios):
    result = runner.invoke(cli, ["find", "c", "--regex", "--type", "json"])
    assert result.exit_code == 0
    assert "config" in result.output
    assert "cells" not in result.output


def test_find_in(runner, folios):
    result = runner.invoke(cli, ["find", "--in", "beta"])
    assert "synapses" in result.output
    assert "cells" not in result.output


def test_find_metadata(runner, folios):
    result = runner.invoke(cli, ["find", "dataset=minnie*", "--metadata"])
    assert result.exit_code == 0
    assert "minnie65" in result.output


def test_no_match_exits_1(runner, folios):
    result = runner.invoke(cli, ["find", "nothing-here"])
    assert result.exit_code == 1
    assert "No matches" in result.output


def test_unreachable_folio_warns(runner, folios, tmp_path):
    datafolio.set_alias("ghost", tmp_path / "ghost")
    result = runner.invoke(cli, ["find", "cells"])
    assert result.exit_code == 0
    assert "Skipping folio 'ghost'" in result.output


def test_bad_regex(runner, folios):
    result = runner.invoke(cli, ["find", "(", "--regex"])
    assert result.exit_code == 1
    assert "Invalid regex" in result.output


def test_bad_type_is_usage_error(runner):
    assert runner.invoke(cli, ["find", "--type", "spreadsheet"]).exit_code == 2


def test_find_desc(runner, folios):
    folios[1].add(
        "tbl_7", pd.DataFrame({"z": [1]}), description="synapse table I liked"
    )
    assert runner.invoke(cli, ["find", "synapse"]).exit_code == 1
    result = runner.invoke(cli, ["find", "synapse", "--desc"])
    assert result.exit_code == 0
    assert "tbl_7" in result.output
    assert "synapse table I liked" in result.output


def test_no_match_hints_at_descriptions(runner, folios):
    folios[1].add("tbl_7", pd.DataFrame({"z": [1]}), description="synapse table")
    result = runner.invoke(cli, ["find", "synapse"])
    assert result.exit_code == 1
    assert "1 item mentions it in its description; add --desc" in result.output


def test_no_match_hint_plural(runner, folios):
    folios[1].add("t1", pd.DataFrame({"z": [1]}), description="synapse a")
    folios[1].add("t2", pd.DataFrame({"z": [1]}), description="synapse b")
    result = runner.invoke(cli, ["find", "synapse"])
    assert "2 items mention it in their descriptions" in result.output


def test_no_match_without_description_hits_has_no_hint(runner, folios):
    result = runner.invoke(cli, ["find", "nothing-here"])
    assert result.exit_code == 1
    assert "--desc" not in result.output


def test_desc_hint_respects_type_filter(runner, folios):
    folios[1].add("tbl_7", pd.DataFrame({"z": [1]}), description="synapse table")
    result = runner.invoke(cli, ["find", "synapse", "--type", "model"])
    assert "--desc" not in result.output
