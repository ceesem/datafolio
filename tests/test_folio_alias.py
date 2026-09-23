"""Tests for alias support in the DataFolio Python API."""

import pandas as pd
import pytest

from datafolio import DataFolio
from datafolio.folio_registry import FolioRegistry


def test_plain_open_never_touches_registry(tmp_path, _isolated_folio_registry):
    DataFolio(tmp_path / "f").add("x", 1)
    DataFolio(tmp_path / "f")
    assert not (_isolated_folio_registry / "registry.json").exists()


def test_create_with_alias_registers(tmp_path):
    DataFolio(tmp_path / "f", alias="demo")
    assert FolioRegistry().get_alias("demo") == str((tmp_path / "f").resolve())


def test_open_by_alias_round_trip(tmp_path):
    folio = DataFolio(tmp_path / "f", alias="demo")
    folio.add("table", pd.DataFrame({"a": [1, 2]}))
    reopened = DataFolio(alias="demo")
    pd.testing.assert_frame_equal(reopened.get("table"), pd.DataFrame({"a": [1, 2]}))


def test_open_by_alias_keeps_other_options(tmp_path):
    DataFolio(tmp_path / "f", alias="demo")
    folio = DataFolio(alias="demo", read_only=True)
    with pytest.raises(Exception, match="read-only"):
        folio.add("x", 1)


def test_unknown_alias(tmp_path):
    with pytest.raises(KeyError, match="No folio registered as 'nope'"):
        DataFolio(alias="nope")


def test_neither_path_nor_alias():
    with pytest.raises(ValueError, match="path or an alias"):
        DataFolio()


def test_invalid_alias_fails_before_creating(tmp_path):
    with pytest.raises(ValueError, match="Alias"):
        DataFolio(tmp_path / "f", alias="bad alias")
    assert not (tmp_path / "f").exists()


def test_alias_conflict_and_overwrite(tmp_path):
    DataFolio(tmp_path / "a", alias="demo")
    with pytest.raises(ValueError, match="already points to"):
        DataFolio(tmp_path / "b", alias="demo")
    DataFolio(tmp_path / "b", alias="demo", overwrite_alias=True)
    assert FolioRegistry().get_alias("demo").endswith("b")


def test_reopen_with_same_alias_is_fine(tmp_path):
    DataFolio(tmp_path / "a", alias="demo")
    DataFolio(tmp_path / "a", alias="demo")


def test_set_alias_method(tmp_path):
    folio = DataFolio(tmp_path / "f")
    assert folio.set_alias("later") is folio
    assert DataFolio(alias="later")._bundle_dir == folio._bundle_dir
    other = DataFolio(tmp_path / "g")
    with pytest.raises(ValueError):
        other.set_alias("later")
    other.set_alias("later", overwrite=True)
    assert FolioRegistry().get_alias("later").endswith("g")
