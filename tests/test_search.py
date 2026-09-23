"""Tests for datafolio.find (cross-registry search)."""

import numpy as np
import pandas as pd
import pytest

import datafolio
from datafolio import DataFolio
from datafolio.folio_registry import FolioRegistry
from datafolio.search import ITEM_COLUMNS, METADATA_COLUMNS


@pytest.fixture
def folios(tmp_path):
    """Two aliased folios and one recent-only folio."""
    a = DataFolio(tmp_path / "a", alias="alpha", metadata={"dataset": "minnie65"})
    a.add("cells", pd.DataFrame({"x": [1]}), description="cell table")
    a.add(
        "tbl_0042",
        pd.DataFrame({"s": [1]}),
        description="the Synapse table I thought was interesting",
    )
    a.add("cells_raw", np.zeros(3))
    a.add("config", {"k": 1})

    b = DataFolio(tmp_path / "b", alias="beta", metadata={"dataset": "v1dd"})
    b.add("cells", pd.DataFrame({"y": [2]}))
    b.add("synapses", pd.DataFrame({"z": [3]}))

    c = DataFolio(tmp_path / "c", metadata={"dataset": "minnie65"})
    c.add("soma_table", pd.DataFrame({"w": [4]}))
    FolioRegistry().record_recent(tmp_path / "c")
    return {"a": a, "b": b, "c": c}


def _pairs(df):
    return sorted(zip(df["alias"].fillna("-"), df["name"]))


def test_glob_across_registry(folios):
    df = datafolio.find("cells*")
    assert list(df.columns) == ITEM_COLUMNS
    assert _pairs(df) == [("alpha", "cells"), ("alpha", "cells_raw"), ("beta", "cells")]


def test_glob_is_whole_name_and_case_insensitive(folios):
    assert _pairs(datafolio.find("CELLS")) == [("alpha", "cells"), ("beta", "cells")]
    assert datafolio.find("CELLS", case_sensitive=True).empty


def test_default_pattern_matches_everything(folios):
    assert len(datafolio.find()) == 7


def test_regex(folios):
    df = datafolio.find("syn|soma", regex=True)
    assert _pairs(df) == [("-", "soma_table"), ("beta", "synapses")]


def test_invalid_regex():
    with pytest.raises(ValueError, match="Invalid regex"):
        datafolio.find("(", regex=True)


@pytest.mark.parametrize(
    "item_type, expected",
    [
        ("table", {"cells", "synapses", "soma_table", "tbl_0042"}),
        ("array", {"cells_raw"}),
        ("json", {"config"}),
        (["array", "json"], {"cells_raw", "config"}),
        ("included_table", {"cells", "synapses", "soma_table", "tbl_0042"}),
    ],
)
def test_item_type_filter(folios, item_type, expected):
    assert set(datafolio.find(item_type=item_type)["name"]) == expected


def test_unknown_item_type():
    with pytest.raises(ValueError, match="Unknown item type"):
        datafolio.find(item_type="spreadsheet")


def test_description_and_path_columns(folios, tmp_path):
    row = datafolio.find("cells", folios="alpha").iloc[0]
    assert row["description"] == "cell table"
    assert row["folio_path"] == str((tmp_path / "a").resolve())
    assert row["created_at"]


def test_metadata_keys_and_values(folios):
    df = datafolio.find("dataset", metadata=True)
    assert list(df.columns) == METADATA_COLUMNS
    assert len(df) == 3
    df = datafolio.find("dataset=minnie*", metadata=True)
    assert sorted(df["alias"].fillna("-")) == ["-", "alpha"]
    assert set(df["value"]) == {"minnie65"}


def test_restrict_folios_by_alias_or_path(folios, tmp_path):
    assert set(datafolio.find(folios="beta")["name"]) == {"cells", "synapses"}
    df = datafolio.find(folios=[str(tmp_path / "a"), "beta"], item_type="table")
    # a path that has an alias is reported under that alias
    assert _pairs(df) == [
        ("alpha", "cells"),
        ("alpha", "tbl_0042"),
        ("beta", "cells"),
        ("beta", "synapses"),
    ]


def test_aliases_only(folios):
    assert "soma_table" not in set(datafolio.find(aliases_only=True)["name"])


def test_archived_items(folios):
    folios["b"].archive("synapses")
    assert "synapses" not in set(datafolio.find(folios="beta")["name"])
    assert "synapses" in set(
        datafolio.find(folios="beta", include_archived=True)["name"]
    )


def test_missing_folio_warns_and_is_skipped(folios, tmp_path):
    datafolio.set_alias("ghost", tmp_path / "ghost")
    with pytest.warns(UserWarning, match="ghost"):
        df = datafolio.find("cells")
    assert len(df) == 2
    assert not (tmp_path / "ghost").exists()  # never created


def test_non_folio_directory_warns(tmp_path):
    (tmp_path / "plain").mkdir()
    datafolio.set_alias("plain", tmp_path / "plain")
    with pytest.warns(UserWarning, match="not a DataFolio bundle"):
        assert datafolio.find().empty


def test_local_only_skips_cloud(folios):
    datafolio.set_alias("remote", "gs://no-such-bucket-datafolio-test/x")
    df = datafolio.find("cells", local_only=True)  # no warning, no network
    assert len(df) == 2


def test_empty_registry():
    df = datafolio.find("*")
    assert df.empty
    assert list(df.columns) == ITEM_COLUMNS


def test_find_does_not_reorder_recents(folios, tmp_path):
    before = FolioRegistry().recent()
    datafolio.find("*")
    assert FolioRegistry().recent() == before


def test_descriptions_off_by_default(folios):
    assert datafolio.find("synapse").empty


def test_descriptions_substring_case_insensitive(folios):
    df = datafolio.find("synapse", descriptions=True)
    assert set(df["name"]) == {"tbl_0042"}
    assert "interesting" in df.iloc[0]["description"]


def test_descriptions_or_name(folios):
    names = set(datafolio.find("syn*", descriptions=True)["name"])
    assert names == {"synapses", "tbl_0042"}


def test_descriptions_regex(folios):
    df = datafolio.find(r"thought.*interest", regex=True, descriptions=True)
    assert set(df["name"]) == {"tbl_0042"}


def test_descriptions_with_type_filter(folios):
    assert datafolio.find("synapse", descriptions=True, item_type="array").empty


def test_descriptions_case_sensitive(folios):
    assert datafolio.find("synapse", descriptions=True, case_sensitive=True).empty
    assert not datafolio.find("Synapse", descriptions=True, case_sensitive=True).empty
