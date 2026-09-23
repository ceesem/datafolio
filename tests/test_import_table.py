"""Tests for DataFolio.import_table and StorageBackend.import_table_file."""

import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datafolio import DataFolio
from datafolio.storage import StorageBackend
from datafolio.storage import backend as backend_module

N = 5000


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "i": np.arange(N, dtype="int64"),
            "x": rng.random(N),
            "s": [f"row{i}" for i in range(N)],
        }
    )


@pytest.fixture
def files(tmp_path, df):
    paths = {
        "parquet": tmp_path / "t.parquet",
        "csv": tmp_path / "t.csv",
        "feather": tmp_path / "t.feather",
    }
    df.to_parquet(paths["parquet"], index=False)
    df.to_csv(paths["csv"], index=False)
    df.to_feather(paths["feather"])
    return paths


@pytest.fixture
def folio(tmp_path):
    return DataFolio(tmp_path / "folio")


@pytest.mark.parametrize("fmt", ["parquet", "csv", "feather"])
def test_round_trip(folio, files, df, fmt):
    folio.import_table(None, files[fmt])
    pd.testing.assert_frame_equal(folio.get("t"), df, check_dtype=False)


@pytest.mark.parametrize("fmt", ["parquet", "csv", "feather"])
def test_manifest_matches_add(folio, files, df, fmt):
    folio.add("from_df", df)
    folio.import_table("imported", files[fmt])
    a, b = folio.item_info("from_df"), folio.item_info("imported")
    assert b["item_type"] == "included_table"
    assert b["table_format"] == "parquet"
    assert b["filename"].endswith(".parquet")
    for key in ("num_rows", "num_cols", "columns"):
        assert a[key] == b[key]
    assert b["checksum"] and b["size_bytes"] > 0


def test_parquet_is_copied_byte_for_byte(folio, files):
    folio.import_table("t", files["parquet"])
    src = hashlib.md5(files["parquet"].read_bytes()).hexdigest()
    assert folio.item_info("t")["checksum"] == src


def test_arrow_extension(folio, tmp_path, df):
    path = tmp_path / "t.arrow"
    df.to_feather(path)
    folio.import_table("t", path)
    assert folio.item_info("t")["num_rows"] == N


def test_explicit_format_overrides_extension(folio, tmp_path, df):
    path = tmp_path / "data.txt"
    df.to_csv(path, index=False)
    folio.import_table("t", path, table_format="csv")
    assert folio.item_info("t")["num_rows"] == N


def test_unknown_extension(folio, tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Cannot infer a table format"):
        folio.import_table("t", path)
    with pytest.raises(ValueError, match="Unsupported table_format"):
        folio.import_table("t", path, table_format="xlsx")


def test_missing_file(folio, tmp_path):
    with pytest.raises(FileNotFoundError):
        folio.import_table("t", tmp_path / "nope.csv")


def test_invalid_derived_name(folio, tmp_path, df):
    path = tmp_path / "bad name!.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="pass a name explicitly"):
        folio.import_table(None, path)


def test_overwrite_rule(folio, files):
    folio.import_table("t", files["csv"])
    with pytest.raises(ValueError, match="already exists"):
        folio.import_table("t", files["csv"])
    folio.import_table("t", files["feather"], overwrite=True, description="v2")
    assert folio.item_info("t")["description"] == "v2"


def test_read_only(tmp_path, files):
    DataFolio(tmp_path / "folio")
    ro = DataFolio(tmp_path / "folio", read_only=True)
    with pytest.raises(Exception, match="read-only"):
        ro.import_table("t", files["csv"])


def test_corrupt_parquet_writes_nothing(folio, tmp_path):
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"definitely not parquet")
    with pytest.raises(Exception):
        folio.import_table("bad", bad)
    assert "bad" not in folio.list_contents()["included_tables"]
    assert not list((folio._bundle_path / "tables").glob("bad*"))


def test_csv_type_mismatch_is_explained(folio, tmp_path):
    path = tmp_path / "mixed.csv"
    rows = ["a"] + [str(i) for i in range(20000)] + ["not-a-number"]
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="--as-file"):
        folio.import_table("mixed", path, block_size=4096)
    assert "mixed" not in folio.list_contents()["included_tables"]


def test_csv_streams_in_blocks(tmp_path, df, files, monkeypatch):
    """Small blocks + small row-group target → many row groups, same data."""
    monkeypatch.setattr(backend_module, "ROW_GROUP_BYTES", 16 * 1024)
    out = str(tmp_path / "out.parquet")
    schema, num_rows = StorageBackend().import_table_file(
        out, files["csv"], "csv", block_size=16 * 1024
    )
    assert num_rows == N
    assert pq.ParquetFile(out).num_row_groups > 1
    pd.testing.assert_frame_equal(pd.read_parquet(out), df, check_dtype=False)


def test_small_batches_are_coalesced(tmp_path, files):
    out = str(tmp_path / "out.parquet")
    StorageBackend().import_table_file(out, files["csv"], "csv", block_size=4096)
    assert pq.ParquetFile(out).num_row_groups == 1


def test_ipc_stream_format(tmp_path, df):
    path = tmp_path / "stream.arrow"
    table = pa.Table.from_pandas(df, preserve_index=False)
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    out = str(tmp_path / "out.parquet")
    _, num_rows = StorageBackend().import_table_file(out, path, "arrow")
    assert num_rows == N


def test_unsupported_storage_format(tmp_path):
    with pytest.raises(ValueError, match="Cannot import table format"):
        StorageBackend().import_table_file(str(tmp_path / "o.parquet"), "x", "xlsx")
