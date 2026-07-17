"""Regression tests for the audit fixes.

Organized by finding number. Written test-first per AGENTS.md.
"""

import numpy as np
import pandas as pd
import pytest

from datafolio import DataFolio

pl = pytest.importorskip("polars")


# =============================================================================
# Finding 1: generic add_data/get_data must enforce the same invariants as the
# type-specific public methods (dup-name, overwrite, snapshot COW, guards).
# =============================================================================


class TestGenericApiInvariants:
    def test_add_data_duplicate_name_raises(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_data("t", pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(ValueError, match="already exists"):
            folio.add_data("t", pd.DataFrame({"a": [9]}))

    def test_add_data_overwrite_allows_replace(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_data("t", pd.DataFrame({"a": [1, 2, 3]}))
        folio.add_data("t", pd.DataFrame({"a": [9, 9]}), overwrite=True)
        assert folio.get_table("t")["a"].to_list() == [9, 9]

    def test_add_data_snapshot_copy_on_write(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_data("t", pd.DataFrame({"a": [1, 2, 3]}))
        folio.create_snapshot("snap")
        folio.add_data("t", pd.DataFrame({"a": [9]}), overwrite=True)
        # snapshot must still see the original data
        assert folio.snapshots["snap"].get_table("t")["a"].to_list() == [1, 2, 3]
        assert folio.get_table("t")["a"].to_list() == [9]

    def test_add_data_numpy_json_timestamp_delegate(self, tmp_path):
        folio = DataFolio(tmp_path / "b")
        folio.add_data("arr", np.array([1, 2, 3]))
        folio.add_data("cfg", {"lr": 0.01})
        folio.add_data("scalar", 0.95)
        assert folio.get_data("arr").tolist() == [1, 2, 3]
        assert folio.get_data("cfg") == {"lr": 0.01}
        assert folio.get_data("scalar") == 0.95

    def test_get_data_polars_only_raises_friendly(self, tmp_path):
        # Build a partitioned dataset and reference it (auto polars_only).
        d = tmp_path / "hive"
        pl.DataFrame({"g": ["a", "b"], "x": [1, 2]}).write_parquet(d, partition_by="g")
        folio = DataFolio(tmp_path / "b")
        folio.reference_table("big", path=d)
        with pytest.raises(ValueError, match="polars-only"):
            folio.get_data("big")

    def test_get_data_respects_eager_guard(self, tmp_path):
        df = pd.DataFrame({"a": range(1000)})
        ext = tmp_path / "e.parquet"
        df.to_parquet(ext, index=False)
        folio = DataFolio(tmp_path / "b", max_eager_bytes=10)
        folio.reference_table("ref", ext)
        with pytest.raises(ValueError, match="eager-load limit"):
            folio.get_data("ref")
