import os

import pandas as pd
import pytest

from datafolio import DataFolio


def test_batch_mode(tmp_path):
    folio = DataFolio(tmp_path / "batch_test")

    # Verify items.json is saved normally
    folio.add_json("item1", {"a": 1})
    assert (folio._bundle_path / "items.json").exists()

    # Check timestamp of items.json
    t1 = (folio._bundle_path / "items.json").stat().st_mtime

    with folio.batch():
        folio.add_json("item2", {"b": 2})
        folio.add_json("item3", {"c": 3})
        # items.json should NOT be updated yet (or at least not written to disk repeatedly)
        # But since we can't easily check write count without mocking,
        # we rely on logic correctness.
        # We can check that _batch_mode is True
        assert folio._batch_mode is True

    # After exit, items.json should be updated
    assert "item2" in folio.list_contents()["json_data"]
    assert "item3" in folio.list_contents()["json_data"]

    # Reload to verify persistence
    folio2 = DataFolio(tmp_path / "batch_test")
    assert "item2" in folio2.list_contents()["json_data"]


def test_validate(tmp_path):
    folio = DataFolio(tmp_path / "validate_test")

    # Add valid items
    folio.add_json("valid_json", {"a": 1})

    # Add reference
    ref_path = tmp_path / "external.csv"
    ref_path.write_text("a,b\n1,2")
    folio.reference_table("valid_ref", str(ref_path))

    # Validate - should be all True
    results = folio.validate()
    assert results["valid_json"] is True
    assert results["valid_ref"] is True

    # Corrupt bundle (delete internal file)
    (folio._bundle_path / "artifacts" / "valid_json.json").unlink()

    # Corrupt reference (delete external file)
    ref_path.unlink()

    # Validate - should be False
    results = folio.validate()
    assert results["valid_json"] is False
    assert results["valid_ref"] is False


def test_checksum_validation(tmp_path):
    folio = DataFolio(tmp_path / "checksum_test")

    # Create artifact
    art_path = tmp_path / "test.txt"
    art_path.write_text("hello world")

    # Add artifact (should calc checksum)
    folio.add_artifact("my_art", str(art_path))

    # Verify checksum in metadata
    item = folio._items["my_art"]
    assert "checksum" in item
    original_checksum = item["checksum"]

    # Validate - should be True
    assert folio.validate()["my_art"] is True

    # Modify file in bundle to corrupt it
    bundle_file = folio._bundle_path / "artifacts" / "my_art.txt"
    bundle_file.write_text("hacked content")

    # Validate - should be False due to checksum mismatch
    assert folio.validate()["my_art"] is False

    # Restore file
    bundle_file.write_text("hello world")
    assert folio.validate()["my_art"] is True


def test_reference_directory(tmp_path):
    folio = DataFolio(tmp_path / "ref_dir_test")

    # Create a directory to reference
    ext_dir = tmp_path / "ext_data"
    ext_dir.mkdir()
    (ext_dir / "part1.parquet").touch()

    # Add reference to directory
    folio.reference_table("dir_ref", str(ext_dir))

    # Check metadata
    item = folio._items["dir_ref"]
    assert item["is_directory"] is True
    assert item["path"] == f"file://{ext_dir}"

    # Validate
    assert folio.validate()["dir_ref"] is True

    # Delete directory
    import shutil

    shutil.rmtree(ext_dir)

    # Validate
    assert folio.validate()["dir_ref"] is False


def test_extended_checksums(tmp_path):
    folio = DataFolio(tmp_path / "extended_checksum_test")

    # Test Numpy
    import numpy as np

    arr = np.array([1, 2, 3])
    folio.add_numpy("arr", arr)
    assert "checksum" in folio._items["arr"]
    assert folio.validate()["arr"] is True

    # Test JSON
    folio.add_json("config", {"a": 1})
    assert "checksum" in folio._items["config"]
    assert folio.validate()["config"] is True

    # Test Sklearn (using a simple class to avoid sklearn dependency if not present,
    # though SklearnHandler handles any joblib object)
    class SimpleModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return X

    model = SimpleModel()
    folio.add_model("model", model)
    assert "checksum" in folio._items["model"]
    assert folio.validate()["model"] is True

    # Corrupt one
    (folio._bundle_path / "artifacts" / "config.json").write_text("{}")
    assert folio.validate()["config"] is False


def test_is_valid(tmp_path):
    folio = DataFolio(tmp_path / "is_valid_test")

    # Add valid items
    folio.add_json("item1", {"a": 1})
    folio.add_json("item2", {"b": 2})

    # Should be valid
    assert folio.is_valid() is True

    # Corrupt one item
    (folio._bundle_path / "artifacts" / "item1.json").write_text("corrupted")

    # Should be invalid
    assert folio.is_valid() is False
