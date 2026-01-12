"""Tests for sklearn handler with multiple serialization formats."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datafolio import DataFolio


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Custom transformer for testing skops portability."""

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_bound_ = np.percentile(X, self.lower, axis=0)
        self.upper_bound_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bound_, self.upper_bound_)


def test_skops_custom_transformer(tmp_path):
    """Test skops serialization with custom transformer."""
    # Create pipeline with custom transformer
    pipeline = Pipeline(
        [
            ("clipper", PercentileClipper(lower=5, upper=95)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )

    # Fit on dummy data
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    pipeline.fit(X, y)

    # Save with skops
    folio = DataFolio(tmp_path / "test_folio")
    folio.add_sklearn("custom_pipeline", pipeline, custom=True)

    # Load and verify
    folio2 = DataFolio(tmp_path / "test_folio")
    loaded_pipeline = folio2.get_sklearn("custom_pipeline")

    # Verify predictions match
    np.testing.assert_array_equal(pipeline.predict(X), loaded_pipeline.predict(X))

    # Verify custom transformer parameters preserved
    assert loaded_pipeline.named_steps["clipper"].lower == 5
    assert loaded_pipeline.named_steps["clipper"].upper == 95


def test_skops_metadata_stored(tmp_path):
    """Test that skops format is stored in metadata."""
    pipeline = Pipeline([("scaler", StandardScaler())])

    folio = DataFolio(tmp_path / "test_folio")
    folio.add_sklearn("pipeline", pipeline, custom=True)

    # Check metadata
    assert folio._items["pipeline"]["serialization_format"] == "skops"
    assert folio._items["pipeline"]["filename"].endswith(".skops")


def test_joblib_still_default(tmp_path):
    """Test that joblib remains the default format."""
    pipeline = Pipeline([("scaler", StandardScaler())])

    folio = DataFolio(tmp_path / "test_folio")
    folio.add_sklearn("pipeline", pipeline)  # No custom flag specified

    # Should default to joblib
    assert folio._items["pipeline"]["serialization_format"] == "joblib"
    assert folio._items["pipeline"]["filename"].endswith(".joblib")


def test_legacy_models_without_format_field(tmp_path):
    """Test backward compatibility for old models without serialization_format."""
    pipeline = Pipeline([("scaler", StandardScaler())])

    folio = DataFolio(tmp_path / "test_folio")
    folio.add_sklearn("pipeline", pipeline)

    # Manually remove serialization_format to simulate legacy model
    del folio._items["pipeline"]["serialization_format"]
    folio._save_items()  # Manually save to persist the change

    # Should still load (defaults to joblib)
    folio2 = DataFolio(tmp_path / "test_folio")
    loaded = folio2.get_sklearn("pipeline")
    assert loaded is not None


def test_add_model_with_custom_flag(tmp_path):
    """Test that custom parameter works through add_model()."""
    pipeline = Pipeline([("scaler", StandardScaler())])

    folio = DataFolio(tmp_path / "test_folio")
    folio.add_model("pipeline", pipeline, custom=True)

    assert folio._items["pipeline"]["serialization_format"] == "skops"


def test_both_formats_in_same_folio(tmp_path):
    """Test mixing joblib and skops models in one folio."""
    p1 = Pipeline([("scaler", StandardScaler())])
    p2 = Pipeline([("clipper", PercentileClipper())])

    X = np.random.randn(50, 3)
    p2.fit(X)

    folio = DataFolio(tmp_path / "test_folio")
    folio.add_sklearn("joblib_model", p1)  # default: custom=False
    folio.add_sklearn("skops_model", p2, custom=True)

    # Load and verify both work
    folio2 = DataFolio(tmp_path / "test_folio")
    m1 = folio2.get_sklearn("joblib_model")
    m2 = folio2.get_sklearn("skops_model")

    assert m1 is not None
    assert m2 is not None


def test_skops_with_numpy_functions(tmp_path):
    """Test that FunctionTransformer with numpy functions works with skops."""
    from sklearn.preprocessing import FunctionTransformer

    # Create pipeline with numpy function (should work with skops)
    pipeline = Pipeline(
        [
            ("sqrt", FunctionTransformer(np.sqrt)),
            ("scaler", StandardScaler()),
        ]
    )

    X = np.random.rand(50, 3) + 1  # Add 1 to ensure positive values for sqrt
    pipeline.fit(X)

    folio = DataFolio(tmp_path / "test_folio")
    folio.add_sklearn("numpy_pipeline", pipeline, custom=True)

    # Load and verify
    folio2 = DataFolio(tmp_path / "test_folio")
    loaded = folio2.get_sklearn("numpy_pipeline")

    # Verify transformation works
    result = loaded.transform(X[:10])
    expected = pipeline.transform(X[:10])
    np.testing.assert_array_almost_equal(result, expected)


def test_skops_roundtrip_preserves_fitted_state(tmp_path):
    """Test that fitted parameters are preserved with skops."""
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X = np.random.randn(100, 3)
    scaler.fit(X)

    # Store fitted scaler
    folio = DataFolio(tmp_path / "test_folio")
    folio.add_sklearn("scaler", scaler, custom=True)

    # Load and verify fitted parameters
    folio2 = DataFolio(tmp_path / "test_folio")
    loaded_scaler = folio2.get_sklearn("scaler")

    # Verify fitted parameters match
    np.testing.assert_array_almost_equal(scaler.center_, loaded_scaler.center_)
    np.testing.assert_array_almost_equal(scaler.scale_, loaded_scaler.scale_)


def test_get_model_detects_format(tmp_path):
    """Test that get_model() works with both formats."""
    p1 = Pipeline([("scaler", StandardScaler())])
    p2 = Pipeline([("clipper", PercentileClipper())])
    X = np.random.randn(50, 3)
    p2.fit(X)

    folio = DataFolio(tmp_path / "test_folio")
    folio.add_sklearn("joblib_model", p1)
    folio.add_sklearn("skops_model", p2, custom=True)

    # Load using get_model() (generic method)
    folio2 = DataFolio(tmp_path / "test_folio")
    m1 = folio2.get_model("joblib_model")
    m2 = folio2.get_model("skops_model")

    assert m1 is not None
    assert m2 is not None
