"""Handler for scikit-learn models."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from datafolio.base.handler import BaseHandler

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class SklearnHandler(BaseHandler):
    """Handler for sklearn-style models stored in bundle.

    This handler manages models with sklearn-style API:
    - Serializes models using joblib to .pkl format
    - Stores metadata (model type, sklearn version if available)
    - Handles lineage tracking
    - Deserializes back to model on read
    - Works with sklearn models AND any joblib-serializable objects (e.g., custom models)

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = SklearnHandler()
        >>> register_handler(handler)
        >>>
        >>> # Handler is used automatically by DataFolio
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> folio.add_sklearn('clf', model)
        >>>
        >>> # Also works with custom models
        >>> class CustomModel:
        ...     def __init__(self, param=1):
        ...         self.param = param
        >>> custom_model = CustomModel(param=42)
        >>> folio.add_model('custom', custom_model)
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "model"

    def can_handle(self, data: Any) -> bool:
        """Check if data is a scikit-learn compatible model.

        Checks for:
        1. sklearn.base.BaseEstimator (sklearn models)
        2. Common ML framework classes (XGBoost, LightGBM, CatBoost)
        3. sklearn-compatible interface (has fit + predict/transform methods)

        Args:
            data: Data to check

        Returns:
            True if data is an sklearn-compatible model
        """
        # Don't handle built-in types (str, int, float, list, dict, etc.)
        if isinstance(
            data,
            (
                str,
                int,
                float,
                bool,
                list,
                dict,
                tuple,
                set,
                frozenset,
                bytes,
                bytearray,
                type(None),
            ),
        ):
            return False

        # Don't handle numpy arrays or pandas DataFrames (have their own handlers)
        try:
            import numpy as np

            if isinstance(data, np.ndarray):
                return False
        except ImportError:
            pass

        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                return False
        except ImportError:
            pass

        # Check for sklearn models
        try:
            from sklearn.base import BaseEstimator

            if isinstance(data, BaseEstimator):
                return True
        except ImportError:
            pass

        # Check for common ML framework classes
        # XGBoost
        try:
            import xgboost as xgb

            if isinstance(data, (xgb.XGBModel, xgb.Booster)):
                return True
        except (ImportError, AttributeError):
            pass

        # LightGBM
        try:
            import lightgbm as lgb

            if isinstance(data, (lgb.LGBMModel, lgb.Booster)):
                return True
        except (ImportError, AttributeError):
            pass

        # CatBoost
        try:
            import catboost as cb

            if isinstance(data, cb.CatBoost):
                return True
        except (ImportError, AttributeError):
            pass

        # Check for sklearn-compatible interface
        # Must have 'fit' method AND at least one of: predict, transform, predict_proba
        has_fit = callable(getattr(data, "fit", None))
        has_predict = callable(getattr(data, "predict", None))
        has_transform = callable(getattr(data, "transform", None))
        has_predict_proba = callable(getattr(data, "predict_proba", None))

        if has_fit and (has_predict or has_transform or has_predict_proba):
            return True

        return False

    def add(
        self,
        folio: "DataFolio",
        name: str,
        model: Any,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add joblib-serializable model to folio.

        Writes the model to storage and builds complete metadata including:
        - Basic info: filename, model type
        - Model info: sklearn version (if sklearn is installed and model is sklearn estimator)
        - Lineage: inputs and description
        - Timestamps: creation time

        Args:
            folio: DataFolio instance
            name: Item name
            model: Model to store (sklearn estimator or any joblib-serializable object)
            description: Optional description
            inputs: Optional lineage inputs
            **kwargs: Additional arguments (currently unused)

        Returns:
            Complete metadata dict for this model

        Raises:
            ImportError: If joblib is not installed
        """
        # Try to get sklearn version if available (optional)
        sklearn_version = None
        try:
            import sklearn

            sklearn_version = sklearn.__version__
        except ImportError:
            pass

        # Build filename
        filename = f"{name}.joblib"
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        # Write model to storage
        folio._storage.write_joblib(filepath, model)

        # Build comprehensive metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "model_type": type(model).__name__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add sklearn version if available
        if sklearn_version:
            metadata["sklearn_version"] = sklearn_version

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    def get(self, folio: "DataFolio", name: str, **kwargs) -> Any:
        """Load joblib-serialized model from folio.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Additional arguments (currently unused)

        Returns:
            Deserialized model object

        Raises:
            KeyError: If item doesn't exist
            ImportError: If joblib is not installed
        """
        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )
        return folio._storage.read_joblib(filepath)
