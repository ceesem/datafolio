"""Data type handlers for DataFolio.

This module contains all built-in handlers and automatically registers them
with the HandlerRegistry on import.

Available handlers:
- PandasHandler: Handles pandas DataFrames (included_table)
- ReferenceTableHandler: Handles external table references (referenced_table)
- NumpyHandler: Handles numpy arrays (numpy_array)
- JsonHandler: Handles JSON-serializable data (json_data)
- TimestampHandler: Handles datetime/Unix timestamps (timestamp)
- ArtifactHandler: Handles arbitrary file artifacts (artifact)
- SklearnHandler: Handles scikit-learn models (model)

Usage:
    Handlers are automatically registered when this module is imported.
    DataFolio uses the registry to find appropriate handlers for data types.

Examples:
    >>> from datafolio.handlers import PandasHandler, NumpyHandler
    >>> from datafolio.base.registry import get_registry
    >>> registry = get_registry()
    >>> registry.get('included_table')
    <PandasHandler object>
"""

from datafolio.base.registry import register_handler
from datafolio.handlers.arrays import NumpyHandler
from datafolio.handlers.artifacts import ArtifactHandler
from datafolio.handlers.json_data import JsonHandler
from datafolio.handlers.sklearn_models import SklearnHandler
from datafolio.handlers.tables import PandasHandler, ReferenceTableHandler
from datafolio.handlers.timestamps import TimestampHandler

# Auto-register all built-in handlers
# Order matters: more specific handlers first, generic handlers last
# This ensures correct auto-detection in add_data() since the first matching handler is used
_pandas_handler = PandasHandler()
_numpy_handler = NumpyHandler()
_sklearn_handler = SklearnHandler()
_artifact_handler = ArtifactHandler()
_reference_handler = ReferenceTableHandler()
_json_handler = JsonHandler()
_timestamp_handler = TimestampHandler()

# Registration order: Most specific → Most generic
register_handler(_pandas_handler)  # 1. pd.DataFrame (very specific)
register_handler(_numpy_handler)  # 2. np.ndarray (very specific)
register_handler(_sklearn_handler)  # 3. ML models with sklearn API (specific)
register_handler(_artifact_handler)  # 4. Existing file paths (specific)
register_handler(_timestamp_handler)  # 5. datetime objects (specific)
register_handler(_reference_handler)  # 6. Never auto-detects (position neutral)
register_handler(_json_handler)  # 7. dict/list (generic)

# Export handlers for direct use if needed
__all__ = [
    "PandasHandler",
    "ReferenceTableHandler",
    "NumpyHandler",
    "JsonHandler",
    "TimestampHandler",
    "ArtifactHandler",
    "SklearnHandler",
]
