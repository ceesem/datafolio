"""Demonstration of how handlers distinguish between data types.

Shows the two mechanisms for type detection:
1. Auto-detection via can_handle() - for add_data()
2. Explicit type via item_type - for get_data()
"""

import pandas as pd
import numpy as np

from datafolio.base import register_handler, detect_handler
from datafolio.handlers.tables import PandasHandler, ReferenceTableHandler


class NumpyHandler:
    """Mock Numpy handler for demonstration."""

    @property
    def item_type(self):
        return "numpy_array"

    def can_handle(self, data):
        return isinstance(data, np.ndarray)

    def get_storage_subdir(self):
        return "artifacts"


class JsonHandler:
    """Mock JSON handler for demonstration."""

    @property
    def item_type(self):
        return "json_data"

    def can_handle(self, data):
        # JSON handles: dict, list, int, float, str, bool, None
        return isinstance(data, (dict, list, int, float, str, bool, type(None)))

    def get_storage_subdir(self):
        return "artifacts"


class ArtifactHandler:
    """Mock Artifact handler for demonstration."""

    @property
    def item_type(self):
        return "artifact"

    def can_handle(self, data):
        # Artifacts are added via explicit method (file paths)
        # Never auto-detected
        return False

    def get_storage_subdir(self):
        return "artifacts"


def demo_auto_detection():
    """Show how handlers auto-detect data types."""

    print("=" * 70)
    print("AUTO-DETECTION: How add_data() knows what handler to use")
    print("=" * 70)

    # Register handlers
    handlers = [
        PandasHandler(),
        NumpyHandler(),
        JsonHandler(),
        ArtifactHandler(),
    ]

    for handler in handlers:
        register_handler(handler)

    # Test different data types
    test_cases = [
        ("pandas DataFrame", pd.DataFrame({'a': [1, 2, 3]})),
        ("numpy array", np.array([1, 2, 3])),
        ("dict", {'key': 'value'}),
        ("list", [1, 2, 3]),
        ("int", 42),
        ("float", 3.14),
        ("string", "hello"),
        ("bool", True),
        ("None", None),
    ]

    print("\nTesting each data type:\n")

    for name, data in test_cases:
        detected = detect_handler(data)
        if detected:
            print(f"✓ {name:20} → {detected.item_type:20} (via {detected.__class__.__name__})")
        else:
            print(f"✗ {name:20} → No handler found")

    print("\n" + "-" * 70)
    print("How it works:")
    print("-" * 70)
    print("""
When you call:
    folio.add_data('my_data', data)

DataFolio does:
    1. handler = detect_handler(data)
       → Tries each handler's can_handle(data) until one returns True

    2. metadata = handler.add(folio, 'my_data', data)
       → Handler writes data and returns metadata

    3. folio._items['my_data'] = metadata
       → Metadata includes 'item_type' for retrieval
""")


def demo_explicit_type():
    """Show how handlers are selected when reading data."""

    print("\n" + "=" * 70)
    print("EXPLICIT TYPE: How get_data() knows what handler to use")
    print("=" * 70)

    print("""
When data is stored, metadata includes 'item_type':

    folio._items['my_data'] = {
        'name': 'my_data',
        'item_type': 'included_table',  ← This tells us which handler
        'filename': 'my_data.parquet',
        'num_rows': 100,
        ...
    }

When you call:
    folio.get_data('my_data')

DataFolio does:
    1. item = folio._items['my_data']
    2. item_type = item['item_type']  → 'included_table'
    3. handler = get_handler('included_table')  → PandasHandler
    4. data = handler.get(folio, 'my_data')
    5. return data  → Returns DataFrame

No guessing needed - the item_type is stored in metadata!
""")


def demo_priority_and_conflicts():
    """Show how detection order matters."""

    print("\n" + "=" * 70)
    print("DETECTION ORDER: What happens with overlapping types?")
    print("=" * 70)

    print("""
Q: What if multiple handlers can handle the same data?
   For example, a dict could be JSON data OR model params OR config.

A: Detection order matters! First match wins.

Example hierarchy (from specific to general):
    1. PandasHandler.can_handle(df)     → isinstance(df, pd.DataFrame)
    2. NumpyHandler.can_handle(arr)     → isinstance(arr, np.ndarray)
    3. JsonHandler.can_handle(obj)      → isinstance(obj, (dict, list, ...))

For a dict:
    • PandasHandler.can_handle(dict) → False (not a DataFrame)
    • NumpyHandler.can_handle(dict)  → False (not an array)
    • JsonHandler.can_handle(dict)   → True ✓ (is a dict)

Result: Dict → JsonHandler

This works because handlers are specific:
    • Tables only match DataFrames
    • Arrays only match ndarrays
    • JSON matches everything else (dict, list, primitives)
""")


def demo_non_detectable_types():
    """Show types that require explicit methods."""

    print("\n" + "=" * 70)
    print("NON-DETECTABLE: Some types need explicit methods")
    print("=" * 70)

    print("""
Some data types CANNOT be auto-detected:

1. External References
   • Can't detect "this is a reference, not actual data"
   • Must use: folio.reference_table('name', path='s3://...')

2. Artifacts (files)
   • Can't detect "this is a file path to copy"
   • Must use: folio.add_artifact('name', source_path='/path/to/file.png')

3. PyTorch Models
   • Could auto-detect isinstance(model, nn.Module)
   • But need extra params (init_args, save_class)
   • Must use: folio.add_pytorch('name', model, init_args={...})

4. Timestamps
   • datetime objects could be auto-detected
   • But treated specially for metadata
   • Must use: folio.add_timestamp('name', datetime.now(UTC))

These handlers set can_handle() → False to prevent auto-detection.
""")

    # Show the actual implementation
    print("\nActual Implementation:")
    print("-" * 70)

    ref_handler = ReferenceTableHandler()
    artifact_handler = ArtifactHandler()

    df = pd.DataFrame({'a': [1, 2, 3]})

    print(f"ReferenceTableHandler.can_handle(DataFrame): {ref_handler.can_handle(df)}")
    print(f"ArtifactHandler.can_handle(DataFrame):      {artifact_handler.can_handle(df)}")
    print(f"\n→ Both return False, so they're never auto-selected")
    print(f"→ Must use explicit methods: reference_table(), add_artifact()")


def demo_complete_flow():
    """Show the complete flow from add to get."""

    print("\n" + "=" * 70)
    print("COMPLETE FLOW: From add_data() to get_data()")
    print("=" * 70)

    print("""
Step-by-step example:

1. USER CODE
   --------
   folio.add_data('results', pd.DataFrame({'a': [1, 2, 3]}))

2. AUTO-DETECTION
   --------------
   detect_handler(DataFrame)
     → Try PandasHandler.can_handle(df)   → True ✓
     → Return PandasHandler

3. HANDLER ADDS DATA
   -----------------
   PandasHandler.add(folio, 'results', df)
     → Writes: {bundle}/tables/results.parquet
     → Returns: {
           'name': 'results',
           'item_type': 'included_table',  ← Stored for later
           'filename': 'results.parquet',
           'num_rows': 3,
           'num_cols': 1,
           'columns': ['a'],
           ...
       }

4. DATAFOLIO STORES METADATA
   -------------------------
   folio._items['results'] = metadata
   folio._save_items()  # Writes items.json

5. USER RETRIEVES DATA
   -------------------
   folio.get_data('results')

6. TYPE LOOKUP
   -----------
   item = folio._items['results']
   item_type = item['item_type']  → 'included_table'

7. GET HANDLER
   -----------
   handler = get_handler('included_table')  → PandasHandler

8. HANDLER READS DATA
   ------------------
   PandasHandler.get(folio, 'results')
     → Reads: {bundle}/tables/results.parquet
     → Returns: DataFrame

9. RETURN TO USER
   --------------
   Returns DataFrame to user

Key insight: Detection happens ONCE during add_data().
             Retrieval uses stored 'item_type' - no guessing!
""")


if __name__ == "__main__":
    demo_auto_detection()
    demo_explicit_type()
    demo_priority_and_conflicts()
    demo_non_detectable_types()
    demo_complete_flow()
