"""Demonstration of the handler architecture.

This script shows how the new handler-based architecture works:
1. Handlers are registered globally
2. DataFolio delegates to handlers
3. Each data type has its own isolated handler
"""

import pandas as pd
import tempfile
from pathlib import Path

# Import the new architecture components
from datafolio.base import BaseHandler, register_handler, get_handler, detect_handler
from datafolio.handlers.tables import PandasHandler, ReferenceTableHandler
from datafolio.storage import StorageBackend

def demo_handler_system():
    """Demonstrate the handler system works independently."""

    print("=" * 70)
    print("HANDLER ARCHITECTURE DEMONSTRATION")
    print("=" * 70)

    # 1. Register handlers
    print("\n1. Registering Handlers")
    print("-" * 70)
    pandas_handler = PandasHandler()
    ref_handler = ReferenceTableHandler()

    register_handler(pandas_handler)
    register_handler(ref_handler)
    print(f"✓ Registered: {pandas_handler.item_type}")
    print(f"✓ Registered: {ref_handler.item_type}")

    # 2. Auto-detection
    print("\n2. Handler Auto-Detection")
    print("-" * 70)
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    detected = detect_handler(df)
    print(f"Data type: pandas.DataFrame")
    print(f"Detected handler: {detected.__class__.__name__}")
    print(f"Handler item_type: {detected.item_type}")

    # 3. Get handler by type
    print("\n3. Get Handler by Type")
    print("-" * 70)
    handler = get_handler('included_table')
    print(f"Requested: 'included_table'")
    print(f"Got: {handler.__class__.__name__}")

    # 4. Handler interface
    print("\n4. Handler Interface")
    print("-" * 70)
    print(f"Handler: {handler.__class__.__name__}")
    print(f"  - item_type: {handler.item_type}")
    print(f"  - can_handle(DataFrame): {handler.can_handle(df)}")
    print(f"  - can_handle(list): {handler.can_handle([1, 2, 3])}")
    print(f"  - storage_subdir: {handler.get_storage_subdir()}")

    print("\n" + "=" * 70)
    print("All components working correctly!")
    print("=" * 70)


def demo_with_datafolio():
    """Show how handlers integrate with DataFolio (conceptually)."""

    print("\n\nHow This Works with DataFolio:")
    print("=" * 70)

    code_example = """
    # User code (unchanged):
    from datafolio import DataFolio
    import pandas as pd

    folio = DataFolio('my-experiment')
    df = pd.DataFrame({'a': [1, 2, 3]})

    # This call now uses the handler system internally:
    folio.add_table('data', df)

    # Behind the scenes:
    # 1. DataFolio.add_table() gets the 'included_table' handler
    # 2. Calls handler.add(folio, 'data', df, ...)
    # 3. Handler writes data via folio._storage
    # 4. Handler returns metadata dict
    # 5. DataFolio stores metadata in folio._items

    # Reading works similarly:
    df_loaded = folio.get_table('data')

    # Behind the scenes:
    # 1. DataFolio.get_table() gets the 'included_table' handler
    # 2. Calls handler.get(folio, 'data')
    # 3. Handler reads from folio._storage
    # 4. Returns DataFrame to user
    """

    print(code_example)

    print("\nKey Benefits:")
    print("-" * 70)
    print("✓ Each data type (pandas, numpy, pytorch, etc.) has isolated logic")
    print("✓ Easy to add new types without touching DataFolio core")
    print("✓ Handlers can be tested independently")
    print("✓ Storage backend is swappable")
    print("✓ Public API remains unchanged - backward compatible")


if __name__ == "__main__":
    demo_handler_system()
    demo_with_datafolio()
