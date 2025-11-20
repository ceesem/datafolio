# DataFolio Handler Architecture

## Overview

DataFolio has been refactored from a monolithic design to a modular handler-based architecture. This document explains how the components fit together.

## Current State (After Phases 1-2)

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      User Code                               │
│  folio = DataFolio('experiment')                            │
│  folio.add_table('data', df)        ← Public API unchanged  │
│  df = folio.get_table('data')                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   DataFolio                                  │
│  - Coordinates operations                                    │
│  - Manages metadata & lineage                                │
│  - Delegates I/O to StorageBackend                          │
│  - Delegates type logic to Handlers (Phase 3+)              │
└─────────┬─────────────────────────┬─────────────────────────┘
          │                         │
    ┌─────▼──────┐          ┌───────▼────────┐
    │  Storage   │          │   Handlers     │
    │  Backend   │          │   (Phase 3+)   │
    │            │          │                 │
    │  Handles:  │          │  Handles:      │
    │  • Files   │          │  • Type logic  │
    │  • JSON    │          │  • Metadata    │
    │  • Parquet │          │  • Validation  │
    │  • PyTorch │          │                 │
    │  • Numpy   │          │                 │
    │  • Cloud   │          │                 │
    └────────────┘          └─────────────────┘
```

## File Structure

```
src/datafolio/
├── __init__.py                 # Public API
├── folio.py                    # Main DataFolio class (~3200 lines)
│
├── base/                       # ✅ Phase 1: Foundation
│   ├── __init__.py
│   ├── handler.py             # BaseHandler abstract class
│   └── registry.py            # HandlerRegistry singleton
│
├── storage/                    # ✅ Phase 2: Storage Backend
│   ├── __init__.py
│   └── backend.py             # StorageBackend class (all I/O)
│
├── handlers/                   # 🚧 Phase 3+: Handlers
│   ├── __init__.py            # Auto-register handlers
│   ├── tables.py              # PandasHandler, ReferenceTableHandler
│   ├── arrays.py              # NumpyHandler (pending)
│   ├── json_data.py           # JsonHandler (pending)
│   ├── sklearn_models.py      # SklearnHandler (pending)
│   ├── pytorch_models.py      # PyTorchHandler (pending)
│   ├── artifacts.py           # ArtifactHandler (pending)
│   └── timestamps.py          # TimestampHandler (pending)
│
├── utils.py                    # Shared utilities
└── readers.py                  # Low-level readers
```

## Component Details

### 1. BaseHandler (Phase 1)

Abstract interface all handlers must implement:

```python
class BaseHandler(ABC):
    @property
    def item_type(self) -> str:
        """Unique ID: 'included_table', 'numpy_array', etc."""

    def can_handle(self, data: Any) -> bool:
        """Can this handler process this data type?"""

    def add(self, folio, name, data, **kwargs) -> Dict:
        """Write data, return metadata"""

    def get(self, folio, name, **kwargs) -> Any:
        """Read and return data"""

    def get_storage_subdir(self) -> str:
        """Where to store: 'tables', 'models', 'artifacts'"""
```

### 2. HandlerRegistry (Phase 1)

Global singleton that manages all handlers:

```python
from datafolio.base import register_handler, get_handler, detect_handler

# Register a handler
register_handler(PandasHandler())

# Get handler by type
handler = get_handler('included_table')

# Auto-detect handler for data
handler = detect_handler(pd.DataFrame())
```

### 3. StorageBackend (Phase 2)

Centralized I/O operations for all file types:

```python
class StorageBackend:
    # File system
    def exists(self, path) -> bool
    def mkdir(self, path) -> None
    def join_paths(self, *parts) -> str
    def delete_file(self, path) -> None
    def copy_file(self, src, dst) -> None

    # Format-specific I/O
    def write_json(self, path, data) -> None
    def read_json(self, path) -> Any
    def write_parquet(self, path, df) -> None
    def read_parquet(self, path) -> DataFrame
    def write_pytorch(self, path, model, ...) -> None
    def read_pytorch(self, path) -> Dict
    # ... etc for numpy, joblib, timestamps
```

**Benefits:**
- Single place for all I/O logic
- Works with local and cloud storage
- Easy to mock for testing
- Swappable implementation

### 4. Handlers (Phase 3+)

Example: PandasHandler

```python
class PandasHandler(BaseHandler):
    @property
    def item_type(self) -> str:
        return "included_table"

    def can_handle(self, data: Any) -> bool:
        import pandas as pd
        return isinstance(data, pd.DataFrame)

    def add(self, folio, name, data, description=None, inputs=None, **kwargs):
        """Add DataFrame to folio."""
        # 1. Build filepath
        filename = f"{name}.parquet"
        filepath = folio._storage.join_paths(
            folio._bundle_dir, TABLES_DIR, filename
        )

        # 2. Write data
        folio._storage.write_parquet(filepath, data)

        # 3. Build metadata
        return {
            "name": name,
            "item_type": "included_table",
            "filename": filename,
            "num_rows": len(data),
            "num_cols": len(data.columns),
            "columns": list(data.columns),
            "dtypes": {...},
            "description": description,
            "inputs": inputs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def get(self, folio, name, **kwargs):
        """Load DataFrame from folio."""
        item = folio._items[name]
        filepath = folio._storage.join_paths(
            folio._bundle_dir, TABLES_DIR, item["filename"]
        )
        return folio._storage.read_parquet(filepath, **kwargs)

    def get_storage_subdir(self) -> str:
        return TABLES_DIR
```

## Data Flow

### Adding Data (Current - Pre-Handler Integration)

```
User Code
   folio.add_table('data', df)
      │
      ▼
DataFolio.add_table()
   • Validates DataFrame
   • Builds filename
      │
      ▼
DataFolio._storage.write_parquet(path, df)
      │
      ▼
StorageBackend.write_parquet()
   • Writes to local or cloud
      │
      ▼
DataFolio._items['data'] = metadata
DataFolio._save_items()
```

### Adding Data (Future - With Handler Integration)

```
User Code
   folio.add_table('data', df)
      │
      ▼
DataFolio.add_table()
   • Gets handler: get_handler('included_table')
      │
      ▼
PandasHandler.add(folio, 'data', df, ...)
   • Builds filepath
   • Calls folio._storage.write_parquet()
   • Returns metadata dict
      │
      ▼
DataFolio._items['data'] = metadata
DataFolio._save_items()
```

**Key Difference:** Type-specific logic moves from DataFolio into handlers.

## Adding New Data Types

### Before (Monolithic)

To add Polars support:
1. Edit `folio.py` (3659 lines)
2. Add `add_polars()` method (~100 lines)
3. Add `get_polars()` method (~50 lines)
4. Update `add_data()` detection (~10 lines)
5. Update `get_data()` dispatch (~5 lines)
6. Update `delete()` handling (~10 lines)
7. Risk breaking existing functionality

### After (Handler Architecture)

To add Polars support:
1. Create `handlers/polars.py` (~150 lines)
2. Implement `PolarsHandler(BaseHandler)`
3. Register: `register_handler(PolarsHandler())`
4. Done! Works with `add_data()` automatically

**Example:**

```python
# handlers/polars.py
class PolarsHandler(BaseHandler):
    @property
    def item_type(self) -> str:
        return "polars_table"

    def can_handle(self, data: Any) -> bool:
        import polars as pl
        return isinstance(data, pl.DataFrame)

    def add(self, folio, name, data, **kwargs):
        # Polars-specific serialization
        filename = f"{name}.parquet"
        filepath = folio._storage.join_paths(...)
        data.write_parquet(filepath)  # Polars method
        return metadata

    def get(self, folio, name, **kwargs):
        import polars as pl
        filepath = ...
        return pl.read_parquet(filepath)

    def get_storage_subdir(self) -> str:
        return TABLES_DIR

# Register it
register_handler(PolarsHandler())

# Now it just works!
folio.add_data('data', polars_df)  # Auto-detects Polars
```

## Testing Strategy

### Unit Tests

Each component can be tested independently:

```python
# Test handler in isolation
def test_pandas_handler():
    handler = PandasHandler()
    assert handler.item_type == 'included_table'
    assert handler.can_handle(pd.DataFrame())
    assert not handler.can_handle([1, 2, 3])

# Test storage backend in isolation
def test_storage_backend():
    storage = StorageBackend()
    storage.write_json('/tmp/test.json', {'key': 'value'})
    data = storage.read_json('/tmp/test.json')
    assert data == {'key': 'value'}

# Test registry in isolation
def test_registry():
    registry = HandlerRegistry()
    handler = PandasHandler()
    registry.register(handler)
    assert registry.get('included_table') is handler
```

### Integration Tests

Existing tests continue to work (backward compatible):

```python
def test_add_table():
    folio = DataFolio('test')
    df = pd.DataFrame({'a': [1, 2, 3]})
    folio.add_table('data', df)
    loaded = folio.get_table('data')
    assert loaded.equals(df)
```

## Progress

### ✅ Completed (Phases 1-2)

- [x] **Phase 1: Foundation**
  - BaseHandler abstract class
  - HandlerRegistry system
  - 13 new tests, all passing

- [x] **Phase 2: Storage Backend**
  - StorageBackend class (all I/O operations)
  - Extracted 16 methods from DataFolio
  - 62 storage method calls updated
  - All 278 tests passing

### 🚧 In Progress (Phase 3+)

- [ ] **Phase 3: First Handler (Prototype)**
  - PandasHandler ✅ (created but not integrated)
  - ReferenceTableHandler ✅ (created but not integrated)
  - Update DataFolio.add_table() to use handler
  - Validate design with real implementation

- [ ] **Phase 4-6: Remaining Handlers**
  - NumpyHandler
  - JsonHandler
  - SklearnHandler
  - PyTorchHandler
  - ArtifactHandler
  - TimestampHandler

### 📋 Future (Phases 7-9)

- [ ] Extract supporting classes (metadata, accessors, display)
- [ ] Documentation and cleanup
- [ ] Release preparation

## Benefits Summary

### For Users
- ✅ **No API changes** - existing code works unchanged
- ✅ **Better performance** - no regression, potentially faster
- ✅ **Future-proof** - easy to add new data types

### For Developers
- ✅ **Easier to understand** - each file has single responsibility
- ✅ **Easier to test** - components can be tested in isolation
- ✅ **Easier to extend** - add new types without touching core
- ✅ **Less risk** - changes to one handler don't affect others

### For Maintainers
- ✅ **Clear ownership** - bug in PyTorch? Fix pytorch_models.py
- ✅ **Smaller files** - ~200 lines vs 3659 lines
- ✅ **Better navigation** - tree structure vs monolith
- ✅ **Modular design** - swap storage, add handlers, etc.

## Design Principles

1. **Backward Compatibility**: Existing code must work unchanged
2. **Single Responsibility**: Each component does one thing well
3. **Open/Closed**: Open for extension, closed for modification
4. **Dependency Injection**: Components receive dependencies (not hard-coded)
5. **Interface Segregation**: Small, focused interfaces
6. **Don't Repeat Yourself**: Single source of truth for each concern

## Next Steps

To complete the refactor:
1. Integrate PandasHandler into DataFolio.add_table()
2. Validate the handler design works end-to-end
3. Implement remaining 6 handlers
4. Extract supporting classes for final cleanup

---

**Status:** 2 of 9 phases complete (Foundation + Storage Backend)
**Est. Completion:** 10-12 more days (8-10 days remaining)
