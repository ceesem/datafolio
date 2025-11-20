# DataFolio Architecture

## Table of Contents
- [1. Overview](#1-overview)
- [2. Core Components](#2-core-components)
- [3. Handler System](#3-handler-system)
- [4. Storage Architecture](#4-storage-architecture)
- [5. Data Flow](#5-data-flow)
- [6. Built-in Handlers](#6-built-in-handlers)
- [7. Testing Strategy](#7-testing-strategy)
- [8. Design Decisions](#8-design-decisions)
- [9. Extensibility](#9-extensibility)
- [10. Performance & Future Considerations](#10-performance--future-considerations)

---

## 1. Overview

DataFolio uses a **handler-based plugin architecture** that separates data type logic from core orchestration. This modular design makes the system extensible, maintainable, and testable.

### 1.1 What is DataFolio?

DataFolio is a system for bundling machine learning experiments—datasets, models, metadata, and artifacts—into portable, shareable packages. It tracks lineage, manages metadata, and supports both local and cloud storage.

### 1.2 Architectural Principles

The architecture is built on these core principles:

1. **Modularity**: Each data type is handled by a separate, focused module (~35-80 lines)
2. **Extensibility**: Add new data types without modifying core code
3. **Single Responsibility**: Each component has one clear purpose
4. **Composition over Inheritance**: Handlers are plugins, not subclasses
5. **Type Safety**: Enums and type hints throughout

### 1.3 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Code                               │
│  folio = DataFolio('experiment')                            │
│  folio.add_data('data', df)        ← Auto-detection         │
│  folio.add_pytorch('model', net)   ← Type-specific          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   DataFolio (764 lines)                      │
│  • Bundle orchestration                                      │
│  • Metadata & lineage tracking                              │
│  • Delegates to handlers for type-specific logic            │
└─────────┬─────────────────────────┬─────────────────────────┘
          │                         │
    ┌─────▼──────┐          ┌───────▼────────┐
    │  Storage   │          │   Handlers     │
    │  Backend   │          │   (8 types)    │
    │  229 lines │          │                 │
    │            │          │  • Tables (2)   │
    │  • Local   │          │  • Models (2)   │
    │  • Cloud   │          │  • Arrays       │
    │  • Formats │          │  • JSON         │
    │            │          │  • Artifacts    │
    │            │          │  • Timestamps   │
    └────────────┘          └─────────────────┘
```

### 1.4 Current State Summary

**Codebase Metrics:**
- **Main file**: 764 lines (reduced from 3,659 - **79% smaller**)
- **8 handlers**: All implemented and tested
- **5 modules at 100% coverage**: metadata.py, timestamps.py, categories.py, handlers/__init__.py, base/registry.py

**Test Coverage:**
- **417 tests passing** (all green)
- **79% overall coverage**
- **108 handler & component unit tests**
- **309 integration tests**

**Supported Data Types:**
1. `included_table` - Pandas DataFrames (Parquet)
2. `referenced_table` - External table references
3. `numpy_array` - NumPy arrays (.npy)
4. `json_data` - Dictionaries and lists (JSON)
5. `model` - Scikit-learn models (Joblib)
6. `pytorch_model` - PyTorch models (.pt with enhanced reconstruction)
7. `artifact` - Arbitrary files
8. `timestamp` - DateTime objects (ISO 8601)

**Storage Support:**
- Local filesystem
- AWS S3 (`s3://bucket/path`)
- Google Cloud Storage (`gs://bucket/path`)
- Any CloudFiles-supported backend

## 2. Core Components

The system consists of six main components that work together to provide the handler-based architecture.

### 2.1 DataFolio Class (`folio.py` - 764 lines)

The main orchestrator that users interact with. Responsibilities:

**Bundle Management:**
- Create/load/save bundles
- Track bundle metadata (created_at, updated_at, description)
- Manage bundle directory structure

**Item Registry:**
- Maintain `_items` dict mapping names to metadata
- Save/load items.json
- Prevent duplicate names

**Lineage Tracking:**
- Record inputs for each item
- Build dependency graph
- Query dependents and dependencies

**API Methods:**
```python
# Generic API (auto-detection)
folio.add_data(name, data, description=None, inputs=None)
folio.get_data(name)
folio.delete(name)

# Type-specific API (explicit handlers)
folio.add_table(name, df, ...)
folio.add_pytorch(name, model, init_args=None, save_class=False, ...)
folio.add_numpy(name, array, ...)
folio.add_json(name, data, ...)
folio.add_model(name, model, ...)  # Auto-detects sklearn/pytorch
folio.add_artifact(name, filepath, ...)
folio.add_timestamp(name, dt, ...)
folio.reference_table(name, path, ...)

# Retrieval (type-specific methods available)
folio.get_table(name)
folio.get_pytorch(name, model_class=None, reconstruct=True)
# ... etc

# Metadata & lineage
folio.describe()
folio.list_contents()
folio.get_inputs(name)
folio.get_dependents(name)

# Validation & Batching
with folio.batch():
    folio.add_data(...)
    folio.add_model(...)

folio.validate()  # Check integrity
folio.is_valid()  # Boolean check
```

### 2.2 BaseHandler (`base/handler.py` - 27 lines)

Abstract base class defining the handler interface. All handlers must inherit from this.

**Required Properties:**
```python
@property
@abstractmethod
def item_type(self) -> str:
    """Unique identifier like 'pytorch_model' or 'included_table'"""
```

**Required Methods:**
```python
@abstractmethod
def add(self, folio: DataFolio, name: str, data: Any, **kwargs) -> Dict[str, Any]:
    """
    Store data and return metadata dictionary.

    Must include: name, item_type, filename, created_at
    May include: description, inputs, type-specific fields
    """

@abstractmethod
def get(self, folio: DataFolio, name: str, **kwargs) -> Any:
    """Load and return the data"""
```

**Optional Methods:**
```python
def can_handle(self, data: Any) -> bool:
    """Return True if this handler can process this data type"""
    return False  # Default: can't auto-detect

def delete(self, folio: DataFolio, name: str) -> None:
    """Delete files. Default implementation handles single file."""
    # Default provided by BaseHandler

def get_storage_subdir(self) -> str:
    """Auto-derived from item_type via ITEM_TYPE_TO_CATEGORY"""
    # Automatic - handlers don't override this
```

### 2.3 HandlerRegistry (`base/registry.py` - 33 lines)

Singleton managing all registered handlers. Uses module-level functions for clean API.

**Key Functions:**
```python
def get_registry() -> HandlerRegistry:
    """Get the global registry singleton"""

registry = get_registry()

# Register a handler
registry.register(handler: BaseHandler)

# Get handler by item_type
handler = registry.get(item_type: str) -> BaseHandler

# Auto-detect handler for data
handler = registry.detect_handler(data: Any) -> Optional[BaseHandler]

# Clear registry (mainly for testing)
registry.clear()
```

**Registration Order:**
Handlers are registered in order of specificity (most specific first):
1. TimestampHandler (datetime objects)
2. PandasHandler (DataFrames)
3. ReferenceTableHandler (table references)
4. PyTorchHandler (torch.nn.Module)
5. SklearnHandler (any joblib-serializable - catch-all)
6. NumpyHandler (numpy arrays)
7. JsonHandler (dicts and lists)
8. ArtifactHandler (file paths)

This ordering ensures correct auto-detection when types overlap.

### 2.4 StorageBackend (`storage/backend.py` - 229 lines)

Handles all I/O operations with unified interface for local and cloud storage.

**File System Operations:**
```python
def exists(self, path: str) -> bool
def mkdir(self, path: str, parents=True, exist_ok=True) -> None
def join_paths(self, *parts) -> str
def delete_file(self, path: str) -> None
def copy_file(self, src: str, dst: str) -> None
def calculate_checksum(self, path: str) -> str  # MD5 hash
```

**Format-Specific I/O:**
```python
# JSON
def write_json(self, path: str, data: dict) -> None
def read_json(self, path: str) -> dict

# Parquet (pandas DataFrames)
def write_parquet(self, path: str, df: pd.DataFrame, **kwargs) -> None
def read_parquet(self, path: str, **kwargs) -> pd.DataFrame

# Joblib (sklearn models, any serializable object)
def write_joblib(self, path: str, obj: Any) -> None
def read_joblib(self, path: str) -> Any

# PyTorch
def write_pytorch(
    self, path: str, model: Any,
    init_args: Optional[Dict] = None,
    save_class: bool = False,
    optimizer_state: Optional[Dict] = None
) -> None
def read_pytorch(self, path: str, **kwargs) -> Dict[str, Any]

# NumPy
def write_numpy(self, path: str, array: np.ndarray) -> None
def read_numpy(self, path: str) -> np.ndarray

# Timestamps
def write_timestamp(self, path: str, timestamp: datetime) -> None
def read_timestamp(self, path: str) -> datetime
```

**Cloud Support:**
Uses `cloudfiles` library to transparently handle cloud paths. Same API works for:
- Local: `/path/to/bundle`
- S3: `s3://bucket/bundle`
- GCS: `gs://bucket/bundle`

### 2.5 StorageCategory (`storage/categories.py` - 14 lines)

Type-safe enum system for organizing items into subdirectories.

**Enum Definition:**
```python
class StorageCategory(Enum):
    TABLES = "tables"      # DataFrames and references
    MODELS = "models"      # ML models (sklearn, pytorch)
    ARTIFACTS = "artifacts"  # Everything else
```

**Item Type Mapping:**
```python
ITEM_TYPE_TO_CATEGORY = {
    "included_table": StorageCategory.TABLES,
    "referenced_table": StorageCategory.TABLES,
    "model": StorageCategory.MODELS,
    "pytorch_model": StorageCategory.MODELS,
    "numpy_array": StorageCategory.ARTIFACTS,
    "json_data": StorageCategory.ARTIFACTS,
    "artifact": StorageCategory.ARTIFACTS,
    "timestamp": StorageCategory.ARTIFACTS,
}
```

**Automatic Derivation:**
Handlers don't manually specify subdirectories. The BaseHandler automatically derives the subdirectory from the handler's `item_type` using this mapping.

**Benefits:**
- Single source of truth for organization
- Type-safe (can't typo category names)
- Easy to reorganize (change mapping, not handlers)
- Extensible for category-specific behavior (compression, permissions, etc.)

### 2.6 Supporting Modules

#### MetadataDict (`metadata.py` - 31 lines, 100% coverage)

Auto-save dictionary for bundle metadata:

```python
folio.metadata['experiment_name'] = 'test'  # Auto-saves
folio.metadata.update({'author': 'Alice'})  # Auto-saves
```

**Features:**
- Inherits from `dict`
- Triggers `folio._save_metadata()` on any modification
- Auto-updates `updated_at` timestamp
- Prevents infinite recursion when setting `updated_at`

**Tested with 25 unit tests** covering all modification methods.

#### DataAccessor & ItemProxy (`accessors.py` - 88 lines, 98% coverage)

Provides convenient data access patterns:

```python
# Dictionary-style access
folio['data'] = df           # Calls add_data()
df = folio['data']           # Calls get_data()
del folio['data']            # Calls delete()

# Attribute-style access (ItemProxy)
folio.data.get()             # Get data
folio.data.metadata          # Get metadata
folio.data.inputs            # Get inputs
folio.data.dependents        # Get dependents
```

#### DisplayFormatter (`display.py` - 265 lines, 73% coverage)

Formats the output of `folio.describe()`:

```python
folio.describe()  # Pretty-printed table of all items
folio.describe(name='specific_item')  # Details for one item
```

Generates formatted text with:
- Item counts by type
- Table with columns: Name, Type, Created, Description
- Detailed metadata for individual items

## 3. Handler System

The handler system is the core of DataFolio's extensibility. Each data type is managed by a dedicated handler that knows how to serialize, deserialize, and validate that type.

### 3.1 Handler Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│ 1. Registration (on module import)                      │
│    handlers/__init__.py imports all handler modules     │
│    Each handler self-registers with HandlerRegistry     │
│    Order matters for auto-detection                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Auto-Detection (folio.add_data)                      │
│    registry.detect_handler(data) → handler              │
│    Tries each handler's can_handle() in order           │
│    Returns first handler that returns True              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Adding Data                                           │
│    handler.add(folio, name, data, **kwargs)             │
│    → Builds filepath from name and storage category     │
│    → Calls folio._storage.write_*() to save             │
│    → Returns metadata dict with all item info           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Metadata Storage                                      │
│    folio._items[name] = metadata                        │
│    folio._save_items() → writes items.json              │
│    Bundle is now in consistent state                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Retrieval (folio.get_data)                           │
│    metadata = folio._items[name]                        │
│    handler = registry.get(metadata['item_type'])        │
│    data = handler.get(folio, name, **kwargs)            │
│    → Builds filepath from metadata['filename']          │
│    → Calls folio._storage.read_*() to load              │
│    → Returns deserialized data                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Deletion (folio.delete)                               │
│    handler.delete(folio, name)                           │
│    → Deletes file(s) from storage                       │
│    → Removes from folio._items                          │
│    → Saves updated items.json                           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Handler Interface Details

#### The `add()` Method

**Signature:**
```python
def add(
    self,
    folio: DataFolio,
    name: str,
    data: Any,
    description: Optional[str] = None,
    inputs: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
```

**Responsibilities:**
1. Build filename (e.g., `f"{name}.parquet"`)
2. Get storage subdirectory: `self.get_storage_subdir()`
3. Build full filepath: `folio._storage.join_paths(bundle_dir, subdir, filename)`
4. Write data: `folio._storage.write_*(filepath, data)`
5. Build metadata dict with required + optional fields
6. Return metadata

**Required Metadata Fields:**
- `name`: Item name
- `item_type`: Handler's item type
- `filename`: Filename used for storage
- `created_at`: UTC timestamp (ISO 8601)

**Optional Metadata Fields:**
- `description`: User-provided description
- `inputs`: List of input item names (lineage)
- Type-specific fields (columns, shape, model_type, etc.)

**Example:**
```python
def add(self, folio, name, data, description=None, inputs=None, **kwargs):
    filename = f"{name}.npy"
    subdir = self.get_storage_subdir()  # → "artifacts"
    filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

    folio._storage.write_numpy(filepath, data)

    return {
        "name": name,
        "item_type": "numpy_array",
        "filename": filename,
        "shape": data.shape,
        "dtype": str(data.dtype),
        "description": description,
        "inputs": inputs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
```

#### The `get()` Method

**Signature:**
```python
def get(
    self,
    folio: DataFolio,
    name: str,
    **kwargs
) -> Any:
```

**Responsibilities:**
1. Get metadata: `item = folio._items[name]`
2. Build filepath from metadata['filename'] and storage subdir
3. Read data: `folio._storage.read_*(filepath, **kwargs)`
4. Return deserialized data

**Example:**
```python
def get(self, folio, name, **kwargs):
    item = folio._items[name]
    subdir = self.get_storage_subdir()
    filepath = folio._storage.join_paths(
        folio._bundle_dir, subdir, item["filename"]
    )
    return folio._storage.read_numpy(filepath)
```

#### The `can_handle()` Method

**Signature:**
```python
def can_handle(self, data: Any) -> bool:
```

**Purpose:** Enable auto-detection in `add_data()`

**Guidelines:**
- Return `True` only if handler can definitely process this data
- Use `isinstance()` checks for type detection
- Import dependencies inside method (avoid import errors)
- Be specific (avoid catching too many types)

**Example:**
```python
def can_handle(self, data: Any) -> bool:
    try:
        import pandas as pd
        return isinstance(data, pd.DataFrame)
    except ImportError:
        return False
```

#### The `delete()` Method

**Signature:**
```python
def delete(self, folio: DataFolio, name: str) -> None:
```

**Default Implementation:**
Handles single-file deletion (works for most handlers):

```python
def delete(self, folio, name):
    item = folio._items[name]
    subdir = self.get_storage_subdir()
    filepath = folio._storage.join_paths(
        folio._bundle_dir, subdir, item["filename"]
    )
    folio._storage.delete_file(filepath)
```

**Override when:**
- Multiple files need deletion
- Special cleanup required

#### The `get_storage_subdir()` Method

**Auto-derived - handlers don't override!**

```python
def get_storage_subdir(self) -> str:
    category = ITEM_TYPE_TO_CATEGORY[self.item_type]
    return category.value  # "tables", "models", or "artifacts"
```

This method is provided by BaseHandler and uses the StorageCategory mapping.

### 3.3 Handler Registration

**Registration happens automatically** on import in `handlers/__init__.py`:

```python
# handlers/__init__.py
from datafolio.base import get_registry
from datafolio.handlers.timestamps import TimestampHandler
from datafolio.handlers.tables import PandasHandler, ReferenceTableHandler
from datafolio.handlers.pytorch_models import PyTorchHandler
from datafolio.handlers.sklearn_models import SklearnHandler
from datafolio.handlers.arrays import NumpyHandler
from datafolio.handlers.json_data import JsonHandler
from datafolio.handlers.artifacts import ArtifactHandler

# Get global registry
_registry = get_registry()

# Register in order of specificity (most specific first)
_registry.register(TimestampHandler())
_registry.register(PandasHandler())
_registry.register(ReferenceTableHandler())
_registry.register(PyTorchHandler())
_registry.register(SklearnHandler())  # Catch-all for joblib-serializable
_registry.register(NumpyHandler())
_registry.register(JsonHandler())
_registry.register(ArtifactHandler())
```

**Why Registration Order Matters:**

When `detect_handler()` is called, handlers are tried in registration order. The first handler whose `can_handle()` returns `True` is selected.

**Example conflict:** SklearnHandler accepts any joblib-serializable object. If registered first, it would catch numpy arrays and dicts! By registering it after NumpyHandler and JsonHandler, more specific handlers get priority.

**Current Order Logic:**
1. **TimestampHandler** - Very specific (datetime objects)
2. **PandasHandler** - Specific (DataFrames)
3. **ReferenceTableHandler** - Doesn't auto-detect (manual registration)
4. **PyTorchHandler** - Specific (torch.nn.Module)
5. **SklearnHandler** - General (any joblib-serializable) ← Catch-all
6. **NumpyHandler** - Specific (numpy arrays)
7. **JsonHandler** - General (dicts/lists)
8. **ArtifactHandler** - General (file paths)

### 3.4 Error Handling

Handlers should raise clear exceptions:

**During `add()`:**
- `TypeError`: Wrong data type (e.g., not a DataFrame)
- `ValueError`: Invalid data (e.g., empty DataFrame)
- `ImportError`: Required library not installed

**During `get()`:**
- `KeyError`: Item doesn't exist (raised by DataFolio)
- `FileNotFoundError`: File missing from storage
- `ImportError`: Required library not installed for reconstruction

**Example:**
```python
def add(self, folio, name, data, **kwargs):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "PyTorch is required to save PyTorch models. "
            "Install with: pip install torch"
        )

    if not isinstance(data, nn.Module):
        raise TypeError(
            f"Expected torch.nn.Module, got {type(data).__name__}"
        )

    # ... rest of implementation
```

## 4. Storage Architecture

DataFolio organizes files in a structured directory layout that works identically for local and cloud storage.

### 4.1 Bundle Directory Structure

```
bundle_dir/
├── metadata.json          # Bundle-level metadata
├── items.json             # Registry of all items
│
├── tables/                # TABLES category
│   ├── train_data.parquet
│   ├── test_data.parquet
│   └── validation.parquet
│
├── models/                # MODELS category
│   ├── random_forest.joblib
│   ├── neural_net.pt
│   └── xgboost_model.joblib
│
└── artifacts/             # ARTIFACTS category
    ├── embeddings.npy
    ├── config.json
    ├── preprocessor.joblib
    ├── plot.png
    └── experiment_end.timestamp
```

**File Descriptions:**

- `metadata.json` - Bundle-level metadata (name, description, created_at, updated_at)
- `items.json` - Array of all item metadata dicts
- `tables/` - DataFrames and table references
- `models/` - ML models (sklearn, PyTorch)
- `artifacts/` - Everything else (numpy, JSON, files, timestamps)

### 4.2 Storage Categories

Items are organized into three categories based on their `item_type`:

#### TABLES Category (`tables/`)
**Item Types:**
- `included_table` - Pandas DataFrames stored as Parquet
- `referenced_table` - External table references (no local copy)

**File Format:** `.parquet`

**Why separate?** Tables often have special handling needs (column tracking, data validation, query optimization).

#### MODELS Category (`models/`)
**Item Types:**
- `model` - Scikit-learn models and any joblib-serializable objects
- `pytorch_model` - PyTorch neural networks

**File Formats:** `.joblib`, `.pt`

**Why separate?** Models are typically large, versioned, and may need special handling (checkpointing, optimization states).

#### ARTIFACTS Category (`artifacts/`)
**Item Types:**
- `numpy_array` - NumPy arrays (`.npy`)
- `json_data` - JSON data (`.json`)
- `artifact` - Arbitrary files (preserves extension)
- `timestamp` - DateTime markers (`.timestamp`)

**Why catch-all?** These types don't need special organization and work well grouped together.

### 4.3 Category Mapping

The mapping from `item_type` to category is centralized in `storage/categories.py`:

```python
ITEM_TYPE_TO_CATEGORY = {
    # TABLES
    "included_table": StorageCategory.TABLES,
    "referenced_table": StorageCategory.TABLES,

    # MODELS
    "model": StorageCategory.MODELS,
    "pytorch_model": StorageCategory.MODELS,

    # ARTIFACTS
    "numpy_array": StorageCategory.ARTIFACTS,
    "json_data": StorageCategory.ARTIFACTS,
    "artifact": StorageCategory.ARTIFACTS,
    "timestamp": StorageCategory.ARTIFACTS,
}
```

**Benefits:**
- Single source of truth - easy to reorganize
- Type-safe enum prevents typos
- Extensible for future category-specific behavior

### 4.4 Cloud Storage Support

The `StorageBackend` uses the `cloudfiles` library to transparently support cloud storage.

**Supported Backends:**
- **Local**: `/path/to/bundle` or `file:///path/to/bundle`
- **AWS S3**: `s3://bucket-name/path/to/bundle`
- **Google Cloud Storage**: `gs://bucket-name/path/to/bundle`
- **Azure Blob Storage**: `https://account.blob.core.windows.net/container/bundle`

**Usage:**
```python
# Local
folio = DataFolio('/local/experiments/exp1')

# S3
folio = DataFolio('s3://my-bucket/experiments/exp1')

# GCS
folio = DataFolio('gs://my-bucket/experiments/exp1')

# Everything works identically
folio.add_table('data', df)  # Writes to cloud
df = folio.get_table('data')  # Reads from cloud
```

**Cloud Path Detection:**
```python
def is_cloud_path(path: str) -> bool:
    """Detect if path is cloud storage"""
    return path.startswith(('s3://', 'gs://', 'https://'))
```

**Temporary File Pattern:**
For cloud writes, data is first written locally then uploaded:

```python
if is_cloud_path(path):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Write to local temp file
        df.to_parquet(tmp.name)

        # Upload to cloud
        cf = CloudFiles(dirname)
        with open(tmp.name, 'rb') as f:
            cf.put(filename, f.read())

        # Cleanup
        Path(tmp.name).unlink()
else:
    # Direct local write
    df.to_parquet(path)
```

### 4.5 File Naming Conventions

**Tables:**
```
{name}.parquet
```

**Models:**
```
{name}.joblib  # Sklearn/joblib-serializable
{name}.pt      # PyTorch
```

**Artifacts:**
```
{name}.npy         # NumPy
{name}.json        # JSON data
{name}.{ext}       # Artifacts (preserve original extension)
{name}.timestamp   # Timestamps
```

**Metadata Files:**
```
metadata.json  # Bundle metadata
items.json     # Item registry
```

### 4.6 Metadata Storage

#### Bundle Metadata (`metadata.json`)

```json
{
  "name": "experiment_1",
  "description": "Initial experiment with baseline model",
  "created_at": "2025-11-20T10:30:00+00:00",
  "updated_at": "2025-11-20T15:45:00+00:00",
  "author": "researcher",
  "tags": ["baseline", "v1"]
}
```

Managed by `MetadataDict` with auto-save on any modification.

#### Item Registry (`items.json`)

Array of all item metadata:

```json
[
  {
    "name": "train_data",
    "item_type": "included_table",
    "filename": "train_data.parquet",
    "num_rows": 1000,
    "num_cols": 10,
    "columns": ["feature1", "feature2", "..."],
    "description": "Training dataset",
    "inputs": [],
    "created_at": "2025-11-20T10:30:00+00:00"
  },
  {
    "name": "model",
    "item_type": "pytorch_model",
    "filename": "model.pt",
    "model_type": "ResNet",
    "torch_version": "2.0.0",
    "description": "Baseline ResNet model",
    "inputs": ["train_data"],
    "created_at": "2025-11-20T12:00:00+00:00"
  }
]
```

Saved after every `add()` or `delete()` operation.

### 4.7 Storage Backend Implementation

The `StorageBackend` provides a unified interface:

**Path Operations:**
```python
backend.exists(path)           # Check if file exists
backend.mkdir(path)            # Create directory
backend.join_paths(*parts)     # Join path components
backend.delete_file(path)      # Delete file
backend.copy_file(src, dst)    # Copy file
```

**Format Operations:**
Each format has read/write pair:
- `write_json()` / `read_json()`
- `write_parquet()` / `read_parquet()`
- `write_joblib()` / `read_joblib()`
- `write_pytorch()` / `read_pytorch()`
- `write_numpy()` / `read_numpy()`
- `write_timestamp()` / `read_timestamp()`

**Error Handling:**
- `FileNotFoundError` - File doesn't exist
- `PermissionError` - Insufficient permissions
- `ImportError` - Required library missing
- Format-specific errors propagate up

## 5. Data Flow

This section shows the detailed flow for the three main operations: adding, retrieving, and deleting data.

### 5.1 Adding Data

**Two API paths:**
1. **Generic API** - Auto-detects handler: `folio.add_data(name, data)`
2. **Type-specific API** - Explicit handler: `folio.add_table(name, df)`

#### Generic API Flow (with Auto-Detection)

```
User Code                    DataFolio              Registry              Handler             Storage
    │                            │                      │                      │                   │
    ├─ add_data('data', df) ────>│                      │                      │                   │
    │                            │                      │                      │                   │
    │                            ├─ detect_handler(df)─>│                      │                   │
    │                            │                      ├─ try each handler   │                   │
    │                            │                      │  can_handle(df)?    │                   │
    │                            │                      │  ... PandasHandler  │                   │
    │                            │                      │      returns True!  │                   │
    │                            │<─── handler ─────────┘                      │                   │
    │                            │                      │                      │                   │
    │                            ├─ add(self, 'data', df, ...) ───────────────>│                   │
    │                            │                      │                      │                   │
    │                            │                      │                      ├─ build filepath   │
    │                            │                      │                      │  = "tables/data.parquet"
    │                            │                      │                      │                   │
    │                            │                      │                      ├─ write_parquet()─>│
    │                            │                      │                      │                   ├─ save
    │                            │                      │                      │<──────────────────┘
    │                            │                      │                      │                   │
    │                            │                      │                      ├─ build metadata   │
    │                            │<───────────────────── metadata ─────────────┘                   │
    │                            │                      │                      │                   │
    │                            ├─ _items['data'] = metadata                  │                   │
    │                            ├─ _save_items() ────────────────────────────────────────────────>│
    │                            │                      │                      │                   ├─ save
    │                            │<────────────────────────────────────────────────────────────────┘
    │<─── folio ─────────────────┘                      │                      │                   │
```

#### Type-Specific API Flow

```
User Code                    DataFolio              Registry              Handler             Storage
    │                            │                      │                      │                   │
    ├─ add_table('data', df) ───>│                      │                      │                   │
    │                            │                      │                      │                   │
    │                            ├─ get('included_table')>│                      │                   │
    │                            │<─── handler ─────────┘                      │                   │
    │                            │                      │                      │                   │
    │                            ├─ add(self, 'data', df, ...) ───────────────>│                   │
    │                            │                      │                      ├─ [same as above]  │
    │                            │                      │                      │                   │
    │<─── folio ─────────────────┘                      │                      │                   │
```

**Key Differences:**
- Generic API calls `detect_handler()` (iterates through handlers)
- Type-specific API calls `get()` directly (O(1) lookup)
- Both use the same `handler.add()` method

### 5.2 Retrieving Data

**Two API paths:**
1. **Generic API**: `folio.get_data(name)`
2. **Type-specific API**: `folio.get_table(name)`, `folio.get_pytorch(name, ...)`

#### Retrieval Flow

```
User Code                    DataFolio              Registry              Handler             Storage
    │                            │                      │                      │                   │
    ├─ get_data('data') ────────>│                      │                      │                   │
    │                            │                      │                      │                   │
    │                            ├─ metadata = _items['data']                  │                   │
    │                            │  item_type = metadata['item_type']          │                   │
    │                            │            = 'included_table'               │                   │
    │                            │                      │                      │                   │
    │                            ├─ get('included_table')>│                      │                   │
    │                            │<─── handler ─────────┘                      │                   │
    │                            │                      │                      │                   │
    │                            ├─ get(self, 'data', **kwargs) ──────────────>│                   │
    │                            │                      │                      │                   │
    │                            │                      │                      ├─ build filepath   │
    │                            │                      │                      │  from metadata    │
    │                            │                      │                      │                   │
    │                            │                      │                      ├─ read_parquet()──>│
    │                            │                      │                      │                   ├─ load
    │                            │                      │                      │<─── df ───────────┘
    │                            │<───────────────────── df ───────────────────┘                   │
    │<─── df ────────────────────┘                      │                      │                   │
```

**Key Points:**
- Metadata lookup determines which handler to use
- Handler reconstructs filepath from metadata
- `**kwargs` passed through to handler (e.g., `reconstruct=False` for PyTorch)

### 5.3 Deleting Data

#### Deletion Flow

```
User Code                    DataFolio              Registry              Handler             Storage
    │                            │                      │                      │                   │
    ├─ delete('data') ──────────>│                      │                      │                   │
    │                            │                      │                      │                   │
    │                            ├─ metadata = _items['data']                  │                   │
    │                            ├─ get(metadata['item_type']) ───────────────>│                   │
    │                            │<─── handler ─────────┘                      │                   │
    │                            │                      │                      │                   │
    │                            ├─ delete(self, 'data') ─────────────────────>│                   │
    │                            │                      │                      │                   │
    │                            │                      │                      ├─ build filepath   │
    │                            │                      │                      ├─ delete_file() ──>│
    │                            │                      │                      │                   ├─ remove
    │                            │                      │                      │<──────────────────┘
    │                            │<───────────────────────────────────────────┘                   │
    │                            │                      │                      │                   │
    │                            ├─ del _items['data']  │                      │                   │
    │                            ├─ _save_items() ────────────────────────────────────────────────>│
    │                            │                      │                      │                   ├─ update
    │                            │<────────────────────────────────────────────────────────────────┘
    │<─── None ──────────────────┘                      │                      │                   │
```

**Key Points:**
- Handler deletes file(s) from storage
- DataFolio removes from `_items` registry
- Items.json is updated (removal persisted)

### 5.4 Code Example: Complete Workflow

```python
from datafolio import DataFolio
import pandas as pd
import torch.nn as nn

# Create bundle
folio = DataFolio('experiment_1')
folio.metadata['description'] = 'Baseline experiment'

# Add data (various types)
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
folio.add_table('data', df, description='Training data')

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
folio.add_pytorch('model', model, init_args={'input_dim': 10})

folio.add_json('config', {'lr': 0.001, 'epochs': 10})

# Lineage: model depends on data
folio.add_numpy('weights', model[0].weight.detach().numpy(), inputs=['model'])

# Inspect
folio.describe()
folio.list_contents()

# Retrieve
loaded_df = folio.get_table('data')
loaded_model = folio.get_pytorch('model', model_class=type(model))
config = folio.get_json('config')

# Query lineage
folio.get_inputs('weights')  # → ['model']
folio.get_dependents('model')  # → ['weights']

# Delete
folio.delete('weights')

# Bundle structure on disk:
# experiment_1/
# ├── metadata.json
# ├── items.json
# ├── tables/
# │   └── data.parquet
# ├── models/
# │   └── model.pt
# └── artifacts/
#     └── config.json
```

### 5.5 Error Handling in Data Flow

**During Add:**
```python
try:
    folio.add_data('item', data)
except ValueError:
    # No handler can process this data type
    print("Unsupported data type")
except TypeError:
    # Handler found but data validation failed
    print("Invalid data for this type")
except ValueError:
    # Name already exists
    print("Item name conflict")
```

**During Get:**
```python
try:
    data = folio.get_data('item')
except KeyError:
    # Item doesn't exist
    print("Item not found")
except FileNotFoundError:
    # Metadata exists but file missing
    print("Bundle corrupted - file missing")
except ImportError:
    # Required library not installed
    print("Install required library")
```

**During Delete:**
```python
try:
    folio.delete('item')
except KeyError:
    # Item doesn't exist
    print("Item not found")
# File errors are logged but don't raise
# (allow cleanup even if files missing)
```

## 6. Built-in Handlers

DataFolio includes 8 built-in handlers covering common ML workflow data types.

**Common Features:**
- All handlers calculate MD5 checksums for data integrity
- All handlers support metadata tracking (creation time, inputs, description)
- All handlers integrate with the storage backend for unified I/O

### 6.1 PandasHandler (`handlers/tables.py`)

**Item Type:** `included_table`

**Purpose:** Store pandas DataFrames as Parquet files

**Features:**
- Efficient columnar storage
- Preserves column types
- Tracks row/column counts
- Stores column names and dtypes

**Usage:**
```python
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
folio.add_table('data', df, description='Training data')

# Retrieve
df = folio.get_table('data')
```

**Metadata Example:**
```json
{
  "name": "data",
  "item_type": "included_table",
  "filename": "data.parquet",
  "num_rows": 3,
  "num_cols": 2,
  "columns": ["x", "y"],
  "dtypes": {"x": "int64", "y": "int64"},
  "created_at": "2025-11-20T10:00:00+00:00"
}
```

### 6.2 ReferenceTableHandler (`handlers/tables.py`)

**Item Type:** `referenced_table`

**Purpose:** Reference external tables without copying data

**Features:**
- Stores path/URL only (no local copy)
- Supports Parquet, CSV, Feather formats
- Supports directory references (e.g., Delta Lake, DuckDB tables)
- Handles `file://` URIs and relative paths
- Useful for large external datasets
- Preserves column tracking

**Usage:**
```python
folio.reference_table(
    'external',
    'data/large_dataset.parquet',
    description='Large external table'
)

# Retrieve (reads from original location)
df = folio.get_table('external')
```

**Metadata Example:**
```json
{
  "name": "external",
  "item_type": "referenced_table",
  "path": "data/large_dataset.parquet",
  "format": "parquet",
  "columns": ["col1", "col2", "..."],
  "num_rows": 1000000,
  "created_at": "2025-11-20T10:00:00+00:00"
}
```

### 6.3 NumpyHandler (`handlers/arrays.py`)

**Item Type:** `numpy_array`

**Purpose:** Store NumPy arrays in .npy format

**Features:**
- Preserves shape and dtype
- Efficient binary format
- Supports all numpy dtypes
- Tracks memory usage

**Usage:**
```python
embeddings = np.random.randn(100, 768)
folio.add_numpy('embeddings', embeddings, description='BERT embeddings')

# Retrieve
embeddings = folio.get_numpy('embeddings')
```

**Metadata Example:**
```json
{
  "name": "embeddings",
  "item_type": "numpy_array",
  "filename": "embeddings.npy",
  "shape": [100, 768],
  "dtype": "float64",
  "created_at": "2025-11-20T10:00:00+00:00"
}
```

### 6.4 JsonHandler (`handlers/json_data.py`)

**Item Type:** `json_data`

**Purpose:** Store dictionaries and lists as JSON

**Features:**
- Uses `orjson` for performance
- Handles nested structures
- Serializes numpy values automatically
- Supports complex data structures

**Usage:**
```python
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'optimizer': {'type': 'Adam', 'params': {...}}
}
folio.add_json('config', config, description='Experiment config')

# Retrieve
config = folio.get_json('config')
```

**Auto-Detection:**
Handles `dict` and `list` types (but not primitive types to avoid conflicts).

### 6.5 SklearnHandler (`handlers/sklearn_models.py`)

**Item Type:** `model`

**Purpose:** Store any joblib-serializable object (primarily sklearn models)

**Features:**
- Uses joblib compression
- Auto-detects model type
- Stores hyperparameters if provided
- Catch-all for serializable objects

**Usage:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

folio.add_model(
    'classifier',
    model,
    description='Random forest baseline',
    hyperparameters={'n_estimators': 100, 'max_depth': 10}
)

# Retrieve
model = folio.get_model('classifier')
```

**Auto-Detection:**
Checks for `fit` and `predict` methods (sklearn duck-typing).

**Note:** This handler accepts **any** joblib-serializable object, not just sklearn. It's registered late to avoid catching numpy arrays and dicts.

### 6.6 PyTorchHandler (`handlers/pytorch_models.py`)

**Item Type:** `pytorch_model`

**Purpose:** Store PyTorch models with enhanced reconstruction

**Features:**
- Stores state_dict (not full model - PyTorch best practice)
- Enhanced reconstruction with multiple strategies:
  - Provide `model_class` manually
  - Auto-reconstruct from metadata
  - Use dill-serialized class
- Supports `init_args` for model instantiation
- Optional optimizer state storage
- Tracks PyTorch version

**Usage:**

**Basic (with model_class):**
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

folio.add_pytorch(
    'model',
    model,
    init_args={'input_dim': 10},
    description='Simple feedforward network'
)

# Retrieve - provide class
loaded = folio.get_pytorch('model', model_class=type(model))
```

**With dill (auto-reconstruction):**
```python
folio.add_pytorch(
    'model',
    model,
    init_args={'input_dim': 10},
    save_class=True  # Serialize class with dill
)

# Retrieve - no model_class needed!
loaded = folio.get_pytorch('model')
```

**State dict only:**
```python
state_dict = folio.get_pytorch('model', reconstruct=False)
model = MyModel(10, 20)
model.load_state_dict(state_dict)
```

**Metadata Example:**
```json
{
  "name": "model",
  "item_type": "pytorch_model",
  "filename": "model.pt",
  "model_type": "Sequential",
  "torch_version": "2.0.0",
  "init_args": {"input_dim": 10},
  "has_serialized_class": true,
  "created_at": "2025-11-20T10:00:00+00:00"
}
```

**Storage Bundle:**
The .pt file contains:
```python
{
    "state_dict": {...},  # Model weights
    "metadata": {
        "model_class": "Sequential",
        "model_module": "torch.nn.modules.container",
        "init_args": {"input_dim": 10}
    },
    "serialized_class": b"...",  # Optional dill bytes
    "optimizer_state": {...}  # Optional
}
```

### 6.7 ArtifactHandler (`handlers/artifacts.py`)

**Item Type:** `artifact`

**Purpose:** Store arbitrary files

**Features:**
- Preserves original file extension
- Copies file to bundle
- No format restrictions
- Useful for plots, configs, checkpoints

**Usage:**
```python
# Save plot
folio.add_artifact('plot', 'results/training_curve.png')

# Retrieve path (file copied to bundle)
path = folio.get_artifact_path('plot')
# → bundle_dir/artifacts/plot.png
```

**Metadata Example:**
```json
{
  "name": "plot",
  "item_type": "artifact",
  "filename": "plot.png",
  "original_path": "results/training_curve.png",
  "created_at": "2025-11-20T10:00:00+00:00"
}
```

### 6.8 TimestampHandler (`handlers/timestamps.py`)

**Item Type:** `timestamp`

**Purpose:** Store datetime markers for experiment tracking

**Features:**
- ISO 8601 format
- Enforces timezone awareness
- Auto-converts to UTC
- Useful for timing experiments

**Usage:**
```python
from datetime import datetime, timezone

folio.add_timestamp(
    'experiment_start',
    datetime.now(timezone.utc),
    description='Experiment started'
)

# Later...
folio.add_timestamp('experiment_end', datetime.now(timezone.utc))

# Retrieve
start = folio.get_timestamp('experiment_start')
# → datetime object in UTC
```

**Validation:**
```python
# ERROR: naive datetime not allowed
folio.add_timestamp('time', datetime.now())
# → ValueError: Timestamp must be timezone-aware
```

**Metadata Example:**
```json
{
  "name": "experiment_start",
  "item_type": "timestamp",
  "filename": "experiment_start.timestamp",
  "timestamp": "2025-11-20T10:00:00+00:00",
  "created_at": "2025-11-20T10:00:00+00:00"
}
```

### 6.9 Handler Comparison Table

| Handler | Item Type | File Format | Auto-Detect | Size Limit | Special Features |
|---------|-----------|-------------|-------------|------------|------------------|
| PandasHandler | `included_table` | .parquet | DataFrame | None | Column tracking |
| ReferenceTableHandler | `referenced_table` | N/A | No | N/A | No local copy |
| NumpyHandler | `numpy_array` | .npy | ndarray | None | Binary, efficient |
| JsonHandler | `json_data` | .json | dict/list | ~100MB | Nested structures |
| SklearnHandler | `model` | .joblib | fit/predict | None | Any serializable |
| PyTorchHandler | `pytorch_model` | .pt | nn.Module | None | Enhanced reconstruction |
| ArtifactHandler | `artifact` | Original | file path | None | Preserves extension |
| TimestampHandler | `timestamp` | .timestamp | datetime | N/A | Timezone validation |

## 7. Testing Strategy

### 7.1 Overview

Datafolio uses a **dual-level testing approach** that combines:
- **Unit tests**: Test individual components in isolation (handlers, storage backend, utilities)
- **Integration tests**: Test end-to-end workflows with real file I/O

This approach ensures both component correctness and system reliability.

### 7.2 Current Test Metrics

**Summary (as of latest run):**
- **Total tests**: 417 passing
- **Overall coverage**: 79%
- **Test execution time**: ~2-3 seconds

**Per-module coverage:**

| Module | Coverage | Tests | Notes |
|--------|----------|-------|-------|
| `metadata.py` | **100%** | 25 | Complete MetadataDict coverage |
| `handlers/pytorch_models.py` | **89%** | 22 | All reconstruction paths tested |
| `handlers/pandas_tables.py` | **95%** | 18 | Include and reference patterns |
| `handlers/sklearn_models.py` | **92%** | 15 | Serialization edge cases |
| `storage_backend.py` | **85%** | 32 | Local and cloud operations |
| `folio.py` | **75%** | 140+ | Core API integration tests |
| `registry.py` | **100%** | 12 | Handler registration logic |
| `base.py` | **100%** | 8 | Base handler interface |

### 7.3 Unit Tests

#### 7.3.1 Handler Unit Tests

Each handler has dedicated unit tests covering:
- **`can_handle()` detection logic**
- **`add()` method with various inputs**
- **`get()` method with edge cases**
- **Error handling** (missing files, invalid formats)
- **Metadata generation**

**Example: PyTorchHandler (`test_handlers_pytorch.py`)**

```python
def test_pytorch_reconstruction_strategies(tmp_path):
    """Test all four reconstruction strategies."""
    import torch
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=20):
            super().__init__()
            self.fc = nn.Linear(input_dim, hidden_dim)

        def forward(self, x):
            return self.fc(x)

    folio = DataFolio(tmp_path / "test")
    model = SimpleModel(input_dim=10, hidden_dim=20)

    # Strategy 1: Save with class serialization
    folio.add_pytorch('model1', model, save_class=True)
    loaded1 = folio.get_pytorch('model1')
    assert isinstance(loaded1, SimpleModel)

    # Strategy 2: Provide model_class on retrieval
    folio.add_pytorch('model2', model, init_args={'input_dim': 10, 'hidden_dim': 20})
    loaded2 = folio.get_pytorch('model2', model_class=SimpleModel)
    assert isinstance(loaded2, SimpleModel)

    # Strategy 3: Just get state_dict
    state_dict = folio.get_pytorch('model1', reconstruct=False)
    assert 'fc.weight' in state_dict

    # Strategy 4: Auto-reconstruct from metadata (requires module path)
    # Tested separately with importable classes
```

**Key tests across all handlers:**
- Input validation (type checking, required fields)
- Output format verification (file extensions, metadata structure)
- Error messages are clear and actionable
- Idempotency (can add/get/delete multiple times)

#### 7.3.2 Storage Backend Tests

Tests for `storage_backend.py` cover:
- **Local file operations**: read/write/delete for all formats
- **Cloud storage integration**: CloudFiles compatibility
- **Path handling**: absolute/relative path resolution
- **Error handling**: missing files, permission errors

**Example patterns:**

```python
def test_write_and_read_parquet(tmp_path):
    """Test parquet round-trip."""
    backend = StorageBackend(tmp_path)
    df = pd.DataFrame({'a': [1, 2, 3]})

    # Write
    backend.write_parquet('test.parquet', df)

    # Read
    loaded = backend.read_parquet('test.parquet')
    pd.testing.assert_frame_equal(df, loaded)

def test_cloud_path_support():
    """Test CloudFiles path detection."""
    backend = StorageBackend("s3://bucket/path")
    assert backend.is_cloud_path
    # Actual cloud operations mocked or skipped in CI
```

#### 7.3.3 Base Infrastructure Tests

**MetadataDict (`test_metadata.py`):**
- All dict methods (`__setitem__`, `__delitem__`, `update`, `clear`, `setdefault`)
- Auto-save triggering after modifications
- Timestamp management (`updated_at` auto-updating)
- Recursion prevention (setting `updated_at` doesn't update itself)

**HandlerRegistry (`test_registry.py`):**
- Handler registration order (specificity-based)
- Auto-detection with multiple handlers
- Explicit type registration
- Error cases (no handler found, ambiguous detection)

### 7.4 Integration Tests

Integration tests validate complete workflows using `DataFolio` as the main entry point.

#### 7.4.1 End-to-End Workflows

**File: `test_folio_basic.py`**

Tests cover:
- Bundle creation and reopening
- Adding/retrieving/deleting items
- Metadata persistence across sessions
- Multi-item bundles

**Example workflow test:**

```python
def test_complete_workflow(tmp_path):
    """Test realistic data science workflow."""
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from datetime import datetime, timezone

    # Create bundle
    folio = DataFolio(
        tmp_path / "experiment_001",
        metadata={
            'project': 'classification',
            'experiment_id': 'exp_001'
        }
    )

    # Add timestamp
    folio.add_timestamp('start', datetime.now(timezone.utc))

    # Add raw data
    raw_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'label': [0, 1, 0]
    })
    folio.add_table('raw_data', raw_data)

    # Add processed features
    X = raw_data[['feature1', 'feature2']].values
    folio.add_numpy('features', X)

    # Train and save model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, raw_data['label'])
    folio.add_model('classifier', model, inputs=['features'])

    # Save config
    folio.add_json('config', {
        'n_estimators': 10,
        'random_state': 42
    })

    # Add end timestamp
    folio.add_timestamp('end', datetime.now(timezone.utc))

    # --- Reload and verify ---
    folio2 = DataFolio(tmp_path / "experiment_001")

    # Check all items present
    assert 'raw_data' in folio2.list_contents()['included_tables']
    assert 'features' in folio2.list_contents()['numpy_arrays']
    assert 'classifier' in folio2.list_contents()['models']
    assert 'config' in folio2.list_contents()['json_data']
    assert 'start' in folio2.list_contents()['timestamps']

    # Verify data integrity
    loaded_data = folio2.get_table('raw_data')
    pd.testing.assert_frame_equal(raw_data, loaded_data)

    loaded_model = folio2.get_model('classifier')
    assert loaded_model.n_estimators == 10

    # Verify lineage tracking
    clf_metadata = folio2._items['classifier']
    assert 'features' in clf_metadata['inputs']
```

#### 7.4.2 Multi-Type Bundle Tests

**File: `test_folio_tables.py`, `test_folio_models.py`**

Tests mixing different item types:
- Tables + models (training pipeline)
- Numpy arrays + JSON (experiment configs)
- Artifacts + timestamps (result tracking)

#### 7.4.3 Deletion and Dependency Tests

**File: `test_folio_basic.py::TestDelete`**

Tests for the `delete()` method:
- Single item deletion
- Bulk deletion (multiple items)
- Dependency warnings (deleting items that others depend on)
- Referenced table deletion (metadata only, file preserved)
- Transaction-like validation (all items must exist before deleting any)

```python
def test_delete_with_dependents_warns(tmp_path):
    """Test that deleting item with dependents shows warning."""
    folio = DataFolio(tmp_path / "test")

    df = pd.DataFrame({'a': [1, 2, 3]})
    folio.add_table('raw', df)
    folio.add_table('processed', df, inputs=['raw'])

    # Should warn about 'processed' depending on 'raw'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        folio.delete('raw')

        assert len(w) == 1
        assert 'used by' in str(w[0].message)
        assert 'processed' in str(w[0].message)
```

### 7.5 Testing Guidelines

#### 7.5.1 Running Tests

```bash
# Full suite with coverage
poe test

# Specific test file
uv run pytest tests/test_handlers_pytorch.py -v

# Specific test with output
uv run pytest tests/test_folio_basic.py::test_complete_workflow -v -s

# Coverage report (HTML)
uv run pytest --cov=datafolio --cov-report=html tests/
open htmlcov/index.html
```

#### 7.5.2 Test Organization

Tests mirror the source structure:

```
tests/
├── test_folio_basic.py         # Core DataFolio functionality
├── test_folio_tables.py        # Table-specific operations
├── test_folio_models.py        # Model-specific operations
├── test_metadata.py            # MetadataDict unit tests
├── test_registry.py            # HandlerRegistry unit tests
├── test_storage_backend.py     # StorageBackend unit tests
├── test_handlers_pandas.py     # PandasHandler tests
├── test_handlers_pytorch.py    # PyTorchHandler tests
├── test_handlers_sklearn.py    # SklearnHandler tests
└── conftest.py                 # Shared fixtures
```

#### 7.5.3 Writing New Tests

**For new handlers:**
1. Create `tests/test_handlers_<name>.py`
2. Test `can_handle()` with valid and invalid inputs
3. Test `add()` with edge cases
4. Test `get()` with missing files
5. Test metadata structure
6. Test error messages

**For new features:**
1. Write integration test first (TDD approach)
2. Add unit tests for new methods
3. Check test coverage: `poe test`
4. Ensure coverage stays above 75%

**Example template:**

```python
# tests/test_handlers_custom.py
import pytest
from datafolio.handlers.custom import CustomHandler

@pytest.fixture
def handler():
    """Create handler instance."""
    return CustomHandler()

def test_can_handle_valid_input(handler):
    """Test detection of valid input types."""
    valid_obj = create_valid_object()
    assert handler.can_handle(valid_obj) is True

def test_can_handle_invalid_input(handler):
    """Test rejection of invalid input types."""
    assert handler.can_handle("string") is False
    assert handler.can_handle(123) is False

def test_add_and_get_roundtrip(handler, tmp_path):
    """Test adding and retrieving data."""
    from datafolio import DataFolio

    folio = DataFolio(tmp_path / "test")
    obj = create_valid_object()

    # Add
    metadata = handler.add(folio, 'test', obj)
    assert metadata['item_type'] == 'custom'

    # Retrieve
    loaded = handler.get(folio, 'test')
    assert_objects_equal(obj, loaded)

def test_error_handling(handler, tmp_path):
    """Test error cases with clear messages."""
    folio = DataFolio(tmp_path / "test")

    with pytest.raises(ValueError, match="expected pattern"):
        handler.add(folio, 'test', invalid_object())
```

#### 7.5.4 Coverage Targets

- **New handlers**: Aim for >90% coverage
- **Core modules** (`folio.py`, `storage_backend.py`): >80% coverage
- **Utility modules**: 100% coverage where feasible
- **Overall project**: Maintain >75% coverage

#### 7.5.5 CI/CD Integration

Tests run automatically on:
- Every commit (via pre-commit hooks if configured)
- Pull requests (GitHub Actions)
- Before releases

**Typical CI workflow:**
```yaml
- name: Run tests
  run: |
    uv sync
    uv run pytest --cov=datafolio tests/

- name: Check coverage threshold
  run: |
    uv run pytest --cov=datafolio --cov-fail-under=75 tests/
```

### 7.6 Testing Challenges and Solutions

#### Challenge 1: PyTorch Import Errors in Tests

**Problem**: Early tests failed when simulating PyTorch unavailability because imports happen inside methods.

**Solution**: Mock `sys.modules` and `builtins.__import__` to simulate import failures:

```python
def test_without_pytorch_installed(handler):
    """Test graceful degradation without PyTorch."""
    import sys

    original_torch = sys.modules.get('torch')
    try:
        # Remove torch from sys.modules
        if 'torch' in sys.modules:
            del sys.modules['torch']

        # Mock import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name.startswith('torch'):
                raise ImportError(f"No module named '{name}'")
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            result = handler.can_handle("anything")
            assert result is False
    finally:
        # Restore original state
        if original_torch is not None:
            sys.modules['torch'] = original_torch
```

#### Challenge 2: Cloud Storage Tests

**Problem**: Can't run actual S3/GCS operations in CI without credentials.

**Solution**:
- Mock CloudFiles operations in unit tests
- Use local filesystem for integration tests
- Document manual cloud testing procedures

#### Challenge 3: Large File Testing

**Problem**: Don't want to commit large test files to repository.

**Solution**:
- Generate test data programmatically
- Use small datasets (10-100 rows)
- Test scalability logic without large files

### 7.7 Test Maintenance

**Regular maintenance tasks:**
- Review and update tests when APIs change
- Add regression tests for bug fixes
- Remove obsolete tests after refactoring
- Keep test documentation up-to-date

**Deprecation workflow:**
1. Mark feature as deprecated in code
2. Add deprecation warnings to tests
3. Update tests to use new API
4. Remove deprecated feature and old tests after grace period

## 8. Design Decisions

This section documents key architectural choices, their rationale, and trade-offs considered during development.

### 8.1 Handler-Based Plugin Architecture

**Decision**: Use handler classes with auto-detection instead of type checking in `DataFolio`.

**Rationale**:
- **Extensibility**: New data types can be added without modifying core `DataFolio` class
- **Separation of concerns**: Each handler encapsulates format-specific logic
- **Testing**: Handlers can be tested independently
- **Maintainability**: Changes to one data type don't affect others

**Alternative considered**: If/elif chains in `DataFolio.add()`:

```python
# Rejected approach
def add(self, name, data):
    if isinstance(data, pd.DataFrame):
        # pandas logic here
    elif isinstance(data, np.ndarray):
        # numpy logic here
    elif hasattr(data, 'fit') and hasattr(data, 'predict'):
        # sklearn logic here
    # ... etc
```

**Why rejected**: This approach leads to a bloated `DataFolio` class with hundreds of lines, tight coupling to specific libraries, and difficulty adding custom types.

**Trade-off**: Handler pattern adds indirection (one extra function call), but benefits far outweigh this negligible performance cost.

### 8.2 PyTorch State Dict Storage

**Decision**: Store only `state_dict()` (model weights), not full model objects.

**Rationale**:
- **Compatibility**: State dicts are forward/backward compatible across PyTorch versions
- **Size**: Weights-only files are smaller (no class definition overhead)
- **Flexibility**: Can load weights into different model architectures
- **Best practice**: Aligns with PyTorch official recommendations

**Alternative considered**: Full model serialization with `torch.save(model, path)`.

**Why rejected**:
- Fragile across PyTorch versions
- Includes unnecessary class definition in file
- Can break when model code changes
- Not portable between projects

**Implementation details**:
We provide **four reconstruction strategies** to balance convenience and flexibility:

1. **Dill-serialized class** (`save_class=True`): Bundles class definition with weights
2. **Provided model_class**: User supplies class at load time
3. **Auto-reconstruction**: Import class from metadata (`model_module` + `model_class_name`)
4. **State dict only** (`reconstruct=False`): Return raw weights

This layered approach gives users control while maintaining best practices.

### 8.3 MetadataDict Auto-Save

**Decision**: `MetadataDict` automatically saves to disk on every modification.

**Rationale**:
- **Consistency**: Metadata always reflects current state
- **Simplicity**: No need to remember to call `save()`
- **Crash safety**: Changes persist immediately
- **User expectations**: Dict-like API should "just work"

**Alternative considered**: Explicit `folio.save_metadata()` calls.

**Why rejected**:
- Users forget to call save
- Easy to lose metadata after crashes
- Adds cognitive overhead

**Trade-off**: More disk I/O, but metadata files are tiny (~1-10KB) so performance impact is negligible.

**Implementation detail**: Auto-save skipped during initialization to avoid saving half-constructed state.

### 8.4 Manifest-Based Tracking (manifest.json)

**Decision**: Store all item metadata in a single `manifest.json` file.

**Rationale**:
- **Atomic operations**: Load/save entire manifest in one transaction
- **Easy browsing**: Single file to inspect bundle contents
- **Fast lookups**: All metadata in memory
- **Simplicity**: No complex database or indexing needed

**Alternative considered**: Distributed metadata (one `.meta` file per data file).

**Why rejected**:
- Harder to list all items (need to glob/scan directory)
- More I/O operations (read N files instead of 1)
- Risk of orphaned metadata files
- Harder to maintain consistency

**Trade-off**: Manifest can grow large for bundles with 1000+ items, but this is rare in typical use cases (most bundles have 10-50 items). If needed, future optimization could shard manifests or add indexing.

**Backup safety**: `manifest.json` is written atomically (write to temp file, then rename) to prevent corruption.

### 8.5 Storage Categories (Enum-Based Organization)

**Decision**: Use `StorageCategory` enum to organize files into subdirectories.

```python
class StorageCategory(str, Enum):
    TABLES = "tables"
    MODELS = "models"
    ARTIFACTS = "artifacts"
```

**Rationale**:
- **Type safety**: Can't accidentally use wrong category string
- **IDE support**: Autocomplete and refactoring tools work
- **Clear structure**: Bundle directory is self-documenting
- **Collision prevention**: Tables and artifacts can have same name

**Alternative considered**: Flat directory (all files in bundle root).

**Why rejected**:
- Name collisions likely (e.g., `model.json` could be JSON data or model metadata)
- Hard to browse large bundles
- Less intuitive for users exploring bundle manually

**Trade-off**: Nested structure adds path complexity, but paths are abstracted away by `StorageBackend`.

### 8.6 CloudFiles Integration

**Decision**: Use `cloudfiles` library for cloud storage abstraction.

**Rationale**:
- **Unified API**: Same code for local, S3, GCS, Azure
- **Automatic detection**: `cloudfiles.CloudFiles(path)` handles all path types
- **Maintained**: Active library with good test coverage
- **Transparent**: Works with existing `read_text()`/`write_binary()` patterns

**Alternative considered**: Boto3 for S3 + separate libraries for GCS/Azure.

**Why rejected**:
- Requires separate code paths for each cloud provider
- More dependencies
- More complex error handling
- Harder to test

**Implementation detail**: `StorageBackend` detects cloud paths with simple heuristic (`://` in path) and wraps operations with `CloudFiles`.

### 8.7 Handler Registration Order

**Decision**: Register handlers in specificity order (most specific first).

```python
registry.register_handler(PyTorchHandler, item_type="pytorch_model")
registry.register_handler(SklearnHandler, item_type="model")
# PyTorch checked before sklearn
```

**Rationale**:
- **Correctness**: PyTorch models have `fit`/`predict` (look like sklearn), but need different handling
- **Predictability**: Explicit order prevents flaky auto-detection
- **Debuggability**: Clear why a handler was chosen

**Alternative considered**: Random registration order (rely on handler `can_handle()` logic).

**Why rejected**:
- Ambiguous when multiple handlers match
- Hard to debug false detections
- Order-dependent behavior hidden

**Trade-off**: Developers must remember to register specific handlers before generic ones, but this is documented clearly.

### 8.8 No Backward Compatibility Guarantees (Current Phase)

**Decision**: Breaking changes allowed without deprecation periods.

**Rationale** (per user):
- Project currently single-user
- Rapid iteration more valuable than stability
- Can make bold architectural changes
- Easier to fix design mistakes early

**Future consideration**: Once API stabilizes, add semantic versioning and deprecation warnings.

**Documentation impact**: ARCHITECTURE.md focuses on current state, not migration guides.

### 8.9 Timestamp Management

**Decision**: Auto-populate `created_at` and auto-update `updated_at` timestamps.

**Rationale**:
- **Auditability**: Always know when data was added/modified
- **Reproducibility**: Track experiment timing
- **Debugging**: Identify stale data
- **Zero effort**: Users don't need to remember to timestamp

**Format**: ISO 8601 with UTC timezone (`2025-11-20T10:00:00+00:00`).

**Alternative considered**: Optional timestamps (user-provided).

**Why rejected**:
- Users forget to add timestamps
- Inconsistent formatting
- Can't trust timestamp accuracy

**Trade-off**: Small metadata overhead (~30 bytes per item), but benefits justify cost.

### 8.10 Bundle Directory Structure

**Decision**: Self-contained directory with standard layout.

```
my-experiment/
├── manifest.json          # Central metadata
├── metadata.json          # User metadata
├── README.md             # Auto-generated documentation
├── tables/
├── models/
└── artifacts/
```

**Rationale**:
- **Portability**: Entire experiment in one directory
- **Self-documenting**: README explains how to reload
- **Git-friendly**: Can commit entire bundle (if small)
- **Shareable**: Zip and send to collaborators

**Alternative considered**: Database-backed storage (SQLite, etc.).

**Why rejected**:
- Overkill for typical data science workflows
- Less transparent (can't `ls` to see contents)
- Not as portable
- Harder to debug

**Trade-off**: Directory-based storage less efficient for very large bundles (1000s of items), but typical use cases have 10-100 items.

### 8.11 Type Hints and Runtime Checks

**Decision**: Full type annotations with limited runtime validation.

**Rationale**:
- **IDE support**: Autocomplete and error detection
- **Documentation**: Types serve as inline docs
- **Static analysis**: Mypy/Pyright can catch bugs
- **Performance**: Runtime checks only where needed (e.g., `add()` methods)

**Where we validate**:
- Handler `can_handle()` methods (lightweight type checks)
- Critical paths (file writes, metadata structure)

**Where we trust types**:
- Internal method calls
- After initial validation

**Alternative considered**: Exhaustive runtime validation (e.g., with Pydantic).

**Why rejected**:
- Performance overhead
- Redundant with type checkers
- More code to maintain

**Trade-off**: Trust callers to pass correct types. This is acceptable for a library (vs. public API with untrusted input).

### 8.12 Reference vs. Included Tables

**Decision**: Provide two storage modes for DataFrames.

```python
# Included: Copy to bundle
folio.add_table('data', df)

# Referenced: Track path only
folio.reference_table('big_data', '/data/warehouse/table.parquet')
```

**Rationale**:
- **Flexibility**: Handle both small and huge datasets
- **Performance**: Don't copy multi-GB files unnecessarily
- **Use cases**: Experiments may reference shared corporate data

**Alternative considered**: Always copy data to bundle.

**Why rejected**:
- Wasteful for large datasets
- Slow for network-mounted storage
- Not practical for read-only data sources

**Trade-off**: Referenced tables can become stale or disappear. We accept this risk and document it clearly.

### 8.13 Lineage Tracking with `inputs=` Parameter

**Decision**: Optional lineage tracking via `inputs` parameter.

```python
folio.add_table('raw', df1)
folio.add_table('processed', df2, inputs=['raw'])
folio.add_model('clf', model, inputs=['processed'])
```

**Rationale**:
- **Reproducibility**: Track data provenance
- **Debugging**: Understand dependencies
- **Deletion warnings**: Warn if deleting item with dependents
- **Optional**: Don't force users to track lineage

**Alternative considered**: Automatic lineage tracking (inspect variables, call stacks).

**Why rejected**:
- Fragile (depends on runtime inspection)
- Magic behavior (hard to understand)
- Can't capture manual transformations (notebooks, scripts)

**Trade-off**: Requires manual effort, but explicit is better than implicit.

**Implementation detail**: Lineage stored as simple list in metadata. No validation that inputs exist (flexibility vs. strictness trade-off).

### 8.14 Error Handling Philosophy

**Decision**: Fail fast with clear, actionable error messages.

**Examples**:

```python
# Bad: Silent failure
if not os.path.exists(path):
    return None

# Good: Explicit error
if not os.path.exists(path):
    raise FileNotFoundError(
        f"Item '{name}' not found. File expected at {path}. "
        f"Available items: {list(folio._items.keys())}"
    )
```

**Rationale**:
- **Debuggability**: Users know exactly what went wrong
- **Correctness**: Don't silently ignore errors
- **Discoverability**: Error messages suggest alternatives

**Alternative considered**: Return `None`/`False` for errors.

**Why rejected**:
- Silent failures hard to debug
- Error handling becomes caller's responsibility
- Easy to ignore errors accidentally

**Trade-off**: More verbose error handling code, but worth it for user experience.

### 8.15 Handler Independence

**Decision**: Handlers are independent and don't interact directly.

**Rationale**:
- **Modularity**: Can develop/test handlers in isolation
- **No coupling**: Changing one handler doesn't affect others
- **Clear boundaries**: Each handler owns its file formats

**Consequence**: If handlers need shared logic, it goes in `StorageBackend` or utility modules, not in handlers themselves.

**Example**: All handlers use `StorageBackend.write_json()` for metadata, ensuring consistent behavior.

**Alternative considered**: Allow handlers to call each other.

**Why rejected**:
- Creates dependency graphs
- Harder to understand data flow
- Risk of circular dependencies

**Trade-off**: Some code duplication (e.g., path construction), but independence is more valuable.

## 9. Extensibility

Datafolio is designed to be extended with custom handlers for new data types. This section explains how to create, register, and test custom handlers.

### 9.1 Creating a Custom Handler

All handlers inherit from `BaseHandler` and implement three key methods:

```python
from datafolio.handlers.base import BaseHandler
from datafolio.storage_backend import StorageCategory
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class CustomHandler(BaseHandler):
    """Handler for custom data type."""

    @staticmethod
    def can_handle(data: Any) -> bool:
        """Check if this handler can process the given data.

        This method is called during auto-detection to determine
        which handler should process the data.

        Args:
            data: Data object to check

        Returns:
            True if this handler can process the data
        """
        # Implement type detection logic
        # Example: Check for specific attributes or types
        return isinstance(data, MyCustomType)

    def add(
        self,
        folio: "DataFolio",
        name: str,
        data: Any,
        description: str = "",
        inputs: list[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add custom data to the bundle.

        Args:
            folio: DataFolio instance
            name: Unique identifier for this item
            data: Data object to store
            description: Human-readable description
            inputs: List of item names this depends on
            **kwargs: Handler-specific parameters

        Returns:
            Metadata dict describing the stored item
        """
        # 1. Determine storage location
        filename = f"{name}.custom"
        filepath = folio._storage.get_path(
            filename, category=StorageCategory.ARTIFACTS
        )

        # 2. Serialize and save data
        self._save_data(data, filepath, folio._storage)

        # 3. Generate metadata
        metadata = {
            "name": name,
            "item_type": "custom",
            "filename": filename,
            "description": description,
            "inputs": inputs or [],
            # Add custom fields
            "custom_field": extract_custom_info(data),
        }

        return metadata

    def get(
        self,
        folio: "DataFolio",
        name: str,
        **kwargs,
    ) -> Any:
        """Retrieve custom data from the bundle.

        Args:
            folio: DataFolio instance
            name: Item identifier
            **kwargs: Handler-specific retrieval options

        Returns:
            Reconstructed data object
        """
        # 1. Get metadata
        metadata = folio._items[name]

        # 2. Construct file path
        filepath = folio._storage.get_path(
            metadata["filename"], category=StorageCategory.ARTIFACTS
        )

        # 3. Load and deserialize
        data = self._load_data(filepath, folio._storage)

        return data

    # Helper methods (optional but recommended)
    def _save_data(self, data: Any, filepath: str, storage) -> None:
        """Save data to file."""
        # Implement serialization logic
        serialized = serialize_custom_type(data)
        storage.write_binary(filepath, serialized)

    def _load_data(self, filepath: str, storage) -> Any:
        """Load data from file."""
        # Implement deserialization logic
        serialized = storage.read_binary(filepath)
        return deserialize_custom_type(serialized)
```

### 9.2 Registering Custom Handlers

#### Method 1: Global Registration (Before Creating DataFolio)

```python
from datafolio.registry import registry
from my_handlers import CustomHandler

# Register handler globally
registry.register_handler(CustomHandler, item_type="custom")

# Now all DataFolio instances can auto-detect this type
folio = DataFolio("path/to/bundle")
folio.add(name="data", data=MyCustomType())  # Auto-detects CustomHandler
```

#### Method 2: Per-Instance Registration

```python
from datafolio import DataFolio
from my_handlers import CustomHandler

folio = DataFolio("path/to/bundle")

# Register handler for this instance only
folio.registry.register_handler(CustomHandler, item_type="custom")

# Use it
folio.add(name="data", data=MyCustomType())
```

#### Method 3: Convenience Methods on DataFolio

For cleaner API, add convenience methods to `DataFolio` (requires modifying core):

```python
# In folio.py (requires PR)
class DataFolio:
    def add_custom(
        self,
        name: str,
        data: MyCustomType,
        description: str = "",
        inputs: list[str] | None = None,
        **kwargs,
    ) -> "DataFolio":
        """Add custom data type to bundle."""
        handler = self.registry.get_handler_by_type("custom")
        metadata = handler.add(self, name, data, description, inputs, **kwargs)
        self._items[name] = metadata
        self._save_manifest()
        return self

    def get_custom(self, name: str, **kwargs) -> MyCustomType:
        """Retrieve custom data from bundle."""
        handler = self.registry.get_handler_by_type("custom")
        return handler.get(self, name, **kwargs)
```

### 9.3 Handler Registration Order

**Critical**: Register more specific handlers before generic ones.

```python
from datafolio.registry import registry

# CORRECT: Specific before generic
registry.register_handler(MySpecificHandler, item_type="specific")
registry.register_handler(MyGenericHandler, item_type="generic")

# WRONG: Generic registered first might intercept specific types
registry.register_handler(MyGenericHandler, item_type="generic")
registry.register_handler(MySpecificHandler, item_type="specific")
```

**Example conflict**: If `GenericHandler.can_handle()` returns `True` for types that `SpecificHandler` should handle, register `SpecificHandler` first.

### 9.4 Best Practices for Custom Handlers

#### 1. Make `can_handle()` Fast and Precise

```python
@staticmethod
def can_handle(data: Any) -> bool:
    """Fast type checking."""
    # Good: Quick isinstance check
    if not isinstance(data, MyType):
        return False

    # Good: Check required attributes
    return hasattr(data, 'required_method')

    # Bad: Slow validation
    try:
        validate_complex_structure(data)  # Expensive!
        return True
    except:
        return False
```

#### 2. Include Comprehensive Metadata

```python
def add(self, folio, name, data, description="", inputs=None, **kwargs):
    metadata = {
        "name": name,
        "item_type": "custom",
        "filename": f"{name}.custom",
        "description": description,
        "inputs": inputs or [],

        # Add useful diagnostic info
        "data_shape": data.shape if hasattr(data, 'shape') else None,
        "data_type": type(data).__name__,
        "version": get_library_version(),

        # Track parameters
        "parameters": kwargs,
    }
    return metadata
```

#### 3. Handle Missing Dependencies Gracefully

```python
@staticmethod
def can_handle(data: Any) -> bool:
    """Handle missing libraries gracefully."""
    try:
        import special_library
        return isinstance(data, special_library.SpecialType)
    except ImportError:
        # Library not installed - can't handle this type
        return False
```

#### 4. Validate Inputs Early

```python
def add(self, folio, name, data, **kwargs):
    # Validate data structure
    if not hasattr(data, 'required_attr'):
        raise ValueError(
            f"Data must have 'required_attr'. Got type: {type(data).__name__}"
        )

    # Validate parameters
    if 'mode' in kwargs and kwargs['mode'] not in ['fast', 'accurate']:
        raise ValueError(
            f"Invalid mode: {kwargs['mode']}. Must be 'fast' or 'accurate'."
        )

    # Proceed with storage...
```

#### 5. Use Appropriate Storage Categories

```python
# For machine learning models
StorageCategory.MODELS

# For data tables (CSV, Parquet, etc.)
StorageCategory.TABLES

# For everything else (images, audio, custom formats)
StorageCategory.ARTIFACTS
```

#### 6. Support Reconstruction Options

```python
def get(self, folio, name, *, load_full=True, lazy=False, **kwargs):
    """Flexible retrieval with options."""
    metadata = folio._items[name]
    filepath = folio._storage.get_path(metadata["filename"])

    if lazy:
        # Return lazy loader instead of full data
        return LazyLoader(filepath)

    data = self._load_data(filepath, folio._storage)

    if not load_full:
        # Return metadata only
        return {"metadata": metadata, "data": None}

    return data
```

### 9.5 Complete Example: YAML Handler

Here's a complete custom handler for YAML configuration files:

```python
"""Handler for YAML configuration files."""

from datafolio.handlers.base import BaseHandler
from datafolio.storage_backend import StorageCategory
from typing import Any, Dict, TYPE_CHECKING
import yaml

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class YAMLHandler(BaseHandler):
    """Handler for YAML configuration files."""

    @staticmethod
    def can_handle(data: Any) -> bool:
        """Check if data is a dict (could be YAML-serializable)."""
        # Note: We're conservative here - only handle explicit YAML types
        return isinstance(data, dict) and '__yaml_config__' in data

    def add(
        self,
        folio: "DataFolio",
        name: str,
        data: Dict[str, Any],
        description: str = "",
        inputs: list[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Save YAML configuration to bundle."""
        # Validate
        if not isinstance(data, dict):
            raise ValueError(f"YAML data must be dict, got {type(data).__name__}")

        # Determine path
        filename = f"{name}.yaml"
        filepath = folio._storage.get_path(
            filename, category=StorageCategory.ARTIFACTS
        )

        # Serialize to YAML
        yaml_str = yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        folio._storage.write_text(filepath, yaml_str)

        # Generate metadata
        metadata = {
            "name": name,
            "item_type": "yaml_config",
            "filename": filename,
            "description": description,
            "inputs": inputs or [],
            "num_keys": len(data),
            "top_level_keys": list(data.keys())[:10],  # First 10 keys
        }

        return metadata

    def get(
        self,
        folio: "DataFolio",
        name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Load YAML configuration from bundle."""
        metadata = folio._items[name]
        filepath = folio._storage.get_path(
            metadata["filename"], category=StorageCategory.ARTIFACTS
        )

        # Load YAML
        yaml_str = folio._storage.read_text(filepath)
        data = yaml.safe_load(yaml_str)

        return data


# Usage
from datafolio import DataFolio
from datafolio.registry import registry

# Register handler
registry.register_handler(YAMLHandler, item_type="yaml_config")

# Use it
folio = DataFolio("experiments/config-test")

config = {
    '__yaml_config__': True,  # Tag for handler detection
    'model': {
        'architecture': 'transformer',
        'layers': 12,
        'hidden_size': 768,
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
    }
}

folio.add(name="config", data=config)

# Retrieve
loaded_config = folio.get("config")
print(loaded_config['model']['layers'])  # → 12
```

### 9.6 Testing Custom Handlers

Create comprehensive tests for your custom handler:

```python
# tests/test_handlers_yaml.py
import pytest
from datafolio import DataFolio
from my_handlers import YAMLHandler


@pytest.fixture
def handler():
    """Create handler instance."""
    return YAMLHandler()


def test_can_handle_valid_yaml_dict(handler):
    """Test detection of YAML-tagged dicts."""
    data = {'__yaml_config__': True, 'key': 'value'}
    assert handler.can_handle(data) is True


def test_can_handle_rejects_regular_dicts(handler):
    """Test that regular dicts are not detected."""
    data = {'key': 'value'}  # No __yaml_config__ tag
    assert handler.can_handle(data) is False


def test_add_and_get_roundtrip(handler, tmp_path):
    """Test adding and retrieving YAML data."""
    folio = DataFolio(tmp_path / "test")
    folio.registry.register_handler(YAMLHandler, item_type="yaml_config")

    data = {
        '__yaml_config__': True,
        'nested': {'key': 'value'},
        'list': [1, 2, 3],
    }

    # Add
    folio.add(name='config', data=data)

    # Retrieve
    loaded = folio.get('config')
    assert loaded['nested']['key'] == 'value'
    assert loaded['list'] == [1, 2, 3]


def test_metadata_includes_key_info(handler, tmp_path):
    """Test that metadata captures useful info."""
    folio = DataFolio(tmp_path / "test")
    folio.registry.register_handler(YAMLHandler, item_type="yaml_config")

    data = {
        '__yaml_config__': True,
        'key1': 'value1',
        'key2': 'value2',
    }

    folio.add(name='config', data=data)
    metadata = folio._items['config']

    assert metadata['item_type'] == 'yaml_config'
    assert metadata['num_keys'] == 3  # Including __yaml_config__
    assert 'key1' in metadata['top_level_keys']


def test_invalid_data_raises_error(handler, tmp_path):
    """Test error handling for invalid data."""
    folio = DataFolio(tmp_path / "test")
    folio.registry.register_handler(YAMLHandler, item_type="yaml_config")

    # Try to add non-dict data
    with pytest.raises(ValueError, match="must be dict"):
        folio.add(name='bad', data="not a dict")
```

### 9.7 Integration Points

#### 9.7.1 Custom Storage Backends

To support custom storage systems (databases, APIs, etc.), extend `StorageBackend`:

```python
from datafolio.storage_backend import StorageBackend

class DatabaseStorageBackend(StorageBackend):
    """Store data in a database instead of files."""

    def __init__(self, db_connection_string: str):
        super().__init__(db_connection_string)
        self.db = connect_to_database(db_connection_string)

    def write_parquet(self, filepath: str, df: pd.DataFrame) -> None:
        """Store DataFrame as table in database."""
        table_name = filepath.replace('/', '_').replace('.parquet', '')
        df.to_sql(table_name, self.db, if_exists='replace')

    def read_parquet(self, filepath: str) -> pd.DataFrame:
        """Load DataFrame from database table."""
        table_name = filepath.replace('/', '_').replace('.parquet', '')
        return pd.read_sql_table(table_name, self.db)

    # Implement other required methods...
```

**Note**: This requires modifying `DataFolio.__init__()` to accept custom storage backends.

#### 9.7.2 Custom Metadata Fields

Add custom fields to bundle-level metadata:

```python
folio = DataFolio("path/to/bundle", metadata={
    'project': 'my-project',
    'experiment_id': 'exp_001',
    'git_commit': get_git_commit_hash(),
    'environment': {
        'python_version': sys.version,
        'gpu': torch.cuda.get_device_name(0),
    }
})

# Access later
print(folio.metadata['git_commit'])
```

#### 9.7.3 Hooks and Callbacks

Add hooks to track bundle operations (useful for logging, monitoring):

```python
class LoggingDataFolio(DataFolio):
    """DataFolio with operation logging."""

    def add(self, name, data, **kwargs):
        """Log additions."""
        logger.info(f"Adding item: {name}, type: {type(data).__name__}")
        result = super().add(name, data, **kwargs)
        logger.info(f"Added item: {name}, size: {self._get_item_size(name)}")
        return result

    def get(self, name, **kwargs):
        """Log retrievals."""
        logger.info(f"Retrieving item: {name}")
        return super().get(name, **kwargs)
```

### 9.8 Example Use Cases

#### Use Case 1: Image Handler

```python
class ImageHandler(BaseHandler):
    """Handler for PIL Images."""

    @staticmethod
    def can_handle(data: Any) -> bool:
        try:
            from PIL import Image
            return isinstance(data, Image.Image)
        except ImportError:
            return False

    def add(self, folio, name, data, format="PNG", **kwargs):
        """Save PIL Image."""
        filename = f"{name}.{format.lower()}"
        filepath = folio._storage.get_path(
            filename, category=StorageCategory.ARTIFACTS
        )

        # Save image
        buffer = io.BytesIO()
        data.save(buffer, format=format)
        folio._storage.write_binary(filepath, buffer.getvalue())

        return {
            "name": name,
            "item_type": "image",
            "filename": filename,
            "format": format,
            "size": data.size,
            "mode": data.mode,
        }
```

#### Use Case 2: Audio Handler (librosa)

```python
class AudioHandler(BaseHandler):
    """Handler for audio data."""

    @staticmethod
    def can_handle(data: Any) -> bool:
        """Check for (audio, sample_rate) tuple."""
        return (isinstance(data, tuple) and
                len(data) == 2 and
                isinstance(data[0], np.ndarray))

    def add(self, folio, name, data, **kwargs):
        """Save audio as WAV."""
        import soundfile as sf

        audio, sr = data
        filename = f"{name}.wav"
        filepath = folio._storage.get_path(
            filename, category=StorageCategory.ARTIFACTS
        )

        # Save audio
        temp_path = f"/tmp/{filename}"
        sf.write(temp_path, audio, sr)

        with open(temp_path, 'rb') as f:
            folio._storage.write_binary(filepath, f.read())

        return {
            "name": name,
            "item_type": "audio",
            "filename": filename,
            "sample_rate": sr,
            "duration": len(audio) / sr,
            "channels": audio.shape[1] if audio.ndim > 1 else 1,
        }
```

### 9.9 Contributing Handlers to Datafolio

If you create a useful handler, consider contributing it:

1. **Implement handler** following patterns in `handlers/` directory
2. **Add comprehensive tests** achieving >90% coverage
3. **Document usage** with docstrings and examples
4. **Add to handler registry** in `registry.py`
5. **Update ARCHITECTURE.md** Section 6 with your handler
6. **Submit pull request** with clear description of use case

**Criteria for inclusion**:
- Handles common data type (used by >5% of data scientists)
- Well-tested (>90% coverage)
- Clear documentation
- Follows datafolio patterns and conventions

## 10. Performance & Future Considerations

### 10.1 Current Performance Characteristics

**Typical Operations (as of latest benchmarks):**

| Operation | Small Bundle (10 items) | Medium Bundle (100 items) | Large Bundle (1000 items) |
|-----------|-------------------------|---------------------------|--------------------------|
| Create bundle | ~50ms | ~60ms | ~80ms |
| Add table (1MB parquet) | ~100ms | ~100ms | ~100ms |
| Add model (10MB joblib) | ~200ms | ~200ms | ~200ms |
| Load manifest | ~5ms | ~20ms | ~150ms |
| List contents | <1ms | <1ms | ~10ms |
| Delete item | ~10ms | ~15ms | ~20ms |

**Memory Usage:**
- Manifest stored in memory: ~1KB per item
- Storage backend: ~100KB overhead
- Handlers: ~50KB per handler (lazy loaded)
- Typical bundle: ~1MB in memory (100 items)

**Disk I/O Patterns:**
- Metadata writes: Atomic (write temp, rename)
- Data writes: Streaming for large files
- Cloud operations: Batched when possible

### 10.2 Known Bottlenecks

#### 10.2.1 Manifest Loading for Large Bundles

**Issue**: Loading `manifest.json` scales linearly with item count.

```python
# 1000 items → ~150ms to load manifest
folio = DataFolio("path/to/huge-bundle")  # Reads entire manifest
```

**Impact**: Noticeable for bundles with >500 items.

**Workaround**: Currently acceptable because typical bundles have 10-100 items.

**Future optimization**:
- Index-based lookup (SQLite or similar)
- Lazy manifest loading (load metadata on-demand)
- Sharded manifests (separate file per category)

#### 10.2.2 Auto-Save on Every Metadata Change

**Issue**: `MetadataDict` saves on every modification.

```python
folio.metadata['key1'] = 'value1'  # Saves metadata.json
folio.metadata['key2'] = 'value2'  # Saves metadata.json again
folio.metadata['key3'] = 'value3'  # Saves metadata.json third time
```

**Impact**: 3 disk writes instead of 1.

**Workaround**: Batch metadata updates:

```python
# Better
folio.metadata.update({
    'key1': 'value1',
    'key2': 'value2',
    'key3': 'value3',
})  # Single save
```

**Future optimization**:
- Debounced saves (wait 100ms before writing)
- Transaction contexts (`with folio.metadata.batch():`)
- Optional manual save mode

#### 10.2.3 Cloud Storage Latency

**Issue**: Cloud operations have inherent latency (50-200ms per file).

```python
# S3 bundle - each operation is a network round-trip
folio = DataFolio("s3://bucket/bundle")
folio.add_table('data', df)  # ~150ms S3 write
```

**Impact**: 5-10x slower than local filesystem.

**Workaround**:
- Use local staging, then sync to cloud
- Batch operations when possible

**Future optimization**:
- Multipart uploads for large files
- Parallel writes for multiple items
- Caching layer

### 10.3 Optimization Opportunities

#### 10.3.1 Lazy Loading

**Current**: All handlers loaded at import time.

**Optimization**: Load handlers on first use.

```python
# Future implementation
class HandlerRegistry:
    def __init__(self):
        self._handler_classes = {}  # Store class references
        self._instances = {}  # Lazy instantiation

    def get_handler(self, data):
        for handler_cls in self._handler_classes:
            if handler_cls not in self._instances:
                self._instances[handler_cls] = handler_cls()  # Lazy load
            handler = self._instances[handler_cls]
            if handler.can_handle(data):
                return handler
```

**Benefit**: Faster startup, lower memory for unused handlers.

#### 10.3.2 Parallel Item Operations

**Current**: Sequential adds/gets.

**Optimization**: Parallel operations for independent items.

```python
# Future API
folio.add_batch([
    ('table1', df1),
    ('table2', df2),
    ('model', clf),
], parallel=True)  # Write all in parallel
```

**Benefit**: 3-5x speedup for large batches.

**Challenge**: Ensuring manifest consistency.

#### 10.3.3 Compression

**Current**: No compression for most formats (parquet self-compresses).

**Optimization**: Optional compression for large files.

```python
folio.add_numpy('embeddings', large_array, compress=True)
# → Stores as embeddings.npy.gz (50% size reduction)
```

**Benefit**: 30-70% disk space savings for large arrays.

**Trade-off**: 10-20% slower write/read times.

#### 10.3.4 Incremental Manifest Updates

**Current**: Entire manifest written on every change.

**Optimization**: Append-only changelog with periodic compaction.

```python
# manifest.json (immutable)
# + manifest.log (append-only changes)
# Periodically compact log into manifest
```

**Benefit**: Faster writes for large bundles (append vs full rewrite).

**Challenge**: More complex loading logic.

### 10.4 Scalability Considerations

#### 10.4.1 Bundle Size Limits

**Current practical limits:**
- Items per bundle: ~1000 (manifest loading becomes slow)
- Bundle size: No hard limit (filesystem/cloud dependent)
- Largest tested: 50GB bundle with 500 items

**Scaling strategies:**
- Split experiments into multiple bundles
- Use referenced tables for huge datasets
- Archive old bundles to cold storage

#### 10.4.2 Concurrent Access

**Current**: Single-writer, multiple-reader safe (filesystem atomicity).

**Limitations**:
- No locking mechanism
- Concurrent writers may corrupt manifest
- Last-write-wins for metadata

**Future considerations**:
- File locking for write operations
- Optimistic concurrency (version tags)
- Read-only mode for safe concurrent reads

#### 10.4.3 Cloud Storage at Scale

**Considerations:**
- S3/GCS costs: $0.023/GB-month storage, $0.09/GB egress
- Large bundles (100GB+) expensive to download
- Multipart uploads required for files >5GB

**Best practices:**
- Keep bundles under 10GB for cloud
- Use cloud-native formats (parquet on S3)
- Consider data warehouses for massive datasets

### 10.5 Future Features

#### 10.5.1 Planned Features (Next 6 Months)

**1. Versioning and Snapshots**

```python
# Save current state as version
folio.create_snapshot('v1.0', description='Final model')

# List snapshots
folio.list_snapshots()
# → [{'name': 'v1.0', 'timestamp': '...', 'items': 15}]

# Load specific version
folio_v1 = DataFolio.load_snapshot('path/to/bundle', snapshot='v1.0')
```

**Use case**: Track experiment iterations without duplicating data.

**2. Bundle Comparison**

```python
# Compare two bundles
diff = DataFolio.compare(folio1, folio2)
# → {
#   'added': ['new_model'],
#   'removed': ['old_data'],
#   'modified': ['config'],
#   'metadata_changes': {...}
# }
```

**Use case**: Understand what changed between experiments.

**3. Provenance Graphs**

```python
# Visualize data lineage
folio.plot_lineage(output='lineage.png')
# → Creates graph showing item dependencies
```

**Use case**: Understand data flow in complex pipelines.

**4. Bundle Merge**

```python
# Merge two bundles
merged = DataFolio.merge(
    [bundle1, bundle2],
    output='merged-bundle',
    conflict_resolution='rename'  # or 'overwrite', 'error'
)
```

**Use case**: Combine results from parallel experiments.

#### 10.5.2 Under Consideration

**1. Streaming APIs**

```python
# Stream large DataFrames in chunks
for chunk in folio.stream_table('huge_data', chunk_size=10000):
    process(chunk)
```

**2. Encryption**

```python
folio = DataFolio('path', encryption_key=key)
# All files encrypted at rest
```

**3. Compression Levels**

```python
folio.add_table('data', df, compression='high')  # Slower but smaller
folio.add_table('data', df, compression='fast')  # Faster but larger
```

**4. Remote Bundles (Read-Only)**

```python
# Open remote bundle without downloading
folio = DataFolio('https://data.example.com/bundle', mode='remote')
df = folio.get_table('data')  # Downloads on demand
```

**5. Metadata Search**

```python
# Find items by metadata
results = folio.search(lambda m: m.get('type') == 'model' and m['accuracy'] > 0.9)
```

### 10.6 Breaking Changes to Anticipate

When API stabilizes (v1.0), these changes may require migration:

**1. Manifest Format Evolution**

Future versions may add fields to `manifest.json`. Plan includes:
- Forward compatibility (ignore unknown fields)
- Backward compatibility (provide defaults for missing fields)
- Migration tools to upgrade old bundles

**2. Handler Interface Changes**

Current `BaseHandler` may gain new methods:
- `validate()` - Pre-storage validation
- `describe()` - Rich metadata generation
- `estimate_size()` - Predict storage needs

**3. Storage Backend Abstraction**

May extract `StorageBackend` as pluggable interface to support:
- Databases (PostgreSQL, MongoDB)
- Object stores (MinIO, Azure Blob)
- Version control (DVC, Git LFS)

### 10.7 Performance Best Practices

#### For Users

**1. Batch Operations**

```python
# Bad: Multiple saves
folio.add_table('data1', df1)
folio.add_table('data2', df2)
folio.add_table('data3', df3)

# Good: Batch add (future API)
folio.add_batch([
    ('data1', df1),
    ('data2', df2),
    ('data3', df3),
])
```

**2. Use Reference Tables for Large Data**

```python
# Bad: Copy 10GB file into bundle
folio.add_table('huge', huge_df)  # Copies entire file

# Good: Reference external file
folio.reference_table('huge', '/data/warehouse/huge.parquet')
```

**3. Preallocate Metadata**

```python
# Good: Set all metadata at once
folio = DataFolio('path', metadata={
    'experiment': 'exp_001',
    'date': today,
    'parameters': {...},
})

# Avoid: Many individual updates
folio.metadata['experiment'] = 'exp_001'  # Save
folio.metadata['date'] = today            # Save again
folio.metadata['parameters'] = {...}      # Save third time
```

**4. Close Bundles When Done**

```python
# Use context manager (future API)
with DataFolio('path') as folio:
    folio.add_table('data', df)
    folio.add_model('model', clf)
# Automatically finalizes bundle, releases resources
```

#### For Developers

**1. Optimize `can_handle()` Methods**

```python
# Fast path
if not isinstance(data, ExpectedType):
    return False

# Avoid expensive checks in can_handle()
# Move to add() method instead
```

**2. Stream Large Files**

```python
def add(self, folio, name, data, **kwargs):
    # Don't load entire file into memory
    with open(source, 'rb') as src:
        with folio._storage.open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst, length=1024*1024)  # 1MB chunks
```

**3. Cache Expensive Computations**

```python
def add(self, folio, name, data, **kwargs):
    # Cache model architecture info
    if not hasattr(data, '_architecture_str'):
        data._architecture_str = compute_architecture(data)  # Expensive

    metadata = {
        'architecture': data._architecture_str,
        # ...
    }
```

### 10.8 Roadmap Summary

**v0.5 (Current)**: Handler-based architecture, 8 built-in handlers, 79% test coverage

**v0.6 (Next 3 months)**:
- Versioning and snapshots
- Bundle comparison tools
- Performance optimizations (lazy loading)

**v0.7 (Next 6 months)**:
- Provenance graph visualization
- Streaming APIs for large data
- Enhanced cloud storage support

**v1.0 (API Stability)**:
- Stable API with semantic versioning
- Comprehensive documentation
- Production-ready performance
- Backward compatibility guarantees

**v1.x (Long-term)**:
- Encryption at rest
- Concurrent access with locking
- Plugin marketplace
- Enterprise features (audit logs, access control)

---

## Appendices

### A. Module Dependency Graph
[TO BE WRITTEN]

### B. File Line Counts
[TO BE WRITTEN]

### C. Related Documentation
- [EXTENDING.md](EXTENDING.md) - Guide for adding custom handlers
- [CLAUDE.md](CLAUDE.md) - Development workflow and testing guidelines
