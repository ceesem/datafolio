---
title: DataFolio API Reference
---

# DataFolio Class - Complete API Reference

This page provides a comprehensive reference of all methods available on the `DataFolio` class, organized by functionality.

## Creating a DataFolio

::: datafolio.DataFolio.__init__
    options:
        show_source: false
        heading_level: 3

---

## Core Item API

Two methods cover reading and writing every data type — the object's type
selects the storage format on write, and the stored type determines what you
get back on read.

::: datafolio.DataFolio.add
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.item_path
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.item_info
    options:
        show_source: false
        heading_level: 3

---

## Models

The one explicit typed pair: `add_model` stores *any* picklable object (not
just auto-detected sklearn estimators), and `get_model` loads it.

::: datafolio.DataFolio.add_model
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get_model
    options:
        show_source: false
        heading_level: 3

---

## Files

::: datafolio.DataFolio.add_file
    options:
        show_source: false
        heading_level: 3

---

## External Table References

Link tables that live outside the folio (S3, GCS, local paths) without
copying them. Creating a reference is offline; `inspect_table` opts into I/O.

::: datafolio.DataFolio.reference_table
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.inspect_table
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.scan_table
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.mutable_references
    options:
        show_source: false
        heading_level: 3

---

## Inspecting Items

::: datafolio.DataFolio.list_contents
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.describe
    options:
        show_source: false
        heading_level: 3

---

## Managing Items

### Updating Item Metadata

::: datafolio.DataFolio.update_item
    options:
        show_source: false
        heading_level: 3

### Deleting Items

::: datafolio.DataFolio.delete
    options:
        show_source: false
        heading_level: 3

### Archiving Items

::: datafolio.DataFolio.archive
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.unarchive
    options:
        show_source: false
        heading_level: 3

### Copying Bundles

::: datafolio.DataFolio.copy
    options:
        show_source: false
        heading_level: 3

### Validation

::: datafolio.DataFolio.validate
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.is_valid
    options:
        show_source: false
        heading_level: 3

---

## Lineage and Dependencies

Methods for working with lineage tracking.

::: datafolio.DataFolio.get_inputs
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get_dependents
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get_lineage_graph
    options:
        show_source: false
        heading_level: 3

---

## Snapshots

Methods for working with snapshots (read-only copies).

::: datafolio.DataFolio.create_snapshot
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.list_snapshots
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.delete_snapshot
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.load_snapshot
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get_snapshot
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get_snapshot_info
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.compare_snapshots
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.diff_from_snapshot
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.restore_snapshot
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.export_snapshot
    options:
        show_source: false
        heading_level: 3

---

## Bundle Management

Methods for managing the DataFolio bundle itself.

::: datafolio.DataFolio.refresh
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.pinned
    options:
        show_source: false
        heading_level: 3

---

## Properties

Useful properties for accessing bundle information and items.

### Core Properties

#### `path`
The path to the DataFolio bundle.

```python
print(folio.path)  # e.g., 'gs://my-bucket/my-bundle' or '/local/path/bundle'
```

#### `metadata`
Bundle-level metadata dictionary.

```python
print(folio.metadata)  # e.g., {'project': 'analysis', 'version': '1.0'}
```

### Item Lists

#### `tables`
List of all table names in the bundle.

```python
print(folio.tables)  # e.g., ['results', 'metadata', 'analysis']
```

#### `models`
List of all model names in the bundle.

```python
print(folio.models)  # e.g., ['classifier', 'regressor']
```

#### `artifacts`
List of all artifact names in the bundle.

```python
print(folio.artifacts)  # e.g., ['config.yaml', 'results.png']
```

### Data Accessor

#### `data`
Accessor for convenient data retrieval with autocomplete support.

```python
df = folio.data.my_table.content  # Equivalent to folio.get('my_table')
model = folio.data.my_model.content  # Equivalent to folio.get_model('my_model')
```

### Status Properties

#### `read_only`
Whether the bundle is in read-only mode.

```python
print(folio.read_only)  # True or False
```

#### `in_snapshot_mode`
Whether the bundle was loaded from a snapshot.

```python
print(folio.in_snapshot_mode)  # True or False
```

#### `loaded_snapshot`
Name of the snapshot this bundle was loaded from (if any).

```python
print(folio.loaded_snapshot)  # e.g., 'v1.0' or None
```

---

## Method Categories Summary

| Category | Methods |
|----------|---------|
| **Core item API** | `add()`, `get()`, `item_path()`, `item_info()` |
| **Models** | `add_model()`, `get_model()` |
| **Files** | `add_file()` |
| **External references** | `reference_table()`, `inspect_table()`, `scan_table()`, `mutable_references()` |
| **Inspecting** | `list_contents()`, `describe()` |
| **Managing Items** | `update_item()`, `delete()`, `archive()`, `unarchive()`, `copy()`, `validate()`, `is_valid()` |
| **Lineage** | `get_inputs()`, `get_dependents()`, `get_lineage_graph()` |
| **Snapshots** | `create_snapshot()`, `list_snapshots()`, `delete_snapshot()`, `load_snapshot()`, `get_snapshot()`, `get_snapshot_info()`, `compare_snapshots()`, `diff_from_snapshot()`, `restore_snapshot()`, `export_snapshot()` |
| **Bundle Management** | `refresh()`, `pinned()`, `batch()` |

---

## Quick Examples

### Basic Usage
```python
import datafolio
import pandas as pd

# Create a new DataFolio
folio = datafolio.DataFolio('my_analysis')

# Add anything — the type picks the format
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
folio.add('results', df, description='Experimental results')
folio.add('config', {'lr': 0.01})
folio.add('accuracy', 0.95)

# Retrieve by name — the stored type determines what comes back
df_loaded = folio.get('results')
config = folio.get('config')

# List contents
print(folio.list_contents())
print(folio.tables)  # Property access

# Use data accessor with autocomplete
df_via_accessor = folio.data.results.content
```

### With Snapshots
```python
# Create a read-only snapshot
folio.create_snapshot('v1.0', description='Release 1.0')

# Load a snapshot (read-only mode)
folio_snapshot = datafolio.DataFolio.load_snapshot('my_analysis', 'v1.0')

# List all snapshots
snapshots = folio.list_snapshots()
for snap in snapshots:
    print(f"{snap['name']}: {snap['description']}")
```

### Lineage Tracking
```python
# Add data with lineage
folio.add('raw_data', raw_df)
folio.add('processed_data', processed_df, inputs=['raw_data'])
folio.add_model('trained_model', model, inputs=['processed_data'])

# Query lineage
inputs = folio.get_inputs('trained_model')  # ['processed_data']
dependents = folio.get_dependents('raw_data')  # ['processed_data']

# Get full lineage graph
graph = folio.get_lineage_graph()
print(graph)  # Shows dependency relationships
```

### Sharing Paths with Collaborators
```python
# For a cloud-hosted folio, get the direct path to any item
folio = datafolio.DataFolio('s3://my-bucket/experiments/run-42')

# One path getter for every item type:
path = folio.item_path('results')
# → 's3://my-bucket/experiments/run-42/tables/results--r2.parquet'

path = folio.item_path('classifier')
# → 's3://my-bucket/experiments/run-42/models/classifier--r3.joblib'

# External references return the external path:
path = folio.item_path('raw_data')      # → 's3://data-lake/raw.parquet' 

# Share with a colleague who doesn't use datafolio:
# import pandas as pd; pd.read_parquet('s3://my-bucket/.../results--r2.parquet')

# Or browse all paths at once with describe()
folio.describe(show_paths=True)
# Tables (2):
#   • raw_data (reference): Input dataset
#     ↳ path: s3://data-lake/raw.parquet
#   • results: Model results
#     ↳ path: s3://my-bucket/experiments/run-42/tables/results--r2.parquet
# Models (1):
#   • classifier: Trained model
#     ↳ path: s3://my-bucket/experiments/run-42/models/classifier--r3.joblib
```
