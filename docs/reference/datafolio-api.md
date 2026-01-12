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

## Adding Data

Methods for adding different types of data to a DataFolio.

### Tables (DataFrames)

::: datafolio.DataFolio.add_table
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.reference_table
    options:
        show_source: false
        heading_level: 3

### Arrays

::: datafolio.DataFolio.add_numpy
    options:
        show_source: false
        heading_level: 3

### JSON Data

::: datafolio.DataFolio.add_json
    options:
        show_source: false
        heading_level: 3

### Timestamps

::: datafolio.DataFolio.add_timestamp
    options:
        show_source: false
        heading_level: 3

### Generic Data

::: datafolio.DataFolio.add_data
    options:
        show_source: false
        heading_level: 3

---

## Adding Models

Methods for saving machine learning models.

### Scikit-learn Models

::: datafolio.DataFolio.add_sklearn
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.add_model
    options:
        show_source: false
        heading_level: 3

---

## Adding Artifacts

Methods for adding arbitrary files and artifacts.

::: datafolio.DataFolio.add_artifact
    options:
        show_source: false
        heading_level: 3

---

## Retrieving Data

Methods for loading data from a DataFolio.

### Tables (DataFrames)

::: datafolio.DataFolio.get_table
    options:
        show_source: false
        heading_level: 3

### Arrays

::: datafolio.DataFolio.get_numpy
    options:
        show_source: false
        heading_level: 3

### JSON Data

::: datafolio.DataFolio.get_json
    options:
        show_source: false
        heading_level: 3

### Timestamps

::: datafolio.DataFolio.get_timestamp
    options:
        show_source: false
        heading_level: 3

### Generic Data

::: datafolio.DataFolio.get_data
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get_data_path
    options:
        show_source: false
        heading_level: 3

---

## Retrieving Models

Methods for loading machine learning models.

### Scikit-learn Models

::: datafolio.DataFolio.get_sklearn
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get_model
    options:
        show_source: false
        heading_level: 3

---

## Retrieving Artifacts

::: datafolio.DataFolio.get_artifact_path
    options:
        show_source: false
        heading_level: 3

---

## Inspecting Items

Methods for getting information about items.

::: datafolio.DataFolio.list_contents
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get_table_info
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get_model_info
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.get_artifact_info
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.describe
    options:
        show_source: false
        heading_level: 3

---

## Managing Items

### Deleting Items

::: datafolio.DataFolio.delete
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

## Caching

Methods for managing the local cache (for remote bundles).

::: datafolio.DataFolio.cache_status
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.clear_cache
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.invalidate_cache
    options:
        show_source: false
        heading_level: 3

::: datafolio.DataFolio.refresh_cache
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

#### `items`
Dictionary of all items in the bundle with their metadata.

```python
print(folio.items)  # e.g., {'table1': {...}, 'model1': {...}}
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
df = folio.data.my_table  # Equivalent to folio.get_table('my_table')
model = folio.data.my_model  # Equivalent to folio.get_model('my_model')
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
| **Adding Data** | `add_table()`, `add_numpy()`, `add_json()`, `add_timestamp()`, `add_data()`, `reference_table()` |
| **Adding Models** | `add_sklearn()`, `add_model()` |
| **Adding Artifacts** | `add_artifact()` |
| **Retrieving Data** | `get_table()`, `get_numpy()`, `get_json()`, `get_timestamp()`, `get_data()`, `get_data_path()` |
| **Retrieving Models** | `get_sklearn()`, `get_model()` |
| **Retrieving Artifacts** | `get_artifact_path()` |
| **Inspecting Items** | `list_contents()`, `get_table_info()`, `get_model_info()`, `get_artifact_info()`, `describe()` |
| **Managing Items** | `delete()`, `copy()`, `validate()`, `is_valid()` |
| **Lineage** | `get_inputs()`, `get_dependents()`, `get_lineage_graph()` |
| **Snapshots** | `create_snapshot()`, `list_snapshots()`, `delete_snapshot()`, `load_snapshot()`, `get_snapshot()`, `get_snapshot_info()`, `compare_snapshots()`, `diff_from_snapshot()`, `restore_snapshot()`, `export_snapshot()` |
| **Caching** | `cache_status()`, `clear_cache()`, `invalidate_cache()`, `refresh_cache()` |
| **Bundle Management** | `refresh()` |

---

## Quick Examples

### Basic Usage
```python
import datafolio
import pandas as pd

# Create a new DataFolio
folio = datafolio.DataFolio('my_analysis')

# Add data
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
folio.add_table('results', df, description='Experimental results')

# Retrieve data
df_loaded = folio.get_table('results')

# List contents
print(folio.list_contents())
print(folio.tables)  # Property access

# Use data accessor with autocomplete
df_via_accessor = folio.data.results
```

### With Caching
```python
# Enable caching for remote bundles
folio = datafolio.DataFolio(
    'gs://my-bucket/my-bundle',
    cache_enabled=True,
    cache_dir='/tmp/my-cache'
)

# First access downloads and caches
df = folio.get_table('large_table')  # Downloads from cloud

# Second access uses cache (much faster!)
df = folio.get_table('large_table')  # Reads from local cache

# Check cache statistics
status = folio.cache_status()
print(f"Cache hits: {status['cache_hits']}")
print(f"Cache misses: {status['cache_misses']}")
```

### With Snapshots
```python
# Create a read-only snapshot
folio.create_snapshot('v1.0', description='Release 1.0')

# Load a snapshot (read-only mode)
folio_snapshot = datafolio.DataFolio.load_snapshot('v1.0')

# List all snapshots
snapshots = folio.list_snapshots()
for snap in snapshots:
    print(f"{snap['name']}: {snap['description']}")
```

### Lineage Tracking
```python
# Add data with lineage
folio.add_table('raw_data', raw_df)
folio.add_table('processed_data', processed_df, inputs=['raw_data'])
folio.add_model('trained_model', model, inputs=['processed_data'])

# Query lineage
inputs = folio.get_inputs('trained_model')  # ['processed_data']
dependents = folio.get_dependents('raw_data')  # ['processed_data']

# Get full lineage graph
graph = folio.get_lineage_graph()
print(graph)  # Shows dependency relationships
```
