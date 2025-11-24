# DataFolio: Parquet Table Enhancements

**Phase 1: Core Table Features for Memory-Efficient Data Processing**

**Status:** Planning Document
**Target:** Academic researchers with large datasets and limited compute resources
**Philosophy:** Provide core primitives + integration points, not a full query engine

---

## Table of Contents

1. [Motivation & Use Cases](#motivation--use-cases)
2. [Design Philosophy](#design-philosophy)
3. [Feature Overview](#feature-overview)
4. [Architecture Design](#architecture-design)
5. [API Specification](#api-specification)
6. [Integration Patterns](#integration-patterns)
7. [Implementation Plan](#implementation-plan)
8. [Testing Strategy](#testing-strategy)
9. [Documentation Updates](#documentation-updates)
10. [Future Work (Phase 2)](#future-work-phase-2)

---

## Motivation & Use Cases

### The Problem

Academic researchers frequently encounter these challenges:

1. **Large datasets, limited RAM**: 50GB genomics data, 8GB laptop
2. **Memory management pain**: Constant `del df; gc.collect()` in notebooks
3. **Workflow inefficiency**: Load → process → save → load again → repeat
4. **Tool fragmentation**: Pandas/Polars/DuckDB/Ibis don't integrate well with versioning
5. **Reproducibility**: Need to track which subset/transformation was used

### Real-World Scenarios

**Genomics Researcher:**
```python
# Current pain: Can't load 50GB variant file
df = pd.read_parquet('variants.parquet')  # ❌ OOM!

# Desired: Process in chunks, save filtered results
for chunk in folio.iter_table('variants', chunk_size=100000):
    filtered = process_variants(chunk)
    folio.append_table('significant_variants', filtered)
```

**Neuroscience Lab:**
```python
# Current pain: Complex SQL needed, but data versioned in DataFolio
# Have to export to database, losing version tracking

# Desired: Query versioned data directly
results = folio.query_table('neural_recordings',
    "SELECT neuron_id, AVG(firing_rate) as mean_rate
     FROM table
     WHERE stimulus = 'visual'
     GROUP BY neuron_id")
```

**Clinical Trial Analyst:**
```python
# Current pain: Pandas for small data, need Polars for speed

# Desired: Use modern tools while maintaining DataFolio versioning
import polars as pl
df_polars = pl.read_parquet(...)
folio.add_table('results', df_polars)  # Auto-detect and store

# Get back in same format
df = folio.get_table('results')  # Returns polars.DataFrame
```

---

## Design Philosophy

### Core Principles

1. **DataFolio's job:** Versioning, organization, metadata tracking
2. **Not DataFolio's job:** Complex query optimization, distributed computing
3. **Approach:** Provide primitives + integration points for power tools

### What We're Adding

✅ **Polars support** - Modern dataframe library (faster than pandas)
✅ **Basic SQL queries** - Via DuckDB for common filtering/aggregation
✅ **Chunked iteration** - Memory-efficient processing primitive
✅ **Path accessors** - Integration with Ibis, DuckDB, Polars
✅ **Table metadata** - Inspect without loading

### What We're NOT Adding

❌ **Query abstraction layer** - That's Ibis's job
❌ **Distributed computing** - That's Dask/Ray/Spark's job
❌ **Backend abstraction** - DataFolio is file-based, stay simple
❌ **Map-reduce helpers** - Users can build with primitives

### Integration Strategy

**Pattern:** Expose internals, let users compose tools

```python
# DataFolio provides the path
path = folio.get_table_path('large_dataset')

# User chooses their tool
import ibis
con = ibis.duckdb.connect()
table = con.read_parquet(path)

# Full Ibis power, DataFolio stays simple
result = (table
    .filter(table.value > 100)
    .group_by('category')
    .aggregate(mean=table.value.mean()))
```

---

## Feature Overview

### Feature 1: Polars Support

**What:** First-class support for Polars DataFrames alongside pandas

**Why:** Polars is faster, more memory-efficient, and increasingly popular in research

**How:** New `PolarsHandler` that auto-detects `polars.DataFrame`

```python
import polars as pl

# Add polars DataFrame (auto-detected)
df_polars = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
folio.add_table('data', df_polars)

# Get back as polars (auto-detected from metadata)
df = folio.get_table('data')  # Returns polars.DataFrame
isinstance(df, pl.DataFrame)  # True

# Or force return type
df_pandas = folio.get_table('data', return_type='pandas')
df_polars = folio.get_table('data', return_type='polars')
```

**Implementation notes:**
- Storage format: Still Parquet (compatible with both pandas/polars)
- Metadata: Add `dataframe_library` field ('pandas' or 'polars')
- Auto-detection: Check `isinstance(data, pl.DataFrame)` in handler
- Separate handler: `PolarsHandler` (follows existing pattern)

---

### Feature 2: DuckDB SQL Queries

**What:** Execute SQL queries on tables without loading full data

**Why:** SQL is familiar, DuckDB is optimized for Parquet, enables filtering/aggregation

**How:** Add `query_table()` method that uses DuckDB under the hood

```python
# Simple filter
df = folio.query_table('large_dataset',
    "SELECT * FROM table WHERE value > 100 AND category = 'A'")

# Aggregation
summary = folio.query_table('measurements',
    """SELECT
        subject_id,
        AVG(score) as mean_score,
        COUNT(*) as n_trials
    FROM table
    GROUP BY subject_id
    HAVING n_trials >= 10""")

# Return as polars
df = folio.query_table('data',
    "SELECT * FROM table WHERE x > 5",
    return_type='polars')
```

**Implementation notes:**
- Use DuckDB's `read_parquet()` for efficient queries
- Register table as `table` in DuckDB (simple convention)
- Support both pandas and polars return types
- Lazy connection: Create DuckDB connection on first use
- Optional dependency: DuckDB is optional, error if not installed

---

### Feature 3: Chunked Iteration

**What:** Iterate through large tables in chunks without loading into memory

**Why:** Process datasets larger than RAM, avoid OOM errors

**How:** Add `iter_table()` generator method

```python
# Basic iteration
for chunk in folio.iter_table('large_dataset', chunk_size=10000):
    # chunk is pandas/polars DataFrame with 10000 rows
    results = process_chunk(chunk)
    folio.append_table('results', results)

# With column selection (only load needed columns)
for chunk in folio.iter_table('data',
                               chunk_size=10000,
                               columns=['id', 'value']):
    process(chunk)

# With filtering (pushdown to DuckDB)
for chunk in folio.iter_table('data',
                               chunk_size=10000,
                               where="p_value < 0.05"):
    process(chunk)

# Return as polars
for chunk in folio.iter_table('data',
                               chunk_size=10000,
                               return_type='polars'):
    process(chunk)
```

**Implementation notes:**
- Use DuckDB for efficient chunked reads (OFFSET/LIMIT)
- Respect original dataframe library (return pandas/polars based on storage)
- Support column projection and filtering (pushdown to DuckDB)
- Memory-efficient: Only one chunk in memory at a time

---

### Feature 4: Path Accessor Methods

**What:** Get file paths to tables for integration with other tools

**Why:** Enable users to use Ibis, DuckDB, Polars directly without DataFolio reimplementing features

**How:** Add simple path accessor methods

```python
# Get path to single table
path = folio.get_table_path('results')
# Returns: '/path/to/bundle/tables/results.parquet'

# Use with any tool
import ibis
con = ibis.duckdb.connect()
table = con.read_parquet(path)

# Get all table paths
paths = folio.get_all_table_paths()
# Returns: {'results': '/path/to/tables/results.parquet',
#           'features': '/path/to/tables/features.parquet'}

# Register multiple tables with DuckDB
import duckdb
con = duckdb.connect()
for name, path in paths.items():
    con.register(path, table_name=name)

# Now query across tables
result = con.sql("""
    SELECT e.*, s.age
    FROM experiments e
    JOIN subjects s ON e.subject_id = s.subject_id
""")
```

**Implementation notes:**
- Simple path resolution from items.json
- Works with both included and referenced tables
- Absolute paths for portability
- No tool-specific logic (pure path exposure)

---

### Feature 5: Table Metadata Queries

**What:** Get table information without loading data

**Why:** Quick exploration, understand data before committing to load

**How:** Add `table_info()` method that reads Parquet metadata

```python
# Get comprehensive info
info = folio.table_info('large_dataset')
# Returns: {
#     'rows': 1000000,
#     'columns': 50,
#     'size_mb': 245.3,
#     'column_names': ['id', 'value', 'category', ...],
#     'dtypes': {'id': 'int64', 'value': 'float64', ...},
#     'parquet_schema': <pyarrow.Schema>,
#     'created_at': '2024-01-15T10:30:00Z'
# }

# Quick preview (first N rows)
preview = folio.preview_table('data', n=10)
# Returns: DataFrame with first 10 rows

# Check if table is large
info = folio.table_info('data')
if info['size_mb'] > 1000:
    # Use chunked iteration
    for chunk in folio.iter_table('data'):
        process(chunk)
else:
    # Small enough to load
    df = folio.get_table('data')
```

**Implementation notes:**
- Use PyArrow for fast metadata reading (no data scan)
- Cache metadata in items.json for repeated access
- Preview uses DuckDB LIMIT for efficiency
- Works with both pandas and polars tables

---

## Architecture Design

### Handler Structure

Following DataFolio's existing handler pattern:

```
handlers/
├── tables.py
│   ├── PandasHandler (existing)
│   │   - item_type: "included_table"
│   │   - Handles: pd.DataFrame
│   ├── PolarsHandler (NEW)
│   │   - item_type: "polars_table"
│   │   - Handles: pl.DataFrame
│   └── ReferenceTableHandler (existing)
│       - item_type: "referenced_table"
│       - Handles: external references
```

**Design decisions:**
1. **Separate handlers**: PolarsHandler separate from PandasHandler (clarity, separate concerns)
2. **Shared storage**: Both use Parquet (interoperable)
3. **Metadata tracking**: Add `dataframe_library` field to distinguish
4. **Auto-detection**: `can_handle()` checks for pl.DataFrame

### Item Type Registration

```python
# In storage/categories.py
ITEM_TYPE_TO_CATEGORY = {
    # ... existing ...
    "included_table": StorageCategory.TABLES,
    "polars_table": StorageCategory.TABLES,    # NEW
    "referenced_table": StorageCategory.TABLES,
}
```

### Metadata Structure

**Enhanced table metadata:**
```python
{
    "name": "results",
    "item_type": "polars_table",  # or "included_table"
    "filename": "results.parquet",
    "table_format": "parquet",
    "dataframe_library": "polars",  # NEW: 'pandas' or 'polars'
    "num_rows": 1000,
    "num_cols": 10,
    "columns": ["id", "value", "category"],
    "dtypes": {"id": "Int64", "value": "Float64", "category": "Utf8"},
    "checksum": "abc123...",
    "created_at": "2024-01-15T10:30:00Z",
    # ... other fields ...
}
```

### Query Engine Design

**DuckDB integration (lazy initialization):**

```python
class DataFolio:
    def __init__(self, ...):
        self._duckdb_con = None  # Lazy init

    def _get_duckdb_connection(self):
        """Get or create DuckDB connection."""
        if self._duckdb_con is None:
            try:
                import duckdb
                self._duckdb_con = duckdb.connect()
            except ImportError:
                raise ImportError(
                    "DuckDB required for query features. "
                    "Install with: pip install duckdb"
                )
        return self._duckdb_con

    def _register_table_with_duckdb(self, name: str):
        """Register a table with DuckDB for querying."""
        con = self._get_duckdb_connection()
        path = self.get_table_path(name)
        # DuckDB can read Parquet directly
        return con.from_parquet(path)
```

---

## API Specification

### Polars Handler API

```python
# Auto-detect and store
import polars as pl

df_polars = pl.DataFrame({"a": [1, 2, 3]})
folio.add_table('data', df_polars)  # Auto-detects polars

# Get back (returns original type)
df = folio.get_table('data')  # Returns polars.DataFrame

# Force return type
df_pandas = folio.get_table('data', return_type='pandas')
df_polars = folio.get_table('data', return_type='polars')

# Also works with add_data (generic method)
folio.add_data('results', pl_df)  # Auto-detects
```

### Enhanced get_table API

```python
def get_table(
    self,
    name: str,
    return_type: Optional[Literal['pandas', 'polars']] = None,
    columns: Optional[list[str]] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Get a table with optional column selection and row range.

    Enhanced to support memory-efficient partial loading and parallel processing.

    Args:
        name: Name of table
        return_type: Override return type ('pandas' or 'polars')
        columns: Optional list of columns to load (memory optimization)
        offset: Number of rows to skip from start (for chunking/parallel processing)
        limit: Maximum number of rows to return (for chunking/parallel processing)

    Returns:
        DataFrame (full table or specified subset)

    Examples:
        >>> # Load full table (existing behavior)
        >>> df = folio.get_table('data')

        >>> # Load only specific columns
        >>> df = folio.get_table('data', columns=['id', 'value'])

        >>> # Load specific row range (for parallel processing)
        >>> chunk = folio.get_table('data', offset=10000, limit=5000)
        >>> # Returns rows 10000-14999

        >>> # Combine parameters
        >>> chunk = folio.get_table('data',
        ...     columns=['id', 'score'],
        ...     offset=0,
        ...     limit=1000,
        ...     return_type='polars')

    Notes:
        - Uses DuckDB for efficient offset/limit operations
        - Column selection happens at read time (no full table load)
        - Useful for parallel processing (each worker gets a chunk)
    """
```

### Query API

```python
def query_table(
    self,
    name: str,
    query: str,
    return_type: Literal['pandas', 'polars'] = 'pandas'
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Execute SQL query on a table using DuckDB.

    The table is available as 'table' in the SQL query.

    Args:
        name: Name of table to query
        query: SQL query string (use 'table' as table name)
        return_type: Return DataFrame type ('pandas' or 'polars')

    Returns:
        Query results as DataFrame

    Examples:
        >>> df = folio.query_table('data',
        ...     "SELECT * FROM table WHERE value > 100")

        >>> summary = folio.query_table('measurements',
        ...     '''SELECT subject_id, AVG(score) as mean_score
        ...        FROM table GROUP BY subject_id''')

    Raises:
        ImportError: If duckdb not installed
        KeyError: If table not found
    """
```

### Iteration API

```python
def iter_table(
    self,
    name: str,
    chunk_size: int = 10000,
    columns: Optional[list[str]] = None,
    where: Optional[str] = None,
    return_type: Optional[Literal['pandas', 'polars']] = None
) -> Iterator[Union[pd.DataFrame, pl.DataFrame]]:
    """Iterate through table in chunks without loading full dataset.

    Uses DuckDB for efficient chunked reading with optional filtering.

    Args:
        name: Name of table to iterate
        chunk_size: Number of rows per chunk
        columns: Optional list of columns to load (memory optimization)
        where: Optional SQL WHERE clause for filtering
        return_type: Override default return type

    Yields:
        DataFrame chunks of size chunk_size (or less for last chunk)

    Examples:
        >>> # Basic iteration
        >>> for chunk in folio.iter_table('large_data', chunk_size=10000):
        ...     process(chunk)

        >>> # With column selection
        >>> for chunk in folio.iter_table('data',
        ...                                chunk_size=5000,
        ...                                columns=['id', 'value']):
        ...     process(chunk)

        >>> # With filtering
        >>> for chunk in folio.iter_table('data',
        ...                                chunk_size=10000,
        ...                                where="p_value < 0.05"):
        ...     process(chunk)

    Raises:
        ImportError: If duckdb not installed
        KeyError: If table not found
    """
```

### Path Accessor API

```python
def get_table_path(self, name: str) -> str:
    """Get file system path to a table's Parquet file.

    Useful for integrating with external tools (Ibis, DuckDB, Polars).

    Args:
        name: Name of table

    Returns:
        Absolute path to Parquet file

    Examples:
        >>> path = folio.get_table_path('results')
        '/path/to/bundle/tables/results.parquet'

        >>> # Use with Ibis
        >>> import ibis
        >>> con = ibis.duckdb.connect()
        >>> table = con.read_parquet(path)

        >>> # Use with Polars
        >>> import polars as pl
        >>> df = pl.read_parquet(path)

    Raises:
        KeyError: If table not found
        ValueError: If item is not a table
    """

def get_all_table_paths(self) -> dict[str, str]:
    """Get paths to all tables in the bundle.

    Returns:
        Dictionary mapping table name to file path

    Examples:
        >>> paths = folio.get_all_table_paths()
        {'results': '/path/to/tables/results.parquet',
         'features': '/path/to/tables/features.parquet'}

        >>> # Register all tables with DuckDB
        >>> import duckdb
        >>> con = duckdb.connect()
        >>> for name, path in paths.items():
        ...     con.register(path, table_name=name)
    """
```

### Metadata API

```python
def table_info(self, name: str) -> dict[str, Any]:
    """Get table metadata without loading data.

    Reads Parquet metadata using PyArrow for fast access.

    Args:
        name: Name of table

    Returns:
        Dictionary with table information:
        - rows: Number of rows
        - columns: Number of columns
        - size_mb: File size in megabytes
        - column_names: List of column names
        - dtypes: Dictionary of column types
        - parquet_schema: PyArrow schema object
        - created_at: Creation timestamp

    Examples:
        >>> info = folio.table_info('large_dataset')
        >>> print(f"Table has {info['rows']:,} rows, {info['size_mb']:.1f} MB")
        Table has 1,000,000 rows, 245.3 MB

        >>> if info['size_mb'] > 1000:
        ...     # Use chunked iteration for large tables
        ...     for chunk in folio.iter_table('large_dataset'):
        ...         process(chunk)
    """

def preview_table(
    self,
    name: str,
    n: int = 10,
    return_type: Optional[Literal['pandas', 'polars']] = None
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Preview first N rows of table without loading full dataset.

    Uses DuckDB LIMIT for efficient preview.

    Args:
        name: Name of table
        n: Number of rows to preview
        return_type: Override default return type

    Returns:
        DataFrame with first n rows

    Examples:
        >>> preview = folio.preview_table('data', n=5)
        >>> print(preview)
           id  value
        0   1    0.5
        1   2    0.7
        ...
    """
```

---

## Integration Patterns

### Pattern 1: Ibis for Complex Queries

```python
import ibis
from datafolio import DataFolio

# Load folio
folio = DataFolio('experiments/analysis')

# Get table path
path = folio.get_table_path('large_dataset')

# Use Ibis for lazy queries
con = ibis.duckdb.connect()
table = con.read_parquet(path)

# Build complex query (lazy)
result = (table
    .filter(table.p_value < 0.05)
    .filter(table.fold_change.abs() > 2)
    .group_by('gene_id')
    .aggregate(
        mean_expr=table.expression.mean(),
        n_samples=table.sample_id.nunique()
    )
    .order_by(ibis.desc('mean_expr'))
    .limit(100))

# Execute when needed
df = result.execute()

# Save back to DataFolio
folio.add_table('significant_genes', df)
folio.create_snapshot('v1.0-filtered')
```

### Pattern 2: Multi-Table Analysis

```python
import duckdb
from datafolio import DataFolio

folio = DataFolio('clinical_trial')

# Get all table paths
paths = folio.get_all_table_paths()

# Register with DuckDB
con = duckdb.connect()
for name, path in paths.items():
    con.register(path, table_name=name)

# Complex join query
result = con.sql("""
    SELECT
        p.patient_id,
        p.age,
        p.treatment_group,
        t.visit_date,
        AVG(t.blood_pressure) as mean_bp,
        COUNT(DISTINCT t.visit_id) as n_visits
    FROM patients p
    JOIN trials t ON p.patient_id = t.patient_id
    WHERE t.visit_date BETWEEN '2024-01-01' AND '2024-12-31'
    GROUP BY p.patient_id, p.age, p.treatment_group, t.visit_date
    HAVING n_visits >= 3
""").df()

# Save results
folio.add_table('patient_summary', result)
```

### Pattern 3: Polars for Speed

```python
import polars as pl
from datafolio import DataFolio

folio = DataFolio('genomics')

# Get path for direct Polars access
path = folio.get_table_path('variants')

# Use Polars for fast processing
df = (pl.read_parquet(path)
    .filter(pl.col('quality') > 30)
    .with_columns([
        (pl.col('alt_count') / pl.col('total_count')).alias('alt_freq')
    ])
    .filter(pl.col('alt_freq') > 0.05)
)

# Save back (auto-detected as polars)
folio.add_table('filtered_variants', df)
```

### Pattern 4: Chunked Processing Pipeline

```python
from datafolio import DataFolio
import pandas as pd

folio = DataFolio('experiments')

# Process large table in chunks
results = []
for chunk in folio.iter_table('raw_data', chunk_size=50000):
    # Heavy processing per chunk
    processed = expensive_computation(chunk)
    results.append(processed)

# Combine results (assuming results are small)
final = pd.concat(results, ignore_index=True)
folio.add_table('processed_results', final)

# Or write progressively to avoid memory issues
for i, chunk in enumerate(folio.iter_table('raw_data', chunk_size=50000)):
    processed = expensive_computation(chunk)
    folio.add_table(f'processed_chunk_{i}', processed)

# Later, use SQL to query across chunks
result = folio.query_table('processed_chunk_0',
    "SELECT * FROM table WHERE score > 0.8 LIMIT 100")
```

### Pattern 5: Parallel Processing with Multiprocessing

```python
from datafolio import DataFolio
from multiprocessing import Pool
from functools import partial
import pandas as pd

def process_chunk(chunk_data, expensive_param=None):
    """
    Process a single chunk of data.

    This function will be called in parallel by worker processes.
    Must be pickle-able (defined at module level, not nested).
    """
    chunk_id, bundle_path, table_name, chunk_size = chunk_data

    # Each worker creates its own DataFolio instance
    # (DataFolio objects may not be pickle-able)
    folio = DataFolio(bundle_path)

    # Get the specific chunk using offset and limit parameters
    chunk = folio.get_table(
        table_name,
        offset=chunk_id * chunk_size,
        limit=chunk_size
    )

    # Heavy computation here
    result = expensive_computation(chunk, expensive_param)

    return result

def expensive_computation(chunk, param):
    """Your actual processing logic."""
    # Example: compute statistics per subject
    return chunk.groupby('subject_id').agg({
        'value': ['mean', 'std', 'count']
    }).reset_index()

# Main script
if __name__ == '__main__':
    folio = DataFolio('experiments/analysis')

    # Get table info to determine number of chunks
    info = folio.table_info('large_dataset')
    chunk_size = 50000
    num_chunks = (info['rows'] + chunk_size - 1) // chunk_size  # Ceiling division

    print(f"Processing {info['rows']:,} rows in {num_chunks} chunks")

    # Prepare chunk specifications
    bundle_path = folio._bundle_dir
    chunk_specs = [
        (i, bundle_path, 'large_dataset', chunk_size)
        for i in range(num_chunks)
    ]

    # Process in parallel
    with Pool(processes=4) as pool:
        # Option 1: Map with partial for extra parameters
        process_func = partial(process_chunk, expensive_param=42)
        results = pool.map(process_func, chunk_specs)

        # Option 2: Use imap for progress tracking
        # results = list(tqdm(
        #     pool.imap(process_func, chunk_specs),
        #     total=num_chunks
        # ))

    # Combine results
    final_result = pd.concat(results, ignore_index=True)

    # Aggregate across chunks if needed
    final_summary = final_result.groupby('subject_id').agg({
        'value_mean': 'mean',
        'value_std': 'mean',  # Average of stds
        'value_count': 'sum'
    }).reset_index()

    # Save back to DataFolio
    folio.add_table('processed_results', final_summary)
    folio.create_snapshot('v1.0-processed')

    print(f"Processed {len(final_summary)} subjects")
```

**Alternative: Using iter_table with multiprocessing.Queue**

For more complex scenarios where you need streaming results:

```python
from datafolio import DataFolio
from multiprocessing import Process, Queue, cpu_count
import pandas as pd

def worker(task_queue, result_queue, bundle_path):
    """Worker process that processes chunks from queue."""
    folio = DataFolio(bundle_path)

    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break

        chunk_id, chunk = task

        # Process chunk
        result = expensive_computation(chunk)
        result_queue.put((chunk_id, result))

def writer(result_queue, output_path, num_chunks):
    """Writer process that saves results as they complete."""
    folio = DataFolio(output_path)
    processed = 0

    while processed < num_chunks:
        chunk_id, result = result_queue.get()
        folio.add_table(f'result_chunk_{chunk_id}', result)
        processed += 1
        print(f"Saved chunk {processed}/{num_chunks}")

if __name__ == '__main__':
    folio = DataFolio('experiments/analysis')

    # Setup queues
    task_queue = Queue(maxsize=10)  # Limit queue size to control memory
    result_queue = Queue()

    # Start worker processes
    num_workers = cpu_count() - 1  # Leave one core for main + writer
    workers = []
    for _ in range(num_workers):
        p = Process(target=worker, args=(task_queue, result_queue, folio._bundle_dir))
        p.start()
        workers.append(p)

    # Start writer process
    chunk_size = 50000
    info = folio.table_info('large_dataset')
    num_chunks = (info['rows'] + chunk_size - 1) // chunk_size

    writer_process = Process(target=writer, args=(result_queue, folio._bundle_dir, num_chunks))
    writer_process.start()

    # Feed tasks to workers
    for chunk_id, chunk in enumerate(folio.iter_table('large_dataset', chunk_size=chunk_size)):
        task_queue.put((chunk_id, chunk))

    # Send poison pills
    for _ in range(num_workers):
        task_queue.put(None)

    # Wait for completion
    for w in workers:
        w.join()

    writer_process.join()

    print("Processing complete!")
```

**Best Practices for Multiprocessing with DataFolio:**

1. **Pickle-able Functions**: Define processing functions at module level (not nested)

2. **Separate DataFolio Instances**: Each process should create its own DataFolio instance
   ```python
   # ❌ Don't pass DataFolio objects to workers
   with Pool() as pool:
       pool.map(lambda chunk: folio.process(chunk), chunks)  # Won't work!

   # ✅ Pass bundle path, create instance in worker
   def worker(bundle_path, chunk):
       folio = DataFolio(bundle_path)
       return folio.process(chunk)
   ```

3. **Chunk Size Tuning**: Balance parallelism vs overhead
   ```python
   # Too small: High overhead from creating DataFolio instances
   chunk_size = 100  # ❌

   # Too large: Not enough chunks for all workers
   chunk_size = 10_000_000  # ❌

   # Good: ~10-100 chunks per worker
   chunk_size = total_rows // (num_workers * 50)  # ✅
   ```

4. **Memory Management**: Limit queue sizes to avoid memory explosion
   ```python
   task_queue = Queue(maxsize=num_workers * 2)  # Limit backlog
   ```

5. **Progress Tracking**: Use `tqdm` with `imap` for progress bars
   ```python
   from tqdm import tqdm
   results = list(tqdm(pool.imap(func, tasks), total=len(tasks)))
   ```

6. **Error Handling**: Catch and log errors in workers
   ```python
   def safe_worker(task):
       try:
           return process_chunk(task)
       except Exception as e:
           print(f"Error processing chunk {task[0]}: {e}")
           return None  # Or return error marker
   ```

**When to Use Multiprocessing:**

✅ **Good use cases:**
- CPU-bound processing (complex calculations per row)
- Independent chunks (no dependencies between chunks)
- Large datasets (justify overhead)
- Available CPU cores (> 4 cores)

❌ **Bad use cases:**
- I/O-bound tasks (use async instead)
- Small datasets (overhead > benefit)
- Few CPU cores (< 4 cores)
- Complex shared state (use threading or single-process)

---

## Implementation Plan

### Phase 1.1: Polars Handler (Week 1)

**Files to modify/create:**

1. **`src/datafolio/handlers/tables.py`**
   - Add `PolarsHandler` class (~60 lines)
   - Implement `can_handle()`, `add()`, `get()`
   - Use polars `.write_parquet()` and `.read_parquet()`

2. **`src/datafolio/storage/categories.py`**
   - Add `"polars_table": StorageCategory.TABLES` to mapping

3. **`src/datafolio/handlers/__init__.py`**
   - Import and register PolarsHandler
   - Add in registration order (after pandas, before json)

4. **`src/datafolio/readers.py`** (optional enhancement)
   - Add `read_polars()` helper function

5. **`pyproject.toml`**
   - Add polars as optional dependency:
     ```toml
     [project.optional-dependencies]
     polars = ["polars>=0.19.0"]
     ```

**Tests to add:**
- `tests/test_handlers_polars.py` - Unit tests for PolarsHandler
- `tests/test_polars_integration.py` - Integration tests with DataFolio

**Steps:**
1. Create PolarsHandler following PandasHandler pattern
2. Add metadata field `dataframe_library = 'polars'`
3. Test auto-detection (priority over JSONHandler)
4. Test round-trip (add → get)
5. Test interop (add polars, get pandas and vice versa)

---

### Phase 1.2: DuckDB Queries (Week 2)

**Files to modify/create:**

1. **`src/datafolio/folio.py`**
   - Add `_duckdb_con` attribute (lazy init)
   - Add `_get_duckdb_connection()` method
   - Add `query_table()` method (~40 lines)
   - Add `_register_table_with_duckdb()` helper

2. **`pyproject.toml`**
   - Add duckdb as optional dependency:
     ```toml
     [project.optional-dependencies]
     query = ["duckdb>=0.9.0"]
     all = ["polars>=0.19.0", "duckdb>=0.9.0"]
     ```

**Tests to add:**
- `tests/test_duckdb_queries.py` - Query functionality tests

**Steps:**
1. Implement lazy DuckDB connection
2. Add query_table() with error handling
3. Support return_type parameter (pandas/polars)
4. Test with filters, aggregations, joins
5. Test error cases (table not found, DuckDB not installed)

---

### Phase 1.3: Chunked Iteration (Week 2-3)

**Files to modify/create:**

1. **`src/datafolio/folio.py`**
   - Add `iter_table()` method (~50 lines)
   - Use DuckDB for chunked reads (OFFSET/LIMIT)
   - Support columns, where, return_type parameters

**Tests to add:**
- `tests/test_table_iteration.py` - Iteration tests

**Steps:**
1. Implement basic iteration with DuckDB
2. Add column selection (SELECT clause)
3. Add filtering (WHERE clause)
4. Test memory efficiency (large tables)
5. Test with both pandas and polars tables

---

### Phase 1.4: Path Accessors (Week 3)

**Files to modify/create:**

1. **`src/datafolio/folio.py`**
   - Add `get_table_path()` method (~20 lines)
   - Add `get_all_table_paths()` method (~15 lines)

**Tests to add:**
- `tests/test_path_accessors.py` - Path accessor tests

**Steps:**
1. Implement path resolution from items.json
2. Validate item is a table
3. Return absolute paths
4. Test with both included and referenced tables
5. Test error cases (not found, not a table)

---

### Phase 1.5: Table Metadata (Week 3)

**Files to modify/create:**

1. **`src/datafolio/folio.py`**
   - Add `table_info()` method (~30 lines)
   - Add `preview_table()` method (~15 lines)
   - Use PyArrow for metadata reading

2. **`pyproject.toml`**
   - PyArrow already a dependency (used for Parquet)

**Tests to add:**
- `tests/test_table_metadata.py` - Metadata query tests

**Steps:**
1. Implement table_info() using PyArrow
2. Add caching for repeated calls
3. Implement preview_table() using DuckDB LIMIT
4. Test with large and small tables
5. Test error cases

---

## Testing Strategy

### Unit Tests

**Handler Tests** (`tests/test_handlers_polars.py`):
```python
def test_polars_handler_can_handle():
    """Test PolarsHandler detects polars DataFrames."""

def test_polars_handler_add():
    """Test adding polars DataFrame stores correctly."""

def test_polars_handler_get():
    """Test retrieving polars DataFrame."""

def test_polars_to_pandas_conversion():
    """Test adding polars, retrieving as pandas."""

def test_pandas_to_polars_conversion():
    """Test adding pandas, retrieving as polars."""
```

**Query Tests** (`tests/test_duckdb_queries.py`):
```python
def test_query_table_basic():
    """Test simple SELECT query."""

def test_query_table_with_filter():
    """Test query with WHERE clause."""

def test_query_table_aggregation():
    """Test GROUP BY aggregation."""

def test_query_table_returns_polars():
    """Test return_type='polars' parameter."""

def test_query_table_missing_duckdb():
    """Test error when DuckDB not installed."""
```

**Iteration Tests** (`tests/test_table_iteration.py`):
```python
def test_iter_table_basic():
    """Test basic chunked iteration."""

def test_iter_table_column_selection():
    """Test columns parameter."""

def test_iter_table_filtering():
    """Test where parameter."""

def test_iter_table_memory_efficiency():
    """Test only one chunk in memory at a time."""

def test_iter_table_returns_polars():
    """Test return_type parameter."""
```

### Integration Tests

**Workflow Tests** (`tests/test_table_workflows.py`):
```python
def test_chunked_processing_workflow():
    """Test complete chunked processing pipeline."""

def test_ibis_integration():
    """Test integration with Ibis."""

def test_multi_table_query():
    """Test querying across multiple tables."""

def test_polars_pandas_interop():
    """Test mixing polars and pandas tables."""
```

### Performance Tests

**Benchmarks** (`tests/test_table_performance.py`):
```python
def test_large_table_iteration():
    """Benchmark iteration over 10GB table."""

def test_query_vs_load_memory():
    """Compare memory usage: query vs full load."""

def test_polars_vs_pandas_speed():
    """Compare polars and pandas read/write speed."""
```

---

## Documentation Updates

### User Guide Additions

**1. New section: `docs/guides/tables.md`**

Content outline:
```markdown
# Working with Tables

## DataFrame Libraries
- Pandas support (existing)
- Polars support (new)
- Auto-detection and conversion

## Memory-Efficient Processing
- Chunked iteration for large tables
- Column and row selection
- Filtering with SQL

## SQL Queries
- Basic queries with query_table()
- Filtering and aggregation
- Return type options

## Integration with Other Tools
- Using Ibis for complex queries
- Direct DuckDB access
- Polars workflows

## Best Practices
- When to use chunked iteration
- Choosing pandas vs polars
- Memory management tips
```

**2. Update: `docs/guides/getting-started.md`**

Add polars example:
```python
# Works with polars too!
import polars as pl
df_polars = pl.DataFrame({"x": [1, 2, 3]})
folio.add_table('data', df_polars)  # Auto-detected
```

**3. Update: `README.md`**

Add to features list:
- ✅ **Polars Support**: First-class support for modern Polars DataFrames
- ✅ **SQL Queries**: Query tables with SQL using DuckDB (memory-efficient)
- ✅ **Chunked Iteration**: Process large datasets without loading into memory
- ✅ **Tool Integration**: Easy integration with Ibis, DuckDB, Polars directly

**4. API Reference: `docs/reference/api.md`**

Add method documentation for:
- `query_table()`
- `iter_table()`
- `get_table_path()`
- `get_all_table_paths()`
- `table_info()`
- `preview_table()`

---

## Future Work (Phase 2)

### Delta Lake Support

**Planned for next iteration:**

```python
# Reference Delta Lake tables
folio.reference_table('events',
    reference='s3://bucket/delta/events/',
    format='delta')

# Time travel
df = folio.get_table('events', version=5)
df = folio.get_table('events', timestamp='2024-01-15')

# Query Delta tables
results = folio.query_table('events',
    "SELECT * FROM table WHERE event_type = 'click'")
```

**Why later:**
- More complex (external dependencies, time travel, schema evolution)
- Builds on Phase 1 infrastructure
- Less commonly used in academic research (but growing)

**Implementation notes:**
- Use `deltalake` Python package
- Extend ReferenceTableHandler
- Integrate with DuckDB (has Delta extension)
- Support version and timestamp parameters

---

## Dependencies Summary

### Required (Core)
- pandas >= 2.0.0 (existing)
- pyarrow >= 14.0.0 (existing)

### Optional (New)
```toml
[project.optional-dependencies]
polars = ["polars>=0.19.0"]
query = ["duckdb>=0.9.0"]
all = ["polars>=0.19.0", "duckdb>=0.9.0"]
```

### Future (Phase 2)
```toml
delta = ["deltalake>=0.10.0"]
```

---

## Success Criteria

### Functionality
- ✅ Polars DataFrames can be stored and retrieved
- ✅ SQL queries execute on tables without full load
- ✅ Chunked iteration works with tables larger than RAM
- ✅ Path accessors enable Ibis integration
- ✅ Table metadata queries are fast (< 100ms)

### Performance
- ✅ Polars read/write is faster than pandas
- ✅ Query uses < 10% memory of full load for filters
- ✅ Chunked iteration maintains constant memory
- ✅ Metadata queries don't scan data

### Usability
- ✅ Auto-detection works reliably (pandas/polars)
- ✅ Error messages are clear (missing dependencies)
- ✅ Integration examples in docs work
- ✅ API is intuitive and consistent with existing DataFolio

### Compatibility
- ✅ All existing tests still pass
- ✅ No breaking changes to existing API
- ✅ Works with snapshots (versioning preserved)
- ✅ Cloud storage compatible (S3, GCS)

---

## Migration & Compatibility

### Backward Compatibility

**All existing code works unchanged:**
```python
# Existing pandas code - no changes needed
folio.add_table('data', pandas_df)  # Still works
df = folio.get_table('data')  # Still returns pandas by default
```

**New features are opt-in:**
```python
# Use new features when you want them
df = folio.query_table('data', "SELECT * WHERE x > 5")  # New
for chunk in folio.iter_table('data'):  # New
    process(chunk)
```

### No Breaking Changes

- Existing handlers unchanged
- Items.json format backward compatible
- Snapshots work with new table types
- Optional dependencies (won't break if not installed)

---

## Timeline Estimate

**Week 1:** Polars handler + tests
**Week 2:** DuckDB queries + chunked iteration
**Week 3:** Path accessors + metadata queries + docs
**Week 4:** Integration tests + performance testing + documentation

**Total:** ~3-4 weeks for Phase 1

**Phase 2 (Delta Lake):** Additional 2-3 weeks (future iteration)

---

## Open Questions

1. **DuckDB connection lifecycle:** Keep persistent or close after each operation?
   - **Recommendation:** Keep persistent per DataFolio instance (performance)

2. **Polars as default?** Should new installations prefer polars over pandas?
   - **Recommendation:** No, pandas remains default for compatibility

3. **Query syntax:** Allow column.value syntax or just SQL strings?
   - **Recommendation:** Just SQL strings (simpler, familiar)

4. **Chunk size defaults:** 10k rows good default for all use cases?
   - **Recommendation:** Yes, with documentation on tuning

5. **Return type inference:** Always infer from stored type or require explicit?
   - **Recommendation:** Infer by default, allow override

---

## Notes & Decisions

**Design Decision Log:**

1. **Separate Polars handler** - Clearer separation of concerns vs merged handler
2. **DuckDB for queries** - More SQL-like vs Polars lazy API
3. **Path exposure over wrappers** - Enable integration vs implement features
4. **Optional dependencies** - Don't force users to install unused tools
5. **Minimal API** - Primitives + integration vs feature-complete query engine

**Trade-offs Accepted:**

- Less feature-rich than Ibis → But better versioning integration
- DuckDB dependency for queries → But optional and lightweight
- Manual tool integration → But maximum flexibility
- No distributed computing → But that's not DataFolio's job

---

**End of Planning Document**

*This document will be updated as implementation progresses and decisions are refined.*
