# Advanced Table Features

DataFolio provides powerful features for working with large tables efficiently, including support for Polars DataFrames, SQL queries, chunked iteration, and convenient query parameters.

## Polars Support

DataFolio automatically detects and handles both pandas and Polars DataFrames, storing them in an interoperable Parquet format.

### Adding Polars Tables

```python
import polars as pl
from datafolio import DataFolio

folio = DataFolio('experiments/polars_demo')

# Create Polars DataFrame
df = pl.DataFrame({
    'id': range(1000),
    'value': range(1000, 2000),
    'category': ['A', 'B', 'C'] * 333 + ['A']
})

# Add to folio - automatically detected as Polars
folio.add_table('data', df,
    description='Demo data in Polars format')

# Retrieve returns Polars DataFrame
result = folio.get_table('data')
assert isinstance(result, pl.DataFrame)
```

### Type Preservation

DataFolio preserves the DataFrame library you used:

```python
# Polars in → Polars out
df_polars = pl.DataFrame({'x': [1, 2, 3]})
folio.add_table('polars_data', df_polars)
result = folio.get_table('polars_data')
assert isinstance(result, pl.DataFrame)

# Pandas in → Pandas out
import pandas as pd
df_pandas = pd.DataFrame({'x': [1, 2, 3]})
folio.add_table('pandas_data', df_pandas)
result = folio.get_table('pandas_data')
assert isinstance(result, pd.DataFrame)
```

### Interoperability

Both pandas and Polars tables are stored as Parquet files, making them interoperable:

```python
# Add with pandas
import pandas as pd
df_pandas = pd.DataFrame({'a': [1, 2, 3]})
folio.add_table('data', df_pandas)

# You can read the underlying Parquet file with either library
import polars as pl
# The file is at: folio.path / 'tables' / 'data.parquet'
```

## Installation

The advanced table features require optional dependencies:

```bash
# Install with Polars support only
pip install datafolio[polars]

# Install with DuckDB query support only
pip install datafolio[query]

# Install with all features
pip install datafolio[all]
```

## SQL Queries with DuckDB

Query tables using SQL without loading them fully into memory. This is powered by DuckDB and works directly on Parquet files.

### Basic Queries

```python
from datafolio import DataFolio
import pandas as pd

folio = DataFolio('experiments/query_demo')

# Add large table
df = pd.DataFrame({
    'timestamp': range(100000),
    'value': range(100000, 200000),
    'status': ['active', 'inactive'] * 50000
})
folio.add_table('data', df)

# Query with SQL - uses DuckDB, doesn't load full table
result = folio.query_table(
    'data',
    'SELECT * FROM $table WHERE value > 150000 LIMIT 10'
)

print(len(result))  # 10 rows
```

The `$table` placeholder is automatically replaced with the Parquet file path.

### Aggregations

```python
# Group and aggregate
result = folio.query_table(
    'data',
    '''SELECT status,
              COUNT(*) as count,
              AVG(value) as avg_value,
              MAX(value) as max_value
       FROM $table
       GROUP BY status'''
)

print(result)
# Output:
#      status  count  avg_value  max_value
# 0    active  50000   149999.5     199999
# 1  inactive  50000   150000.0     200000
```

### Filtering and Sorting

```python
# Complex filtering
result = folio.query_table(
    'data',
    '''SELECT * FROM $table
       WHERE status = 'active'
         AND value > 180000
       ORDER BY value DESC
       LIMIT 100'''
)
```

### Return Types

Choose between pandas and Polars for the result:

```python
# Return as pandas DataFrame (default)
result_pandas = folio.query_table(
    'data',
    'SELECT * FROM $table LIMIT 10',
    return_type='pandas'
)

# Return as Polars DataFrame
result_polars = folio.query_table(
    'data',
    'SELECT * FROM $table LIMIT 10',
    return_type='polars'
)
```

## Chunked Iteration

Process large tables in memory-efficient chunks without loading the entire table at once.

### Basic Iteration

```python
from datafolio import DataFolio
import pandas as pd

folio = DataFolio('experiments/chunked_demo')

# Add large table
df = pd.DataFrame({
    'id': range(100000),
    'value': range(100000, 200000)
})
folio.add_table('large_data', df)

# Iterate in chunks of 10,000 rows
for chunk in folio.iter_table('large_data', chunk_size=10000):
    # Process each chunk
    print(f"Processing {len(chunk)} rows")
    # ... do analysis on chunk ...
```

### Column Selection

Select only the columns you need to reduce memory usage:

```python
# Only load specific columns
for chunk in folio.iter_table(
    'data',
    chunk_size=5000,
    columns=['id', 'value']  # Only these columns
):
    process_chunk(chunk)
```

### Filtering During Iteration

Apply WHERE filters to reduce the amount of data returned:

```python
# Only iterate over matching rows
for chunk in folio.iter_table(
    'data',
    chunk_size=5000,
    where='value > 150000 AND status = "active"'
):
    process_chunk(chunk)
```

### Combined Example

```python
# Column selection + filtering + chunking
total = 0
for chunk in folio.iter_table(
    'large_data',
    chunk_size=10000,
    columns=['value'],
    where='value > 150000'
):
    total += chunk['value'].sum()

print(f"Total: {total}")
```

### Return Types

Choose the output format for each chunk:

```python
# Return pandas DataFrames (default)
for chunk in folio.iter_table('data', chunk_size=5000, return_type='pandas'):
    assert isinstance(chunk, pd.DataFrame)

# Return Polars DataFrames
for chunk in folio.iter_table('data', chunk_size=5000, return_type='polars'):
    assert isinstance(chunk, pl.DataFrame)
```

## Convenience Query Parameters

For common operations, `get_table()` now supports convenient parameters that don't require full SQL syntax.

### Limit Rows

Get only the first N rows:

```python
# Get first 100 rows
preview = folio.get_table('data', limit=100)

# Get first 10 rows for quick inspection
sample = folio.get_table('large_table', limit=10)
```

### Offset (Skip Rows)

Skip the first N rows:

```python
# Skip first 1000 rows
later_data = folio.get_table('data', offset=1000)

# Get middle section
middle = folio.get_table('data', offset=500, limit=100)  # Rows 500-599
```

### Pagination

Combine `limit` and `offset` for pagination:

```python
# Page 1 (rows 0-99)
page1 = folio.get_table('data', limit=100, offset=0)

# Page 2 (rows 100-199)
page2 = folio.get_table('data', limit=100, offset=100)

# Page 3 (rows 200-299)
page3 = folio.get_table('data', limit=100, offset=200)
```

### Simple Filtering with WHERE

Filter rows without writing full SQL - just provide the condition:

```python
# Filter by value
high_values = folio.get_table('data', where='value > 9000')

# Filter by string
active_only = folio.get_table('data', where="status = 'active'")

# Multiple conditions
filtered = folio.get_table(
    'data',
    where="status = 'active' AND value > 1000"
)

# Combine with limit
top_active = folio.get_table(
    'data',
    where="status = 'active'",
    limit=100
)
```

The `where` parameter takes just the condition part (what comes after `WHERE` in SQL), and DataFolio builds the full query automatically.

### Type Preservation with Query Parameters

The convenience parameters preserve your DataFrame type:

```python
# Polars table returns Polars, even with query params
df_polars = pl.DataFrame({'value': range(1000)})
folio.add_table('polars_data', df_polars)

result = folio.get_table('polars_data', where='value > 500', limit=100)
assert isinstance(result, pl.DataFrame)  # Still Polars!

# Pandas table returns pandas
df_pandas = pd.DataFrame({'value': range(1000)})
folio.add_table('pandas_data', df_pandas)

result = folio.get_table('pandas_data', where='value > 500')
assert isinstance(result, pd.DataFrame)  # Still pandas!
```

### Complete Example

```python
from datafolio import DataFolio
import pandas as pd

folio = DataFolio('experiments/queries')

# Add data
df = pd.DataFrame({
    'timestamp': range(10000),
    'value': range(10000, 20000),
    'status': ['active', 'inactive'] * 5000,
    'category': (['A'] * 2500 + ['B'] * 2500) * 2
})
folio.add_table('data', df)

# Quick preview
print(folio.get_table('data', limit=5))

# Get active items only
active = folio.get_table('data', where="status = 'active'")
print(f"Active items: {len(active)}")

# Paginate through results
page_size = 1000
for page_num in range(3):
    page = folio.get_table(
        'data',
        limit=page_size,
        offset=page_num * page_size
    )
    print(f"Page {page_num + 1}: {len(page)} rows")

# Complex filtering with limit
high_value_a = folio.get_table(
    'data',
    where="category = 'A' AND value > 15000",
    limit=100
)
print(f"High-value category A (top 100): {len(high_value_a)}")
```

## Comparison: When to Use Each Feature

| Feature | Use Case | Memory Efficient | Requires DuckDB |
|---------|----------|------------------|-----------------|
| `get_table()` | Small to medium tables | No (loads full table) | No |
| `get_table(limit=N)` | Quick previews, sampling | Yes (loads only N rows) | Yes |
| `get_table(where=...)` | Simple filtering | Yes (loads only matches) | Yes |
| `query_table()` | Complex SQL, aggregations | Yes | Yes |
| `iter_table()` | Very large tables, streaming | Yes (chunked) | Yes |

### Choosing the Right Tool

**Use `get_table()` without parameters when:**
- Table fits comfortably in memory
- You need the entire dataset
- You don't need DuckDB installed

**Use `get_table(limit=N)` when:**
- You want a quick preview
- You're sampling data
- You want a simple interface without SQL

**Use `get_table(where=...)` when:**
- You need simple filtering
- You don't want to write SQL
- The filtered result fits in memory

**Use `query_table()` when:**
- You need complex SQL (JOINs, subqueries, CTEs)
- You want aggregations or GROUP BY
- You need fine control over the query

**Use `iter_table()` when:**
- Table is very large (millions of rows)
- You're doing streaming/online processing
- You want to process data in batches
- Memory is constrained

## Examples

### Preview Large Tables

```python
# Quick inspection without loading everything
print(folio.get_table('huge_table', limit=10))

# Sample from different sections
print(folio.get_table('huge_table', offset=1000, limit=10))
print(folio.get_table('huge_table', offset=50000, limit=10))
```

### Filter and Sample

```python
# Get 100 active records
active_sample = folio.get_table(
    'data',
    where="status = 'active'",
    limit=100
)

# Get recent records
recent = folio.get_table(
    'data',
    where='timestamp > 20240101',
    limit=1000
)
```

### Memory-Efficient Aggregation

```python
# Calculate statistics without loading full table
total = 0
count = 0

for chunk in folio.iter_table('large_data', chunk_size=10000):
    total += chunk['value'].sum()
    count += len(chunk)

mean = total / count
print(f"Mean: {mean}")
```

### Data Export

```python
# Export filtered data in chunks
with open('export.csv', 'w') as f:
    # Write header
    first_chunk = True

    for chunk in folio.iter_table(
        'data',
        chunk_size=5000,
        where="status = 'active'"
    ):
        chunk.to_csv(f, header=first_chunk, index=False, mode='a')
        first_chunk = False
```

### Progressive Processing

```python
# Process data progressively, showing progress
import pandas as pd

results = []

for i, chunk in enumerate(folio.iter_table('data', chunk_size=10000)):
    # Process chunk
    processed = expensive_operation(chunk)
    results.append(processed)

    # Show progress
    print(f"Processed chunk {i + 1} ({len(chunk)} rows)")

# Combine results
final_result = pd.concat(results, ignore_index=True)
```

### Mixed DataFrame Libraries

```python
# Store pandas
df_pandas = pd.DataFrame({'x': range(1000)})
folio.add_table('pandas_data', df_pandas)

# Store Polars
df_polars = pl.DataFrame({'y': range(1000)})
folio.add_table('polars_data', df_polars)

# Both can be queried with same interface
pandas_result = folio.query_table('pandas_data', 'SELECT * FROM $table WHERE x > 500')
polars_result = folio.query_table('polars_data', 'SELECT * FROM $table WHERE y > 500')

# Type preservation in iteration
for chunk in folio.iter_table('polars_data', chunk_size=100, return_type='polars'):
    assert isinstance(chunk, pl.DataFrame)
```

## Performance Tips

1. **Use `limit` for previews** - Much faster than loading and then slicing
   ```python
   # Fast
   preview = folio.get_table('data', limit=10)

   # Slower
   full_table = folio.get_table('data')
   preview = full_table.head(10)
   ```

2. **Filter early** - Apply WHERE conditions to reduce data transfer
   ```python
   # Good - filters at read time
   active = folio.get_table('data', where="status = 'active'")

   # Bad - loads everything then filters
   all_data = folio.get_table('data')
   active = all_data[all_data['status'] == 'active']
   ```

3. **Select only needed columns** - When iterating, specify columns
   ```python
   # Good - loads only 2 columns
   for chunk in folio.iter_table('data', chunk_size=10000, columns=['id', 'value']):
       process(chunk)

   # Bad - loads all columns
   for chunk in folio.iter_table('data', chunk_size=10000):
       process(chunk[['id', 'value']])
   ```

4. **Use appropriate chunk sizes**
   - Smaller chunks (1,000-5,000): Lower memory, more overhead
   - Medium chunks (10,000-50,000): Good balance for most cases
   - Larger chunks (100,000+): Less overhead, more memory

5. **Combine operations** - Use WHERE + LIMIT together
   ```python
   # Efficient - one operation
   result = folio.get_table('data', where='value > 5000', limit=100)

   # Less efficient - two operations
   filtered = folio.get_table('data', where='value > 5000')
   result = filtered.head(100)
   ```

## Limitations

### Referenced Tables

The query and convenience parameters only work with **included tables** (tables stored in the bundle), not referenced tables:

```python
# This works - included table
folio.add_table('data', df)
result = folio.get_table('data', limit=100)  # ✓

# This doesn't work - referenced table
folio.reference_table('external', 's3://bucket/data.parquet')
result = folio.get_table('external', limit=100)  # ✗ ValueError
```

For referenced tables, use the reference directly:

```python
import pandas as pd

# Get the reference path
path = folio.data.external.path

# Use pandas/pyarrow directly with query pushdown
df = pd.read_parquet(path, filters=[('value', '>', 1000)])
```

### DuckDB Requirement

Features requiring DuckDB will raise clear errors if it's not installed:

```python
# Without duckdb installed
folio.query_table('data', 'SELECT * FROM $table')
# ImportError: DuckDB is required for query operations. Install with: pip install datafolio[query]
```

## Table Utility Methods

DataFolio provides several utility methods for inspecting and accessing tables without loading them fully into memory.

### Get Table Path

Get the filesystem path to a table's Parquet file for use with external tools:

```python
from datafolio import DataFolio
import pandas as pd

folio = DataFolio('experiments/analysis')

# Add table
df = pd.DataFrame({'x': [1, 2, 3]})
folio.add_table('data', df)

# Get path to Parquet file
path = folio.get_table_path('data')
print(path)
# /path/to/experiments/analysis/tables/data.parquet

# Use with external tools
import polars as pl
df_polars = pl.read_parquet(path)

import ibis
con = ibis.duckdb.connect()
table = con.read_parquet(path)
```

**Use cases:**
- Integration with Ibis, DuckDB, Polars
- Custom processing pipelines
- Direct file access for inspection

### Table Info

Get metadata about a table without loading it into memory:

```python
from datafolio import DataFolio
import pandas as pd

folio = DataFolio('experiments/analysis')

# Add large table
df = pd.DataFrame({'value': range(1000000)})
folio.add_table('large_data', df)

# Check table metadata
info = folio.table_info('large_data')

print(f"Rows: {info['num_rows']:,}")
print(f"Columns: {info['num_columns']}")
print(f"Size: {info['size_mb']:.1f} MB")
print(f"Column names: {info['columns']}")

# Output:
# Rows: 1,000,000
# Columns: 1
# Size: 3.8 MB
# Column names: ['value']

# Use info to decide processing strategy
if info['size_mb'] > 1000:
    # Large table: use chunked iteration
    for chunk in folio.iter_table('large_data', chunk_size=10000):
        process(chunk)
else:
    # Small table: load directly
    df = folio.get_table('large_data')
```

**Returns:**
- `num_rows`: Number of rows
- `num_columns`: Number of columns
- `size_bytes`: File size in bytes
- `size_mb`: File size in megabytes
- `columns`: List of column names
- `schema`: PyArrow schema object

**Requirements:** PyArrow (included with pandas)

### Preview Table

Quick preview of table contents without loading the full dataset:

```python
from datafolio import DataFolio
import pandas as pd

folio = DataFolio('experiments/analysis')

# Add table
df = pd.DataFrame({'value': range(10000), 'category': ['A', 'B'] * 5000})
folio.add_table('data', df)

# Quick preview (default: 10 rows)
preview = folio.preview_table('data')
print(preview)
#    value category
# 0      0        A
# 1      1        B
# 2      2        A
# ...

# Custom preview size
preview = folio.preview_table('data', n=5)
print(len(preview))  # 5

# Force return type
preview = folio.preview_table('data', n=5, return_type='polars')
```

**Use cases:**
- Quick data inspection
- Schema validation
- Sanity checks before full load

### Complete Example

```python
from datafolio import DataFolio
import pandas as pd

folio = DataFolio('experiments/analysis')

# Add large dataset
df = pd.DataFrame({
    'timestamp': range(1000000),
    'value': range(1000000),
    'status': ['active', 'inactive'] * 500000
})
folio.add_table('large_data', df)

# 1. Check metadata first
info = folio.table_info('large_data')
print(f"Table size: {info['size_mb']:.1f} MB with {info['num_rows']:,} rows")

# 2. Preview to check structure
preview = folio.preview_table('large_data', n=5)
print(f"Columns: {list(preview.columns)}")
print(f"Dtypes: {preview.dtypes.to_dict()}")

# 3. Get path for external tools
path = folio.get_table_path('large_data')

# 4. Use Ibis for complex query
import ibis
con = ibis.duckdb.connect()
table = con.read_parquet(path)

result = (
    table
    .filter(table.status == 'active')
    .filter(table.value > 500000)
    .aggregate(count=table.count(), avg_value=table.value.mean())
)

print(result.execute())

# 5. Save results back to DataFolio
folio.add_table('analysis_result', result.execute(),
    description='Filtered active records > 500k',
    inputs=['large_data'])
```

## Advanced Querying with Ibis

For more complex query operations beyond DataFolio's built-in features—such as joins across multiple tables, window functions, or complex analytics—we recommend using [Ibis](https://ibis-project.org/).

Ibis is a Python dataframe library that provides a unified interface for querying data from multiple backends, including DuckDB. Since DataFolio stores tables as Parquet files, you can easily use Ibis to query them directly.

### Why Ibis?

- **Join tables**: DataFolio's `query_table()` works on single tables, but Ibis can join multiple tables
- **Complex expressions**: Window functions, pivots, advanced aggregations
- **Lazy evaluation**: Build complex queries that execute efficiently
- **Multiple backends**: Same API works with DuckDB, Polars, pandas, SQL databases
- **Type safety**: Strongly typed expressions with excellent IDE support

### Basic Pattern

```python
import ibis
from datafolio import DataFolio

# Create folio with multiple tables
folio = DataFolio('experiments/analytics')

# Connect Ibis to DuckDB
con = ibis.duckdb.connect()

# Get paths to Parquet files
customers_path = folio.path / 'tables' / 'customers.parquet'
orders_path = folio.path / 'tables' / 'orders.parquet'

# Register tables with Ibis
customers = con.read_parquet(customers_path, table_name='customers')
orders = con.read_parquet(orders_path, table_name='orders')

# Query with joins
result = (
    orders
    .join(customers, orders.customer_id == customers.id)
    .group_by(customers.country)
    .aggregate(
        total_orders=orders.count(),
        total_revenue=orders.amount.sum()
    )
)

# Execute and get results
df = result.execute()
```

### Joining Multiple DataFolio Tables

```python
import ibis
from pathlib import Path

folio = DataFolio('experiments/ecommerce')

# Helper to get table path
def get_table_path(name: str) -> Path:
    return folio.path / 'tables' / f'{name}.parquet'

# Connect Ibis to DuckDB
con = ibis.duckdb.connect()

# Load multiple tables
users = con.read_parquet(get_table_path('users'), table_name='users')
sessions = con.read_parquet(get_table_path('sessions'), table_name='sessions')
events = con.read_parquet(get_table_path('events'), table_name='events')

# Complex query with multiple joins
funnel = (
    events
    .join(sessions, events.session_id == sessions.id)
    .join(users, sessions.user_id == users.id)
    .filter(events.event_type.isin(['page_view', 'add_to_cart', 'purchase']))
    .group_by([users.cohort, events.event_type])
    .aggregate(count=events.count())
    .pivot_wider(
        names_from='event_type',
        values_from='count'
    )
)

result = funnel.execute()
```

### Window Functions

```python
import ibis

folio = DataFolio('experiments/timeseries')
con = ibis.duckdb.connect()

# Load time series data
path = folio.path / 'tables' / 'metrics.parquet'
metrics = con.read_parquet(path, table_name='metrics')

# Calculate moving average and rank
analysis = metrics.mutate(
    moving_avg=metrics.value.mean().over(
        ibis.window(order_by=metrics.timestamp, preceding=6, following=0)
    ),
    rank=metrics.value.rank().over(
        ibis.window(partition_by=metrics.category, order_by=metrics.value.desc())
    )
)

result = analysis.execute()
```

### Saving Query Results Back to DataFolio

```python
import ibis
import pandas as pd

folio = DataFolio('experiments/analytics')
con = ibis.duckdb.connect()

# Query multiple tables
# ... build complex query with Ibis ...

# Execute and get pandas DataFrame
result_df = query.execute()

# Save back to DataFolio
folio.add_table('analysis_result', result_df,
    description='Customer cohort analysis',
    inputs=['customers', 'orders', 'sessions'])
```

### Complete Example: Customer Analytics

```python
import ibis
import pandas as pd
from datafolio import DataFolio

# Setup
folio = DataFolio('experiments/customer_analytics')
con = ibis.duckdb.connect()

# Helper function
def load_table(name: str):
    path = folio.path / 'tables' / f'{name}.parquet'
    return con.read_parquet(path, table_name=name)

# Load tables
customers = load_table('customers')
orders = load_table('orders')
products = load_table('products')

# Complex analytics query
customer_summary = (
    orders
    .join(customers, orders.customer_id == customers.id)
    .join(products, orders.product_id == products.id)
    .group_by([
        customers.id,
        customers.name,
        customers.segment,
        customers.region
    ])
    .aggregate(
        total_orders=orders.count(),
        total_spent=orders.amount.sum(),
        avg_order=orders.amount.mean(),
        unique_products=products.id.nunique(),
        first_order=orders.order_date.min(),
        last_order=orders.order_date.max()
    )
    .mutate(
        customer_lifetime_days=(
            customers.last_order - customers.first_order
        ).days,
        avg_days_between_orders=(
            customers.customer_lifetime_days / customers.total_orders
        )
    )
    .order_by(customers.total_spent.desc())
)

# Execute
result = customer_summary.execute()

# Save results
folio.add_table('customer_summary', result,
    description='Customer lifetime value analysis',
    inputs=['customers', 'orders', 'products'])

print(f"Analyzed {len(result)} customers")
```

### When to Use Each Tool

| Task | Tool | Why |
|------|------|-----|
| Single table queries | `folio.query_table()` | Simpler, built-in |
| Simple filtering/limits | `folio.get_table(where=...)` | Most convenient |
| Joins across tables | Ibis | DataFolio doesn't support joins |
| Window functions | Ibis | Advanced SQL features |
| Complex aggregations | Ibis | More expressive API |
| Chunked processing | `folio.iter_table()` | Memory-efficient iteration |

### Best Practices

1. **Use DataFolio for storage** - Let DataFolio handle versioning, lineage, and metadata
2. **Use Ibis for complex queries** - Joins, window functions, pivots
3. **Save results back** - Store important query results in DataFolio with lineage tracking
4. **Document dependencies** - Use `inputs=` parameter when saving derived tables

```python
# Good pattern
folio = DataFolio('experiments/analysis')

# 1. Load data from DataFolio
con = ibis.duckdb.connect()
table1 = con.read_parquet(folio.path / 'tables' / 'table1.parquet')
table2 = con.read_parquet(folio.path / 'tables' / 'table2.parquet')

# 2. Query with Ibis
result = table1.join(table2, ...).aggregate(...).execute()

# 3. Save back to DataFolio with lineage
folio.add_table('combined_analysis', result,
    description='Joined analysis of table1 and table2',
    inputs=['table1', 'table2'])  # Track dependencies!
```

### Resources

- [Ibis Documentation](https://ibis-project.org/)
- [Ibis DuckDB Backend](https://ibis-project.org/backends/duckdb/)
- [Ibis Tutorial](https://ibis-project.org/tutorial/)

## Next Steps

- Learn about [Snapshots](snapshots.md) for versioning experiments
- Check the [API Reference](../reference/api.md) for complete method signatures
- See [Getting Started](getting-started.md) for basic DataFolio usage
- Explore [Ibis](https://ibis-project.org/) for advanced query capabilities
