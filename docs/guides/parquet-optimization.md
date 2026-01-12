# Advanced Parquet Operations

DataFolio stores tables as Parquet files, leveraging Parquet's powerful filtering and optimization capabilities. This guide shows you how to work efficiently with large datasets using column selection, row filtering, and other advanced techniques.

## Why Parquet?

Parquet is a columnar storage format designed for analytics workloads:

- **Column pruning** - Read only the columns you need
- **Row filtering** - Filter data at the file level before loading into memory
- **Compression** - Efficient storage with various compression algorithms
- **Metadata** - Store schema, statistics for query optimization
- **Cloud-optimized** - Works well with S3, GCS, and other cloud storage

## Quick Example: The Problem

```python
# Problem: 100GB table, but you only need 2 columns and 1% of rows
folio = DataFolio('s3://huge-bucket/analysis')

# Bad: Loads entire 100GB into memory! 💥
df = folio.get_table('transactions')  # OOM error!
df_filtered = df[df['amount'] > 1000][['user_id', 'amount']]

# Good: Loads only ~1GB (filtered subset) ✅
df_filtered = folio.get_table('transactions',
    columns=['user_id', 'amount'],
    filters=[('amount', '>', 1000)],
    engine='pyarrow')
```

This is the power of column selection and row filtering - work with massive datasets efficiently.

## Column Selection (Column Pruning)

Read only the columns you need, reducing memory usage and load time.

### Basic Column Selection

```python
folio = DataFolio('experiments/analysis')

# Full table has 50 columns
df_full = folio.get_table('user_events')  # Loads all 50 columns

# Read only specific columns
df_subset = folio.get_table('user_events',
    columns=['user_id', 'timestamp', 'event_type'])  # Loads only 3 columns
```

### Performance Impact

```python
import time

# Example: 10M rows × 50 columns = 4GB
start = time.time()
df_full = folio.get_table('large_table')
full_time = time.time() - start
print(f"Full table: {full_time:.2f}s, {df_full.memory_usage().sum() / 1e9:.2f} GB")
# Full table: 12.3s, 4.2 GB

# Read only 5 columns
start = time.time()
df_subset = folio.get_table('large_table',
    columns=['col1', 'col2', 'col3', 'col4', 'col5'])
subset_time = time.time() - start
print(f"5 columns: {subset_time:.2f}s, {df_subset.memory_usage().sum() / 1e9:.2f} GB")
# 5 columns: 1.8s, 0.4 GB
# 6.8x faster, 10x less memory!
```

### Use Cases

**Data Exploration**
```python
# Quick peek at data without loading everything
summary = folio.get_table('user_data',
    columns=['user_id', 'signup_date', 'country'])

print(summary.head())
print(summary['country'].value_counts())
```

**Feature Engineering**
```python
# Load only features needed for this model
features = ['age', 'income', 'education', 'job_title']
target = 'churn'

df = folio.get_table('training_data',
    columns=features + [target])

X = df[features]
y = df[target]
```

## Row Filtering (Predicate Pushdown)

Filter rows at the Parquet file level before loading into memory. This is **much** more efficient than loading everything and filtering in pandas.

### Basic Filtering

Row filtering requires the `pyarrow` engine:

```python
# Filter rows where amount > 1000
df = folio.get_table('transactions',
    filters=[('amount', '>', 1000)],
    engine='pyarrow')

# Multiple conditions (AND)
df = folio.get_table('transactions',
    filters=[
        ('amount', '>', 1000),
        ('country', '==', 'US')
    ],
    engine='pyarrow')
```

### Filter Syntax

PyArrow supports these filter operators:

```python
# Comparisons
[('age', '>', 18)]           # Greater than
[('age', '>=', 18)]          # Greater than or equal
[('age', '<', 65)]           # Less than
[('age', '<=', 65)]          # Less than or equal
[('status', '==', 'active')] # Equal
[('status', '!=', 'banned')] # Not equal

# Set membership
[('country', 'in', ['US', 'CA', 'MX'])]
[('country', 'not in', ['XX', 'YY'])]
```

### Complex Filters

Combine multiple conditions:

```python
# AND conditions (list of tuples)
df = folio.get_table('users',
    filters=[
        ('age', '>=', 18),
        ('age', '<', 65),
        ('country', 'in', ['US', 'CA', 'UK'])
    ],
    engine='pyarrow')

# OR conditions (list of lists)
# Get users who are either (age < 18) OR (country == 'US')
df = folio.get_table('users',
    filters=[
        [('age', '<', 18)],
        [('country', '==', 'US')]
    ],
    engine='pyarrow')

# Complex: (age >= 18 AND country == 'US') OR (status == 'premium')
df = folio.get_table('users',
    filters=[
        [('age', '>=', 18), ('country', '==', 'US')],
        [('status', '==', 'premium')]
    ],
    engine='pyarrow')
```

### Performance Impact

```python
# Example: 100M rows, filter returns 1M rows

# Bad: Load all 100M rows, then filter (slow!) 💥
df = folio.get_table('events')  # 30s, 8GB memory
df_filtered = df[df['event_type'] == 'purchase']  # Another 5s

# Good: Filter at file level (fast!) ✅
df_filtered = folio.get_table('events',
    filters=[('event_type', '==', 'purchase')],
    engine='pyarrow')  # 2s, 80MB memory
# 17.5x faster, 100x less memory!
```

## Combining Column Selection and Filtering

The real power comes from combining both techniques:

```python
# Dataset: 1B rows × 100 columns = 800GB
# Goal: Get user_id and amount for large transactions in 2024

# Optimal approach: Filter first, select columns
df = folio.get_table('transactions',
    columns=['user_id', 'amount', 'timestamp'],  # Only 3 columns
    filters=[
        ('timestamp', '>=', '2024-01-01'),
        ('timestamp', '<', '2025-01-01'),
        ('amount', '>', 10000)
    ],
    engine='pyarrow')

# Loads only ~500MB instead of 800GB! 🎉
```

### Real-World Example: Fraud Detection

```python
from datafolio import DataFolio

folio = DataFolio('s3://fraud-detection/production',
    cache_enabled=True)  # Cache for repeated access

# Full dataset: 10TB of transaction data
# We need: Recent high-value transactions from flagged countries

df = folio.get_table('transactions',
    # Only these columns
    columns=[
        'transaction_id',
        'user_id',
        'amount',
        'country',
        'timestamp',
        'merchant_id'
    ],
    # Filter to recent, high-value, suspicious countries
    filters=[
        ('timestamp', '>=', '2024-12-01'),  # Last month
        ('amount', '>', 5000),              # Large amounts
        ('country', 'in', ['XX', 'YY', 'ZZ'])  # Flagged countries
    ],
    engine='pyarrow')

# Result: ~2GB loaded instead of 10TB
# Now train fraud detection model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
# ... feature engineering and training ...

# Save everything
folio.add_table('flagged_transactions', df)
folio.add_model('fraud_detector', model,
    inputs=['transactions'])
folio.metadata['filtered_rows'] = len(df)
folio.metadata['date_range'] = '2024-12'
```

## Working with Large Datasets

### Pattern 1: Iterative Processing

For datasets too large to fit in memory, process in chunks:

```python
# Get list of unique dates
info = folio.get_table_info('huge_dataset')
# Assume we know the date range: 2020-01-01 to 2024-12-31

from datetime import datetime, timedelta
import pandas as pd

results = []
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)
current_date = start_date

while current_date < end_date:
    next_date = current_date + timedelta(days=30)

    # Load one month at a time
    df_month = folio.get_table('huge_dataset',
        filters=[
            ('date', '>=', current_date.strftime('%Y-%m-%d')),
            ('date', '<', next_date.strftime('%Y-%m-%d'))
        ],
        engine='pyarrow')

    # Process this month
    monthly_stats = df_month.groupby('category').agg({
        'amount': ['sum', 'mean', 'count']
    })
    results.append(monthly_stats)

    current_date = next_date

# Combine results
final_results = pd.concat(results)
folio.add_table('monthly_statistics', final_results)
```

### Pattern 2: Sample First, Analyze Later

```python
# 1. Get a sample to develop your analysis
df_sample = folio.get_table('massive_dataset',
    filters=[('random_partition', '==', 0)],  # 1% sample
    engine='pyarrow')

# 2. Develop and test your analysis on the sample
def analyze_data(df):
    # Your analysis logic
    return df.groupby('category').agg({'value': 'mean'})

# 3. Test on sample
result_sample = analyze_data(df_sample)
print("Sample results:", result_sample)

# 4. Run on full dataset
df_full = folio.get_table('massive_dataset')
result_full = analyze_data(df_full)
folio.add_table('analysis_results', result_full)
```

### Pattern 3: Progressive Loading

```python
# Load increasingly detailed data as needed

# Step 1: Load summary columns for filtering
df_summary = folio.get_table('user_behavior',
    columns=['user_id', 'total_spend', 'num_purchases'])

# Step 2: Identify users of interest
high_value_users = df_summary[df_summary['total_spend'] > 10000]['user_id']

# Step 3: Load detailed data only for those users
df_detailed = folio.get_table('user_behavior',
    filters=[('user_id', 'in', high_value_users.tolist())],
    engine='pyarrow')
# All columns, but only for high-value users
```

## Memory Optimization Tips

### 1. Check Available Columns First

```python
# See what columns are available without loading data
info = folio.get_table_info('my_table')
print("Available columns:", info.get('columns', []))
print("Total rows:", info.get('num_rows'))
print("Total columns:", info.get('num_cols'))

# Now load only what you need
df = folio.get_table('my_table',
    columns=['col1', 'col2', 'col3'])
```

### 2. Use Appropriate Data Types

```python
# Load with optimized dtypes
df = folio.get_table('user_data',
    columns=['user_id', 'age', 'country'])

# Convert to more efficient types
df['user_id'] = df['user_id'].astype('uint32')  # If IDs are small integers
df['age'] = df['age'].astype('uint8')           # Age fits in uint8
df['country'] = df['country'].astype('category') # Categorical for repeated strings

# Memory savings can be 50-90%!
```

### 3. Read Only What You'll Use

```python
# Bad: Load everything, then filter
df = folio.get_table('events')  # 10GB
active_users = df[df['status'] == 'active']['user_id'].unique()

# Good: Filter first, then load
df_active = folio.get_table('events',
    columns=['user_id', 'status'],
    filters=[('status', '==', 'active')],
    engine='pyarrow')  # 100MB
active_users = df_active['user_id'].unique()
```

## Advanced: PyArrow Tables

For maximum performance, work with PyArrow tables directly:

```python
import pyarrow.parquet as pq

# Get the file path
folio = DataFolio('experiments/data')

# For included tables, get the bundle path
bundle_path = folio.path
table_info = folio.get_table_info('my_table')
file_path = f"{bundle_path}/tables/{table_info['filename']}"

# Read as PyArrow table (zero-copy, very fast)
table = pq.read_table(file_path,
    columns=['col1', 'col2'],
    filters=[('col1', '>', 100)])

# Work with PyArrow (often faster than pandas)
print(f"Rows: {table.num_rows}")
print(f"Schema: {table.schema}")

# Convert to pandas only when needed
df = table.to_pandas()

# Or: Reference tables work directly
ref_path = folio.get_data_path('referenced_table')
table = pq.read_table(ref_path, columns=['a', 'b'])
```

## Integration with DuckDB

For complex queries on large datasets, use DuckDB:

```python
import duckdb

folio = DataFolio('s3://my-bucket/analysis')

# Get path to table
table_path = folio.get_data_path('large_table')

# Query with DuckDB (very fast, minimal memory)
result = duckdb.query(f"""
    SELECT
        category,
        AVG(amount) as avg_amount,
        COUNT(*) as count
    FROM read_parquet('{table_path}')
    WHERE amount > 100
    AND date >= '2024-01-01'
    GROUP BY category
    ORDER BY avg_amount DESC
    LIMIT 100
""").to_df()

# Save results
folio.add_table('category_summary', result,
    description='Top 100 categories by average amount',
    inputs=['large_table'])
```

## Cloud Storage Optimization

### Caching for Repeated Access

```python
# Enable caching for cloud data
folio = DataFolio('s3://my-bucket/experiment',
    cache_enabled=True,
    cache_dir='/fast/local/disk')

# First access: Downloads from S3 (slow)
df1 = folio.get_table('data',
    filters=[('country', '==', 'US')],
    engine='pyarrow')  # 30s

# Second access: Reads from local cache (fast!)
df2 = folio.get_table('data',
    filters=[('country', '==', 'CA')],
    engine='pyarrow')  # 0.5s

# Note: Different filters still benefit from cached file
```

### Minimize Data Transfer

```python
# Bad: Transfer entire file from S3, then filter locally
folio = DataFolio('s3://huge-bucket/data')
df = folio.get_table('transactions')  # Transfers 50GB!
df_us = df[df['country'] == 'US']     # Then filters locally

# Good: Filter on S3, transfer only results
df_us = folio.get_table('transactions',
    columns=['id', 'amount', 'date'],
    filters=[('country', '==', 'US')],
    engine='pyarrow')  # Transfers only 500MB
```

## Best Practices

### 1. Always Specify Columns

```python
# Default: Read all columns
df = folio.get_table('data')  # All columns

# Better: Specify exactly what you need
df = folio.get_table('data',
    columns=['essential_col1', 'essential_col2'])
```

### 2. Use Filters for Time-Series Data

```python
# Analyze only recent data
df = folio.get_table('events',
    filters=[('timestamp', '>=', '2024-12-01')],
    engine='pyarrow')
```

### 3. Combine with Good Data Organization

```python
# When saving data, include useful columns for filtering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

folio.add_table('data_with_partitions', df)

# Later: Fast filtering
df_2024 = folio.get_table('data_with_partitions',
    filters=[('year', '==', 2024)],
    engine='pyarrow')
```

### 4. Check Your Filters Work

```python
# Verify filter syntax before processing huge files
df_test = folio.get_table('data',
    filters=[('category', 'in', ['A', 'B'])],
    engine='pyarrow')

if len(df_test) > 0:
    print("Filter works! Processing full dataset...")
else:
    print("No data matched filter - check column names/values")
```

## Common Patterns

### Pattern: Top N Analysis

```python
# Find top users by spend, without loading everyone
df_all = folio.get_table('user_spending',
    columns=['user_id', 'total_spend'])

top_users = df_all.nlargest(1000, 'total_spend')['user_id']

# Load full details only for top users
df_top = folio.get_table('user_details',
    filters=[('user_id', 'in', top_users.tolist())],
    engine='pyarrow')
```

### Pattern: Time-Window Analysis

```python
from datetime import datetime, timedelta

# Analyze last 7 days
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

df = folio.get_table('events',
    columns=['user_id', 'event_type', 'timestamp', 'value'],
    filters=[
        ('timestamp', '>=', start_date.strftime('%Y-%m-%d')),
        ('timestamp', '<', end_date.strftime('%Y-%m-%d'))
    ],
    engine='pyarrow')

# Analyze this week's data
summary = df.groupby('event_type').agg({'value': 'sum'})
folio.add_table(f'weekly_summary_{start_date:%Y%m%d}', summary)
```

### Pattern: A/B Test Analysis

```python
# Load only treatment group data
df_treatment = folio.get_table('experiment_data',
    columns=['user_id', 'conversion', 'revenue'],
    filters=[('group', '==', 'treatment')],
    engine='pyarrow')

df_control = folio.get_table('experiment_data',
    columns=['user_id', 'conversion', 'revenue'],
    filters=[('group', '==', 'control')],
    engine='pyarrow')

# Compare
print(f"Treatment: {df_treatment['conversion'].mean():.2%}")
print(f"Control: {df_control['conversion'].mean():.2%}")
```

## Troubleshooting

### Error: "filter '...' not supported"

Use `engine='pyarrow'`:
```python
# Wrong
df = folio.get_table('data', filters=[('a', '>', 1)])  # Error!

# Right
df = folio.get_table('data',
    filters=[('a', '>', 1)],
    engine='pyarrow')  # Works!
```

### Error: "Column 'x' does not exist"

Check available columns:
```python
info = folio.get_table_info('data')
print("Available columns:", info.get('columns', []))
```

### Slow Performance Despite Filtering

- Ensure you're using `engine='pyarrow'`
- Check filter selectivity (how many rows match)
- Consider breaking into smaller chunks
- Enable caching for cloud data

## Summary

| Technique | Use When | Performance Gain |
|-----------|----------|-----------------|
| **Column selection** | You need only some columns | 5-50x faster, 5-50x less memory |
| **Row filtering** | You need only some rows | 10-1000x faster, 10-1000x less memory |
| **Both combined** | Large datasets | 50-10000x faster! |
| **Caching** | Cloud storage, repeated access | 10-100x faster on subsequent reads |
| **PyArrow engine** | Any filtering | Required for filters |
| **Chunked processing** | Data too large for memory | Enables processing unlimited data |

**Golden Rule:** Always filter and select at the Parquet level, never load everything into memory first!

## Learn More

- [Getting Started Guide](getting-started.md) - Basic DataFolio usage
- [Snapshots Guide](snapshots.md) - Version your filtered datasets
- [DataFolio API Reference](../reference/datafolio-api.md) - All `get_table()` options
- [Pandas read_parquet docs](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html) - Full parameter reference
- [PyArrow filtering docs](https://arrow.apache.org/docs/python/parquet.html#filtering) - Advanced filter syntax
