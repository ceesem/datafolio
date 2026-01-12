# Caching for Fast Remote Access

DataFolio's caching system dramatically speeds up access to cloud-stored data by maintaining a local copy. This guide shows you how to use caching effectively for remote bundles.

## Why Use Caching?

### The Problem

Working with cloud-stored bundles can be slow due to network latency:

```python
# Remote bundle on S3
folio = DataFolio('s3://my-bucket/experiments/model_v1')

# Every access downloads from S3 (slow!)
df1 = folio.get_table('results')  # 30 seconds
df2 = folio.get_table('features') # 25 seconds
model = folio.get_model('classifier') # 15 seconds
# Total: 70 seconds just for data access
```

### The Solution: Local Caching

Enable caching to download once, read many times:

```python
# Enable caching
folio = DataFolio('s3://my-bucket/experiments/model_v1',
    cache_enabled=True)

# First access: Downloads and caches (30s)
df1 = folio.get_table('results')  # 30 seconds

# Second access: Reads from cache (fast!)
df1_again = folio.get_table('results')  # 0.1 seconds
# 300x faster! 🚀
```

## Quick Start

### Basic Usage

```python
from datafolio import DataFolio

# Enable caching with defaults
folio = DataFolio('s3://my-bucket/data',
    cache_enabled=True)  # That's it!

# Work normally - caching is automatic
df = folio.get_table('my_data')  # Cached on first access
model = folio.get_model('my_model')  # Cached on first access

# Check cache statistics
status = folio.cache_status()
print(f"Cache hits: {status['cache_hits']}")
print(f"Cache misses: {status['cache_misses']}")
print(f"Total size: {status['total_size_bytes'] / 1e9:.2f} GB")
```

### Custom Cache Directory

```python
# Store cache on a fast SSD
folio = DataFolio('gs://my-bucket/data',
    cache_enabled=True,
    cache_dir='/fast/ssd/cache')

# Or use a specific directory for this bundle
folio = DataFolio('s3://bucket/experiment1',
    cache_enabled=True,
    cache_dir='/data/cache/experiment1')
```

## How Caching Works

### Cache Storage Structure

DataFolio creates a cache directory with this structure:

```
~/.datafolio_cache/               # Default cache directory
├── bundles/
│   └── <bundle-id>/              # One directory per bundle
│       ├── tables/
│       │   └── results.parquet   # Cached table files
│       ├── models/
│       │   └── classifier.joblib # Cached model files
│       └── ...
└── .locks/                       # Lock files for thread safety
```

With explicit `cache_dir`:
```
/my/cache/                        # Your specified directory
├── tables/
│   └── results.parquet
├── models/
│   └── classifier.joblib
└── ...
```

### Cache Behavior

1. **First access**: Downloads from remote → saves to cache → returns data
2. **Subsequent accesses**: Reads from cache (no network access)
3. **Cache invalidation**: Automatic when remote file changes

```python
folio = DataFolio('s3://bucket/data', cache_enabled=True)

# Access 1: Cache miss (downloads from S3)
df = folio.get_table('data')  # Downloads, caches, returns
# Status: 1 miss, 0 hits

# Access 2: Cache hit (reads from local cache)
df = folio.get_table('data')  # Reads from cache
# Status: 1 miss, 1 hit

# Access 3: Cache hit
df = folio.get_table('data')  # Reads from cache
# Status: 1 miss, 2 hits
```

### Checksum-Based Invalidation

DataFolio uses checksums to detect changes:

```python
folio = DataFolio('s3://bucket/data', cache_enabled=True)

# First access: Downloads and caches
df1 = folio.get_table('results')  # Cache miss

# Meanwhile, someone updates the remote file...
# (e.g., another process overwrites results.parquet)

# Next access: Detects change, re-downloads
df2 = folio.get_table('results')  # Cache miss (checksum changed)
# Automatic cache invalidation! ✅
```

## Configuration Options

### Cache Directory

```python
# Default: ~/.datafolio_cache/bundles/<bundle-id>
folio = DataFolio('s3://bucket/data', cache_enabled=True)

# Custom: Exact directory for this bundle
folio = DataFolio('s3://bucket/data',
    cache_enabled=True,
    cache_dir='/fast/ssd/my_cache')

# Environment variable
import os
os.environ['DATAFOLIO_CACHE_DIR'] = '/shared/cache'
folio = DataFolio('s3://bucket/data', cache_enabled=True)
```

### Cache Enabled/Disabled

```python
# Disable caching (default)
folio = DataFolio('s3://bucket/data')
# or explicitly:
folio = DataFolio('s3://bucket/data', cache_enabled=False)

# Enable caching
folio = DataFolio('s3://bucket/data', cache_enabled=True)

# Environment variable
os.environ['DATAFOLIO_CACHE_ENABLED'] = 'true'
folio = DataFolio('s3://bucket/data')  # Caching enabled
```

### Multiple Bundles, Shared Cache

```python
# Default cache_dir: Each bundle gets its own subdirectory
cache_base = '~/.datafolio_cache'  # Default

folio1 = DataFolio('s3://bucket/exp1', cache_enabled=True)
# Cache: ~/.datafolio_cache/bundles/<exp1-id>/

folio2 = DataFolio('s3://bucket/exp2', cache_enabled=True)
# Cache: ~/.datafolio_cache/bundles/<exp2-id>/

# Custom: Separate caches for different projects
folio_a = DataFolio('s3://bucket/project_a',
    cache_enabled=True,
    cache_dir='/cache/project_a')

folio_b = DataFolio('s3://bucket/project_b',
    cache_enabled=True,
    cache_dir='/cache/project_b')
```

## Cache Management

### Check Cache Status

```python
folio = DataFolio('s3://bucket/data', cache_enabled=True)

# Work with data
df = folio.get_table('table1')
df = folio.get_table('table2')
model = folio.get_model('model1')

# Check cache statistics
status = folio.cache_status()

print(f"Cache enabled: {status['cache_enabled']}")
print(f"Cache directory: {status['cache_dir']}")
print(f"Total files: {status['total_files']}")
print(f"Total size: {status['total_size_bytes'] / 1e9:.2f} GB")
print(f"Cache hits: {status['cache_hits']}")
print(f"Cache misses: {status['cache_misses']}")
print(f"Hit rate: {status['cache_hits'] / (status['cache_hits'] + status['cache_misses']) * 100:.1f}%")
```

Example output:
```
Cache enabled: True
Cache directory: /Users/me/.datafolio_cache/bundles/abc123
Total files: 15
Total size: 2.3 GB
Cache hits: 45
Cache misses: 15
Hit rate: 75.0%
```

### Clear Cache

```python
# Clear entire cache for this bundle
folio.clear_cache()

# Verify cache is empty
status = folio.cache_status()
assert status['total_files'] == 0
assert status['total_size_bytes'] == 0
```

### Invalidate Specific Item

```python
# Force re-download on next access
folio.invalidate_cache('my_table')

# Next access will download fresh copy
df = folio.get_table('my_table')  # Cache miss (downloads)
```

### Refresh Cache

```python
# Re-download all cached items
folio.refresh_cache()

# Useful when you know remote data has changed
# but checksums aren't updated yet
```

## Performance Examples

### Example 1: Iterative Data Analysis

```python
# Working with cloud data in a Jupyter notebook
folio = DataFolio('s3://analytics/user_study',
    cache_enabled=True)

# Analysis iteration 1: Download data (slow)
df = folio.get_table('user_data')  # 45 seconds
df.head()

# Oops, need to filter differently
df_filtered = folio.get_table('user_data')  # 0.1 seconds (cached!)
df_filtered[df_filtered['age'] > 18]

# Try different aggregation
df_agg = folio.get_table('user_data')  # 0.1 seconds (cached!)
df_agg.groupby('country').mean()

# Total time: 45s instead of 135s (3x access)
# Cache saved 90 seconds!
```

### Example 2: Model Training Pipeline

```python
# Training pipeline with multiple runs
folio = DataFolio('s3://ml-data/experiment',
    cache_enabled=True,
    cache_dir='/fast/nvme/cache')

# First training run
df_train = folio.get_table('training_data')  # 2 minutes (5GB download)
df_test = folio.get_table('test_data')      # 1 minute (2GB download)
model = train_model(df_train, df_test)
# Total: 3 minutes data loading

# Second training run (different hyperparameters)
df_train = folio.get_table('training_data')  # 0.5 seconds (cached!)
df_test = folio.get_table('test_data')      # 0.2 seconds (cached!)
model = train_model(df_train, df_test)
# Total: 0.7 seconds data loading

# 10 training runs: 3 min + (9 × 0.7s) = ~3.1 minutes
# Without caching: 10 × 3 min = 30 minutes
# Cache saved 27 minutes! 🚀
```

### Example 3: Team Collaboration

```python
# Shared cache on network drive
folio = DataFolio('s3://team-bucket/shared-analysis',
    cache_enabled=True,
    cache_dir='/nfs/team-cache/analysis')

# Alice downloads data first
df = folio.get_table('large_dataset')  # 10 minutes (downloads)

# Bob uses same cache later
folio_bob = DataFolio('s3://team-bucket/shared-analysis',
    cache_enabled=True,
    cache_dir='/nfs/team-cache/analysis')
df_bob = folio_bob.get_table('large_dataset')  # 5 seconds (from cache!)

# Charlie also benefits
folio_charlie = DataFolio('s3://team-bucket/shared-analysis',
    cache_enabled=True,
    cache_dir='/nfs/team-cache/analysis')
df_charlie = folio_charlie.get_table('large_dataset')  # 5 seconds!

# Team total: 10 min + 5s + 5s vs 30 min without cache
```

## Use Cases

### Local Development with Cloud Data

```python
# Develop locally with production data from S3
folio = DataFolio('s3://production/analytics',
    cache_enabled=True)

# First run: Downloads data
df = folio.get_table('transactions')  # Slow
# Develop your analysis...

# Restart notebook, re-run cells: Reads from cache (fast!)
df = folio.get_table('transactions')  # Fast
# Much better development experience!
```

### CI/CD Pipelines

```python
# Cache data across CI runs
import os

# In CI environment
cache_dir = os.environ.get('CI_CACHE_DIR', '/tmp/cache')
folio = DataFolio('s3://ml-models/production',
    cache_enabled=True,
    cache_dir=cache_dir)

# First CI run: Downloads models
model = folio.get_model('production_model')  # Downloads
run_tests(model)

# Subsequent CI runs: Uses cache
model = folio.get_model('production_model')  # From cache
run_tests(model)  # Much faster CI!
```

### Offline Work

```python
# Pre-populate cache while online
folio = DataFolio('s3://bucket/data',
    cache_enabled=True,
    cache_dir='/local/cache')

# Download everything while connected
df1 = folio.get_table('data1')
df2 = folio.get_table('data2')
model = folio.get_model('model')

# Now work offline (airplane, no internet)
# All data available from cache!
df1 = folio.get_table('data1')  # Works offline! ✈️
```

## Best Practices

### 1. Enable Caching for Remote Bundles

```python
# Good: Enable caching for cloud bundles
folio = DataFolio('s3://bucket/data', cache_enabled=True)

# Unnecessary: Don't cache local bundles
folio = DataFolio('/local/path/data')  # Already local
```

### 2. Use Fast Storage for Cache

```python
# Best: NVMe SSD
folio = DataFolio('s3://bucket/data',
    cache_enabled=True,
    cache_dir='/nvme/cache')

# Good: Regular SSD
cache_dir='/ssd/cache'

# Slow: HDD (still faster than network!)
cache_dir='/hdd/cache'
```

### 3. Monitor Cache Size

```python
# Check cache size periodically
status = folio.cache_status()
cache_gb = status['total_size_bytes'] / 1e9

if cache_gb > 100:  # If cache > 100GB
    print(f"Cache is large: {cache_gb:.1f} GB")
    print("Consider clearing old data")
    # folio.clear_cache()
```

### 4. Clear Cache When Switching Contexts

```python
# Finished with this analysis
folio = DataFolio('s3://bucket/old_analysis', cache_enabled=True)
# ... work done ...
folio.clear_cache()  # Free up disk space

# Start new analysis
folio = DataFolio('s3://bucket/new_analysis', cache_enabled=True)
```

### 5. Use Separate Caches for Different Projects

```python
# Project A
folio_a = DataFolio('s3://bucket/project_a',
    cache_enabled=True,
    cache_dir='/cache/project_a')

# Project B
folio_b = DataFolio('s3://bucket/project_b',
    cache_enabled=True,
    cache_dir='/cache/project_b')

# Easy to manage and clear separately
```

## Interaction with Parquet Filtering

Caching affects Parquet filtering performance:

### Without Caching (Predicate Pushdown)

```python
# No caching: Filter on S3, download only matching rows
folio = DataFolio('s3://bucket/data')  # cache_enabled=False

df = folio.get_table('huge_table',
    filters=[('country', '==', 'US')],
    engine='pyarrow')
# Downloads: ~1GB (filtered data only)
# Time: 30 seconds
```

### With Caching (Download Then Filter)

```python
# With caching: Download full file, then filter locally
folio = DataFolio('s3://bucket/data', cache_enabled=True)

# First access: Downloads full file
df = folio.get_table('huge_table',
    filters=[('country', '==', 'US')],
    engine='pyarrow')
# Downloads: 100GB (full file)
# Time: 10 minutes
# But subsequent accesses are very fast!

# Second access: Filters cached file
df_ca = folio.get_table('huge_table',
    filters=[('country', '==', 'CA')],
    engine='pyarrow')
# Downloads: 0GB (from cache)
# Time: 5 seconds
```

**Guideline**: For one-time queries on large files, disable caching to use predicate pushdown. For repeated queries, enable caching.

See [Parquet Optimization Guide](parquet-optimization.md) for more details.

## Troubleshooting

### Cache Not Working

**Problem**: Data still downloads every time

**Check**:
```python
status = folio.cache_status()
print(f"Cache enabled: {status['cache_enabled']}")
print(f"Cache dir: {status['cache_dir']}")
print(f"Cache hits: {status['cache_hits']}")
```

**Solutions**:
- Verify `cache_enabled=True` in constructor
- Check cache directory is writable
- Check disk space available

### Stale Cache Data

**Problem**: Cache contains old data

**Solution**:
```python
# Option 1: Invalidate specific item
folio.invalidate_cache('my_table')

# Option 2: Refresh all cache
folio.refresh_cache()

# Option 3: Clear and rebuild
folio.clear_cache()
df = folio.get_table('my_table')  # Fresh download
```

### Cache Too Large

**Problem**: Cache consuming too much disk space

**Solutions**:
```python
# Check current size
status = folio.cache_status()
print(f"Cache size: {status['total_size_bytes'] / 1e9:.2f} GB")

# Clear cache
folio.clear_cache()

# Or: Use smaller cache_dir with limited space
folio = DataFolio('s3://bucket/data',
    cache_enabled=True,
    cache_dir='/limited/disk/cache')  # E.g., 50GB partition
```

### Permission Errors

**Problem**: Can't write to cache directory

**Solution**:
```python
# Use a directory you have write access to
folio = DataFolio('s3://bucket/data',
    cache_enabled=True,
    cache_dir='/home/me/datafolio_cache')  # Your home directory

# Or create directory first
import os
os.makedirs('/tmp/my_cache', exist_ok=True)
folio = DataFolio('s3://bucket/data',
    cache_enabled=True,
    cache_dir='/tmp/my_cache')
```

### Cache Hits Not Improving Performance

**Problem**: Cache hits still slow

**Possible causes**:
1. **Slow cache storage**: HDD instead of SSD
   ```python
   # Move cache to faster storage
   folio = DataFolio('s3://bucket/data',
       cache_enabled=True,
       cache_dir='/fast/ssd/cache')  # Use SSD
   ```

2. **Large files**: Even local reads take time
   ```python
   # Use column selection to read less data
   df = folio.get_table('huge_table',
       columns=['id', 'value'])  # Smaller, faster
   ```

3. **CPU bottleneck**: Parquet decompression
   ```python
   # Check if CPU is bottleneck
   import time
   start = time.time()
   df = folio.get_table('compressed_data')
   print(f"Time: {time.time() - start:.2f}s")
   # If slow despite cache hit, compression/CPU may be bottleneck
   ```

## Advanced: Cache Internals

### Cache Key Generation

DataFolio generates cache keys based on:
- Item name
- Item type (table, model, etc.)
- Checksum (if available)

```python
# Internally, cache key might be:
# tables/my_data.parquet → cached as <cache_dir>/tables/my_data.parquet
```

### Thread Safety

Cache operations are thread-safe:

```python
from concurrent.futures import ThreadPoolExecutor

folio = DataFolio('s3://bucket/data', cache_enabled=True)

def load_table(name):
    return folio.get_table(name)

# Multiple threads can safely access cache
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(load_table, f'table_{i}') for i in range(10)]
    results = [f.result() for f in futures]
# No race conditions! ✅
```

### Cache Locking

DataFolio uses file locks to prevent corruption:
```
~/.datafolio_cache/
└── .locks/
    └── <bundle-id>.lock  # Lock file for this bundle
```

## Performance Benchmarks

| Scenario | No Cache | With Cache (First) | With Cache (Subsequent) | Speedup |
|----------|----------|-------------------|------------------------|---------|
| 100MB S3 file | 5s | 5s | 0.1s | 50x |
| 1GB S3 file | 45s | 45s | 0.5s | 90x |
| 10GB S3 file | 8m | 8m | 5s | 96x |
| 100GB S3 file | 80m | 80m | 45s | 107x |
| 10 small files | 30s | 30s | 1s | 30x |

**Note**: Speedup is for cache hits. First access has no speedup (must download).

## Summary

| Feature | Description | Command |
|---------|-------------|---------|
| **Enable caching** | Cache remote data locally | `cache_enabled=True` |
| **Set cache dir** | Specify cache location | `cache_dir='/path'` |
| **Check status** | View cache statistics | `folio.cache_status()` |
| **Clear cache** | Delete all cached files | `folio.clear_cache()` |
| **Invalidate item** | Force re-download | `folio.invalidate_cache('name')` |
| **Refresh cache** | Re-download everything | `folio.refresh_cache()` |

**When to use caching:**
- ✅ Remote bundles (S3, GCS, etc.)
- ✅ Repeated data access
- ✅ Iterative development
- ✅ Team collaboration with shared cache
- ✅ Offline work preparation

**When NOT to use caching:**
- ❌ Local bundles (already fast)
- ❌ One-time data access
- ❌ Limited disk space
- ❌ Very large files with selective filters (use predicate pushdown)

## Learn More

- [Getting Started Guide](getting-started.md) - Basic DataFolio usage
- [Parquet Optimization Guide](parquet-optimization.md) - Filtering and column selection
- [Snapshots Guide](snapshots.md) - Version your cached data
- [DataFolio API Reference](../reference/datafolio-api.md) - Cache management methods
