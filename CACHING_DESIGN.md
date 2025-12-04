# DataFolio Local Caching System - Design Document

**Author:** Claude
**Date:** 2024-12-03
**Status:** Proposal
**Target Version:** 0.5.0

---

## Executive Summary

This document proposes a local caching layer for cloud-based DataFolios to improve performance and reduce redundant data transfers. The system uses MD5 checksums + configurable TTL (30-60 min) to validate cache freshness while maintaining data integrity.

**Key Benefits:**
- 10-100x faster access to frequently-used cloud data
- Reduced cloud storage costs (fewer GET requests)
- Offline access to recently-used data
- Transparent integration with existing API

---

## Use Cases

### Use Case 1: Dynamic Collaboration (1:1 Research)

**Scenario:** Alice and Bob collaborating on active experiment

```python
# Alice's notebook (updates data frequently)
folio = DataFolio('gs://shared-project/experiment-001')
folio.add('training_data', updated_dataframe)  # Uploads to cloud

# Bob's notebook (runs analysis repeatedly)
folio = DataFolio('gs://shared-project/experiment-001', cache_enabled=True)
for epoch in range(100):
    data = folio.get('training_data')  # Uses cache, checks for updates every 30min
    train_model(data)
```

**Requirements:**
- Fast local access (avoid re-downloading every iteration)
- Detect updates from collaborators (TTL-based checks)
- Minimal overhead when data hasn't changed
- Clear warnings when using potentially stale data

**Expected Behavior:**
- First `get()`: Downloads from cloud, caches locally (~2-10s for large files)
- Subsequent `get()` within TTL: Uses cache instantly (~0.01s)
- After TTL expires: Checks cloud manifest, re-downloads only if changed
- If cloud updated: Downloads new version, warns user "training_data updated by alice@email.com"

---

### Use Case 2: Static Public Snapshots

**Scenario:** Research paper with immutable public data

```python
# Published paper data (never changes)
folio = DataFolio('gs://public-data/v1dd-paper-2024', cache_enabled=True, cache_ttl=86400)

# First researcher downloads once
df = folio.get('soma_features')  # Downloads ~500MB, caches

# Second notebook (same user, different session)
df = folio.get('soma_features')  # Instant load from cache

# Third researcher (different user, same institution)
df = folio.get('soma_features')  # Could share cache if configured
```

**Requirements:**
- Download once, use forever (immutable data)
- Verify integrity (checksum validation)
- Share cache across sessions/notebooks
- Optional: Institutional cache sharing

**Expected Behavior:**
- First access: Downloads and caches with checksum verification
- All subsequent access: Instant load from cache
- No remote checks for static snapshots (user-configurable)
- Cache survives restarts, kernel crashes, etc.

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      DataFolio Instance                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    Cache Layer                         │ │
│  │  • TTL validation                                      │ │
│  │  • Checksum verification                               │ │
│  │  • Cache metadata management                           │ │
│  └────────────────────────────────────────────────────────┘ │
│         ↓ (cache miss)              ↑ (cache hit)           │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │  Remote Storage  │        │   Local Cache    │          │
│  │  (cloud/network) │        │  (~/.datafolio)  │          │
│  └──────────────────┘        └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Cache Directory Structure

```
~/.datafolio_cache/                           # Default cache root
├── config.json                               # Global cache config
├── index.json                                # Cache index (fast lookup)
├── bundles/                                  # Per-bundle caches
│   ├── gs_bucket_name_bundle_001/            # Normalized bundle path
│   │   ├── .cache_meta.json                  # Cache metadata
│   │   ├── tables/
│   │   │   ├── training_data.parquet
│   │   │   └── validation_data.parquet
│   │   ├── models/
│   │   │   └── classifier.pkl
│   │   └── artifacts/
│   │       └── plot.png
│   └── local_path_another_bundle/
│       └── ...
└── .locks/                                   # Lock files for concurrency
    └── gs_bucket_name_bundle_001.lock
```

**Design Decisions:**
- **Per-bundle caching:** Preserves DataFolio structure, easy to understand
- **Normalized paths:** `gs://bucket/name` → `gs_bucket_name` (filesystem-safe)
- **Separate locks directory:** Avoids lock file proliferation

---

### Cache Metadata Format

**File:** `~/.datafolio_cache/bundles/{bundle_id}/.cache_meta.json`

```json
{
  "schema_version": "1.0",
  "bundle_path": "gs://shared-project/experiment-001",
  "bundle_checksum": "abc123...",
  "created_at": "2024-03-15T10:00:00Z",
  "last_manifest_check": "2024-03-15T10:30:00Z",
  "cache_config": {
    "ttl_seconds": 1800,
    "checksum_algorithm": "md5",
    "enabled": true
  },
  "items": {
    "training_data": {
      "item_type": "included_table",
      "filename": "training.parquet",
      "remote_checksum": "def456...",
      "local_checksum": "def456...",
      "cached_at": "2024-03-15T10:30:00Z",
      "last_verified": "2024-03-15T10:30:00Z",
      "size_bytes": 1024000,
      "access_count": 42,
      "last_accessed": "2024-03-15T11:00:00Z"
    },
    "validation_data": {
      "item_type": "included_table",
      "filename": "validation.parquet",
      "remote_checksum": "ghi789...",
      "local_checksum": "ghi789...",
      "cached_at": "2024-03-15T10:31:00Z",
      "last_verified": "2024-03-15T10:31:00Z",
      "size_bytes": 512000,
      "access_count": 15,
      "last_accessed": "2024-03-15T10:45:00Z"
    }
  },
  "stats": {
    "total_size_bytes": 1536000,
    "total_items": 2,
    "cache_hits": 57,
    "cache_misses": 2,
    "remote_checks": 4
  }
}
```

**Key Fields:**
- `bundle_checksum`: MD5 of manifest, detects structural changes
- `remote_checksum` vs `local_checksum`: Validates data integrity
- `last_verified`: TTL base timestamp
- `access_count`: For LRU eviction (future)
- `stats`: Performance monitoring

---

### Cache Validation Logic

**Decision Tree for `folio.get(item_name)`:**

```
START: User calls folio.get('item_name')
│
├─ Is caching enabled?
│  ├─ NO → Fetch from remote (existing behavior)
│  └─ YES ↓
│
├─ Does item exist in local cache?
│  ├─ NO → CACHE MISS
│  │       ├─ Fetch from remote
│  │       ├─ Compute checksum
│  │       ├─ Write to cache
│  │       └─ Update metadata
│  │       └─ RETURN data
│  └─ YES ↓
│
├─ Has TTL expired? (now - last_verified > ttl_seconds)
│  ├─ NO → CACHE HIT (fast path)
│  │       ├─ Load from local cache
│  │       ├─ Update access stats
│  │       └─ RETURN data
│  └─ YES ↓
│
├─ Check remote manifest for changes
│  ├─ Network error? → Use stale cache + warn user
│  ├─ Manifest unchanged? → CACHE HIT
│  │    ├─ Update last_verified timestamp
│  │    ├─ Load from local cache
│  │    └─ RETURN data
│  └─ Item checksum changed? → CACHE INVALIDATION
│       ├─ Log: "Item 'item_name' updated remotely"
│       ├─ Fetch new version from remote
│       ├─ Overwrite local cache
│       ├─ Update metadata (checksums, timestamps)
│       └─ RETURN data
```

**Performance Characteristics:**
- **Cache hit (within TTL):** ~0.01s (disk I/O only)
- **Cache hit (TTL expired, no changes):** ~0.1s (manifest check)
- **Cache miss or invalidation:** ~2-10s (full download)
- **Speedup factor:** 10-1000x depending on file size

---

## Implementation Details

### Phase 1: Core Caching Infrastructure

**Files to Modify:**
1. `src/datafolio/folio.py` - Add cache parameters to constructor
2. `src/datafolio/storage/backend.py` - Add cache layer
3. `src/datafolio/base/handler.py` - Update base handler interface
4. `src/datafolio/handlers/*.py` - Implement caching in each handler

**New Files to Create:**
1. `src/datafolio/cache/__init__.py`
2. `src/datafolio/cache/manager.py` - Cache manager class
3. `src/datafolio/cache/metadata.py` - Metadata operations
4. `src/datafolio/cache/validation.py` - Checksum & TTL validation
5. `src/datafolio/cache/config.py` - Configuration management

### API Design

#### Global Configuration

```python
from datafolio import DataFolio

# Configure caching globally (applies to all instances unless overridden)
DataFolio.configure_cache(
    enabled=True,
    cache_dir='~/.datafolio_cache',  # or '/shared/cache' for institutional
    default_ttl=1800,  # 30 minutes
    max_cache_size='50GB',  # Auto-cleanup when exceeded
    checksum_algorithm='md5',
    strict_mode=False  # If True, fail on checksum mismatch (else warn)
)

# Query global cache
stats = DataFolio.cache_stats()
# Returns: {
#   'total_size': '2.3GB',
#   'total_bundles': 5,
#   'total_items': 47,
#   'cache_hit_rate': 0.87,
#   'oldest_access': '2024-03-10T08:00:00Z'
# }

# Clear all caches
DataFolio.clear_all_caches()  # Use with caution!
```

#### Per-Instance Configuration

```python
# Minimal: Enable caching with defaults
folio = DataFolio('gs://bucket/bundle', cache_enabled=True)

# Full control
folio = DataFolio(
    'gs://bucket/bundle',
    cache_enabled=True,
    cache_dir='~/.datafolio_cache',  # Override global
    cache_ttl=3600,  # 1 hour (override global)
    cache_mode='read-write',  # 'read-only', 'write-through', 'disabled'
    cache_on_add=True,  # Cache items when adding (write-through)
)

# Static snapshot (never check for updates)
folio = DataFolio(
    'gs://public-data/paper-2024',
    cache_enabled=True,
    cache_ttl=None,  # Never expire (check checksum once only)
)
```

#### Cache Management Methods

```python
# Check cache status for specific item
status = folio.cache_status('training_data')
# Returns: {
#   'cached': True,
#   'cache_path': '~/.datafolio_cache/bundles/gs_bucket_bundle/tables/training.parquet',
#   'size_bytes': 1024000,
#   'cached_at': '2024-03-15T10:30:00Z',
#   'last_verified': '2024-03-15T10:30:00Z',
#   'ttl_expires_in': 1200,  # seconds
#   'checksum_valid': True,
#   'access_count': 42
# }

# Force refresh single item (ignore TTL)
folio.refresh_item('training_data', force=True)

# Refresh entire bundle (check all items)
folio.refresh_cache()

# Clear cache for this bundle only
folio.clear_cache()

# Pre-warm cache (download all items)
folio.warm_cache(items=['training_data', 'validation_data'])
```

#### Integration with Existing API

```python
# Existing API works unchanged
folio = DataFolio('gs://bucket/bundle', cache_enabled=True)

# get() transparently uses cache
df = folio.get('training_data')  # Cache hit: ~0.01s
model = folio.get('classifier')  # Cache miss: ~2s, then cached

# add() can optionally write-through to cache
folio.add('new_data', dataframe)  # Uploads to cloud
# With cache_on_add=True, also writes to local cache immediately
```

---

### Checksum Integration

**Where Checksums Come From:**

1. **On `add()`:** Compute MD5 while uploading
   ```python
   # In handler.add()
   checksum = compute_md5(data_bytes)
   metadata = {
       'filename': 'data.parquet',
       'checksum': checksum,
       'checksum_algorithm': 'md5',
       ...
   }
   ```

2. **On `get()` (cache miss):** Compute MD5 while downloading
   ```python
   # In cache_manager.py
   content = remote_storage.download(path)
   checksum = compute_md5(content)
   if remote_checksum and checksum != remote_checksum:
       raise ChecksumMismatchError(...)
   write_to_cache(content, checksum)
   ```

3. **On `get()` (cache validation):** Compare stored checksums
   ```python
   # In cache_manager.py
   remote_meta = fetch_item_metadata(item_name)
   local_meta = load_cache_metadata(item_name)
   if remote_meta['checksum'] != local_meta['checksum']:
       invalidate_cache(item_name)
   ```

**Manifest Updates:**

Extend `items.json` to include checksums:

```json
{
  "training_data": {
    "item_type": "included_table",
    "filename": "training.parquet",
    "size_bytes": 1024000,
    "checksum": "5d41402abc4b2a76b9719d911017c592",
    "checksum_algorithm": "md5",
    "created_at": "2024-03-15T10:00:00Z",
    "updated_at": "2024-03-15T10:30:00Z"
  }
}
```

**Backward Compatibility:**
- If `checksum` field missing (old bundles), compute on first access
- TTL-only validation until checksum available
- Gradual migration: next `add()` operation adds checksums

---

### Concurrency & Locking

**Problem:** Multiple processes/notebooks accessing same cache

**Solution:** File-based locking with atomic operations

```python
# In cache_manager.py
from filelock import FileLock

class CacheManager:
    def get_with_cache(self, item_name):
        lock_path = self._get_lock_path(item_name)
        with FileLock(lock_path, timeout=30):
            # Check cache validity
            if self._is_cache_valid(item_name):
                return self._load_from_cache(item_name)

            # Download and update cache
            data = self._fetch_from_remote(item_name)
            self._write_to_cache(item_name, data)
            return data
```

**Lock Granularity:**
- Per-item locks (not per-bundle) for parallelism
- Read locks for cache hits (shared)
- Write locks for cache updates (exclusive)

**Lock Timeout:**
- Default: 30 seconds
- Handles crashed processes (stale locks)
- User-configurable for large files

---

### Cache Eviction & Limits

**When Cache Grows Too Large:**

```python
# Global config
DataFolio.configure_cache(
    max_cache_size='50GB',  # Total cache limit
    eviction_policy='lru',  # or 'lfu', 'fifo'
    eviction_threshold=0.9  # Cleanup at 90% full
)
```

**LRU (Least Recently Used) Eviction:**
1. Track `last_accessed` timestamp for each item
2. When cache exceeds threshold, sort by `last_accessed`
3. Remove oldest items until under threshold
4. Log evictions: `"Evicted 3 items (2.1GB) from cache"`

**Priority Overrides:**
```python
# Pin important items (never evict)
folio.pin_cache('training_data')

# Assign priority (higher = keep longer)
folio.set_cache_priority('validation_data', priority=10)
```

**Manual Cleanup:**
```python
# Remove items not accessed in 7 days
DataFolio.cleanup_cache(older_than='7d')

# Remove all caches for bundles no longer on disk/cloud
DataFolio.cleanup_orphaned_caches()
```

---

### CLI Commands for Cache Management

**Problem:** Easy cache maintenance without writing Python code

**Solution:** Built-in CLI commands via `datafolio cache` subcommand

```bash
# Show cache statistics
datafolio cache stats
# Output:
# Cache directory: ~/.datafolio_cache
# Total size: 12.3 GB (24% of 50GB limit)
# Bundles cached: 8
# Total items: 145
# Cache hit rate: 87%
# Oldest access: 2024-03-01 (12 days ago)

# List all cached bundles
datafolio cache list
# Output:
# gs://project/exp-001        2.1 GB    23 items    Last: 2h ago
# gs://project/exp-002        5.4 GB    45 items    Last: 1d ago
# local/my-bundle            156 MB     8 items     Last: 5h ago

# Clear cache for specific bundle
datafolio cache clear gs://project/exp-001
# Output: Removed 2.1 GB (23 items) from cache

# Clear entire cache (with confirmation)
datafolio cache clear --all
# Output: WARNING: This will delete 12.3 GB of cached data
#         Continue? [y/N]: y
#         Cleared all caches (8 bundles, 145 items)

# Remove old/unused caches
datafolio cache cleanup --older-than 30d
# Output: Removed 3 bundles (4.2 GB) not accessed in 30 days

# Remove orphaned caches (bundles deleted)
datafolio cache cleanup --orphaned
# Output: Removed 2 orphaned bundles (1.8 GB)

# Set cache size limit
datafolio cache config --max-size 100GB
# Output: Updated cache limit: 100 GB

# Disable caching globally
datafolio cache config --disable
# Output: Caching disabled globally (existing cache preserved)
```

**Implementation:**
```python
# In src/datafolio/cli/cache.py
import click
from datafolio import DataFolio

@click.group()
def cache():
    """Manage DataFolio cache"""
    pass

@cache.command()
def stats():
    """Show cache statistics"""
    stats = DataFolio.cache_stats()
    click.echo(f"Cache directory: {stats['cache_dir']}")
    click.echo(f"Total size: {format_size(stats['total_size'])} ({stats['percent_used']}% of {stats['max_size']})")
    # ... more stats

@cache.command()
@click.option('--all', is_flag=True, help='Clear all caches')
@click.option('--bundle', help='Clear specific bundle')
@click.confirmation_option(prompt='This will delete cached data. Continue?')
def clear(all, bundle):
    """Clear cache"""
    if all:
        DataFolio.clear_all_caches()
        click.echo("Cleared all caches")
    elif bundle:
        # Clear specific bundle
        pass
```

**User Benefits:**
- No Python needed for common tasks
- Quick cleanup before deploys/releases
- Easy monitoring of cache usage
- Integration with shell scripts/CI

---

### Large File Policies & Best Practices

**Problem:** Large files (>1GB) can bloat cache and slow operations

**Policy Recommendations:**

#### 1. Size-Based Storage Strategy

```python
# Automatic policy based on file size
DataFolio.configure_large_file_policy(
    size_threshold='1GB',  # Files larger than this
    default_action='reference',  # Store as reference by default
    warn_on_include=True,  # Warn if user tries to include
    cache_large_files=False  # Don't cache files over threshold
)
```

**Behavior:**
```python
# Adding a 2GB file
folio = DataFolio('gs://bucket/bundle')

# Option 1: Default policy (reference)
folio.add('large_data', df)  # 2GB dataframe
# ⚠️  WARNING: File size 2GB exceeds threshold (1GB)
# ⚠️  Storing as REFERENCE (not included in bundle)
# ⚠️  To include, use: folio.add('large_data', df, force_include=True)

# Option 2: Force include (override policy)
folio.add('large_data', df, force_include=True)
# ⚠️  WARNING: Including 2GB file in bundle (not recommended)
# ⚠️  Consider using reference instead for better performance

# Option 3: Explicit reference
folio.add('large_data', df, as_reference=True)
# ✓ Stored as reference: gs://bucket/bundle/tables/large_data.parquet
```

#### 2. Cache Behavior for Large Files

```python
# Configure cache to skip large files
DataFolio.configure_cache(
    max_cache_size='50GB',
    cache_large_files=False,  # Skip files over threshold
    large_file_threshold='1GB',
    stream_large_files=True  # Stream instead of loading into memory
)
```

**Behavior:**
```python
folio = DataFolio('gs://bucket/bundle', cache_enabled=True)

# Small file (<1GB): Cached normally
small_df = folio.get('training_data')  # 500MB → cached

# Large file (>1GB): Streamed, not cached
large_df = folio.get('huge_dataset')  # 5GB → streamed directly, no cache
# ℹ️  Streaming large file (5GB), not cached
# ℹ️  To cache, use: folio.get('huge_dataset', force_cache=True)

# Force cache for large file (if really needed)
large_df = folio.get('huge_dataset', force_cache=True)
# ⚠️  Caching 5GB file (may impact performance)
```

#### 3. Reference Management

```python
# List all references
refs = folio.list_references()
# Returns: {
#   'huge_dataset': {
#     'url': 'gs://bucket/bundle/tables/huge_dataset.parquet',
#     'size': '5GB',
#     'type': 'parquet'
#   }
# }

# Check if item is reference or included
folio.is_reference('huge_dataset')  # True
folio.is_included('training_data')  # True

# Convert included → reference (to save space)
folio.convert_to_reference('medium_data')
# ✓ Converted 'medium_data' to reference
# ✓ Bundle size reduced by 800MB

# Convert reference → included (for portability)
folio.convert_to_included('huge_dataset')
# ⚠️  This will increase bundle size by 5GB
# Continue? [y/N]: y
```

#### 4. Bundle Size Warnings

```python
# Automatic warnings on bundle creation
folio = DataFolio('experiments/exp-001')
folio.add('data1', df1)  # 500MB
folio.add('data2', df2)  # 800MB
folio.add('data3', df3)  # 1.2GB

# ⚠️  WARNING: Bundle size now 2.5GB (3 items)
# ⚠️  Consider using references for large files to improve performance
# ⚠️  Run: folio.optimize_storage() to convert large items to references

# Optimize storage automatically
folio.optimize_storage(threshold='1GB')
# ✓ Converted 'data3' to reference (1.2GB)
# ✓ Bundle size reduced: 2.5GB → 1.3GB
```

#### 5. Best Practices Summary

**Use INCLUDED for:**
- Small files (<100MB)
- Frequently accessed data
- Data that needs to travel with bundle
- Immutable reference data

**Use REFERENCE for:**
- Large files (>1GB) ✅
- Infrequently accessed data
- Data that changes frequently
- Shared datasets across experiments

**Cache Strategy:**
```python
# Small included files: Cache with short TTL
folio = DataFolio('gs://bucket/bundle',
    cache_enabled=True,
    cache_ttl=1800,  # 30 min
    cache_large_files=False)

# Large referenced files: Don't cache, stream directly
# (Automatic if file > threshold)
```

#### 6. Migration Tools

```bash
# Analyze bundle for optimization opportunities
datafolio analyze experiments/exp-001
# Output:
# Bundle: experiments/exp-001
# Total size: 8.2 GB
# Included: 8.1 GB (12 items)
# References: 100 MB (3 items)
#
# Optimization suggestions:
#   - Convert 'huge_dataset' to reference (5GB)
#   - Convert 'large_model' to reference (2GB)
#   - Potential savings: 7GB (85% reduction)

# Auto-optimize bundle
datafolio optimize experiments/exp-001 --threshold 1GB
# ✓ Converted 2 items to references
# ✓ Bundle size: 8.2GB → 1.2GB (85% reduction)
```

---

## Advanced Features (Future Phases)

### Content-Addressable Storage (CAS)

**Problem:** Same data in multiple bundles wastes space

**Solution:** Store files by checksum, reference from bundles

```
~/.datafolio_cache/
├── blobs/
│   ├── 5d41402abc4b2a76b9719d911017c592  # training.parquet (1GB)
│   └── 098f6bcd4621d373cade4e832627b4f6  # classifier.pkl (100MB)
└── bundles/
    ├── bundle_001/
    │   └── .cache_meta.json  # references blobs by checksum
    └── bundle_002/
        └── .cache_meta.json  # same training.parquet, shares blob
```

**Benefits:**
- Deduplication across bundles
- Immutable storage (safe for concurrent access)
- Easier to implement cache warming from peers

---

### Offline Mode

```python
# Work entirely from cache (fail if not cached)
folio = DataFolio('gs://bucket/bundle', cache_mode='offline')

try:
    data = folio.get('training_data')  # Uses cache or raises
except CacheError:
    print("Item not in cache, download when online")
```

---

### Cache Warming & Sharing

```python
# Pre-download all items (background)
folio.warm_cache(background=True)

# Export cache for sharing (e.g., USB drive)
folio.export_cache('/mnt/usb/bundle_cache.tar.gz')

# Import cache from peer
folio.import_cache('/mnt/usb/bundle_cache.tar.gz', verify_checksums=True)

# Institutional cache (read-only shared cache)
DataFolio.configure_cache(
    cache_dir='~/.datafolio_cache',  # Local R/W cache
    shared_cache_dir='/shared/lab/datafolio_cache',  # Shared R/O cache
    shared_cache_priority='fallback'  # Check local first
)
```

---

### Partial Caching (Lazy Loading)

**For Large Files:** Cache only accessed chunks

```python
# Parquet with 1000 columns, only need 5
df = folio.get('huge_table', columns=['col_a', 'col_b', 'col_c'])
# Only downloads & caches selected columns

# Cache table schema separately (fast)
schema = folio.get_schema('huge_table')  # ~1KB, cached
df = folio.get('huge_table', columns=schema.names[:10])  # Partial fetch
```

---

## Configuration File Format

**File:** `~/.datafolio_cache/config.json`

```json
{
  "version": "1.0",
  "enabled": true,
  "cache_dir": "~/.datafolio_cache",
  "defaults": {
    "ttl_seconds": 1800,
    "checksum_algorithm": "md5",
    "strict_mode": false,
    "cache_mode": "read-write"
  },
  "limits": {
    "max_cache_size_bytes": 53687091200,
    "eviction_policy": "lru",
    "eviction_threshold": 0.9
  },
  "performance": {
    "parallel_downloads": 4,
    "chunk_size_bytes": 8388608,
    "compression": "auto"
  },
  "logging": {
    "level": "INFO",
    "log_cache_hits": false,
    "log_cache_misses": true,
    "log_evictions": true
  },
  "advanced": {
    "content_addressable_storage": false,
    "shared_cache_dir": null,
    "offline_mode": false
  }
}
```

**Override Priority:**
1. Function call parameters (highest)
2. Constructor parameters
3. Environment variables (`DATAFOLIO_CACHE_DIR`, etc.)
4. Config file
5. Built-in defaults (lowest)

---

## Error Handling & Edge Cases

### Network Failures During Validation

```python
# Scenario: TTL expired, but network down
try:
    data = folio.get('training_data')
except NetworkError:
    # Option 1: Fallback to stale cache
    data = folio.get('training_data', use_stale=True)
    warnings.warn("Using stale cache due to network error")

    # Option 2: Fail fast (default)
    raise CacheError("Cannot validate cache and network unavailable")
```

**Configuration:**
```python
DataFolio.configure_cache(
    fallback_to_stale=True,  # Use stale cache on network error
    stale_warning=True,  # Warn user
    max_stale_age=86400  # Max 24 hours stale
)
```

---

### Checksum Mismatches

```python
# Scenario: Local file corrupted or tampered with
try:
    data = folio.get('training_data')
except ChecksumMismatchError as e:
    # Automatic behavior:
    # 1. Log error
    # 2. Invalidate cache
    # 3. Re-download from remote
    # 4. Verify new checksum
    print(f"Cache corruption detected, re-downloaded: {e}")
```

**Strict Mode:**
```python
DataFolio.configure_cache(strict_mode=True)
# Raises exception instead of auto-fixing (for security-critical apps)
```

---

### Partial Downloads / Interrupted Transfers

```python
# Use atomic writes with temp files
def write_to_cache(item_name, data):
    temp_path = cache_path + '.tmp'
    checksum = compute_md5_while_writing(data, temp_path)

    # Only rename if complete
    if checksum == expected_checksum:
        os.rename(temp_path, cache_path)  # Atomic on POSIX
    else:
        os.remove(temp_path)
        raise ChecksumMismatchError(...)
```

---

### Cache Metadata Corruption

```python
# Robust metadata handling
def load_cache_metadata(bundle_path):
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # Metadata corrupted, rebuild from filesystem
        warnings.warn("Cache metadata corrupted, rebuilding...")
        return rebuild_metadata_from_files(bundle_path)
```

---

### Concurrent Modifications

**Problem:** Alice and Bob both update same bundle simultaneously

```python
# Bob's session
folio = DataFolio('gs://shared/bundle', cache_enabled=True)
data = folio.get('training_data')  # Cached version

# [Meanwhile, Alice updates the remote]

# Bob's next access (after TTL)
data = folio.get('training_data')  # Detects change, re-downloads
warnings.warn("Item 'training_data' was updated remotely")
```

**Notification:**
```python
# Optional: Register callback for cache invalidations
def on_cache_invalidated(item_name, old_version, new_version):
    print(f"⚠️  {item_name} changed remotely!")
    print(f"   Old: {old_version['updated_at']}")
    print(f"   New: {new_version['updated_at']}")

folio.on_cache_invalidation(on_cache_invalidated)
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_cache_manager.py
def test_cache_hit_within_ttl():
    """Test that cache is used when TTL not expired"""
    cache = CacheManager(ttl=3600)
    cache.write('item', data, checksum='abc123')

    # Should load from cache without remote check
    result = cache.get('item')
    assert result == data
    assert cache.stats['cache_hits'] == 1
    assert cache.stats['remote_checks'] == 0

def test_cache_miss_after_ttl():
    """Test that remote is checked after TTL expires"""
    cache = CacheManager(ttl=1)  # 1 second TTL
    cache.write('item', data, checksum='abc123')

    time.sleep(2)

    # Should check remote
    result = cache.get('item')
    assert cache.stats['remote_checks'] == 1

def test_checksum_mismatch_invalidates():
    """Test that checksum mismatch triggers re-download"""
    cache = CacheManager()
    cache.write('item', data, checksum='abc123')

    # Simulate remote checksum change
    remote_meta = {'checksum': 'xyz789'}
    with pytest.raises(ChecksumMismatchError):
        cache.validate('item', remote_meta, strict=True)
```

---

### Integration Tests

```python
# tests/test_cache_integration.py
def test_full_workflow_with_updates(tmp_path, mock_cloud):
    """Test complete workflow: cache, update, invalidate"""

    # Initial download
    folio = DataFolio(mock_cloud, cache_dir=tmp_path, cache_ttl=1)
    data1 = folio.get('item')
    assert folio.cache_status('item')['cached'] == True

    # Cache hit
    data2 = folio.get('item')
    assert data2 == data1
    assert folio.cache_stats()['cache_hit_rate'] == 0.5  # 1 hit, 1 miss

    # Simulate remote update
    mock_cloud.update('item', new_data)
    time.sleep(2)  # Wait for TTL

    # Should detect change and re-download
    data3 = folio.get('item')
    assert data3 == new_data
    assert data3 != data1

def test_concurrent_access(tmp_path, mock_cloud):
    """Test multiple processes accessing same cache"""
    import multiprocessing as mp

    def worker(i):
        folio = DataFolio(mock_cloud, cache_dir=tmp_path)
        data = folio.get('large_item')
        return len(data)

    # Run 10 workers in parallel
    with mp.Pool(10) as pool:
        results = pool.map(worker, range(10))

    # All should get same data
    assert len(set(results)) == 1

    # Only one download should have occurred
    assert mock_cloud.download_count == 1
```

---

### Performance Tests

```python
# tests/test_cache_performance.py
def test_cache_speedup(tmp_path, mock_cloud):
    """Verify cache provides significant speedup"""

    # Without cache
    folio_no_cache = DataFolio(mock_cloud, cache_enabled=False)
    t1 = timeit.timeit(lambda: folio_no_cache.get('large_item'), number=10)

    # With cache (first miss, then hits)
    folio_cached = DataFolio(mock_cloud, cache_dir=tmp_path, cache_enabled=True)
    folio_cached.get('large_item')  # Prime cache
    t2 = timeit.timeit(lambda: folio_cached.get('large_item'), number=10)

    speedup = t1 / t2
    assert speedup > 10, f"Expected >10x speedup, got {speedup:.1f}x"

def test_large_cache_performance(tmp_path):
    """Test cache performance with many items"""
    cache = CacheManager(cache_dir=tmp_path)

    # Add 10,000 items
    for i in range(10000):
        cache.write(f'item_{i}', f'data_{i}', checksum=f'sum_{i}')

    # Lookup should still be fast
    t = timeit.timeit(lambda: cache.get('item_9999'), number=1000)
    assert t < 0.1, f"Lookup too slow: {t:.3f}s"
```

---

## Migration Path

### Phase 1: Core Infrastructure (v0.5.0)
**Scope:** Basic caching with TTL validation
- [ ] Cache manager implementation
- [ ] TTL-based validation
- [ ] Per-bundle cache directories
- [ ] Basic API (cache_enabled parameter)
- [ ] Unit tests

**Timeframe:** 2-3 weeks
**Risk:** Low (additive, disabled by default)

---

### Phase 2: Checksum Validation (v0.6.0)
**Scope:** Add MD5 checksums to manifests
- [ ] Compute checksums on add()
- [ ] Store checksums in items.json
- [ ] Checksum validation on get()
- [ ] Migration for existing bundles
- [ ] Integration tests

**Timeframe:** 1-2 weeks
**Risk:** Medium (manifest format change)

---

### Phase 3: Advanced Features (v0.7.0+)
**Scope:** Cache eviction, warming, sharing
- [ ] LRU eviction
- [ ] Cache size limits
- [ ] Cache warming API
- [ ] Export/import cache
- [ ] Performance optimization
- [ ] Content-addressable storage (optional)

**Timeframe:** 3-4 weeks
**Risk:** Low (opt-in features)

---

## Open Questions

1. **Should we support cache encryption for sensitive data?**
   - Use case: PHI, credentials, proprietary data
   - Implementation: Encrypt cache files with user key
   - Trade-off: Performance overhead vs. security

2. **How to handle very large files (>10GB)?**
   - Option A: Don't cache (skip if > threshold)
   - Option B: Partial caching (byte ranges)
   - Option C: Streaming with cache-aside pattern

3. **Should cache be per-user or per-system?**
   - Per-user: `~/.datafolio_cache` (privacy, isolation)
   - Per-system: `/var/cache/datafolio` (shared, space-efficient)
   - Hybrid: Per-user + opt-in shared cache

4. **How to handle cache for referenced_table items?**
   - These don't have data to cache (just references)
   - Could cache resolved data if fetched via URL
   - Skip caching for pure references?

5. **Should we log cache activity by default?**
   - Pros: Visibility, debugging, usage analytics
   - Cons: Disk space, privacy concerns
   - Compromise: Opt-in logging, or in-memory stats only

---

## Success Metrics

### Performance
- **Cache hit rate:** >80% for typical workflows
- **Speedup:** >10x for cached reads vs. remote
- **Overhead:** <5% slowdown for uncached reads (validation cost)

### User Experience
- **Transparent:** No code changes required to enable
- **Predictable:** Clear TTL behavior, no surprises
- **Debuggable:** Easy to check cache status, clear errors

### Reliability
- **Data integrity:** 100% checksum validation (no corrupt data)
- **Concurrency:** Safe for parallel notebook sessions
- **Robustness:** Graceful handling of network failures, disk full, etc.

---

## Appendix A: API Reference Summary

```python
# Global configuration
DataFolio.configure_cache(
    enabled: bool = True,
    cache_dir: str = '~/.datafolio_cache',
    default_ttl: int = 1800,
    max_cache_size: str = '50GB',
    checksum_algorithm: str = 'md5',
    strict_mode: bool = False
)

# Constructor
DataFolio(
    path: str,
    cache_enabled: bool = False,
    cache_dir: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    cache_mode: str = 'read-write',
    cache_on_add: bool = False
)

# Instance methods
folio.cache_status(item_name: str) -> dict
folio.refresh_item(item_name: str, force: bool = False)
folio.refresh_cache()
folio.clear_cache()
folio.warm_cache(items: Optional[list] = None, background: bool = False)
folio.cache_stats() -> dict

# Global methods
DataFolio.cache_stats() -> dict
DataFolio.clear_all_caches()
DataFolio.cleanup_cache(older_than: str = '30d')
DataFolio.cleanup_orphaned_caches()
```

---

## Appendix B: Comparison with Other Systems

| Feature | DataFolio Cache | Git LFS | DVC | Hugging Face Hub |
|---------|----------------|---------|-----|------------------|
| Transparent API | ✅ | ❌ (separate commands) | ❌ (separate commands) | ✅ |
| TTL validation | ✅ | ❌ | ❌ | ❌ |
| Checksum validation | ✅ | ✅ | ✅ | ✅ |
| Automatic eviction | ✅ | ❌ | ❌ | ❌ |
| Cloud-native | ✅ | ❌ (requires server) | ✅ | ✅ |
| Offline mode | ✅ (planned) | ✅ | ✅ | ✅ |
| Content deduplication | 🔄 (planned) | ✅ | ✅ | ✅ |

**Key Differentiators:**
- DataFolio: Transparent, automatic, TTL-based (optimized for active research)
- Git LFS / DVC: Explicit versioning (optimized for reproducibility)
- HF Hub: Model-centric (optimized for ML models)

---

## Appendix C: Checksum Algorithm Comparison

| Algorithm | Speed | Collision Resistance | File Size | Use Case |
|-----------|-------|---------------------|-----------|----------|
| MD5 | Fast (500 MB/s) | Low (known collisions) | 16 bytes | Cache validation ✅ |
| SHA-1 | Medium (350 MB/s) | Medium (theoretical collisions) | 20 bytes | Legacy systems |
| SHA-256 | Slower (150 MB/s) | High (no known collisions) | 32 bytes | Security-critical |
| xxHash | Fastest (1500 MB/s) | N/A (non-cryptographic) | 8 bytes | High-performance cache |

**Recommendation:** MD5 for cache validation
- Fast enough for large files
- Collision risk negligible for cache use case
- Widely supported, small footprint
- Can upgrade to SHA-256 later if needed (versioned metadata)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-03 | Claude | Initial design document |

---

**Next Steps:**
1. Review design with maintainers
2. Prototype Phase 1 implementation
3. Performance benchmarking with real datasets
4. Iterate based on feedback
