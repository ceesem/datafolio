# Changelog

## 2.0.0

### Unified manifest (format v2)

- `items.json` is now the SINGLE authoritative manifest:
  `{schema_version: 2, revision, metadata, items, snapshots}`. The
  `metadata.json`/`snapshots.json` sidecars are gone (their content is
  embedded), so every commit — items, metadata, snapshots — is one atomic
  file replace with one revision. v0/v1 folios load transparently and
  migrate on the first write (sidecars are removed then); 1.x writers
  refuse v2 folios cleanly via the schema_version gate.
- Snapshot membership is derived from the snapshot registry's pinned
  version ids; the denormalized `in_snapshots` markers are no longer
  persisted (legacy markers are ignored and stripped).

### Unified item API (breaking)

- **One write, one read**: `add(name, obj)` and `get(name)` replace the
  per-type method families (`add_table`/`add_numpy`/`add_json`/
  `add_timestamp`/`add_data`, `get_table`/`get_numpy`/`get_json`/
  `get_timestamp`/`get_data`, every `get_*_path` and `get_*_info`).
  `item_path(name)` and `item_info(name)` cover paths and manifest entries
  for all types. See the [migration guide](guides/migrating-to-2.0.md).
- **Explicit verbs** kept where being explicit matters: `add_model`/
  `get_model` (any picklable object; skops models need `trusted=True`),
  `add_file` (files never enter via string sniffing), and
  `reference_table`/`inspect_table`/`scan_table`.
- **Membership**: `name in folio` mirrors `get()` — True exactly when
  `folio.get(name)` would succeed, so archived items count as present and
  non-string keys are simply not members. It auto-refreshes like every other
  read entry point.

  ```python
  if 'results' not in folio:
      folio.add('results', df)
  ```

- Existing 1.x folios open in 2.0 without a manual conversion step and migrate
  to the unified manifest on their first write.

### Correctness

- Snapshots now honor "owned data frozen" everywhere: `delete()` preserves
  snapshotted versions, `restore_snapshot` restores deleted items and never
  copies bytes, `export_snapshot` works with external references and
  preserves lineage/descriptions, `create_snapshot` inside `batch()` raises,
  and `update_item` no longer edits snapshot views.
- Read-only mode is enforced on every mutation path.
- Item names are validated (traversal-safe segment grammar; `/` namespacing
  kept).
- Cloud fixes: `exists()` no longer swallows errors (a transient auth
  failure can no longer cause a fresh bundle to be written over a real
  one); included-table reads go through cloudfiles instead of a second
  credential chain; missing cloud objects raise `FileNotFoundError`.
- `diff_from_snapshot()` and `datafolio snapshot status` compare against
  the newest snapshot (previously the oldest).
- Git context for snapshots is captured from the working directory (the
  running code), not the bundle directory.

### Added

Cloud reads are round-trip bound, and datafolio was paying for far more
round trips than the data required. Measured on 60 items at a 250ms round
trip, reading them all went from **45.9s to 1.1s** across these four
changes.

- **The staleness check is one round trip, not two.** Every read re-verifies
  the on-disk manifest revision; it used to probe with `exists()` before
  reading. A missing manifest already reads as "not stale" via the read's
  own error path, so the probe was pure latency — and on cloud it could cost
  two operations by itself (object check, then a prefix listing).

- `pinned()` — a re-entrant context manager that suspends the per-read
  staleness recheck entirely. Inside the block the check happens once, on
  entry: 60 `get()` calls drop from 60 manifest round trips to 1. The trade
  is explicit — other writers' changes are not seen until the block exits,
  so reads are internally consistent rather than up to date. Writes inside
  the block behave normally, including the fail-closed stale-writer check
  that rejects a pinned write over an externally advanced manifest.

  ```python
  with folio.pinned():
      for name in folio.tables:
          process(folio.get(name))
  ```

- `get_many(names, threads=20)` — read several items concurrently, returning
  `{name: content}`. Pure sugar: it is exactly a `pinned()` block around a
  thread pool mapping `get()`, and writing that loop yourself is equally
  supported and equally fast. Overlapping round trips is what pays — 60
  reads go from 15.6s pinned-sequential to 1.1s at 20 threads.

  ```python
  tables = folio.get_many(['train', 'test', 'holdout'])
  ```

  The pin is a correctness requirement there, not only a speed one: a
  concurrent auto-refresh rebuilds the item table in place, so a thread
  reading it mid-rebuild can miss an item that is really there.

- `validate()` resolves existence for every item in one batched call
  (`StorageBackend.exists_many`), so validating a cloud bundle costs a few
  threaded round trips rather than one per item. Semantics are unchanged,
  including the directory fallback for sharded payloads and the rule that a
  per-item failure is reported as `False` rather than raising.

### Changed

- `add(..., preserve_index=True)` (still supported) now stores the index as
  plain columns recorded in the manifest (`index_columns`) and restores them
  on pandas reads — the parquet file stays readable as ordinary columns by
  any tool.

### Removed

- The caching subsystem (`cache_enabled`/`cache_dir`/`cache_ttl` and the
  four `cache_*` methods).
- Shadow-API trims (datafolio stays lightweight and offloads to other
  tools): `code=` kwargs (git owns code history),
  `add_model(hyperparameters=)` (`estimator.get_params()` owns it),
  `models=` table lineage (use `inputs`; legacy fields still read),
  `reproduce_instructions()`/`snapshot reproduce` (data lives in
  `get_snapshot_info()`), requirements.txt embedding in environment
  capture (uv.lock hash kept), the `.polars` accessor property, and
  reader kwargs on `get()` (use `scan_table()` or
  `pd.read_parquet(folio.item_path(name), ...)`).

## 1.3.0

### Polars DataFrame Support

- `add_table()` and `get_table()` now accept **Polars DataFrames** in addition to pandas DataFrames.
- Serialization now routes through **PyArrow** for both libraries, preserving exact column types such as nullable `Int64` and struct fields that were previously degraded via `pandas.to_parquet`.
- `PandasHandler` renamed to `DataframeHandler`; `PandasHandler` remains as a backward-compatible alias.

## 1.2.0

### Generic Path Methods

- **`get_data_path(name)`** refactored into a universal dispatcher: works for all item types (tables, models, arrays, JSON, timestamps, artifacts) rather than only referenced tables.
- **New type-specific path methods**: `get_table_path()`, `get_model_path()`, `get_numpy_path()`, `get_json_path()`, `get_timestamp_path()` — each returns the full path to the stored file for that item type.
- **`get_table_path()`** now works for both included and referenced tables. Previously `get_data_path()` raised an error for included tables; now it returns the parquet file path inside the bundle.

## 1.1.0

### Item Curation

#### Archive / Unarchive
- **New `archive()` method**: Mark one or more items as hidden without deleting them.
  - Accepts a single name, a list of names, or a glob pattern (e.g. `folio.archive("intermediate/*")`)
  - Archived items are excluded from `list_contents()`, `describe()`, and `copy()` by default
  - Data remains fully accessible via `get_data()` / `get_table()` / etc.
  - `create_snapshot()` still captures archived items (snapshots record complete state)
- **New `unarchive()` method**: Restore archived items to active status.
  - Same flexible name/list/glob interface as `archive()`
- **`include_archived=True` parameter** added to `list_contents()`, `describe()`, and `copy()`
  — pass this flag to reveal or include archived items in any of those views.

#### Lineage-Aware Copy
- **New `follow_lineage=True` parameter on `copy()`**: When combined with `include_items`,
  automatically resolves all transitive upstream dependencies of the named items.
  - `folio.copy("pub", include_items=["final_model"], follow_lineage=True)` copies `final_model`
    plus every item it depends on, recursively.
  - Items referenced in lineage metadata that are not in this folio (external tables, etc.) are
    silently skipped — the lineage metadata is still preserved.
  - Works together with `include_archived=True` to control whether archived upstream items
    are included or excluded.

### Major Features

#### Generic Data Interface
- **New `add_data()` method**: Universal data addition method that automatically detects data type and routes to the appropriate handler
  - Supports DataFrames, numpy arrays, dicts, lists, scalars, and external references
  - Single, intuitive interface for all data types
- **New `get_data()` method**: Universal data retrieval method that automatically returns data in its original format
  - No need to remember which getter to use for each data type

#### Numpy Array Support
- **New `add_numpy()` method**: Store numpy arrays as `.npy` files with full metadata
  - Preserves shape, dtype, and array properties
  - Supports lineage tracking (inputs, code context)
- **New `get_numpy()` method**: Retrieve numpy arrays with original shape and dtype

#### JSON Data Support
- **New `add_json()` method**: Store JSON-serializable data (dicts, lists, scalars)
  - Supports nested structures
  - Type information stored in metadata
  - Supports lineage tracking
- **New `get_json()` method**: Retrieve JSON data in original format

#### Timestamp Support
- **New `add_timestamp()` method**: Store datetime objects with proper timezone handling
  - Accepts timezone-aware `datetime.datetime` objects or Unix timestamps (int/float)
  - Rejects naive datetimes to prevent timezone ambiguity
  - Automatically converts all timestamps to UTC for consistent storage
  - Stores as ISO 8601 strings in JSON format for human readability
  - Supports lineage tracking (inputs, code context)
- **New `get_timestamp()` method**: Retrieve timestamps in multiple formats
  - Returns UTC-aware datetime by default
  - Optional `as_unix=True` parameter to return Unix timestamp (float)
  - Always reads fresh from disk (not cached)
- **Integration features**:
  - Timestamps appear in `list_contents()` under `"timestamps"` key
  - Timestamps display in `describe()` output with human-readable formatting
  - Full support for `folio.data.timestamp_name` accessor pattern
  - Round-trip preservation of microsecond precision

### Enhanced Features

#### Improved `describe()` Method
- **Compact output format**: More readable, information-dense display
- **New parameters**:
  - `return_string=True`: Returns description as string instead of printing
  - `show_empty=True`: Shows empty sections in output
  - `max_metadata_fields=10`: Limit number of metadata fields displayed (default: 10)
  - `show_paths=True`: Show the file path for every item — especially useful for
    cloud-hosted folios where paths can be copied and sent to collaborators directly
- **Unified data sections**: Tables section now combines referenced and included tables
- **Better metadata display**: Shows shape, dtype, init_args, and other relevant info inline
- **Improved lineage display**: Clearer visualization of data dependencies
- **Smart metadata display**: New metadata section with intelligent truncation
  - Automatically filters out internal fields (`_datafolio`, `created_at`, `updated_at`)
  - Truncates long strings with ellipsis (shows first 50 chars)
  - Shows type and item count for collections (lists, dicts)
  - Limits display to configurable number of fields with "... and N more fields" indicator

#### Path Sharing for Collaborators

- **New `get_item_path()` method**: Returns the full path to any item's data file by name,
  regardless of item type (table, model, artifact, array, JSON, or timestamp).
  - For items stored in the bundle returns the full local or cloud URI
    (e.g. `s3://bucket/my-run/tables/results.parquet`)
  - For referenced tables returns the external path recorded at reference time
  - Makes it easy to hand off individual files to colleagues who don't use datafolio

#### New `delete()` Method
- **Delete items from DataFolio**: Remove items and their associated files
- **Flexible input**: Accepts single string or list of strings
- **Transaction-like validation**: Checks all items exist before deleting any
- **Dependency warnings**: Warns (but doesn't block) when deleting items with dependents
- **Parameters**:
  - `name`: Item name(s) to delete (string or list)
  - `warn_dependents=True`: Print warning if deleted items have dependents
- **Method chaining**: Returns `Self` for fluent API
- **Complete cleanup**: Removes both manifest entries and physical files

#### Autocomplete-Friendly Data Access (`folio.data`)
- **New `data` property**: Access items with IDE autocomplete support
- **Dual access patterns**:
  - Attribute-style: `folio.data.my_table.content`
  - Dictionary-style: `folio.data['my_table'].content`
- **ItemProxy properties**: Each item provides rich metadata access
  - `.content`: Returns data in appropriate format (DataFrame, array, dict, model, or file path)
  - `.description`: Item description string
  - `.type`: Item type identifier
  - `.path`: File path (for referenced tables and artifacts)
  - `.inputs`: List of lineage inputs
  - `.dependents`: List of dependent items
  - `.metadata`: Full metadata dictionary
- **IPython/Jupyter support**: Full autocomplete via `__dir__` implementation
- **Type-appropriate returns**: Automatically returns correct data type based on item type

#### Enhanced `list_contents()` Method
- **New keys in return dict**:
  - `numpy_arrays`: List of numpy array items
  - `json_data`: List of JSON data items
  - `timestamps`: List of timestamp items

### Internal Improvements
- **Refactored to handler-based architecture**: Separated data type logic into modular handlers for improved maintainability and extensibility
  - Core `folio.py` reduced from 3,659 → 764 lines (79% smaller)
  - 7 specialized handlers for different data types
  - Zero breaking changes - all existing APIs preserved
- **Improved test coverage**: 69% → 80% coverage with 694 passing tests (up from 265)
- **Enhanced code quality**: Complete type hints, no circular dependencies, clean linting

### Documentation
- Comprehensive documentation update with examples for all new features
- Added Quick Start guide with generic interface examples
- Added complete ML workflow example using the new generic interface
- Updated directory structure documentation

## 0.1.0

Initial release
