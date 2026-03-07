# Changelog

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