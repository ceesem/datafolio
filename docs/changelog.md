# Changelog

## 0.2.0 (Unreleased)

### Major Features

#### Advanced Table Features

**Polars DataFrame Support**
- **First-class Polars integration**: DataFolio now automatically detects and handles Polars DataFrames
  - New `PolarsHandler` for seamless Polars DataFrame storage and retrieval
  - Type preservation: Polars in → Polars out, pandas in → pandas out
  - Shared Parquet storage format ensures pandas/Polars interoperability
  - Auto-detection via handler registry - no manual specification needed
  - Full metadata tracking including dtypes and dataframe library
- **Optional dependency**: Install with `pip install datafolio[polars]` or `pip install datafolio[all]`

**SQL Queries with DuckDB**
- **New `query_table()` method**: Query tables using SQL without loading full data into memory
  - Powered by DuckDB for efficient Parquet file queries
  - Use `$table` placeholder for automatic file path substitution
  - Supports complex SQL: WHERE, JOIN, GROUP BY, ORDER BY, LIMIT, OFFSET
  - Memory-efficient: Only matching rows loaded into memory
  - Return type options: `return_type="pandas"` or `return_type="polars"`
- **Example**: `folio.query_table('data', 'SELECT * FROM $table WHERE value > 1000 LIMIT 100')`
- **Optional dependency**: Install with `pip install datafolio[query]` or `pip install datafolio[all]`

**Chunked Iteration**
- **New `iter_table()` method**: Memory-efficient iteration over large tables
  - Generator-based: Process data in chunks without loading entire table
  - Column selection: `columns=['col1', 'col2']` to load only needed columns
  - WHERE filtering: `where='value > 1000'` to filter rows during iteration
  - Configurable chunk size: `chunk_size=10000` (default)
  - Return type options: `return_type="pandas"` or `return_type="polars"`
  - Uses DuckDB OFFSET/LIMIT for efficient chunking
- **Example**:
  ```python
  for chunk in folio.iter_table('large_data', chunk_size=10000, where='status = "active"'):
      process(chunk)
  ```
- **Requires DuckDB**: Install with `pip install datafolio[query]` or `pip install datafolio[all]`

**Convenient Query Parameters in `get_table()`**
- **Enhanced `get_table()` method**: New convenience parameters for common operations
  - `limit`: Return only first N rows (e.g., `limit=100`)
  - `offset`: Skip first N rows (e.g., `offset=1000`)
  - `where`: Simple filter without SQL boilerplate (e.g., `where='value > 9000'`)
  - Type preservation: Automatically returns pandas or Polars based on storage
  - Can be combined: `get_table('data', where='active', limit=100, offset=50)`
- **Example**: `folio.get_table('data', where="status = 'active'", limit=100)`
- **Backward compatible**: Original behavior preserved when no parameters specified
- **Requires DuckDB**: Install with `pip install datafolio[query]` or `pip install datafolio[all]`
- **Limitation**: Only works with included tables (not referenced tables)

**Table Utility Methods**
- **New `get_table_path()` method**: Get filesystem path to table's Parquet file
  - Enables integration with external tools (Ibis, DuckDB, Polars)
  - Returns absolute path to Parquet file
  - Works with both pandas and Polars tables
- **New `table_info()` method**: Inspect table metadata without loading data
  - Uses PyArrow to read Parquet metadata efficiently
  - Returns row count, column count, file size, schema
  - Helps decide processing strategy (load vs iterate)
- **New `preview_table()` method**: Quick preview of first N rows
  - Convenience wrapper around `get_table(limit=n)`
  - Default: 10 rows
  - Type preservation (pandas/Polars)
- **Example**: `path = folio.get_table_path('data'); df = pl.read_parquet(path)`

**Documentation**
- **New Advanced Tables Guide** (`docs/guides/advanced-tables.md`):
  - Complete documentation for Polars support
  - SQL query examples and patterns
  - Chunked iteration use cases
  - Convenience parameter reference
  - Table utility methods documentation
  - Performance tips and best practices
  - **Ibis integration guide**: Recommended patterns for complex operations
    - Joins across multiple DataFolio tables
    - Window functions and pivots
    - Complete customer analytics example
    - When to use each tool (DataFolio vs Ibis)
- Updated main documentation with installation options and feature highlights

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

#### PyTorch Model Support
- **New `add_pytorch()` method**: Full support for PyTorch models
  - Saves state dict using `torch.save()`
  - Optional class serialization with dill for full reconstruction
  - Stores model metadata (class name, module, init_args)
  - Supports hyperparameters, lineage, and code tracking
- **New `get_pytorch()` method**: Three ways to load PyTorch models
  - State dict only: `get_pytorch(name, reconstruct=False)`
  - With provided class: `get_pytorch(name, model_class=MyModel)`
  - Auto-reconstruction: Uses metadata or serialized class
- **Enhanced `add_model()` method**: Now automatically detects PyTorch vs sklearn models
  - Routes to appropriate handler (`add_pytorch` or `add_sklearn`)
  - Unified interface for all model types
- **Enhanced `get_model()` method**: Automatically detects stored model type and uses correct loader

### Enhanced Features

#### Improved `describe()` Method
- **Compact output format**: More readable, information-dense display
- **New parameters**:
  - `return_string=True`: Returns description as string instead of printing
  - `show_empty=True`: Shows empty sections in output
  - `max_metadata_fields=10`: Limit number of metadata fields displayed (default: 10)
- **Unified data sections**: Tables section now combines referenced and included tables
- **Better metadata display**: Shows shape, dtype, init_args, and other relevant info inline
- **Improved lineage display**: Clearer visualization of data dependencies
- **Smart metadata display**: New metadata section with intelligent truncation
  - Automatically filters out internal fields (`_datafolio`, `created_at`, `updated_at`)
  - Truncates long strings with ellipsis (shows first 50 chars)
  - Shows type and item count for collections (lists, dicts)
  - Limits display to configurable number of fields with "... and N more fields" indicator

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
  - `pytorch_models`: List of PyTorch model items

### Internal Improvements
- **Refactored to handler-based architecture**: Separated data type logic into modular handlers for improved maintainability and extensibility
  - Core `folio.py` reduced from 3,659 → 764 lines (79% smaller)
  - 8 specialized handlers for different data types
  - Zero breaking changes - all existing APIs preserved
- **Improved test coverage**: 69% → 79% coverage with 423 passing tests (up from 265)
- **Enhanced code quality**: Complete type hints, no circular dependencies, clean linting

### Documentation
- Comprehensive documentation update with examples for all new features
- Added Quick Start guide with generic interface examples
- Added PyTorch deep learning workflow example
- Added complete ML workflow example using the new generic interface
- Updated directory structure documentation

## 0.1.0

Initial release