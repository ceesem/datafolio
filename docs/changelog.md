# Changelog

## 0.2.0 (Unreleased)

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

### Documentation
- Comprehensive documentation update with examples for all new features
- Added Quick Start guide with generic interface examples
- Added PyTorch deep learning workflow example
- Added complete ML workflow example using the new generic interface
- Updated directory structure documentation

## 0.1.0

Initial release