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
- **Unified data sections**: Tables section now combines referenced and included tables
- **Better metadata display**: Shows shape, dtype, init_args, and other relevant info inline
- **Improved lineage display**: Clearer visualization of data dependencies

#### Enhanced `list_contents()` Method
- **New keys in return dict**:
  - `numpy_arrays`: List of numpy array items
  - `json_data`: List of JSON data items
  - `pytorch_models`: List of PyTorch model items

### Documentation
- Comprehensive documentation update with examples for all new features
- Added Quick Start guide with generic interface examples
- Added PyTorch deep learning workflow example
- Added complete ML workflow example using the new generic interface
- Updated directory structure documentation

## 0.1.0

Initial release