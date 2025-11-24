# DataFolio

**A lightweight, filesystem-based data versioning and experiment tracking library for Python.**

DataFolio helps you organize, version, and track your data science experiments by storing datasets, models, and artifacts in a simple, transparent directory structure. Everything is saved as plain files (Parquet, JSON, pickle) that you can inspect, version with git, or backup to any storage system.

## Features

- **Universal Data Management**: Single `add_data()` method automatically handles DataFrames, numpy arrays, dicts, lists, and scalars
- **Model Support**: Save and load scikit-learn and PyTorch models with full metadata tracking
- **Data Lineage**: Track inputs and dependencies between datasets and models
- **External References**: Point to data stored externally (S3, local paths) without copying
- **Multi-Instance Sync**: Automatic refresh when multiple notebooks/processes access the same bundle
- **Autocomplete Access**: IDE-friendly `folio.data.item_name.content` syntax with full autocomplete support
- **Smart Metadata Display**: Automatic metadata truncation and formatting in `describe()`
- **Item Management**: Delete items with dependency tracking and warnings
- **Git-Friendly**: All data stored as standard file formats in a simple directory structure
- **Type-Safe**: Full type hints and comprehensive error handling
- **Snapshots**: Create immutable checkpoints of experiments with copy-on-write versioning
- **CLI Tools**: Command-line interface for snapshot management and bundle operations

## Quick Start

```python
from datafolio import DataFolio
import pandas as pd
import numpy as np

# Create a new folio
folio = DataFolio('experiments/my_experiment')

# Add any type of data with a single method
folio.add_data('results', df)                          # DataFrame
folio.add_data('embeddings', np.array([1, 2, 3]))    # Numpy array
folio.add_data('config', {'lr': 0.01})                # Dict/JSON
folio.add_data('accuracy', 0.95)                      # Scalar

# Retrieve data (automatically returns correct type)
df = folio.get_data('results')           # Returns DataFrame
arr = folio.get_data('embeddings')       # Returns numpy array
config = folio.get_data('config')        # Returns dict

# Or use autocomplete-friendly access
df = folio.data.results.content          # Same as get_data()
arr = folio.data.embeddings.content
config = folio.data.config.content

# View everything (including custom metadata)
folio.describe()

# Clean up temporary items
folio.delete('temp_data')
```

## Installation

```bash
pip install datafolio
```

## Core Concepts

### Generic Data Methods

The `add_data()` and `get_data()` methods provide a unified interface for all data types:

```python
# add_data() automatically detects type and uses the appropriate method
folio.add_data('my_data', data)  # Works with DataFrame, array, dict, list, scalar

# get_data() automatically detects stored type and returns correct format
data = folio.get_data('my_data')  # Returns original type
```

Supported data types:
- **DataFrames** (`pd.DataFrame`) → stored as Parquet
- **Numpy arrays** (`np.ndarray`) → stored as `.npy`
- **JSON data** (`dict`, `list`, `int`, `float`, `str`, `bool`, `None`) → stored as JSON
- **External references** → metadata only, data stays in original location

### Multi-Instance Access

DataFolio automatically keeps multiple instances synchronized when accessing the same bundle:

```python
# Notebook 1: Create and update bundle
folio1 = DataFolio('experiments/shared')
folio1.add_data('results', df)

# Notebook 2: Open same bundle
folio2 = DataFolio('experiments/shared')

# Notebook 1: Add more data
folio1.add_data('analysis', new_df)

# Notebook 2: Automatically sees new data!
folio2.describe()  # Shows both 'results' and 'analysis'
analysis = folio2.get_data('analysis')  # Works immediately ✅
```

All read operations (`describe()`, `list_contents()`, `get_*()` methods, and `folio.data` accessors) automatically refresh from disk when changes are detected, ensuring you always see the latest data without manual intervention.

### Data Types

#### Tables (DataFrames)

Store pandas DataFrames as Parquet files:

```python
# Add table
folio.add_table('training_data', df,
    description='Training dataset with 10k samples',
    inputs=['raw_data'])

# Reference external data (no copy)
folio.reference_table('raw_data',
    reference='s3://bucket/data.parquet',
    description='Original raw data')

# Retrieve
df = folio.get_table('training_data')
```

#### Numpy Arrays

Store numpy arrays with full metadata:

```python
# Add array
array = np.random.randn(100, 128)
folio.add_numpy('embeddings', array,
    description='Model embeddings',
    inputs=['training_data'])

# Retrieve
arr = folio.get_numpy('embeddings')  # Returns numpy array with original shape/dtype
```

Metadata includes:
- Shape and dtype
- Description and inputs (lineage)
- Code context (optional)

#### JSON Data

Store configuration, metrics, or any JSON-serializable data:

```python
# Store different types
folio.add_json('config', {'lr': 0.01, 'batch_size': 32})
folio.add_json('metrics', {'accuracy': 0.95, 'f1': 0.92})
folio.add_json('classes', ['cat', 'dog', 'bird'])
folio.add_json('best_score', 0.95)

# Retrieve
config = folio.get_json('config')  # Returns dict
classes = folio.get_json('classes')  # Returns list
score = folio.get_json('best_score')  # Returns float
```

### Models

#### Scikit-learn Models

```python
from sklearn.ensemble import RandomForestClassifier

# Train and save
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

folio.add_model('classifier', clf,
    description='Random forest classifier',
    hyperparameters={'n_estimators': 100},
    inputs=['training_data'])

# Load
clf = folio.get_model('classifier')
```

#### PyTorch Models

Full support for PyTorch models with optional class serialization:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Save model
model = MyModel(input_dim=10, hidden_dim=50)
folio.add_pytorch('neural_net', model,
    description='Simple feedforward network',
    init_args={'input_dim': 10, 'hidden_dim': 50},
    save_class=True)  # Optional: serialize class definition

# Load model (three options)
# 1. State dict only
state_dict = folio.get_pytorch('neural_net', reconstruct=False)

# 2. Provide model class
model = folio.get_pytorch('neural_net', model_class=MyModel)

# 3. Auto-reconstruct (if class is importable)
model = folio.get_pytorch('neural_net')
```

#### Universal Model Interface

Use `add_model()` and `get_model()` for automatic model type detection:

```python
# Works for both sklearn and PyTorch
folio.add_model('my_model', model)  # Auto-detects type
loaded = folio.get_model('my_model')  # Auto-detects type
```

### Data Lineage

Track dependencies between datasets and models:

```python
# Create dependency chain
folio.reference_table('raw', reference='s3://bucket/raw.parquet')
folio.add_table('clean', cleaned_df, inputs=['raw'])
folio.add_table('features', feature_df, inputs=['clean'])
folio.add_model('model', clf, inputs=['features'])

# Lineage is preserved in metadata and shown in describe()
```

### Describe Your Folio

The `describe()` method provides a compact overview of all data with smart metadata display:

```python
# Print to console
folio.describe()

# Get as string
summary = folio.describe(return_string=True)

# Show empty sections
folio.describe(show_empty=True)

# Limit metadata fields shown (default: 10)
folio.describe(max_metadata_fields=5)
```

Example output:

```
DataFolio: experiments/my_experiment
====================================

Tables (2):
  • raw (reference): Original raw data
    ↳ path: s3://bucket/raw.parquet
  • features: Engineered features
    ↳ inputs: clean
    ↳ shape: [10000, 50]

Numpy Arrays (1):
  • embeddings: Model embeddings
    ↳ shape: [100, 128], dtype: float64
    ↳ inputs: training_data

JSON Data (2):
  • config: Model configuration
    ↳ type: dict
  • accuracy: Final accuracy score
    ↳ type: float

PyTorch Models (1):
  • neural_net: Feedforward network
    ↳ init_args: input_dim=10, hidden_dim=50
    ↳ inputs: features

Metadata (5):
  • experiment_name: my_experiment
  • model_version: v2.1
  • learning_rate: 0.001
  • tags: ['neural_net', 'classification', 'production'] (list, 3 items)
  • description: My experiment description that is quite lo... (truncated)
  ... and 3 more fields
```

The metadata section automatically:
- Filters out internal fields (like `_datafolio`, `created_at`, `updated_at`)
- Truncates long strings with ellipsis
- Shows type and count for collections (lists, dicts)
- Limits display to `max_metadata_fields` (default: 10)

### Delete Items

Remove items from your folio with the `delete()` method:

```python
# Delete single item
folio.delete('old_model')

# Delete multiple items
folio.delete(['temp_data', 'debug_plot', 'old_model'])

# Delete without dependency warnings
folio.delete('item', warn_dependents=False)
```

The `delete()` method:
- Removes items from the manifest and deletes associated files
- Validates all items exist before deleting any (transaction-like)
- Warns if deleted items have dependents (but allows deletion)
- Supports method chaining
- Works with both single items (string) and multiple items (list)

```python
# Example: Clean up temporary items
folio = DataFolio('experiments/test')
folio.add_data('temp1', [1, 2, 3])
folio.add_data('temp2', [4, 5, 6])
folio.add_data('final', [7, 8, 9], inputs=['temp1', 'temp2'])

# Delete temporary data (warns about 'final' dependency)
folio.delete(['temp1', 'temp2'])
# Warning: Deleting 'temp1' which is used by: final. Those items may have broken lineage.
# Warning: Deleting 'temp2' which is used by: final. Those items may have broken lineage.

# Delete without warnings
folio.delete(['temp1', 'temp2'], warn_dependents=False)
```

### Autocomplete-Friendly Data Access

Access your data with autocomplete support using the `folio.data` property:

```python
# Attribute-style access (autocomplete-friendly!)
df = folio.data.results.content          # Get DataFrame
desc = folio.data.results.description    # Get description
type_str = folio.data.results.type       # Get item type
inputs = folio.data.results.inputs       # Get lineage inputs
deps = folio.data.results.dependents     # Get dependents
meta = folio.data.results.metadata       # Get full metadata dict

# Dictionary-style access
df = folio.data['results'].content
model = folio.data['classifier'].content

# Works for all data types
arr = folio.data.embeddings.content      # numpy array
cfg = folio.data.config.content          # dict
model = folio.data.classifier.content    # model object

# Artifacts return file path
with open(folio.data.plot.content, 'rb') as f:
    img = f.read()

# Path property for referenced tables and artifacts
external_path = folio.data.raw_data.path  # e.g., 's3://bucket/raw.parquet'
```

The `ItemProxy` returned by `folio.data.item_name` provides:
- `.content` - Returns the actual data (DataFrame, array, dict, model, or file path)
- `.description` - Description string
- `.type` - Item type ('referenced_table', 'included_table', 'model', etc.)
- `.path` - File path (for referenced tables and artifacts)
- `.inputs` - List of input dependencies
- `.dependents` - List of items that depend on this item
- `.metadata` - Full metadata dictionary

In IPython/Jupyter, `folio.data.<TAB>` shows all available items with autocomplete!

## Directory Structure

DataFolio creates a transparent directory structure:

```
experiments/my_experiment/
├── metadata.json              # Folio metadata
├── tables/
│   ├── results.parquet       # DataFrame storage
│   └── results.json          # Table metadata
├── numpy/
│   ├── embeddings.npy        # Numpy array storage
│   └── embeddings.json       # Array metadata
├── json/
│   ├── config.json           # JSON data
│   └── config_meta.json      # JSON metadata
├── models/
│   ├── classifier.pkl        # Sklearn model
│   └── classifier.json       # Model metadata
├── pytorch_models/
│   ├── neural_net.pth        # PyTorch state dict
│   └── neural_net.json       # PyTorch metadata
└── artifacts/
    └── plot.png              # Any file type
```

## API Reference

See the [API Reference](reference/api.md) for complete method documentation.

## Examples

### Complete ML Workflow

```python
from datafolio import DataFolio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Initialize
folio = DataFolio('experiments/classifier_v1')

# Reference external data
folio.add_data('raw', reference='s3://bucket/raw.csv',
    description='Raw training data from database')

# Add processed data
folio.add_data('clean', cleaned_df,
    description='Cleaned and preprocessed data',
    inputs=['raw'])

# Add features
folio.add_data('features', feature_df,
    description='Engineered features',
    inputs=['clean'])

# Save embeddings
embeddings = np.random.randn(1000, 128)
folio.add_data('embeddings', embeddings,
    description='Model embeddings',
    inputs=['features'])

# Save configuration
folio.add_data('config', {
    'model_type': 'random_forest',
    'n_estimators': 100,
    'max_depth': 10
})

# Train and save model
clf = RandomForestClassifier(**folio.get_data('config'))
clf.fit(X_train, y_train)

folio.add_model('classifier', clf,
    description='Random forest classifier',
    hyperparameters=folio.get_data('config'),
    inputs=['features'])

# Save metrics
folio.add_data('metrics', {
    'accuracy': 0.95,
    'f1': 0.92,
    'precision': 0.94
})

# Add custom metadata to the folio itself
folio.metadata['experiment_name'] = 'rf_baseline'
folio.metadata['tags'] = ['classification', 'production']
folio.metadata['notes'] = 'Baseline random forest model for production deployment'

# View summary (shows data and custom metadata)
folio.describe()

# Access data with autocomplete
config = folio.data.config.content
metrics = folio.data.metrics.content
trained_model = folio.data.classifier.content

# Clean up intermediate data
folio.delete('embeddings')
```

### PyTorch Deep Learning Workflow

```python
import torch
import torch.nn as nn
from datafolio import DataFolio

# Define model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.fc = nn.Linear(32 * 30 * 30, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Create folio
folio = DataFolio('experiments/cnn_v1')

# Save training config
config = {
    'num_classes': 10,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}
folio.add_data('config', config)

# Train model
model = CNN(num_classes=config['num_classes'])
# ... training code ...

# Save model
folio.add_pytorch('cnn', model,
    description='CNN for image classification',
    init_args={'num_classes': 10},
    save_class=True)

# Save training history
history = {
    'train_loss': [0.5, 0.3, 0.2],
    'val_loss': [0.6, 0.4, 0.3],
    'val_acc': [0.7, 0.8, 0.85]
}
folio.add_data('history', history)

# Later: load and use
loaded_model = folio.get_pytorch('cnn', model_class=CNN)
loaded_model.eval()
```

## Best Practices

1. **Use descriptive names**: `add_data('training_features', ...)` not `add_data('data1', ...)`
2. **Track lineage**: Always specify `inputs` to track data dependencies
3. **Add descriptions**: Help future you understand what each item contains
4. **Use custom metadata**: Store experiment context in `folio.metadata` for better tracking
5. **Leverage autocomplete**: Use `folio.data.item_name.content` for cleaner, more discoverable code
6. **Clean up regularly**: Use `delete()` to remove temporary or obsolete items
7. **Version control**: Commit your folio directories to git (data is stored efficiently)
8. **Use references**: For large external datasets, use `reference` to avoid copying
9. **Check describe()**: Regularly review your folio with `folio.describe()` to see data and metadata
10. **Share across notebooks**: Multiple DataFolio instances can safely access the same bundle - changes are automatically detected and synchronized
11. **Snapshot before major changes**: Create snapshots before experimenting with new approaches—it's free insurance
12. **Tag snapshots meaningfully**: Use tags like `baseline`, `production`, `paper` to organize versions

## Development

See [CLAUDE.md](https://github.com/caseysm/datafolio/blob/main/CLAUDE.md) for development guidelines.

## License

MIT License - see LICENSE file for details.
