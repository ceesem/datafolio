# DataFolio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-265%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-69%25-green.svg)](tests/)

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

### Autocomplete-Friendly Access

Access your data with autocomplete support using the `folio.data` property:

```python
# Attribute-style access (autocomplete-friendly!)
df = folio.data.results.content          # Get DataFrame
desc = folio.data.results.description    # Get description
type_str = folio.data.results.type       # Get item type
inputs = folio.data.results.inputs       # Get lineage inputs

# Works for all data types
arr = folio.data.embeddings.content      # numpy array
cfg = folio.data.config.content          # dict
model = folio.data.classifier.content    # model object
```

In IPython/Jupyter, `folio.data.<TAB>` shows all available items with autocomplete!

## Directory Structure

DataFolio creates a transparent directory structure:

```text
experiments/my_experiment/
├── metadata.json              # Folio metadata
├── items.json                 # Unified manifest
├── tables/
│   └── results.parquet       # DataFrame storage
├── models/
│   └── classifier.joblib     # Sklearn models
└── artifacts/
    ├── embeddings.npy        # Numpy arrays
    ├── config.json           # JSON data
    └── plot.png              # Any file type
```

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

# Train and save model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

folio.add_model('classifier', clf,
    description='Random forest classifier',
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

# View summary (shows data and custom metadata)
folio.describe()

# Access data with autocomplete
config = folio.data.config.content
metrics = folio.data.metrics.content
trained_model = folio.data.classifier.content
```

### PyTorch Deep Learning

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

## Development

```bash
# Clone the repo
git clone https://github.com/caseysm/datafolio.git
cd datafolio

# Install with dev dependencies
uv sync

# Run tests
poe test

# Preview documentation
poe doc-preview

# Lint
uv run ruff check src/ tests/

# Bump version
poe bump patch  # or minor, major
```

## Documentation

For complete API documentation and detailed guides, see the [full documentation](docs/index.md).

## Requirements

- Python 3.10+
- pandas >= 2.0.0
- pyarrow >= 14.0.0
- joblib >= 1.3.0
- orjson >= 3.9.0
- cloudpickle >= 3.0.0

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`poe test`)
5. Submit a pull request

See [CLAUDE.md](CLAUDE.md) for development guidelines.

---

Made with ❤️ for data scientists who need simple, lightweight experiment tracking.
