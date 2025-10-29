# DataFolio

**A lightweight, filesystem-based data versioning and experiment tracking library for Python.**

DataFolio helps you organize, version, and track your data science experiments by storing datasets, models, and artifacts in a simple, transparent directory structure. Everything is saved as plain files (Parquet, JSON, pickle) that you can inspect, version with git, or backup to any storage system.

## Features

- **Universal Data Management**: Single `add_data()` method automatically handles DataFrames, numpy arrays, dicts, lists, and scalars
- **Model Support**: Save and load scikit-learn and PyTorch models with full metadata tracking
- **Data Lineage**: Track inputs and dependencies between datasets and models
- **External References**: Point to data stored externally (S3, local paths) without copying
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

# View everything
folio.describe()
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

The `describe()` method provides a compact overview of all data:

```python
# Print to console
folio.describe()

# Get as string
summary = folio.describe(return_string=True)

# Show empty sections
folio.describe(show_empty=True)
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
```

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

# View summary
folio.describe()
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
4. **Version control**: Commit your folio directories to git (data is stored efficiently)
5. **Use references**: For large external datasets, use `reference` to avoid copying
6. **Check describe()**: Regularly review your folio with `folio.describe()`

## Development

See [CLAUDE.md](https://github.com/caseysm/datafolio/blob/main/CLAUDE.md) for development guidelines.

## License

MIT License - see LICENSE file for details.
