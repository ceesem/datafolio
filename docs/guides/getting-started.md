# Getting Started with DataFolio

DataFolio is a lightweight, filesystem-based experiment tracking library that helps you organize data science experiments by storing datasets, models, and artifacts in a simple, transparent directory structure.

## Why DataFolio?

Traditional experiment tracking solutions can be heavyweight, require servers, or lock you into specific platforms. DataFolio takes a different approach:

- **Filesystem-based** - Everything is just files on disk
- **Transparent** - Standard formats (Parquet, JSON, pickle) you can inspect
- **Portable** - Works locally, in notebooks, on clusters, or in the cloud
- **Git-friendly** - Version control your entire experiment
- **Simple** - Intuitive Python API with minimal boilerplate

## Installation

```bash
pip install datafolio
```

This installs both the Python library and the `datafolio` CLI tool.

## Quick Start

### Your First Bundle

A "bundle" is DataFolio's way of organizing an experiment. Think of it as a project folder:

```python
from datafolio import DataFolio
import pandas as pd

# Create a new bundle
folio = DataFolio('experiments/my_first_experiment')

# Add some data
df = pd.DataFrame({
    'feature_1': [1, 2, 3],
    'feature_2': [4, 5, 6],
    'target': [0, 1, 0]
})

folio.add_data('training_data', df)

# View what's in the bundle
folio.describe()
```

Output:
```
DataFolio: experiments/my_first_experiment
==========================================

Tables (1):
  • training_data
    ↳ shape: [3, 3]
```

### What Just Happened?

DataFolio created a directory structure:

```
experiments/my_first_experiment/
├── metadata.json          # Bundle metadata
├── items.json            # Manifest of all items
└── tables/
    └── training_data.parquet  # Your DataFrame
```

Everything is saved automatically. No need to call `save()` or `commit()`.

## Core Concepts

### Bundles

A bundle is a self-contained experiment with:

- **Data items** (tables, arrays, JSON, models, files)
- **Metadata** (custom key-value pairs about the experiment)
- **Lineage** (relationships between data items)

```python
# Create or open a bundle
folio = DataFolio('path/to/bundle')

# Add custom metadata
folio.metadata['experiment_name'] = 'baseline_v1'
folio.metadata['date'] = '2025-01-20'
folio.metadata['tags'] = ['classification', 'baseline']

# Metadata is automatically saved
```

### Data Items

DataFolio supports multiple data types, each optimized for its use case:

| Type | Examples | Storage Format |
|------|----------|---------------|
| **Tables** | DataFrames | Parquet |
| **Numpy Arrays** | Embeddings, tensors | `.npy` |
| **JSON** | Configs, metrics, lists | `.json` |
| **Models** | sklearn | `.joblib`, `.skops` |
| **Artifacts** | Images, PDFs, any file | Original format |
| **References** | External data (S3, etc.) | Metadata only |

### The Universal `add_data()` Method

For simplicity, use `add_data()` which automatically detects the type:

```python
# Automatically handles different types
folio.add_data('df', dataframe)           # Table
folio.add_data('embeddings', np_array)    # Numpy
folio.add_data('config', {'lr': 0.01})    # JSON
folio.add_data('model', sklearn_model)    # Model
folio.add_data('score', 0.95)             # JSON (scalar)
```

Or use type-specific methods for more control:

```python
folio.add_table('df', dataframe, description='Training data')
folio.add_numpy('embeddings', array, description='Word embeddings')
folio.add_json('config', config_dict, description='Model config')
folio.add_model('clf', model, description='Random forest')
```

## Working with Data

### Adding Data

```python
import pandas as pd
import numpy as np

# Tables (DataFrames)
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
folio.add_table('data', df,
    description='Experimental data',
    inputs=['raw_data'])  # Optional: track lineage

# Numpy arrays
embeddings = np.random.randn(100, 128)
folio.add_numpy('embeddings', embeddings,
    description='Model embeddings')

# JSON data (configs, metrics, lists)
config = {'learning_rate': 0.01, 'batch_size': 32}
folio.add_json('config', config,
    description='Training configuration')

# Scalars are stored as JSON
folio.add_json('accuracy', 0.95)

# Files/artifacts
folio.add_artifact('plot.png', 'path/to/plot.png',
    description='Training curve')
```

### Retrieving Data

```python
# Get by type-specific method
df = folio.get_table('data')
arr = folio.get_numpy('embeddings')
config = folio.get_json('config')

# Or use universal get_data()
df = folio.get_data('data')        # Returns DataFrame
arr = folio.get_data('embeddings') # Returns numpy array
config = folio.get_data('config')  # Returns dict
```

### Autocomplete-Friendly Access

For a better developer experience, use the `folio.data` accessor:

```python
# Attribute-style access (great for autocomplete!)
df = folio.data.training_data.content
config = folio.data.config.content
model = folio.data.classifier.content

# Access metadata
desc = folio.data.training_data.description
inputs = folio.data.training_data.inputs
item_type = folio.data.training_data.type

# In Jupyter/IPython, use TAB completion
folio.data.<TAB>  # Shows all available items
```

### Overwriting Data

```python
# Add initial data
folio.add_data('model', model_v1)

# Overwrite with new version
folio.add_data('model', model_v2, overwrite=True)

# Without overwrite=True, you'll get an error
folio.add_data('model', model_v3)  # Error: item exists!
```

### Deleting Data

```python
# Delete single item
folio.delete('old_model')

# Delete multiple items
folio.delete(['temp1', 'temp2', 'debug_data'])

# DataFolio warns if deleted items have dependents
folio.delete('train_data')  # Warns if other items depend on it
folio.delete('train_data', warn_dependents=False)  # Skip warning
```

## Working with Models

### Scikit-learn Models

```python
from sklearn.ensemble import RandomForestClassifier

# Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(X_train, y_train)

# Save model
folio.add_model('classifier', clf,
    description='Random forest classifier',
    hyperparameters={'n_estimators': 100, 'max_depth': 10},
    inputs=['training_data'])

# Load model
loaded_clf = folio.get_model('classifier')
predictions = loaded_clf.predict(X_test)
```

### Custom Models with Skops

DataFolio supports custom sklearn-compatible models using [skops](https://skops.readthedocs.io/). This is particularly useful for pipelines with custom transformers that need to be portable across environments.

**When to use skops format (`custom=True`):**
- Pipelines with custom transformers that need to work across different machines
- Models that need to be deployed without access to the original class definitions
- Better security for model deployment (skops provides secure serialization)

**Key requirement:** Custom transformers must inherit from sklearn base classes:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# ✅ CORRECT: Inherits from sklearn mixins
class PercentileClipper(BaseEstimator, TransformerMixin):
    """Custom transformer that clips values to percentile bounds."""

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_bound_ = np.percentile(X, self.lower, axis=0)
        self.upper_bound_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bound_, self.upper_bound_)

# ❌ WRONG: Plain class without sklearn mixins
class BadTransformer:  # Won't work with skops!
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
```

**Why inherit from `BaseEstimator` and `TransformerMixin`?**
- `BaseEstimator`: Provides `get_params()` and `set_params()` methods required by sklearn
- `TransformerMixin`: Provides `fit_transform()` method automatically
- Ensures compatibility with sklearn's Pipeline and other utilities
- Required for skops to properly serialize and deserialize your custom class

**Using custom transformers in pipelines:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create pipeline with custom transformer
pipeline = Pipeline([
    ('clipper', PercentileClipper(lower=5, upper=95)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# Fit pipeline
X_train = np.random.randn(100, 5)
y_train = np.random.randint(0, 2, 100)
pipeline.fit(X_train, y_train)

# Save with skops format (custom=True)
folio.add_sklearn('custom_pipeline', pipeline,
    custom=True,  # Uses skops for portability
    description='Pipeline with custom percentile clipper')

# Load in a different environment (doesn't need PercentileClipper class!)
folio2 = DataFolio('path/to/bundle')
loaded_pipeline = folio2.get_sklearn('custom_pipeline')
predictions = loaded_pipeline.predict(X_test)
```

**Comparison of serialization formats:**

| Format | When to Use | Pros | Cons |
|--------|------------|------|------|
| **joblib** (default) | Standard sklearn models, XGBoost, LightGBM | Fast, widely supported | Requires class definitions on load |
| **skops** (`custom=True`) | Custom transformers, deployment | Portable, more secure | Slightly slower |

```python
# Joblib format (default)
folio.add_sklearn('model', pipeline)  # Uses joblib

# Skops format (portable)
folio.add_sklearn('model', pipeline, custom=True)  # Uses skops

# Both work through generic add_model() too
folio.add_model('model', pipeline, custom=True)
```

**Best practices for custom transformers:**

1. **Always inherit from sklearn base classes:**
   ```python
   from sklearn.base import BaseEstimator, TransformerMixin

   class MyTransformer(BaseEstimator, TransformerMixin):
       ...
   ```

2. **Store fitted parameters with trailing underscore:**
   ```python
   def fit(self, X, y=None):
       self.mean_ = np.mean(X)  # Fitted params end with _
       return self
   ```

3. **Initialize all parameters in `__init__`:**
   ```python
   def __init__(self, threshold=0.5):
       self.threshold = threshold  # Store all params
   ```

4. **Always return `self` from `fit()`:**
   ```python
   def fit(self, X, y=None):
       # ... fitting logic ...
       return self  # Required for sklearn API
   ```

## Data Lineage

Track dependencies between your data items to understand your workflow:

```python
# Reference external data
folio.reference_table('raw_data',
    reference='s3://bucket/raw_data.parquet',
    description='Original raw data from database')

# Add processed data with lineage
folio.add_table('cleaned_data', cleaned_df,
    description='Cleaned and preprocessed',
    inputs=['raw_data'])  # Depends on raw_data

# Add features
folio.add_table('features', feature_df,
    description='Engineered features',
    inputs=['cleaned_data'])  # Depends on cleaned_data

# Add model
folio.add_model('classifier', model,
    description='Trained classifier',
    inputs=['features'])  # Depends on features

# View the lineage chain
folio.describe()
```

Output shows the dependency chain:
```
Tables (2):
  • raw_data (reference): Original raw data from database
    ↳ path: s3://bucket/raw_data.parquet
  • cleaned_data: Cleaned and preprocessed
    ↳ inputs: raw_data
    ↳ shape: [10000, 25]
  • features: Engineered features
    ↳ inputs: cleaned_data
    ↳ shape: [10000, 50]

Models (1):
  • classifier: Trained classifier
    ↳ inputs: features
```

### Why Track Lineage?

- **Understand workflows** - See how data flows through your pipeline
- **Debug issues** - Trace problems back to their source
- **Reproduce results** - Know exactly which data created which results
- **Cleanup safely** - DataFolio warns when deleting items with dependents

## External References

For large datasets stored elsewhere (S3, network drives, etc.), use references instead of copying:

```python
# Reference data without copying
folio.reference_table('huge_dataset',
    reference='s3://my-bucket/data/train.parquet',
    description='10GB training dataset')

# Reference with additional metadata
folio.reference_table('cloud_data',
    reference='gs://bucket/data.csv',
    description='Data in Google Cloud Storage',
    num_rows=1_000_000,
    num_cols=500)

# Later, access the path
path = folio.data.huge_dataset.path  # 's3://my-bucket/data/train.parquet'

# Load with pandas/pyarrow
import pandas as pd
df = pd.read_parquet(path)  # Reads directly from S3
```

## Bundle Metadata

Store experiment-level information in the bundle metadata:

```python
# Add custom metadata
folio.metadata['experiment_name'] = 'baseline_v1'
folio.metadata['researcher'] = 'Alice'
folio.metadata['date_started'] = '2025-01-20'
folio.metadata['hypothesis'] = 'Random forest will outperform logistic regression'
folio.metadata['tags'] = ['classification', 'baseline', 'production']
folio.metadata['notes'] = 'First experiment with cleaned dataset'

# Metadata is automatically saved

# Access metadata
print(folio.metadata['experiment_name'])

# View all metadata
folio.describe()  # Shows metadata section
```

The `describe()` method automatically formats and displays your custom metadata.

## Describing Your Bundle

Get a comprehensive overview of your bundle:

```python
# Print to console (default)
folio.describe()

# Get as string
summary = folio.describe(return_string=True)
print(summary)

# Show empty sections
folio.describe(show_empty=True)

# Limit metadata fields shown
folio.describe(max_metadata_fields=5)
```

Example output:
```
DataFolio: experiments/classifier_v1
====================================

Tables (2):
  • raw_data (reference): Original raw data
    ↳ path: s3://bucket/raw.parquet
  • features: Engineered features
    ↳ inputs: cleaned_data
    ↳ shape: [10000, 50]

Numpy Arrays (1):
  • embeddings: Model embeddings
    ↳ shape: [100, 128], dtype: float64
    ↳ inputs: features

Models (1):
  • classifier: Random forest classifier
    ↳ inputs: features
    ↳ hyperparameters: {'n_estimators': 100, 'max_depth': 10}

Metadata (5):
  • experiment_name: baseline_v1
  • researcher: Alice
  • tags: ['classification', 'baseline'] (list, 2 items)
  • hypothesis: Random forest will outperform logistic... (truncated)
  ... and 2 more fields
```

## Multi-Instance Access

Multiple notebooks or processes can safely access the same bundle:

```python
# Notebook 1: Create bundle
folio1 = DataFolio('experiments/shared')
folio1.add_data('results', df1)

# Notebook 2: Open same bundle
folio2 = DataFolio('experiments/shared')
print(folio2.describe())  # Shows 'results'

# Notebook 1: Add more data
folio1.add_data('analysis', df2)

# Notebook 2: Automatically sees new data!
folio2.describe()  # Now shows both 'results' and 'analysis'
data = folio2.get_data('analysis')  # Works immediately ✅
```

All read operations automatically refresh from disk, so you always see the latest state.

## Complete Workflow Example

Here's a complete example from data loading to model deployment:

```python
from datafolio import DataFolio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 1. Initialize bundle
folio = DataFolio('experiments/fraud_detection_v1')
folio.metadata['experiment_name'] = 'fraud_detection_baseline'
folio.metadata['date'] = '2025-01-20'
folio.metadata['tags'] = ['classification', 'fraud', 'baseline']

# 2. Reference external raw data
folio.reference_table('raw_data',
    reference='s3://data-lake/fraud/raw_2024.parquet',
    description='Raw transaction data from 2024',
    num_rows=1_000_000,
    num_cols=25)

# 3. Load and clean data
raw_df = pd.read_parquet('s3://data-lake/fraud/raw_2024.parquet')
cleaned_df = clean_data(raw_df)  # Your cleaning function

folio.add_table('cleaned_data', cleaned_df,
    description='Cleaned transaction data',
    inputs=['raw_data'])

# 4. Engineer features
features_df = engineer_features(cleaned_df)  # Your feature engineering

folio.add_table('features', features_df,
    description='Engineered features for classification',
    inputs=['cleaned_data'])

# 5. Train/test split
X = features_df.drop('is_fraud', axis=1)
y = features_df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 8. Save model and results
folio.add_model('classifier', clf,
    description='Random forest fraud classifier',
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    inputs=['features'])

folio.add_json('metrics', {
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
})

folio.add_json('feature_importance', {
    feature: float(importance)
    for feature, importance in zip(X.columns, clf.feature_importances_)
})

# 9. Update metadata with results
folio.metadata['accuracy'] = float(accuracy)
folio.metadata['f1_score'] = float(f1)
folio.metadata['status'] = 'completed'

# 10. View summary
folio.describe()

# 11. Later: Load and use in production
production_folio = DataFolio('experiments/fraud_detection_v1')
model = production_folio.get_model('classifier')
metrics = production_folio.get_json('metrics')

print(f"Deploying model with accuracy: {metrics['accuracy']}")
predictions = model.predict(new_transactions)
```

## Directory Structure

DataFolio creates an intuitive directory structure:

```
experiments/my_experiment/
├── metadata.json              # Bundle metadata
├── items.json                # Manifest of all items
├── snapshots.json            # Snapshot registry (if using snapshots)
│
├── tables/
│   └── features.parquet      # DataFrames
│
├── models/
│   └── classifier.joblib     # Scikit-learn models
│
├── numpy/
│   └── embeddings.npy       # Numpy arrays
│
└── artifacts/
    ├── config.json          # JSON data
    ├── plot.png            # Images
    └── report.pdf          # Any file type
```

All files use standard formats:
- **Parquet** for DataFrames (efficient, columnar)
- **JSON** for configs and metrics (human-readable)
- **Joblib/Skops** for scikit-learn models
- **Numpy** `.npy` for arrays

You can inspect any file directly without DataFolio!

## Tips and Tricks

### 1. Use Descriptive Names

```python
# Good
folio.add_data('training_features_v2', df)
folio.add_model('random_forest_baseline', model)

# Bad
folio.add_data('data1', df)
folio.add_model('model', model)
```

### 2. Add Descriptions

```python
# Always add descriptions
folio.add_table('features', df,
    description='Engineered features with PCA and polynomial terms')

# Future you will thank present you
```

### 3. Track Lineage

```python
# Always specify inputs
folio.add_table('features', feature_df,
    inputs=['cleaned_data'])

# This helps you understand the data flow
```

### 4. Use Custom Metadata

```python
# Store experiment context
folio.metadata['experiment_type'] = 'hyperparameter_tuning'
folio.metadata['best_params'] = {'n_estimators': 100, 'max_depth': 10}
folio.metadata['notes'] = 'Best results from grid search over 50 configs'
```

### 5. Clean Up Regularly

```python
# Delete temporary data
folio.delete(['debug_data', 'temp_results', 'old_model_v1'])

# Check before deleting
folio.describe()  # Review what you have
```

### 6. Use References for Large Data

```python
# Don't copy huge datasets
folio.reference_table('training_data',
    reference='s3://bucket/huge_data.parquet')

# Load directly from source when needed
df = pd.read_parquet(folio.data.training_data.path)
```

### 7. Leverage Autocomplete

```python
# This is more discoverable
config = folio.data.config.content
model = folio.data.classifier.content

# Than this
config = folio.get_data('config')
model = folio.get_data('classifier')
```

### 8. Commit to Git

```python
# Your bundle is git-friendly
cd experiments/my_experiment
git add .
git commit -m "Baseline model - 89% accuracy"
git push
```

### 9. Use Snapshots for Versions

```python
# Create snapshots at milestones
folio.create_snapshot('v1.0-baseline',
    description='Initial baseline model')

# Experiment freely
folio.add_model('classifier', new_model, overwrite=True)

# Return to baseline anytime
baseline = DataFolio.load_snapshot('experiments/exp', 'v1.0-baseline')
```

See the [Snapshots Guide](snapshots.md) for more details.

### 10. Use the CLI

```bash
# Describe bundle from terminal
datafolio describe

# List snapshots
datafolio snapshot list

# Compare versions
datafolio snapshot compare v1.0 v2.0
```

## Common Patterns

### Experiment Template

```python
def run_experiment(name, config):
    # Initialize
    folio = DataFolio(f'experiments/{name}')
    folio.metadata.update(config)
    folio.metadata['status'] = 'running'

    # Load data
    data = load_data(config['data_source'])
    folio.add_data('data', data)

    # Train
    model = train_model(data, config)
    folio.add_model('model', model)

    # Evaluate
    metrics = evaluate_model(model, data)
    folio.add_json('metrics', metrics)
    folio.metadata.update(metrics)
    folio.metadata['status'] = 'completed'

    return folio

# Run experiments
exp1 = run_experiment('baseline', {'lr': 0.01, 'data_source': 'train.csv'})
exp2 = run_experiment('tuned', {'lr': 0.001, 'data_source': 'train.csv'})
```

### A/B Test Comparison

```python
# Load two experiments
baseline = DataFolio('experiments/baseline')
variant = DataFolio('experiments/variant_a')

# Compare
print(f"Baseline accuracy: {baseline.metadata['accuracy']}")
print(f"Variant accuracy: {variant.metadata['accuracy']}")

# Deploy winner
if variant.metadata['accuracy'] > baseline.metadata['accuracy']:
    model = variant.get_model('classifier')
else:
    model = baseline.get_model('classifier')
```

### Hyperparameter Grid Search

```python
from itertools import product

# Grid
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

results = []

# Try each combination
for n_est, depth in product(params['n_estimators'], params['max_depth']):
    # Create bundle
    folio = DataFolio(f'experiments/grid_search/n{n_est}_d{depth}')

    # Train
    model = RandomForestClassifier(n_estimators=n_est, max_depth=depth)
    model.fit(X_train, y_train)

    # Evaluate
    acc = model.score(X_test, y_test)

    # Save
    folio.add_model('model', model)
    folio.metadata['n_estimators'] = n_est
    folio.metadata['max_depth'] = depth
    folio.metadata['accuracy'] = acc

    results.append((n_est, depth, acc))

# Find best
best = max(results, key=lambda x: x[2])
print(f"Best: n_estimators={best[0]}, max_depth={best[1]}, acc={best[2]}")
```

## Next Steps

- **Learn about snapshots** - See the [Snapshots Guide](snapshots.md) for versioning experiments
- **API Reference** - Check the [API docs](../reference/api.md) for all methods
- **Examples** - Browse the main [documentation](../index.md) for more examples
- **CLI Tools** - Use `datafolio --help` to explore the command-line interface

## Common Questions

**Q: How is this different from MLflow/Weights & Biases?**

A: DataFolio is filesystem-based and self-contained. No servers, no databases, no accounts. Everything is just files you can inspect, version with git, and move around.

**Q: Can I use this in production?**

A: Yes! DataFolio bundles are self-contained and can be deployed anywhere. Load a bundle, get your model, and run inference.

**Q: Does it work with cloud storage?**

A: Yes! DataFolio supports any storage backend via `cloud-files` (S3, GCS, Azure, etc.). Just use cloud paths:

```python
folio = DataFolio('s3://my-bucket/experiments/exp1')
```

**Q: How do I share bundles with colleagues?**

A: Just share the directory! Everything is self-contained. You can:
- Commit to git
- Copy to shared storage
- Zip and email
- Mount network drives

**Q: What about versioning?**

A: Use [Snapshots](snapshots.md)! They let you create immutable checkpoints without duplicating data.

**Q: Can I use this with Jupyter notebooks?**

A: Absolutely! DataFolio works great in notebooks. Multiple notebooks can even access the same bundle simultaneously.

## Need Help?

- **Documentation**: Check the [full docs](../index.md)
- **Issues**: Report bugs on [GitHub](https://github.com/ceesem/datafolio/issues)
- **Examples**: See the repository for example notebooks
