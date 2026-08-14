# Getting Started with DataFolio

DataFolio is an annotated working directory for related data. It saves common
Python objects in ordinary formats, records how to load them in one readable
catalog, and lets you reopen the whole collection with one path.

Its purpose is deliberately narrow: remove repetitive save/load wiring and
leave behind a directory that future you—or someone without DataFolio—can
understand.

## The mental model

A folio consists of:

1. **A directory** containing ordinary Parquet, JSON, NumPy, model, and
   artifact files.
2. **An `items.json` catalog** recording names, descriptions, formats,
   relationships, references, and snapshots.
3. **A dispatcher** that connects `add()` and `get()` to the appropriate
   writer and reader.

DataFolio organizes the boundary around your data. It does not replace pandas,
Polars, PyArrow, object-store tools, or workflow systems.

## Installation

```bash
pip install datafolio
```

This installs both the Python library and the `datafolio` CLI tool.

## Quick Start

### Your first folio

A folio is just a directory with a catalog. It can represent an experiment,
an analysis, a prepared dataset, or any other small collection of related
objects.

```python
from datafolio import DataFolio
import pandas as pd

# Create a new folio (or open it if it already exists)
folio = DataFolio('experiments/my_first_experiment')

# Add some data
df = pd.DataFrame({
    'feature_1': [1, 2, 3],
    'feature_2': [4, 5, 6],
    'target': [0, 1, 0]
})

folio.add(
    'training_data',
    df,
    description='Three-row example table used by the introductory notebook',
)

# View what's in the bundle
folio.describe()
```

Output:
```
DataFolio: experiments/my_first_experiment
==========================================

Tables (1):
  • training_data: Three-row example table used by the introductory notebook
    ↳ size: 2.1 KB
```

### What Just Happened?

DataFolio created an ordinary directory and recorded the table in its catalog:

```
experiments/my_first_experiment/
├── items.json            # Metadata, item catalog, snapshots, revision
├── CONTENTS.md           # Human-readable inventory
└── tables/
    └── training_data--r2.parquet
```

The dataframe was dispatched to Parquet and its name, description, format, and
loading information were written to `items.json`. There is no separate
`save()` or `commit()` call.

## Core Concepts

### The folio path

The path is the handle for the entire collection. You can pass it between
notebooks or use the same analysis code with another similarly organized
folio:

```python
folio = DataFolio(FOLIO_PATH)

features = folio.get('features')
labels = folio.get('labels')
config = folio.get('config')
```

### Descriptions

Descriptions are first-class catalog information. Add them while the meaning
of an object is still obvious:

```python
folio.add(
    'labels_reviewed',
    labels,
    description='Manual labels after the March review; ambiguous rows removed',
)
```

Descriptions appear in `describe()` and `CONTENTS.md`. They are often the most
useful part of the folio when returning to a project months later.

### Included objects and references

An included object is saved inside the folio and moves with the directory. An
external reference stores an absolute link but does not copy or own the data:

```python
folio.reference_table(
    'raw',
    path='gs://bucket/releases/2026-07/raw.parquet',
    description='Published source table; not owned by this folio',
)
```

Snapshots preserve included data. For references, they preserve the recorded
link rather than the bytes at the external location.

### Supported objects

DataFolio uses a small set of sensible storage conventions:

| Type | Examples | Storage Format |
|------|----------|---------------|
| **Tables** | pandas / Polars DataFrames | Parquet |
| **Numpy Arrays** | Embeddings, tensors | `.npy` |
| **JSON** | Configs, metrics, lists | `.json` |
| **Models** | sklearn | `.joblib`, `.skops` |
| **Files** | Images, PDFs, any file | Original format |
| **References** | External tables (local, GCS, S3, etc.) | Catalog entry only |

### Save and load dispatch

`add()` detects common object types and selects the matching writer:

```python
# Automatically handles different types
folio.add('df', dataframe)           # Table
folio.add('embeddings', np_array)    # Numpy
folio.add('config', {'lr': 0.01})    # JSON
folio.add('model', sklearn_model)    # Model (real sklearn/XGBoost/LightGBM/CatBoost estimators)
folio.add('score', 0.95)             # JSON (scalar)
```

Strings are always stored as JSON — a string that looks like a path is never
treated as a file. Use the dedicated methods for what `add()` doesn't cover:

```python
folio.add_model('clf', model, description='Random forest')      # any picklable object
folio.add_file('path/to/plot.png', name='plot', description='Training curve')
folio.reference_table('raw', path='s3://bucket/raw.parquet')    # external data, no copy
```

`get()` consults the catalog and uses the corresponding reader. For tables,
`scan_table()` hands off to Polars when lazy execution is the better tool.

## Working with Data

### Adding Data

```python
import pandas as pd
import numpy as np

# Tables (pandas or Polars DataFrames)
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
folio.add('data', df,
    description='Experimental data',
    inputs=['raw_data'])  # Optional: track lineage

# A non-default pandas index warns that it is dropped;
# keep it explicitly with preserve_index=True
folio.add('indexed_data', df_with_index, preserve_index=True)

# Numpy arrays
embeddings = np.random.randn(100, 128)
folio.add('embeddings', embeddings,
    description='Model embeddings')

# JSON data (configs, metrics, lists)
config = {'learning_rate': 0.01, 'batch_size': 32}
folio.add('config', config,
    description='Training configuration')

# Scalars are stored as JSON (note: NaN/inf serialize as null)
folio.add('accuracy', 0.95)

# Files
folio.add_file('path/to/plot.png', name='plot.png',
    description='Training curve')
```

### Retrieving Data

```python
# Universal get() returns the right type
df = folio.get('data')        # Returns DataFrame
arr = folio.get('embeddings') # Returns numpy array
config = folio.get('config')  # Returns dict

pl_df = folio.get('data', frame='polars')  # eager Polars DataFrame

# get() on a file item returns the payload path
plot_path = folio.get('plot.png')
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
folio.add('model', model_v1)

# Overwrite with new version
folio.add('model', model_v2, overwrite=True)

# Without overwrite=True, you'll get an error
folio.add('model', model_v3)  # Error: item exists!
```

`overwrite=True` is required to replace *any* existing item. If the item is
pinned by a snapshot, the prior version is preserved via copy-on-write.

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
folio.add_model('custom_pipeline', pipeline,
    custom=True,  # Uses skops for portability
    description='Pipeline with custom percentile clipper')

# Load in a different environment (doesn't need PercentileClipper class!)
# skops refuses unknown types by default, so trusted=True is required
folio2 = DataFolio('path/to/bundle')
loaded_pipeline = folio2.get_model('custom_pipeline', trusted=True)
predictions = loaded_pipeline.predict(X_test)
```

**Comparison of serialization formats:**

| Format | When to Use | Pros | Cons |
|--------|------------|------|------|
| **joblib** (default) | Standard sklearn models, XGBoost, LightGBM | Fast, widely supported | Requires class definitions on load |
| **skops** (`custom=True`) | Custom transformers, deployment | Portable, more secure | Slightly slower |

```python
# Joblib format (default)
folio.add_model('model', pipeline)  # Uses joblib

# Skops format (portable)
folio.add_model('model', pipeline, custom=True)  # Uses skops

# Loading a skops model requires opting in to trust its contents
model = folio.get_model('model', trusted=True)
```

Joblib models are pickle under the hood, so the same rule applies everywhere:
only load folios you trust.

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
    path='s3://bucket/raw_data.parquet',
    description='Original raw data from database')

# Add processed data with lineage
folio.add('cleaned_data', cleaned_df,
    description='Cleaned and preprocessed',
    inputs=['raw_data'])  # Depends on raw_data

# Add features
folio.add('features', feature_df,
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
Tables (3):
  • raw_data (reference): Original raw data from database
    ↳ path: s3://bucket/raw_data.parquet
  • cleaned_data: Cleaned and preprocessed
    ↳ size: 1.5 MB
    ↳ inputs: raw_data
  • features: Engineered features
    ↳ size: 3.2 MB
    ↳ inputs: cleaned_data

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
    path='s3://my-bucket/data/train.parquet',
    description='10GB training dataset')

# Reference with additional metadata
folio.reference_table('cloud_data',
    path='gs://bucket/data.csv',
    description='Data in Google Cloud Storage',
    num_rows=1_000_000,
    num_cols=500)

# Later, access the path
path = folio.data.huge_dataset.path  # 's3://my-bucket/data/train.parquet'

# Load with pandas/pyarrow
import pandas as pd
df = pd.read_parquet(path)  # Reads directly from S3
```

## Folio Metadata

Store collection-level context alongside the item catalog:

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

!!! note "NaN/inf in metadata values"
    Metadata is persisted as JSON, which has no representation for non-finite
    floats — the same NaN/inf → `null` conversion that applies to JSON items
    also applies to metadata values (`folio.metadata['score'] = float('nan')`
    reads back as `None`).

## Describing Your Folio

Get a comprehensive overview of the directory and its contents:

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
    ↳ size: 3.2 MB
    ↳ inputs: cleaned_data

Numpy Arrays (1):
  • embeddings: Model embeddings
    ↳ shape: [100, 128], dtype: float64
    ↳ inputs: features

Models (1):
  • classifier: Random forest classifier
    ↳ inputs: features

Metadata (5):
  • experiment_name: baseline_v1
  • researcher: Alice
  • tags: ['classification', 'baseline'] (list, 2 items)
  • hypothesis: Random forest will outperform logistic... (truncated)
  ... and 2 more fields
```

## Multi-Instance Access

Multiple notebooks can share a folio under a **many readers, one active
writer** model:

```python
# Notebook 1: Create a folio
folio1 = DataFolio('experiments/shared')
folio1.add('results', df1)

# Notebook 2: Open same bundle
folio2 = DataFolio('experiments/shared')
print(folio2.describe())  # Shows 'results'

# Notebook 1: Add more data
folio1.add('analysis', df2)

# Notebook 2: Automatically sees new data!
folio2.describe()  # Now shows both 'results' and 'analysis'
data = folio2.get('analysis')  # Works immediately ✅
```

All read operations automatically refresh from disk, so readers always see
the latest committed state. Writing is single-writer: if another notebook has
advanced the folio since yours last loaded it, your write raises
`ConcurrentWriteError` instead of clobbering their work — call `refresh()`
and retry. Local writers are serialized with a lock file; cloud folios have
no cross-machine lock and should be treated as single-writer.

### Reading Many Items at Once

Cloud reads are round-trip bound, so a loop of small reads spends its time
waiting, not moving data. Two things help, and they compose.

`pinned()` checks freshness once on entry and suspends the per-read recheck
for the block:

```python
with folio.pinned():
    for name in folio.tables():
        process(folio.get(name))   # no per-read staleness round trips
```

The trade is explicit and is the point: **another writer's changes are not
visible until the block exits**, so the reads are one coherent view rather
than individually up-to-date. Nesting is re-entrant, writes inside work
normally (including the stale-writer check), and an exception still unpins.
Don't hold one open across a long loop that must observe another process's
writes.

`get_many()` goes further and overlaps the payload reads themselves:

```python
tables = folio.get_many(folio.tables())     # {name: DataFrame}
```

It is pure sugar for a `pinned()` block around a thread pool mapping
`get()` — write that loop yourself if you prefer, it is equally fast:

```python
with folio.pinned():
    with ThreadPoolExecutor(20) as ex:
        results = list(ex.map(folio.get, names))
```

Measured on 60 items at a 250ms round trip: 45.9s naive, 15.6s pinned,
**1.1s with `get_many`**. Use the pin whenever you thread reads by hand —
it isn't only about speed. A concurrent auto-refresh rebuilds the item
table in place, and a thread reading it mid-rebuild can miss an item that
is really there.

## Taking a Cloud Folio Offline

A folio is ordinary files plus a self-contained, relative-path catalog—so
downloading one is a job for the tool that owns downloading, not for
DataFolio:

```bash
gsutil -m rsync -r gs://bucket/experiments/my-exp ~/analysis/my-exp
# or: aws s3 sync s3://bucket/experiments/my-exp ~/analysis/my-exp
# or: rclone sync remote:bucket/experiments/my-exp ~/analysis/my-exp
```

```python
folio = DataFolio('~/analysis/my-exp')   # a complete, normal local folio
```

You get parallel, resumable, incremental transfer, and the copy is complete:
`items.json`, all included payloads, and every snapshot version. Re-running the
sync later picks up only what changed. (One caution for two-way sync
tools: exclude `items.json.lock` — it is the local write lock, and deleting
it out from under a live writer breaks write serialization.)

The one thing that stays where it is: external references
(`reference_table`) keep pointing at their original locations — datafolio
never copies data it doesn't own. `folio.mutable_references()` lists them.

(`folio.copy()` also works cloud→local, but it *forks* rather than clones:
current item versions only, no snapshot history, fresh bundle identity. Use
it for derived experiments; use a sync tool for downloads.)

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
    path='s3://data-lake/fraud/raw_2024.parquet',
    description='Raw transaction data from 2024',
    num_rows=1_000_000,
    num_cols=25)

# 3. Load and clean data
raw_df = pd.read_parquet('s3://data-lake/fraud/raw_2024.parquet')
cleaned_df = clean_data(raw_df)  # Your cleaning function

folio.add('cleaned_data', cleaned_df,
    description='Cleaned transaction data',
    inputs=['raw_data'])

# 4. Engineer features
features_df = engineer_features(cleaned_df)  # Your feature engineering

folio.add('features', features_df,
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
    inputs=['features'])

folio.add('metrics', {
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
})

folio.add('feature_importance', {
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
metrics = production_folio.get('metrics')

print(f"Deploying model with accuracy: {metrics['accuracy']}")
predictions = model.predict(new_transactions)
```

## Directory Structure

DataFolio creates a small, inspectable directory structure:

```
experiments/my_experiment/
├── items.json                 # Metadata, item catalog, snapshots, revision
├── CONTENTS.md                # Human-readable inventory
│
├── tables/
│   └── features--r4.parquet   # DataFrames
│
├── models/
│   └── classifier--r6.joblib  # Scikit-learn models
│
└── artifacts/
    ├── embeddings--r3.npy    # Numpy arrays
    ├── config--r5.json       # JSON data
    ├── plot--r7.png          # Images
    └── report--r8.pdf        # Any file type
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
folio.add('training_features_v2', df)
folio.add_model('random_forest_baseline', model)

# Bad
folio.add('data1', df)
folio.add_model('model', model)
```

### 2. Add Descriptions

```python
# Always add descriptions
folio.add('features', df,
    description='Engineered features with PCA and polynomial terms')

# Future you will thank present you
```

### 3. Track Lineage

```python
# Always specify inputs
folio.add('features', feature_df,
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
    path='s3://bucket/huge_data.parquet')

# Load directly from source when needed
df = pd.read_parquet(folio.data.training_data.path)
```

### 7. Leverage Autocomplete

```python
# This is more discoverable
config = folio.data.config.content
model = folio.data.classifier.content

# Than this
config = folio.get('config')
model = folio.get('classifier')
```

### 8. Move and share with ordinary tools

```bash
# Use the storage tool that already fits your workflow
gsutil -m rsync -r analysis/my_experiment gs://team-analysis/my_experiment
```

The recipient does not need DataFolio to inspect `CONTENTS.md`, read
`items.json`, or open the ordinary payload files. External references remain
absolute and continue to point at their original locations.

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
    folio.add('data', data)

    # Train
    model = train_model(data, config)
    folio.add_model('model', model)

    # Evaluate
    metrics = evaluate_model(model, data)
    folio.add('metrics', metrics)
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

**Q: What is DataFolio actually for?**

A: It removes repetitive, format-specific save/load code while keeping a small
collection of related objects and their descriptions together. It is a
directory, a readable catalog, and a dispatcher—not a general experiment
tracking platform.

**Q: Can someone use a folio without installing DataFolio?**

A: Yes. Included data uses ordinary formats, `CONTENTS.md` provides a readable
inventory, and `items.json` records the loading information. External
references may still require access to their original locations.

**Q: Does it work with cloud storage?**

A: Yes! DataFolio supports any storage backend via `cloud-files` (S3, GCS, Azure, etc.). Just use cloud paths:

```python
folio = DataFolio('s3://my-bucket/experiments/exp1')
```

**Q: How do I share a folio with colleagues?**

A: Copy or sync the directory to shared storage. Included payloads and snapshot
versions move with it; absolute external references do not move and may not be
accessible to the recipient.

**Q: What about versioning?**

A: Use [Snapshots](snapshots.md)! They checkpoint your owned data without duplicating it (external references preserve the recorded link, not the bytes behind it).

**Q: Can I use this with Jupyter notebooks?**

A: Absolutely. Notebooks are a primary use case. Multiple readers may open the
same folio; writes follow the documented one-active-writer model.

## Need Help?

- **Documentation**: Check the [full docs](../index.md)
- **Issues**: Report bugs on [GitHub](https://github.com/caseysm/datafolio/issues)
- **Examples**: See the repository for example notebooks
