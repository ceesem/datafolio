# datafolio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-112%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-83%25-green.svg)](tests/)

**Lightweight metadata tracking for exploratory data analysis workflows.**

`datafolio` helps you bundle analysis artifacts (metadata, models, plots) with references to large datasets, making it easy to track provenance and reproduce experiments without copying huge files.

## The Problem

Research workflows involve:
- 🗂️ Multiple linked dataframes (some large in datalakes, some small intermediate results)
- 🤖 Trained models (sklearn, xgboost, etc.)
- 📊 Metadata (parameters, timestamps, data versions)
- 📈 Artifacts (plots, configs, reports)

**But:** pandas doesn't handle metadata well (`.attrs` is fragile), and we need to package related artifacts together without copying large datasets.

## The Solution

`datafolio` creates lightweight bundles (tarfiles) that contain:
- ✅ **Metadata** - Analysis parameters, timestamps, arbitrary key-value data
- ✅ **Data references** - Paths to external data (S3, GCS, local) - *not copied*
- ✅ **Included data** - Small dataframes/results *are* copied into bundle
- ✅ **Models** - Trained models serialized with joblib
- ✅ **Artifacts** - Plots, configs, any files

**Key features:**
- 🪶 **Lightweight** - Large datasets referenced by path, not copied
- ⚡ **Lazy loading** - Files extracted only when accessed
- ☁️ **Cloud-native** - Works with S3, GCS, Azure
- 🔗 **Unified API** - Same interface for included and referenced data
- 🐍 **Simple** - Feels like saving/loading dataframes

## Installation

```bash
# Basic installation
pip install datafolio

# With cloud storage support (S3, GCS, Azure)
pip install datafolio[cloud]

# With Delta Lake support
pip install datafolio[delta]

# Everything
pip install datafolio[all]
```

Or with `uv`:
```bash
uv pip install datafolio[all]
```

## Quick Start

```python
from datafolio import DataFolio
import pandas as pd

# Create a bundle
folio = DataFolio(metadata={
    'experiment': 'protein_analysis_2024',
    'model_version': 'v2.1',
    'parameters': {'learning_rate': 0.01}
})

# Reference large dataset (path only, not copied!)
folio.reference_table(
    'training_data',
    path='s3://my-datalake/features/v3.parquet',
    table_format='parquet',
    num_rows=10_000_000
)

# Include small results
results_df = pd.DataFrame({'metric': ['accuracy'], 'value': [0.95]})
folio.add_table('results', results_df)

# Include model
folio.add_model('classifier', trained_model)

# Include artifacts
folio.add_artifact('loss_curve', 'plots/training.png', category='plots')

# Save (local or cloud)
folio.save('analysis.tar.gz')
# or: folio.save('s3://my-bucket/analysis.tar.gz')

# Load later
loaded = DataFolio.load('analysis.tar.gz')
results = loaded.get_table('results')        # From bundle
training = loaded.get_table('training_data') # From S3!
model = loaded.get_model('classifier')
```

## Core Concepts

### Two Modes for Data

**1. Reference (path only, not copied)** - For large datasets
```python
folio.reference_table(
    'big_data',
    path='s3://datalake/data.parquet',
    table_format='parquet',
    num_rows=1_000_000
)
```

**2. Include (copied into bundle)** - For small results
```python
results_df = pd.DataFrame({'metric': ['acc'], 'value': [0.95]})
folio.add_table('results', results_df)
```

**Unified retrieval:**
```python
# Same API for both!
df1 = folio.get_table('big_data')  # Reads from S3
df2 = folio.get_table('results')   # Extracts from bundle
```

### Bundle Structure

```
analysis.tar.gz (2-3 KB)
├── metadata/
│   ├── metadata.json          # Your analysis metadata
│   ├── data_references.json   # External data paths
│   ├── included_data.json     # Manifest of included tables
│   └── included_items.json    # Models & artifacts manifest
├── tables/
│   └── results.parquet        # Small included dataframes
├── models/
│   └── classifier.joblib      # Trained models
└── artifacts/
    └── loss_curve.png         # Plots, configs, etc.
```

## Usage Examples

### Basic Workflow

```python
from datafolio import DataFolio
import pandas as pd

# Create and populate
folio = DataFolio(metadata={'experiment': 'exp_001'})
df = pd.DataFrame({'a': [1, 2, 3]})
folio.add_table('results', df)
folio.save('experiment.tar.gz')

# Load and use
loaded = DataFolio.load('experiment.tar.gz')
print(loaded.metadata)
df = loaded.get_table('results')
```

### Mixed Local and Cloud Data

```python
folio = DataFolio(metadata={'date': '2024-01-15'})

# Reference cloud datasets (not copied)
folio.reference_table('features', path='s3://lake/features.parquet')
folio.reference_table('labels', path='s3://lake/labels.parquet')

# Include small outputs
summary = pd.DataFrame({'stat': ['mean', 'std'], 'value': [42.5, 12.3]})
folio.add_table('summary', summary)

# Save bundle (small - just summary and metadata)
folio.save('experiment.tar.gz')

# Load and access everything
loaded = DataFolio.load('experiment.tar.gz')
features = loaded.get_table('features')  # Reads from S3
summary = loaded.get_table('summary')     # From bundle
```

### Complete Analysis Bundle

```python
from datafolio import DataFolio
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Train a model
X_train = pd.read_parquet('s3://data/train.parquet')
y_train = pd.read_csv('s3://data/labels.csv')
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create bundle
folio = DataFolio(metadata={
    'experiment': 'rf_classification',
    'timestamp': '2024-01-15T10:30:00',
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 10
    }
})

# Reference training data
folio.reference_table(
    'training_data',
    path='s3://data/train.parquet',
    table_format='parquet',
    num_rows=len(X_train)
)

# Include predictions
predictions = model.predict(X_test)
pred_df = pd.DataFrame({'pred': predictions, 'true': y_test})
folio.add_table('predictions', pred_df)

# Include model
folio.add_model('rf_model', model)

# Include plots
folio.add_artifact('confusion_matrix', 'plots/cm.png', category='plots')
folio.add_artifact('config', 'config.yaml', category='configs')

# Save to cloud
folio.save('s3://my-bucket/experiments/exp_001.tar.gz')

# Later: Reproduce
loaded = DataFolio.load('s3://my-bucket/experiments/exp_001.tar.gz')
print(loaded.metadata['hyperparameters'])
model = loaded.get_model('rf_model')
training_data = loaded.get_table('training_data')
```

### Cross-Notebook Sharing

**Notebook 1: Training**
```python
from datafolio import DataFolio

folio = DataFolio(metadata={'stage': 'training'})
folio.reference_table('data', path='s3://lake/data.parquet')
folio.add_model('model', trained_model)
folio.save('models/model_v1.tar.gz')
```

**Notebook 2: Evaluation**
```python
from datafolio import DataFolio

# Load the bundle from training
folio = DataFolio.load('models/model_v1.tar.gz')

# Access everything
model = folio.get_model('model')
data = folio.get_table('data')

# Evaluate
results = evaluate(model, data)

# Create new bundle with results
eval_folio = DataFolio(metadata={
    'stage': 'evaluation',
    'parent': folio.metadata
})
eval_folio.add_table('eval_results', results)
eval_folio.save('models/eval_v1.tar.gz')
```

## API Reference

### DataFolio

**Constructor:**
```python
DataFolio(metadata: Optional[Dict[str, Any]] = None)
```

**Data Methods:**
- `reference_table(name, path, table_format='parquet', ...)` - Reference external data
- `add_table(name, data, description=None)` - Include DataFrame in bundle
- `get_table(name)` - Retrieve table (works for both types)
- `get_data_path(name)` - Get path to referenced data

**Model Methods:**
- `add_model(name, model, description=None)` - Include trained model
- `get_model(name)` - Load model from bundle

**Artifact Methods:**
- `add_artifact(name, path, category=None, description=None)` - Include file
- `get_artifact_path(name)` - Get path to artifact

**Bundle Methods:**
- `save(path, compression='gz')` - Save bundle (local or cloud)
- `load(path, lazy=True)` - Load bundle (classmethod)
- `list_contents()` - Show what's in the bundle

### Supported Formats

**Data formats:**
- Parquet (via pyarrow)
- CSV (via pandas)
- Delta Lake (optional, via deltalake)

**Storage backends:**
- Local filesystem
- Amazon S3 (`s3://`)
- Google Cloud Storage (`gs://`)
- Azure Blob Storage (`az://`)

## Cloud Storage

See [CLOUD_STORAGE.md](CLOUD_STORAGE.md) for detailed cloud storage documentation.

**Quick cloud example:**
```python
from datafolio import DataFolio

# Save to S3
folio = DataFolio(metadata={'exp': 'test'})
folio.save('s3://my-bucket/experiments/test.tar.gz')

# Load from S3
loaded = DataFolio.load('s3://my-bucket/experiments/test.tar.gz')
```

**Requirements:**
```bash
pip install datafolio[cloud]
```

## Best Practices

### ✅ Do's

1. **Reference large datasets, include small results**
   ```python
   # Good
   folio.reference_table('features', path='s3://data/features.parquet')  # 10GB
   folio.add_table('metrics', metrics_df)  # 10 rows
   ```

2. **Store rich metadata**
   ```python
   folio = DataFolio(metadata={
       'experiment_id': 'exp_001',
       'timestamp': datetime.now().isoformat(),
       'git_commit': get_git_commit(),
       'parameters': {...},
       'data_versions': {...}
   })
   ```

3. **Use method chaining**
   ```python
   folio.reference_table('data1', path='s3://...') \
        .add_table('results', df) \
        .add_model('model', model)
   ```

4. **Organize cloud storage**
   ```python
   # Use structured paths
   path = f's3://bucket/{project}/{experiment_id}/bundle.tar.gz'
   ```

### ❌ Don'ts

1. **Don't include large datasets in bundles**
   ```python
   # Bad - bundle will be huge!
   huge_df = pd.read_parquet('10GB_file.parquet')
   folio.add_table('data', huge_df)  # ❌ Don't do this!

   # Good - just store the path
   folio.reference_table('data', path='10GB_file.parquet')  # ✅
   ```

2. **Don't lose references**
   ```python
   # Bad - referenced file might move/delete
   folio.reference_table('data', path='/tmp/data.parquet')  # ❌

   # Good - use stable storage
   folio.reference_table('data', path='s3://bucket/data.parquet')  # ✅
   ```

3. **Don't skip metadata**
   ```python
   # Bad - no context
   folio = DataFolio()  # ❌

   # Good - rich metadata
   folio = DataFolio(metadata={...})  # ✅
   ```

## Development

```bash
# Clone the repo
git clone https://github.com/your-org/datafolio.git
cd datafolio

# Install with dev dependencies
uv sync

# Run tests
poe test

# Run tests with coverage
poe test

# Preview documentation
poe doc-preview

# Lint
uv run ruff check src/ tests/

# Bump version
poe bump patch  # or minor, major
```

## Testing

```bash
# Run full test suite
poe test

# Run specific test file
uv run pytest tests/test_readers.py -v

# Run with coverage report
uv run pytest --cov=datafolio --cov-report=html tests/
```

**Test coverage: 83%** (112 tests passing)

## Requirements

- Python 3.10+
- pandas >= 2.0.0
- pyarrow >= 14.0.0
- joblib >= 1.3.0
- orjson >= 3.9.0

**Optional:**
- cloud-files >= 4.0.0 (for cloud storage)
- deltalake >= 0.15.0 (for Delta Lake support)

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

## Related Projects

- **pandas** - Data manipulation
- **joblib** - Model serialization
- **cloud-files** - Cloud storage I/O
- **DVC** - Data version control (heavier, git-based)
- **MLflow** - ML experiment tracking (heavier, server-based)

**datafolio is lighter-weight:** No servers, no databases, just tarfiles with metadata.

## FAQ

**Q: Why not just use pickle?**
A: Pickle is fragile across Python versions and doesn't handle metadata or large dataset references well.

**Q: Why not MLflow?**
A: MLflow requires a server and is heavier. datafolio is just tarfiles - no infrastructure needed.

**Q: Can I version control the bundles?**
A: Small bundles (< 100MB) can be versioned with git. Large bundles should go in cloud storage.

**Q: What about data lineage?**
A: Store lineage info in metadata:
```python
folio = DataFolio(metadata={
    'derived_from': previous_folio.metadata,
    'transformations': ['filter', 'aggregate']
})
```

**Q: Does it work with Polars/Dask?**
A: Currently pandas only. Convert to pandas before adding to bundle.

---

Made with ❤️ for data scientists who need simple, lightweight experiment tracking.