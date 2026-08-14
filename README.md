# DataFolio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-912%20passing-brightgreen.svg)](tests/)

**A small, human-readable home for the data associated with an analysis.**

DataFolio saves common Python objects in ordinary formats, records what they
are in one readable catalog, and lets you reopen the whole collection with one
path. It is useful when you are tired of wiring a different writer, loader,
filename, and cloud path for every object—and when you want future you to know
which of several similar files is which.

A folio is intentionally just three things:

1. **A directory** you can inspect, copy, upload, or share.
2. **One `items.json` catalog** containing descriptions and loading information.
3. **A dispatcher** connecting `add()` and `get()` to sensible ordinary formats.

DataFolio organizes and links your data, then gets out of the way. Querying,
transforming, and scaling remain pandas, Polars, and PyArrow's job. Moving a
directory remains your filesystem or object-store tool's job.

## What it does

- Saves DataFrames, arrays, JSON values, models, and files through a small,
  consistent API
- Loads each object by name with the appropriate reader
- Keeps descriptions, relationships, and external references beside the data
- Gives notebooks one folio path instead of a collection of unrelated paths
- Produces a human-readable inventory for people who do not use DataFolio
- Records snapshots of folio-owned state when useful
- Works with local directories and supported object storage

## Intentional limits

DataFolio is not a database, dataframe engine, workflow orchestrator,
distributed catalog, garbage collector, or multi-writer collaboration system.
It is designed for one person—or a small team sharing mostly read-only work—
managing a few to dozens of understandable objects.

## Quick Start

```python
from datafolio import DataFolio
import pandas as pd
import numpy as np

# Create a new folio
folio = DataFolio('experiments/my_experiment')

# Add ordinary objects and record what they mean
folio.add('results', df,
    description='Reviewed results used in the final analysis')
folio.add('embeddings', np.array([1, 2, 3]),
    description='Three-dimensional demonstration embedding')
folio.add('config', {'lr': 0.01},
    description='Parameters used to produce results')
folio.add('accuracy', 0.95)                       # Scalar

# Retrieve data (automatically returns correct type)
df = folio.get('results')           # Returns DataFrame
arr = folio.get('embeddings')       # Returns numpy array
config = folio.get('config')        # Returns dict

# Or use autocomplete-friendly access
df = folio.data.results.content          # Same as get()
arr = folio.data.embeddings.content
config = folio.data.config.content

# View names, descriptions, formats, and relationships
folio.describe()

# Clean up items you no longer need
folio.delete('accuracy')
```

## Installation

```bash
pip install datafolio
```

This includes the `datafolio` command-line tool for inspecting folios and
managing snapshots.

## Core Concepts

### Generic Data Methods

The `add()` and `get()` methods provide a unified interface for all data types:

```python
# add() automatically detects type and uses the appropriate handler
folio.add('my_data', data)  # Works with DataFrame, array, dict, list, scalar

# get() automatically detects stored type and returns correct format
data = folio.get('my_data')  # Returns original type

# in tests membership by name, mirroring get()
if 'my_data' not in folio:
    folio.add('my_data', data)
```

Supported data types:

- **DataFrames** (`pd.DataFrame`, `pl.DataFrame`, `pl.LazyFrame`) → stored as Parquet
- **Numpy arrays** (`np.ndarray`) → stored as `.npy`
- **JSON data** (`dict`, `list`, `int`, `float`, `str`, `bool`, `None`) → stored as JSON (strings are always stored as JSON, never treated as file paths)
- **Timezone-aware datetimes** (`datetime`) → stored as timestamps. For Unix numbers, convert first: `folio.add('run_at', datetime.fromtimestamp(x, tz=timezone.utc))` (a bare number like `1705318200` is stored as JSON)
- **Sklearn-family estimators** (scikit-learn, XGBoost, LightGBM, CatBoost) → auto-detected; use `add_model()` for any other picklable object
- **External references** → use `reference_table(name, path='s3://...')`; metadata only, data stays in original location

Replacing an existing item always requires `overwrite=True` — snapshotted versions are preserved via copy-on-write.

### Multi-Instance Access

DataFolio automatically refreshes readers that access the same folio:

```python
# Notebook 1: Create and update a folio
folio1 = DataFolio('experiments/shared')
folio1.add('results', df)

# Notebook 2: Open the same folio
folio2 = DataFolio('experiments/shared')

# Notebook 1: Add more data
folio1.add('analysis', new_df)

# Notebook 2: Automatically sees new data!
folio2.describe()  # Shows both 'results' and 'analysis'
analysis = folio2.get('analysis')  # Works immediately ✅
```

All read operations (`describe()`, `list_contents()`, `get()`/`get_model()`, and `folio.data` accessors) automatically refresh from disk when changes are detected. Datafolio supports many readers and one active writer. Local writers are serialized; a stale writer fails with `ConcurrentWriteError` and must `refresh()` and retry. Cloud folios should be treated as single-writer.

Cloud reads are round-trip bound. `pinned()` checks freshness once and suspends the per-read recheck for the block — other writers' changes then aren't visible until it exits — and `get_many()` additionally overlaps the reads themselves:

```python
with folio.pinned():
    for name in folio.tables():
        process(folio.get(name))     # one staleness check, not one per read

tables = folio.get_many(folio.tables())   # ...and read them concurrently
```

Measured on 60 items at a 250ms round trip: 45.9s naive → 15.6s pinned → 1.1s with `get_many`.

### Data Lineage

Track dependencies between datasets and models:

```python
# Create dependency chain
folio.reference_table('raw', path='s3://bucket/raw.parquet')
folio.add('clean', cleaned_df, inputs=['raw'])
folio.add('features', feature_df, inputs=['clean'])
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
path = folio.data.results.path           # Payload path (works for every type)

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
├── items.json                 # The single manifest: metadata, item catalog,
│                              # snapshots, and revision, in one JSON document
├── tables/
│   └── results--r2.parquet   # DataFrame storage (versioned filenames)
├── models/
│   └── classifier--r3.joblib # Model storage
└── artifacts/
    ├── embeddings--r1.npy    # Numpy arrays
    ├── config--r4.json       # JSON data
    └── plot--r1.png          # Any file type
```

## Polars, Lazy Scans & External References

Parquet is the canonical table format. Tables come back as **pandas** by default
(unchanged), but you can also work with **polars** — eager or lazy:

```python
import polars as pl

folio.add('t', df)                             # pandas or polars DataFrame
folio.add('big', lazyframe)                    # LazyFrame → streamed to parquet (bounded memory)

folio.get('t')                                 # pandas DataFrame (default)
folio.get('t', frame='polars')                 # eager polars DataFrame
folio.scan_table('t')                          # genuinely lazy pl.LazyFrame
folio.scan_table('t').filter(pl.col('a') > 0).select('b').collect()  # pushdown
```

**External references** link to data you don't copy into the bundle. Creating a
reference is a cheap, offline manifest write — it performs **no** network I/O:

```python
folio.reference_table('raw', path='s3://bucket/huge.parquet')  # no download, no stat
folio.scan_table('raw')            # lazy scan over the external parquet (pushdown, no full read)
folio.inspect_table('raw')         # opt-in: reads schema/size/identity now
```

Sharded / hive-partitioned directory references are read lazily via polars; a
plain pandas `get` on one raises a clear "polars-only" error. `scan_table`
is genuinely lazy (local, `s3://`, `gs://`, `az://`, `http(s)://`) and raises
rather than silently downloading a scheme it can't scan lazily. Delta/Iceberg
are not supported — convert to Parquet first.

> **Snapshots & references:** snapshots freeze *owned* items (included tables,
> models, files). A referenced table's external bytes are **not** owned and
> may change; snapshots preserve the link, not the content. See
> `folio.mutable_references()` and `folio.inspect_table()`.

## Snapshots: Remembered Folio States

Snapshots record which owned files and catalog information constituted the
folio at a useful moment. They let you compare or reopen earlier states while
sharing unchanged payloads.

> **What a snapshot preserves:** folio-owned data (included tables, models,
> files, ...) and the *recorded state* of external references. It does **not**
> preserve or guarantee the contents of referenced data — datafolio never copies
> or freezes external bytes, so the data behind a `reference_table` can change
> after a snapshot is taken.

### Why Snapshots?

**The Problem**: You train a model with 89% accuracy, then experiment with improvements. The new version gets 85%—worse! But you've already overwritten your good model. You need to recreate it from git history.

**The Solution**: Create snapshots before experimenting. Snapshots preserve exact states while sharing unchanged data to save disk space.

### Quick Start with Snapshots

```python
from datafolio import DataFolio

# Create your experiment
folio = DataFolio('experiments/classifier')
folio.add('train_data', train_df)
folio.add_model('model', baseline_model)
folio.metadata['accuracy'] = 0.89

# Create a snapshot before experimenting
folio.create_snapshot('v1.0-baseline',
    description='Baseline random forest model',
    tags=['baseline', 'production'])

# Experiment freely - the snapshot is preserved
folio.add_model('model', experimental_model, overwrite=True)
folio.metadata['accuracy'] = 0.85  # Worse!

# Load the original version
baseline = DataFolio.load_snapshot('experiments/classifier', 'v1.0-baseline')
model = baseline.get_model('model')  # Original model with 89% accuracy!
```

### CLI for Snapshot Management

DataFolio includes a command-line tool for easy snapshot operations:

```bash
# Create a snapshot
datafolio snapshot create v1.0 -d "Baseline model" -t baseline

# List all snapshots
datafolio snapshot list

# Show snapshot details
datafolio snapshot show v1.0

# Compare two snapshots
datafolio snapshot compare v1.0 v2.0

# Delete old snapshots and cleanup
datafolio snapshot delete experimental-v5 --cleanup
```

### Key Features

- **Owned data frozen**: A snapshot's owned items never change (external references preserve the link, not the bytes)
- **Space-efficient**: Uses copy-on-write versioning—only changed items create new files
- **Git integration**: Automatically captures commit hash, branch, and dirty status
- **Environment tracking**: Optionally records the Python version, platform, and uv.lock hash (never full dependency contents — git owns those)
- **Metadata preservation**: Snapshots include complete metadata state at that moment
- **Multiple snapshots**: Load different versions simultaneously for comparison

### Use Cases

**Paper Submission**: Snapshot your owned data, models, and recorded reference descriptors when submitting (plus the git commit of your code). Months later, you can reload exactly what the folio owned — referenced external data is linked, not guaranteed.

**A/B Testing**: Create snapshots for baseline and experimental versions, deploy both, and compare performance metrics.

**Hyperparameter Tuning**: Snapshot each configuration, then compare results to find the best settings.

**Production Deployment**: Tag production-ready snapshots and deploy specific versions with confidence.

For complete snapshot documentation, see [docs/guides/snapshots.md](docs/guides/snapshots.md).

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
folio.reference_table('raw', path='s3://bucket/raw.parquet',
    description='Raw training data from database')

# Add processed data
folio.add('clean', cleaned_df,
    description='Cleaned and preprocessed data',
    inputs=['raw'])

# Add features
folio.add('features', feature_df,
    description='Engineered features',
    inputs=['clean'])

# Train and save model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

folio.add_model('classifier', clf,
    description='Random forest classifier',
    inputs=['features'])

# Save metrics
folio.add('metrics', {
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

## Best Practices

1. **Use descriptive names**: `add('training_features', ...)` not `add('data1', ...)`
2. **Track lineage**: Always specify `inputs` to track data dependencies
3. **Add descriptions**: Help future you understand what each item contains
4. **Use custom metadata**: Store experiment context in `folio.metadata` for better tracking
5. **Leverage autocomplete**: Use `folio.data.item_name.content` for cleaner, more discoverable code
6. **Clean up regularly**: Use `delete()` to remove temporary or obsolete items
7. **Use ordinary transfer tools**: Copy or sync folios with the filesystem or object-store tool that already fits your workflow
8. **Use references**: For large external datasets, use `reference_table()` to avoid copying
9. **Check describe()**: Regularly review your folio with `folio.describe()` to see data and metadata
10. **Share across notebooks**: many readers, one active writer — readers auto-refresh; a second writer fails safely with `ConcurrentWriteError` and should `refresh()` and retry
11. **Snapshot before major changes**: Create snapshots before experimenting with new approaches—it's free insurance
12. **Tag snapshots meaningfully**: Use tags like `baseline`, `production`, `paper` to organize versions

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
- skops >= 0.10.0
- orjson >= 3.9.0
- cloud-files >= 5.8.1
- click >= 8.1.0 (for CLI)
- rich >= 13.0.0 (for CLI formatting)
- filelock >= 3.12.0

Optional extras:

- polars >= 1.0.0 (`pip install datafolio[polars]`)

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

Built for people who want less save/load wiring and more understandable data directories.
