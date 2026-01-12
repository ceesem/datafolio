# DataFolio

**A lightweight, filesystem-based data versioning and experiment tracking library for Python.**

DataFolio helps you organize, version, and track your data science experiments by storing datasets, models, and artifacts in a simple, transparent directory structure. Everything is saved as plain files (Parquet, JSON, pickle) that you can inspect, version with git, or backup to any storage system.

## Why DataFolio?

Ever trained a model with great results, then lost it while experimenting? Or struggled to remember which dataset produced which model? Or needed to reproduce results from months ago?

DataFolio solves these problems with a simple, filesystem-based approach—no servers, no databases, just files you can inspect and version control.

## Quick Example: The Story of a Good Model

```python
from datafolio import DataFolio
from sklearn.ensemble import RandomForestClassifier

# You've been working on a classification problem
folio = DataFolio('experiments/fraud_detection')

# Process your data
folio.add_table('training_data', processed_df,
    description='Cleaned transaction data with engineered features')

# Train a model - it gets 89% accuracy! 🎉
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Save it with metadata
folio.add_model('classifier', model,
    description='Random forest classifier',
    inputs=['training_data'])
folio.metadata['accuracy'] = 0.89
folio.metadata['status'] = 'promising'

# View everything
folio.describe()

# Create a snapshot before experimenting - it's free insurance!
folio.create_snapshot('v1-baseline',
    description='89% accuracy baseline model',
    tags=['baseline', 'validated'])

# Now experiment freely - try a neural network
new_model = train_experimental_model()
folio.add_model('classifier', new_model, overwrite=True)
folio.metadata['accuracy'] = 0.85  # Worse! 😞

# No problem - load the good version back
baseline = DataFolio.load_snapshot('experiments/fraud_detection', 'v1-baseline')
good_model = baseline.get_model('classifier')  # Your 89% model is safe!

# Deploy the good one to production
deploy_to_production(good_model)
```

This is the core DataFolio workflow: track your data, models, and results; snapshot before experimenting; never lose good work.

## Key Features

- **Universal Data Management** - Single `add_data()` method handles DataFrames, numpy arrays, dicts, lists, and scalars
- **Model Support** - Save and load scikit-learn and PyTorch models with full metadata
- **Snapshots** - Create immutable checkpoints of experiments with copy-on-write versioning (no data duplication!)
- **Data Lineage** - Track inputs and dependencies between datasets and models
- **Autocomplete Access** - IDE-friendly `folio.data.item_name.content` syntax with full autocomplete
- **Multi-Instance Sync** - Multiple notebooks/processes can safely access the same bundle
- **Cloud Storage** - Works with local paths, S3, GCS, Azure, and more
- **Caching** - Smart caching for remote data reduces download times
- **Git-Friendly** - All data stored as standard file formats in a simple directory structure
- **CLI Tools** - Command-line interface for snapshot management and bundle operations

## Installation

```bash
pip install datafolio
```

## Learn More

**New to DataFolio?** Start with the [Getting Started Guide](guides/getting-started.md) for a comprehensive tutorial.

**Specific Topics:**
- [Snapshots Guide](guides/snapshots.md) - Version control for experiments
- [DataFolio API Reference](reference/datafolio-api.md) - All methods and properties
- [CLI Reference](reference/cli.md) - Command-line tools
- [Complete API](reference/api.md) - Full API documentation

## Common Use Cases

### Experiment Tracking

```python
# Track everything about your experiment
folio = DataFolio('experiments/model_v2')
folio.metadata['experiment'] = 'hyperparameter_tuning'
folio.metadata['date'] = '2025-01-20'

# Save data, models, and results
folio.add_table('features', feature_df)
folio.add_model('model', trained_model)
folio.add_json('metrics', {'accuracy': 0.92, 'f1': 0.89})

# Create snapshot at milestones
folio.create_snapshot('v2.0-production', tags=['production'])
```

### Reproducible Research

```python
# Paper submission: snapshot your exact results
folio.create_snapshot('neurips-2025-submission',
    description='Results in paper Table 3',
    tags=['paper', 'published'])

# Six months later: reviewers ask for clarification
paper_version = DataFolio.load_snapshot('research/exp', 'neurips-2025-submission')
exact_model = paper_version.get_model('classifier')
exact_data = paper_version.get_table('test_data')
```

### Team Collaboration

```python
# Use cloud storage for team access
folio = DataFolio('s3://team-bucket/shared-experiment',
    cache_enabled=True)  # Cache for faster local access

# Everyone sees the same data
df = folio.get_table('results')
model = folio.get_model('classifier')

# Compare different team members' approaches
baseline = DataFolio.load_snapshot('s3://team-bucket/shared', 'alice-baseline')
variant = DataFolio.load_snapshot('s3://team-bucket/shared', 'bob-neural-net')
```

## What Makes DataFolio Different?

| | DataFolio | MLflow | Weights & Biases |
|---|---|---|---|
| **Setup** | Zero - just a directory | Requires server | Requires account |
| **Storage** | Files on disk/cloud | Database + artifacts | Cloud service |
| **Inspection** | Direct file access | Via API | Via web UI |
| **Versioning** | Snapshots (copy-on-write) | Runs (separate copies) | Versions (cloud) |
| **Sharing** | Copy directory/git | Share server access | Share workspace |
| **Cost** | Free | Free (self-hosted) | Free tier + paid |

DataFolio is perfect when you want:
- Full control over your data
- Simple filesystem-based storage
- Git-friendly versioning
- No external dependencies
- Cloud storage without cloud services

## Directory Structure

DataFolio creates an intuitive, inspectable directory structure:

```
experiments/my_experiment/
├── items.json                # Manifest of all items
├── metadata.json             # Bundle metadata
├── snapshots.json            # Snapshot registry
│
├── tables/
│   └── features.parquet      # DataFrames as Parquet
│
├── models/
│   └── classifier.joblib     # Scikit-learn models
│
├── pytorch_models/
│   ├── neural_net.pth        # PyTorch state dicts
│   └── neural_net_class.pkl  # PyTorch class definitions
│
├── numpy/
│   └── embeddings.npy        # Numpy arrays
│
└── artifacts/
    ├── config.json           # JSON data
    ├── plot.png              # Images
    └── report.pdf            # Any file type
```

All files use standard formats you can open with any tool!

## Quick CLI Reference

```bash
# Initialize a new bundle
datafolio init my_experiment

# Describe bundle contents
datafolio describe

# Create a snapshot
datafolio snapshot create v1.0 -d "Baseline model" --tags baseline,production

# List snapshots
datafolio snapshot list

# Compare two snapshots
datafolio snapshot compare v1.0 v2.0

# Show current status vs last snapshot
datafolio snapshot status
```

See the [CLI Reference](reference/cli.md) for complete documentation.

## Best Practices

1. **Use descriptive names** - `'training_features'` not `'data1'`
2. **Track lineage** - Always specify `inputs` parameter
3. **Add descriptions** - Help future you understand your work
4. **Snapshot before major changes** - It's free insurance
5. **Use tags** - Organize snapshots with `baseline`, `production`, `paper`
6. **Leverage autocomplete** - Use `folio.data.item_name.content`
7. **Clean up regularly** - Delete temporary items with `folio.delete()`
8. **Version control** - Commit bundles to git for team collaboration

## Get Started

Ready to organize your experiments? Check out the [Getting Started Guide](guides/getting-started.md) for a step-by-step tutorial.

## Development

See [CLAUDE.md](https://github.com/ceesem/datafolio/blob/main/CLAUDE.md) for development guidelines.

## License

MIT License - see LICENSE file for details.
