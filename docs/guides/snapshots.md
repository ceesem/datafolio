# Snapshots: Version Control for Experiments

Snapshots provide immutable checkpoints of your DataFolio bundles, allowing you to version experiments, compare results, and maintain reproducibility without duplicating data.

## Why Use Snapshots?

### The Problem

Imagine this scenario:

1. You train a model that achieves 89% accuracy—great results!
2. You experiment with a new approach, overwriting the model
3. The new version gets 85%—worse!
4. You want to go back, but the good model is gone

Without snapshots, you'd need to either:

- Duplicate entire experiment directories (wasting disk space)
- Manually track versions in separate folders
- Try to recreate the model from git history

### The Solution

Snapshots let you create immutable checkpoints before experimenting:

```python
from datafolio import DataFolio

# Create your experiment
folio = DataFolio('experiments/classifier')
folio.add_model('model', baseline_model)
folio.metadata['accuracy'] = 0.89

# Snapshot before experimenting
folio.create_snapshot('v1.0-baseline',
    description='Baseline random forest - 89% accuracy',
    tags=['baseline', 'production'])

# Experiment freely!
folio.add_model('model', experimental_model, overwrite=True)
folio.metadata['accuracy'] = 0.85  # Worse, but that's OK

# Return to the good version anytime
baseline = DataFolio.load_snapshot('experiments/classifier', 'v1.0-baseline')
good_model = baseline.get_model('model')  # Original 89% model
```

## Key Benefits

**🔒 Immutable** - Once created, snapshots never change
**💾 Space-efficient** - Only changed items create new files (copy-on-write)
**🔄 Git integration** - Automatically captures commit hash and status
**📝 Metadata preservation** - Complete experiment state at that moment
**🔍 Comparison tools** - Built-in diff and comparison functions
**⚡ Fast** - No data copying for unchanged items

## Creating Snapshots

### Basic Snapshot

```python
folio = DataFolio('experiments/my_experiment')

# Add your data
folio.add_table('train_data', df)
folio.add_model('model', trained_model)
folio.metadata['accuracy'] = 0.92

# Create snapshot
folio.create_snapshot('v1.0')
```

### Snapshot with Metadata

```python
folio.create_snapshot(
    'v1.0-baseline',
    description='Baseline random forest model',
    tags=['baseline', 'production', 'paper'],
    capture_git=True,           # Capture git info (default: True)
    capture_environment=True,   # Capture Python env (default: True)
    capture_execution=True      # Capture execution info (default: True)
)
```

### What Gets Captured

When you create a snapshot, DataFolio automatically captures:

- **Item versions** - Current state of all data, models, and artifacts
- **Metadata** - Complete folio metadata
- **Git information** - Commit hash, branch, dirty status (if in a git repo)
- **Environment** - Python version, dependencies
- **Execution context** - Working directory, entry point
- **Timestamp** - When the snapshot was created

### Security: Credential Protection

DataFolio automatically protects against accidental credential leakage when capturing git information:

**Git Remote URLs are Sanitized**

When `capture_git=True` (the default), DataFolio captures your git remote URL for reproducibility, but **automatically removes any embedded credentials**:

```python
# Original git remote (potentially with token)
# https://ghp_token123@github.com/user/repo.git

# What gets stored in snapshot (credentials removed)
# https://github.com/user/repo.git
```

**What is Sanitized:**
- ✅ HTTPS URLs with tokens: `https://token@github.com/repo.git` → `https://github.com/repo.git`
- ✅ HTTPS URLs with username:password: `https://user:pass@gitlab.com/repo.git` → `https://gitlab.com/repo.git`
- ✅ SSH URLs preserved as-is (no credentials embedded): `git@github.com:user/repo.git`

**What is Preserved:**
- Repository location (host and path)
- Commit hash (essential for reproducibility)
- Branch name
- Dirty status (whether there are uncommitted changes - but not which files)

**What is NOT Captured (for security):**
- List of uncommitted files (could reveal sensitive filenames like `.env`, `secrets.yaml`)
- The dirty flag tells you *if* there were uncommitted changes, but not *what* they were

This ensures you can safely share snapshots with collaborators without worrying about exposing access tokens, passwords, or sensitive filenames.

**Best Practices:**
- Review snapshots before sharing: `datafolio snapshot show <name>`
- Avoid putting secrets in commit messages or metadata
- Use environment variables or config files for credentials (not command-line arguments)

## Loading Snapshots

There are three ways to access snapshots, depending on your needs:

### Method 1: Load Snapshot as DataFolio (Full Access)

Load any snapshot to get the exact state with full DataFolio functionality:

```python
# Using classmethod (when you don't have a folio instance yet)
snapshot = DataFolio.load_snapshot('experiments/classifier', 'v1.0-baseline')

# Using instance method (when you already have a folio)
folio = DataFolio('experiments/classifier')
snapshot = folio.get_snapshot('v1.0-baseline')  # Equivalent to above

# Access data exactly as it was
model = snapshot.get_model('model')
data = snapshot.get_table('train_data')
accuracy = snapshot.metadata['accuracy']

# Snapshot is read-only by default
print(snapshot.read_only)  # True
```

### Method 2: Quick View (Read Metadata)

For lightweight access to snapshot metadata without loading all data:

```python
folio = DataFolio('experiments/classifier')

# Access snapshot via accessor (returns SnapshotView)
view = folio.snapshots['v1.0-baseline']

# Quick metadata access
print(view.metadata)
print(view.name)
print(view.timestamp)

# Can also get data, but more limited than full DataFolio
data = view.get_table('train_data')
```

**When to use which:**
- Use `load_snapshot()` or `get_snapshot()` when you need full access to data and models
- Use `snapshots['name']` for quick metadata inspection or simple data access

### Method 3: Load Multiple Snapshots

You can load multiple snapshots at once to compare:

```python
# Using classmethod
v1 = DataFolio.load_snapshot('experiments/exp', 'v1.0')
v2 = DataFolio.load_snapshot('experiments/exp', 'v2.0')
v3 = DataFolio.load_snapshot('experiments/exp', 'v3.0')

# Or using instance method
folio = DataFolio('experiments/exp')
v1 = folio.get_snapshot('v1.0')
v2 = folio.get_snapshot('v2.0')
v3 = folio.get_snapshot('v3.0')

# Compare results
print(f"v1: {v1.metadata['accuracy']}")  # 0.89
print(f"v2: {v2.metadata['accuracy']}")  # 0.91
print(f"v3: {v3.metadata['accuracy']}")  # 0.87

# Deploy the best one
best_model = v2.get_model('classifier')
```

## Managing Snapshots

### List Snapshots

```python
# List all snapshots
snapshots = folio.list_snapshots()
for snap in snapshots:
    print(f"{snap['name']}: {snap['description']}")

# Filter by tags
production_snaps = folio.list_snapshots(tags=['production'])
```

### Get Snapshot Info

```python
# Get detailed information
info = folio.get_snapshot_info('v1.0')

print(info['description'])
print(info['timestamp'])
print(info['item_versions'])  # Which versions of items
print(info['metadata_snapshot'])  # Metadata state
print(info['git'])  # Git information
```

### Compare Snapshots

```python
# Compare two snapshots
diff = folio.compare_snapshots('v1.0', 'v2.0')

print("Added items:", diff['added_items'])
print("Removed items:", diff['removed_items'])
print("Modified items:", diff['modified_items'])
print("Metadata changes:", diff['metadata_changes'])
```

### Delete Snapshots

```python
# Delete a snapshot (keeps files unless orphaned)
folio.delete_snapshot('experimental-v5')

# Delete and cleanup orphaned versions
folio.delete_snapshot('experimental-v5', cleanup_orphans=True)
```

### Cleanup Orphaned Versions

Over time, old item versions not used by any snapshot can accumulate:

```python
# See what would be deleted
orphans = folio.cleanup_orphaned_versions(dry_run=True)
print(f"Would delete: {orphans}")

# Actually delete them
deleted = folio.cleanup_orphaned_versions()
print(f"Deleted {len(deleted)} orphaned versions")
```

## Copy-on-Write Versioning

Snapshots use copy-on-write versioning to save disk space:

```python
# Initial data (5GB file)
folio.add_table('big_data', huge_df)
folio.create_snapshot('v1.0')

# Create 10 more snapshots - still only 5GB!
folio.create_snapshot('v1.1')
folio.create_snapshot('v1.2')
# ... no new files created for 'big_data'

# Only when you overwrite an item in a snapshot do we create a new version
folio.add_table('big_data', modified_df, overwrite=True)
# Now we have: big_data.parquet (5GB) and big_data_v2.parquet (5GB)

folio.create_snapshot('v2.0')

# v1.x snapshots still reference original file
# v2.0 references the new file
```

## Reproduction Instructions

Get human-readable instructions for reproducing a snapshot:

```python
instructions = folio.reproduce_instructions('v1.0')
print(instructions)
```

Output:
```
To reproduce snapshot 'v1.0':

1. Restore code:
   git checkout abc123

2. Restore environment:
   Python version: 3.11.5
   uv sync

3. Load bundle:
   folio = DataFolio.load_snapshot('experiments/exp', 'v1.0')

4. Expected results:
   accuracy: 0.89
   f1_score: 0.87
```

## CLI Tools

DataFolio includes a command-line tool for snapshot management:

### Create Snapshots

```bash
# Basic creation
datafolio snapshot create v1.0 -d "Baseline model"

# With tags
datafolio snapshot create v1.0 \
  -d "Production model" \
  -t baseline -t production

# Skip git/env capture
datafolio snapshot create v1.0 --no-git --no-env
```

### List and Show

```bash
# List all snapshots
datafolio snapshot list

# Filter by tag
datafolio snapshot list --tag production

# Show details
datafolio snapshot show v1.0

# Show reproduction instructions
datafolio snapshot reproduce v1.0
```

### Compare and Manage

```bash
# Compare two snapshots
datafolio snapshot compare v1.0 v2.0

# Delete a snapshot
datafolio snapshot delete experimental-v5

# Cleanup orphaned versions
datafolio snapshot gc --dry-run  # See what would be deleted
datafolio snapshot gc            # Actually delete
```

### Bundle Path Options

```bash
# Work in current directory
cd experiments/my-experiment
datafolio snapshot list

# Specify bundle path
datafolio --bundle experiments/my-experiment snapshot list

# Use environment variable
export DATAFOLIO_BUNDLE=experiments/my-experiment
datafolio snapshot list
```

## Common Workflows

### Paper Submission

```python
# September: Finalize results for paper
folio = DataFolio('research/protein-analysis')
folio.add_table('data', processed_data)
folio.add_model('classifier', final_model)
folio.metadata['accuracy'] = 0.92

# Snapshot for paper
folio.create_snapshot(
    'neurips-2025-submission',
    description='Exact version submitted to NeurIPS 2025',
    tags=['paper', 'neurips', 'submitted']
)

# February: Reviewers ask for changes
# Load original version
paper_folio = DataFolio.load_snapshot(
    'research/protein-analysis',
    'neurips-2025-submission'
)

# Run additional experiments with original data/model
original_model = paper_folio.get_model('classifier')
original_data = paper_folio.get_table('data')
```

### A/B Testing

```python
# Deploy two versions for A/B test
baseline = DataFolio.load_snapshot('models/recommender', 'v2.0-baseline')
experimental = DataFolio.load_snapshot('models/recommender', 'v3.0-experimental')

# Deploy to different endpoints
deploy_model(baseline.get_model('model'), endpoint='prod-a')
deploy_model(experimental.get_model('model'), endpoint='prod-b')

# Compare results after test
print(f"Baseline p95: {baseline.metadata['p95_latency']}")
print(f"Experimental p95: {experimental.metadata['p95_latency']}")

# Winner! Create new baseline
folio = DataFolio('models/recommender')
folio.create_snapshot('v3.0-baseline',
    description='New production baseline',
    tags=['production', 'baseline'])
```

### Hyperparameter Tuning

```python
folio = DataFolio('experiments/tuning')

# Try different hyperparameters
for lr in [0.001, 0.01, 0.1]:
    for depth in [5, 10, 20]:
        model = train_model(lr=lr, max_depth=depth)
        accuracy = evaluate(model)

        folio.add_model('model', model, overwrite=True)
        folio.metadata['lr'] = lr
        folio.metadata['max_depth'] = depth
        folio.metadata['accuracy'] = accuracy

        # Snapshot each config
        folio.create_snapshot(f'lr{lr}_depth{depth}')

# Find best config
snapshots = folio.list_snapshots()
best = max(snapshots, key=lambda s: s['metadata_snapshot']['accuracy'])

# Load best model
best_folio = DataFolio.load_snapshot('experiments/tuning', best['name'])
production_model = best_folio.get_model('model')
```

## Best Practices

### When to Snapshot

✅ **Do snapshot:**
- Before major experiments
- After achieving good results
- Before paper submission
- Before deploying to production
- At important milestones

❌ **Don't snapshot:**
- After every tiny change
- During active development
- For temporary experiments

### Naming Conventions

Use semantic, descriptive names:

- **Semantic versioning**: `v1.0.0`, `v1.1.0`, `v2.0.0`
- **Date-based**: `2025-01-20-baseline`, `2025-02-15-production`
- **Milestone-based**: `paper-submission`, `production-v1`, `baseline`
- **Descriptive**: `random-forest-baseline`, `neural-net-experiment`

Avoid:
- `final`, `final2`, `final-final` (use versions instead!)
- Generic names like `test`, `temp`, `backup`

### Tags

Use tags to organize snapshots:

- `baseline` - Initial or reference versions
- `production` - Models in production
- `paper` - Research paper versions
- `experimental` - Unproven approaches
- `archived` - Old versions kept for reference

### Cleanup Strategy

- Keep all snapshots for active experiments
- Delete experimental snapshots that didn't work
- Periodically run `cleanup_orphaned_versions()`
- Archive old snapshots if needed

### Git Integration

For best reproducibility:

1. Commit your code before creating snapshots
2. Create snapshot with git info enabled
3. Optionally commit the snapshot metadata files

```bash
# Good workflow
git add .
git commit -m "Implement baseline model"
datafolio snapshot create v1.0 -d "Baseline"

# Commit snapshot metadata
git add snapshots.json items.json
git commit -m "Snapshot v1.0"
```

## Troubleshooting

### "Snapshot already exists"

Snapshots are immutable. Use a different name:

```python
# Error
folio.create_snapshot('v1.0')  # Already exists!

# Fix
folio.create_snapshot('v1.1')  # New name
```

### Large number of versions

Too many item versions? Clean up:

```python
# See what would be deleted
orphans = folio.cleanup_orphaned_versions(dry_run=True)

# Delete them
folio.cleanup_orphaned_versions()
```

### Can't delete item

Items in snapshots can't be deleted. Delete the snapshot first:

```python
# Error
folio.delete('model')  # Used by snapshot v1.0!

# Fix
folio.delete_snapshot('v1.0', cleanup_orphans=True)
folio.delete('model')  # Now works
```

## Advanced Topics

### Snapshot Internals

Snapshots are stored in `snapshots.json`:

```json
{
  "snapshots": {
    "v1.0": {
      "timestamp": "2025-01-20T15:00:00Z",
      "description": "Baseline model",
      "tags": ["baseline"],
      "item_versions": {
        "model": 1,
        "data": 1
      },
      "metadata_snapshot": {...},
      "git": {...},
      "environment": {...}
    }
  }
}
```

Item versions are tracked in `items.json`:

```json
{
  "items": [
    {
      "name": "model",
      "filename": "model.joblib",
      "version": 1,
      "in_snapshots": ["v1.0", "v1.1"]
    },
    {
      "name": "model",
      "filename": "model_v2.joblib",
      "version": 2,
      "in_snapshots": ["v2.0"]
    }
  ]
}
```

### Programmatic Snapshot Analysis

```python
# Get all snapshots
snapshots = folio.list_snapshots()

# Find snapshots with specific criteria
production_snaps = [
    s for s in snapshots
    if 'production' in s.get('tags', [])
]

# Find best-performing snapshot
best = max(snapshots,
    key=lambda s: s['metadata_snapshot'].get('accuracy', 0))

# Track accuracy over time
import matplotlib.pyplot as plt

times = [s['timestamp'] for s in snapshots]
accuracies = [s['metadata_snapshot'].get('accuracy', 0) for s in snapshots]

plt.plot(times, accuracies)
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('Model Performance Over Time')
```

## FAQ

**Q: How much disk space do snapshots use?**

A: Very little! Snapshots only create new files when you overwrite items. Unchanged items are shared across all snapshots.

**Q: Can I modify a snapshot?**

A: No, snapshots are immutable. This is essential for reproducibility.

**Q: Can I export/share a snapshot?**

A: Yes! Just share the entire bundle directory. Others can load the same snapshot.

**Q: Do snapshots work with cloud storage?**

A: Yes! DataFolio works with any storage backend (local, S3, GCS, etc.). Snapshots work the same everywhere.

**Q: Can I snapshot only part of my bundle?**

A: No, snapshots capture the complete state. But only changed items create new files, so it's efficient.

**Q: How do snapshots compare to git?**

A: Snapshots are complementary to git. Git tracks code, snapshots track data/models/results. Use both together for full reproducibility!

## Next Steps

- See the [API Reference](../reference/api.md) for complete snapshot method documentation
- Check out the [changelog](../changelog.md) for what's new
- Read the [full design document](https://github.com/ceesem/datafolio/blob/main/snapshots.md) for implementation details
