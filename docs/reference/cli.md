---
title: Command Line Interface (CLI)
---

# DataFolio CLI Reference

DataFolio provides a command-line interface for managing bundles and snapshots without writing Python code.

## Installation

The CLI is automatically available after installing datafolio:

```bash
pip install datafolio
```

## Global Options

All commands support these global options:

```bash
datafolio [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
|--------|-------------|
| `-f, --folio PATH` | Path to DataFolio bundle (default: current directory or `DATAFOLIO_PATH` env var) |
| `--version` | Show version and exit |
| `--help` | Show help message |

### Setting Default Folio Path

You can set a default folio path using the environment variable:

```bash
export DATAFOLIO_PATH=/path/to/my/bundle
datafolio describe  # Uses DATAFOLIO_PATH
```

Or specify it explicitly:

```bash
datafolio -f /path/to/my/bundle describe
```

---

## Commands

### `init` - Initialize a Bundle

Create a new DataFolio bundle.

```bash
datafolio init [PATH]
```

**Arguments:**
- `PATH` (optional): Directory to create bundle in (default: current directory)

**Examples:**

```bash
# Create bundle in current directory
datafolio init

# Create bundle in specific directory
datafolio init my_analysis

# Create with custom path
datafolio init /data/experiments/exp_001
```

**Output:**
```
✓ Initialized new DataFolio at: /data/experiments/exp_001
```

---

### `describe` - Show Bundle Information

Display detailed information about a bundle including items, metadata, and lineage.

```bash
datafolio describe [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--json` | Output in JSON format |
| `--verbose, -v` | Show detailed item information |

**Examples:**

```bash
# Basic description
datafolio describe

# Detailed output
datafolio describe --verbose

# JSON output for scripting
datafolio describe --json > bundle_info.json

# Describe specific bundle
datafolio -f /path/to/bundle describe
```

**Output:**
```
DataFolio: my_analysis
Path: /data/experiments/exp_001

Bundle Metadata:
  project: analysis
  created: 2024-01-15

Items (5):
  Tables (3):
    - raw_data (100 rows, 5 cols)
    - processed_data (100 rows, 8 cols)
    - results (50 rows, 3 cols)

  Models (1):
    - classifier (sklearn_model)

  Artifacts (1):
    - config.yaml
```

---

### `validate` - Validate Bundle

Check if a directory is a valid DataFolio bundle.

```bash
datafolio validate [PATH]
```

**Arguments:**
- `PATH` (optional): Directory to validate (default: current directory)

**Examples:**

```bash
# Validate current directory
datafolio validate

# Validate specific path
datafolio validate /data/experiments/exp_001
```

**Output:**

✅ Valid bundle:
```
✓ Valid DataFolio bundle
  - items.json: valid
  - 5 items found
  - No issues detected
```

❌ Invalid bundle:
```
✗ Not a valid DataFolio bundle
  - Missing items.json
  - Directory structure incomplete
```

**Exit Codes:**
- `0`: Valid bundle
- `1`: Invalid bundle

---

## Snapshot Commands

Manage snapshots (read-only copies) of your bundle state.

### `snapshot create` - Create Snapshot

Create a new snapshot of the current bundle state.

```bash
datafolio snapshot create NAME [OPTIONS]
```

**Arguments:**
- `NAME`: Unique name for the snapshot (e.g., 'v1.0', 'baseline', '2024-01-15')

**Options:**
| Option | Description |
|--------|-------------|
| `-d, --description TEXT` | Description of this snapshot |
| `--tags TEXT` | Comma-separated tags |
| `--metadata KEY=VALUE` | Additional metadata (can be used multiple times) |

**Examples:**

```bash
# Simple snapshot
datafolio snapshot create v1.0

# With description
datafolio snapshot create baseline -d "Initial baseline results"

# With tags
datafolio snapshot create exp_001 --tags "experiment,baseline,validated"

# With custom metadata
datafolio snapshot create v2.0 \
  -d "Improved model" \
  --metadata accuracy=0.95 \
  --metadata model=transformer
```

**Output:**
```
✓ Created snapshot 'v1.0'
  Items: 5 tables, 1 model, 1 artifact
  Time: 2024-01-15 14:30:00 UTC
```

---

### `snapshot list` - List Snapshots

List all snapshots in the bundle.

```bash
datafolio snapshot list [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--json` | Output in JSON format |
| `--verbose, -v` | Show detailed information |

**Examples:**

```bash
# List all snapshots
datafolio snapshot list

# Detailed listing
datafolio snapshot list --verbose

# JSON output
datafolio snapshot list --json
```

**Output:**
```
Snapshots (3):

  v1.0
    Created: 2024-01-15 14:30:00
    Items: 7
    Description: Initial baseline

  v1.1
    Created: 2024-01-16 10:15:00
    Items: 8
    Description: Added validation data

  v2.0
    Created: 2024-01-17 15:45:00
    Items: 9
    Description: Improved model (accuracy=0.95)
```

---

### `snapshot show` - Show Snapshot Details

Display detailed information about a specific snapshot.

```bash
datafolio snapshot show NAME
```

**Arguments:**
- `NAME`: Snapshot name

**Examples:**

```bash
datafolio snapshot show v1.0
```

**Output:**
```
Snapshot: v1.0
Created: 2024-01-15 14:30:00 UTC
Description: Initial baseline results

Items (7):
  Tables (5):
    - raw_data (v1)
    - processed_data (v1)
    - train_data (v1)
    - test_data (v1)
    - results (v1)

  Models (1):
    - classifier (v1)

  Artifacts (1):
    - config.yaml (v1)

Metadata:
  accuracy: 0.92
  model_type: random_forest
```

---

### `snapshot compare` - Compare Snapshots

Compare two snapshots to see what changed.

```bash
datafolio snapshot compare SNAPSHOT1 SNAPSHOT2
```

**Arguments:**
- `SNAPSHOT1`: First snapshot name
- `SNAPSHOT2`: Second snapshot name

**Examples:**

```bash
datafolio snapshot compare v1.0 v2.0
```

**Output:**
```
Comparing v1.0 → v2.0

Added (2):
  + new_features (table)
  + updated_model (model)

Modified (1):
  ~ results (table): rows changed 50 → 75

Removed (0):

Summary:
  2 additions, 1 modification, 0 deletions
```

---

### `snapshot diff` - Diff Against Snapshot

Show changes between current state and a snapshot.

```bash
datafolio snapshot diff [SNAPSHOT]
```

**Arguments:**
- `SNAPSHOT` (optional): Snapshot name (default: latest snapshot)

**Examples:**

```bash
# Compare with latest snapshot
datafolio snapshot diff

# Compare with specific snapshot
datafolio snapshot diff v1.0
```

**Output:**
```
Changes since v1.0:

Modified (2):
  ~ results (table): updated
  ~ classifier (model): updated

Added (1):
  + validation_results (table)

Current state has 3 changes from snapshot v1.0
```

---

### `snapshot status` - Show Bundle Status

Show current bundle state compared to the last snapshot.

```bash
datafolio snapshot status
```

**Examples:**

```bash
datafolio snapshot status
```

**Output:**
```
Current Status:

Last snapshot: v2.0 (2024-01-17 15:45:00)

Changes since v2.0:
  Modified: 1 item
  Added: 0 items
  Deleted: 0 items

Modified items:
  ~ results (table): 75 → 100 rows

💡 Tip: Create a new snapshot to save current state
      datafolio snapshot create v2.1
```

---

### `snapshot delete` - Delete Snapshot

Delete a snapshot from the bundle.

```bash
datafolio snapshot delete NAME [OPTIONS]
```

**Arguments:**
- `NAME`: Snapshot name to delete

**Options:**
| Option | Description |
|--------|-------------|
| `--force` | Skip confirmation prompt |
| `--cleanup-orphans` | Also remove orphaned item versions |

**Examples:**

```bash
# Delete with confirmation
datafolio snapshot delete old_experiment

# Force delete without confirmation
datafolio snapshot delete old_experiment --force

# Delete and cleanup orphaned versions
datafolio snapshot delete old_experiment --cleanup-orphans
```

**Output:**
```
⚠ Warning: This will permanently delete snapshot 'old_experiment'
Continue? [y/N]: y
✓ Deleted snapshot 'old_experiment'
```

---

### `snapshot gc` - Garbage Collection

Clean up orphaned item versions that are no longer referenced by any snapshot.

```bash
datafolio snapshot gc [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would be deleted without actually deleting |
| `--verbose, -v` | Show detailed information |

**Examples:**

```bash
# Dry run to see what would be deleted
datafolio snapshot gc --dry-run

# Actually perform cleanup
datafolio snapshot gc

# Verbose output
datafolio snapshot gc --verbose
```

**Output:**
```
Scanning for orphaned versions...

Would delete (3):
  - results.v1.parquet (orphaned since v1.0 deleted)
  - old_model.v2.pkl (no longer referenced)
  - temp_data.v1.parquet (orphaned)

Total space to free: 45.2 MB

Run without --dry-run to perform cleanup
```

---

### `snapshot reproduce` - Show Reproduction Instructions

Generate instructions for reproducing a snapshot.

```bash
datafolio snapshot reproduce NAME
```

**Arguments:**
- `NAME`: Snapshot name

**Examples:**

```bash
datafolio snapshot reproduce v1.0
```

**Output:**
```
Reproduction Instructions for Snapshot: v1.0

To reproduce this exact state:

1. Load the snapshot:
   ```python
   import datafolio
   folio = datafolio.DataFolio.load_snapshot('v1.0')
   ```

2. Items in this snapshot:
   - raw_data (table)
   - processed_data (table)
   - classifier (model)
   - config.yaml (artifact)

3. Dependencies:
   raw_data → processed_data → classifier

4. Metadata:
   - Created: 2024-01-15 14:30:00 UTC
   - Python: 3.10.2
   - datafolio: 0.2.0

5. To export this snapshot:
   ```python
   folio.export_snapshot('v1.0', '/path/to/export')
   ```
```

---

## Usage Examples

### Common Workflows

#### 1. Create and Manage a Bundle

```bash
# Initialize new bundle
datafolio init my_analysis

# Work with Python to add data...
# (see Python API documentation)

# Create snapshot when ready
datafolio -f my_analysis snapshot create baseline -d "Initial results"

# View bundle info
datafolio -f my_analysis describe
```

#### 2. Track Progress with Snapshots

```bash
# After initial analysis
datafolio snapshot create v1.0 -d "Initial model"

# Continue working...

# Create another snapshot
datafolio snapshot create v1.1 -d "Improved preprocessing"

# Compare versions
datafolio snapshot compare v1.0 v1.1

# Check what changed since last snapshot
datafolio snapshot diff
```

#### 3. Validate and Inspect Bundles

```bash
# Validate bundle structure
datafolio validate /path/to/bundle

# View detailed description
datafolio -f /path/to/bundle describe --verbose

# List all snapshots
datafolio -f /path/to/bundle snapshot list
```

#### 4. Cleanup Old Snapshots

```bash
# List all snapshots
datafolio snapshot list

# Delete old experiments
datafolio snapshot delete old_experiment

# Clean up orphaned versions
datafolio snapshot gc
```

---

## Integration with Python API

The CLI complements the Python API. A typical workflow:

```python
# Python: Create and populate bundle
import datafolio
import pandas as pd

folio = datafolio.DataFolio('my_analysis')
folio.add_table('results', df)
folio.add_model('classifier', model)
```

```bash
# CLI: Create snapshot
datafolio -f my_analysis snapshot create v1.0 -d "Initial results"

# CLI: Validate
datafolio -f my_analysis validate

# CLI: View status
datafolio -f my_analysis describe
```

```python
# Python: Load snapshot later
folio = datafolio.DataFolio.load_snapshot('v1.0')
results = folio.get_table('results')
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATAFOLIO_PATH` | Default path for folio operations |
| `DATAFOLIO_CACHE_ENABLED` | Enable caching ('true'/'false') |
| `DATAFOLIO_CACHE_DIR` | Cache directory path |
| `DATAFOLIO_CACHE_TTL` | Cache TTL in seconds |

**Example:**

```bash
export DATAFOLIO_PATH=/data/experiments/current
export DATAFOLIO_CACHE_ENABLED=true
export DATAFOLIO_CACHE_DIR=/tmp/datafolio_cache

# Now CLI commands use these defaults
datafolio describe
datafolio snapshot list
```

---

## Scripting with the CLI

The CLI is designed for use in scripts and automation:

### Bash Script Example

```bash
#!/bin/bash

# Validate bundle
if ! datafolio validate /data/bundle; then
    echo "Invalid bundle!"
    exit 1
fi

# Create dated snapshot
DATE=$(date +%Y-%m-%d)
datafolio -f /data/bundle snapshot create "daily_$DATE" \
    -d "Daily backup" \
    --tags "automated,backup"

# Cleanup old snapshots (keep last 7 days)
# ... (custom logic to delete old snapshots)

echo "Backup complete: daily_$DATE"
```

### JSON Output for Processing

```bash
# Get bundle info as JSON
INFO=$(datafolio describe --json)

# Extract item count using jq
TABLE_COUNT=$(echo "$INFO" | jq '.tables | length')
echo "Bundle has $TABLE_COUNT tables"

# List snapshots as JSON
SNAPSHOTS=$(datafolio snapshot list --json)
LATEST=$(echo "$SNAPSHOTS" | jq -r '.[0].name')
echo "Latest snapshot: $LATEST"
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid arguments, bundle not found, operation failed) |
| 2 | Validation failed (for `validate` command) |

---

## Getting Help

For any command, use `--help`:

```bash
datafolio --help
datafolio snapshot --help
datafolio snapshot create --help
```

For more detailed documentation, see:
- [Python API Reference](datafolio-api.md)
- [Getting Started Guide](../guides/getting-started.md)
- [Snapshots Guide](../guides/snapshots.md)
