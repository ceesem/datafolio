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

!!! note "Limitation: local paths only"
    The `-f/--folio` flag and `DATAFOLIO_PATH` only accept **local** filesystem
    paths — the CLI checks that the path exists on disk, so cloud URIs like
    `gs://bucket/experiment` fail with `Folio not found`. This is a known,
    deliberate limitation in v2.0. To work with a cloud-hosted folio, use the
    Python API (`DataFolio('gs://bucket/experiment')`) instead.
    (`datafolio init` is the one exception: it accepts a cloud URI as its
    `PATH` argument to create a new cloud bundle.)

---

## Commands

### `init` - Initialize a Bundle

Create a new DataFolio bundle.

```bash
datafolio init [OPTIONS] [PATH]
```

**Arguments:**
- `PATH` (optional): Directory to create bundle in (default: current directory)

**Options:**
| Option | Description |
|--------|-------------|
| `-d, --description TEXT` | Bundle description |
| `-n, --name TEXT` | Bundle name (default: directory name) |

**Examples:**

```bash
# Create bundle in current directory
datafolio init

# Create bundle in specific directory, with a description
datafolio init my_folio -d "Demo analysis bundle"

# Create a cloud bundle (init only; other commands can't target cloud paths)
datafolio init gs://my-bucket/experiment -d "Cloud experiment"
```

**Output:**
```
Initializing bundle in: /data/experiments/my_folio

✓ Initialized DataFolio bundle: my_folio
  Path: /data/experiments/my_folio
  Description: Demo analysis bundle

Next steps:
  # Add data to your bundle
  cd /data/experiments/my_folio
  python -c "from datafolio import DataFolio; folio = DataFolio('.'); ..."

  # Create a snapshot when ready
  datafolio snapshot create v1.0 -d 'Initial version'
```

---

### `describe` - Show Bundle Information

Display detailed information about a bundle including items, metadata, and snapshots.

```bash
datafolio describe [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--max-metadata INTEGER` | Maximum metadata fields to show (default: 10) |
| `-s, --snapshot TEXT` | Describe a specific snapshot instead of the full bundle |

**Examples:**

```bash
# Basic description
datafolio describe

# Describe a specific snapshot's state
datafolio describe --snapshot v1.0

# Describe specific bundle
datafolio -f /path/to/bundle describe
```

**Output:**
```
DataFolio: ./my_folio
=====================

Created: Today at 8:09 PM EDT
Updated: Today at 8:10 PM EDT

Metadata (2 fields):
  • description: Demo analysis bundle
  • project: demo

Snapshots (1):
  • v1.0: Initial baseline
    ↳ created: Today at 8:10 PM EDT, items: 2
    ↳ tags: baseline

Tables (2):
  • raw_data: Raw measurements
    ↳ size: 3.5 KB
  • results: Group means
    ↳ size: 1.6 KB
```

With `--snapshot`, the description covers the bundle as it existed at snapshot time:

```
Snapshot: v1.1
==============

Description: Added validation data

Created: Today at 8:10 PM EDT
Tags: validated

Items (3):

  Tables (3):
    • raw_data: Raw measurements
    • results: Group means with counts
    • validation_data: Held-out validation set
```

---

### `validate` - Validate Bundle

Check that a directory is a valid DataFolio bundle. This runs a real integrity
check: every item's payload must exist, and for locally-owned items the stored
checksum must match.

```bash
datafolio validate [PATH]
```

**Arguments:**
- `PATH` (optional): Directory to validate (default: current directory or `DATAFOLIO_PATH`)

**Examples:**

```bash
# Validate current directory
datafolio validate

# Validate specific path
datafolio validate /data/experiments/my_folio
```

**Output:**

✅ Valid bundle:
```
✓ Valid DataFolio bundle: ./my_folio
  Items: 2 (all payloads verified)
  Snapshots: 1
  Description: Demo analysis bundle
```

❌ Not a bundle:
```
Error: Not a DataFolio bundle: /tmp
Missing required files: items.json and metadata.json
Tip: Use 'datafolio init' to create a new folio, or use --folio/-f to specify the correct path.
```

**Exit Codes:**
- `0`: Valid bundle
- `1`: Invalid bundle (not a folio, or one or more items have missing/corrupt payloads)

---

## Snapshot Commands

Manage snapshots (read-only records) of your bundle state.

### `snapshot create` - Create Snapshot

Create a new snapshot of the current bundle state.

By default, only git information is captured (with credentials automatically
removed from remote URLs). Environment and execution context are opt-in via flags.

```bash
datafolio snapshot create NAME [OPTIONS]
```

**Arguments:**
- `NAME`: Unique name for the snapshot (e.g., 'v1.0', 'baseline', '2024-01-15'). Letters, numbers, hyphens, underscores, and dots only.

**Options:**
| Option | Description |
|--------|-------------|
| `-d, --description TEXT` | Description of this snapshot |
| `-t, --tag TEXT` | Tag for the snapshot (can be used multiple times) |
| `--no-git` | Don't capture git information |
| `--env` | Capture environment information (Python version, packages) |
| `--exec` | Capture execution context (entry point, working dir) |

**Examples:**

```bash
# Simple snapshot
datafolio snapshot create v1.0

# With description
datafolio snapshot create baseline -d "Initial baseline results"

# With tags (repeat -t for each tag)
datafolio snapshot create exp_001 -t experiment -t baseline -t validated

# Include environment and execution info
datafolio snapshot create v2.0 -d "Improved model" --env --exec
```

**Output:**
```
✓ Created snapshot 'v1.0'
  Items: 2
  Description: Initial baseline
  Tags: baseline
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
| `-t, --tag TEXT` | Filter by tag |

**Examples:**

```bash
# List all snapshots
datafolio snapshot list

# Only snapshots with a given tag
datafolio snapshot list --tag baseline
```

**Output:**
```
                               Snapshots (2)
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Name ┃ Description           ┃ Items ┃ Created              ┃ Tags      ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ v1.1 │ Added validation data │     3 │ 2026-07-17 20:10 EDT │ validated │
│ v1.0 │ Initial baseline      │     2 │ 2026-07-17 20:10 EDT │ baseline  │
└──────┴───────────────────────┴───────┴──────────────────────┴───────────┘
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
============================================================

Description: Initial baseline
Tags: baseline
Timestamp: 2026-07-17 20:10:01 EDT

Items (2):
  • raw_data
  • results

Metadata (2 fields):
  • description: Demo analysis bundle
  • project: demo
```

If the snapshot captured git or environment information, those sections
(commit, branch, dirty status; Python version) are shown as well.

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
datafolio snapshot compare v1.0 v1.1
```

**Output:**
```
Comparing v1.0 → v1.1
============================================================

Added (1):
  + validation_data

Modified (1):
  ~ results

Unchanged (1):
  = raw_data

Metadata Changes (1):
  updated_at: 2026-07-18T00:09:50.364141+00:00 → 2026-07-18T00:10:19.543277+00:00

Summary:
  Added: 1
  Removed: 0
  Modified: 1
  Unchanged: 1
```

---

### `snapshot diff` - Diff Against Snapshot

Show changes between the current state and a snapshot.

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
Comparing current state to snapshot 'v1.0'
============================================================

Added (1):
  + validation_data

Modified (1):
  ~ results

Unchanged (1):
  = raw_data

Summary:
  Added: 1
  Removed: 0
  Modified: 1
  Unchanged: 1
```

---

### `snapshot status` - Show Bundle Status

Show current bundle state compared to the last snapshot. Similar to `git status`.

```bash
datafolio snapshot status
```

**Output:**
```
Current bundle: ./my_folio
Last snapshot: v1.0 (2026-07-17)

Changes since last snapshot:

Added (1):
  + validation_data

Modified (1):
  ~ results

Unchanged: 1 items
```

If no snapshots exist yet, it suggests creating your first one. If nothing
changed, it prints `✓ No changes since last snapshot`.

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
| `-y, --yes` | Skip confirmation prompt |
| `--cleanup / --no-cleanup` | Cleanup orphaned versions after deletion (default: no cleanup) |

**Examples:**

```bash
# Delete with confirmation prompt
datafolio snapshot delete old_experiment

# Delete without confirmation
datafolio snapshot delete old_experiment -y

# Delete and clean up orphaned item versions
datafolio snapshot delete old_experiment --cleanup
```

**Output:**
```
Delete snapshot 'v1.0'? [y/N]: y
✓ Deleted snapshot 'v1.0'
```

---

### `snapshot gc` - Garbage Collection

Clean up orphaned item versions that are no longer referenced by any snapshot
or by the current bundle state.

```bash
datafolio snapshot gc [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would be deleted without actually deleting |

**Examples:**

```bash
# Dry run to see what would be deleted
datafolio snapshot gc --dry-run

# Actually perform cleanup
datafolio snapshot gc
```

**Output:**

Dry run:
```
Would delete 1 orphaned version(s):
  • results--r3.parquet
```

Real run:
```
✓ Deleted 1 orphaned version(s)
```

---

## Usage Examples

### Common Workflows

#### 1. Create and Manage a Bundle

```bash
# Initialize new bundle
datafolio init my_analysis -d "My analysis"

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
# Validate bundle structure and payload integrity
datafolio validate /path/to/bundle

# View detailed description
datafolio -f /path/to/bundle describe

# List all snapshots
datafolio -f /path/to/bundle snapshot list
```

#### 4. Cleanup Old Snapshots

```bash
# List all snapshots
datafolio snapshot list

# Delete old experiments
datafolio snapshot delete old_experiment -y

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
folio.add('results', df)
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
# Python: Load snapshot later (path first, then snapshot name)
folio = datafolio.DataFolio.load_snapshot('my_analysis', 'v1.0')
results = folio.get('results')
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATAFOLIO_PATH` | Default path for folio operations (local paths only) |

**Example:**

```bash
export DATAFOLIO_PATH=/data/experiments/current

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
    -t automated -t backup

echo "Backup complete: daily_$DATE"
```

### Structured Output for Processing

The CLI prints human-readable output only. For machine-readable data, use the
Python API, which returns plain dicts and lists:

```bash
# Name of the most recent snapshot
LATEST=$(python -c "from datafolio import DataFolio; \
    print(DataFolio('/data/bundle', read_only=True).list_snapshots()[0]['name'])")
echo "Latest snapshot: $LATEST"
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid arguments, bundle not found, validation failed, operation failed) |

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
