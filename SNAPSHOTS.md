# DataFolio Snapshots: Design & Implementation Plan

**Version:** 1.0
**Status:** Design Phase
**Target Release:** v0.6
**Author:** Design discussion 2025-11-20

---

## Table of Contents
- [1. Motivation](#1-motivation)
- [2. User Stories](#2-user-stories)
- [3. Design Principles](#3-design-principles)
- [4. Architecture](#4-architecture)
- [5. Data Structures](#5-data-structures)
- [6. API Design](#6-api-design)
- [7. CLI Design](#7-cli-design)
- [8. Implementation Plan](#8-implementation-plan)
- [9. Testing Strategy](#9-testing-strategy)
- [10. Migration Path](#10-migration-path)

---

## 1. Motivation

### The Problem

**Current pain point:** Iterative ML research requires experimenting with different models, hyperparameters, and preprocessing steps. Researchers face a dilemma:

1. **Overwrite artifacts** → Lose ability to return to previous good versions
2. **Duplicate bundles** → Disk space explosion (10 iterations × 5GB = 50GB)
3. **Manual versioning** → Folder chaos: `exp-v2-final-FINAL-actually-final/`

**Real-world scenario:**
```python
# Week 1: Achieve 89% accuracy
folio.add_model('classifier', good_model)
folio.metadata['accuracy'] = 0.89

# Week 2: Experiment with new approach
folio.add_model('classifier', experimental_model, overwrite=True)  # Old model LOST!
folio.metadata['accuracy'] = 0.85  # Worse!

# Week 3: Product team wants to ship the 89% model
# Problem: It no longer exists. Must recreate from git history.
```

### Why Existing Tools Don't Solve This

| Tool | What It Does | Why It's Not Enough |
|------|--------------|---------------------|
| **DVC** | Git for data | Requires external storage, Git-centric, complex setup |
| **MLflow** | Experiment tracking | Server-based, doesn't bundle artifacts, can't "load v1.0 into memory" |
| **Git LFS** | Large file versioning | No semantic understanding of ML artifacts, no lineage |
| **Pachyderm** | Data pipelines | Production infrastructure, overkill for research |

**The gap:** Nobody combines self-contained bundles + semantic ML artifacts + lineage + cheap snapshots + simple Python API.

### The Solution

**Snapshots:** Immutable, named checkpoints of bundle state with copy-on-write versioning.

**Key features:**
- ✅ Zero data duplication for unchanged items
- ✅ Automatic git commit tracking
- ✅ Environment capture (Python version, dependencies)
- ✅ Entry point recording (how to reproduce)
- ✅ Full metadata state preservation
- ✅ Simple Python API

---

## 2. User Stories

### Story 1: Paper Submission & Revision

**Sarah, ML Researcher:**

```python
# September: Working on paper
folio = DataFolio('research/protein-analysis')
folio.add_table('train_data', train_df)
folio.add_model('classifier', model_v1)
folio.add_json('hyperparameters', params)
folio.metadata['accuracy'] = 0.89

# Ready to submit
folio.create_snapshot(
    'neurips-2025-submission',
    description='Exact version submitted to NeurIPS',
    entry_point='python train.py --config config.json',
    tags=['paper', 'neurips']
)

# November: Continue research (overwrites model)
folio.add_model('classifier', improved_model, overwrite=True)
folio.metadata['accuracy'] = 0.91

# February: Reviewers ask for rerun with changes
paper_folio = DataFolio.load_snapshot(
    'research/protein-analysis',
    'neurips-2025-submission'
)

# EXACT state from September!
assert paper_folio.metadata['accuracy'] == 0.89
model = paper_folio.get_model('classifier')  # Original model
```

**Value:** Full reproducibility for paper submission without duplicating GBs of data.

### Story 2: A/B Testing in Production

**DevOps Team:**

```python
# Deploy baseline model to production
baseline = DataFolio.load_snapshot('models/recommender', 'v2.0-baseline')
deploy_model(baseline.get_model('recommender'), endpoint='prod-a')

# Deploy experimental model for A/B test
experimental = DataFolio.load_snapshot('models/recommender', 'v3.0-experimental')
deploy_model(experimental.get_model('recommender'), endpoint='prod-b')

# Compare performance
baseline.metadata['p95_latency']      # 50ms
experimental.metadata['p95_latency']  # 45ms

# Winner! Promote experimental to new baseline
folio = DataFolio('models/recommender')
folio.create_snapshot('v3.0-baseline', description='New production baseline')
```

**Value:** Deploy specific versions, compare in production, promote winners.

### Story 3: Model Iteration Without Fear

**Data Scientist:**

```python
folio = DataFolio('experiments/fraud-detection')

# Week 1: Baseline
folio.add_model('detector', baseline_model)
folio.create_snapshot('v1.0-baseline')

# Week 2: Try neural net (worse!)
folio.add_model('detector', neural_net, overwrite=True)
folio.create_snapshot('v2.0-neural')  # Save even though worse

# Week 3: Try ensemble (better!)
folio.add_model('detector', ensemble, overwrite=True)
folio.create_snapshot('v3.0-ensemble')

# Week 4: Product wants baseline for now
prod = DataFolio.load_snapshot('experiments/fraud-detection', 'v1.0-baseline')

# Research continues with ensemble...
# No conflict! Both versions available.
```

**Value:** Experiment aggressively while maintaining production stability.

---

## 3. Design Principles

### Principle 1: Snapshots Are Immutable

Once created, snapshots cannot be modified. This ensures reproducibility.

```python
folio.create_snapshot('v1.0')
folio.create_snapshot('v1.0')  # ❌ Error: Snapshot already exists

# To update, create new snapshot
folio.create_snapshot('v1.1')  # ✅ New snapshot
```

**Rationale:** Immutability is essential for reproducibility. If snapshots can change, they're not snapshots.

### Principle 2: Copy-on-Write Versioning

Only create new file versions when overwriting items that exist in snapshots.

```python
# Item NOT in any snapshot
folio.add_model('m', v1)
folio.add_model('m', v2, overwrite=True)  # Replaces file (no versioning)

# Item IS in snapshot
folio.create_snapshot('s1')
folio.add_model('m', v3, overwrite=True)  # Creates m_v2.joblib (preserves v1)
```

**Rationale:** Minimize disk usage. Only version what's necessary for snapshot integrity.

### Principle 3: Automatic Context Capture

Snapshots automatically capture:
- Git commit hash (if in repo)
- Python version
- Environment hash (uv.lock or requirements.txt)
- Timestamp
- All item versions
- All metadata state

**Rationale:** Full reproducibility requires code + environment + data. Automate this.

### Principle 4: Shared Immutable Data

Unchanged items are shared across snapshots (reference, not copy).

```python
folio.add_table('data', big_df)  # 5GB file
folio.create_snapshot('v1.0')
folio.create_snapshot('v2.0')
folio.create_snapshot('v3.0')

# Storage: 5GB (not 15GB!)
# All snapshots reference same data.parquet
```

**Rationale:** Disk space efficiency. Most data doesn't change between iterations.

---

## 4. Architecture

### 4.1 File Structure

```
bundle/
├── metadata.json              # Current metadata
├── items.json                 # All item versions (extended)
├── snapshots.json             # Snapshot registry (NEW!)
│
├── tables/
│   ├── train_data.parquet           # v1 (shared by v1.0 and v2.0)
│   └── train_data_v2.parquet        # v2 (only in v3.0)
│
├── models/
│   ├── classifier.joblib            # v1 (in v1.0 snapshot)
│   ├── classifier_v2.joblib         # v2 (in v2.0 snapshot)
│   └── classifier_v3.joblib         # v3 (current, not in snapshot yet)
│
└── artifacts/
    └── config.json              # v1 (shared by all snapshots)
```

### 4.2 Snapshot Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│ WORKING STATE (Current)                                          │
│ - Latest version of each item                                    │
│ - Current metadata                                               │
│ - Mutable: Can add, overwrite, delete                           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ create_snapshot('v1.0')
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ SNAPSHOT 'v1.0' (Immutable)                                      │
│ - References specific versions of items                         │
│ - Frozen metadata state                                         │
│ - Git commit, environment captured                              │
│ - Immutable: Cannot modify                                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ add_model('m', new, overwrite=True)
                              │ (creates new version, preserves old)
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ WORKING STATE (Updated)                                          │
│ - New version of 'm' (old preserved for snapshot)               │
│ - Can continue experimenting                                    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ create_snapshot('v2.0')
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ SNAPSHOT 'v2.0' (Immutable)                                      │
│ - References new version of 'm'                                 │
│ - May share other items with 'v1.0'                            │
└──────────────────────────────────────────────────────────────────┘
```

### 4.3 Version Management

**Item versioning rules:**

1. **New item** → Version 1, no snapshot reference
2. **Overwrite item NOT in snapshot** → Replace file, increment version (optional)
3. **Overwrite item IN snapshot** → Create new versioned file, preserve old
4. **Create snapshot** → Mark all current items as "in this snapshot"
5. **Delete item** → Fail if referenced by any snapshot

**Garbage collection:**
- Items not in any snapshot AND not current → Can be deleted
- Items in at least one snapshot → Must be preserved
- User-triggered: `folio.cleanup_orphaned_versions()`

---

## 5. Data Structures

### 5.1 Extended items.json

```json
{
  "items": [
    {
      "name": "classifier",
      "item_type": "model",
      "filename": "classifier.joblib",
      "version": 1,
      "checksum": "abc123...",
      "in_snapshots": ["v1.0-baseline"],
      "created_at": "2025-11-20T10:00:00Z",
      "num_rows": 1000,
      "num_cols": 10
    },
    {
      "name": "classifier",
      "item_type": "model",
      "filename": "classifier_v2.joblib",
      "version": 2,
      "checksum": "def456...",
      "in_snapshots": ["v2.0-neural"],
      "replaces_version": 1,
      "created_at": "2025-11-27T14:30:00Z"
    },
    {
      "name": "train_data",
      "item_type": "included_table",
      "filename": "train_data.parquet",
      "version": 1,
      "checksum": "xyz789...",
      "in_snapshots": ["v1.0-baseline", "v2.0-neural"],
      "created_at": "2025-11-20T10:00:00Z",
      "num_rows": 10000,
      "num_cols": 50
    }
  ],
  "current_versions": {
    "classifier": 2,
    "train_data": 1
  }
}
```

**New fields:**
- `version`: Integer version number (starts at 1)
- `in_snapshots`: List of snapshot names referencing this version
- `replaces_version`: Points to previous version (for history)

**Changes to existing structure:**
- `items` can now contain multiple entries with same `name` but different `version`
- `current_versions` tracks which version is "active" for each item name

### 5.2 snapshots.json (NEW!)

```json
{
  "snapshots": {
    "v1.0-baseline": {
      "timestamp": "2025-11-20T15:00:00Z",
      "description": "Baseline random forest model",
      "tags": ["baseline", "paper"],

      "item_versions": {
        "classifier": 1,
        "train_data": 1,
        "config": 1
      },

      "metadata_snapshot": {
        "accuracy": 0.89,
        "f1_score": 0.87,
        "experiment": "diagnosis-v1"
      },

      "git": {
        "commit": "a3f2b8c1d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0",
        "commit_short": "a3f2b8c",
        "branch": "main",
        "remote": "https://github.com/lab/protein-analysis.git",
        "dirty": false
      },

      "environment": {
        "python_version": "3.11.5",
        "platform": "Linux-5.15.0-x86_64",
        "uv_lock_hash": "def456...",
        "requirements": "numpy==1.24.0\npandas==2.0.0\n..."
      },

      "execution": {
        "entry_point": "python train.py --config config.json",
        "working_dir": "/home/user/research/protein"
      }
    },

    "v2.0-neural": {
      "timestamp": "2025-11-27T16:30:00Z",
      "description": "Experimental neural network",
      "tags": ["experimental"],

      "item_versions": {
        "classifier": 2,
        "train_data": 1,
        "config": 1
      },

      "metadata_snapshot": {
        "accuracy": 0.85,
        "architecture": "neural_net"
      },

      "git": {
        "commit": "b4a3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5",
        "commit_short": "b4a3c2d",
        "branch": "experiment/neural-net",
        "remote": "https://github.com/lab/protein-analysis.git",
        "dirty": true
      },

      "environment": {
        "python_version": "3.11.5",
        "platform": "Linux-5.15.0-x86_64",
        "uv_lock_hash": "ghi789..."
      },

      "execution": {
        "entry_point": "python train_neural.py",
        "working_dir": "/home/user/research/protein"
      }
    }
  }
}
```

**Snapshot metadata fields:**
- `timestamp`: When snapshot was created (ISO 8601)
- `description`: Human-readable description
- `tags`: List of tags for organization
- `item_versions`: Map of item_name → version_number
- `metadata_snapshot`: Full copy of metadata at snapshot time
- `git`: Git repository state (optional, auto-captured)
- `environment`: Python environment state (optional, auto-captured)
- `execution`: How to reproduce (optional, user-provided)

---

## 6. API Design

### 6.1 Core Snapshot Methods

#### create_snapshot()

```python
def create_snapshot(
    self,
    name: str,
    description: Optional[str] = None,
    capture_git: bool = True,
    capture_env: bool = True,
    entry_point: Optional[str] = None,
    tags: Optional[list[str]] = None
) -> None:
    """Create immutable snapshot of current bundle state.

    Args:
        name: Unique snapshot name (e.g., 'v1.0', 'paper-submission')
        description: Human-readable description
        capture_git: Auto-capture git commit info (default: True)
        capture_env: Auto-capture Python environment (default: True)
        entry_point: Command to reproduce results (e.g., 'python train.py')
        tags: Optional tags for organization

    Raises:
        ValueError: If snapshot name already exists

    Examples:
        Basic snapshot:
        >>> folio.create_snapshot('v1.0', 'Initial baseline model')

        Full reproducibility:
        >>> folio.create_snapshot(
        ...     'neurips-2025-submission',
        ...     description='Paper submission version',
        ...     entry_point='python train.py --config config.json',
        ...     tags=['paper', 'neurips', 'final']
        ... )

        Skip git capture (if not in repo):
        >>> folio.create_snapshot('v1.0', capture_git=False)
    """
```

#### load_snapshot() (Class Method)

```python
@classmethod
def load_snapshot(
    cls,
    path: Union[str, Path],
    snapshot: str
) -> "DataFolio":
    """Load bundle in specific snapshot state.

    Args:
        path: Path to bundle directory
        snapshot: Snapshot name to load

    Returns:
        DataFolio instance in snapshot state (read-only mode)

    Raises:
        KeyError: If snapshot doesn't exist

    Examples:
        Load specific version:
        >>> folio = DataFolio.load_snapshot(
        ...     'research/protein-analysis',
        ...     'neurips-2025-submission'
        ... )
        >>> assert folio.metadata['accuracy'] == 0.89
        >>> model = folio.get_model('classifier')  # v1

        Compare snapshots:
        >>> v1 = DataFolio.load_snapshot('research/exp', 'v1.0')
        >>> v2 = DataFolio.load_snapshot('research/exp', 'v2.0')
        >>> print(v1.metadata['accuracy'], v2.metadata['accuracy'])
        0.89 0.91
    """
```

#### list_snapshots()

```python
def list_snapshots(
    self,
    tags: Optional[list[str]] = None,
    sort_by: str = 'timestamp'
) -> list[Dict[str, Any]]:
    """List all snapshots in bundle.

    Args:
        tags: Filter by tags (returns snapshots with ANY of these tags)
        sort_by: Sort key ('timestamp', 'name')

    Returns:
        List of snapshot info dicts

    Examples:
        All snapshots:
        >>> snapshots = folio.list_snapshots()
        >>> for snap in snapshots:
        ...     print(f"{snap['name']}: {snap['description']}")
        v1.0-baseline: Initial model
        v2.0-neural: Experimental neural net

        Filter by tag:
        >>> paper_snaps = folio.list_snapshots(tags=['paper'])
        >>> print(paper_snaps[0]['name'])
        neurips-2025-submission
    """
```

#### delete_snapshot()

```python
def delete_snapshot(
    self,
    name: str,
    cleanup_orphans: bool = False
) -> None:
    """Delete a snapshot.

    Args:
        name: Snapshot name to delete
        cleanup_orphans: If True, delete item versions no longer in any snapshot

    Examples:
        Delete snapshot only (keeps files):
        >>> folio.delete_snapshot('experimental-v5')

        Delete snapshot and cleanup:
        >>> folio.delete_snapshot('experimental-v5', cleanup_orphans=True)
        # Deletes snapshot AND any item versions only in this snapshot
    """
```

#### compare_snapshots()

```python
def compare_snapshots(
    self,
    snapshot1: str,
    snapshot2: str
) -> Dict[str, Any]:
    """Compare two snapshots.

    Args:
        snapshot1: First snapshot name
        snapshot2: Second snapshot name

    Returns:
        Dictionary with comparison results

    Examples:
        >>> diff = folio.compare_snapshots('v1.0', 'v2.0')
        >>> print(diff)
        {
            'metadata_changes': {
                'accuracy': (0.89, 0.91),
                'architecture': ('random_forest', 'neural_net')
            },
            'added_items': ['neural_config'],
            'removed_items': [],
            'modified_items': ['classifier'],
            'shared_items': ['train_data', 'config']
        }
    """
```

### 6.2 Utility Methods

#### restore_snapshot()

```python
def restore_snapshot(
    self,
    snapshot: str,
    confirm: bool = False
) -> None:
    """Restore working state to snapshot (DESTRUCTIVE).

    Args:
        snapshot: Snapshot name to restore
        confirm: Must be True to proceed (safety check)

    Raises:
        ValueError: If confirm=False

    Examples:
        >>> folio.restore_snapshot('v1.0', confirm=True)
        # Working state now matches v1.0 snapshot
        # WARNING: Overwrites current state!
    """
```

#### cleanup_orphaned_versions()

```python
def cleanup_orphaned_versions(
    self,
    dry_run: bool = False
) -> list[str]:
    """Delete item versions not in any snapshot and not current.

    Args:
        dry_run: If True, return what would be deleted without deleting

    Returns:
        List of deleted filenames

    Examples:
        See what would be deleted:
        >>> orphans = folio.cleanup_orphaned_versions(dry_run=True)
        >>> print(f"Would delete {len(orphans)} files: {orphans}")

        Actually delete:
        >>> deleted = folio.cleanup_orphaned_versions()
        >>> print(f"Deleted {len(deleted)} orphaned versions")
    """
```

#### get_snapshot_info()

```python
def get_snapshot_info(
    self,
    snapshot: str
) -> Dict[str, Any]:
    """Get detailed snapshot information.

    Args:
        snapshot: Snapshot name

    Returns:
        Full snapshot metadata

    Examples:
        >>> info = folio.get_snapshot_info('v1.0')
        >>> print(info['git']['commit'])
        a3f2b8c
        >>> print(info['metadata_snapshot']['accuracy'])
        0.89
    """
```

#### reproduce_instructions()

```python
def reproduce_instructions(
    self,
    snapshot: Optional[str] = None
) -> str:
    """Get human-readable reproduction instructions.

    Args:
        snapshot: Snapshot name (if loaded via load_snapshot, uses that)

    Returns:
        Formatted string with reproduction steps

    Examples:
        >>> folio = DataFolio.load_snapshot('research/exp', 'v1.0')
        >>> print(folio.reproduce_instructions())

        To reproduce snapshot 'v1.0':

        1. Restore code:
           git clone https://github.com/lab/protein-analysis.git
           cd protein-analysis
           git checkout a3f2b8c

        2. Restore environment:
           python --version  # Should be 3.11.5
           uv sync

        3. Run training:
           python train.py --config config.json

        4. Results should match:
           - accuracy: 0.89
           - f1_score: 0.87
    """
```

### 6.3 Modified Existing Methods

#### add_*() methods with overwrite

**Behavior change:** When `overwrite=True` and item is in a snapshot, create new version instead of replacing.

```python
# Before snapshots:
folio.add_model('m', v1)
folio.add_model('m', v2, overwrite=True)  # Replaces m.joblib

# After snapshots:
folio.add_model('m', v1)
folio.create_snapshot('s1')
folio.add_model('m', v2, overwrite=True)  # Creates m_v2.joblib, keeps m.joblib
```

**No API changes required** - versioning happens automatically behind the scenes.

---

## 7. CLI Design

### 7.1 Motivation

A command-line interface for snapshots provides several key benefits:

1. **Natural git-like workflow** - Snapshots feel like version control operations
2. **Better UX for warnings** - Interactive prompts and colored output for uncommitted changes
3. **Scripting and automation** - Easy to integrate into shell scripts and CI/CD
4. **Separation of concerns** - Research code (Python API) vs. metadata management (CLI)
5. **Discoverability** - `datafolio snapshot create v1.0` is more obvious than remembering Python API

**Philosophy:** CLI-first for metadata operations, Python API for programmatic access.

### 7.2 Bundle Path Specification

Users need flexible ways to specify which bundle to operate on:

#### Pattern 1: Current Directory (Default - Most Common)

```bash
# Work directly in bundle directory
cd data/v1dd-experiment
datafolio snapshot create v1.0
datafolio snapshot list
datafolio bundle describe
```

**Best for:** Interactive development, when you're working on one experiment.

#### Pattern 2: Global `--folio` Flag

```bash
# Specify bundle from anywhere
datafolio --folio data/v1dd-experiment snapshot create v1.0
datafolio -f data/v1dd-experiment snapshot list

# Short form
datafolio -f data/v1dd-experiment snapshot create v1.0
```

**Best for:** Managing multiple bundles, scripts, automation.

#### Pattern 3: Environment Variable

```bash
# Set once, use many times
export DATAFOLIO_PATH=data/v1dd-experiment

datafolio snapshot create v1.0
datafolio snapshot list
```

**Best for:** CI/CD pipelines, batch processing.

#### Precedence Order

```
1. --folio / -f flag (highest priority)
2. DATAFOLIO_PATH environment variable
3. Current directory (default)
```

**Example:**

```bash
cd /path/to/experiment1
export DATAFOLIO_PATH=/path/to/experiment2

# Uses current directory (experiment1)
datafolio snapshot list

# Uses environment variable (experiment2) - WRONG, flag overrides!
# Actually uses experiment3 because flag has highest priority
datafolio --folio /path/to/experiment3 snapshot list
```

### 7.3 Command Structure

```bash
datafolio [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGS]

# Global options (before command)
  -b, --folio PATH    Path to bundle directory
  -C PATH              Change to directory (git-style)
  --help               Show help
  --version            Show version

# Commands
  init                 Initialize new bundle
  snapshot             Snapshot management (subcommand group)
  bundle               Bundle operations (describe, validate)
  list                 List bundles in directory
```

### 7.4 Snapshot Commands

#### create

```bash
datafolio snapshot create NAME [OPTIONS]

# Create snapshot with description
datafolio snapshot create v1.0 -m "Baseline random forest model"

# Create with full metadata
datafolio snapshot create v1.0 \
  --message "NeurIPS 2025 submission" \
  --tag paper --tag neurips \
  --entry-point "python train.py --config config.json" \
  --commit

# Interactive mode (prompts for all fields)
datafolio snapshot create v1.0 --interactive

Options:
  -m, --message TEXT       Snapshot description
  --tag TEXT               Tag for organization (multiple allowed)
  --entry-point TEXT       Command to reproduce results
  --commit                 Auto-commit snapshot metadata to git
  --interactive            Prompt for all fields
  --no-git                 Skip git capture
  --no-env                 Skip environment capture
```

#### list

```bash
datafolio snapshot list [OPTIONS]

# List all snapshots
datafolio snapshot list

# Filter by tag
datafolio snapshot list --tag paper

# Sort by different fields
datafolio snapshot list --sort-by name
datafolio snapshot list --sort-by timestamp

Options:
  --tag TEXT              Filter by tag
  --sort-by [name|timestamp]  Sort order (default: timestamp)
  --format [table|json|yaml]  Output format
```

#### show

```bash
datafolio snapshot show NAME [OPTIONS]

# Show snapshot details
datafolio snapshot show v1.0

# Show reproduction instructions
datafolio snapshot show v1.0 --reproduce

Options:
  --reproduce            Show reproduction instructions
  --format [table|json|yaml]
```

#### compare

```bash
datafolio snapshot compare SNAPSHOT1 SNAPSHOT2

# Compare two snapshots
datafolio snapshot compare v1.0 v2.0

# Output shows:
# - Metadata changes
# - Added/removed/modified items
# - Git commit differences
```

#### diff

```bash
datafolio snapshot diff [SNAPSHOT]

# Show what changed since last snapshot
datafolio snapshot diff

# Show what changed since specific snapshot
datafolio snapshot diff v1.0

# Similar to 'git status' for snapshots
```

#### status

```bash
datafolio snapshot status

# Show current state
# Output:
#   Current bundle: /path/to/bundle
#   Last snapshot: v2.0 (2025-11-20)
#
#   Items modified since v2.0:
#     M  classifier (model)
#     M  config (json)
#   Unchanged items:
#          train_data (table)
#
#   Git status: dirty
#     M train.py
#     M models/neural.py
```

#### delete

```bash
datafolio snapshot delete NAME [OPTIONS]

# Delete snapshot
datafolio snapshot delete v1.0

# Delete and cleanup orphaned files
datafolio snapshot delete v1.0 --cleanup

Options:
  --cleanup              Remove orphaned item versions
  --force                Skip confirmation prompt
```

#### restore

```bash
datafolio snapshot restore NAME [OPTIONS]

# Restore working state to snapshot
datafolio snapshot restore v1.0 --confirm

Options:
  --confirm              Required safety flag
```

#### gc

```bash
datafolio snapshot gc [OPTIONS]

# Garbage collect orphaned versions
datafolio snapshot gc

# Dry run (show what would be deleted)
datafolio snapshot gc --dry-run

Options:
  --dry-run              Show what would be deleted
```

#### reproduce

```bash
datafolio snapshot reproduce NAME

# Show reproduction instructions
datafolio snapshot reproduce v1.0

# Output:
#   To reproduce snapshot 'v1.0':
#
#   1. Restore code:
#      git checkout abc123
#
#   2. Restore environment:
#      python --version  # Should be 3.11.5
#      uv sync
#
#   3. Run training:
#      python train.py --config config.json
#
#   4. Verify results:
#      Expected accuracy: 0.89
```

### 7.5 Bundle Commands

```bash
# Initialize new bundle
datafolio init PATH [OPTIONS]
datafolio init data/new-experiment --description "My experiment"

# Describe bundle
datafolio describe
datafolio --folio data/exp1 describe

# Validate bundle integrity
datafolio validate
datafolio --folio data/exp1 validate

# List bundles in directory
datafolio list --search data/
```

### 7.6 Implementation

**Technology stack:**
- **Click** - Python CLI framework (declarative, well-documented)
- **Rich** - Terminal formatting (colors, tables, progress bars)
- **Pygments** - Syntax highlighting for code/diffs

**File structure:**

```
src/datafolio/
├── cli/
│   ├── __init__.py
│   ├── main.py              # Entry point, global options
│   ├── snapshot.py          # Snapshot commands
│   ├── bundle.py            # Bundle commands
│   ├── utils.py             # Formatting, prompts, colors
│   └── validators.py        # Input validation
```

**Entry point in pyproject.toml:**

```toml
[project.scripts]
datafolio = "datafolio.cli.main:cli"
```

**Core implementation:**

```python
# src/datafolio/cli/main.py
import click
import os
from pathlib import Path
from rich.console import Console

console = Console()

@click.group()
@click.option(
    '-b', '--folio',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    envvar='DATAFOLIO_PATH',
    default='.',
    help='Path to bundle directory'
)
@click.option(
    '-C',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Change to directory (git-style)'
)
@click.version_option()
@click.pass_context
def cli(ctx, bundle, c):
    """DataFolio: Manage ML experiment bundles.

    Examples:

        # Work in current directory
        cd data/experiment
        datafolio snapshot create v1.0

        # Specify bundle path
        datafolio --folio data/experiment snapshot list

        # Git-style
        datafolio -C data/experiment snapshot create v1.0
    """
    # Use -C if provided (git-style), otherwise use --folio
    bundle_path = Path(c if c else bundle).resolve()

    # Check if it's a valid bundle
    if not (bundle_path / 'items.json').exists():
        console.print(
            f"[red]Error:[/red] '{bundle_path}' is not a DataFolio bundle"
        )
        console.print(f"[cyan]Tip:[/cyan] Initialize with: datafolio init {bundle_path}")
        ctx.exit(1)

    # Store in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['bundle_path'] = bundle_path
    ctx.obj['console'] = console


# Import subcommands
from datafolio.cli import snapshot, bundle

cli.add_command(snapshot.snapshot)
cli.add_command(bundle.bundle)
```

```python
# src/datafolio/cli/snapshot.py
import click
from rich.console import Console
from rich.table import Table
from datafolio import DataFolio

@click.group()
def snapshot():
    """Manage bundle snapshots."""
    pass


@snapshot.command('create')
@click.argument('name')
@click.option('-m', '--message', help='Snapshot description')
@click.option('--tag', multiple=True, help='Tags (can be used multiple times)')
@click.option('--entry-point', help='Command to reproduce')
@click.option('--commit', is_flag=True, help='Auto-commit to git')
@click.option('--interactive', is_flag=True, help='Interactive mode')
@click.pass_context
def create(ctx, name, message, tag, entry_point, commit, interactive):
    """Create a new snapshot.

    Examples:

        # Basic snapshot
        datafolio snapshot create v1.0 -m "Baseline model"

        # With metadata
        datafolio snapshot create v1.0 \\
          --message "NeurIPS submission" \\
          --tag paper --tag neurips \\
          --entry-point "python train.py"

        # Interactive mode
        datafolio snapshot create v1.0 --interactive
    """
    bundle_path = ctx.obj['bundle_path']
    console = ctx.obj['console']

    # Interactive prompts
    if interactive:
        if not message:
            message = click.prompt('Description')
        if not entry_point:
            entry_point = click.prompt('Entry point (command to reproduce)', default='')
        tag_input = click.prompt('Tags (comma-separated)', default='')
        tag = tuple(t.strip() for t in tag_input.split(',') if t.strip())

    console.print(f"[dim]Creating snapshot in:[/dim] {bundle_path}")

    # Load bundle
    folio = DataFolio(bundle_path)

    # Check git status
    git_info = folio._capture_git_info()
    if git_info and git_info['dirty']:
        console.print("\n[yellow]⚠ Warning:[/yellow] Git has uncommitted changes")
        console.print("  Consider committing your changes before creating a snapshot")

        if not click.confirm('\nContinue anyway?', default=False):
            console.print("[red]Snapshot cancelled[/red]")
            ctx.exit(1)

    # Create snapshot
    try:
        folio.create_snapshot(
            name,
            description=message,
            tags=list(tag) if tag else None,
            entry_point=entry_point or None
        )

        console.print(f"\n[green]✓ Snapshot '{name}' created[/green]")

        if git_info:
            console.print(f"  Git: {git_info['commit_short']}", style="dim")
            if git_info['dirty']:
                console.print("  Status: [yellow]dirty[/yellow]")

        # Auto-commit
        if commit:
            import subprocess
            import os

            os.chdir(bundle_path)
            subprocess.run(['git', 'add', 'snapshots.json', 'items.json'])
            result = subprocess.run([
                'git', 'commit', '-m',
                f"Snapshot {name}\n\n{message or 'Created snapshot'}"
            ])

            if result.returncode == 0:
                console.print("[green]✓ Snapshot metadata committed[/green]")
            else:
                console.print("[red]✗ Failed to commit[/red]")
        else:
            console.print("\n[cyan]To commit snapshot metadata:[/cyan]")
            console.print(f"  cd {bundle_path}")
            console.print(f"  git add snapshots.json items.json")
            console.print(f"  git commit -m 'Snapshot {name}'")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        ctx.exit(1)


@snapshot.command('list')
@click.option('--tag', help='Filter by tag')
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
@click.pass_context
def list_snapshots(ctx, tag, format):
    """List all snapshots."""
    bundle_path = ctx.obj['bundle_path']
    console = ctx.obj['console']

    folio = DataFolio(bundle_path)
    snapshots = folio.list_snapshots(tags=[tag] if tag else None)

    if not snapshots:
        console.print("[yellow]No snapshots found[/yellow]")
        return

    if format == 'json':
        import json
        console.print(json.dumps(snapshots, indent=2))
        return

    # Table format
    table = Table(title="Snapshots")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Created", style="dim")
    table.add_column("Tags", style="magenta")

    for snap in snapshots:
        table.add_row(
            snap['name'],
            snap.get('description', ''),
            snap['timestamp'][:10],  # Just date
            ', '.join(snap.get('tags', []))
        )

    console.print(table)


@snapshot.command('show')
@click.argument('name')
@click.option('--reproduce', is_flag=True, help='Show reproduction instructions')
@click.pass_context
def show(ctx, name, reproduce):
    """Show snapshot details."""
    bundle_path = ctx.obj['bundle_path']
    console = ctx.obj['console']

    folio = DataFolio(bundle_path)

    if reproduce:
        instructions = folio.reproduce_instructions(snapshot=name)
        console.print(instructions)
    else:
        info = folio.get_snapshot_info(name)

        console.print(f"\n[bold]{info['name']}[/bold]")
        console.print(f"[dim]{info.get('description', 'No description')}[/dim]\n")
        console.print(f"Created: {info['timestamp']}")

        if 'git' in info:
            console.print(f"\nGit:")
            console.print(f"  Commit: {info['git']['commit_short']}")
            console.print(f"  Branch: {info['git']['branch']}")
            if info['git']['dirty']:
                console.print(f"  [yellow]Status: dirty[/yellow]")

        if 'execution' in info:
            console.print(f"\nReproduction:")
            console.print(f"  {info['execution']['entry_point']}")

        console.print(f"\nItems ({len(info['item_versions'])}):")
        for item_name, version in info['item_versions'].items():
            console.print(f"  {item_name} (v{version})")


@snapshot.command('status')
@click.pass_context
def status(ctx):
    """Show current bundle state (like git status)."""
    bundle_path = ctx.obj['bundle_path']
    console = ctx.obj['console']

    folio = DataFolio(bundle_path)

    console.print(f"[bold]Current bundle:[/bold] {bundle_path}")

    # Last snapshot
    snapshots = folio.list_snapshots()
    if snapshots:
        last = snapshots[-1]
        console.print(f"[bold]Last snapshot:[/bold] {last['name']} ({last['timestamp'][:10]})")

        # Compare current to last
        # (Would need to implement this comparison)
        console.print("\n[bold]Changes since last snapshot:[/bold]")
        console.print("  [yellow]M[/yellow]  classifier (model)")
        console.print("  [green]A[/green]  new_config (json)")
    else:
        console.print("[yellow]No snapshots yet[/yellow]")

    # Git status
    git_info = folio._capture_git_info()
    if git_info:
        console.print(f"\n[bold]Git status:[/bold]")
        if git_info['dirty']:
            console.print("  [yellow]dirty[/yellow]")
            for file in git_info['uncommitted_files']:
                console.print(f"    [yellow]M[/yellow] {file}")
        else:
            console.print("  [green]clean[/green]")
```

### 7.7 Usage Examples

#### Interactive Development

```bash
# Navigate to experiment
cd ~/research/protein-analysis

# Run experiment
python train.py

# Create snapshot
datafolio snapshot create v1.0 \
  -m "Baseline random forest" \
  --tag baseline \
  --entry-point "python train.py --config baseline.json"

# Output:
# Creating snapshot in: /Users/me/research/protein-analysis
#
# ✓ Snapshot 'v1.0' created
#   Git: abc123 (clean)
#
# To commit snapshot metadata:
#   cd /Users/me/research/protein-analysis
#   git add snapshots.json items.json
#   git commit -m 'Snapshot v1.0'

# List snapshots
datafolio snapshot list

# Continue experimenting
python train_v2.py
datafolio snapshot create v2.0 -m "Neural network" --tag experimental
```

#### Managing Multiple Bundles

```bash
# From central location
cd ~/research

# Create snapshots in different bundles
datafolio -b protein-analysis snapshot create v1.0 -m "Protein baseline"
datafolio -b genomics-study snapshot create v1.0 -m "Genomics baseline"

# Compare
datafolio -b protein-analysis snapshot list
datafolio -b genomics-study snapshot list
```

#### Scripting

```bash
#!/bin/bash
# run_experiments.sh

BUNDLE_DIR="experiments/$(date +%Y%m%d)"

# Initialize bundle
datafolio init "$BUNDLE_DIR" --description "Daily experiment"

# Set environment variable
export DATAFOLIO_PATH="$BUNDLE_DIR"

# Run experiment
cd "$BUNDLE_DIR"
python ../train.py --config config.json

# Create snapshot with auto-commit
datafolio snapshot create "run-$(date +%H%M%S)" \
  --message "Automated experiment run" \
  --tag automated \
  --entry-point "python train.py --config config.json" \
  --commit

echo "Results saved to: $BUNDLE_DIR"
```

#### CI/CD

```yaml
# .github/workflows/experiment.yml
name: Run Experiment

on: [push]

jobs:
  experiment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install datafolio
          uv sync

      - name: Run experiment
        run: python train.py

      - name: Create snapshot
        env:
          DATAFOLIO_PATH: experiments/ci-${{ github.run_id }}
        run: |
          datafolio snapshot create "ci-${{ github.sha }}" \
            --message "CI run from commit ${{ github.sha }}" \
            --tag ci --tag automated \
            --entry-point "python train.py" \
            --commit
```

### 7.8 Help Output

```bash
$ datafolio --help

Usage: datafolio [OPTIONS] COMMAND [ARGS]...

  DataFolio: Manage ML experiment bundles.

  Examples:

      # Work in current directory
      cd data/experiment
      datafolio snapshot create v1.0

      # Specify bundle path
      datafolio --folio data/experiment snapshot list

      # Git-style
      datafolio -C data/experiment snapshot create v1.0

Options:
  -b, --folio PATH  Path to bundle directory [env: DATAFOLIO_PATH]
  -C PATH            Change to directory (git-style)
  --version          Show version
  --help             Show this message

Commands:
  init      Initialize new bundle
  snapshot  Manage bundle snapshots
  bundle    Bundle operations
  list      List bundles in directory


$ datafolio snapshot --help

Usage: datafolio snapshot [OPTIONS] COMMAND [ARGS]...

  Manage bundle snapshots.

Commands:
  create    Create new snapshot
  list      List all snapshots
  show      Show snapshot details
  compare   Compare two snapshots
  diff      Show changes since snapshot
  status    Show current state
  delete    Delete snapshot
  restore   Restore to snapshot state
  gc        Garbage collect orphaned versions
  reproduce Show reproduction instructions
```

### 7.9 Error Handling and User Experience

**Validation errors:**

```bash
$ datafolio snapshot create v1.0
Error: Snapshot 'v1.0' already exists. Snapshots are immutable.

Tip: Use a different name or view existing snapshot:
  datafolio snapshot show v1.0
```

**Git warnings:**

```bash
$ datafolio snapshot create v2.0 -m "Experimental model"

⚠ Warning: Git has uncommitted changes:
  M train.py
  M models/neural.py

These changes will not be captured in snapshot git reference.
Continue anyway? [y/N]: n

Snapshot cancelled

Tip: Commit code first for full reproducibility:
  git add train.py models/neural.py
  git commit -m "Training improvements"
  datafolio snapshot create v2.0
```

**Not in bundle directory:**

```bash
$ cd /tmp
$ datafolio snapshot list

Error: '/tmp' is not a DataFolio bundle

Tip: Initialize with: datafolio init /tmp
     Or navigate to bundle: cd /path/to/bundle
```

---

## 8. Implementation Plan

### Phase 1: Core Data Structures (Week 1)

**Goal:** Extend items.json with versioning, create snapshots.json

**Tasks:**
1. Add version tracking to items.json
   - Add `version`, `in_snapshots`, `replaces_version` fields
   - Add `current_versions` top-level key
   - Update TypedDict definitions in utils.py

2. Create snapshots.json structure
   - Define SnapshotMetadata TypedDict
   - Create SnapshotRegistry class to manage snapshots
   - Implement save/load for snapshots.json

3. Update DataFolio._load_manifests()
   - Load both items.json and snapshots.json
   - Build version index for fast lookups

**Deliverables:**
- [ ] Extended items.json with version fields
- [ ] snapshots.json structure defined
- [ ] Tests for manifest loading with versions

**Validation:**
- Existing bundles load correctly (backward compatible)
- New version fields serialize/deserialize properly

### Phase 2: Version Management (Week 1-2)

**Goal:** Implement copy-on-write versioning for items

**Tasks:**
1. Implement `_create_new_version()` method
   - Generate versioned filename (e.g., `classifier_v2.joblib`)
   - Write new version using handler
   - Update items list with new version entry
   - Track which version is current

2. Update `add_*()` methods
   - Check if item is in any snapshot before overwriting
   - Call `_create_new_version()` if needed
   - Otherwise replace normally

3. Implement `_get_item_by_version()` method
   - Look up specific version of an item
   - Used by snapshot loading

4. Add version tracking to handlers
   - Handlers accept optional `filename` parameter
   - Used to create versioned filenames

**Deliverables:**
- [ ] Versioning logic implemented
- [ ] All add_*() methods support versioning
- [ ] Tests for overwrite with/without snapshots

**Validation:**
- `overwrite=True` replaces file when no snapshot
- `overwrite=True` creates new version when in snapshot
- Old versions are preserved

### Phase 3: Snapshot Creation (Week 2)

**Goal:** Implement create_snapshot() with context capture

**Tasks:**
1. Implement core `create_snapshot()` method
   - Capture current item versions
   - Snapshot metadata state
   - Generate snapshot metadata

2. Implement `_capture_git_info()`
   - Use subprocess to call git commands
   - Get commit, branch, remote, dirty status
   - Handle non-git repositories gracefully

3. Implement `_capture_env_info()`
   - Get Python version
   - Hash uv.lock or requirements.txt
   - Capture platform info

4. Update items on snapshot creation
   - Add snapshot name to `in_snapshots` list for each item

**Deliverables:**
- [ ] create_snapshot() working
- [ ] Git capture implemented
- [ ] Environment capture implemented
- [ ] Tests for snapshot creation

**Validation:**
- Snapshots are immutable
- Git info captured correctly
- Warnings shown if repo is dirty
- Works without git (capture_git=False)

### Phase 4: Snapshot Loading (Week 3)

**Goal:** Implement load_snapshot() to restore snapshot state

**Tasks:**
1. Implement `load_snapshot()` class method
   - Load bundle normally
   - Read snapshot metadata
   - Set `_current_versions` to snapshot versions
   - Restore metadata state

2. Add snapshot mode flag
   - `_snapshot_mode = True` when loaded from snapshot
   - `_loaded_snapshot = name` to track which snapshot

3. Make snapshot mode read-only (optional)
   - Prevent add/delete operations in snapshot mode
   - Or allow but warn

4. Implement `get_*()` methods with version awareness
   - Use version from `_current_versions`
   - Look up correct item version in items list

**Deliverables:**
- [ ] load_snapshot() working
- [ ] Snapshot mode implemented
- [ ] All get_*() methods work with snapshots
- [ ] Tests for snapshot loading

**Validation:**
- load_snapshot() restores exact state
- Metadata matches snapshot
- get_*() returns correct versions
- Can load multiple snapshots simultaneously

### Phase 5: Snapshot Management (Week 3-4)

**Goal:** Implement list, delete, compare, cleanup operations

**Tasks:**
1. Implement `list_snapshots()`
   - Return sorted list of snapshots
   - Support tag filtering
   - Include summary info

2. Implement `delete_snapshot()`
   - Remove from snapshots.json
   - Remove from `in_snapshots` in items
   - Optionally cleanup orphans

3. Implement `compare_snapshots()`
   - Compare item versions
   - Compare metadata
   - Return diff structure

4. Implement `cleanup_orphaned_versions()`
   - Find items not in any snapshot and not current
   - Delete files and remove from items.json
   - Support dry_run mode

5. Implement `restore_snapshot()`
   - Copy snapshot state to current
   - Require confirmation flag

**Deliverables:**
- [ ] All management methods implemented
- [ ] Tests for each operation
- [ ] Integration tests for workflows

**Validation:**
- Can list and filter snapshots
- Can delete snapshots safely
- Compare shows correct diffs
- Cleanup only deletes orphans

### Phase 6: Utility Methods (Week 4)

**Goal:** Implement helper methods for usability

**Tasks:**
1. Implement `get_snapshot_info()`
   - Return full snapshot metadata
   - Format for display

2. Implement `reproduce_instructions()`
   - Generate human-readable steps
   - Include git, environment, execution info
   - Format as markdown or plain text

3. Add `describe()` integration
   - Show snapshots in bundle description
   - List which snapshot items belong to

4. Add snapshot metadata to `__repr__()`
   - Show number of snapshots
   - Show current vs snapshot mode

**Deliverables:**
- [ ] Helper methods implemented
- [ ] Documentation in docstrings
- [ ] Examples in docstrings

**Validation:**
- Methods produce useful output
- Instructions are clear and actionable

### Phase 7: CLI Tool (Week 5)

**Goal:** Implement command-line interface for snapshot management

**Tasks:**
1. Set up Click framework
   - Create cli/ package structure
   - Set up entry point in pyproject.toml
   - Add Rich for terminal formatting

2. Implement core CLI infrastructure
   - Global options (--folio, -C)
   - Bundle path resolution (flag > env > current dir)
   - Context passing to subcommands
   - Error handling and formatting

3. Implement snapshot commands
   - `snapshot create` with all options
   - `snapshot list` with filtering
   - `snapshot show` with details
   - `snapshot status` (git status-like)
   - `snapshot compare` for diffs
   - `snapshot delete` with confirmations
   - `snapshot gc` for cleanup
   - `snapshot reproduce` for instructions

4. Implement bundle commands
   - `init` for new bundles
   - `describe` for bundle info
   - `validate` for integrity checks
   - `list` for finding bundles

5. Add interactive features
   - Colored output (success=green, error=red, warning=yellow)
   - Progress indicators for long operations
   - Interactive prompts (--interactive mode)
   - Confirmation prompts for destructive operations
   - Table formatting for list outputs

6. Git integration
   - Auto-commit with --commit flag
   - Git status warnings
   - Reproduction instructions with git checkout

**Deliverables:**
- [ ] CLI package structure created
- [ ] All snapshot commands implemented
- [ ] Bundle management commands
- [ ] Rich formatting and colors
- [ ] Interactive mode support
- [ ] Git integration working
- [ ] Help text for all commands
- [ ] Tests for CLI commands

**Validation:**
- CLI commands work from any directory
- Bundle path resolution correct
- Git warnings show properly
- Auto-commit works
- Error messages are helpful
- Help output is clear

### Phase 8: Testing & Documentation (Week 6)

**Goal:** Comprehensive tests and documentation

**Tasks:**
1. Unit tests
   - Test each method independently
   - Test error cases
   - Test edge cases (empty snapshots, etc.)

2. Integration tests
   - Test full workflows (create, load, compare)
   - Test paper submission scenario
   - Test A/B testing scenario

3. Documentation
   - Update ARCHITECTURE.md with snapshot design
   - Create examples/ directory with notebooks
   - Update README.md with snapshot feature

4. Migration guide
   - How existing bundles work (no changes needed)
   - How to start using snapshots
   - Best practices

**Deliverables:**
- [ ] Test coverage > 90% for snapshot code
- [ ] All workflows tested
- [ ] Documentation complete
- [ ] Example notebooks

**Validation:**
- All tests pass
- Examples run successfully
- Documentation is clear

---

## 9. Testing Strategy

### 8.1 Unit Tests

**File:** `tests/test_snapshots.py`

```python
class TestSnapshotCreation:
    def test_create_snapshot_basic(tmp_path):
        """Test basic snapshot creation."""
        folio = DataFolio(tmp_path / 'test')
        folio.add_table('data', df)
        folio.create_snapshot('v1.0', 'Initial version')

        assert 'v1.0' in folio._snapshots
        assert folio._snapshots['v1.0']['description'] == 'Initial version'

    def test_create_snapshot_duplicate_fails(tmp_path):
        """Test that duplicate snapshot names fail."""
        folio = DataFolio(tmp_path / 'test')
        folio.create_snapshot('v1.0')

        with pytest.raises(ValueError, match='already exists'):
            folio.create_snapshot('v1.0')

    def test_create_snapshot_captures_git(tmp_path):
        """Test git info capture."""
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=tmp_path)
        subprocess.run(['git', 'add', '.'], cwd=tmp_path)
        subprocess.run(['git', 'commit', '-m', 'Initial'], cwd=tmp_path)

        folio = DataFolio(tmp_path / 'test')
        folio.create_snapshot('v1.0')

        assert 'git' in folio._snapshots['v1.0']
        assert 'commit' in folio._snapshots['v1.0']['git']
        assert folio._snapshots['v1.0']['git']['dirty'] == False

class TestVersioning:
    def test_overwrite_without_snapshot_replaces(tmp_path):
        """Test that overwrite replaces file when no snapshot."""
        folio = DataFolio(tmp_path / 'test')
        folio.add_model('m', model_v1)
        folio.add_model('m', model_v2, overwrite=True)

        # Only one version exists
        items = [i for i in folio._items if i['name'] == 'm']
        assert len(items) == 1

    def test_overwrite_with_snapshot_creates_version(tmp_path):
        """Test that overwrite creates version when in snapshot."""
        folio = DataFolio(tmp_path / 'test')
        folio.add_model('m', model_v1)
        folio.create_snapshot('v1.0')
        folio.add_model('m', model_v2, overwrite=True)

        # Two versions exist
        items = [i for i in folio._items if i['name'] == 'm']
        assert len(items) == 2
        assert items[0]['version'] == 1
        assert items[1]['version'] == 2

class TestSnapshotLoading:
    def test_load_snapshot_restores_state(tmp_path):
        """Test loading snapshot restores exact state."""
        # Create bundle with snapshot
        folio = DataFolio(tmp_path / 'test')
        folio.add_table('data', df_v1)
        folio.metadata['accuracy'] = 0.89
        folio.create_snapshot('v1.0')

        # Modify
        folio.add_table('data', df_v2, overwrite=True)
        folio.metadata['accuracy'] = 0.91

        # Load snapshot
        snap_folio = DataFolio.load_snapshot(tmp_path / 'test', 'v1.0')

        # Verify state
        assert snap_folio.metadata['accuracy'] == 0.89
        loaded_df = snap_folio.get_table('data')
        pd.testing.assert_frame_equal(loaded_df, df_v1)
```

### 8.2 Integration Tests

**File:** `tests/test_snapshots_integration.py`

```python
def test_paper_submission_workflow(tmp_path):
    """Test realistic paper submission scenario."""
    # Initial research
    folio = DataFolio(tmp_path / 'protein-analysis')
    folio.add_table('train_data', train_df)
    folio.add_table('test_data', test_df)
    folio.add_model('classifier', model_v1)
    folio.metadata['accuracy'] = 0.89

    # Snapshot for paper
    folio.create_snapshot(
        'neurips-2025-submission',
        description='Paper submission',
        entry_point='python train.py'
    )

    # Continue research
    folio.add_model('classifier', model_v2, overwrite=True)
    folio.metadata['accuracy'] = 0.91
    folio.create_snapshot('post-review')

    # Load paper version
    paper = DataFolio.load_snapshot(tmp_path / 'protein-analysis', 'neurips-2025-submission')

    # Verify exact state
    assert paper.metadata['accuracy'] == 0.89
    loaded_model = paper.get_model('classifier')
    assert type(loaded_model) == type(model_v1)

    # Verify files shared
    # train_data should be same file for both snapshots
    train_item = [i for i in folio._items if i['name'] == 'train_data'][0]
    assert 'neurips-2025-submission' in train_item['in_snapshots']
    assert 'post-review' in train_item['in_snapshots']

def test_ab_testing_workflow(tmp_path):
    """Test A/B testing with multiple snapshots."""
    folio = DataFolio(tmp_path / 'recommender')

    # Baseline
    folio.add_model('model', baseline)
    folio.metadata['p95_latency'] = 50
    folio.create_snapshot('v1.0-baseline')

    # Experiment
    folio.add_model('model', experimental, overwrite=True)
    folio.metadata['p95_latency'] = 45
    folio.create_snapshot('v2.0-experimental')

    # Load both
    baseline_folio = DataFolio.load_snapshot(tmp_path / 'recommender', 'v1.0-baseline')
    experimental_folio = DataFolio.load_snapshot(tmp_path / 'recommender', 'v2.0-experimental')

    # Compare
    assert baseline_folio.metadata['p95_latency'] == 50
    assert experimental_folio.metadata['p95_latency'] == 45
```

### 8.3 Edge Cases

```python
def test_snapshot_empty_bundle(tmp_path):
    """Test snapshot of empty bundle."""
    folio = DataFolio(tmp_path / 'test')
    folio.create_snapshot('empty')

    snap = DataFolio.load_snapshot(tmp_path / 'test', 'empty')
    assert len(snap._items) == 0

def test_delete_item_in_snapshot_fails(tmp_path):
    """Test deleting item referenced by snapshot fails."""
    folio = DataFolio(tmp_path / 'test')
    folio.add_table('data', df)
    folio.create_snapshot('v1.0')

    with pytest.raises(ValueError, match='referenced by snapshot'):
        folio.delete('data')

def test_cleanup_preserves_snapshot_items(tmp_path):
    """Test cleanup doesn't delete snapshot items."""
    folio = DataFolio(tmp_path / 'test')
    folio.add_model('m', v1)
    folio.create_snapshot('v1.0')
    folio.add_model('m', v2, overwrite=True)
    folio.add_model('m', v3, overwrite=True)  # Current

    # v2 is orphan (not in snapshot, not current)
    deleted = folio.cleanup_orphaned_versions()
    assert 'm_v2.joblib' in deleted
    assert 'm.joblib' not in deleted  # v1 in snapshot
    assert 'm_v3.joblib' not in deleted  # v3 is current
```

### 9.4 CLI Tests

**File:** `tests/test_cli_snapshots.py`

```python
from click.testing import CliRunner
from datafolio.cli.main import cli

def test_snapshot_create_basic(tmp_path):
    """Test basic snapshot creation via CLI."""
    # Setup bundle
    folio = DataFolio(tmp_path / 'test')
    folio.add_table('data', df)

    # Run CLI command
    runner = CliRunner()
    result = runner.invoke(cli, [
        '--folio', str(tmp_path / 'test'),
        'snapshot', 'create', 'v1.0',
        '-m', 'Test snapshot'
    ])

    assert result.exit_code == 0
    assert 'Snapshot \'v1.0\' created' in result.output

def test_snapshot_list(tmp_path):
    """Test listing snapshots via CLI."""
    folio = DataFolio(tmp_path / 'test')
    folio.create_snapshot('v1.0', 'First')
    folio.create_snapshot('v2.0', 'Second')

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-C', str(tmp_path / 'test'),
        'snapshot', 'list'
    ])

    assert result.exit_code == 0
    assert 'v1.0' in result.output
    assert 'v2.0' in result.output

def test_git_warning(tmp_path):
    """Test that CLI warns about uncommitted changes."""
    # Initialize git repo with changes
    subprocess.run(['git', 'init'], cwd=tmp_path)
    Path(tmp_path / 'test.py').write_text('# test')

    folio = DataFolio(tmp_path / 'test')
    folio.add_table('data', df)

    runner = CliRunner()
    result = runner.invoke(cli, [
        '--folio', str(tmp_path / 'test'),
        'snapshot', 'create', 'v1.0'
    ], input='n\n')  # Respond 'no' to confirmation

    assert 'uncommitted changes' in result.output.lower()
    assert result.exit_code == 1

def test_bundle_path_precedence(tmp_path):
    """Test bundle path resolution precedence."""
    # Create bundles
    bundle1 = tmp_path / 'bundle1'
    bundle2 = tmp_path / 'bundle2'

    DataFolio(bundle1).add_table('data', df)
    DataFolio(bundle2).add_table('data', df)

    # Flag should override env var
    runner = CliRunner()
    result = runner.invoke(cli, [
        '--folio', str(bundle1),
        'snapshot', 'list'
    ], env={'DATAFOLIO_PATH': str(bundle2)})

    # Should use bundle1 (flag has priority)
    assert result.exit_code == 0
```

---

## 10. Migration Path

### 9.1 Backward Compatibility

**Existing bundles work unchanged:**
- No snapshots.json → No snapshots, normal operation
- items.json without version fields → Auto-upgrade on first write
- All existing APIs work exactly as before

**Upgrade path:**
```python
# Open old bundle (pre-snapshots)
folio = DataFolio('old-bundle')

# Use normally - no changes needed
folio.add_table('data', df)

# Start using snapshots
folio.create_snapshot('v1.0')

# items.json auto-upgraded with version fields
# snapshots.json created
```

### 9.2 Data Migration

**Auto-upgrade on first snapshot:**
```python
def create_snapshot(self, name: str, **kwargs) -> None:
    # Check if this is first snapshot
    if not hasattr(self, '_snapshots') or self._snapshots is None:
        self._snapshots = {}

        # Upgrade items to versioned format
        for item in self._items:
            if 'version' not in item:
                item['version'] = 1
                item['in_snapshots'] = []

        # Initialize current_versions
        self._current_versions = {
            item['name']: item['version']
            for item in self._items
        }

    # Continue with snapshot creation...
```

### 9.3 Best Practices

**When to create snapshots:**
- ✅ Before major experiments
- ✅ After achieving good results
- ✅ Before paper submission
- ✅ Before deploying to production
- ❌ After every single model iteration (too many snapshots)

**Naming conventions:**
- Semantic versioning: `v1.0.0`, `v1.1.0`, `v2.0.0`
- Date-based: `2025-11-20-baseline`, `2025-12-15-final`
- Milestone-based: `paper-submission`, `production-v1`, `baseline`
- Avoid: `final`, `final2`, `final-final` (use versions instead!)

**Cleanup strategy:**
- Keep all snapshots for active experiments
- Delete experimental snapshots that didn't work out
- Archive old snapshots to separate storage
- Run cleanup periodically: `folio.cleanup_orphaned_versions()`

---

## Appendix A: Example Workflows

### Workflow 1: Model Development Lifecycle

```python
# Week 1: Baseline
folio = DataFolio('experiments/fraud-detection')
folio.add_table('train_data', train_df)
folio.add_model('detector', logistic_regression)
folio.metadata['accuracy'] = 0.85
folio.create_snapshot('v1.0-baseline', 'Logistic regression baseline')

# Week 2: Try random forest
folio.add_model('detector', random_forest, overwrite=True)
folio.metadata['accuracy'] = 0.89
folio.create_snapshot('v2.0-random-forest', 'Improved with random forest')

# Week 3: Try neural network (worse!)
folio.add_model('detector', neural_net, overwrite=True)
folio.metadata['accuracy'] = 0.87
folio.create_snapshot('v3.0-neural-net', 'Experimental neural net')

# Week 4: Deploy best model (v2.0)
prod = DataFolio.load_snapshot('experiments/fraud-detection', 'v2.0-random-forest')
deploy_model(prod.get_model('detector'))

# Continue experimenting without affecting production
folio.add_model('detector', ensemble, overwrite=True)
```

### Workflow 2: Reproducible Research

```python
# September: Research & submission
folio = DataFolio('research/genomics')
folio.add_table('sequences', sequences_df)
folio.add_model('classifier', model)
folio.add_json('parameters', params)
folio.metadata['accuracy'] = 0.92

folio.create_snapshot(
    'nature-2025-submission',
    description='Submitted to Nature Medicine',
    entry_point='python analyze.py --config nature.json',
    tags=['paper', 'nature', 'submitted']
)

# February: Reviewers ask for ablation study
paper = DataFolio.load_snapshot('research/genomics', 'nature-2025-submission')

# Run ablation with original data & model
original_model = paper.get_model('classifier')
original_data = paper.get_table('sequences')

ablation_results = run_ablation(original_model, original_data)

# Save results in new bundle
ablation_folio = DataFolio('research/genomics-ablation')
ablation_folio.add_json('ablation_results', ablation_results)
ablation_folio.add_json('original_params', paper.get_json('parameters'))
ablation_folio.create_snapshot('reviewer-response', 'Ablation study for reviewers')
```

### Workflow 3: Hyperparameter Tuning

```python
# Grid search
folio = DataFolio('experiments/tuning')

best_accuracy = 0
for lr in [0.001, 0.01, 0.1]:
    for depth in [5, 10, 20]:
        model = train_model(lr=lr, max_depth=depth)
        accuracy = evaluate(model)

        folio.add_model('model', model, overwrite=True)
        folio.metadata['lr'] = lr
        folio.metadata['max_depth'] = depth
        folio.metadata['accuracy'] = accuracy

        # Snapshot each config
        folio.create_snapshot(
            f'lr{lr}_depth{depth}',
            description=f'lr={lr}, depth={depth}, acc={accuracy:.3f}'
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = f'lr{lr}_depth{depth}'

# Load best model
best = DataFolio.load_snapshot('experiments/tuning', best_config)
production_model = best.get_model('model')

# Clean up non-best snapshots
for snap in folio.list_snapshots():
    if snap['name'] != best_config:
        folio.delete_snapshot(snap['name'], cleanup_orphans=True)
```

---

## Appendix B: Implementation Checklist

### Phase 1: Core Data Structures
- [ ] Add version fields to TypedDicts
- [ ] Create SnapshotMetadata TypedDict
- [ ] Extend items.json loading
- [ ] Create snapshots.json structure
- [ ] Tests for manifest loading

### Phase 2: Version Management
- [ ] Implement _create_new_version()
- [ ] Update add_*() methods
- [ ] Implement _get_item_by_version()
- [ ] Update handlers for versioned filenames
- [ ] Tests for versioning logic

### Phase 3: Snapshot Creation
- [ ] Implement create_snapshot()
- [ ] Implement _capture_git_info()
- [ ] Implement _capture_env_info()
- [ ] Update items on snapshot creation
- [ ] Tests for snapshot creation
- [ ] Tests for git capture
- [ ] Tests for dirty repo warnings

### Phase 4: Snapshot Loading
- [ ] Implement load_snapshot()
- [ ] Add snapshot mode flag
- [ ] Update get_*() methods
- [ ] Tests for snapshot loading
- [ ] Tests for version restoration

### Phase 5: Snapshot Management
- [ ] Implement list_snapshots()
- [ ] Implement delete_snapshot()
- [ ] Implement compare_snapshots()
- [ ] Implement cleanup_orphaned_versions()
- [ ] Implement restore_snapshot()
- [ ] Tests for each method

### Phase 6: Utility Methods
- [ ] Implement get_snapshot_info()
- [ ] Implement reproduce_instructions()
- [ ] Update describe()
- [ ] Update __repr__()
- [ ] Tests for utilities

### Phase 7: CLI Tool
- [ ] Set up Click + Rich framework
- [ ] Implement global options (--folio, -C)
- [ ] Implement snapshot create command
- [ ] Implement snapshot list command
- [ ] Implement snapshot show command
- [ ] Implement snapshot status command
- [ ] Implement snapshot compare command
- [ ] Implement snapshot delete command
- [ ] Implement snapshot gc command
- [ ] Implement snapshot reproduce command
- [ ] Implement bundle init command
- [ ] Implement bundle describe command
- [ ] Implement bundle validate command
- [ ] Add interactive mode (--interactive)
- [ ] Add git auto-commit (--commit)
- [ ] Add colored output and formatting
- [ ] Tests for CLI commands
- [ ] Help text for all commands

### Phase 8: Testing & Documentation
- [ ] Unit tests (>90% coverage)
- [ ] Integration tests (workflows)
- [ ] Edge case tests
- [ ] Update ARCHITECTURE.md
- [ ] Create example notebooks
- [ ] Update README.md
- [ ] Migration guide

### Phase 8: Release
- [ ] Code review
- [ ] Performance testing
- [ ] Documentation review
- [ ] Create v0.6 release
- [ ] Announce feature

---

## Appendix C: Open Questions

1. **Snapshot size limits?**
   - Should we warn if snapshots.json gets very large?
   - Recommend archiving old snapshots?

2. **Snapshot export/import?**
   - Export snapshot as standalone bundle?
   - Import snapshot from another bundle?

3. **Snapshot diff visualization?**
   - Create visual diff for compare_snapshots()?
   - Integrate with notebooks?

4. **Snapshot metadata search?**
   - Query snapshots by metadata values?
   - E.g., "find all snapshots with accuracy > 0.9"

5. **Partial snapshot restore?**
   - Restore only specific items from snapshot?
   - E.g., "restore just the model from v1.0"

6. **Cloud-optimized storage?**
   - Use cloud-native versioning (S3 versioning)?
   - Leverage object storage features?

---

**End of Document**
