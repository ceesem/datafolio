---
title: CLI
---

# Command line

The CLI is for looking at a folio and managing snapshots without opening a
notebook. It does not add or read data — that is Python's job.

```bash
datafolio --help
datafolio --version
```

## Choosing the folio

Three ways, highest priority first:

```bash
datafolio -f analysis/experiment-12 describe   # explicit flag
export DATAFOLIO_PATH=analysis/experiment-12   # environment variable
cd analysis/experiment-12 && datafolio describe # current directory
```

!!! note "Local folios only"
    `-f` resolves a filesystem path, so cloud folios (`gs://…`, `s3://…`)
    cannot be targeted by `describe`, `validate`, or the snapshot commands.
    `init` is the exception — it can create a cloud folio. Use Python for
    everything else on a cloud folio, or sync it down first.

Commands exit `0` on success and `1` on failure (folio not found, invalid
bundle, unknown snapshot, validation failure).

## `datafolio init`

Create a new folio.

```bash
datafolio init                                   # in the current directory
datafolio init experiments/new-exp -d "March review rerun"
datafolio init gs://my-bucket/experiment -d "Cloud experiment"
```

| Option | |
| --- | --- |
| `-d, --description TEXT` | bundle description, stored in metadata |
| `-n, --name TEXT` | bundle name (default: the directory name) |

## `datafolio describe`

Print the folio: metadata, snapshots, and every item with its description,
size, and lineage. Same output as `folio.describe()` in Python.

```bash
datafolio describe
datafolio describe --snapshot v1.0
datafolio describe --max-metadata 25
```

| Option | |
| --- | --- |
| `--max-metadata INTEGER` | maximum metadata fields to show |
| `-s, --snapshot TEXT` | describe a snapshot instead of the current state |

## `datafolio validate`

Check that a directory is a folio and that every payload is present and
matches its recorded checksum.

```bash
datafolio validate
datafolio validate /path/to/folio
```

```text
✓ Valid DataFolio bundle: /Users/casey/analysis/experiment-12
  Items: 12 (all payloads verified)
  Snapshots: 1
```

```text
✗ Invalid DataFolio bundle: /Users/casey/analysis/experiment-12
  Items: 12 (1 failing)
  ✗ raw_measurements: missing or corrupt payload
```

A failing external reference usually means "unreachable from here" (no
credentials, no network) rather than "gone".

## `datafolio snapshot`

```bash
datafolio snapshot --help
```

### `create`

```bash
datafolio snapshot create v1.0 -d "Baseline model" -t baseline -t production
datafolio snapshot create v2.0 --env --exec
datafolio snapshot create scratch --no-git
```

| Option | |
| --- | --- |
| `-d, --description TEXT` | snapshot description |
| `-t, --tag TEXT` | tag; repeat the flag for multiple tags |
| `--no-git` | skip git capture (captured by default) |
| `--env` | capture Python version, platform, and package versions |
| `--exec` | capture entry point and working directory |

Git remote URLs are sanitized of embedded credentials before storage, and
environment variables are never captured.

### `list`

```bash
datafolio snapshot list
datafolio snapshot list --tag baseline
```

```text
                         Snapshots (1)
┏━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Name ┃ Description ┃ Items ┃ Created              ┃ Tags     ┃
┡━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ v1   │ baseline    │    12 │ 2026-08-14 13:45 EDT │ baseline │
└──────┴─────────────┴───────┴──────────────────────┴──────────┘
```

### `show`

```bash
datafolio snapshot show v1
```

```text
Snapshot: v1
============================================================

Description: baseline
Tags: baseline
Timestamp: 2026-08-14 13:45:03 EDT

Items (12):
  • classifier
  • embedding
  • features
  ... and 2 more

Git:
  Commit: 9d1a60b
  Branch: v2-refactor
  Status: dirty (uncommitted changes)

Metadata (2 fields):
  • experiment: exp-12
  • analyst: casey
```

### `status` and `diff`

`status` is the `git status` of a folio: what changed since the most recent
snapshot. `diff` does the same against a named one.

```bash
datafolio snapshot status
datafolio snapshot diff            # vs. the newest snapshot
datafolio snapshot diff v1.0
```

```text
Comparing current state to snapshot 'v1'
============================================================

Modified (1):
  ~ features

Unchanged (11):
  = classifier
  = embedding
  ... and 6 more

Summary:
  Added: 0
  Removed: 0
  Modified: 1
  Unchanged: 11
```

"Modified" means the item's current version is not the one the snapshot
pinned — a version comparison, not a content diff.

### `compare`

```bash
datafolio snapshot compare v1.0 v2.0
```

### `delete` and `gc`

```bash
datafolio snapshot delete v0.1 --cleanup
datafolio snapshot delete experimental-v5 -y
datafolio snapshot gc --dry-run
datafolio snapshot gc
```

| Option | |
| --- | --- |
| `--cleanup / --no-cleanup` | remove versions orphaned by the deletion (default: no) |
| `-y, --yes` | skip the confirmation prompt |
| `--dry-run` (`gc`) | list what would be removed, remove nothing |

Nothing is garbage-collected automatically; `gc` is the sweep.

## Not in the CLI

`restore_snapshot()` and `export_snapshot()` are Python-only — both rewrite or
create bundles, which is not something to trigger from a one-line command by
accident. Adding and reading data is Python-only by design; see the
[API cheat sheet](datafolio-api.md).

## Scripting against a folio

There is no `--json` output. For anything scriptable, use Python — it is the
same information without a parsing layer:

```bash
python -c "
from datafolio import DataFolio
f = DataFolio('analysis/experiment-12', read_only=True)
print('\n'.join(f.tables))
"
```

Or read `items.json` directly — it is a documented, stable format. See
[Reading a folio without DataFolio](../guides/format.md).
