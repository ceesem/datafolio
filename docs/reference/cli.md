---
title: CLI
---

# Command line

The CLI is for looking at a folio, adding files to it, finding things across
folios, and managing snapshots without opening a notebook. Reading data is
Python's job.

```bash
datafolio --help
datafolio --version
```

## Choosing the folio

Four ways, highest priority first:

```bash
datafolio -a exp12 describe                     # registered alias
datafolio -f analysis/experiment-12 describe    # explicit path
export DATAFOLIO_PATH=analysis/experiment-12    # environment variable
cd analysis/experiment-12 && datafolio describe  # current directory
```

`-a` and `-f` are separate flags, so a name is never mistaken for a path.
If you pass an alias to `-f` or `--to` by mistake, the error suggests `-a`.
Register aliases with [`datafolio folios alias`](#datafolio-folios) or
`init --alias`. Every folio a command uses successfully is also remembered
in a recent list, which [`datafolio find`](#datafolio-find) searches.

!!! note "Cloud folios"
    `add`, `find`, `folios` and `init` accept cloud folios (`gs://…`,
    `s3://…`), by path or by alias. `describe`, `validate`, and the snapshot
    commands work on local folios only. Use Python for those on a cloud
    folio, or sync it down first.

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
| `--alias TEXT` | register the new folio under this alias |

## `datafolio add`

Add a file to a folio: `datafolio add [--to PATH | -a ALIAS] FILE`.

```bash
datafolio add --to analysis/experiment-12 ~/Downloads/blah.parquet
datafolio add -a exp12 synapses.parquet -d "synapse table I thought was interesting"
datafolio add -a exp12 cells.csv --name cells
datafolio add -a exp12 notes.csv --as-file
datafolio add -a exp12 /data/huge.parquet --reference
```

The file type decides what gets stored:

| File | Stored as |
| --- | --- |
| `.parquet`, `.csv`, `.feather`, `.arrow` | table, as parquet (via `folio.import_table`) |
| `.npy` | numpy array |
| `.json` | JSON data |
| anything else | the file unchanged (artifact) |

Tables are streamed rather than loaded, so files larger than memory work.
Parquet is copied byte for byte. CSV and feather are converted a block at a
time. CSV column types are inferred from the start of the file; if a later
row doesn't fit, the command fails and suggests `--as-file` or
`--reference`.

| Option | |
| --- | --- |
| `--to PATH` | folio path, local or cloud |
| `-a, --alias TEXT` | folio alias (instead of `--to`) |
| `-n, --name TEXT` | item name (default: file name without extension) |
| `-d, --description TEXT` | item description |
| `--as-file` | store any file unchanged, with no conversion |
| `--reference` | link a parquet/csv file in place, without copying |
| `--overwrite` | replace an existing item |

Without `--to` or `-a`, the folio comes from the global `-f`/`-a`,
`DATAFOLIO_PATH`, or the current directory. `add` never creates a folio; use
`init` first.

## `datafolio folios`

Manage the per-user registry in `~/.datafolio` (set `DATAFOLIO_HOME` to keep
it elsewhere). It holds aliases you choose, and a list of the 50 folios most
recently used from the CLI.

```bash
datafolio folios list                               # aliases, then recents
datafolio folios alias exp12 analysis/experiment-12 # register an alias
datafolio folios alias shared gs://team/folios/shared
datafolio folios alias exp12 other/path --overwrite # rebind
datafolio folios unalias exp12
datafolio folios forget ~/scratch/tmp-folio         # drop from recents
datafolio folios prune                              # drop missing local folios
```

`folios alias NAME` with no path aliases the current folio. Rebinding an
existing alias needs `--overwrite`. `prune` never removes cloud folios.

## `datafolio find`

Find items by name across every aliased and recently used folio. Only each
folio's `items.json` is read, so this is fast even across many folios.

```bash
datafolio find 'cells*'                        # glob on the whole name
datafolio find 'synapse|soma' --regex --type table
datafolio find --in exp12 --type model         # everything of a type in one folio
datafolio find 'dataset=minnie*' --metadata    # folio metadata key=value
datafolio find synapse --desc                  # also search descriptions
```

```text
                  Matches for 'cells*'
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Folio  ┃ Item      ┃ Type           ┃ Description  ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ exp12  │ cells     │ included_table │ cell table   │
│ exp13  │ cells_raw │ numpy_array    │              │
└────────┴───────────┴────────────────┴──────────────┘
```

| Option | |
| --- | --- |
| `--regex` | treat the pattern as a regular expression (matched anywhere) |
| `--type TYPE` | `table`, `model`, `artifact`, `array`, `json`, `timestamp`; repeatable |
| `--metadata` | match folio metadata keys (`KEY` or `KEY=VALUE`) instead of item names |
| `--in ALIAS_OR_PATH` | only search this folio; repeatable |
| `--aliases-only` | skip recent folios that have no alias |
| `--local-only` | skip cloud folios |
| `--include-archived` | include archived items |
| `--case-sensitive` | match case-sensitively (default: case-insensitive) |
| `--desc` | also match items whose description contains the pattern |

A folio that can't be opened (moved, deleted, no credentials) is reported
with a warning and skipped. Exits `1` when nothing matches; if the
pattern does appear in item descriptions, the message says so and suggests
`--desc`. The same search
is available in Python as `datafolio.find()`, which returns a DataFrame.

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
accident. Reading data is Python-only by design; see the
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
