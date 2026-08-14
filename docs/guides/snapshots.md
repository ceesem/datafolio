# Snapshots

A snapshot gives a name to the folio's current contents so you can come back to
them: *the tables the paper used*, *the model before the March retrain*.

```python
folio.create_snapshot("paper-v1", description="Figures 2–4 in the submission")

# months later
paper = DataFolio.load_snapshot("analysis/experiment-12", "paper-v1")
features = paper.get("features")     # exactly the bytes from that day
```

Reach for a snapshot at a moment you might need to defend or reproduce later.
For everything else, the ordinary folio is enough.

## What a snapshot actually is

It is a named record of which *version* of each item was current, plus a copy
of the folio metadata and (by default) your git commit and branch. It is not a
copy of your data.

```python
folio.get_snapshot_info("paper-v1")
{
  "name": "paper-v1",
  "timestamp": "2026-08-14T17:45:03+00:00",
  "item_versions": {"features": "features--r2", "classifier": "classifier--r8", …},
  "metadata_snapshot": {"experiment": "exp-12", "analyst": "casey", …},
  "tags": ["baseline"],
  "description": "Figures 2–4 in the submission",
  "git": {"commit": "9d1a60b…", "branch": "v2-refactor", "dirty": true,
          "remote": "https://github.com/ceesem/datafolio.git"},
}
```

Snapshots are cheap because nothing is copied when you take one. A second file
appears only when you *overwrite* an item a snapshot has pinned:

```python
folio.add("big_table", df)             # tables/big_table--r2.parquet (5 GB)
folio.create_snapshot("v1")            # no new bytes
folio.create_snapshot("v2")            # still no new bytes
folio.create_snapshot("v3")            # still 5 GB on disk

folio.add("big_table", df2, overwrite=True)
# now two files: big_table--r2.parquet (pinned by v1–v3) and big_table--r9.parquet (current)
```

Overwriting an unsnapshotted item does not accumulate versions — the old
payload is removed once the new catalog is committed.

### What it does not cover

| | |
| --- | --- |
| Owned items — tables, models, arrays, JSON, files you `add()`ed | **frozen** |
| External `reference_table()` data | **not frozen** — the link is preserved, the bytes are not yours |
| Your analysis code | not stored — the git commit is *recorded*, that is all |
| The Python environment | only if you pass `capture_environment=True` |

```python
folio.mutable_references()      # ['raw_measurements'] — references in this folio
folio.get_snapshot_info("v1")   # flags them under 'mutable_references'
```

If a snapshot must survive a source dataset changing, `add()` a copy of that
table into the folio instead of referencing it. And record source identity at
link time if you want drift to be detectable later:

```python
folio.inspect_table("raw_measurements")   # records size, schema, source identity
```

## Creating

```python
folio.create_snapshot(
    "paper-v1",
    description="Figures 2–4 in the submission",
    tags=["paper", "final"],
)
```

Options: `capture_git=True` (default), `capture_environment=False`,
`capture_execution=False`. Environment and execution capture are opt-in
because they record more about your machine than most folios need. Environment
variables are **never** captured, and git remote URLs are stripped of embedded
credentials before storage.

Two rules worth knowing up front: snapshot names must be unique (creating one
twice raises), and `create_snapshot()` cannot be called inside a `batch()`
block — the batch's items are not committed yet.

## Reading one back

**As a full folio** — this is usually what you want:

```python
paper = DataFolio.load_snapshot("analysis/experiment-12", "paper-v1")
# or, if you already have the folio open:
paper = folio.get_snapshot("paper-v1")

paper.describe()
paper.get("features")
paper.get_model("classifier")
paper.metadata["accuracy"]        # the metadata as it was then
paper.read_only                   # True — always
```

A loaded snapshot behaves like any read-only folio: `get`, `get_many`,
`scan_table`, `item_path`, `describe`, lineage. It cannot be written to.

**As a quick view** — when you only want one item or the metadata:

```python
snap = folio.snapshots["paper-v1"]
snap.description, snap.timestamp, snap.tags
snap.get("features")
snap.scan_table("features")

"paper-v1" in folio.snapshots
list(folio.snapshots)
```

**Without loading anything:**

```python
folio.describe(snapshot="paper-v1")
```

## Comparing

```python
folio.list_snapshots()                      # name, timestamp, description, tags
folio.diff_from_snapshot("paper-v1")        # current state vs. a snapshot
folio.diff_from_snapshot()                  # vs. the newest snapshot
folio.compare_snapshots("v1", "v2")         # snapshot vs. snapshot
```

```python
{'snapshot_name': 'paper-v1',
 'added_items': [], 'removed_items': [], 'modified_items': ['features'],
 'unchanged_items': ['classifier', 'labels', 'params', …],
 'metadata_changes': {}}
```

"Modified" means the item's pinned version is no longer the current one. It is
a version comparison, not a diff of the data itself — DataFolio does not open
payloads to compare their contents.

## Going back

**Restore** rewinds the working folio to a snapshot: pinned versions become
current again, items deleted since the snapshot come back, and items added
since it are removed from the working set. It is destructive to the current
state, so it insists you say so:

```python
folio.restore_snapshot("paper-v1", confirm=True)
```

**Export** is the safer and usually better move — it writes a clean, standalone
folio containing only that snapshot's items, leaving the original untouched:

```python
shared = folio.export_snapshot("paper-v1", "share/paper-final")
```

That is what to hand a collaborator or attach to a submission: no version
history, no unrelated items, just the state you named.

## Cleaning up

Snapshots retain old payloads. When you delete a snapshot, the versions it was
the last to pin become orphans, and you decide when the bytes go:

```python
folio.delete_snapshot("scratch-v3")
folio.delete_snapshot("scratch-v3", cleanup_orphans=True)

folio.cleanup_orphaned_versions(dry_run=True)    # list what would be removed
folio.cleanup_orphaned_versions()                # actually remove it
```

DataFolio never garbage-collects on its own. Reclaiming disk is always an
explicit call — which also means an interrupted write can leave an
unreferenced file behind, harmless until you sweep it.

## From the terminal

```bash
datafolio snapshot create v1.0 -d "Baseline" -t baseline -t production
datafolio snapshot list
datafolio snapshot list -t baseline
datafolio snapshot show v1.0
datafolio snapshot status              # like git status, vs. the last snapshot
datafolio snapshot diff v1.0
datafolio snapshot compare v1.0 v2.0
datafolio snapshot delete v0.1 --cleanup
datafolio snapshot gc --dry-run
```

```text
Current bundle: /Users/casey/analysis/experiment-12
Last snapshot: v1 (2026-08-14)

Changes since last snapshot:

Modified (1):
  ~ features

Unchanged: 11 items
```

Restore and export are Python-only — both rewrite or create bundles, which is
not something to do from a one-line command by accident.

## When *not* to snapshot

- **As a backup.** A snapshot lives in the same directory as the data. If the
  directory is lost, so is the snapshot. Back up the folio itself.
- **For code history.** Git already does this. A snapshot records the commit
  hash so you can pair them; it stores no code.
- **Per run of a loop.** Twenty snapshots of a hyperparameter sweep is worse
  than twenty named items and one snapshot of the conclusion.
- **As a substitute for a fork.** If two lines of work will diverge and both
  continue, `folio.copy()` gives each a folio of its own.
- **For a moment you will not return to.** A description on the item usually
  carries the same information at no cost.

Good practice: snapshot at boundaries a person would recognize — *baseline*,
*submitted*, *before the retrain*, *shipped* — with a description that says why
you stopped there.

## Next

- **[Sharing a folio](sharing.md)** — exporting and handing over.
- **[Reading a folio without DataFolio](format.md)** — how snapshots look on disk.
- **[What DataFolio is not](limits.md)**.
