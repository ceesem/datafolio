# Everyday patterns

The moves that come up over and over once a folio is part of your workflow,
roughly in the order you will need them.

## One path per analysis

The single highest-value habit. A notebook takes a folio path and item names —
never filenames, never formats, never a cloud client.

```python
from datafolio import DataFolio

FOLIO = "analysis/experiment-12"          # the only thing that varies per run

folio = DataFolio(FOLIO)
features = folio.get("features")
labels = folio.get("labels")
params = folio.get("params")
```

Point the same notebook at `experiment-13` and it works, provided that folio
uses the same names. Passing one path between notebooks — or to a colleague —
replaces passing a growing list of individual files.

A useful convention: keep a small set of *canonical* names (`features`,
`labels`, `params`, `model`) that every folio in a project uses, and namespace
everything else (`qc/*`, `scratch/*`).

## Record where things came from

`inputs=` records lineage at write time. It costs one argument and makes the
folio explain its own dependency structure.

```python
folio.add("features", features, inputs=["raw_measurements"])
folio.add("train", train_df, inputs=["features", "labels"])
folio.add_model("classifier", clf, inputs=["train"])
folio.add("predictions", preds, inputs=["classifier", "holdout"])
```

Then query it:

```python
folio.get_inputs("predictions")        # ['classifier', 'holdout']
folio.get_dependents("features")       # ['train']
folio.get_lineage_graph()              # {item: [inputs], ...}
```

Lineage shows up in `describe()` under each item, and it makes two other
operations meaningful:

```python
folio.delete("features")
# UserWarning: Deleting 'features' which is used by: train. Those items may
# have broken lineage.

folio.copy("share/final", include_items=["predictions"], follow_lineage=True)
# also copies classifier, holdout, and their upstream items
```

Lineage is documentation, not enforcement. DataFolio will not stop you deleting
an input, recompute anything, or detect that you edited a table outside the
folio. It records what you told it.

## Folio-level metadata

Per-item descriptions say what an object is. Folio metadata says what the
*analysis* is — parameters of the run, final scores, the ticket number, who
did it.

```python
folio.metadata["accuracy"] = 0.94
folio.metadata["review_round"] = "March 2026"
folio.metadata.update({"n_cells": 1043, "pipeline": "morphology-v3"})
```

Writes commit immediately, at any point in the folio's life. (The constructor's
`metadata=` argument only seeds a *new* folio; reopening an existing one loads
what is on disk and ignores it.) One caveat worth knowing:

```python
folio.metadata["scores"] = {"acc": 0.9}
folio.metadata["scores"]["acc"] = 0.95      # does NOT save — nested, in-place
folio.metadata["scores"] = {"acc": 0.95}    # reassign the key instead
```

For anything structured or large, prefer a JSON item — it is a first-class,
described, versioned object, and metadata stays small and scannable:

```python
folio.add("cv_scores", scores_dict, description="5-fold CV scores per model")
```

## Read many items at once

Sequential `get()` calls are fine locally. Against cloud storage, each read pays
a round trip, and a loop of small reads spends most of its time waiting.

```python
tables = folio.get_many(["train", "test", "holdout"])
tables["train"].shape
```

`get_many()` reads concurrently and returns a name → object dict. It is exactly
a thread pool over `get()` inside a pinned block, which you are equally welcome
to write yourself. On a 250 ms round trip, 60 items goes from ~16 s sequential
to ~1.3 s with the default 20 threads. Locally it changes nothing.

Type options apply to every name, so batch by type:

```python
frames = folio.get_many(folio.tables, frame="polars")
```

If you are doing a series of reads that should see one consistent state — and
you want to skip the per-read staleness check — pin the manifest:

```python
with folio.pinned():
    for name in folio.tables:
        summarize(folio.get(name))
```

Inside `pinned()`, another process's writes are not visible until the block
exits. That is the trade: a coherent view instead of an up-to-date one.

## Write many items at once

`batch()` defers the catalog commit to the end of the block, so 100 additions
publish once instead of 100 times.

```python
with folio.batch():
    for name, arr in arrays.items():
        folio.add(f"embeddings/{name}", arr)
```

The batch is all-or-nothing: if an exception escapes the block, nothing is
committed and the folio returns to its previous on-disk state. Snapshots cannot
be created inside a batch (the items are not committed yet), and batches do not
nest.

## Tidy up

```python
folio.update_item("features", description="Post-review features (v2)")
folio.update_item("features", description="")        # clear the description
folio.update_item("model", inputs=["train"])

folio.archive("scratch/*")        # hide from describe()/list_contents(), keep readable
folio.unarchive("scratch/step2")
folio.delete(["tmp_a", "tmp_b"])  # remove catalog entry and payload
```

Check that the folio still matches the disk:

```python
folio.validate()
# {'features': True, 'labels': True, 'raw_measurements': False, ...}
folio.is_valid()   # True only if every item passes
```

`False` means the payload is missing, unreachable, or its checksum no longer
matches. An external reference to a bucket you cannot currently reach reports
`False` too — that is existence, not corruption. Checksums are computed at
write time and only re-checked here, never during a normal `get()`.

## Fork a folio

`copy()` makes a derived folio: current versions only, no snapshot history,
archived items excluded, fresh identity.

```python
tuned = folio.copy(
    "analysis/experiment-12-tuned",
    metadata_updates={"parent": folio.path, "change": "alpha sweep"},
)

final = folio.copy(
    "share/paper-final",
    include_items=["predictions", "classifier"],
    follow_lineage=True,
)
```

To *mirror* a folio rather than fork it — same history, same snapshots — use an
ordinary sync tool (`gsutil rsync`, `aws s3 sync`, `rclone`, `rsync`). A folio
is plain files; a synced directory is a complete, working folio.

## Hand someone a path

Every item has a real, openable path — including on cloud storage. This is how
you collaborate with someone who does not use the package:

```python
folio.item_path("features")
# 'gs://team-analysis/experiment-12/tables/features--r2.parquet'

folio.describe(show_paths=True)   # every item's path, ready to copy out
```

They open it with `pd.read_parquet(...)` and never install anything.

The same escape hatch is how you sidestep the library when you need to:

```python
import pandas as pd

cols = pd.read_parquet(folio.item_path("features"), columns=["cell_id", "width"])
```

## Peek from the terminal

```bash
cd analysis/experiment-12
datafolio describe
datafolio validate
datafolio snapshot list
```

Or from anywhere:

```bash
datafolio -f analysis/experiment-12 describe
export DATAFOLIO_PATH=analysis/experiment-12
```

The CLI is read-oriented (plus `init` and snapshot management) and works on
local folios. See the [CLI reference](../reference/cli.md).

## Two notebooks, one folio

Opening the same folio twice is normal and safe for reading. A reader picks up
another process's committed writes automatically on its next read; a writer
whose view is behind the on-disk state is rejected rather than allowed to
clobber it.

```python
folio.refresh()   # force a reload if you want to be certain
```

Writing from two processes at once is only really safe on a local filesystem.
Treat a cloud folio as single-writer. Details in
[Sharing a folio](sharing.md#multiple-readers-one-writer).

## Next

- **[Tables](tables.md)** — large tables, references, and lazy scans.
- **[Snapshots](snapshots.md)** — when you need to return to a past state.
- **[What DataFolio is not](limits.md)** — the boundaries these patterns stop at.
