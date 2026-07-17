# Using a folio without Datafolio

A datafolio bundle is designed to remain useful **without** the `datafolio`
package. It degrades into an ordinary directory of standard files plus a
readable JSON manifest. This page documents the on-disk format as a public
compatibility surface: what the files are, how to identify the current items,
and how to resolve and open each one using only the filesystem and standard
format libraries.

## Directory layout

```
my-folio/
├── items.json       # AUTHORITATIVE catalog of every item version
├── metadata.json    # user metadata + timestamps
├── snapshots.json   # named snapshots (if any)
├── README.md        # human-readable orientation
├── CONTENTS.md      # DERIVED inventory (never authoritative)
├── tables/          # Parquet files for included (owned) tables
├── models/          # serialized models (joblib / skops)
└── artifacts/       # numpy (.npy), JSON (.json), images, text, other files
```

`items.json` is the single source of truth. `CONTENTS.md` is a convenience view
regenerated from it — never parse `CONTENTS.md`; read `items.json`.

## The manifest: `items.json`

```json
{
  "schema_version": 1,
  "revision": 7,
  "items": [ { ...item... }, ... ]
}
```

- **`schema_version`** — the manifest format version. A reader that only knows
  older versions should refuse a *newer* `schema_version` rather than guess.
  Very old folios may be a bare JSON list (`[ {...}, ... ]`) with no wrapper —
  treat that as `schema_version` 0.
- **`revision`** — a monotonically increasing integer bumped on every write.
  Used for stale-writer detection (see [Concurrency](#concurrency)).
- **`items`** — a flat list of item *versions*. A logical item may appear more
  than once: the current version plus any prior versions retained by a snapshot.

### Identifying the current items

Use only entries where **`is_current`** is `true` (or absent, in old folios).
Entries with `is_current: false` are prior versions preserved for a snapshot;
ignore them unless you are resolving a specific snapshot.

```python
import json

manifest = json.load(open("my-folio/items.json"))
items = manifest["items"] if isinstance(manifest, dict) else manifest
current = {it["name"]: it for it in items if it.get("is_current", True)}
```

### Common item fields

| Field | Meaning |
| --- | --- |
| `name` | Logical item name (stable across versions). |
| `item_type` | `included_table`, `referenced_table`, `model`, `numpy_array`, `json_data`, `timestamp`, `artifact`. |
| `is_current` | Whether this is the live version. |
| `version_id` | Stable id for this exact version (snapshots pin it). |
| `filename` | Owned items: file name **relative to the type's subdirectory**. |
| `path` | External references: an **absolute** path or URI (no `filename`). |
| `description` | Human note on why this item exists / which to use. |
| `checksum` | Content hash of the owned payload (where applicable). |

## Resolving and opening owned items

Owned items carry a `filename` relative to a subdirectory derived from
`item_type`:

| `item_type` | Subdirectory | Open with |
| --- | --- | --- |
| `included_table` | `tables/` | `pandas.read_parquet`, `polars.read_parquet`, `pyarrow.parquet` |
| `model` | `models/` | `joblib.load` (or `skops` if `serialization_format == "skops"`) |
| `numpy_array` | `artifacts/` | `numpy.load` |
| `json_data` | `artifacts/` | any JSON reader |
| `timestamp` | `artifacts/` | JSON reader (ISO 8601 string + unix timestamp) |
| `artifact` | `artifacts/` | open by extension (image, text, csv, ...) |

```python
import json
import numpy as np
import pandas as pd

root = "my-folio"
subdir = {"included_table": "tables", "model": "models"}

def path_of(item):
    sub = subdir.get(item["item_type"], "artifacts")
    return f"{root}/{sub}/{item['filename']}"

df = pd.read_parquet(path_of(current["features"]))
emb = np.load(path_of(current["embeddings"]))
cfg = json.load(open(path_of(current["config"])))
```

!!! note "Payload filenames are versioned"
    Owned payloads are written under collision-safe, versioned names such as
    `features--r7.parquet` — never overwriting a file a committed manifest (or a
    snapshot) still references. Always resolve the file from the manifest's
    `filename`, not from the logical name.

## External references vs owned objects

An `item_type` of `referenced_table` is an **external reference**. It has a
`path` (absolute local path, `file://` URI, or cloud URI like `s3://`/`gs://`)
instead of a `filename`, and its data is **not stored in the bundle**.

- datafolio does not copy or own referenced data.
- The reference records a *link*, which may point at data that has changed —
  referenced bytes are **not** guaranteed or frozen, and are not preserved by
  snapshots.
- Read a reference directly from its `path` using the appropriate library.

New references must be absolute; legacy folios may contain a relative `path`,
which older tooling resolved against the bundle directory.

## Snapshots: `snapshots.json`

```json
{ "snapshots": { "v1.0": { "item_versions": {"features": "features--r3"}, ... } } }
```

Each snapshot maps item names to a `version_id`. To read an item as of a
snapshot, find the manifest entry whose `version_id` matches, and resolve its
`filename`/`path` exactly as above. A snapshot preserves owned data and the
*recorded* reference descriptors — not the referenced bytes.

## Concurrency

Datafolio supports **many readers and one active writer** per bundle. Local
writers are serialized by a per-folio lock, and a stale writer (one whose loaded
`revision` is behind the on-disk `revision`) is rejected with a
`ConcurrentWriteError` and must call `refresh()` before retrying. Readers never
take the lock.

The complete local mutation — writing the new payload *and* publishing the
manifest — happens under the lock, and new payloads use fresh versioned
filenames, so a stale or interrupted writer can never corrupt a payload the
committed manifest references. A failed write may leave an unreferenced orphan
file; that is harmless (datafolio does not garbage-collect).

!!! warning "Cloud (GCS/S3) writers"
    Object stores lack the conditional-write primitive used for local locking,
    so cross-process write safety on cloud backends is **best-effort**: the
    revision still advances and stale writers are usually detected, but two
    simultaneous cloud writers can still race. Treat a cloud bundle as
    single-writer.
