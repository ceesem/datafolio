# DataFolio

**A small, human-readable home for the data associated with an analysis.**

Give DataFolio a dataframe, array, model, JSON value, file, or external table
reference. It saves the object in an ordinary format, records what it is, and
adds it to a readable catalog. Later, open the directory with one path and load
the object by name.

DataFolio is intentionally not a database or data platform. It removes the
repetitive save/load wiring around small collections of related data—and helps
you remember what every file was for.

## The whole idea

A folio is three small things:

1. **A directory** you can inspect, copy, upload, or share.
2. **One readable catalog** describing the directory and its contents.
3. **A save/load dispatcher** that chooses a sensible format and the matching
   loader for each object.

```python
from datafolio import DataFolio

folio = DataFolio("analysis/experiment-12")

folio.add(
    "features",
    features,
    description="One row per neuron; normalized morphology features",
)
folio.add(
    "labels",
    labels,
    description="Manual labels after the March review",
)
folio.reference_table(
    "raw_measurements",
    "gs://lab-data/run-12/measurements.parquet",
    description="Source measurements for this analysis",
)
```

Six months later—or from another project notebook—the directory explains
itself:

```python
folio = DataFolio("analysis/experiment-12")

folio.describe()
features = folio.get("features")
labels = folio.get("labels")
```

You do not need to remember which loader, cloud client, or nearly identical
filename produced each object. The folio remembers that information next to
the data.

## Descriptions are part of the point

Often the most valuable metadata is simply a sentence explaining which of
several similar files this one is.

```python
folio.add(
    "features_reviewed",
    reviewed_features,
    description=(
        "Feature table after the March review; excludes failed segmentations "
        "and retains the original cell IDs"
    ),
)
```

Descriptions appear in `folio.describe()` and the generated `CONTENTS.md`.
They are stored in the catalog, so future you—or a collaborator without
DataFolio—can understand the directory without reconstructing the notebook
that created it.

## Ordinary files, not a proprietary container

A folio remains an understandable directory:

```text
experiment-12/
├── items.json                 # Metadata, item catalog, snapshots, revision
├── CONTENTS.md                # Human-readable inventory
├── tables/
│   ├── features--r4.parquet
│   └── labels--r5.parquet
├── models/
└── artifacts/
    ├── parameters--r2.json
    └── diagnostic-plot--r6.png
```

Tables are Parquet, arrays are `.npy`, JSON remains JSON, and files keep their
original formats. Someone who does not use DataFolio can read `CONTENTS.md`,
inspect `items.json`, and open the payloads directly with standard tools.

## Save once; reopen by name

`add()` dispatches common Python objects to their ordinary storage formats:

```python
folio.add("table", dataframe)                 # pandas, Polars, or LazyFrame → Parquet
folio.add("embeddings", numpy_array)          # → .npy
folio.add("parameters", {"alpha": 0.1})      # → JSON
folio.add("score", 0.94)                      # → JSON
folio.add_model("classifier", model)          # → joblib or skops
folio.add_file("plots/qc.png", name="qc")    # → original file format
```

`get()` consults the catalog and chooses the corresponding loader:

```python
table = folio.get("table")
embeddings = folio.get("embeddings")
parameters = folio.get("parameters")

# Tables can also stay lazy when that is the better tool for the job.
query = folio.scan_table("table")
```

DataFolio stops at that boundary. Querying and transforming data remain the
job of pandas, Polars, PyArrow, and the rest of the Python data ecosystem.

## One path can scope an analysis

A notebook can depend on names rather than a collection of format-specific
paths:

```python
folio = DataFolio(FOLIO_PATH)

features = folio.get("features")
labels = folio.get("labels")
parameters = folio.get("parameters")
```

Point the same analysis at a similarly organized folio and often the only
thing that changes is `FOLIO_PATH`. The same path can also be passed between
notebooks instead of passing a growing collection of individual filenames.

## Included data and external references

DataFolio distinguishes between data the folio owns and data it only points
to:

- **Included objects** are saved inside the directory and move with it.
- **External references** store an absolute, static link. The external data is
  not copied and may require separate access permissions.

```python
folio.reference_table(
    "raw",
    "gs://lab-data/releases/2026-07/raw.parquet",
    description="Published source table; not owned by this folio",
)
```

This keeps small working outputs together without pretending that every large
source dataset belongs in the same directory.

## Snapshots remember folio state

A snapshot records which owned files and catalog information constituted the
folio at a useful moment:

```python
folio.create_snapshot(
    "reviewed-analysis",
    description="Tables and labels used for the final review",
)

reviewed = DataFolio.load_snapshot(
    "analysis/experiment-12",
    "reviewed-analysis",
)
```

Snapshots preserve folio-owned files. For an external reference, they preserve
the recorded link—not the bytes at that external location.

## Local, cloud, and sharing

The folio path can be local or on supported object storage such as GCS or S3:

```python
local = DataFolio("analysis/experiment-12")
cloud = DataFolio("gs://team-analysis/experiment-12")
```

Use ordinary file-copy and sync tools to move a complete folio. Included files
and snapshots move with the directory; absolute external references continue
to point at their original locations.

Multiple readers can open the same folio. Local writes are serialized and a
stale notebook fails safely instead of silently overwriting a newer commit.
Cloud folios should be treated as single-writer because object stores do not
provide the same cross-machine lock.

## Intentional limits

DataFolio stays useful by declining to become a general data system. It is not:

- a database or dataframe engine
- a distributed data catalog
- a workflow orchestrator
- a garbage collector or repair service
- a multi-writer collaboration system
- a replacement for object-store versioning
- a proprietary storage format

It is designed for one person—or a small team sharing mostly read-only work—
managing a few to dozens of understandable objects. Keeping that scope narrow
is what makes a folio easy to inspect, move, and trust.

## When it fits

DataFolio is a good fit when you want to:

- stop rewriting format- and destination-specific save/load code in notebooks
- keep related feature tables, labels, models, metadata, and artifacts together
- reopen an analysis later and understand which file is which
- pass one path between notebooks or projects
- share ordinary files with someone who does not use the library
- keep large source data external while cataloging how it relates to local work

If you need concurrent cloud writers, database queries, automated workflow
execution, or management of thousands of objects, use the tool built for that
job and let DataFolio remain small.

## Relationship to data version control

A folio is just an ordinary directory, so tools like [DVC](https://dvc.org/) can
version it over time while DataFolio explains what lives inside it. But once you
need detailed data version control or pipeline management—reproducing a run,
diffing data across commits, rerunning changed stages—reach for a tool built for
that job. DataFolio deliberately stops before it.

## Installation

```bash
pip install datafolio
```

## Continue

- [Getting Started](guides/getting-started.md) — create and use your first folio
- [Using a folio without DataFolio](guides/format.md) — understand the files directly
- [Polars & References](guides/polars.md) — lazy tables and external data
- [Snapshots](guides/snapshots.md) — record and reopen folio states
- [API Reference](reference/datafolio-api.md) — complete method documentation
- [Migrating to 2.0](guides/migrating-to-2.0.md) — update older code
