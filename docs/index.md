# DataFolio

**A small, self-documenting home for the data behind one analysis.**

You already have the objects: a feature table, some labels, an array of
embeddings, a fitted model, a parameter dict, a QC plot. DataFolio gives them
one directory, one name each, and one sentence each explaining what they are —
then hands them back by name months later, from any notebook.

```python
from datafolio import DataFolio

folio = DataFolio("analysis/experiment-12")

folio.add("features", features, description="One row per neuron; normalized morphology")
folio.add("labels", labels, description="Manual labels after the March review")
folio.add("params", {"alpha": 0.1, "seed": 7})
folio.add_model("classifier", clf, inputs=["features", "labels"])
```

```python
# A different notebook, six months later. One path, no filenames.
folio = DataFolio("analysis/experiment-12")

folio.describe()
features = folio.get("features")
clf = folio.get_model("classifier")
```

That is the whole product. Everything else on this site is a refinement of it.

## Who this is for

DataFolio is built for **one researcher, or a small team sharing mostly
read-only work**, holding a few to a few dozen understandable objects per
analysis. It assumes you are comfortable in pandas, Polars, numpy, and
scikit-learn, and that you do *not* want another framework between you and
them.

It is a good fit when you want to:

- stop writing `pd.read_parquet(BASE / "features_v3_reviewed.parquet")` in
  every notebook
- remember which of five similar files is the one the paper used
- keep tables, labels, models, metrics, and figures for one analysis together
- hand a colleague a single path instead of a folder tour
- keep large source data where it lives, while recording how your work relates
  to it

It is the wrong tool if you need concurrent cloud writers, database queries,
pipeline orchestration, or thousands of objects. See
[What DataFolio is not](guides/limits.md) — the limits are deliberate and
short.

## What you get

**Descriptions live next to the data.** The most valuable metadata is usually
one sentence saying which of several near-identical files this is. That
sentence is stored in the catalog and shown by `describe()`, so it survives the
notebook that produced it.

**One write path, one read path.** `add()` picks the storage format from the
object's type; `get()` reads the catalog and returns the matching Python
object. There is no per-type API to remember, and no format-specific save/load
code in your notebook.

```python
folio.add("table", dataframe)            # pandas / Polars / LazyFrame -> Parquet
folio.add("embeddings", array)           # numpy -> .npy
folio.add("params", {"alpha": 0.1})      # dict / list / scalar / str -> JSON
folio.add("score", 0.94)                 # -> JSON
folio.add_model("classifier", clf)       # -> joblib (or skops)
folio.add_file("plots/qc.png")           # -> the file, unchanged
```

**Ordinary files, not a container format.** A folio is a directory you can
`ls`, `rsync`, zip, or open in Finder:

```text
experiment-12/
├── items.json                    # the authoritative catalog (+ metadata + snapshots)
├── CONTENTS.md                   # derived, human-readable inventory
├── README.md                     # how to read this directory without datafolio
├── tables/
│   ├── features--r2.parquet
│   └── labels--r3.parquet
├── models/
│   └── classifier--r8.joblib
└── artifacts/
    ├── params--r5.json
    └── embedding--r7.npy
```

Tables are Parquet, arrays are `.npy`, JSON is JSON, files keep their original
format. A collaborator without the package reads `CONTENTS.md`, or parses
`items.json` and opens payloads with pandas directly — that is a supported,
documented path, not a fallback. See
[Reading a folio without DataFolio](guides/format.md).

**Local or cloud, same code.** The path can be `gs://…` or `s3://…`; nothing
else changes.

```python
folio = DataFolio("gs://team-analysis/experiment-12")
```

**It stops where the ecosystem starts.** DataFolio has no query API, no
dataframe operations, no plotting, no scheduler. Big tables are handed to
Polars as a lazy scan or handed to you as a path. That boundary is the design,
not a gap.

## The one thing to internalize

**Owned data vs. external references.** A folio *owns* what you `add()` — those
bytes live in the directory and move with it. A `reference_table()` records a
*link* to data the folio does not own and never copies:

```python
folio.reference_table(
    "raw_measurements",
    "gs://lab-data/run-12/measurements.parquet",
    description="Published source table; not owned by this folio",
)
```

References keep a 200 GB source dataset out of your working directory while
still cataloging it. The tradeoff is real and stated everywhere it matters:
the folio cannot guarantee the referenced bytes still exist or still say the
same thing. Snapshots preserve owned data; for a reference they preserve the
link only.

## Where to go next

Read in this order — each page assumes the one before it.

1. **[Your first folio](guides/getting-started.md)** — ten minutes, start here.
2. **[Everyday patterns](guides/everyday.md)** — the handful of moves that
   cover most real use: lineage, folio metadata, bulk reads, tidying up.
3. **[Tables: big, external, and lazy](guides/tables.md)** — Parquet, Polars,
   references, and the eager-read guard.
4. **[Models](guides/models.md)** — joblib vs. skops, and the security rules.
5. **[Sharing a folio](guides/sharing.md)** — cloud paths, read-only opens,
   multiple readers, handing over a path.
6. **[Snapshots](guides/snapshots.md)** — freezing a state you may need to
   return to.
7. **[Reading a folio without DataFolio](guides/format.md)** — the on-disk
   format as a public surface.
8. **[What DataFolio is not](guides/limits.md)** — limits, sharp edges, and
   what to reach for instead.

Reference: [API cheat sheet](reference/datafolio-api.md) ·
[CLI](reference/cli.md) · [Package API](reference/api.md) ·
[Migrating from 1.x](guides/migrating-to-2.0.md)

## Installation

```bash
pip install datafolio            # core
pip install 'datafolio[polars]'  # + lazy scans and Polars frames
```

Python 3.10+.
