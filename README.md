# DataFolio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-ceesem.github.io%2Fdatafolio-blue.svg)](https://ceesem.github.io/datafolio/)

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

## Install

```bash
pip install datafolio            # core
pip install 'datafolio[polars]'  # + lazy scans and Polars frames
```

Python 3.10+.

## Who it's for

One researcher, or a small team sharing mostly read-only work, holding a few to
a few dozen understandable objects per analysis. It assumes you are comfortable
in pandas, Polars, numpy, and scikit-learn and do not want another framework
between you and them.

Good fit when you want to stop writing
`pd.read_parquet(BASE / "features_v3_reviewed.parquet")` in every notebook,
remember which of five similar files the paper used, keep one analysis's
objects together, or hand a colleague a single path.

## What you get

**One write path, one read path.** `add()` picks the storage format from the
object's type; `get()` reads the catalog and returns the matching Python object.

```python
folio.add("table", dataframe)            # pandas / Polars / LazyFrame -> Parquet
folio.add("embeddings", array)           # numpy -> .npy
folio.add("params", {"alpha": 0.1})      # dict / list / scalar / str -> JSON
folio.add_model("classifier", clf)       # -> joblib (or skops)
folio.add_file("plots/qc.png")           # -> the file, unchanged
folio.reference_table("raw", "gs://lab-data/raw.parquet")   # link, don't copy
```

**Descriptions live next to the data.** The most valuable metadata is usually
one sentence saying which of several near-identical files this is. It is stored
in the catalog, shown by `describe()`, and readable by people who never install
the package.

**Ordinary files, not a container format.**

```text
experiment-12/
├── items.json                    # the authoritative catalog (+ metadata + snapshots)
├── CONTENTS.md                   # derived, human-readable inventory
├── README.md                     # how to read this directory without datafolio
├── tables/features--r2.parquet
├── models/classifier--r8.joblib
└── artifacts/params--r5.json
```

**Local or cloud, same code.** `DataFolio("gs://team-analysis/experiment-12")`.

**Lineage, snapshots, and a CLI** when you need them:

```python
folio.add("predictions", preds, inputs=["classifier", "holdout"])
folio.create_snapshot("paper-v1", description="Figures 2–4 in the submission")
```

```bash
datafolio describe
datafolio snapshot status
```

## What it is not

Not a database or query engine, not a workflow orchestrator, not data version
control, not an experiment tracker, not a multi-writer system, not a backup.
Many readers, one writer — treat a cloud folio as single-writer. Comfortable at
a few dozen items per folio, not thousands.

DataFolio's rule is to be as lightweight as possible and hand off to better
tools as soon as possible: big tables go to Polars as a lazy scan, or to you as
a path. See [What DataFolio is not](https://ceesem.github.io/datafolio/guides/limits/)
for the full list, including the sharp edges worth knowing before you rely on it.

## Documentation

Read in this order:

1. [Your first folio](https://ceesem.github.io/datafolio/guides/getting-started/) — ten minutes
2. [Everyday patterns](https://ceesem.github.io/datafolio/guides/everyday/)
3. [Tables: big, external, and lazy](https://ceesem.github.io/datafolio/guides/tables/)
4. [Models](https://ceesem.github.io/datafolio/guides/models/)
5. [Sharing a folio](https://ceesem.github.io/datafolio/guides/sharing/)
6. [Snapshots](https://ceesem.github.io/datafolio/guides/snapshots/)
7. [Reading a folio without DataFolio](https://ceesem.github.io/datafolio/guides/format/)
8. [What DataFolio is not](https://ceesem.github.io/datafolio/guides/limits/)

Reference: [API cheat sheet](https://ceesem.github.io/datafolio/reference/datafolio-api/) ·
[CLI](https://ceesem.github.io/datafolio/reference/cli/) ·
[Migrating from 1.x](https://ceesem.github.io/datafolio/guides/migrating-to-2.0/)

## Development

```bash
uv sync
poe test                      # pytest with coverage
uv run ruff check src/ tests/
poe doc-preview               # local docs server
```

## License

MIT
