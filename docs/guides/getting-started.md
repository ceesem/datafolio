# Your first folio

Ten minutes, one directory, six objects. By the end you will have created a
folio, put a mixed bag of Python objects in it, read them back by name, and
understood what is on disk.

```bash
pip install 'datafolio[polars]'
```

Polars is optional; install it if you want lazy scans of large tables. Everything
below works without it except `scan_table()`.

## Create a folio

A folio is a directory. You name it, and open it the same way whether it exists
yet or not.

```python
from datafolio import DataFolio

folio = DataFolio(
    "analysis/experiment-12",
    metadata={"experiment": "exp-12", "analyst": "casey"},
)
```

The `metadata` dict is folio-level context — who, what, when, which run. It
describes the folio as a whole, not any one item, and you can keep adding to it
for as long as the folio exists:

```python
folio.metadata["accuracy"] = 0.94                 # commits immediately
folio.metadata.update({"n_cells": 1043, "stage": "review"})
folio.metadata["accuracy"]
```

The constructor argument is just the *initial* value, so it only takes effect
when the folio is created. Reopening an existing folio loads the metadata
already on disk and ignores anything passed to `metadata=` — it will not merge
or overwrite. To change metadata on a folio that already exists, assign to
`folio.metadata` as above.

The path can be local or cloud (`gs://…`, `s3://…`). Nothing else in this guide
changes.

## Add objects

`add()` is the only write verb you need for data. It picks the storage format
from the object's type.

```python
folio.add(
    "features",
    features_df,
    description="One row per neuron; normalized morphology features",
)
folio.add("labels", labels_df, description="Manual labels after the March review")
folio.add(
    "embedding",
    embedding_array,
    inputs=["features"],
    description="8-d embedding of features",
)
folio.add("params", {"alpha": 0.1, "seed": 7}, description="Fit parameters")
folio.add("score", 0.94)
```

| You pass | It stores | You get back from `get()` |
| --- | --- | --- |
| pandas / Polars DataFrame, Polars LazyFrame | Parquet in `tables/` | pandas DataFrame (or Polars on request) |
| numpy array | `.npy` in `artifacts/` | numpy array |
| dict, list, str, int, float, bool, None | JSON in `artifacts/` | the same value |
| timezone-aware `datetime` | JSON timestamp | UTC-aware `datetime` |
| scikit-learn estimator | joblib in `models/` | the fitted model |

Three things need an explicit verb, because guessing would be wrong:

```python
folio.add_model(                                     # any picklable model
    "classifier", clf,
    inputs=["features", "labels"],
    description="Baseline logistic regression",
)
folio.add_file("plots/qc.png", category="plots")     # copy a file in
folio.reference_table(                               # link, don't copy
    "raw_measurements",
    "gs://lab-data/run-12/measurements.parquet",
    description="Source measurements",
)
```

!!! note "A string is data, never a filename"
    `folio.add("note", "results.csv")` stores the *string* `"results.csv"`.
    Files enter a folio only through `add_file()`. This is deliberate: in
    earlier versions a string was silently treated as a path if a file by that
    name happened to exist in your working directory, which made the same code
    behave differently depending on where you ran it.

### Descriptions are the point

The `description=` argument is what makes a folio worth using six months later.
It is the difference between `features_v3_final.parquet` and a sentence that
says which review it came after.

```python
folio.add(
    "features_reviewed",
    reviewed,
    description=(
        "Feature table after the March review; excludes failed segmentations "
        "and retains the original cell IDs"
    ),
)
```

Descriptions are stored in the catalog and shown by `describe()` and
`CONTENTS.md` — including to people who never install the package.

### Names

Names are letters, digits, `.`, `_`, `-`, starting with a letter or digit. You
can namespace with `/`:

```python
folio.add("qc/step1", {"dropped": 12})
folio.add("qc/step2", {"dropped": 3})
folio.describe("qc/*")     # glob over names
```

Invalid names fail loudly at `add()` time:

```python
folio.add("my name", df)
# ValueError: Invalid item name 'my name': segment 'my name' must start with a
# letter or digit and contain only letters, digits, '.', '_', and '-'
```

## Read them back

`get()` is the matching read verb. It consults the catalog and returns the
natural Python object.

```python
features = folio.get("features")            # pandas DataFrame
embedding = folio.get("embedding")          # numpy array
params = folio.get("params")                # dict
score = folio.get("score")                  # float

clf = folio.get_model("classifier")         # models have their own getter
path = folio.get("qc")                      # a file item returns its path
```

Two exceptions to "you get the object back", both intentional:

- **Files** (`add_file`) return the payload's path. A file's value *is* a file;
  handing you an open buffer would just be a worse `open()`.
- **Models** use `get_model()`, because loading a model executes code (see
  [Models](models.md)).

Missing names and wrong options fail immediately:

```python
folio.get("nope")
# KeyError: Item 'nope' not found in DataFolio

folio.get("params", frame="polars")
# TypeError: Unknown option(s) for item type 'json_data': ['frame']
```

Check membership before writing:

```python
if "features" not in folio:
    folio.add("features", compute_features())
```

## See what's there

`describe()` prints the folio, grouped by kind, with descriptions and lineage:

```python
folio.describe()
```

```text
DataFolio: /Users/casey/analysis/experiment-12
==============================================

Created: Today at 1:44 PM EDT
Updated: Today at 1:44 PM EDT

Metadata (2 fields):
  • analyst: casey
  • experiment: exp-12

Tables (3):
  • raw_measurements (reference): Source measurements
    ↳ path: gs://lab-data/run-12/measurements.parquet
  • features: One row per neuron; normalized morphology features
    ↳ size: 2.2 KB
  • labels: Manual labels after the March review
    ↳ size: 1.6 KB

Numpy Arrays (1):
  • embedding: 8-d embedding of features
    ↳ shape: [3, 8], dtype: float64
    ↳ size: 320 B
    ↳ inputs: features

JSON Data (2):
  • params: Fit parameters
    ↳ type: dict
    ↳ size: 31 B
  • score: (no description)
    ↳ type: float
    ↳ size: 4 B

Models (1):
  • classifier: Baseline logistic regression
    ↳ size: 1.1 KB
    ↳ inputs: features, labels
```

Useful variants: `describe("qc/*")` to filter by name, `describe(show_paths=True)`
to print each item's full path (handy for cloud folios you are sharing), and
`describe(return_string=True)` to capture it instead of printing.

Programmatic equivalents:

```python
folio.tables            # ['raw_measurements', 'features', 'labels']
folio.models            # ['classifier']
folio.artifacts         # file items
folio.list_contents()   # dict of every kind -> names

folio.item_info("features")   # the catalog entry: columns, dtypes, rows, size, checksum
folio.item_path("features")   # '/…/experiment-12/tables/features--r2.parquet'
```

In Jupyter, tab completion works through the `data` accessor:

```python
folio.data.<TAB>
folio.data.features.content        # same as folio.get('features')
folio.data.features.description
folio.data.features.inputs
```

## Change and remove things

Replacing an item is explicit, always:

```python
folio.add("features", cleaned, overwrite=True)
```

Without `overwrite=True` you get a `ValueError`. This is uniform across every
item type — there is no type where a silent replacement happens.

Edit an item's description or lineage after the fact:

```python
folio.update_item("features", description="Post-review features (v2)")
folio.update_item("classifier", inputs=["features", "labels"])
```

Remove, or just hide:

```python
folio.delete("scratch_table")               # gone, payload removed
folio.delete(["tmp_a", "tmp_b"])
folio.archive("qc/*")                       # hidden from describe()/list_contents()
folio.unarchive("qc/step1")
```

Archived items are still readable by `get()`; they are simply out of the way.
Use `delete` for mistakes, `archive` for intermediates you want to stop looking
at.

## What's on disk

```text
analysis/experiment-12/
├── items.json                  # the authoritative catalog
├── CONTENTS.md                 # derived inventory, human-readable
├── README.md                   # how to read this directory without datafolio
├── tables/
│   ├── features--r2.parquet
│   └── labels--r3.parquet
├── models/
│   └── classifier--r8.joblib
└── artifacts/
    ├── embedding--r7.npy
    ├── params--r5.json
    └── qc/step1--r9.json
```

Payload filenames carry a revision suffix (`--r2`) so a new version never
overwrites bytes an older catalog entry or a snapshot still points at. Always
resolve files through `item_path()` or `items.json`, never by guessing the
filename.

From a terminal, without Python:

```bash
cd analysis/experiment-12
datafolio describe
datafolio validate
```

## Reopen it later

That is the payoff:

```python
folio = DataFolio("analysis/experiment-12")

folio.describe()
features = folio.get("features")
params = folio.get("params")
```

If you keep the same item names across analyses, the only thing a notebook
needs to change to point at a different experiment is the path:

```python
FOLIO = "analysis/experiment-13"

folio = DataFolio(FOLIO)
features, labels = folio.get("features"), folio.get("labels")
```

## Next

- **[Everyday patterns](everyday.md)** — lineage, folio metadata, bulk reads,
  batching, and tidying up.
- **[Tables](tables.md)** — when a table is too big to `get()`.
- **[Sharing a folio](sharing.md)** — cloud paths and read-only opens.
- **[What DataFolio is not](limits.md)** — worth five minutes before you build
  on it.
