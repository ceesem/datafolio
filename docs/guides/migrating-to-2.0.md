# Migrating to DataFolio 2.0

DataFolio 2.0 consolidates the item API: one `add()` and one `get()` replace
the per-type method families, plus a handful of explicit verbs where being
explicit matters. **The on-disk format is unchanged** — folios written by 1.x
open in 2.0 (and vice versa) with no migration step.

## Method mapping

| 1.x | 2.0 |
| --- | --- |
| `add_table(name, df, ...)` | `add(name, df, ...)` |
| `add_numpy(name, arr, ...)` | `add(name, arr, ...)` |
| `add_json(name, x, ...)` | `add(name, x, ...)` |
| `add_timestamp(name, dt, ...)` | `add(name, dt, ...)` (timezone-aware datetime) |
| `add_data(name, x, ...)` | `add(name, x, ...)` |
| `add_data(name, reference=p)` | `reference_table(name, path=p)` |
| `add_sklearn(name, m, ...)` | `add_model(name, m, ...)` |
| `add_artifact(name, path, ...)` | `add_file(path, name=name, ...)` |
| `get_table(name)` | `get(name)` |
| `get_table(name, frame='polars')` | `get(name, frame='polars')` |
| `get_numpy` / `get_json` / `get_data` / `get_timestamp` | `get(name)` |
| `get_sklearn(name)` | `get_model(name)` |
| `get_lazy(name)` | `scan_table(name)` |
| `get_artifact_path(name)` | `get(name)` or `item_path(name)` |
| `get_table_path` / `get_model_path` / `get_numpy_path` / `get_json_path` / `get_timestamp_path` / `get_data_path` / `get_item_path` | `item_path(name)` |
| `get_table_info` / `get_model_info` / `get_artifact_info` | `item_info(name)` |
| `cache_status` / `clear_cache` / `invalidate_cache` / `refresh_cache` | removed (caching subsystem cut) |
| `DataFolio(..., cache_enabled=, cache_dir=, cache_ttl=)` | removed |
| `code=` (on any add method / `update_item`) | removed — git owns code history |
| `add_model(..., hyperparameters=)` | removed — `estimator.get_params()` owns this; store it as a JSON item if you want it in the folio |
| `add(..., models=[...])` on tables | removed — put model names in `inputs` (old manifests' `models` fields still count as lineage on read) |
| `reproduce_instructions()` / `datafolio snapshot reproduce` | removed — the captured git/env data is in `get_snapshot_info()` |
| `folio.data.name.polars` | `get(name, frame='polars')` or `scan_table(name).collect()` |

Unchanged: `reference_table`, `inspect_table`, `scan_table`, `describe`,
`list_contents`, `delete`, `update_item`, `archive`/`unarchive`, `copy`,
`validate`/`is_valid`, `batch`, `refresh`, all lineage methods, the whole
snapshots API, `folio.data` accessor syntax, and `folio.metadata`.

## Behavior changes

**Overwrite is uniform.** Replacing *any* existing item requires
`overwrite=True` — including items pinned by a snapshot (the pinned version
is preserved via copy-on-write). In 1.x, snapshotted tables could be
silently replaced; that no longer happens.

**Strings are always JSON.** `add(name, "results.txt")` stores the string.
In 1.x, a string was stored as a file artifact whenever a file by that name
happened to exist in the working directory. Files enter the folio only via
`add_file(path)`.

**Model auto-detection is narrower.** `add()` recognizes actual estimator
instances (scikit-learn `BaseEstimator`, XGBoost, LightGBM, CatBoost). An
arbitrary object that merely has `fit`/`predict` methods must be stored
explicitly with `add_model()` — pickling something is an explicit act.

**Numbers are JSON, timestamps are datetimes.** `add(name, 1705318200)`
stores a JSON number. To store a timestamp, pass a timezone-aware datetime:

```python
from datetime import datetime, timezone
folio.add('start', datetime.fromtimestamp(1705318200, tz=timezone.utc))
folio.get('start', as_unix=True)  # -> 1705318200.0
```

**Unknown options raise.** A typo'd keyword (`folio.get('cfg', colums=[...])`)
now raises `TypeError` instead of being silently ignored.

**skops models require explicit trust.** Models saved with `custom=True`
(skops format) refuse non-standard types on load unless you opt in:
`get_model(name, trusted=True)`. Joblib-format models are pickle — only
load folios you trust.

**Object-dtype numpy arrays are rejected at `add()` time.** In 1.x the write
succeeded and every read failed. Convert to a concrete dtype, or store via
`add_model()`.

**A non-default pandas index warns.** Parquet stores columns; a non-default
index is dropped. 2.0 warns when this happens and adds an escape hatch:
`add(name, df, preserve_index=True)` — this stores the index as ordinary
columns (readable by any tool) and records them in the manifest as
`index_columns`, so the pandas read path can `set_index()` them back.
polars and direct file readers simply see the columns.

**Snapshots no longer embed requirements.txt.** `capture_environment=True`
records the Python version, platform, and uv.lock hash; the full dependency
text belongs to git.

**Item names are validated.** Names are segments of letters, digits, `.`,
`_`, `-` (starting with a letter or digit), optionally namespaced with `/`
(e.g. `'examples/weights'`). Names containing `..`, leading `_`, or spaces
are rejected.

**Snapshot integrity fixes** (also in 2.0): `delete()` of a snapshotted item
preserves the pinned version; `restore_snapshot(confirm=True)` also restores
items deleted after the snapshot; `create_snapshot()` inside `batch()`
raises; `update_item()` on a snapshotted item no longer edits the snapshot's
view; `diff_from_snapshot()` defaults to the *newest* snapshot.

## Mechanical migration

Most code migrates with search-and-replace:

```bash
# in your notebooks/scripts
sed -i '' \
  -e 's/\.add_table(/\.add(/g'  -e 's/\.add_numpy(/\.add(/g' \
  -e 's/\.add_json(/\.add(/g'   -e 's/\.add_data(/\.add(/g' \
  -e 's/\.add_sklearn(/\.add_model(/g' \
  -e 's/\.get_table(/\.get(/g'  -e 's/\.get_numpy(/\.get(/g' \
  -e 's/\.get_json(/\.get(/g'   -e 's/\.get_data(/\.get(/g' \
  -e 's/\.get_timestamp(/\.get(/g' -e 's/\.get_sklearn(/\.get_model(/g' \
  -e 's/\.get_lazy(/\.scan_table(/g' \
  your_script.py
```

Then handle the argument-order change for `add_artifact` → `add_file`, any
`add_timestamp(unix_number)` calls, and `overwrite=True` where you relied on
silently replacing snapshotted items.
