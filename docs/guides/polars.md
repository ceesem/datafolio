# Polars, Lazy Scans & External References

DataFolio keeps things simple: **Parquet is the canonical table format** and
**pandas is the eager default**, so existing code is unchanged. On top of that,
you can use **polars** — eager or lazy — and link to large external tables you
don't want to copy.

## Reading tables: pandas, eager polars, lazy polars

Every table (whether stored in the bundle or referenced externally) supports the
same three access modes:

```python
import polars as pl

folio.get("t")                    # pandas.DataFrame  (default, unchanged)
folio.get("t", frame="polars")    # eager polars.DataFrame (downloads/collects)
folio.scan_table("t")             # genuinely lazy polars.LazyFrame
```

`scan_table` returns a `pl.LazyFrame`, so you get predicate/projection pushdown
and never materialize more than you ask for:

```python
(
    folio.scan_table("events")
    .filter(pl.col("ts") > cutoff)
    .select(["user_id", "amount"])
    .group_by("user_id")
    .agg(pl.col("amount").sum())
    .collect()
)
```

### Which lazy scans are *genuinely* lazy?

`scan_table` is honest about laziness. It performs a real, streaming scan
(footer/range reads, no full download up front) for:

- local paths
- `s3://`, `gs://`, `az://` object stores
- `http(s)://` (polars range-requests where the server supports it)

For any scheme it **cannot** scan lazily, it raises a clear error instead of
silently downloading the whole object and pretending it was lazy. If you want an
eager read that downloads, use `get(..., frame="polars")`.

## Working with large tables

A design principle of datafolio is to be **as lightweight as possible and
offload to better tools as quickly as possible**. It has no query API of its
own: for anything that shouldn't be loaded whole, datafolio's job is to hand
you a lazy frame or a file path and get out of the way.

An eager `get()` on a table above the folio's `max_eager_bytes` ceiling
(500 MB by default) raises rather than silently loading it — the error points
you at the two paths below. Pass `allow_full_load=True` (or set
`max_eager_bytes=None`) if you really do want everything in memory.

**1) polars (recommended): `scan_table` and normal polars**

```python
df = (
    folio.scan_table("transactions")          # lazy — nothing is read yet
    .filter(pl.col("amount") > 1000)          # pushdown: filters at the file
    .select(["user_id", "amount"])            # pushdown: reads two columns
    .collect()                                # only now does I/O happen
)
```

Everything polars can do — grouping, joins, window functions, streaming
collection of larger-than-memory results (`collect(engine="streaming")`) —
works on the LazyFrame; see the
[polars lazy API docs](https://docs.pola.rs/user-guide/lazy/). This is also
the only path for sharded/partitioned datasets and works over external
references without downloading them.

**2) pandas (no polars): take the path to pyarrow**

If you're staying in pandas, get the payload path and use pandas'
pyarrow-backed reader directly — column pruning and predicate pushdown are
pandas/pyarrow features, not datafolio ones:

```python
import pandas as pd

df = pd.read_parquet(
    folio.item_path("transactions"),
    columns=["user_id", "amount"],
    filters=[("amount", ">", 1000)],
    engine="pyarrow",
)
```

`item_path` works for included tables and external references alike (for
cloud folios it returns the object URI; reading it with pandas requires the
matching fsspec backend, e.g. `s3fs`).

## Writing tables

`add` accepts a pandas DataFrame, a polars DataFrame, or a polars
**LazyFrame**. A LazyFrame is materialized with a streaming `sink_parquet`
(bounded memory — the full result is never held at once), and its schema/row
count are read back cheaply from the written Parquet footer:

```python
lf = pl.scan_parquet("raw/*.parquet").filter(pl.col("keep")).select(["a", "b"])
folio.add("clean", lf)                # streamed to the bundle, bounded memory
```

Cloud writes are bounded too: the file is serialized to a temp file and streamed
up (no whole-file in-memory buffer).

## External references (no copy)

`reference_table` records a link to external data **without copying it**.
Creating a reference is a cheap, offline manifest write — it makes **no network
requests** (no stat, schema read, row count, or existence check), so linking a
private or currently-unreachable URI never blocks:

```python
folio.reference_table("raw", path="s3://bucket/huge.parquet")   # instant, offline
folio.scan_table("raw")                                          # lazy read, no full download
```

To enrich the manifest with schema, size, row count, and point-in-time source
identity, call `inspect_table` explicitly (this *does* perform I/O and raises
actionable errors if the object is unreachable):

```python
info = folio.inspect_table("raw")
info["columns"], info["num_rows"], info["size_bytes"], info["source_identity"]
```

Use `validate()` to check existence.

### Sharded / partitioned datasets are polars-only

A reference to a directory of Parquet shards (including hive-partitioned
datasets) is read lazily via polars and marked **polars-only**: `scan_table` and
`get(frame="polars")` work, but a plain pandas `get` raises a clear
error (pandas mishandles such layouts). DataFolio *reads* sharded data; it does
not create sharded layouts itself.

```python
folio.reference_table("events", path="s3://bucket/events/")  # hive-partitioned dir
folio.scan_table("events").collect()                          # ok (lazy)
folio.get("events")                                           # raises: polars-only
```

### References and snapshots

Snapshots freeze **owned** items (included tables, models, files). A
referenced table's external bytes are **not** owned and may change over time — a
snapshot preserves the *link*, not the content. `get_snapshot_info` flags any
mutable references, and `folio.mutable_references()` lists them. Use
`inspect_table` to record a `source_identity` for detecting drift.

## Unsupported formats

Only Parquet (canonical) and CSV (reference) are supported. Delta and Iceberg
are rejected up front with an actionable error — convert them to Parquet first.

## Installation

Polars is an optional extra:

```bash
pip install 'datafolio[polars]'
```

Lazy/polars methods raise a clear `ImportError` with this hint if polars isn't
installed; pandas access always works without it.
