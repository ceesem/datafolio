# Tables: big, external, and lazy

Tables are the most common thing in a folio, and the only kind of item where
size forces decisions. This page covers the normal case in four lines and then
the two situations that actually need thought: the table is too big to load,
or the table is not yours.

**Parquet is the canonical format. pandas is the default frame.** Everything
else is opt-in.

## The normal case

```python
folio.add("features", df)                     # pandas or Polars, in it goes
df = folio.get("features")                    # pandas DataFrame back
pl_df = folio.get("features", frame="polars") # Polars DataFrame back
lf = folio.scan_table("features")             # lazy Polars scan
```

What was recorded alongside it:

```python
folio.item_info("features")
# {'item_type': 'included_table', 'filename': 'features--r2.parquet',
#  'num_rows': 3, 'num_cols': 3, 'columns': ['cell_id', 'width', 'depth'],
#  'dtypes': {'cell_id': 'int64', 'width': 'double', 'depth': 'double'},
#  'size_bytes': 2290, 'checksum': '…', 'description': '…', …}
```

Column names, dtypes, and row count are in the catalog, so `describe()` and any
reader can answer "what is in this table" without opening it.

### Two pandas gotchas

**A non-default index is dropped.** Parquet stores columns. DataFolio warns
when it drops one, and gives you the escape hatch:

```python
folio.add("indexed", df)
# UserWarning: Table 'indexed' has a non-default pandas index that will NOT be
# stored (parquet keeps columns only). Call reset_index() first to keep it as a
# column, or pass preserve_index=True to store it.

folio.add("indexed", df, preserve_index=True)   # index stored as ordinary columns
```

With `preserve_index=True` the index becomes real columns (readable by any
tool) and is recorded in the catalog as `index_columns`, so a pandas `get()`
restores it. Polars and direct file readers just see the columns.

**Writing a LazyFrame is streamed.** `add()` accepts a Polars LazyFrame and
sinks it to Parquet with bounded memory — the full result is never held at
once:

```python
lf = pl.scan_parquet("raw/*.parquet").filter(pl.col("keep")).select(["a", "b"])
folio.add("clean", lf)
```

Cloud writes are streamed to a temp file and uploaded, so they are bounded too.

## When the table is too big to load

DataFolio has no query API and will never grow one. For anything that should
not be loaded whole, its job is to hand you a lazy frame or a path and get out
of the way.

An eager `get()` above the folio's ceiling (`max_eager_bytes`, 500 MB by
default) refuses rather than quietly filling memory:

```python
folio.get("transactions")
# ValueError: Table 'transactions' is ~1433.6 MB, above the ~500.0 MB eager-load
# limit. Use scan_table('transactions') for a lazy polars scan with
# predicate/projection pushdown (pip install 'datafolio[polars]' if needed), or
# pass allow_full_load=True (or set max_eager_bytes=None) to load it all anyway.
```

**With Polars — the recommended path:**

```python
import polars as pl

df = (
    folio.scan_table("transactions")     # lazy: nothing read yet
    .filter(pl.col("amount") > 1000)     # pushdown: filtered at the file
    .select(["user_id", "amount"])       # pushdown: two columns read
    .collect()                           # I/O happens here
)
```

Everything Polars can do works on that LazyFrame — joins, window functions,
`collect(engine="streaming")` for larger-than-memory results. See the
[Polars lazy API](https://docs.pola.rs/user-guide/lazy/). This is also the only
path for sharded datasets, and it works on external references without
downloading them.

`scan_table()` is honest about laziness. It streams (footer and range reads,
no up-front download) for local paths, `s3://`, `gs://`, `az://`, and `http(s)://`
where the server supports range requests. For a location it *cannot* scan
lazily, it raises rather than silently downloading the object and calling it
lazy. If you want the download, ask for it with `get(name, frame="polars")`.

**Staying in pandas:** take the path and use pyarrow directly. Column pruning
and predicate pushdown are pandas/pyarrow features, not DataFolio ones.

```python
import pandas as pd

df = pd.read_parquet(
    folio.item_path("transactions"),
    columns=["user_id", "amount"],
    filters=[("amount", ">", 1000)],
    engine="pyarrow",
)
```

**Overriding the guard** when you really do want it all:

```python
folio.get("transactions", allow_full_load=True)
DataFolio(path, max_eager_bytes=None)          # disable the guard for this folio
```

## When the table is not yours

`reference_table()` catalogs external data **without copying it**. Use it for
released datasets, lab archives, anything large or owned by someone else.

```python
folio.reference_table(
    "raw_measurements",
    "gs://lab-data/run-12/measurements.parquet",
    description="Published source table; not owned by this folio",
)
```

Creating a reference is a cheap, offline catalog write. It makes **no network
requests** — no stat, no schema read, no existence check — so linking a private
or currently-unreachable URI never blocks on credentials or the network.

Read it exactly like any other table:

```python
folio.get("raw_measurements")                  # eager (subject to the size guard)
folio.scan_table("raw_measurements")           # lazy, no download
folio.item_path("raw_measurements")            # 'gs://lab-data/run-12/measurements.parquet'
```

### Enriching a reference

When you *do* want schema, size, row count, and point-in-time source identity
recorded, ask for it explicitly. This performs I/O and raises actionable errors
if the object is unreachable:

```python
info = folio.inspect_table("raw_measurements")
info["columns"], info["num_rows"], info["size_bytes"], info["source_identity"]
```

`inspect_table()` works on included tables too, refreshing their recorded stats.

### What a reference does and does not promise

A reference is a link. DataFolio never copies, owns, or freezes the bytes at
the other end.

- The referenced data can change or disappear without the folio noticing.
- Reading it may need credentials the folio knows nothing about.
- `validate()` reports `False` for a reference that is unreachable *from
  here* — that is "could not verify", not necessarily "gone".
- **Snapshots preserve the link, not the content.** `folio.mutable_references()`
  lists them, and `get_snapshot_info()` flags them in any snapshot.

If you need the bytes frozen, `add()` a copy instead of referencing it. That is
the tradeoff, stated plainly: cheap and non-duplicating, or owned and
guaranteed. Pick per table.

### Sharded and partitioned datasets are Polars-only

A reference to a directory of Parquet shards (including hive-partitioned
layouts) is marked polars-only: lazy scans and Polars frames work, plain pandas
`get()` raises a clear error because pandas mishandles such layouts.

```python
folio.reference_table("events", "s3://bucket/events/")   # hive-partitioned dir
folio.scan_table("events").collect()                     # ok
folio.get("events")                                      # raises: polars-only
```

DataFolio reads sharded data. It does not create sharded layouts.

## Formats

Parquet is canonical; CSV is accepted for references. Delta and Iceberg are
rejected up front with an actionable error — convert to plain Parquet and
reference that. Supporting table formats with their own transaction logs would
mean owning a second consistency model, which is exactly the kind of weight
DataFolio declines.

```bash
pip install 'datafolio[polars]'
```

Polars is an optional extra. Without it, pandas access works normally and the
lazy methods raise an `ImportError` that says so.

## Next

- **[Models](models.md)** — the other item type with real caveats.
- **[Sharing a folio](sharing.md)** — cloud folios and handing over paths.
- **[What DataFolio is not](limits.md)**.
