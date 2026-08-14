# What DataFolio is not

DataFolio stays small on purpose. Its design rule is *be as lightweight as
possible and hand off to better tools as soon as possible* — so the honest
version of "what it does" includes a clear list of what it declines to do.
Read this before you build something on top of it.

## What it is genuinely good at

| Strength | Why it holds |
| --- | --- |
| Reopening an analysis and understanding it | Descriptions and lineage live in the catalog, next to the data |
| Removing save/load boilerplate | One `add()`, one `get()`; the type picks the format |
| Surviving the library | Plain Parquet/`.npy`/JSON plus a documented JSON catalog |
| Moving between local and cloud | The path is the only thing that changes |
| Sharing with people who do not use it | Every item has a real, openable path; `CONTENTS.md` is readable by anyone |
| Not losing a write to a stale process | Versioned payload filenames + a revision check on every commit |
| Cheap named states | Snapshots pin versions; bytes are copied only on overwrite |

## What it is not

**Not a database or query engine.** There is no filter, join, or aggregate API,
and there will not be one. `scan_table()` hands you a Polars LazyFrame;
`item_path()` hands you a path for pandas/pyarrow/DuckDB. Query with the tool
built for querying.

**Not a workflow orchestrator.** Nothing recomputes, schedules, caches, or
detects that an input changed. `inputs=` is documentation you chose to write,
not a dependency graph anyone executes. If you need "rerun the stages whose
inputs changed", use Snakemake, Nextflow, or DVC pipelines.

**Not data version control.** Snapshots name states inside one folio; they do
not diff data, merge branches, or track history across a repository. A folio is
an ordinary directory, so DVC or lakeFS can version *it* while DataFolio
explains what is inside. Once you need real data VCS, use one.

**Not an experiment tracker.** No runs, metrics-over-time, or dashboards. If
you want MLflow or Weights & Biases, use them — a folio holds the artifacts
they point at.

**Not a multi-writer system.** Many readers, one writer. Local writes are
serialized by a lock; on cloud storage that lock does not exist and two
simultaneous writers can race. Distributed coordination requires a database.

**Not a backup or a permission system.** Snapshots live in the same directory
as the data. Access control is whatever your filesystem or bucket enforces.

**Not a catalog server.** There is no index across folios, no search, no
service. Discovery is you knowing the path.

**Not a data platform for thousands of objects.** See below.

## Scale: where it stops being a good idea

The catalog is one JSON document rewritten atomically on every commit. That is
what makes the format readable and the commits atomic, and it is also the
ceiling: write cost grows with the number of item versions in the folio.

- **Comfortable:** a few to a few dozen items per folio, human-scale, one
  analysis.
- **Fine:** a couple of hundred items, especially with `batch()` for bulk
  writes.
- **Wrong tool:** thousands of items, or a folio written to in a tight loop.
  Group them into a partitioned Parquet dataset and reference *that* as one
  item, or use a real catalog.

Individual items can be large — a reference costs nothing, and lazy scans do
not download. It is the *count* that matters, not the size.

## Sharp edges worth knowing

Each of these is a deliberate choice with a visible failure mode, not a bug.

**JSON loses non-finite floats.** `nan` and `inf` have no JSON representation
and are stored as `null`, coming back as `None`. This applies to JSON items and
to folio metadata values. Store numeric data with NaNs as a numpy array or a
table.

**A non-default pandas index is dropped.** Parquet stores columns. You get a
warning and an opt-in (`preserve_index=True`), not a silent round trip.

**Some objects are refused at write time, on purpose.** Object-dtype numpy
arrays (they pickle, and reads are unreliable), masked arrays, naive
`datetime`s, and Delta/Iceberg table paths all raise with an actionable
message rather than storing something that reads back wrong.

**Strings are data.** `add(name, "results.csv")` stores the string. Files enter
only through `add_file()`.

**Checksums are write-time only.** They are recorded when an item is written
and verified when you call `validate()` — never during a normal `get()`. They
catch corruption and truncation, not a deliberate substitution by someone with
write access.

**`validate()` returning `False` means "could not verify".** Missing,
unreadable, or unreachable all collapse to `False` — including an external
reference in a bucket you currently lack credentials for.

**Nothing is garbage-collected.** Old versions retained by snapshots, and
orphans left by an interrupted write, stay until you call
`cleanup_orphaned_versions()` or `delete_snapshot(cleanup_orphans=True)`.
Reclaiming disk is always explicit.

**Snapshots do not freeze external references.** They preserve the link. The
bytes at the other end can change or vanish. `mutable_references()` lists which
items this applies to.

**Loading a joblib model executes pickle.** skops (`custom=True`) refuses
unknown types unless you pass `trusted=True` — but in *either* format the
defining class must still be importable at load time. skops buys safety, not
portability.

**Nested metadata edits do not save.**
`folio.metadata["a"]["b"] = 1` mutates in place and commits nothing; reassign
the key or use `update()`.

**The CLI works on local folios.** `datafolio -f gs://…` cannot resolve a cloud
path (`init` is the exception). Use Python for cloud folios.

**2.0 folios are not writable by 1.x.** Old versions refuse them cleanly rather
than corrupting them; 1.x folios open in 2.0 and migrate on the first write.

## Reach for something else when…

| You need | Use |
| --- | --- |
| SQL or fast analytical queries | DuckDB, Polars, pyarrow |
| Larger-than-memory transforms | Polars lazy / streaming, Dask |
| Reruns when inputs change | Snakemake, Nextflow, DVC pipelines |
| Data history across commits/branches | DVC, lakeFS, Git LFS |
| Metrics, runs, and dashboards | MLflow, Weights & Biases |
| ACID tables, time travel, concurrent writers | Delta Lake, Iceberg |
| A searchable catalog across projects | A real data catalog, or a database |

DataFolio is meant to sit comfortably next to any of these, holding the small
pile of named, described objects that one analysis actually produced.

## Next

- **[Reading a folio without DataFolio](format.md)** — the compatibility
  surface these limits rest on.
- **[Everyday patterns](everyday.md)** — the shape of use that stays inside them.
