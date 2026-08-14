# Sharing a folio

A folio is a directory of ordinary files with a readable catalog. That is what
makes sharing simple: most of this page is about *which* ordinary mechanism to
use, not about DataFolio features.

## Put it where the team can see it

The path can be local or cloud storage. Nothing else changes.

```python
folio = DataFolio("gs://team-analysis/experiment-12")
folio.add("features", df, description="…")
folio.get("features")
```

Credentials come from your environment via
[cloudfiles](https://github.com/seung-lab/cloud-files) — the same configuration
your other GCS/S3 tooling uses. DataFolio holds no credentials of its own and
never records them.

For a public bucket you only want to read:

```python
folio = DataFolio("gs://public-data/atlas-v3", use_https=True)
```

`http(s)://` paths are always opened read-only.

## Give collaborators the least dangerous thing that works

**A path, for people who do not use DataFolio.** Every item has a real,
openable location:

```python
folio.item_path("features")
# 'gs://team-analysis/experiment-12/tables/features--r2.parquet'

folio.describe(show_paths=True)     # the whole inventory with paths
```

They run `pd.read_parquet(...)` and are done. Or point them at
`CONTENTS.md` in the directory, which lists every item, its location, and its
description in a Markdown table — no tooling required. See
[Reading a folio without DataFolio](format.md).

**A read-only open, for people who do:**

```python
folio = DataFolio("gs://team-analysis/experiment-12", read_only=True)
folio.get("features")      # fine
folio.add("x", df)         # RuntimeError: Cannot modify a read-only DataFolio
```

Use this in any notebook that is only meant to consume a shared folio. It costs
nothing and removes a whole category of accident.

**A frozen, self-contained folio, for a submission or a handoff:**

```python
folio.create_snapshot("paper-v1", description="Figures 2–4")
folio.export_snapshot("paper-v1", "gs://team-analysis/paper-final")
```

The export contains only that snapshot's items — no history, no scratch work.

**A fork, for someone continuing the work:**

```python
folio.copy("gs://team-analysis/experiment-13",
           metadata_updates={"parent": folio.path})
```

## Multiple readers, one writer

Several people (or notebooks) can read one folio at the same time. A reader
picks up another process's committed writes automatically on its next read, so
this works without ceremony:

```python
# notebook A
folio_a.add("results", df)

# notebook B, already open
folio_b.get("results")     # sees it
folio_b.refresh()          # or force a reload explicitly
```

Writing is where the guarantees narrow, and they are worth stating exactly.

**Locally**, writes are serialized by a per-folio lock, and the whole
mutation — payload write plus catalog publish — happens under it. A writer
whose view is behind the on-disk state is rejected rather than allowed to
overwrite newer work:

```python
folio.add("results", df)
# ConcurrentWriteError: items.json was modified by another writer (on-disk
# revision 12 > loaded 11). Call refresh() and re-apply your change (datafolio
# supports many readers but one writer per bundle).

folio.refresh()      # reload, then re-apply
```

New payloads always get fresh versioned filenames, so an interrupted or stale
writer can never damage a file the committed catalog points at. The cost is
that a failed write may leave an unreferenced orphan file, which is harmless
and swept only when you ask (`cleanup_orphaned_versions()`).

**On cloud storage**, object stores do not offer the primitive that lock
depends on. Stale writers are usually still detected — the catalog revision
check runs the same way — but two simultaneous cloud writers can race.

!!! warning "Treat a cloud folio as single-writer"
    One person (or one job) writes; everyone else opens with `read_only=True`.
    DataFolio does not implement distributed locking and is not going to. If
    you need genuine multi-writer coordination, you need a database, not a
    directory.

If you want a block of reads to see one consistent state — and to skip the
per-read staleness check, which costs round trips on cloud storage — pin it:

```python
with folio.pinned():
    for name in folio.tables:
        summarize(folio.get(name))
```

## Moving a folio around

A folio is plain files with a relative-path catalog, so ordinary sync tools
work and produce a complete, working folio:

```bash
gsutil -m rsync -r gs://team-analysis/experiment-12 ~/analysis/experiment-12
aws s3 sync s3://bucket/experiments/exp-12 ~/analysis/exp-12
rclone sync remote:bucket/exp-12 ~/analysis/exp-12
rsync -a colleague-machine:/data/exp-12/ ~/analysis/exp-12/
```

```python
offline = DataFolio("~/analysis/experiment-12")   # works, including snapshots
```

Use a sync tool when you want a *mirror* (same history, same snapshots); use
`folio.copy()` when you want a *fork* (current versions only, fresh identity).

One thing does not travel: **external references still point where they always
pointed.** Syncing a folio does not bring the referenced dataset with it, and
the new reader may not have access to it. `folio.validate()` will tell you:

```python
folio.validate()
# {'features': True, 'raw_measurements': False, …}   # reference unreachable from here
```

## Trust

Opening a folio someone else wrote means reading files they wrote.

- **`get_model()` on a joblib model executes pickle.** Only load models from
  folios you trust. Models saved with `custom=True` (skops) refuse unknown
  types unless you pass `trusted=True` after reviewing the type list — see
  [Models](models.md).
- Tables, arrays, and JSON are inert data formats; reading them is safe.
- Checksums are recorded at write time and verified by `validate()`. They
  detect corruption and truncation, not a deliberate substitution by someone
  with write access to the bucket.

## Next

- **[Reading a folio without DataFolio](format.md)** — what to tell a
  collaborator who will never install it.
- **[Snapshots](snapshots.md)** — naming the state you are handing over.
- **[What DataFolio is not](limits.md)**.
