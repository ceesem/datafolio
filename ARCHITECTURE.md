# DataFolio Architecture

DataFolio is a lightweight, human-readable working directory for small
collections of linked data — datasets, models, metadata, and files — that can
live locally or in cloud object storage. This document describes the 2.0
design. For the on-disk format itself (usable without datafolio), see
`docs/guides/format.md`; for the user API, see the guides and API reference.

## 1. Design principles

1. **As lightweight as possible; offload as quickly as possible.** datafolio
   organizes, links, and hands off. It has no query API, no compute layer, and
   no bespoke formats. Big-table work belongs to polars/pyarrow, reached via
   `scan_table()` (a plain `pl.LazyFrame`) or `item_path()` (a plain path).
   Downloading belongs to file-sync tools. Code history belongs to git.
2. **Ordinary files, one readable manifest.** A folio must remain usable
   without datafolio: standard formats (Parquet, JSON, `.npy`, joblib/skops)
   plus a single self-contained `items.json` —
   ``{schema_version, revision, metadata, items, snapshots}`` — with relative
   payload paths and a generated README explaining how to read it by hand.
   (Legacy v0/v1 folios with `metadata.json`/`snapshots.json` sidecars load
   transparently and migrate on the next write.)
3. **Descriptions are primary data.** Descriptions, lineage, and item metadata
   live in the manifest and are never discarded implicitly.
4. **References are links, not copies.** External references are absolute,
   static descriptors. Their bytes are not owned, not copied, and not
   guaranteed. Creating a reference is offline; `inspect_table()` opts into I/O.
5. **Many readers, one active writer.** Readers auto-refresh. Local writers are
   serialized by a lock file; a stale writer fails with `ConcurrentWriteError`
   before touching anything, and must `refresh()` and retry. Cloud object
   stores have no cross-machine lock (the installed CloudFiles API exposes no
   conditional writes), so cloud folios should be treated as single-writer.
6. **`items.json` is the commit record.** Every mutation publishes it
   atomically (temp file + `os.replace` locally); nothing is "committed" until
   it is. Failed operations may leave harmless unreferenced payload files, but
   must never damage committed data.
7. **Out of scope, deliberately:** garbage-collection frameworks, repair
   tooling, distributed locking, multi-writer merging, caching layers.

## 2. Components

```
User code
   │  add() / get() / item_path() / item_info()      ← unified core API
   │  add_model() / add_file() / reference_table()   ← explicit verbs
   ▼
DataFolio  (folio.py — facade, composed of:)
   ├── SnapshotMixin        (snapshots.py — registry, views, restore/export)
   ├── ContextCaptureMixin  (context.py — git/env capture for snapshots)
   │
   ├── HandlerRegistry      (base/ — one handler per item type)
   │      DataframeHandler, ReferenceTableHandler, SklearnHandler,
   │      NumpyHandler, JsonHandler, ArtifactHandler, TimestampHandler
   │
   └── StorageBackend       (storage/ — local + cloud I/O via cloudfiles)
```

**DataFolio** owns the manifest (`_items`, `_snapshot_versions`,
`_snapshots`), the mutation protocol, and the public facade. The facade is
intentionally explicit — a small set of verbs whose boilerplate is product
design, not accident.

**Handlers** know how to serialize one item type: `add()` writes the payload
and returns its metadata dict, `get()` reads it back, `get_lazy()`/`inspect()`
are optional table hooks. Auto-detection (`can_handle`) is deliberately
narrow: exact DataFrame/array/datetime types and real estimator classes only.
Adding an item type internally means one handler file plus an entry in
`storage/categories.py` (`ITEM_TYPE_TO_CATEGORY` maps item types to the
`tables/`, `models/`, `artifacts/` subdirectories). This is an internal
seam, not a public plugin API.

**StorageBackend** unifies local and cloud I/O behind one path-string
interface. All cloud traffic goes through cloudfiles (one credential chain);
local manifest writes are atomic (temp + fsync + `os.replace`); cloud parquet
writes are bounded-memory (temp file + streamed upload, LazyFrames sinked).
Errors are never swallowed in existence checks — a transient cloud failure
must not look like "bundle absent."

## 3. The mutation protocol

Every write path flows through the transaction spine in `folio.py` —
one place that defines what a mutation means:

```
_mutation_guard()                 reentrant per-folio lock (local lock file)
  └─ verify committed revision    fail closed: unreadable/stale → raise
       └─ snapshot in-memory state (one deep copy of the small collections)
            └─ [build payload FIRST, then mutate state, then _save_items()]
                 │                 atomic single-file publish; revision += 1
                 ├─ on an exception escaping the guard: restore the state
                 │                 snapshot — no partial mutation can leak
                 │                 into a later commit
                 └─ [best-effort delete newly unreferenced payloads]
```

Nested guards join the outer transaction (its snapshot and its rollback),
so composed operations — and `batch()`, which is simply one long
transaction with the publish at the end — abort as a unit. No individual
method implements rollback.

Ordering rules the protocol enforces:

- **Payload before demotion** (`_commit_owned_item`): the replacement payload
  is written and its metadata validated *before* a snapshotted prior version
  is demoted. A failed handler leaves the folio untouched plus, at worst, an
  unreferenced payload file.
- **Manifest before deletion**: `delete()`, `restore_snapshot()`, and orphan
  cleanup publish the manifest first, then best-effort delete what it no
  longer references. The committed manifest never points at deleted bytes.
- **Failure → restore**: any exception inside the guard (payload error,
  mid-loop exception, failed publish) restores the entry-time state
  snapshot — in memory, with no disk read required.
- **Versioned payload filenames** (`name--r<revision><ext>`): every persisted
  version gets its own file, so overwriting never touches bytes an existing
  manifest or snapshot references, and snapshot operations are manifest
  surgery rather than byte copies.

**Metadata** (`folio.metadata`, a `MetadataDict`) participates in the same
protocol: each mutation enters the guard (stale writers fail before their
in-memory dict changes) and commits through `_save_items()`, so metadata-only
writes advance the same bundle revision other notebooks check.

**`batch()`** holds the guard for the whole block and publishes once at exit.
An exception that *escapes* the block aborts the batch as a unit — the guard's
state snapshot restores everything (staged items, metadata, copy-on-write
moves, deferred deletions), leaving only orphan payload files. An exception
caught and handled *inside* the block does not abort: the batch continues and
publishes normally at exit. Nested batches raise; snapshot creation/deletion
and restore/cleanup inside a batch raise.

## 4. Snapshots

A snapshot pins the current `version_id` of every item, plus a copy of the
bundle metadata and optional git/environment context. **It preserves
folio-owned data and the recorded state of external references. It does not
preserve or guarantee the contents of referenced data.**

- **Copy-on-write:** replacing or deleting a pinned item demotes its
  descriptor to the snapshot-versions list (payload retained); the working
  set moves on. Metadata-only edits (`update_item`) copy the descriptor and
  share the payload file; all deletion paths check `_payload_is_shared` first.
- **One-file commits:** since manifest v2, the registry lives inside the one
  manifest, so a snapshot create/delete is a single atomic publish — there is
  no cross-file ordering and no partially visible registry state. Snapshot
  membership is DERIVED from the registry's pinned `version_id`s (the old
  denormalized `in_snapshots` markers are gone; legacy ones are stripped on
  load).
- **Fail closed:** `load_snapshot()`, `SnapshotView`, and
  `restore_snapshot()` resolve every pinned `version_id` (legacy snapshots
  recorded checksums; both tokens resolve) before changing any state, and
  raise if one is missing — newer data is never silently served under a
  snapshot name.
- **Restore is manifest surgery:** versioned filenames mean restoring
  repoints descriptors; no payload is copied or moved, so an interrupted
  restore cannot lose data.
- `get_snapshot_info()` and `SnapshotView` return defensive copies.

## 5. Concurrency model

- Readers never take the lock; every read entry point calls
  `_refresh_if_needed()` (a cheap comparison of the on-disk `items.json`
  revision against the loaded one).
- Payload checksums are computed at write time and verified only by
  `validate()` — `get()` never verifies them on read.
- Local writers serialize on `items.json.lock` (a `filelock` reentrant lock)
  and check the manifest revision before any change. The lock file lives in
  the bundle; sync tools that delete remote-absent files should exclude it,
  since unlinking a held lock breaks mutual exclusion. Two notebooks writing
  concurrently: the second fails with `ConcurrentWriteError`, refreshes, and
  retries — nothing is clobbered and nothing half-applies.
- Cloud folios: the revision check still runs (best effort), but without
  conditional writes two simultaneous cloud writers can race. Treat cloud
  folios as single-writer.

## 6. Testing

`poe test` runs the suite (integration tests against real file I/O plus
focused unit tests). Integrity behavior is pinned by dedicated regression
files: `test_v2_integrity.py` (stale writers, exception injection,
manifest-publish failures, batch aborts), `test_v2_phase2/3.py` (name
grammar, snapshot preservation), and `test_v2_defensive.py` (defensive
copies, cloud path handling). V1 manifests (bare-list or unversioned) load
and migrate forward on the next write; `schema_version` gates newer formats
with a clear refusal.
