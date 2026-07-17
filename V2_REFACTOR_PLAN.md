# DataFolio 2.0 refactor plan

**Status: proposal for review — no code has been changed.**

Decisions already made (2026-07-17):
- Unified `add()`/`get()` core API; type-specific methods **removed outright** (clean break, version 2.0.0).
- `add_model` stays an explicit verb; other per-type options are not expected to be commonly used.
- Cache subsystem is **cut** for now.
- Handler registry remains the internal extensibility seam; external/third-party extensibility is not a goal and docs stop advertising it.

Guiding constraint: **2.0 is an API break, not a format break.** The on-disk layout
(`items.json` schema_version 1, versioned payload filenames, `snapshots.json`) is
unchanged; bundles written by 1.x open in 2.0 and vice versa.

---

## Target public API

### Constructor

```python
DataFolio(path, metadata=None, random_suffix=False, read_only=False,
          use_https=False, max_eager_bytes=500 * 1024 * 1024)
```

Removed kwargs: `cache_enabled`, `cache_dir`, `cache_ttl`.

### Core item operations (new)

```python
folio.add(name, obj, *, description=None, inputs=None, overwrite=False,
          code=None, **type_opts)          # registry dispatch, one guarded commit path
folio.get(name, **type_opts)               # e.g. frame='polars', columns=[...] for tables
folio.item_path(name)                      # payload path (external path for references)
folio.item_info(name)                      # manifest entry (read-only copy)
folio.delete(name | [names], warn_dependents=True)
folio.update_item(name, description=..., inputs=..., code=...)
folio.archive(name) / folio.unarchive(name)
```

`add()` semantics:
- Dispatch via `detect_handler`; JSON fallback for primitives (`int/float/str/bool/None`).
- **Strings are always data** (JSON). Files enter via `add_file` only — kills the
  CWD-dependent artifact sniffing (audit H2, handlers).
- Model auto-detection narrows to `isinstance(obj, sklearn.base.BaseEstimator)`
  (when sklearn is importable). Arbitrary fit/predict duck-typed objects must use
  `add_model` explicitly — pickling something is an explicit act.
- Uniform overwrite rule for **all** types: existing name ⇒ `overwrite=True` required,
  snapshotted or not (fixes the tables-vs-everything inconsistency).
- All invariants live here once: `_check_read_only`, name validation, overwrite check,
  copy-on-write, `_mutation_guard`, description-preservation, `_save_items`.

`get()` semantics:
- Tables: pandas by default, `frame='polars'` for eager polars; `polars_only` and
  `max_eager_bytes` guards as today.
- Artifacts: returns the payload **path** (documented — it is the only type whose
  "value" is a file).
- Everything else: the deserialized object.

### Kept explicit verbs

```python
folio.add_model(name, model, *, custom=False, hyperparameters=None, ...)
folio.get_model(name)                       # only typed getter that survives
folio.add_file(path, name=None, category=None, ...)
folio.reference_table(name, path, ...)      # linking ≠ storing; different verb is right
folio.inspect_table(name)
folio.scan_table(name, **kwargs)            # genuinely lazy
```

Everything else that survives unchanged: `metadata`, `data` accessor, `describe`,
`list_contents`, `validate`/`is_valid`, `batch`, `refresh`, lineage methods
(`get_inputs`, `get_dependents`, `get_lineage_graph`), `copy`, snapshots API
(`create_snapshot`, `snapshots`, `list_snapshots`, `delete_snapshot`,
`compare_snapshots`, `diff_from_snapshot`, `restore_snapshot`, `export_snapshot`,
`load_snapshot`, `get_snapshot`, `get_snapshot_info`, `mutable_references`,
`reproduce_instructions`, `cleanup_orphaned_versions`).

### Removed in 2.0

- `add_data`, `add_table`, `add_numpy`, `add_json`, `add_timestamp`, `add_sklearn`,
  `add_artifact` (folded into `add`; `add_file` remains the file verb)
- `get_data`, `get_table`, `get_numpy`, `get_json`, `get_timestamp`, `get_sklearn`,
  `get_lazy` (alias), `get_artifact_path`
- `get_data_path`, `get_item_path`, `get_table_path`, `get_model_path`,
  `get_numpy_path`, `get_json_path`, `get_timestamp_path` → `item_path`
- `get_table_info`, `get_model_info`, `get_artifact_info` → `item_info`
  (`get_table_info` callers who want the eager-size fields use `inspect_table`)
- `cache_status`, `clear_cache`, `invalidate_cache`, `refresh_cache`, and the whole
  `datafolio.cache` package
- The 14 dead private I/O helpers in `folio.py` (lines ~611–980)

Net effect on `DataFolio`: ~25 public methods removed, ~4 added.

---

## Phases

Ordered so each phase lands green and reviewable on its own. Phase 0 is manual;
1–3 are fixes on the existing API (so regression tests are written against
current method names and then mechanically migrated in phase 4).

### Phase 0 — baseline (you)
Commit or land the in-flight `feature/reference-tables-polars-lazy` work. The 2.0
branch starts from that state (the polars/reference features are part of the 2.0
story).

### Phase 1 — cut the cache (mechanical, independent)
- Delete `src/datafolio/cache/` (5 files, ~1,100 lines).
- Delete `_get_with_cache`, `_cache_*` state, constructor kwargs, the four
  `cache_*` methods; getters call handlers directly.
- Delete `tests/test_cache_manager.py`, `test_cache_validation.py`,
  `test_cache_integration.py` (keep `test_refresh.py` — that's manifest refresh).
- Docs: remove caching sections (README features list, getting-started, API ref).
- Kills audit findings: cache H3–H6, M1–M6, `refresh_cache` dead branch.

### Phase 2 — infrastructure correctness fixes
1. **Item-name grammar** (`utils.validate_item_name`): segments of
   `[A-Za-z0-9][A-Za-z0-9._-]*` joined by `/` (namespacing like `examples/weights`
   stays supported — `describe('examples/*')` is documented); reject `..` and `.`
   segments, leading `_`, leading `/`, empty segments. Payload filenames encode `/`
   (e.g. `examples/weights` → `examples__weights--r7.parquet`) so payloads never
   leave the category dir. Fixes path traversal (handlers H1).
2. **Accessor hardening** (`accessors.py`): only `setattr` names that are valid
   identifiers, don't start with `_`, and don't shadow class attributes; others
   remain reachable via `folio.data['name']`. Fixes the `_folio` RecursionError
   class; the leading-`_` name rejection in (1) makes it moot for new items but
   old manifests may contain anything.
3. **`StorageBackend.exists()`**: replace bare `except:` with specific cloudfiles
   exceptions; re-raise auth/permission errors instead of returning False (a
   transient error must never look like "bundle absent" — that path re-inits over
   real data). Replace the `"." in name` file/dir heuristic with: exact-key stat
   first, then a `list(prefix, limit=1)` fallback.
4. **`StorageBackend.read_parquet()` cloud path**: route through cloudfiles
   (download to temp file) instead of raw `pd.read_parquet(s3://...)` — one
   credential chain for read and write, `use_https` honored.
5. **`CloudFiles.get() is None`** → `FileNotFoundError` in `read_joblib`,
   `read_numpy`, `read_skops`; `try/finally` around numpy temp files.
6. **`MetadataDict`**: add `pop`, `popitem`, `__ior__` overrides (read-only check +
   auto-save).
7. **`diff_from_snapshot(None)`** and CLI `snapshot status`/`diff`: `[-1]` → `[0]`.
8. **Git capture**: run in `Path.cwd()` (the code repo), not the bundle dir.
9. **CLI `init`**: don't `Path.resolve()` cloud paths.

### Phase 3 — snapshot integrity
The theme: versioned payload filenames already mean *no operation ever needs to
copy or move snapshot bytes* — several bugs come from pre-versioning copy logic.
1. **`delete()`**: enter `_mutation_guard`; if the item is snapshotted, mark
   `is_current=False` and move the descriptor to `_snapshot_versions` (payload
   retained) instead of unlinking; only unsnapshotted payloads are deleted.
   Enforce read-only.
2. **`restore_snapshot()`**: rewrite as manifest surgery — for each pinned
   `version_id`, repoint `_items[name]` at that descriptor (payload already on
   disk); re-add items deleted since the snapshot; remove exception swallowing;
   no delete-before-copy (fixes the cloud data-loss path since nothing is copied).
3. **`export_snapshot()`**: copy payload **files** directly (no get/add
   round-trip); references exported as references; descriptions, lineage, and
   type-specific metadata preserved verbatim.
4. **`create_snapshot()` inside `batch()`**: raise a clear error ("commit the
   batch first"). Simpler and more honest than deferred snapshot writes.
5. **`update_item()`**: copy-on-write before mutating a snapshotted item; fix the
   docstring (empty string clears, `None` is no-op).
6. **Read-only enforcement**: `_check_read_only()` moves into `_commit_owned_item`
   / `_mutation_guard` entry, covering every mutation uniformly.
7. Regression tests for each (the audit repro scripts become tests):
   delete-then-read-snapshot, restore-after-delete, export-with-reference,
   snapshot-in-batch, read-only per mutation, multi-snapshot ordering.

### Phase 4 — API consolidation (the 2.0 break)
1. Implement `add`/`get`/`item_path`/`item_info` on the guarded commit path;
   auto-detection changes (strings, BaseEstimator) land here.
2. Delete the removed methods and the dead I/O helper block.
3. Split `folio.py` (~6,000 lines) into modules: `folio.py` (core + item ops),
   `snapshots.py` (SnapshotView/Accessor + snapshot methods as a mixin),
   `context.py` (git/env/execution capture). Pure moves, no behavior change.
4. Update `accessors.py` (ItemProxy.content → `folio.get`), `display.py`, and
   `cli/main.py` call sites.
5. Handler round-trip fixes that surface through `add()`:
   - JSON NaN/inf: keep orjson's `null` coercion (resolved decision 4) and
     document it in the `add()` and JSON-handler docstrings;
   - reject object-dtype numpy arrays at add time (today: write succeeds, read
     always fails);
   - warn (once) when a non-default DataFrame index is dropped, with
     `preserve_index=True` opt-in stored in metadata;
   - skops loading requires explicit `trusted=True` (or a types list) instead of
     auto-trusting everything.
6. Bump `MANIFEST_SCHEMA_VERSION`? **No** — nothing on disk changes.

### Phase 5 — tests, docs, release
1. Mechanical test migration (`add_table(` → `add(`, etc.); the suite's 824 tests
   are the safety net — port, don't rewrite.
2. Docs: README (features, quick start, badges, requirements list), guides,
   ARCHITECTURE.md, API reference, the generated in-bundle README (fix the
   `ceesem`/`caseysm` URL drift), and a short `docs/guides/migrating-to-2.0.md`
   (a 20-line before/after table covers almost all of it).
3. `poe bump major` → 2.0.0.

### Phase 6 — deferred (post-2.0, tracked but not blocking)
CLI polish (errors to stderr, `gc` confirmation, `init --description` on existing
bundles, dedupe timestamp/diff rendering), manifest file mode vs. umask,
cloud checksums, auto-refresh debounce for cloud bundles, display fixes
(snapshot describe shows pinned metadata, timestamps in listing).

---

## Resolved decisions (Casey, 2026-07-17)

1. **`item_path` / `item_info`** — approved as named.
2. **Artifact naming** — the "artifact" name is dropped from the public API
   (`add_file` is the only file verb; item_type stays `artifact` on disk).
3. **Snapshotted `delete()`** — preserves the version silently, matching
   overwrite-CoW semantics.
4. **NaN policy for JSON** — keep the current orjson behavior
   (`OPT_SERIALIZE_NUMPY`; NaN/inf serialize as `null`) as the default rather
   than rejecting — ambiguous intent shouldn't be an error. Phase 4 documents the
   coercion in `add()`/JSON handler docstrings instead of adding validation.

## Estimated shape

| Phase | Scope | Net LOC (src) |
|---|---|---|
| 1 | cache cut | −1,300 |
| 2 | infra fixes | ±150 |
| 3 | snapshot integrity | ±300 |
| 4 | API consolidation + split | −2,000 to −2,500 |
| 5 | tests/docs | large but mechanical |

End state: `folio.py` on the order of 2,000 lines across three focused modules,
~8 core public methods plus the table/model verbs, one mutation path, one read
path, on-disk format untouched.
