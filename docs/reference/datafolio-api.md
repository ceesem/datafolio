---
title: API cheat sheet
---

# API cheat sheet

Every public method, grouped by what you are trying to do. Full docstrings are
on the [Package API](api.md) page and in `help(DataFolio)`.

The shape of the API in one line: **one write verb (`add`), one read verb
(`get`), and a short list of explicit verbs for the cases where guessing would
be wrong.**

## Open a folio

```python
DataFolio(
    path,                                # local path or gs:// / s3:// URI
    metadata=None,                       # INITIAL folio metadata; ignored when
                                         # opening an existing folio (write via
                                         # folio.metadata instead)
    random_suffix=False,                 # append a memorable suffix to the name
    read_only=False,                     # refuse every write
    allow_existing=False,                # allow creating inside a non-folio directory
    use_https=False,                     # HTTPS reads of public buckets
    max_eager_bytes=500 * 1024 * 1024,   # eager-read ceiling; None disables
    alias=None,                          # register under this alias; or, with
                                         # no path, open the aliased folio
    overwrite_alias=False,               # rebind an alias that points elsewhere
)
```

`DataFolio(alias="exp12")` opens a registered folio. `DataFolio(path)` alone
never touches the registry.

Creates the folio if the path is empty, opens it if it already holds one.
`http(s)://` paths are always read-only.

```python
DataFolio.load_snapshot(path, snapshot)   # classmethod -> read-only folio
```

## Add items

| Call | For |
| --- | --- |
| `add(name, obj, *, description=None, inputs=None, overwrite=False, **type_opts)` | DataFrames, Series, arrays, dicts/lists/tuples/sets/scalars/strings, dataclasses, tz-aware datetimes, estimators |
| `add_model(name, model, *, description=None, inputs=None, overwrite=False, custom=False)` | any picklable model-like object; `custom=True` uses skops |
| `add_file(path, name=None, *, category=None, description=None, overwrite=False)` | copy a file into the folio |
| `import_table(name, path, *, table_format=None, description=None, inputs=None, overwrite=False, block_size=None)` | parquet/CSV/feather file → table, without loading it into memory |
| `reference_table(name, path, table_format="parquet", num_rows=None, version=None, description=None, inputs=None, overwrite=False, allow_full_load=False, polars_only=None)` | link an external table without copying |

`**type_opts` for `add()`: `preserve_index=True` (tables), `custom=True`
(models), `category=...` (files). Unknown options raise `TypeError`.

A pandas Series is stored as a one-column table that includes its index, and
`get()` returns it as a Series again. JSON has no tuples, sets, or non-string
keys, so tuples and ranges load back as lists. Sets load back as sorted lists
and non-string dict keys as strings; those two conversions emit a
`UserWarning`.
A dataclass instance is stored as the JSON of its fields, and `get()` returns
that dict. Rebuild it with `MyConfig(**folio.get("cfg"))`.

Replacing any existing item requires `overwrite=True`. A version pinned by a
snapshot is preserved via copy-on-write.

## Read items

| Call | Returns |
| --- | --- |
| `get(name, **type_opts)` | the natural object; a **path** for file items |
| `get_many(names, *, threads=20, **type_opts)` | `{name: object}`, read concurrently |
| `get_model(name, trusted=False)` | the loaded model (`trusted=True` for skops) |
| `scan_table(name, **kwargs)` | a genuinely lazy `polars.LazyFrame` |
| `item_path(name)` | the payload path (external path for references) |
| `item_info(name)` | a copy of the catalog entry |

`**type_opts` for `get()`: `frame="pandas"|"polars"` and `allow_full_load=True`
(tables), `as_unix=True` (timestamps), `trusted=True` (models).

```python
with folio.pinned():     # one staleness check for the whole block
    ...
folio.refresh()          # force a reload from disk
```

## Inspect

```python
folio.describe(pattern=None, return_string=False, show_empty=False,
               max_metadata_fields=10, snapshot=None,
               include_archived=False, show_paths=False)

folio.list_contents(include_archived=False)   # {kind: [names]}
folio.tables                                  # names of included + referenced tables
folio.models
folio.artifacts                               # file items
"features" in folio                           # True exactly when get() would work

folio.inspect_table(name)     # read the source, record schema/size/identity
folio.validate()              # {name: bool}
folio.is_valid()              # all of the above
```

Properties: `path`, `metadata`, `read_only`, `in_snapshot_mode`,
`loaded_snapshot`, `data`.

### The `data` accessor

```python
folio.data.features.content        # == folio.get('features')
folio.data.features.lazy           # == folio.scan_table('features')
folio.data.features.description
folio.data.features.type
folio.data.features.path
folio.data.features.inputs
folio.data.features.dependents
folio.data.features.metadata
folio.data["qc/step1"].content     # names that are not valid identifiers
```

Built for tab completion in Jupyter. Equivalent to the plain methods.

## Change and remove

```python
folio.update_item(name, description=None, inputs=None)   # "" / [] clear a field
folio.delete(name_or_names, warn_dependents=True)
folio.archive(name_or_names_or_glob)
folio.unarchive(name_or_names_or_glob)

folio.metadata["key"] = value        # commits immediately
folio.metadata.update({...})
folio.metadata = {...}               # wholesale replacement

with folio.batch():                  # one commit for the whole block
    ...
```

## Lineage

```python
folio.get_inputs(name)          # what this was derived from
folio.get_dependents(name)      # what was derived from this
folio.get_lineage_graph()       # {name: [inputs]}
```

Recorded by `inputs=` at write time (or `update_item`). Documentation, not
enforcement.

## Copy and export

```python
folio.copy(path, name=None, metadata_updates=None, include_items=None,
           exclude_items=None, random_suffix=False, follow_lineage=False,
           include_archived=False)
```

A **fork**: current versions only, no snapshot history, archived items excluded,
fresh identity. To *mirror* a folio, use `rsync`/`gsutil rsync`/`aws s3 sync`.

## Snapshots

```python
folio.create_snapshot(name, description=None, tags=None,
                      capture_git=True, capture_environment=False,
                      capture_execution=False)

folio.snapshots                     # dict-like accessor -> SnapshotView
folio.list_snapshots()              # [{name, timestamp, description, tags, …}]
folio.get_snapshot_info(name)       # full record incl. pinned versions and git
folio.mutable_references()          # references whose bytes are not owned

folio.get_snapshot(name)                        # read-only DataFolio
DataFolio.load_snapshot(path, name)             # same, without an open folio

folio.compare_snapshots(a, b)
folio.diff_from_snapshot(name=None)             # None -> newest snapshot

folio.restore_snapshot(name, confirm=True)      # DESTRUCTIVE
folio.export_snapshot(name, target_path, *, include_snapshot_metadata=True)

folio.delete_snapshot(name, cleanup_orphans=False)
folio.cleanup_orphaned_versions(dry_run=False)
```

A `SnapshotView` (`folio.snapshots["v1"]`) exposes `name`, `description`,
`timestamp`, `tags`, `metadata`, `item_versions`, `get(name)`,
`get_table(name)`, and `scan_table(name)`.

## Exceptions

| Exception | Raised when |
| --- | --- |
| `ConcurrentWriteError` | another writer advanced the catalog; `refresh()` and retry |
| `ManifestReadError` | the catalog is missing, unreadable, or was replaced externally |
| `UnsupportedManifestVersionError` | the folio was written by a newer DataFolio |

```python
from datafolio import (
    DataFolio, ConcurrentWriteError, ManifestReadError,
    UnsupportedManifestVersionError,
)
```

Everything else in the package — handlers, storage backends, readers — is
internal.

## Common option quick-reference

| Option | Where | Effect |
| --- | --- | --- |
| `overwrite=True` | every add verb | required to replace an existing item |
| `inputs=[...]` | add verbs, `update_item` | lineage |
| `description="..."` | add verbs, `update_item` | the sentence future-you reads |
| `preserve_index=True` | `add` (tables) | store a non-default pandas index as columns |
| `custom=True` | `add_model` | skops instead of joblib |
| `frame="polars"` | `get` (tables) | Polars DataFrame instead of pandas |
| `allow_full_load=True` | `get` (tables), `reference_table` | bypass the eager-size guard |
| `trusted=True` | `get_model` | accept a skops file's non-standard types |
| `as_unix=True` | `get` (timestamps) | float seconds instead of a datetime |
| `read_only=True` | constructor | refuse every write |
| `follow_lineage=True` | `copy` | pull in upstream dependencies of `include_items` |
| `dry_run=True` | `cleanup_orphaned_versions` | list, do not delete |

## Folio registry and search

Module-level functions for the per-user registry in `~/.datafolio`
(`DATAFOLIO_HOME` overrides the location).

| Call | For |
| --- | --- |
| `folio.set_alias(alias, overwrite=False)` | register this folio under an alias |
| `datafolio.set_alias(alias, path, overwrite=False)` | register any folio path or URI |
| `datafolio.remove_alias(alias)` | remove an alias (the folio is untouched) |
| `datafolio.list_folios()` | DataFrame of aliases and CLI-recent folios |
| `datafolio.find(pattern="*", *, regex=False, item_type=None, metadata=False, folios=None, aliases_only=False, local_only=False, include_archived=False, case_sensitive=False, descriptions=False)` | DataFrame of matching items (or metadata keys) across the registry |
