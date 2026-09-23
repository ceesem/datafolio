"""Search items and metadata across every folio in the user registry.

Opening a folio only reads its small ``items.json`` manifest, so searching
dozens of folios is cheap — no table or model data is ever loaded.

Examples:
    >>> import datafolio
    >>> datafolio.find('cells*', item_type='table')
    >>> datafolio.find('synapse', regex=True, folios=['v1', 'v2'])
    >>> datafolio.find('dataset=minnie*', metadata=True)
"""

import fnmatch
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional, Union

from datafolio.folio_registry import FolioRegistry, normalize_folio_path
from datafolio.utils import ITEMS_FILE, METADATA_FILE, is_cloud_path

# Friendly item-type names accepted by ``find(item_type=...)`` and the CLI
# ``--type`` option. Exact item_type values are accepted too.
ITEM_TYPE_GROUPS: dict[str, frozenset[str]] = {
    "table": frozenset({"included_table", "referenced_table"}),
    "model": frozenset({"model"}),
    "artifact": frozenset({"artifact"}),
    "array": frozenset({"numpy_array"}),
    "json": frozenset({"json_data"}),
    "timestamp": frozenset({"timestamp"}),
}

ITEM_COLUMNS = ["alias", "folio_path", "name", "item_type", "description", "created_at"]
METADATA_COLUMNS = ["alias", "folio_path", "key", "value"]

DEFAULT_WORKERS = 16


def _resolve_item_types(
    item_type: Optional[Union[str, list[str]]],
) -> Optional[set[str]]:
    if item_type is None:
        return None
    names = [item_type] if isinstance(item_type, str) else list(item_type)
    known = set().union(*ITEM_TYPE_GROUPS.values())
    types: set[str] = set()
    for t in names:
        if t in ITEM_TYPE_GROUPS:
            types |= ITEM_TYPE_GROUPS[t]
        elif t in known:
            types.add(t)
        else:
            raise ValueError(
                f"Unknown item type '{t}'. Use one of "
                f"{sorted(ITEM_TYPE_GROUPS)} or an exact type {sorted(known)}."
            )
    return types


def _matcher(pattern: str, regex: bool, case_sensitive: bool) -> Callable[[str], bool]:
    if regex:
        compiled = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        return lambda s: compiled.search(s) is not None
    if case_sensitive:
        return lambda s: fnmatch.fnmatchcase(s, pattern)
    lowered = pattern.lower()
    return lambda s: fnmatch.fnmatchcase(s.lower(), lowered)


def _contains_matcher(
    pattern: str, regex: bool, case_sensitive: bool
) -> Callable[[str], bool]:
    """Match a pattern anywhere in free text (used for descriptions)."""
    if regex:
        return _matcher(pattern, regex, case_sensitive)
    return _matcher(f"*{pattern.strip('*')}*", regex, case_sensitive)


def _resolve_targets(
    registry: FolioRegistry,
    folios: Optional[Union[str, Path, list[Union[str, Path]]]],
    aliases_only: bool,
) -> list[tuple[Optional[str], str]]:
    if folios is None:
        return registry.all_targets(aliases_only=aliases_only)
    requested = [folios] if isinstance(folios, (str, Path)) else list(folios)
    aliases = registry.aliases()
    by_path: dict[str, str] = {}
    for name, p in aliases.items():
        by_path.setdefault(p, name)
    out: list[tuple[Optional[str], str]] = []
    seen: set[str] = set()
    for target in requested:
        target = str(target)
        if target in aliases:
            alias: Optional[str] = target
            path = aliases[target]
        else:
            path = normalize_folio_path(target)
            alias = by_path.get(path)
        if path not in seen:
            seen.add(path)
            out.append((alias, path))
    return out


def open_existing(path: str, read_only: bool = True) -> Any:
    """Open an existing folio without ever creating one.

    ``DataFolio(path)`` creates a new folio when none exists; searching (and
    CLI commands like ``add``) must not, so existence is checked first.

    Args:
        path: Folio path or cloud URI.
        read_only: Open read-only (default: True).

    Returns:
        The opened DataFolio.

    Raises:
        FileNotFoundError: If no folio exists at ``path``.
    """
    from datafolio.folio import DataFolio
    from datafolio.storage import StorageBackend

    storage = StorageBackend()
    if not is_cloud_path(path) and not Path(path).is_dir():
        raise FileNotFoundError(f"folio not found: {path}")
    if not (
        storage.exists(storage.join_paths(path, ITEMS_FILE))
        or storage.exists(storage.join_paths(path, METADATA_FILE))
    ):
        raise FileNotFoundError(f"not a DataFolio bundle: {path}")
    return DataFolio(path, read_only=read_only)


def _search_folio(
    alias: Optional[str],
    path: str,
    match: Callable[[str], bool],
    value_match: Optional[Callable[[str], bool]],
    types: Optional[set[str]],
    metadata: bool,
    include_archived: bool,
    desc_match: Optional[Callable[[str], bool]] = None,
) -> list[dict[str, Any]]:
    folio = open_existing(path)
    rows: list[dict[str, Any]] = []
    if metadata:
        for key, value in folio.metadata.items():
            if not match(str(key)):
                continue
            if value_match is not None and not value_match(str(value)):
                continue
            rows.append(
                {"alias": alias, "folio_path": path, "key": key, "value": value}
            )
        return rows

    for names in folio.list_contents(include_archived=include_archived).values():
        for name in names:
            info = folio.item_info(name)
            description = info.get("description") or ""
            if not (match(name) or (desc_match and desc_match(description))):
                continue
            if types is not None and info.get("item_type") not in types:
                continue
            rows.append(
                {
                    "alias": alias,
                    "folio_path": path,
                    "name": name,
                    "item_type": info.get("item_type"),
                    "description": info.get("description"),
                    "created_at": info.get("created_at"),
                }
            )
    return rows


def find(
    pattern: str = "*",
    *,
    regex: bool = False,
    item_type: Optional[Union[str, list[str]]] = None,
    metadata: bool = False,
    folios: Optional[Union[str, Path, list[Union[str, Path]]]] = None,
    aliases_only: bool = False,
    local_only: bool = False,
    include_archived: bool = False,
    case_sensitive: bool = False,
    descriptions: bool = False,
    max_workers: int = DEFAULT_WORKERS,
) -> Any:
    """Find items (or metadata keys) across the folios in the user registry.

    Searches every aliased and recently used folio (see
    :func:`datafolio.list_folios`) unless ``folios`` narrows the search. Only
    each folio's manifest is read, never item data. Folios are read
    concurrently; one that can't be opened (moved, deleted, no credentials)
    produces a ``UserWarning`` and is skipped.

    Args:
        pattern: Glob pattern matched against the whole item name (``'*'``
            matches everything). With ``metadata=True`` it matches metadata
            keys, and ``'key=value'`` also matches the value's string form.
        regex: Treat the pattern(s) as regular expressions, matched anywhere
            in the string (``re.search``) instead of as globs.
        item_type: Restrict to item types: ``'table'``, ``'model'``,
            ``'artifact'``, ``'array'``, ``'json'``, ``'timestamp'``, an exact
            item type such as ``'referenced_table'``, or a list of these.
            Ignored with ``metadata=True``.
        metadata: Search folio-level metadata keys instead of item names.
        folios: Alias(es) or path(s) to search instead of the whole registry.
        aliases_only: Search only aliased folios, not recents.
        local_only: Skip cloud folios.
        include_archived: Include archived items.
        case_sensitive: Match case-sensitively (default: case-insensitive).
        descriptions: Also match item descriptions. An item matches if its
            name matches the pattern *or* its description contains it
            (``'synapse'`` finds "the synapse table I liked"). Globs match
            anywhere in the description; regexes already do.
        max_workers: Number of folios read in parallel.

    Returns:
        pandas DataFrame, one row per match. Item searches have columns
        ``alias``, ``folio_path``, ``name``, ``item_type``, ``description``
        and ``created_at``; metadata searches have ``alias``, ``folio_path``,
        ``key`` and ``value``. Empty (with those columns) when nothing matches.

    Raises:
        ValueError: For an unknown ``item_type`` or an invalid regex.

    Examples:
        >>> datafolio.find('cells*')
        >>> datafolio.find('synapse|soma', regex=True, item_type='table')
        >>> datafolio.find('*', folios='demo', item_type='model')
        >>> datafolio.find('dataset=minnie*', metadata=True)
        >>> datafolio.find('synapse', descriptions=True)
    """
    import pandas as pd

    types = _resolve_item_types(item_type)
    value_match: Optional[Callable[[str], bool]] = None
    try:
        if metadata and "=" in pattern:
            key_pattern, value_pattern = pattern.split("=", 1)
            value_match = _matcher(value_pattern, regex, case_sensitive)
            match = _matcher(key_pattern, regex, case_sensitive)
        else:
            match = _matcher(pattern, regex, case_sensitive)
    except re.error as exc:
        raise ValueError(f"Invalid regex '{pattern}': {exc}") from None

    desc_match: Optional[Callable[[str], bool]] = None
    if descriptions and not metadata:
        desc_match = _contains_matcher(pattern, regex, case_sensitive)

    targets = _resolve_targets(FolioRegistry(), folios, aliases_only)
    if local_only:
        targets = [(a, p) for a, p in targets if not is_cloud_path(p)]

    def _run(
        target: tuple[Optional[str], str],
    ) -> Union[list[dict[str, Any]], Exception]:
        try:
            return _search_folio(
                target[0],
                target[1],
                match,
                value_match,
                types,
                metadata,
                include_archived,
                desc_match,
            )
        except Exception as exc:  # noqa: BLE001 - reported as a warning
            return exc

    rows: list[dict[str, Any]] = []
    if targets:
        with ThreadPoolExecutor(
            max_workers=max(1, min(max_workers, len(targets)))
        ) as pool:
            results = list(pool.map(_run, targets))
        for (alias, path), result in zip(targets, results):
            if isinstance(result, Exception):
                label = f"'{alias}' ({path})" if alias else path
                warnings.warn(
                    f"Skipping folio {label}: {result}", UserWarning, stacklevel=2
                )
            else:
                rows.extend(result)

    columns = METADATA_COLUMNS if metadata else ITEM_COLUMNS
    return pd.DataFrame(rows, columns=columns)
