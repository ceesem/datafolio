"""Per-user registry of folio aliases and recently used folios.

The registry lives in ``~/.datafolio/registry.json`` (override the directory
with the ``DATAFOLIO_HOME`` environment variable). It holds two things:

- **aliases**: explicit, user-chosen names for folios (``my-folio`` →
  ``/data/analysis/my-folio`` or ``gs://bucket/my-folio``). Created only on
  request — ``DataFolio(path, alias=...)``, ``folio.set_alias(...)``,
  :func:`set_alias`, or ``datafolio folios alias``.
- **recent**: a capped most-recently-used list of folio paths, recorded by the
  CLI only. The Python API never records recents, so opening a folio in a
  script or notebook has no side effects on the user's home directory.

This is unrelated to :mod:`datafolio.base.registry`, which is the registry of
item-type *handlers*.
"""

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import orjson
from filelock import FileLock

from datafolio.utils import is_cloud_path, validate_snapshot_name

logger = logging.getLogger(__name__)

REGISTRY_VERSION = 1
REGISTRY_FILE = "registry.json"

# Maximum number of entries kept in the recent list (least recent dropped).
MAX_RECENT = 50

# Seconds to wait for the registry lock before giving up.
LOCK_TIMEOUT = 10.0


def registry_home() -> Path:
    """Return the registry directory (``$DATAFOLIO_HOME`` or ``~/.datafolio``).

    Returns:
        Path to the directory holding ``registry.json``.

    Examples:
        >>> registry_home()
        PosixPath('/Users/me/.datafolio')
    """
    env = os.environ.get("DATAFOLIO_HOME")
    return Path(env).expanduser() if env else Path.home() / ".datafolio"


def normalize_folio_path(path: Union[str, Path]) -> str:
    """Normalize a folio path for storage in the registry.

    Local paths become absolute and resolved (``~`` expanded); cloud URIs are
    kept verbatim apart from a trailing slash. A ``file://`` URI is treated as
    the local path it names.

    Args:
        path: Local path or cloud URI.

    Returns:
        Normalized path string.

    Examples:
        >>> normalize_folio_path('gs://bucket/folio/')
        'gs://bucket/folio'
        >>> normalize_folio_path('~/data/folio')
        '/Users/me/data/folio'
    """
    path_str = str(path)
    if path_str.startswith("file://"):
        path_str = path_str[len("file://") :]
    if is_cloud_path(path_str):
        return path_str.rstrip("/")
    return str(Path(path_str).expanduser().resolve())


def validate_alias(alias: str) -> None:
    """Validate an alias name.

    Aliases follow the snapshot-name rules: letters, digits, ``.``, ``_``
    and ``-`` only.

    Args:
        alias: Alias to validate.

    Raises:
        ValueError: If the alias is empty or has invalid characters.
    """
    try:
        validate_snapshot_name(alias)
    except (ValueError, TypeError) as exc:
        raise ValueError(str(exc).replace("Snapshot name", "Alias")) from None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_local_missing(path: str) -> bool:
    return not is_cloud_path(path) and not Path(path).exists()


class FolioRegistry:
    """Read and edit the per-user folio registry.

    Every method reads the registry file fresh, so separate instances (and
    separate processes) always see each other's changes. Writes take a file
    lock and replace the file atomically.

    Args:
        home: Registry directory. Defaults to :func:`registry_home`.

    Examples:
        >>> reg = FolioRegistry()
        >>> reg.set_alias('my-folio', '~/analysis/my-folio')
        >>> reg.get_alias('my-folio')
        '/Users/me/analysis/my-folio'
    """

    def __init__(self, home: Optional[Union[str, Path]] = None) -> None:
        self.home: Path = Path(home) if home is not None else registry_home()
        self.path: Path = self.home / REGISTRY_FILE

    # ── low-level I/O ────────────────────────────────────────────────────────

    def _empty(self) -> dict[str, Any]:
        return {"version": REGISTRY_VERSION, "aliases": {}, "recent": []}

    def _read(self) -> dict[str, Any]:
        """Read the registry, tolerating a missing or corrupt file."""
        try:
            data = orjson.loads(self.path.read_bytes())
        except FileNotFoundError:
            return self._empty()
        except (OSError, orjson.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable folio registry %s: %s", self.path, exc)
            return self._empty()
        if not isinstance(data, dict):
            return self._empty()
        data.setdefault("version", REGISTRY_VERSION)
        if not isinstance(data.get("aliases"), dict):
            data["aliases"] = {}
        if not isinstance(data.get("recent"), list):
            data["recent"] = []
        return data

    def _write(self, data: dict[str, Any]) -> None:
        """Atomically replace the registry file (caller holds the lock)."""
        self.home.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.home, prefix=".registry-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _lock(self) -> FileLock:
        self.home.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.path) + ".lock", timeout=LOCK_TIMEOUT)

    # ── aliases ──────────────────────────────────────────────────────────────

    def set_alias(
        self, alias: str, path: Union[str, Path], overwrite: bool = False
    ) -> str:
        """Register ``alias`` as a name for the folio at ``path``.

        Args:
            alias: Alias name (letters, digits, ``.``, ``_``, ``-``).
            path: Folio path (local or cloud).
            overwrite: Rebind the alias if it already points elsewhere.

        Returns:
            The normalized path the alias now points to.

        Raises:
            ValueError: If the alias is invalid, or already points to a
                different path and ``overwrite`` is False.

        Examples:
            >>> FolioRegistry().set_alias('demo', '/data/demo')
            '/data/demo'
        """
        validate_alias(alias)
        target = normalize_folio_path(path)
        with self._lock():
            data = self._read()
            existing = data["aliases"].get(alias)
            if existing is not None:
                if existing.get("path") == target:
                    return target
                if not overwrite:
                    raise ValueError(
                        f"Alias '{alias}' already points to {existing.get('path')}. "
                        f"Pass overwrite=True (CLI: --overwrite) to rebind it."
                    )
            data["aliases"][alias] = {"path": target, "created_at": _now()}
            self._write(data)
        return target

    def get_alias(self, alias: str) -> str:
        """Return the path registered for ``alias``.

        Args:
            alias: Alias name.

        Returns:
            The registered folio path.

        Raises:
            KeyError: If the alias is not registered.
        """
        aliases = self.aliases()
        if alias not in aliases:
            known = ", ".join(sorted(aliases)) or "(none)"
            raise KeyError(f"No folio registered as '{alias}'. Known aliases: {known}")
        return aliases[alias]

    def remove_alias(self, alias: str) -> None:
        """Remove an alias. The folio itself is untouched.

        Args:
            alias: Alias name.

        Raises:
            KeyError: If the alias is not registered.
        """
        with self._lock():
            data = self._read()
            if alias not in data["aliases"]:
                raise KeyError(f"No folio registered as '{alias}'")
            del data["aliases"][alias]
            self._write(data)

    def aliases(self) -> dict[str, str]:
        """Return all aliases as ``{alias: path}``, sorted by alias."""
        data = self._read()
        return {
            name: entry["path"]
            for name, entry in sorted(data["aliases"].items())
            if isinstance(entry, dict) and "path" in entry
        }

    def aliases_for(self, path: Union[str, Path]) -> list[str]:
        """Return every alias pointing at ``path``.

        Args:
            path: Folio path (normalized before comparison).

        Returns:
            Sorted list of alias names (empty if none).
        """
        target = normalize_folio_path(path)
        return [name for name, p in self.aliases().items() if p == target]

    # ── recents ──────────────────────────────────────────────────────────────

    def record_recent(self, path: Union[str, Path]) -> None:
        """Record that the folio at ``path`` was just used.

        Best-effort: any failure (read-only home directory, lock timeout, ...)
        is logged at debug level and swallowed, so it can never break the
        command that triggered it.

        Args:
            path: Folio path (local or cloud).
        """
        try:
            target = normalize_folio_path(path)
            with self._lock():
                data = self._read()
                recent = [
                    e
                    for e in data["recent"]
                    if isinstance(e, dict) and e.get("path") != target
                ]
                recent.insert(0, {"path": target, "last_accessed": _now()})
                data["recent"] = recent[:MAX_RECENT]
                self._write(data)
        except Exception as exc:  # noqa: BLE001 - must never propagate
            logger.debug("Could not record recent folio %s: %s", path, exc)

    def recent(self) -> list[dict[str, str]]:
        """Return recent folios, most recent first.

        Returns:
            List of ``{'path': ..., 'last_accessed': ...}`` dicts.
        """
        return [
            {"path": e["path"], "last_accessed": e.get("last_accessed", "")}
            for e in self._read()["recent"]
            if isinstance(e, dict) and "path" in e
        ]

    def forget(self, path: Union[str, Path]) -> bool:
        """Remove a path from the recent list (aliases are kept).

        Args:
            path: Folio path.

        Returns:
            True if the path was in the recent list.
        """
        target = normalize_folio_path(path)
        with self._lock():
            data = self._read()
            before = len(data["recent"])
            data["recent"] = [
                e
                for e in data["recent"]
                if not (isinstance(e, dict) and e.get("path") == target)
            ]
            removed = len(data["recent"]) != before
            if removed:
                self._write(data)
        return removed

    def prune(self) -> list[str]:
        """Drop local entries (aliases and recents) whose path no longer exists.

        Cloud paths are never pruned: checking them needs network access and
        a transient failure must not delete an alias.

        Returns:
            Sorted list of the paths that were removed.
        """
        removed: set[str] = set()
        with self._lock():
            data = self._read()
            aliases = {}
            for name, entry in data["aliases"].items():
                p = entry.get("path", "") if isinstance(entry, dict) else ""
                if p and _is_local_missing(p):
                    removed.add(p)
                else:
                    aliases[name] = entry
            recent = []
            for entry in data["recent"]:
                p = entry.get("path", "") if isinstance(entry, dict) else ""
                if p and _is_local_missing(p):
                    removed.add(p)
                else:
                    recent.append(entry)
            if removed:
                data["aliases"] = aliases
                data["recent"] = recent
                self._write(data)
        return sorted(removed)

    # ── combined views ───────────────────────────────────────────────────────

    def all_targets(
        self, aliases_only: bool = False
    ) -> list[tuple[Optional[str], str]]:
        """Return every known folio as ``(alias, path)`` pairs, deduplicated.

        Aliased folios come first (sorted by alias), then recents not already
        covered by an alias (most recent first). A folio with several aliases
        is listed once, under its alphabetically first alias.

        Args:
            aliases_only: Only return aliased folios.

        Returns:
            List of ``(alias or None, path)`` tuples.
        """
        data = self._read()
        seen: set[str] = set()
        out: list[tuple[Optional[str], str]] = []
        for name, entry in sorted(data["aliases"].items()):
            p = entry.get("path") if isinstance(entry, dict) else None
            if p and p not in seen:
                seen.add(p)
                out.append((name, p))
        if not aliases_only:
            for entry in data["recent"]:
                p = entry.get("path") if isinstance(entry, dict) else None
                if p and p not in seen:
                    seen.add(p)
                    out.append((None, p))
        return out


# ── module-level conveniences (exported from ``datafolio``) ──────────────────


def set_alias(alias: str, path: Union[str, Path], overwrite: bool = False) -> str:
    """Register ``alias`` for the folio at ``path`` in the user registry.

    Args:
        alias: Alias name (letters, digits, ``.``, ``_``, ``-``).
        path: Folio path (local or cloud).
        overwrite: Rebind the alias if it already points elsewhere.

    Returns:
        The normalized path the alias points to.

    Examples:
        >>> import datafolio
        >>> datafolio.set_alias('demo', '~/analysis/demo')
        >>> folio = datafolio.DataFolio(alias='demo')
    """
    return FolioRegistry().set_alias(alias, path, overwrite=overwrite)


def remove_alias(alias: str) -> None:
    """Remove an alias from the user registry (the folio is untouched).

    Args:
        alias: Alias name.

    Examples:
        >>> datafolio.remove_alias('demo')
    """
    FolioRegistry().remove_alias(alias)


def list_folios() -> Any:
    """List registered aliases and recently used folios as a DataFrame.

    Returns:
        pandas DataFrame with columns ``alias``, ``path``, ``last_accessed``
        and ``exists`` (``None`` for cloud paths, which are not checked).
        Aliased folios come first.

    Examples:
        >>> datafolio.list_folios()
             alias                    path  last_accessed  exists
        0     demo  /Users/me/analysis/demo           None    True
    """
    import pandas as pd

    reg = FolioRegistry()
    last = {e["path"]: e["last_accessed"] for e in reg.recent()}
    rows = []
    for alias, path in reg.all_targets():
        rows.append(
            {
                "alias": alias,
                "path": path,
                "last_accessed": last.get(path),
                "exists": None if is_cloud_path(path) else Path(path).exists(),
            }
        )
    return pd.DataFrame(rows, columns=["alias", "path", "last_accessed", "exists"])
