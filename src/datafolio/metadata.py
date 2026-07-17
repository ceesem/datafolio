"""Metadata dictionary with auto-save functionality.

This module provides MetadataDict, a specialized dictionary that automatically
commits to the bundle whenever it's modified. Metadata participates in the
same single-writer protocol as items: each mutation enters the folio's
mutation guard (so a stale notebook fails BEFORE its in-memory state changes)
and commits through the items manifest, advancing the bundle revision that
other notebooks detect.
"""

import contextlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class MetadataDict(dict):
    """Dictionary that auto-commits to the bundle on any modification.

    This class extends dict to automatically trigger a guarded commit on the
    parent DataFolio whenever the metadata is modified. It also automatically
    updates the 'updated_at' timestamp on commit.

    Examples:
        >>> folio = DataFolio('experiment')
        >>> folio.metadata['experiment_name'] = 'test'
        # Automatically commits to metadata.json (and advances the bundle
        # revision in items.json)

        >>> folio.metadata.update({'author': 'Alice', 'version': '1.0'})
        # One commit for the whole update
    """

    def __init__(self, parent: "DataFolio", *args, **kwargs):
        """Initialize MetadataDict with parent reference.

        Args:
            parent: Parent DataFolio instance for callbacks
            *args: Positional arguments for dict
            **kwargs: Keyword arguments for dict
        """
        # Initialize parent AFTER super().__init__() to avoid triggering saves during initialization
        super().__init__(*args, **kwargs)
        self._parent = parent

    def _check_writable(self) -> None:
        """Raise if the parent folio is read-only (no-op during __init__)."""
        if hasattr(self, "_parent") and self._parent._read_only:
            raise RuntimeError("Cannot modify metadata of a read-only DataFolio")

    def _mutation(self):
        """The parent folio's guarded metadata-commit scope.

        Falls back to a null context during ``__init__`` (before ``_parent``
        exists), where mutations are plain dict operations.
        """
        if hasattr(self, "_parent"):
            return self._parent._metadata_mutation()
        return contextlib.nullcontext()

    def _touch(self) -> None:
        """Bump ``updated_at`` in-memory (persisted by the commit)."""
        if hasattr(self, "_parent"):
            super().__setitem__("updated_at", datetime.now(timezone.utc).isoformat())

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item and commit."""
        self._check_writable()
        with self._mutation():
            super().__setitem__(key, value)
            if key != "updated_at":
                self._touch()

    def __delitem__(self, key: str) -> None:
        """Delete item and commit."""
        self._check_writable()
        with self._mutation():
            super().__delitem__(key)
            self._touch()

    def update(self, *args, **kwargs) -> None:
        """Update dict and commit once."""
        self._check_writable()
        with self._mutation():
            super().update(*args, **kwargs)
            self._touch()

    def __ior__(self, other: Any) -> "MetadataDict":
        """Support ``metadata |= other`` with commit and read-only semantics."""
        self._check_writable()
        with self._mutation():
            super().update(other)
            self._touch()
        return self

    def clear(self) -> None:
        """Clear dict and commit."""
        self._check_writable()
        with self._mutation():
            super().clear()
            self._touch()

    def pop(self, key: str, *default: Any) -> Any:
        """Pop a key and commit (respects read-only mode)."""
        if key in self:
            self._check_writable()
            with self._mutation():
                value = super().pop(key)
                self._touch()
            return value
        return super().pop(key, *default)

    def popitem(self) -> tuple:
        """Pop an arbitrary item and commit (respects read-only mode)."""
        self._check_writable()
        with self._mutation():
            result = super().popitem()
            self._touch()
        return result

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default and commit if the key was added."""
        if key in self:
            return super().setdefault(key, default)
        self._check_writable()
        with self._mutation():
            result = super().setdefault(key, default)
            self._touch()
        return result
