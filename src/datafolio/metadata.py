"""Metadata dictionary with auto-save functionality.

This module provides MetadataDict, a specialized dictionary that automatically
saves to file whenever it's modified.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class MetadataDict(dict):
    """Dictionary that auto-saves to file on any modification.

    This class extends dict to automatically trigger saves to the parent
    DataFolio whenever the metadata is modified. It also automatically
    updates the 'updated_at' timestamp on modifications.

    Examples:
        >>> folio = DataFolio('experiment')
        >>> folio.metadata['experiment_name'] = 'test'
        # Automatically saves to metadata.json

        >>> folio.metadata.update({'author': 'Alice', 'version': '1.0'})
        # Automatically saves and updates timestamp
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

    def _touch_and_save(self) -> None:
        """Bump ``updated_at`` and persist (no-op during __init__)."""
        if hasattr(self, "_parent"):
            super().__setitem__("updated_at", datetime.now(timezone.utc).isoformat())
            self._parent._save_metadata()

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item and trigger save."""
        self._check_writable()
        super().__setitem__(key, value)
        if hasattr(self, "_parent"):  # Skip during initialization
            # Update timestamp (avoid infinite loop by not triggering for 'updated_at')
            if key != "updated_at":
                super().__setitem__(
                    "updated_at", datetime.now(timezone.utc).isoformat()
                )
            self._parent._save_metadata()

    def __delitem__(self, key: str) -> None:
        """Delete item and trigger save."""
        self._check_writable()
        super().__delitem__(key)
        self._touch_and_save()

    def update(self, *args, **kwargs) -> None:
        """Update dict and trigger save."""
        self._check_writable()
        super().update(*args, **kwargs)
        self._touch_and_save()

    def __ior__(self, other: Any) -> "MetadataDict":
        """Support ``metadata |= other`` with save and read-only semantics."""
        self._check_writable()
        super().update(other)
        self._touch_and_save()
        return self

    def clear(self) -> None:
        """Clear dict and trigger save."""
        self._check_writable()
        super().clear()
        self._touch_and_save()

    def pop(self, key: str, *default: Any) -> Any:
        """Pop a key and trigger save (respects read-only mode)."""
        if key in self:
            self._check_writable()
            value = super().pop(key)
            self._touch_and_save()
            return value
        return super().pop(key, *default)

    def popitem(self) -> tuple:
        """Pop an arbitrary item and trigger save (respects read-only mode)."""
        self._check_writable()
        result = super().popitem()
        self._touch_and_save()
        return result

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default and trigger save if key was added."""
        if key not in self:
            self._check_writable()
        had_key = key in self
        result = super().setdefault(key, default)
        if not had_key:
            self._touch_and_save()
        return result
