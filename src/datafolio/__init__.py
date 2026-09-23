"""datafolio: Lightweight wrapping of dataframes, models, and metadata to track analyses."""

__version__ = "2.2.0"

from datafolio.folio import (
    ConcurrentWriteError,
    DataFolio,
    ManifestReadError,
    UnsupportedManifestVersionError,
)
from datafolio.folio_registry import list_folios, remove_alias, set_alias
from datafolio.search import find

__all__ = [
    "DataFolio",
    "ConcurrentWriteError",
    "ManifestReadError",
    "UnsupportedManifestVersionError",
    "find",
    "list_folios",
    "remove_alias",
    "set_alias",
    "__version__",
]
