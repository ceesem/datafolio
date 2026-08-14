"""datafolio: Lightweight wrapping of dataframes, models, and metadata to track analyses."""

__version__ = "2.0.0"

from datafolio.folio import (
    ConcurrentWriteError,
    DataFolio,
    ManifestReadError,
    UnsupportedManifestVersionError,
)

__all__ = [
    "DataFolio",
    "ConcurrentWriteError",
    "ManifestReadError",
    "UnsupportedManifestVersionError",
    "__version__",
]
