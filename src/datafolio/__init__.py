"""datafolio: Lightweight wrapping of dataframes, models, and metadata to track analyses."""

__version__ = "1.3.0"

from datafolio.folio import ConcurrentWriteError, DataFolio

__all__ = ["DataFolio", "ConcurrentWriteError", "__version__"]
