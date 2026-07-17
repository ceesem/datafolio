"""Readers for different table formats."""

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import cloudfiles
import joblib
import pandas as pd

from datafolio.utils import is_cloud_path

from .utils import resolve_path

if TYPE_CHECKING:
    import polars as pl


def _require_polars() -> Any:
    """Import polars or raise a friendly error.

    Returns:
        The imported ``polars`` module.

    Raises:
        ImportError: If polars is not installed, with an install hint.
    """
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "polars is required for lazy/polars table access. "
            "Install with: pip install 'datafolio[polars]'"
        ) from exc
    return pl


def _read_file(
    filename: str, cf: Optional[cloudfiles.CloudFiles], use_https: bool = True
) -> bytes:
    """Read file content as bytes, raising FileNotFoundError if it doesn't exist."""
    # Check if file exists for local paths
    if cf is None and not is_cloud_path(str(filename)):
        path_obj = Path(filename)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {filename}")

    if cf is None:
        fname = resolve_path(filename)
        cf = cloudfiles.CloudFile(fname)
        fbin = cf.get()
    else:
        fbin = cf.get(filename)

    # Check if content is empty (likely file doesn't exist)
    if not fbin:
        raise FileNotFoundError(f"File not found: {filename}")

    return fbin


def read_parquet(
    filename: str,
    cf: cloudfiles.CloudFiles = None,
    use_https: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a Parquet file from local or cloud storage.

    Args:
        path: Path to the Parquet file (local or cloud)
        **kwargs: Additional arguments passed to pd.read_parquet()

    Returns:
        pandas DataFrame

    Raises:
        ImportError: If cloud-files is needed but not installed
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> df = read_parquet('/local/path/data.parquet')
        >>> df = read_parquet('s3://bucket/data.parquet')
    """
    fbin = _read_file(filename, cf, use_https=use_https)
    return pd.read_parquet(io.BytesIO(fbin), **kwargs)


def read_arrow(
    filename: str,
    cf: cloudfiles.CloudFiles = None,
    use_https: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an Arrow IPC / Feather file from local or cloud storage.

    Args:
        filename: Path to the Arrow IPC file (local or cloud)
        **kwargs: Additional arguments passed to pyarrow.feather.read_feather()

    Returns:
        pandas DataFrame

    Raises:
        ImportError: If cloud-files is needed but not installed
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> df = read_arrow('/local/path/data.arrow')
        >>> df = read_arrow('s3://bucket/data.arrow')
    """
    import pyarrow.feather as feather

    fbin = _read_file(filename, cf, use_https=use_https)
    return feather.read_feather(io.BytesIO(fbin), **kwargs)


def read_csv(
    filename: str,
    cf: cloudfiles.CloudFiles = None,
    use_https: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file from local or cloud storage.

    Args:
        path: Path to the CSV file (local or cloud)
        **kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        pandas DataFrame

    Raises:
        ImportError: If cloud-files is needed but not installed
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> df = read_csv('/local/path/data.csv')
        >>> df = read_csv('s3://bucket/data.csv')
    """
    fbin = _read_file(filename, cf, use_https=use_https)
    return pd.read_csv(io.BytesIO(fbin), **kwargs)


def read_joblib(
    filename: str,
    cf: cloudfiles.CloudFiles = None,
    use_https: bool = True,
    **kwargs: Any,
) -> Any:
    """Read a joblib file from local or cloud storage.

    Args:
        filename: Path to the joblib file (local or cloud)
        cf: CloudFiles instance for cloud storage
        use_https: Whether to use HTTPS for cloud storage
        **kwargs: Additional arguments passed to joblib.load()

    Returns:
        Loaded object from joblib file

    Raises:
        ImportError: If cloud-files is needed but not installed
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> obj = read_joblib('/local/path/data.joblib')
        >>> obj = read_joblib('s3://bucket/data.joblib')
    """
    fbin = _read_file(filename, cf, use_https=use_https)
    return joblib.load(io.BytesIO(fbin), **kwargs)


def read_table(
    path: Union[str, Path],
    table_format: str,
    cf: cloudfiles.CloudFiles = None,
    use_https: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a table in any supported format.

    Dispatches to the appropriate reader based on format.

    Args:
        path: Path to the table (local or cloud)
        table_format: Format of the table ('parquet', 'delta', 'csv')
        **kwargs: Additional arguments passed to the format-specific reader

    Returns:
        pandas DataFrame

    Raises:
        ValueError: If format is not supported
        ImportError: If required dependencies are missing
        FileNotFoundError: If file/table doesn't exist

    Examples:
        >>> df = read_table('/path/data.parquet', 'parquet')
        >>> df = read_table('s3://bucket/delta', 'delta', version=3)
    """
    match table_format:
        case "parquet":
            return read_parquet(path, cf=cf, use_https=use_https, **kwargs)
        case "csv":
            return read_csv(path, cf=cf, use_https=use_https, **kwargs)
        case "arrow":
            return read_arrow(path, cf=cf, use_https=use_https, **kwargs)
        case "joblib":
            return read_joblib(path, cf=cf, use_https=use_https, **kwargs)
        case _:
            raise ValueError(
                f"Unsupported table format: {table_format}. Supported formats: parquet, arrow, csv"
            )


# =============================================================================
# Lazy / polars scanning
# =============================================================================

# Cloud schemes polars' object-store can scan natively (true lazy pushdown).
# Anything else (http(s)://, cloudfiles-specific schemes) uses the byte fallback.
_NATIVE_SCAN_PREFIXES = ("s3://", "gs://", "gcs://", "az://", "azure://")


def _polars_scan_path(path: str) -> str:
    """Normalize a datafolio path into one polars' scanners accept.

    Strips a leading ``file://`` (polars wants a bare local path) and normalizes
    ``gcs://`` to ``gs://`` (polars' preferred GCS scheme).

    Args:
        path: Resolved datafolio path.

    Returns:
        A path string suitable for ``pl.scan_*``.
    """
    if path.startswith("file://"):
        path = path[len("file://") :]
    if path.startswith("gcs://"):
        path = "gs://" + path[len("gcs://") :]
    return path


def _natively_scannable(path: str) -> bool:
    """Whether polars can scan this path natively (lazily), without downloading.

    Local paths and the object-store cloud schemes are natively scannable;
    ``http(s)://`` and any other scheme fall back to a full byte read.

    Args:
        path: Resolved datafolio path.

    Returns:
        True if a native ``pl.scan_*`` should be used.
    """
    if path.startswith("file://") or not is_cloud_path(path):
        return True
    return path.startswith(_NATIVE_SCAN_PREFIXES)


def scan_parquet(
    path: str,
    cf: cloudfiles.CloudFiles = None,
    use_https: bool = True,
    storage_options: Optional[dict] = None,
    **kwargs: Any,
) -> "pl.LazyFrame":
    """Lazily scan a Parquet file as a polars LazyFrame.

    Uses polars' native ``scan_parquet`` (true lazy, predicate/projection
    pushdown, footer-only reads) for local and object-store cloud paths. For
    schemes polars can't scan natively (e.g. ``http(s)://``), falls back to a
    cloudfiles byte read wrapped as a LazyFrame (whole file pulled, same API).

    Args:
        path: Path to the Parquet file (local or cloud).
        cf: Optional CloudFiles client for the byte-fallback path.
        use_https: Whether to use HTTPS for the byte fallback.
        storage_options: Optional credentials/config forwarded to native scan.
        **kwargs: Additional arguments forwarded to ``pl.scan_parquet``.

    Returns:
        polars LazyFrame.
    """
    pl = _require_polars()
    if _natively_scannable(path):
        return pl.scan_parquet(
            _polars_scan_path(path), storage_options=storage_options, **kwargs
        )
    fbin = _read_file(path, cf, use_https=use_https)
    return pl.read_parquet(io.BytesIO(fbin)).lazy()


def scan_csv(
    path: str,
    cf: cloudfiles.CloudFiles = None,
    use_https: bool = True,
    storage_options: Optional[dict] = None,
    **kwargs: Any,
) -> "pl.LazyFrame":
    """Lazily scan a CSV file as a polars LazyFrame.

    Native ``pl.scan_csv`` for local/object-store paths, byte fallback otherwise.

    Args:
        path: Path to the CSV file (local or cloud).
        cf: Optional CloudFiles client for the byte-fallback path.
        use_https: Whether to use HTTPS for the byte fallback.
        storage_options: Optional credentials/config forwarded to native scan.
        **kwargs: Additional arguments forwarded to ``pl.scan_csv``.

    Returns:
        polars LazyFrame.
    """
    pl = _require_polars()
    if _natively_scannable(path):
        return pl.scan_csv(
            _polars_scan_path(path), storage_options=storage_options, **kwargs
        )
    fbin = _read_file(path, cf, use_https=use_https)
    return pl.read_csv(io.BytesIO(fbin)).lazy()


def scan_table(
    path: Union[str, Path],
    table_format: str,
    cf: cloudfiles.CloudFiles = None,
    use_https: bool = True,
    **kwargs: Any,
) -> "pl.LazyFrame":
    """Lazily scan a table in any supported format as a polars LazyFrame.

    Dispatches to the appropriate scanner based on format. Mirrors
    :func:`read_table`, but returns a lazy ``pl.LazyFrame`` instead of an eager
    pandas DataFrame.

    Args:
        path: Path to the table (local or cloud).
        table_format: Format of the table ('parquet', 'csv').
        cf: Optional CloudFiles client for the byte-fallback path.
        use_https: Whether to use HTTPS for the byte fallback.
        **kwargs: Additional arguments forwarded to the format-specific scanner.

    Returns:
        polars LazyFrame.

    Raises:
        NotImplementedError: If the format has no lazy scanner.
    """
    match table_format:
        case "parquet":
            return scan_parquet(str(path), cf=cf, use_https=use_https, **kwargs)
        case "csv":
            return scan_csv(str(path), cf=cf, use_https=use_https, **kwargs)
        case _:
            raise NotImplementedError(
                f"Lazy scan is not supported for table format: {table_format}. "
                "Supported lazy formats: parquet, csv."
            )
