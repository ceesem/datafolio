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
        table_format: Format of the table ('parquet' or 'csv')
        **kwargs: Additional arguments passed to the format-specific reader

    Returns:
        pandas DataFrame

    Raises:
        ValueError: If format is not supported
        ImportError: If required dependencies are missing
        FileNotFoundError: If file/table doesn't exist

    Examples:
        >>> df = read_table('/path/data.parquet', 'parquet')
        >>> df = read_table('s3://bucket/data.csv', 'csv')
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

# Schemes polars can scan natively (a genuine lazy scan with predicate/
# projection pushdown — no full download up front). Includes the object stores
# and http(s) (polars range-requests the footer where the server supports it).
# A scheme NOT listed here is refused by the lazy API rather than being silently
# downloaded and wrapped as a fake LazyFrame.
_NATIVE_SCAN_PREFIXES = (
    "s3://",
    "gs://",
    "gcs://",
    "az://",
    "azure://",
    "http://",
    "https://",
)


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
    """Whether polars can perform a genuine lazy scan of this path.

    Local paths and the schemes in ``_NATIVE_SCAN_PREFIXES`` (object stores and
    http(s)) are genuinely scannable. Any other scheme is not, and the lazy API
    refuses it rather than downloading the whole object.

    Args:
        path: Resolved datafolio path.

    Returns:
        True if a native ``pl.scan_*`` should be used.
    """
    if path.startswith("file://") or not is_cloud_path(path):
        return True
    return path.startswith(_NATIVE_SCAN_PREFIXES)


def _lazy_scheme_error(path: str, fmt: str) -> ValueError:
    """Error raised when a lazy scan of ``path`` isn't genuinely possible."""
    scheme = path.split("://", 1)[0] if "://" in path else "<local>"
    return ValueError(
        f"Cannot perform a genuine lazy {fmt} scan of '{path}' (scheme "
        f"'{scheme}'). Lazy scanning avoids full downloads and is only "
        f"supported for local paths and {', '.join(_NATIVE_SCAN_PREFIXES)}. "
        f"Use get(name, frame='polars') for an eager read that downloads the object."
    )


def scan_parquet(
    path: str,
    cf: cloudfiles.CloudFiles = None,
    use_https: bool = True,
    storage_options: Optional[dict] = None,
    **kwargs: Any,
) -> "pl.LazyFrame":
    """Genuinely lazily scan a Parquet file as a polars LazyFrame.

    Uses polars' native ``scan_parquet`` (true lazy, predicate/projection
    pushdown, footer-only reads) for local paths and the object-store/http(s)
    schemes. This never downloads the whole object up front: for a scheme polars
    cannot scan lazily it raises rather than silently downloading and wrapping
    the eager result as a fake LazyFrame. (http laziness depends on the server
    supporting range requests.)

    Args:
        path: Path to the Parquet file (local or cloud).
        cf: Unused; kept for signature compatibility with the eager readers.
        use_https: Unused; kept for signature compatibility.
        storage_options: Optional credentials/config forwarded to native scan.
        **kwargs: Additional arguments forwarded to ``pl.scan_parquet``.

    Returns:
        polars LazyFrame.

    Raises:
        ValueError: If the path's scheme cannot be scanned lazily.
    """
    pl = _require_polars()
    if not _natively_scannable(path):
        raise _lazy_scheme_error(path, "parquet")
    return pl.scan_parquet(
        _polars_scan_path(path), storage_options=storage_options, **kwargs
    )


def scan_csv(
    path: str,
    cf: cloudfiles.CloudFiles = None,
    use_https: bool = True,
    storage_options: Optional[dict] = None,
    **kwargs: Any,
) -> "pl.LazyFrame":
    """Genuinely lazily scan a CSV file as a polars LazyFrame.

    Native ``pl.scan_csv`` for local paths and the object-store/http(s) schemes.
    Raises for schemes polars cannot scan lazily rather than downloading.

    Args:
        path: Path to the CSV file (local or cloud).
        cf: Unused; kept for signature compatibility with the eager readers.
        use_https: Unused; kept for signature compatibility.
        storage_options: Optional credentials/config forwarded to native scan.
        **kwargs: Additional arguments forwarded to ``pl.scan_csv``.

    Returns:
        polars LazyFrame.

    Raises:
        ValueError: If the path's scheme cannot be scanned lazily.
    """
    pl = _require_polars()
    if not _natively_scannable(path):
        raise _lazy_scheme_error(path, "csv")
    return pl.scan_csv(
        _polars_scan_path(path), storage_options=storage_options, **kwargs
    )


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
        cf: Unused; kept for signature compatibility (unsupported schemes
            raise rather than falling back to a full download).
        use_https: Unused; kept for signature compatibility.
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
