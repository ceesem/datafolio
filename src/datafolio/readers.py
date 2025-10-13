"""Readers for different table formats."""

import io
from pathlib import Path
from typing import Any, Optional, Union
from unittest import case

import cloudfiles
import joblib
import pandas as pd

from datafolio.utils import is_cloud_path

from .utils import resolve_path


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
    return pd.read_arrow(io.BytesIO(fbin), **kwargs)


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
