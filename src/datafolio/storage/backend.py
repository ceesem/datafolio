"""Storage backend for handling file I/O operations.

This module provides abstraction for file system operations, supporting both
local and cloud storage (via cloudfiles).
"""

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from datafolio.utils import is_cloud_path


class StorageBackend:
    """Handles all file I/O operations for DataFolio.

    Provides unified interface for local and cloud storage operations including:
    - File system operations (exists, mkdir, join_paths, copy, delete)
    - Format-specific I/O (JSON, Parquet, Joblib, PyTorch, Numpy)
    - Timestamp handling

    Examples:
        >>> storage = StorageBackend()
        >>> storage.write_json('/path/to/data.json', {'key': 'value'})
        >>> data = storage.read_json('/path/to/data.json')
    """

    def __init__(self, use_https: bool = False):
        """Initialize StorageBackend.

        Args:
            use_https: If True, use HTTPS URLs for CloudFiles read operations (default: False)
        """
        self._use_https = use_https

    def _split_cloud_path(self, path: str) -> tuple[str, str]:
        """Split cloud path into directory and filename.

        Args:
            path: Cloud path (e.g., 's3://bucket/dir/file.txt')

        Returns:
            Tuple of (dir_path, filename). If no directory, returns ('', filename).

        Examples:
            >>> storage._split_cloud_path('s3://bucket/dir/file.txt')
            ('s3://bucket/dir', 'file.txt')
            >>> storage._split_cloud_path('s3://bucket/file.txt')
            ('s3://bucket', 'file.txt')
        """
        parts = path.rsplit("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return "", parts[0]

    def _get_cloud_client(self, path: str, use_https: bool = False) -> Any:
        """Get CloudFiles client for a given path.

        Args:
            path: Cloud path (e.g., 's3://bucket/dir/file.txt')
            use_https: If True, use HTTPS URLs for read operations

        Returns:
            CloudFiles instance configured for the path

        Examples:
            >>> storage._get_cloud_client('s3://bucket/dir/file.txt')
            <CloudFiles instance>
        """
        from cloudfiles import CloudFiles

        dir_path, filename = self._split_cloud_path(path)
        if dir_path:
            return CloudFiles(dir_path, use_https=use_https), filename
        else:
            return CloudFiles(path, use_https=use_https), filename

    def _cloud_write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to cloud storage.

        Args:
            path: Cloud path
            data: Bytes to write

        Examples:
            >>> storage._cloud_write_bytes('s3://bucket/file.txt', b'data')
        """
        cf, filename = self._get_cloud_client(path)
        cf.put(filename, data)

    def _cloud_read_bytes(self, path: str) -> bytes:
        """Read bytes from cloud storage.

        Args:
            path: Cloud path

        Returns:
            File contents as bytes

        Examples:
            >>> data = storage._cloud_read_bytes('s3://bucket/file.txt')
        """
        cf, filename = self._get_cloud_client(path, use_https=self._use_https)
        return cf.get(filename)

    def exists(self, path: str) -> bool:
        """Check if a path exists (local or cloud).

        Args:
            path: Path to check

        Returns:
            True if path exists

        Examples:
            >>> storage = StorageBackend()
            >>> storage.exists('/path/to/file.json')
            True
        """
        if path.startswith("file://"):
            return Path(path[7:]).exists()

        if is_cloud_path(path):
            # Errors are deliberately NOT swallowed here: a transient auth or
            # network failure must never masquerade as "does not exist" —
            # DataFolio.__init__ uses this check to decide whether to create a
            # fresh bundle, and a false negative would overwrite a real one.
            from cloudfiles import CloudFiles

            # Exact-object check first (metadata-only HEAD/stat, never a
            # download). Works for any key, extension or not.
            dir_path, filename = self._split_cloud_path(path)
            if dir_path:
                cf = CloudFiles(dir_path, use_https=self._use_https)
                if bool(cf.exists(filename)):
                    return True
            # Fall back to a prefix check so directories (bundles, sharded
            # datasets — including ones with dots in the name) are found.
            # Stop at the first listed object instead of materializing the
            # full listing.
            cf = CloudFiles(path, use_https=self._use_https)
            return next(iter(cf.list()), None) is not None
        else:
            return Path(path).exists()

    def exists_many(self, paths: Iterable[str]) -> Dict[str, bool]:
        """Check existence of many paths at once (local or cloud).

        Cloud paths sharing a directory are checked in a single batched,
        threaded ``CloudFiles.exists()`` call instead of one round trip each,
        which is what makes whole-bundle sweeps (:meth:`DataFolio.validate`)
        cheap against object storage. Local paths are checked directly.

        Semantics match :meth:`exists` exactly, including the directory
        fallback: the batch answers the exact-object question, and any path
        it reports missing is re-checked individually so directory-style
        payloads (sharded datasets) are still found via a prefix listing.

        Args:
            paths: Paths to check. Duplicates are collapsed.

        Returns:
            Dict mapping every input path to its existence.

        Examples:
            >>> storage = StorageBackend()
            >>> storage.exists_many(['gs://b/d/a.parquet', 'gs://b/d/x.npy'])
            {'gs://b/d/a.parquet': True, 'gs://b/d/x.npy': False}
        """
        unique = list(dict.fromkeys(paths))
        results: Dict[str, bool] = {}
        by_dir: Dict[str, list] = {}

        for path in unique:
            dir_path = ""
            if is_cloud_path(path) and not path.startswith("file://"):
                dir_path, filename = self._split_cloud_path(path)
            if dir_path:
                by_dir.setdefault(dir_path, []).append((filename, path))
            else:
                # Local paths, and cloud paths with no directory component,
                # have nothing to batch.
                results[path] = self.exists(path)

        if by_dir:
            from cloudfiles import CloudFiles

            for dir_path, entries in by_dir.items():
                cf = CloudFiles(dir_path, use_https=self._use_https)
                found = cf.exists([filename for filename, _ in entries])
                for filename, path in entries:
                    # A miss falls back to the single-path check, which adds
                    # the prefix listing that finds directory payloads.
                    results[path] = True if found.get(filename) else self.exists(path)

        return results

    def mkdir(self, path: str, parents: bool = True, exist_ok: bool = True) -> None:
        """Create a directory (local or cloud).

        Args:
            path: Directory path to create
            parents: Create parent directories if needed
            exist_ok: Don't error if directory exists

        Examples:
            >>> storage = StorageBackend()
            >>> storage.mkdir('/path/to/dir')
        """
        if is_cloud_path(path):
            # Cloud storage is object-based, no need to create directories
            # They're created implicitly when you write files
            pass
        else:
            Path(path).mkdir(parents=parents, exist_ok=exist_ok)

    def _ensure_parent_dir(self, path: str) -> None:
        """Create parent directory for a file path (local only; cloud is a no-op).

        Args:
            path: File path whose parent directory should be created

        Examples:
            >>> storage._ensure_parent_dir('/path/to/subdir/file.parquet')
            # Creates /path/to/subdir/ if it doesn't exist
        """
        if not is_cloud_path(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def join_paths(self, *parts: str) -> str:
        """Join path components (local or cloud).

        Args:
            *parts: Path components to join

        Returns:
            Joined path string

        Examples:
            >>> storage = StorageBackend()
            >>> storage.join_paths('path', 'to', 'file.txt')
            'path/to/file.txt'
        """
        if any(is_cloud_path(str(p)) for p in parts):
            # Cloud path - use forward slashes
            return "/".join(str(p).rstrip("/") for p in parts)
        else:
            # Local path
            return str(Path(*parts))

    def delete_file(self, path: str) -> None:
        """Delete a file (local or cloud).

        Args:
            path: File path to delete

        Examples:
            >>> storage = StorageBackend()
            >>> storage.delete_file('/path/to/file.json')
        """
        if is_cloud_path(path):
            cf, filename = self._get_cloud_client(path)
            cf.delete(filename)
        else:
            path_obj = Path(path)
            if path_obj.exists():
                path_obj.unlink()

    def copy_file(self, src: Union[str, Path], dst: str) -> None:
        """Copy a file (local to local/cloud).

        Args:
            src: Source file path (local)
            dst: Destination path (local or cloud)

        Examples:
            >>> storage = StorageBackend()
            >>> storage.copy_file('/local/file.txt', 's3://bucket/file.txt')
        """
        if is_cloud_path(dst):
            # Stream the file object rather than buffering the whole file.
            self._upload_file(dst, str(src))
        else:
            import shutil

            self._ensure_parent_dir(dst)
            shutil.copy2(src, dst)

    def calculate_checksum(self, path: str) -> Optional[str]:
        """Calculate MD5 checksum of a file (local or cloud).

        Args:
            path: File path

        Returns:
            MD5 hex string, or None if path is a directory or calculation fails
        """
        import hashlib

        if is_cloud_path(path):
            # For cloud, getting checksum might be expensive (downloading)
            # or we might trust the ETag if available.
            # For now, let's skip checksum for cloud files to avoid massive downloads
            # unless we can get it from metadata.
            return None
        else:
            p = Path(path)
            if p.is_dir():
                return None
            if not p.exists():
                return None

            hash_md5 = hashlib.md5()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

    def file_size(self, path: str) -> Optional[int]:
        """Return the size in bytes of a file or directory (local or cloud).

        Best-effort: returns ``None`` when the size can't be determined cheaply
        (e.g. a cloud object whose backend doesn't expose size without a full
        read). For directories, sums the sizes of contained files.

        Args:
            path: File or directory path (local or cloud)

        Returns:
            Size in bytes, or ``None`` if it can't be determined.
        """
        if path.startswith("file://"):
            path = path[7:]

        if is_cloud_path(path):
            try:
                cf, filename = self._get_cloud_client(path, use_https=self._use_https)
                size = cf.size(filename)
                # cloudfiles may return an int, or None if unknown
                return int(size) if size is not None else None
            except Exception:
                return None
        else:
            p = Path(path)
            if p.is_dir():
                return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            if p.exists():
                return p.stat().st_size
            return None

    # =========================================================================
    # JSON I/O
    # =========================================================================

    def write_json(self, path: str, data: Any) -> None:
        """Write JSON data to file (local or cloud).

        Local writes are atomic: the content is written to a temp file in the
        same directory and then ``os.replace``-d over the target, so a reader
        never observes a half-written manifest and a crash mid-write can't
        corrupt it. (Cloud object puts are atomic per object already.)

        Args:
            path: File path
            data: Data to serialize

        Examples:
            >>> storage = StorageBackend()
            >>> storage.write_json('/path/to/data.json', {'key': 'value'})
        """
        import orjson

        content = orjson.dumps(
            data, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
        )

        if is_cloud_path(path):
            cf, filename = self._get_cloud_client(path)
            # Disable caching for manifest files to ensure fresh reads
            cf.put(filename, content, cache_control="no-cache")
        else:
            import os
            import tempfile

            self._ensure_parent_dir(path)
            directory = os.path.dirname(path) or "."
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)  # atomic on POSIX/Windows same-fs
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

    def read_json(self, path: str) -> Any:
        """Read JSON data from file (local or cloud).

        Args:
            path: File path

        Returns:
            Deserialized data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file content is empty or invalid

        Examples:
            >>> storage = StorageBackend()
            >>> data = storage.read_json('/path/to/data.json')
        """
        import orjson

        if is_cloud_path(path):
            content = self._cloud_read_bytes(path)

            # Handle case where file doesn't exist or is empty
            if content is None:
                raise FileNotFoundError(f"File not found in cloud storage: {path}")
            if not content:
                raise ValueError(f"Empty file in cloud storage: {path}")

            # CloudFiles might return string or bytes - handle both
            if isinstance(content, str):
                content = content.encode("utf-8")

            return orjson.loads(content)
        else:
            with open(path, "rb") as f:
                return orjson.loads(f.read())

    # =========================================================================
    # Parquet I/O
    # =========================================================================

    def write_parquet(self, path: str, df: Any) -> None:
        """Write a DataFrame or PyArrow Table to parquet (local or cloud).

        Polars DataFrames are written with Polars' own writer so that the
        resulting parquet statistics are readable by Polars' predicate-pushdown
        engine. Pandas DataFrames and PyArrow Tables are written via PyArrow.

        Args:
            path: File path
            df: Polars DataFrame, pandas DataFrame, or pyarrow.Table

        Examples:
            >>> storage = StorageBackend()
            >>> storage.write_parquet('/path/to/data.parquet', df)
        """
        if is_cloud_path(path):
            # Bounded memory: serialize to a temp local file, then stream-upload
            # the file object (no whole-file BytesIO). Temp file always cleaned.
            import os
            import tempfile

            fd, tmp = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            try:
                self._write_parquet_local(tmp, df)
                self._upload_file(path, tmp)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        else:
            self._ensure_parent_dir(path)
            self._write_parquet_local(path, df)

    def _write_parquet_local(self, path: str, df: Any) -> None:
        """Write a DataFrame / PyArrow Table to a local Parquet file.

        Polars DataFrames use Polars' own writer (so parquet statistics stay
        readable by Polars' predicate-pushdown engine); everything else goes
        through PyArrow.

        Args:
            path: Local file path
            df: Polars DataFrame, pandas DataFrame, or pyarrow.Table
        """
        try:
            import polars as pl

            if isinstance(df, pl.DataFrame):
                df.write_parquet(path)
                return
        except ImportError:
            pass

        import pyarrow as pa
        import pyarrow.parquet as pq

        table = (
            df
            if isinstance(df, pa.Table)
            else pa.Table.from_pandas(df, preserve_index=False)
        )
        pq.write_table(table, path)

    def _upload_file(self, dst: str, local_src: str) -> None:
        """Upload a local file to a cloud destination without buffering it.

        Passes an open file object to CloudFiles.put (which accepts a
        ``BinaryIO``), so the whole file is never read into a Python bytes
        buffer. This is the bounded-memory cloud upload path.

        Args:
            dst: Cloud destination path
            local_src: Local source file path
        """
        cf, filename = self._get_cloud_client(dst)
        with open(local_src, "rb") as f:
            cf.put(filename, f)

    def sink_parquet(self, path: str, lazyframe: Any) -> None:
        """Stream a Polars LazyFrame to Parquet with bounded memory.

        Uses Polars' streaming ``sink_parquet`` so the full result is never
        held in memory at once. For cloud destinations the stream is written to
        a temporary local file and then uploaded (see :meth:`_upload_file`),
        keeping memory bounded; the temp file is always cleaned up.

        Args:
            path: Destination path (local or cloud)
            lazyframe: A ``polars.LazyFrame`` to materialize
        """
        if is_cloud_path(path):
            import os
            import tempfile

            fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            try:
                lazyframe.sink_parquet(tmp_path)
                self._upload_file(path, tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            self._ensure_parent_dir(path)
            lazyframe.sink_parquet(path)

    def sink_parquet_with_footer(self, path: str, lazyframe: Any) -> tuple[Any, int]:
        """Stream a LazyFrame to Parquet and return ``(arrow_schema, num_rows)``.

        Like :meth:`sink_parquet`, but reads the schema/row-count footer from the
        **local staging file** (for cloud destinations) rather than re-reading
        the just-uploaded object — avoiding a full cloud round-trip download just
        to learn the schema.

        Args:
            path: Destination path (local or cloud).
            lazyframe: A ``polars.LazyFrame`` to materialize.

        Returns:
            Tuple of (pyarrow.Schema, number of rows).
        """
        if is_cloud_path(path):
            import os
            import tempfile

            fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            try:
                lazyframe.sink_parquet(tmp_path)
                footer = self.parquet_footer(tmp_path)  # local read, no download
                self._upload_file(path, tmp_path)
                return footer
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            self._ensure_parent_dir(path)
            lazyframe.sink_parquet(path)
            return self.parquet_footer(path)

    def source_identity(self, path: str) -> Dict[str, Any]:
        """Best-effort point-in-time identity for an external object.

        Captures whatever the backend can cheaply report — size, ETag,
        last-modified time — so a caller can tell whether a mutable external
        reference has changed since it was inspected. This is informational, not
        a hard pin: datafolio does not own or freeze the external content.

        Args:
            path: File path (local or cloud); directories report size only.

        Returns:
            JSON-serializable dict, possibly empty (best-effort).
        """
        import os

        ident: Dict[str, Any] = {}
        size = self.file_size(path)
        if size is not None:
            ident["size"] = size

        head_path = path
        if not is_cloud_path(head_path) and not head_path.startswith("file://"):
            head_path = "file://" + os.path.abspath(head_path)
        try:
            cf, filename = self._get_cloud_client(head_path, use_https=self._use_https)
            head = cf.head(filename)
            if head.get("ETag"):
                ident["etag"] = head["ETag"]
            last_modified = head.get("Last-Modified")
            if last_modified is not None:
                ident["last_modified"] = (
                    last_modified.isoformat()
                    if hasattr(last_modified, "isoformat")
                    else str(last_modified)
                )
        except Exception:
            # Directories and some backends won't support head(); size alone
            # is still useful. Best-effort by design.
            pass
        return ident

    def parquet_footer(self, path: str) -> tuple[Any, int]:
        """Return ``(arrow_schema, num_rows)`` from a Parquet file's footer.

        Reads only file metadata (schema + row count) where possible, which is
        cheap relative to reading the data. For local files this reads just the
        footer; for cloud files the object is fetched (cloudfiles reads whole
        objects) as a best-effort fallback.

        Args:
            path: Path to a Parquet file (local or cloud)

        Returns:
            Tuple of (pyarrow.Schema, number of rows).
        """
        import pyarrow.parquet as pq

        local = path[7:] if path.startswith("file://") else path
        if is_cloud_path(local):
            data = self._cloud_read_bytes(local)
            pf = pq.ParquetFile(io.BytesIO(data))
        else:
            pf = pq.ParquetFile(local)
        return pf.schema_arrow, pf.metadata.num_rows

    def read_parquet(self, path: str, **kwargs) -> Any:
        """Read parquet file to DataFrame (local or cloud).

        Args:
            path: File path
            **kwargs: Additional arguments passed to pd.read_parquet()

        Returns:
            pandas DataFrame

        Examples:
            >>> storage = StorageBackend()
            >>> df = storage.read_parquet('/path/to/data.parquet')
        """
        import pandas as pd

        if is_cloud_path(path) and not path.startswith("file://"):
            # Download via cloudfiles (the same credential chain and
            # use_https setting that wrote the file), then read locally.
            # Handing the cloud URI to pandas directly would route through
            # fsspec/s3fs/gcsfs — a different auth stack that may not be
            # installed or configured even though the bundle is reachable.
            import tempfile

            content = self._cloud_read_bytes(path)
            if content is None:
                raise FileNotFoundError(f"File not found in cloud storage: {path}")
            tmp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".parquet"
                ) as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)
                return pd.read_parquet(tmp_path, **kwargs)
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

        if path.startswith("file://"):
            path = path[7:]
        return pd.read_parquet(path, **kwargs)

    # =========================================================================
    # Joblib I/O
    # =========================================================================

    def write_joblib(self, path: str, obj: Any) -> None:
        """Write object with joblib (local or cloud).

        Args:
            path: File path
            obj: Object to serialize

        Examples:
            >>> storage = StorageBackend()
            >>> storage.write_joblib('/path/to/model.joblib', model)
        """
        import joblib

        if is_cloud_path(path):
            # Serialize to BytesIO
            buffer = io.BytesIO()
            joblib.dump(obj, buffer)
            data = buffer.getvalue()

            self._cloud_write_bytes(path, data)
        else:
            self._ensure_parent_dir(path)
            joblib.dump(obj, path)

    def read_joblib(self, path: str) -> Any:
        """Read object with joblib (local or cloud).

        Args:
            path: File path

        Returns:
            Deserialized object

        Examples:
            >>> storage = StorageBackend()
            >>> model = storage.read_joblib('/path/to/model.joblib')
        """
        import joblib

        if is_cloud_path(path):
            data = self._cloud_read_bytes(path)
            if data is None:
                raise FileNotFoundError(f"File not found in cloud storage: {path}")

            # Deserialize from BytesIO
            buffer = io.BytesIO(data)
            return joblib.load(buffer)
        else:
            return joblib.load(path)

    def write_skops(self, path: str, obj: Any) -> None:
        """Write object with skops (local or cloud).

        Args:
            path: File path
            obj: Object to serialize

        Examples:
            >>> storage = StorageBackend()
            >>> storage.write_skops('/path/to/model.skops', model)
        """
        import skops.io as sio

        # Serialize to bytes
        data = sio.dumps(obj)

        if is_cloud_path(path):
            self._cloud_write_bytes(path, data)
        else:
            self._ensure_parent_dir(path)
            Path(path).write_bytes(data)

    def read_skops(self, path: str, trusted: Any = False) -> Any:
        """Read object with skops (local or cloud).

        Args:
            path: File path
            trusted: False (default) refuses files containing non-standard
                types; True trusts every type found in the file; a list
                trusts exactly those type names

        Returns:
            Deserialized object

        Examples:
            >>> storage = StorageBackend()
            >>> model = storage.read_skops('/path/to/model.skops')
        """
        import skops.io as sio

        if is_cloud_path(path):
            data = self._cloud_read_bytes(path)
            if data is None:
                raise FileNotFoundError(f"File not found in cloud storage: {path}")
        else:
            data = Path(path).read_bytes()

        # skops refuses non-standard types by design (CVE-2024-37065).
        # Loading them requires the caller to opt in: trusted=True trusts
        # everything found in this file; a list trusts exactly those types.
        unknown_types = sio.get_untrusted_types(data=data)
        if not unknown_types:
            return sio.loads(data, trusted=[])
        if trusted is True:
            return sio.loads(data, trusted=unknown_types)
        if isinstance(trusted, (list, tuple, set)):
            return sio.loads(data, trusted=list(trusted))
        raise ValueError(
            f"skops file at {path} contains non-standard types that are not "
            f"trusted by default: {unknown_types}. If you trust the folio's "
            f"author, load with trusted=True (e.g. "
            f"folio.get_model(name, trusted=True))."
        )

    # =========================================================================
    # Numpy I/O
    # =========================================================================

    def write_numpy(self, path: str, array: Any) -> None:
        """Write numpy array to file (local or cloud).

        Args:
            path: File path
            array: numpy array to save

        Examples:
            >>> storage = StorageBackend()
            >>> storage.write_numpy('/path/to/array.npy', arr)
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "NumPy is required to save numpy arrays. "
                "Install with: pip install numpy"
            )

        if is_cloud_path(path):
            # Write to temp file, then upload
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                tmp_path = Path(tmp.name)
            try:
                np.save(tmp_path, array)
                content = tmp_path.read_bytes()
                self._cloud_write_bytes(path, content)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            self._ensure_parent_dir(path)
            np.save(path, array)

    def read_numpy(self, path: str, **kwargs) -> Any:
        """Read numpy array from file (local or cloud).

        Args:
            path: File path
            **kwargs: Additional arguments passed to np.load()

        Returns:
            numpy array

        Examples:
            >>> storage = StorageBackend()
            >>> arr = storage.read_numpy('/path/to/array.npy')
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "NumPy is required to load numpy arrays. "
                "Install with: pip install numpy"
            )

        if is_cloud_path(path):
            # Download to temp file, then load
            import tempfile

            content = self._cloud_read_bytes(path)
            if content is None:
                raise FileNotFoundError(f"File not found in cloud storage: {path}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                tmp_path = Path(tmp.name)
            try:
                tmp_path.write_bytes(content)
                return np.load(tmp_path, **kwargs)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            return np.load(path, **kwargs)

    # =========================================================================
    # Timestamp I/O
    # =========================================================================

    def write_timestamp(self, path: str, timestamp: datetime) -> None:
        """Write timestamp to JSON file (local or cloud).

        Args:
            path: File path
            timestamp: datetime object (must be timezone-aware, will be converted to UTC)

        Examples:
            >>> storage = StorageBackend()
            >>> storage.write_timestamp('/path/to/time.json', datetime.now(timezone.utc))
        """
        # Convert to UTC and create ISO 8601 string
        utc_timestamp = timestamp.astimezone(timezone.utc)
        iso_string = utc_timestamp.isoformat()

        # Write as JSON using existing method
        self.write_json(path, {"iso_string": iso_string})

    def read_timestamp(self, path: str) -> datetime:
        """Read timestamp from JSON file (local or cloud).

        Args:
            path: File path

        Returns:
            UTC-aware datetime object

        Examples:
            >>> storage = StorageBackend()
            >>> ts = storage.read_timestamp('/path/to/time.json')
        """
        # Read JSON using existing method
        data = self.read_json(path)
        iso_string = data["iso_string"]

        # Parse ISO 8601 string to datetime
        return datetime.fromisoformat(iso_string)

    # =========================================================================
    # High-level table reading (delegates to readers.py)
    # =========================================================================

    def read_table(self, path: str, table_format: str, **kwargs) -> Any:
        """Read a table in any supported format.

        This is a convenience method that delegates to readers.py.

        Args:
            path: Path to the table
            table_format: Format of the table ('parquet', 'csv', 'arrow')
            **kwargs: Additional arguments passed to the format-specific reader

        Returns:
            pandas DataFrame

        Examples:
            >>> storage = StorageBackend()
            >>> df = storage.read_table('/path/to/data.parquet', 'parquet')
        """
        from datafolio.readers import read_table

        return read_table(path, table_format, **kwargs)

    def scan_table(self, path: str, table_format: str, **kwargs) -> Any:
        """Lazily scan a table as a polars LazyFrame (local or cloud).

        Convenience method that delegates to readers.scan_table. Uses polars'
        native scanners (with predicate/projection pushdown) where possible and
        a byte fallback otherwise.

        Args:
            path: Path to the table
            table_format: Format of the table ('parquet', 'csv')
            **kwargs: Additional arguments passed to the format-specific scanner

        Returns:
            polars LazyFrame

        Raises:
            ImportError: If polars is not installed
            NotImplementedError: If the format has no lazy scanner

        Examples:
            >>> storage = StorageBackend()
            >>> lf = storage.scan_table('s3://bucket/data.parquet', 'parquet')
        """
        from datafolio.readers import scan_table

        return scan_table(path, table_format, use_https=self._use_https, **kwargs)
