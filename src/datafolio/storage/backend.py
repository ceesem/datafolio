"""Storage backend for handling file I/O operations.

This module provides abstraction for file system operations, supporting both
local and cloud storage (via cloudfiles).
"""

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
        if is_cloud_path(path):
            # For cloud, we'll try to list contents
            try:
                from cloudfiles import CloudFiles

                cf = CloudFiles(path)
                # Try to list - if it works, directory exists
                list(cf.list())
                return True
            except:
                return False
        else:
            return Path(path).exists()

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
            from cloudfiles import CloudFiles

            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
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
            from cloudfiles import CloudFiles

            with open(src, "rb") as f:
                content = f.read()
            parts = dst.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(dst)
            cf.put(filename, content)
        else:
            import shutil

            shutil.copy2(src, dst)

    # =========================================================================
    # JSON I/O
    # =========================================================================

    def write_json(self, path: str, data: Any) -> None:
        """Write JSON data to file (local or cloud).

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
            from cloudfiles import CloudFiles

            # Extract directory and filename
            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
            cf.put(filename, content)
        else:
            with open(path, "wb") as f:
                f.write(content)

    def read_json(self, path: str) -> Any:
        """Read JSON data from file (local or cloud).

        Args:
            path: File path

        Returns:
            Deserialized data

        Examples:
            >>> storage = StorageBackend()
            >>> data = storage.read_json('/path/to/data.json')
        """
        import orjson

        if is_cloud_path(path):
            from cloudfiles import CloudFiles

            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
            content = cf.get(filename)
            return orjson.loads(content)
        else:
            with open(path, "rb") as f:
                return orjson.loads(f.read())

    # =========================================================================
    # Parquet I/O
    # =========================================================================

    def write_parquet(self, path: str, df: Any) -> None:
        """Write DataFrame to parquet (local or cloud).

        Args:
            path: File path
            df: pandas DataFrame

        Examples:
            >>> storage = StorageBackend()
            >>> storage.write_parquet('/path/to/data.parquet', df)
        """
        # pandas.to_parquet handles cloud paths if fsspec/cloud libs installed
        df.to_parquet(path, index=False)

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
            # Write to temp file, then upload
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                joblib.dump(obj, tmp.name)
                with open(tmp.name, "rb") as f:
                    content = f.read()
                # Upload
                from cloudfiles import CloudFiles

                parts = path.rsplit("/", 1)
                if len(parts) == 2:
                    dir_path, filename = parts
                else:
                    dir_path = ""
                    filename = parts[0]
                cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
                cf.put(filename, content)
                # Cleanup
                Path(tmp.name).unlink()
        else:
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
            # Download to temp file, then load
            import tempfile

            from cloudfiles import CloudFiles

            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
            content = cf.get(filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                tmp.write(content)
                tmp.flush()
                obj = joblib.load(tmp.name)
                Path(tmp.name).unlink()
                return obj
        else:
            return joblib.load(path)

    # =========================================================================
    # PyTorch I/O
    # =========================================================================

    def write_pytorch(
        self,
        path: str,
        model: Any,
        init_args: Optional[Dict[str, Any]] = None,
        save_class: bool = False,
        optimizer_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write PyTorch model state dict with metadata (local or cloud).

        Saves a dictionary containing:
        - state_dict: Model weights
        - metadata: Class name, module, and optional init args
        - serialized_class: Optional dill-serialized class (if save_class=True)
        - optimizer_state: Optional optimizer state dict

        Args:
            path: File path
            model: PyTorch model (will save state_dict)
            init_args: Optional dict of args needed to instantiate the model
            save_class: If True, use dill to serialize the model class
            optimizer_state: Optional optimizer state dict

        Examples:
            >>> storage = StorageBackend()
            >>> storage.write_pytorch('/path/to/model.pt', model, save_class=True)
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required to save PyTorch models. "
                "Install with: pip install torch"
            )

        # Prepare save bundle
        save_bundle = {
            "state_dict": model.state_dict(),
            "metadata": {
                "model_class": model.__class__.__name__,
                "model_module": model.__class__.__module__,
            },
        }

        if init_args is not None:
            save_bundle["metadata"]["init_args"] = init_args

        if optimizer_state is not None:
            save_bundle["optimizer_state"] = optimizer_state

        # Optionally serialize class with dill
        if save_class:
            try:
                import dill

                save_bundle["serialized_class"] = dill.dumps(model.__class__)
            except ImportError:
                raise ImportError(
                    "dill is required when save_class=True. "
                    "Install with: pip install dill"
                )

        if is_cloud_path(path):
            # Write to temp file, then upload
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                torch.save(save_bundle, tmp.name)
                with open(tmp.name, "rb") as f:
                    content = f.read()
                # Upload
                from cloudfiles import CloudFiles

                parts = path.rsplit("/", 1)
                if len(parts) == 2:
                    dir_path, filename = parts
                else:
                    dir_path = ""
                    filename = parts[0]
                cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
                cf.put(filename, content)
                # Cleanup
                Path(tmp.name).unlink()
        else:
            torch.save(save_bundle, path)

    def read_pytorch(self, path: str, **kwargs) -> Dict[str, Any]:
        """Read PyTorch model bundle (local or cloud).

        Args:
            path: File path
            **kwargs: Additional arguments passed to torch.load()

        Returns:
            Dictionary containing state_dict, metadata, and optionally serialized_class

        Examples:
            >>> storage = StorageBackend()
            >>> bundle = storage.read_pytorch('/path/to/model.pt')
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required to load PyTorch models. "
                "Install with: pip install torch"
            )

        if is_cloud_path(path):
            # Download to temp file, then load
            import tempfile

            from cloudfiles import CloudFiles

            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
            content = cf.get(filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                tmp.write(content)
                tmp.flush()
                bundle = torch.load(tmp.name, weights_only=False, **kwargs)
                Path(tmp.name).unlink()
                return bundle
        else:
            return torch.load(path, weights_only=False, **kwargs)

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
                np.save(tmp.name, array)
                with open(tmp.name, "rb") as f:
                    content = f.read()
                # Upload
                from cloudfiles import CloudFiles

                parts = path.rsplit("/", 1)
                if len(parts) == 2:
                    dir_path, filename = parts
                else:
                    dir_path = ""
                    filename = parts[0]
                cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
                cf.put(filename, content)
                # Cleanup
                Path(tmp.name).unlink()
        else:
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

            from cloudfiles import CloudFiles

            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_path, filename = parts
            else:
                dir_path = ""
                filename = parts[0]
            cf = CloudFiles(dir_path) if dir_path else CloudFiles(path)
            content = cf.get(filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                tmp.write(content)
                tmp.flush()
                array = np.load(tmp.name, **kwargs)
                Path(tmp.name).unlink()
                return array
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
