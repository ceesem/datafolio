"""Handler for JSON-serializable data."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from datafolio.base.handler import BaseHandler

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


def _json_key(key: Any) -> str:
    """Return the string orjson would write for a non-str dict key."""
    import orjson

    try:
        encoded = orjson.dumps(
            {key: None},
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )
    except (TypeError, orjson.JSONEncodeError) as exc:
        raise TypeError(
            f"dict key {key!r} ({type(key).__name__}) cannot be stored as JSON"
        ) from exc
    return next(iter(orjson.loads(encoded)))


def to_json_compatible(obj: Any) -> tuple[Any, list[str]]:
    """Normalize container types that JSON has no direct equivalent for.

    Recursively converts tuples, ranges, sets and frozensets to lists and
    non-str dict keys to strings, so the result can be written by orjson.
    Other values are passed through untouched (unsupported leaves still fail
    at serialization time with the usual error).

    Tuples and ranges convert silently — orjson already writes nested tuples
    as arrays. Sets and non-str keys change how the data reads back, so they
    are reported in ``notes`` for the caller to warn about.

    Args:
        obj: Object to normalize.

    Returns:
        ``(converted, notes)`` where ``notes`` is a sorted list of the lossy
        conversions performed (``"set"``, ``"non-str dict keys"``).

    Raises:
        ValueError: If two dict keys map to the same string
            (e.g. ``{1: 'a', '1': 'b'}``), which would silently drop data.
        TypeError: If a dict key has no JSON string form.

    Examples:
        >>> to_json_compatible({'a': (1, 2), 'b': {3, 1}})
        ({'a': [1, 2], 'b': [1, 3]}, ['set'])
        >>> to_json_compatible({1: 'x'})
        ({'1': 'x'}, ['non-str dict keys'])
    """
    notes: set[str] = set()

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    notes.add("non-str dict keys")
                    key = _json_key(key)
                if key in out:
                    raise ValueError(
                        f"dict keys collide as JSON key {key!r}; rename one "
                        f"of them before storing."
                    )
                out[key] = convert(item)
            return out
        if isinstance(value, (set, frozenset)):
            notes.add("set")
            items = [convert(item) for item in value]
            try:
                return sorted(items)
            except TypeError:
                return items
        if isinstance(value, (list, tuple, range)):
            return [convert(item) for item in value]
        return value

    return convert(obj), sorted(notes)


class JsonHandler(BaseHandler):
    """Handler for JSON-serializable data stored in bundle.

    This handler manages JSON data (dicts, lists, scalars):
    - Serializes data to .json format
    - Stores metadata (data type)
    - Handles lineage tracking
    - Deserializes back to Python objects on read

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = JsonHandler()
        >>> register_handler(handler)
        >>>
        >>> # Handler is used automatically by DataFolio
        >>> config = {'learning_rate': 0.01, 'batch_size': 32}
        >>> folio.add('config', config)
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "json_data"

    def can_handle(self, data: Any) -> bool:
        """Check if data is JSON-serializable.

        Only handles dict, list, and basic JSON types (int, float, str, bool, None).
        Does NOT handle complex objects, DataFrames, arrays, etc.

        Args:
            data: Data to check

        Returns:
            True if data is a basic JSON-serializable type
        """
        # Only handle dict and list explicitly
        # Exclude primitives (int, float, str, bool, None) to avoid conflicts with other handlers
        # DataFolio.add() routes primitives to this handler directly as JSON
        if isinstance(data, (dict, list)):
            try:
                import orjson

                orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
                return True
            except (TypeError, ValueError, ImportError):
                return False
        return False

    def add(
        self,
        folio: "DataFolio",
        name: str,
        data: Union[dict, list, int, float, str, bool, None],
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add JSON data to folio.

        Writes the data to storage and builds complete metadata including:
        - Basic info: filename, data type
        - Lineage: inputs and description
        - Timestamps: creation time

        Args:
            folio: DataFolio instance
            name: Item name
            data: JSON-serializable data to store
            description: Optional description
            inputs: Optional lineage inputs
            **kwargs: Additional arguments (currently unused)

        Returns:
            Complete metadata dict for this JSON data

        Raises:
            TypeError: If data is not JSON-serializable
        """
        import orjson

        # Validate JSON-serializability
        try:
            orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Data is not JSON-serializable: {e}")

        # Build filename (folio injects a collision-safe versioned name)
        filename = kwargs.get("_filename") or f"{name}.json"
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        # Write data to storage
        folio._storage.write_json(filepath, data)

        # Calculate checksum
        checksum = folio._storage.calculate_checksum(filepath)

        # Build comprehensive metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "checksum": checksum,
            "data_type": kwargs.get("_data_type") or type(data).__name__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = list(inputs)

        return metadata

    def get(self, folio: "DataFolio", name: str, **kwargs) -> Any:
        """Load JSON data from folio.

        Args:
            folio: DataFolio instance
            name: Item name
            **kwargs: Additional arguments (currently unused)

        Returns:
            Deserialized JSON data (dict, list, scalar, etc.)

        Raises:
            KeyError: If item doesn't exist
        """
        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )

        return folio._storage.read_json(filepath)
