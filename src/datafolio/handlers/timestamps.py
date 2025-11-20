"""Handler for timestamp data."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from datafolio.base.handler import BaseHandler

if TYPE_CHECKING:
    from datafolio.folio import DataFolio


class TimestampHandler(BaseHandler):
    """Handler for timestamp data stored in bundle.

    This handler manages timestamps:
    - Converts datetime/Unix timestamps to UTC ISO format
    - Stores metadata (ISO string, Unix timestamp)
    - Handles lineage tracking
    - Deserializes back to datetime objects on read

    Examples:
        >>> from datafolio.base.registry import register_handler
        >>> handler = TimestampHandler()
        >>> register_handler(handler)
        >>>
        >>> # Handler is used automatically by DataFolio
        >>> from datetime import datetime, timezone
        >>> event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        >>> folio.add_timestamp('event_time', event_time)
    """

    @property
    def item_type(self) -> str:
        """Return item type identifier."""
        return "timestamp"

    def can_handle(self, data: Any) -> bool:
        """Check if data is a datetime object.

        Only auto-detects datetime objects for add_data().
        Unix timestamps (int/float) can still be stored via explicit add_timestamp() calls.

        Args:
            data: Data to check

        Returns:
            True if data is a datetime object
        """
        return isinstance(data, datetime)

    def add(
        self,
        folio: "DataFolio",
        name: str,
        data: Union[datetime, int, float],
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add timestamp to folio.

        Writes the timestamp to storage and builds complete metadata including:
        - ISO string representation (UTC)
        - Unix timestamp (float)
        - Lineage: inputs and description
        - Timestamps: creation time

        Args:
            folio: DataFolio instance
            name: Item name
            data: Timezone-aware datetime or Unix timestamp (int/float)
            description: Optional description
            inputs: Optional lineage inputs
            **kwargs: Additional arguments (currently unused)

        Returns:
            Complete metadata dict for this timestamp

        Raises:
            TypeError: If data is not datetime or numeric
            ValueError: If datetime is naive (no timezone)
        """
        # Convert Unix timestamp to datetime if needed
        if isinstance(data, (int, float)):
            # Unix timestamp - convert to UTC datetime
            dt = datetime.fromtimestamp(data, tz=timezone.utc)
        elif isinstance(data, datetime):
            # Validate timezone-awareness
            if data.tzinfo is None:
                raise ValueError(
                    "Naive datetime objects are not allowed. "
                    "Please use timezone-aware datetime objects. "
                    "Example: datetime.now(timezone.utc) or dt.replace(tzinfo=timezone.utc)"
                )
            dt = data
        else:
            raise TypeError(
                f"Expected datetime or Unix timestamp (int/float), got {type(data).__name__}"
            )

        # Convert to UTC
        utc_dt = dt.astimezone(timezone.utc)
        iso_string = utc_dt.isoformat()
        unix_ts = utc_dt.timestamp()

        # Build filename
        filename = f"{name}.json"
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(folio._bundle_dir, subdir, filename)

        # Write data to storage
        folio._storage.write_timestamp(filepath, utc_dt)

        # Build comprehensive metadata
        metadata = {
            "name": name,
            "item_type": self.item_type,
            "filename": filename,
            "iso_string": iso_string,
            "unix_timestamp": unix_ts,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields
        if description:
            metadata["description"] = description
        if inputs:
            metadata["inputs"] = inputs

        return metadata

    def get(
        self, folio: "DataFolio", name: str, as_unix: bool = False, **kwargs
    ) -> Union[datetime, float]:
        """Load timestamp from folio.

        Args:
            folio: DataFolio instance
            name: Item name
            as_unix: If True, return Unix timestamp; if False, return datetime
            **kwargs: Additional arguments (currently unused)

        Returns:
            UTC-aware datetime object or Unix timestamp

        Raises:
            KeyError: If item doesn't exist
        """
        item = folio._items[name]
        subdir = self.get_storage_subdir()
        filepath = folio._storage.join_paths(
            folio._bundle_dir, subdir, item["filename"]
        )
        dt = folio._storage.read_timestamp(filepath)

        return dt.timestamp() if as_unix else dt
