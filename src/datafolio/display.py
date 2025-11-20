"""Display formatting for DataFolio describe() output.

This module provides formatting utilities for generating human-readable
descriptions of DataFolio bundles, including item listings, metadata,
lineage information, and file sizes.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from datafolio.folio import DataFolio

from datafolio.utils import ARTIFACTS_DIR, MODELS_DIR, TABLES_DIR, is_cloud_path


class DisplayFormatter:
    """Formatter for generating human-readable DataFolio descriptions.

    This class handles all display formatting for the describe() method,
    including timestamp formatting, file size formatting, and metadata
    value truncation.
    """

    def __init__(self, folio: "DataFolio"):
        """Initialize DisplayFormatter.

        Args:
            folio: DataFolio instance to format
        """
        self._folio = folio

    def describe(
        self,
        return_string: bool = False,
        show_empty: bool = False,
        max_metadata_fields: int = 10,
    ) -> Optional[str]:
        """Generate a human-readable description of all items in the bundle.

        Includes lineage information showing inputs and dependencies.

        Args:
            return_string: If True, return the description as a string instead of printing
            show_empty: If True, show empty sections (e.g., "Models (0): (none)")
            max_metadata_fields: Maximum number of metadata fields to display (default: 10)

        Returns:
            None if return_string=False (prints to stdout), otherwise returns the description string

        Examples:
            Print description (default):
            >>> folio = DataFolio('experiments/test')
            >>> folio.describe()
            DataFolio: experiments/test
            ===========================
            Tables (2):
              • raw_data (reference): Training data
              • results: Model results
                ↳ inputs: raw_data

            Get description as string:
            >>> folio = DataFolio('experiments/test')
            >>> text = folio.describe(return_string=True)

            Show empty sections:
            >>> folio.describe(show_empty=True)
            Tables (2):
              • raw_data: Training data
              • results: Model results
            Models (0):
              (none)

            Limit metadata fields shown:
            >>> folio.describe(max_metadata_fields=5)
        """
        # Auto-refresh if bundle was updated externally
        self._folio._refresh_if_needed()

        lines = []
        lines.append(f"DataFolio: {self._folio._bundle_dir}")
        lines.append("=" * len(lines[0]))
        lines.append("")

        # Add timestamps if available (with human-readable formatting)
        if "created_at" in self._folio.metadata:
            formatted_created = self._format_timestamp(
                self._folio.metadata["created_at"]
            )
            lines.append(f"Created: {formatted_created}")
        if "updated_at" in self._folio.metadata:
            formatted_updated = self._format_timestamp(
                self._folio.metadata["updated_at"]
            )
            lines.append(f"Updated: {formatted_updated}")
        if "parent_bundle" in self._folio.metadata:
            lines.append(f"Parent: {self._folio.metadata['parent_bundle']}")
        if any(
            k in self._folio.metadata
            for k in ["created_at", "updated_at", "parent_bundle"]
        ):
            lines.append("")

        # Add custom metadata section (filter out internal fields)
        internal_fields = {"created_at", "updated_at", "parent_bundle", "_datafolio"}
        custom_metadata = {
            k: v for k, v in self._folio.metadata.items() if k not in internal_fields
        }

        if custom_metadata:
            lines.append(f"Metadata ({len(custom_metadata)} fields):")
            # Sort keys for consistent display
            sorted_keys = sorted(custom_metadata.keys())
            displayed_keys = sorted_keys[:max_metadata_fields]

            for key in displayed_keys:
                value = custom_metadata[key]
                formatted_value = self._format_metadata_value(value)
                lines.append(f"  • {key}: {formatted_value}")

            # Show count of remaining fields if any
            remaining_count = len(custom_metadata) - len(displayed_keys)
            if remaining_count > 0:
                lines.append(f"  ... ({remaining_count} more fields)")

            lines.append("")

        contents = self._folio.list_contents()

        # Combine referenced and included tables
        ref_tables = contents["referenced_tables"]
        inc_tables = contents["included_tables"]
        all_tables = ref_tables + inc_tables

        if all_tables or show_empty:
            lines.append(f"Tables ({len(all_tables)}):")
            if all_tables:
                # Show referenced tables first
                for name in ref_tables:
                    item = self._folio._items[name]
                    desc = item.get("description", "(no description)")
                    lines.append(f"  • {name} (reference): {desc}")
                    # Show path for referenced tables
                    if "path" in item:
                        lines.append(f"    ↳ path: {item['path']}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")

                # Then included tables
                for name in inc_tables:
                    item = self._folio._items[name]
                    desc = item.get("description", "(no description)")
                    lines.append(f"  • {name}: {desc}")
                    # Show file size if available
                    filesize = self._get_item_filesize(item)
                    if filesize is not None:
                        lines.append(f"    ↳ size: {self._format_filesize(filesize)}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
                    if "models" in item and item["models"]:
                        lines.append(f"    ↳ models: {', '.join(item['models'])}")
            else:
                lines.append("  (none)")
            lines.append("")

        # Numpy arrays
        numpy_arrays = contents["numpy_arrays"]
        if numpy_arrays or show_empty:
            lines.append(f"Numpy Arrays ({len(numpy_arrays)}):")
            if numpy_arrays:
                for name in numpy_arrays:
                    item = self._folio._items[name]
                    desc = item.get("description", "(no description)")
                    shape = item.get("shape", "unknown")
                    dtype = item.get("dtype", "unknown")
                    lines.append(f"  • {name}: {desc}")
                    lines.append(f"    ↳ shape: {shape}, dtype: {dtype}")
                    # Show file size if available
                    filesize = self._get_item_filesize(item)
                    if filesize is not None:
                        lines.append(f"    ↳ size: {self._format_filesize(filesize)}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
            else:
                lines.append("  (none)")
            lines.append("")

        # JSON data
        json_data = contents["json_data"]
        if json_data or show_empty:
            lines.append(f"JSON Data ({len(json_data)}):")
            if json_data:
                for name in json_data:
                    item = self._folio._items[name]
                    desc = item.get("description", "(no description)")
                    data_type = item.get("data_type", "unknown")
                    lines.append(f"  • {name}: {desc}")
                    lines.append(f"    ↳ type: {data_type}")
                    # Show file size if available
                    filesize = self._get_item_filesize(item)
                    if filesize is not None:
                        lines.append(f"    ↳ size: {self._format_filesize(filesize)}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
            else:
                lines.append("  (none)")
            lines.append("")

        # Models
        models = contents["models"]
        if models or show_empty:
            lines.append(f"Models ({len(models)}):")
            if models:
                for name in models:
                    item = self._folio._items[name]
                    desc = item.get("description", "(no description)")
                    lines.append(f"  • {name}: {desc}")
                    # Show file size if available
                    filesize = self._get_item_filesize(item)
                    if filesize is not None:
                        lines.append(f"    ↳ size: {self._format_filesize(filesize)}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
                    if "hyperparameters" in item and item["hyperparameters"]:
                        # Show a few key hyperparameters
                        hps = item["hyperparameters"]
                        hp_str = ", ".join(f"{k}={v}" for k, v in list(hps.items())[:3])
                        if len(hps) > 3:
                            hp_str += f", ... ({len(hps) - 3} more)"
                        lines.append(f"    ↳ hyperparameters: {hp_str}")
            else:
                lines.append("  (none)")
            lines.append("")

        # PyTorch Models
        pytorch_models = contents["pytorch_models"]
        if pytorch_models or show_empty:
            lines.append(f"PyTorch Models ({len(pytorch_models)}):")
            if pytorch_models:
                for name in pytorch_models:
                    item = self._folio._items[name]
                    desc = item.get("description", "(no description)")
                    lines.append(f"  • {name}: {desc}")
                    # Show file size if available
                    filesize = self._get_item_filesize(item)
                    if filesize is not None:
                        lines.append(f"    ↳ size: {self._format_filesize(filesize)}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
                    if "hyperparameters" in item and item["hyperparameters"]:
                        # Show a few key hyperparameters
                        hps = item["hyperparameters"]
                        hp_str = ", ".join(f"{k}={v}" for k, v in list(hps.items())[:3])
                        if len(hps) > 3:
                            hp_str += f", ... ({len(hps) - 3} more)"
                        lines.append(f"    ↳ hyperparameters: {hp_str}")
                    if "init_args" in item and item["init_args"]:
                        # Show init args
                        init_args = item["init_args"]
                        init_str = ", ".join(
                            f"{k}={v}" for k, v in list(init_args.items())[:3]
                        )
                        if len(init_args) > 3:
                            init_str += f", ... ({len(init_args) - 3} more)"
                        lines.append(f"    ↳ init_args: {init_str}")
            else:
                lines.append("  (none)")
            lines.append("")

        # Artifacts
        artifacts = contents["artifacts"]
        if artifacts or show_empty:
            lines.append(f"Artifacts ({len(artifacts)}):")
            if artifacts:
                for name in artifacts:
                    item = self._folio._items[name]
                    desc = item.get("description", "(no description)")
                    category = item.get("category", "")
                    category_str = f" ({category})" if category else ""
                    lines.append(f"  • {name}{category_str}: {desc}")
                    # Show file size if available
                    filesize = self._get_item_filesize(item)
                    if filesize is not None:
                        lines.append(f"    ↳ size: {self._format_filesize(filesize)}")
            else:
                lines.append("  (none)")
            lines.append("")

        # Timestamps
        timestamps = contents["timestamps"]
        if timestamps or show_empty:
            lines.append(f"Timestamps ({len(timestamps)}):")
            if timestamps:
                for name in timestamps:
                    item = self._folio._items[name]
                    desc = item.get("description", "(no description)")
                    lines.append(f"  • {name}: {desc}")
                    # Show formatted timestamp
                    iso_string = item.get("iso_string", "")
                    if iso_string:
                        formatted_time = self._format_datetime_for_display(iso_string)
                        lines.append(f"    ↳ time: {formatted_time}")
                    # Show file size if available
                    filesize = self._get_item_filesize(item)
                    if filesize is not None:
                        lines.append(f"    ↳ size: {self._format_filesize(filesize)}")
                    # Show lineage if present
                    if "inputs" in item and item["inputs"]:
                        lines.append(f"    ↳ inputs: {', '.join(item['inputs'])}")
            else:
                lines.append("  (none)")

        # Build final output
        output = "\n".join(lines)

        # Print or return based on parameter
        if return_string:
            return output
        else:
            print(output)
            return None

    def _format_timestamp(self, iso_timestamp: str) -> str:
        """Format an ISO timestamp into a human-readable string using local timezone.

        Args:
            iso_timestamp: ISO format timestamp string

        Returns:
            Human-readable timestamp in local time (e.g., "Today at 2:34 PM EST")

        Examples:
            >>> self._format_timestamp("2024-01-15T14:34:56.789Z")
            'January 15, 2024 at 2:34 PM EST'
        """
        try:
            # Parse ISO timestamp (in UTC)
            dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))

            # Convert to local timezone
            dt_local = dt.astimezone()

            # Get current time in local timezone for comparison
            now = datetime.now().astimezone()

            # Check if today or yesterday
            dt_date = dt_local.date()
            now_date = now.date()

            time_str = dt_local.strftime("%I:%M %p %Z").lstrip(
                "0"
            )  # Remove leading zero from hour

            if dt_date == now_date:
                return f"Today at {time_str}"
            elif (now_date - dt_date).days == 1:
                return f"Yesterday at {time_str}"
            else:
                date_str = dt_local.strftime("%B %d, %Y")
                return f"{date_str} at {time_str}"

        except (ValueError, AttributeError):
            # If parsing fails, return the original
            return iso_timestamp

    def _format_datetime_for_display(self, iso_string: str) -> str:
        """Format a timestamp item's datetime for display in describe().

        Args:
            iso_string: ISO 8601 timestamp string (typically in UTC)

        Returns:
            Human-readable timestamp string (e.g., "2024-01-15 10:30:00 UTC")

        Examples:
            >>> self._format_datetime_for_display("2024-01-15T10:30:00+00:00")
            '2024-01-15 10:30:00 UTC'
        """
        try:
            dt = datetime.fromisoformat(iso_string)
            # Format as: YYYY-MM-DD HH:MM:SS TZ
            return (
                dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                or f"{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )
        except (ValueError, AttributeError):
            # If parsing fails, return the original
            return iso_string

    def _format_metadata_value(self, value: Any, max_length: int = 60) -> str:
        """Format a metadata value for display with smart truncation.

        Args:
            value: The metadata value to format
            max_length: Maximum length for string values before truncation

        Returns:
            Formatted string representation

        Examples:
            >>> self._format_metadata_value("short")
            'short'
            >>> self._format_metadata_value("a" * 100)
            'aaaaaaaaaa... (90 more chars)'
            >>> self._format_metadata_value([1, 2, 3])
            '[1, 2, 3]'
            >>> self._format_metadata_value(list(range(20)))
            '[0, 1, 2, 3, 4, ...] (15 more items)'
        """
        # Handle strings
        if isinstance(value, str):
            if len(value) <= max_length:
                return value
            else:
                truncated = value[:max_length]
                remaining = len(value) - max_length
                return f"{truncated}... ({remaining} more chars)"

        # Handle lists
        if isinstance(value, list):
            if len(value) <= 5:
                return str(value)
            else:
                preview = value[:5]
                remaining = len(value) - 5
                preview_str = str(preview)[:-1]  # Remove closing bracket
                return f"{preview_str}, ...] ({remaining} more items)"

        # Handle dicts
        if isinstance(value, dict):
            if len(value) <= 3:
                return str(value)
            else:
                # Show first 3 items
                preview_items = list(value.items())[:3]
                preview_dict = {k: v for k, v in preview_items}
                remaining = len(value) - 3
                preview_str = str(preview_dict)[:-1]  # Remove closing brace
                return f"{preview_str}, ...}} ({remaining} more fields)"

        # Default: convert to string
        value_str = str(value)
        if len(value_str) <= max_length:
            return value_str
        else:
            truncated = value_str[:max_length]
            remaining = len(value_str) - max_length
            return f"{truncated}... ({remaining} more chars)"

    def _format_filesize(self, size_bytes: int) -> str:
        """Format file size in bytes to human-readable string.

        Args:
            size_bytes: File size in bytes

        Returns:
            Human-readable file size string (e.g., "1.5 MB", "23.4 KB")

        Examples:
            >>> self._format_filesize(1024)
            '1.0 KB'
            >>> self._format_filesize(1536000)
            '1.5 MB'
        """
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(size_bytes)
        unit_index = 0

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        # Format with appropriate precision
        if unit_index == 0:  # Bytes - no decimal
            return f"{int(size)} {units[unit_index]}"
        else:  # Larger units - show one decimal place
            return f"{size:.1f} {units[unit_index]}"

    def _get_item_filesize(self, item: Dict[str, Any]) -> Optional[int]:
        """Get the file size for an item if it has an associated file.

        Args:
            item: Item metadata dictionary

        Returns:
            File size in bytes, or None if no file exists or size cannot be determined

        Examples:
            >>> item = {"item_type": "included_table", "filename": "table.parquet"}
            >>> self._get_item_filesize(item)
            1024000
        """
        # Only included items have local files
        if "filename" not in item:
            return None

        item_type = item.get("item_type", "")

        # Determine the subdirectory based on item type
        if item_type == "included_table":
            subdir = TABLES_DIR
        elif item_type in ["model", "pytorch_model"]:
            subdir = MODELS_DIR
        elif item_type in ["artifact", "numpy_array", "json_data", "timestamp"]:
            subdir = ARTIFACTS_DIR
        else:
            return None

        # Construct full path
        file_path = self._folio._bundle_path / subdir / item["filename"]

        # Get file size if it exists
        try:
            if is_cloud_path(str(self._folio._bundle_dir)):
                # For cloud paths, we can't easily get file size without fetching
                return None
            else:
                # Local path
                return os.path.getsize(file_path)
        except (OSError, FileNotFoundError):
            return None
