"""Utility functions for datafolio."""

import random
from pathlib import Path
from typing import Any, Optional, Union

from typing_extensions import TypedDict

# Bundle structure constants
METADATA_DIR = "metadata"
TABLES_DIR = "tables"
MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts"

METADATA_FILE = "metadata.json"
ITEMS_FILE = "items.json"  # Unified manifest for all items


# TypedDict for unified items manifest
class TableReference(TypedDict, total=False):
    """Metadata for a referenced table (external data)."""

    name: str
    item_type: str  # 'referenced_table'
    path: str
    table_format: str  # 'parquet', 'delta', 'csv'
    num_rows: Optional[int]
    version: Optional[int]  # For Delta tables
    description: Optional[str]
    # Lineage fields
    inputs: Optional[list[str]]  # Names of items this was derived from
    code: Optional[str]  # Code snippet that created this
    created_at: Optional[str]  # ISO 8601 timestamp


class IncludedTable(TypedDict, total=False):
    """Metadata for an included table (stored in bundle)."""

    name: str
    item_type: str  # 'included_table'
    filename: str  # Relative path within bundle
    num_rows: int
    num_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    description: Optional[str]
    # Lineage fields
    inputs: Optional[list[str]]  # Names of tables used to create this
    models: Optional[list[str]]  # Names of models used to create this
    code: Optional[str]  # Code snippet that created this
    created_at: Optional[str]  # ISO 8601 timestamp


class IncludedItem(TypedDict, total=False):
    """Metadata for included models and artifacts."""

    name: str
    item_type: str  # 'model' or 'artifact'
    filename: str  # Relative path within bundle
    category: Optional[str]  # For artifacts: 'plots', 'configs', etc.
    description: Optional[str]
    # Lineage fields (primarily for models)
    inputs: Optional[list[str]]  # Training data, etc.
    hyperparameters: Optional[dict[str, Any]]  # Model hyperparameters
    code: Optional[str]  # Training code snippet
    created_at: Optional[str]  # ISO 8601 timestamp


class TimestampItem(TypedDict, total=False):
    """Metadata for timestamp items."""

    name: str
    item_type: str  # 'timestamp'
    filename: str  # Relative path within bundle (e.g., 'event_time.json')
    iso_string: str  # ISO 8601 timestamp in UTC (e.g., '2024-01-15T10:30:00+00:00')
    unix_timestamp: float  # Unix timestamp for quick reference
    description: Optional[str]
    # Lineage fields
    inputs: Optional[list[str]]  # Names of items this was derived from
    code: Optional[str]  # Code snippet that created this
    created_at: Optional[str]  # ISO 8601 timestamp of when item was added


def is_cloud_path(path: Union[str, Path]) -> bool:
    """Check if a path is a cloud storage path.

    Args:
        path: Path to check

    Returns:
        True if path starts with cloud storage protocol

    Examples:
        >>> is_cloud_path('s3://bucket/file.parquet')
        True
        >>> is_cloud_path('/local/path/file.parquet')
        False
        >>> is_cloud_path('gs://bucket/file.parquet')
        True
    """
    path_str = str(path)
    cloud_prefixes = (
        "s3://",
        "gs://",
        "gcs://",
        "az://",
        "azure://",
        "https://",
        "http://",
        "file://",
    )
    return path_str.startswith(cloud_prefixes)


def resolve_path(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    make_absolute: bool = True,
) -> str:
    """Resolve a path to absolute or relative form.

    Args:
        path: Path to resolve
        base_dir: Base directory for relative paths (defaults to cwd)
        make_absolute: If True, return absolute path; if False, return relative

    Returns:
        Resolved path as string

    Examples:
        >>> resolve_path('data/file.csv')
        '/current/working/dir/data/file.csv'
        >>> resolve_path('s3://bucket/file.parquet')
        's3://bucket/file.parquet'
    """
    # Cloud paths are always returned as-is
    if is_cloud_path(path):
        return str(path)

    path_obj = Path(path)

    # Convert to absolute if requested
    if make_absolute and not path_obj.is_absolute():
        if base_dir:
            path_obj = Path(base_dir) / path_obj
        path_obj = path_obj.resolve()

    return "file://" + str(path_obj)


def validate_table_format(table_format: str) -> None:
    """Validate that table format is supported.

    Args:
        table_format: Format string to validate

    Raises:
        ValueError: If format is not supported
    """
    supported_formats = {"parquet", "delta", "csv"}
    if table_format not in supported_formats:
        raise ValueError(
            f"Unsupported table format: {table_format}. "
            f"Supported formats: {supported_formats}"
        )


def get_file_extension(table_format: str) -> str:
    """Get the file extension for a table format.

    Args:
        table_format: Format name ('parquet', 'csv', etc.)

    Returns:
        File extension including the dot (e.g., '.parquet')

    Examples:
        >>> get_file_extension('parquet')
        '.parquet'
        >>> get_file_extension('csv')
        '.csv'
    """
    extensions = {
        "parquet": ".parquet",
        "csv": ".csv",
        "arrow": ".arrow",
        "delta": "",  # Delta Lake is a directory
    }
    return extensions.get(table_format, f".{table_format}")


# Word lists for random name generation (~60 each = 216K combinations)
_COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "teal",
    "coral",
    "navy",
    "gold",
    "silver",
    "cyan",
    "magenta",
    "lime",
    "indigo",
    "violet",
    "amber",
    "jade",
    "ruby",
    "emerald",
    "sapphire",
    "pearl",
    "bronze",
    "crimson",
    "scarlet",
    "maroon",
    "burgundy",
    "plum",
    "lavender",
    "mint",
    "olive",
    "khaki",
    "tan",
    "beige",
    "ivory",
    "cream",
    "chocolate",
    "coffee",
    "slate",
    "steel",
    "iron",
    "copper",
    "brass",
    "pewter",
    "charcoal",
    "ash",
    "smoke",
    "fog",
    "mist",
    "cloud",
    "sky",
    "ocean",
    "sea",
    "lake",
    "river",
    "forest",
    "meadow",
    "sunset",
    "sunrise",
    "twilight",
    "dusk",
    "dawn",
]

_ADJECTIVES = [
    "happy",
    "bright",
    "calm",
    "bold",
    "swift",
    "gentle",
    "proud",
    "wise",
    "clever",
    "brave",
    "kind",
    "noble",
    "quick",
    "warm",
    "cool",
    "fierce",
    "graceful",
    "mighty",
    "serene",
    "vivid",
    "zealous",
    "eager",
    "lively",
    "keen",
    "agile",
    "nimble",
    "sturdy",
    "steady",
    "solid",
    "stable",
    "strong",
    "tough",
    "smooth",
    "sleek",
    "glossy",
    "shiny",
    "brilliant",
    "radiant",
    "glowing",
    "luminous",
    "silent",
    "quiet",
    "loud",
    "sonic",
    "rapid",
    "speedy",
    "fleet",
    "hasty",
    "patient",
    "careful",
    "mindful",
    "alert",
    "sharp",
    "acute",
    "astute",
    "crafty",
    "daring",
    "fearless",
    "valiant",
    "heroic",
    "epic",
    "grand",
    "majestic",
]

_NOUNS = [
    "falcon",
    "tiger",
    "dolphin",
    "eagle",
    "wolf",
    "bear",
    "fox",
    "hawk",
    "lion",
    "otter",
    "raven",
    "deer",
    "badger",
    "lynx",
    "puma",
    "shark",
    "whale",
    "owl",
    "panther",
    "cobra",
    "python",
    "viper",
    "dragon",
    "phoenix",
    "griffin",
    "pegasus",
    "sphinx",
    "hydra",
    "kraken",
    "basilisk",
    "unicorn",
    "centaur",
    "leopard",
    "cheetah",
    "jaguar",
    "cougar",
    "bobcat",
    "caracal",
    "serval",
    "ocelot",
    "condor",
    "albatross",
    "pelican",
    "heron",
    "crane",
    "stork",
    "ibis",
    "egret",
    "salmon",
    "trout",
    "marlin",
    "barracuda",
    "tuna",
    "swordfish",
    "manta",
    "orca",
    "bison",
    "moose",
    "elk",
    "antelope",
    "gazelle",
    "impala",
    "zebra",
]


def generate_random_name() -> str:
    """Generate a random readable name in format: color-adjective-noun.

    Returns:
        Random name like 'blue-happy-falcon'

    Examples:
        >>> name = generate_random_name()
        >>> len(name.split('-'))
        3
        >>> all(part.isalpha() for part in name.split('-'))
        True
    """
    color = random.choice(_COLORS)
    adjective = random.choice(_ADJECTIVES)
    noun = random.choice(_NOUNS)
    return f"{adjective}-{color}-{noun}"


def make_bundle_name(prefix: Optional[str] = None) -> str:
    """Create a bundle name with optional prefix and random suffix.

    Args:
        prefix: Optional prefix for the bundle name

    Returns:
        Bundle name in format: 'prefix-color-adjective-noun' or 'color-adjective-noun'

    Examples:
        >>> name = make_bundle_name('my-experiment')
        >>> name.startswith('my-experiment-')
        True
        >>> len(name.split('-'))
        5
        >>> name = make_bundle_name()
        >>> len(name.split('-'))
        3
    """
    random_suffix = generate_random_name()
    if prefix:
        return f"{prefix}-{random_suffix}"
    return random_suffix
