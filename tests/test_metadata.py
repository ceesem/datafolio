"""Unit tests for MetadataDict.

These tests directly test the MetadataDict class to ensure full coverage
of the auto-save functionality and timestamp management.
"""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from datafolio.metadata import MetadataDict


@pytest.fixture
def mock_folio():
    """Create a mock DataFolio instance."""
    folio = Mock()
    folio._save_metadata = Mock()
    folio._read_only = False  # Not in read-only mode
    return folio


# ============================================================================
# Test initialization
# ============================================================================


def test_init_empty(mock_folio):
    """Test initializing empty MetadataDict."""
    md = MetadataDict(mock_folio)

    assert len(md) == 0
    assert md._parent is mock_folio
    # Should not trigger save during initialization
    mock_folio._save_metadata.assert_not_called()


def test_init_with_data(mock_folio):
    """Test initializing MetadataDict with initial data."""
    initial_data = {"key1": "value1", "key2": "value2"}
    md = MetadataDict(mock_folio, initial_data)

    assert len(md) == 2
    assert md["key1"] == "value1"
    assert md["key2"] == "value2"
    assert md._parent is mock_folio
    # Should not trigger save during initialization
    mock_folio._save_metadata.assert_not_called()


def test_init_with_kwargs(mock_folio):
    """Test initializing MetadataDict with keyword arguments."""
    md = MetadataDict(mock_folio, name="test", version="1.0")

    assert len(md) == 2
    assert md["name"] == "test"
    assert md["version"] == "1.0"
    # Should not trigger save during initialization
    mock_folio._save_metadata.assert_not_called()


# ============================================================================
# Test __setitem__
# ============================================================================


def test_setitem_triggers_save(mock_folio):
    """Test that setting an item triggers save."""
    md = MetadataDict(mock_folio)

    md["key"] = "value"

    assert md["key"] == "value"
    assert "updated_at" in md
    mock_folio._save_metadata.assert_called_once()


def test_setitem_updates_timestamp(mock_folio):
    """Test that setting an item updates the timestamp."""
    md = MetadataDict(mock_folio)

    # Set initial timestamp
    old_time = "2024-01-01T00:00:00+00:00"
    md._parent._save_metadata = Mock()  # Reset mock
    super(MetadataDict, md).__setitem__("updated_at", old_time)

    # Set a new item
    md["key"] = "value"

    # Timestamp should be updated
    assert md["updated_at"] != old_time
    # Parse and verify it's a valid ISO timestamp
    timestamp = datetime.fromisoformat(md["updated_at"])
    assert timestamp.tzinfo == timezone.utc


def test_setitem_updated_at_no_recursion(mock_folio):
    """Test that setting 'updated_at' doesn't cause infinite recursion."""
    md = MetadataDict(mock_folio)
    mock_folio._save_metadata.reset_mock()

    # Setting updated_at directly should not update itself again
    md["updated_at"] = "2024-01-01T00:00:00+00:00"

    assert md["updated_at"] == "2024-01-01T00:00:00+00:00"
    # Should still trigger save, but not update timestamp again
    mock_folio._save_metadata.assert_called_once()


def test_setitem_overwrite_value(mock_folio):
    """Test overwriting an existing value."""
    md = MetadataDict(mock_folio)
    md["key"] = "value1"
    mock_folio._save_metadata.reset_mock()

    md["key"] = "value2"

    assert md["key"] == "value2"
    mock_folio._save_metadata.assert_called_once()


# ============================================================================
# Test __delitem__
# ============================================================================


def test_delitem_triggers_save(mock_folio):
    """Test that deleting an item triggers save."""
    md = MetadataDict(mock_folio)
    md["key"] = "value"
    mock_folio._save_metadata.reset_mock()

    del md["key"]

    assert "key" not in md
    assert "updated_at" in md
    mock_folio._save_metadata.assert_called_once()


def test_delitem_updates_timestamp(mock_folio):
    """Test that deleting an item updates the timestamp."""
    md = MetadataDict(mock_folio)
    md["key"] = "value"

    # Set old timestamp
    old_time = "2024-01-01T00:00:00+00:00"
    super(MetadataDict, md).__setitem__("updated_at", old_time)
    mock_folio._save_metadata.reset_mock()

    del md["key"]

    # Timestamp should be updated
    assert md["updated_at"] != old_time
    timestamp = datetime.fromisoformat(md["updated_at"])
    assert timestamp.tzinfo == timezone.utc


def test_delitem_nonexistent_raises_error(mock_folio):
    """Test that deleting non-existent key raises KeyError."""
    md = MetadataDict(mock_folio)

    with pytest.raises(KeyError):
        del md["nonexistent"]


# ============================================================================
# Test update
# ============================================================================


def test_update_with_dict(mock_folio):
    """Test updating with a dictionary."""
    md = MetadataDict(mock_folio)
    mock_folio._save_metadata.reset_mock()

    md.update({"key1": "value1", "key2": "value2"})

    assert md["key1"] == "value1"
    assert md["key2"] == "value2"
    assert "updated_at" in md
    mock_folio._save_metadata.assert_called_once()


def test_update_with_kwargs(mock_folio):
    """Test updating with keyword arguments."""
    md = MetadataDict(mock_folio)
    mock_folio._save_metadata.reset_mock()

    md.update(key1="value1", key2="value2")

    assert md["key1"] == "value1"
    assert md["key2"] == "value2"
    assert "updated_at" in md
    mock_folio._save_metadata.assert_called_once()


def test_update_with_both(mock_folio):
    """Test updating with both dict and kwargs."""
    md = MetadataDict(mock_folio)
    mock_folio._save_metadata.reset_mock()

    md.update({"key1": "value1"}, key2="value2")

    assert md["key1"] == "value1"
    assert md["key2"] == "value2"
    assert "updated_at" in md
    mock_folio._save_metadata.assert_called_once()


def test_update_updates_timestamp(mock_folio):
    """Test that update updates the timestamp."""
    md = MetadataDict(mock_folio)

    # Set old timestamp
    old_time = "2024-01-01T00:00:00+00:00"
    super(MetadataDict, md).__setitem__("updated_at", old_time)
    mock_folio._save_metadata.reset_mock()

    md.update({"key": "value"})

    # Timestamp should be updated
    assert md["updated_at"] != old_time
    timestamp = datetime.fromisoformat(md["updated_at"])
    assert timestamp.tzinfo == timezone.utc


def test_update_empty_dict(mock_folio):
    """Test updating with empty dict still triggers save."""
    md = MetadataDict(mock_folio)
    mock_folio._save_metadata.reset_mock()

    md.update({})

    # Even empty update should trigger save and timestamp update
    assert "updated_at" in md
    mock_folio._save_metadata.assert_called_once()


# ============================================================================
# Test clear
# ============================================================================


def test_clear_removes_all(mock_folio):
    """Test that clear removes all items."""
    md = MetadataDict(mock_folio, {"key1": "value1", "key2": "value2"})
    mock_folio._save_metadata.reset_mock()

    md.clear()

    # All items should be removed except updated_at
    assert len(md) == 1
    assert "key1" not in md
    assert "key2" not in md
    assert "updated_at" in md
    mock_folio._save_metadata.assert_called_once()


def test_clear_updates_timestamp(mock_folio):
    """Test that clear updates the timestamp."""
    md = MetadataDict(mock_folio, {"key": "value"})

    # Set old timestamp
    old_time = "2024-01-01T00:00:00+00:00"
    super(MetadataDict, md).__setitem__("updated_at", old_time)
    mock_folio._save_metadata.reset_mock()

    md.clear()

    # Timestamp should be updated
    assert md["updated_at"] != old_time
    timestamp = datetime.fromisoformat(md["updated_at"])
    assert timestamp.tzinfo == timezone.utc


def test_clear_empty_dict(mock_folio):
    """Test clearing an already empty dict."""
    md = MetadataDict(mock_folio)
    mock_folio._save_metadata.reset_mock()

    md.clear()

    # Should still trigger save and add timestamp
    assert len(md) == 1
    assert "updated_at" in md
    mock_folio._save_metadata.assert_called_once()


# ============================================================================
# Test setdefault
# ============================================================================


def test_setdefault_new_key(mock_folio):
    """Test setdefault with a new key triggers save."""
    md = MetadataDict(mock_folio)
    mock_folio._save_metadata.reset_mock()

    result = md.setdefault("key", "default")

    assert result == "default"
    assert md["key"] == "default"
    assert "updated_at" in md
    mock_folio._save_metadata.assert_called_once()


def test_setdefault_existing_key(mock_folio):
    """Test setdefault with existing key does not trigger save."""
    md = MetadataDict(mock_folio)
    md["key"] = "existing"
    mock_folio._save_metadata.reset_mock()

    result = md.setdefault("key", "default")

    assert result == "existing"
    assert md["key"] == "existing"
    # Should NOT trigger save since key existed
    mock_folio._save_metadata.assert_not_called()


def test_setdefault_none_default(mock_folio):
    """Test setdefault with None as default value."""
    md = MetadataDict(mock_folio)
    mock_folio._save_metadata.reset_mock()

    result = md.setdefault("key")

    assert result is None
    assert md["key"] is None
    assert "updated_at" in md
    mock_folio._save_metadata.assert_called_once()


def test_setdefault_updates_timestamp_only_if_new(mock_folio):
    """Test that setdefault only updates timestamp for new keys."""
    md = MetadataDict(mock_folio)
    md["existing"] = "value"

    # Set old timestamp
    old_time = "2024-01-01T00:00:00+00:00"
    super(MetadataDict, md).__setitem__("updated_at", old_time)
    mock_folio._save_metadata.reset_mock()

    # setdefault on existing key - should not update timestamp
    md.setdefault("existing", "default")
    assert md["updated_at"] == old_time
    mock_folio._save_metadata.assert_not_called()

    # setdefault on new key - should update timestamp
    md.setdefault("new_key", "new_value")
    assert md["updated_at"] != old_time
    mock_folio._save_metadata.assert_called_once()


# ============================================================================
# Test dict behavior
# ============================================================================


def test_behaves_like_dict(mock_folio):
    """Test that MetadataDict behaves like a regular dict."""
    md = MetadataDict(mock_folio, {"key1": "value1"})
    md["key2"] = "value2"

    # Test len
    assert len(md) >= 2  # At least key1, key2 (plus updated_at)

    # Test in operator
    assert "key1" in md
    assert "key2" in md
    assert "nonexistent" not in md

    # Test get
    assert md.get("key1") == "value1"
    assert md.get("nonexistent") is None
    assert md.get("nonexistent", "default") == "default"

    # Test keys, values, items
    assert "key1" in md.keys()
    assert "value1" in md.values()
    assert ("key1", "value1") in md.items()


def test_iteration(mock_folio):
    """Test iterating over MetadataDict."""
    md = MetadataDict(mock_folio, {"key1": "value1", "key2": "value2"})

    # Test iteration over keys
    keys = list(md)
    assert "key1" in keys
    assert "key2" in keys

    # Test iteration over items
    items = list(md.items())
    assert ("key1", "value1") in items
    assert ("key2", "value2") in items


# ============================================================================
# Integration-style tests
# ============================================================================


def test_multiple_operations_sequence(mock_folio):
    """Test a sequence of operations."""
    md = MetadataDict(mock_folio)
    mock_folio._save_metadata.reset_mock()

    # Add items
    md["key1"] = "value1"
    md["key2"] = "value2"

    # Update
    md.update({"key3": "value3"})

    # Delete
    del md["key1"]

    # Clear
    md.clear()

    # Every operation should have triggered a save
    assert mock_folio._save_metadata.call_count >= 4
    # Only updated_at should remain
    assert len(md) == 1
    assert "updated_at" in md
