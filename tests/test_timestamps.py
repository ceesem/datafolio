"""Tests for timestamp handling functionality."""

from datetime import datetime, timezone

import pytest

from datafolio import DataFolio


class TestAddTimestamp:
    """Tests for add_timestamp method."""

    def test_add_timestamp_basic_utc(self, tmp_path):
        """Test adding a basic UTC timestamp."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time)

        assert "event_time" in folio._items
        assert folio._items["event_time"]["item_type"] == "timestamp"
        assert folio._items["event_time"]["iso_string"] == "2024-01-15T10:30:00+00:00"
        # Check unix timestamp matches the datetime we provided
        expected_unix = event_time.timestamp()
        assert folio._items["event_time"]["unix_timestamp"] == pytest.approx(
            expected_unix
        )

    def test_add_timestamp_with_description(self, tmp_path):
        """Test adding timestamp with description."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time, description="Event occurred")

        item = folio._items["event_time"]
        assert item["description"] == "Event occurred"

    def test_add_timestamp_with_lineage(self, tmp_path):
        """Test adding timestamp with lineage metadata."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add(
            "event_time",
            event_time,
            inputs=["event_log"],
            code='timestamp = event_log.iloc[0]["timestamp"]',
        )

        item = folio._items["event_time"]
        assert item["inputs"] == ["event_log"]
        assert item["code"] == 'timestamp = event_log.iloc[0]["timestamp"]'

    def test_add_timestamp_from_unix_int(self, tmp_path):
        """Unix timestamps enter as datetimes; bare ints store as JSON."""
        folio = DataFolio(tmp_path / "test")
        unix_ts = 1705318200
        expected_dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        folio.add("start_time", expected_dt)

        item = folio._items["start_time"]
        assert item["iso_string"] == expected_dt.isoformat()
        assert item["unix_timestamp"] == pytest.approx(float(unix_ts))

        # A bare number is JSON data, not a timestamp
        folio.add("raw_number", unix_ts)
        assert folio._items["raw_number"]["item_type"] == "json_data"

    def test_add_timestamp_from_unix_float(self, tmp_path):
        """Fractional Unix timestamps survive the datetime round-trip."""
        folio = DataFolio(tmp_path / "test")
        dt = datetime.fromtimestamp(1705318200.5, tz=timezone.utc)
        folio.add("start_time", dt)

        item = folio._items["start_time"]
        assert item["unix_timestamp"] == pytest.approx(1705318200.5)

    def test_add_timestamp_non_utc_timezone(self, tmp_path):
        """Test adding timestamp with non-UTC timezone (converts to UTC)."""
        folio = DataFolio(tmp_path / "test")

        # Create a timezone-aware datetime in a different timezone
        # Using fixed offset for testing (UTC+5)
        from datetime import timedelta

        tz_plus_5 = timezone(timedelta(hours=5))
        local_time = datetime(2024, 1, 15, 15, 30, 0, tzinfo=tz_plus_5)

        folio.add("local_event", local_time)

        item = folio._items["local_event"]
        # Should be converted to UTC (15:30 UTC+5 = 10:30 UTC)
        assert item["iso_string"] == "2024-01-15T10:30:00+00:00"

    def test_add_timestamp_method_chaining(self, tmp_path):
        """Test method chaining works."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = folio.add("event_time", event_time)

        assert result is folio  # Returns self

    def test_add_timestamp_overwrite_false(self, tmp_path):
        """Test error when adding duplicate timestamp without overwrite."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time)

        with pytest.raises(ValueError, match="already exists"):
            folio.add("event_time", event_time)

    def test_add_timestamp_overwrite_true(self, tmp_path):
        """Test overwriting existing timestamp with overwrite=True."""
        folio = DataFolio(tmp_path / "test")
        event_time1 = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        event_time2 = datetime(2024, 1, 16, 10, 30, 0, tzinfo=timezone.utc)

        folio.add("event_time", event_time1)
        folio.add("event_time", event_time2, overwrite=True)

        item = folio._items["event_time"]
        assert item["iso_string"] == "2024-01-16T10:30:00+00:00"

    def test_add_timestamp_naive_datetime_raises_error(self, tmp_path):
        """Test error when adding naive datetime."""
        folio = DataFolio(tmp_path / "test")
        naive_time = datetime(2024, 1, 15, 10, 30, 0)  # No timezone

        with pytest.raises(ValueError, match="Naive datetime objects are not allowed"):
            folio.add("event_time", naive_time)

    def test_add_date_string_is_json(self, tmp_path):
        """A date-like string stores as JSON — only real datetimes are
        timestamps."""
        folio = DataFolio(tmp_path / "test")

        folio.add("event_time", "2024-01-15")
        assert folio._items["event_time"]["item_type"] == "json_data"
        assert folio.get("event_time") == "2024-01-15"

    def test_add_timestamp_appears_in_list_contents(self, tmp_path):
        """Test timestamp appears in list_contents."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time)

        contents = folio.list_contents()
        assert "event_time" in contents["timestamps"]

    def test_add_timestamp_creates_file(self, tmp_path):
        """Test that timestamp file is created on disk."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time)

        # Check file exists (payload filename is versioned)
        timestamp_file = (
            tmp_path / "test" / "artifacts" / folio._items["event_time"]["filename"]
        )
        assert timestamp_file.exists()


class TestGetTimestamp:
    """Tests for get_timestamp method."""

    def test_get_timestamp_as_datetime(self, tmp_path):
        """Test retrieving timestamp as datetime (default)."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time)

        retrieved = folio.get("event_time")

        assert isinstance(retrieved, datetime)
        assert retrieved.tzinfo is not None  # Timezone-aware
        assert retrieved.isoformat() == "2024-01-15T10:30:00+00:00"

    def test_get_timestamp_as_unix(self, tmp_path):
        """Test retrieving timestamp as Unix timestamp."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time)

        retrieved = folio.get("event_time", as_unix=True)

        assert isinstance(retrieved, float)
        expected_unix = event_time.timestamp()
        assert retrieved == pytest.approx(expected_unix)

    def test_get_timestamp_not_found(self, tmp_path):
        """Test error when timestamp doesn't exist."""
        folio = DataFolio(tmp_path / "test")

        with pytest.raises(KeyError, match="not found"):
            folio.get("nonexistent")

    def test_get_non_timestamp_returns_stored_type(self, tmp_path):
        """get() returns the stored type; as_unix on a non-timestamp errors."""
        folio = DataFolio(tmp_path / "test")
        folio.add("config", {"key": "value"})

        assert folio.get("config") == {"key": "value"}
        with pytest.raises(TypeError, match="as_unix"):
            folio.get("config", as_unix=True)

    def test_get_timestamp_round_trip(self, tmp_path):
        """Test adding and retrieving timestamp maintains value."""
        folio = DataFolio(tmp_path / "test")
        original = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        folio.add("event_time", original)

        retrieved = folio.get("event_time")

        assert retrieved == original

    def test_get_timestamp_round_trip_with_microseconds(self, tmp_path):
        """Test timestamp with microseconds is preserved."""
        folio = DataFolio(tmp_path / "test")
        original = datetime(2024, 1, 15, 10, 30, 45, 123456, tzinfo=timezone.utc)
        folio.add("event_time", original)

        retrieved = folio.get("event_time")

        assert retrieved == original


class TestTimestampIntegration:
    """Integration tests for timestamp functionality."""

    def test_timestamp_save_and_load(self, tmp_path):
        """Test saving and loading DataFolio with timestamps."""
        bundle_path = tmp_path / "test"

        # Create and save
        folio1 = DataFolio(bundle_path)
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio1.add("event_time", event_time, description="Event occurred")

        # Load in new instance
        folio2 = DataFolio(bundle_path)

        assert "event_time" in folio2._items
        retrieved = folio2.get("event_time")
        assert retrieved == event_time

    def test_timestamp_in_describe(self, tmp_path):
        """Test timestamp appears in describe() output."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time, description="Event occurred")

        output = folio.describe(return_string=True)

        assert "Timestamps" in output
        assert "event_time" in output
        assert "Event occurred" in output

    def test_timestamp_via_data_accessor(self, tmp_path):
        """Test accessing timestamp via folio.data.name pattern."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time)

        # Access via .data accessor
        retrieved = folio.data.event_time.content

        assert isinstance(retrieved, datetime)
        assert retrieved == event_time

    def test_timestamp_data_accessor_metadata(self, tmp_path):
        """Test accessing timestamp metadata via data accessor."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time, description="Event occurred")

        # Access metadata via data accessor
        assert folio.data.event_time.description == "Event occurred"
        # Check item_type in the underlying items dict
        assert folio._items["event_time"]["item_type"] == "timestamp"

    def test_multiple_timestamps(self, tmp_path):
        """Test adding multiple timestamps."""
        folio = DataFolio(tmp_path / "test")

        start = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)

        folio.add("start_time", start, description="Start")
        folio.add("end_time", end, description="End")

        contents = folio.list_contents()
        assert len(contents["timestamps"]) == 2
        assert "start_time" in contents["timestamps"]
        assert "end_time" in contents["timestamps"]

    def test_timestamp_with_unix_zero(self, tmp_path):
        """Test the epoch as a timestamp."""
        folio = DataFolio(tmp_path / "test")
        folio.add("epoch", datetime.fromtimestamp(0, tz=timezone.utc))

        retrieved = folio.get("epoch")
        assert retrieved == datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_timestamp_with_negative_unix(self, tmp_path):
        """Test a pre-epoch timestamp."""
        folio = DataFolio(tmp_path / "test")
        folio.add(
            "before_epoch", datetime.fromtimestamp(-86400, tz=timezone.utc)
        )  # 1 day before epoch

        retrieved = folio.get("before_epoch")
        assert retrieved == datetime(1969, 12, 31, 0, 0, 0, tzinfo=timezone.utc)

    def test_timestamp_list_contents_empty(self, tmp_path):
        """Test list_contents shows empty timestamps list."""
        folio = DataFolio(tmp_path / "test")

        contents = folio.list_contents()
        assert "timestamps" in contents
        assert contents["timestamps"] == []

    def test_timestamp_describe_empty(self, tmp_path):
        """Test describe() handles empty timestamps section."""
        folio = DataFolio(tmp_path / "test")

        output = folio.describe(return_string=True, show_empty=True)
        assert "Timestamps (0):" in output
        assert "(none)" in output


class TestTimestampEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_timestamp_very_large_unix_timestamp(self, tmp_path):
        """Test very large timestamp (far future)."""
        folio = DataFolio(tmp_path / "test")
        # Year 2100
        folio.add("future", datetime.fromtimestamp(4102444800, tz=timezone.utc))

        retrieved = folio.get("future", as_unix=True)
        assert retrieved == pytest.approx(4102444800.0)

    def test_timestamp_with_special_characters_in_name(self, tmp_path):
        """Test timestamp name with special characters."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        # Use underscores and numbers
        folio.add("event_2024_01", event_time)

        retrieved = folio.get("event_2024_01")
        assert retrieved == event_time

    def test_timestamp_metadata_has_created_at(self, tmp_path):
        """Test timestamp metadata includes created_at field."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("event_time", event_time)

        item = folio._items["event_time"]
        assert "created_at" in item
        # created_at should be a valid ISO string
        datetime.fromisoformat(item["created_at"])

    def test_timestamp_filename_format(self, tmp_path):
        """Test timestamp is stored with correct filename."""
        folio = DataFolio(tmp_path / "test")
        event_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        folio.add("my_event", event_time)

        item = folio._items["my_event"]
        # Payload filename is versioned but derived from the name + .json.
        assert item["filename"].startswith("my_event--r")
        assert item["filename"].endswith(".json")
