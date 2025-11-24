"""Tests for CLI snapshot commands.

These tests verify:
- All snapshot CLI commands work correctly
- Bundle initialization via CLI
- Error handling and user feedback
- Output formatting
"""

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from datafolio import DataFolio
from datafolio.cli.main import cli


class TestSnapshotCreate:
    """Test snapshot create command."""

    def test_create_basic(self, tmp_path):
        """Test basic snapshot creation via CLI."""
        # Setup bundle
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Run CLI command
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--bundle", str(tmp_path / "test"), "snapshot", "create", "v1.0"],
        )

        assert result.exit_code == 0
        assert "Created snapshot 'v1.0'" in result.output

        # Verify snapshot was created
        folio_reloaded = DataFolio(tmp_path / "test")
        assert "v1.0" in folio_reloaded._snapshots

    def test_create_with_description_and_tags(self, tmp_path):
        """Test snapshot creation with description and tags."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--bundle",
                str(tmp_path / "test"),
                "snapshot",
                "create",
                "v1.0",
                "-d",
                "Test snapshot",
                "-t",
                "baseline",
                "-t",
                "test",
            ],
        )

        assert result.exit_code == 0
        assert "Created snapshot 'v1.0'" in result.output

        # Verify snapshot metadata
        folio_reloaded = DataFolio(tmp_path / "test")
        snapshot_info = folio_reloaded.get_snapshot_info("v1.0")
        assert snapshot_info["description"] == "Test snapshot"
        assert "baseline" in snapshot_info["tags"]
        assert "test" in snapshot_info["tags"]

    def test_create_duplicate_fails(self, tmp_path):
        """Test that duplicate snapshot names fail."""
        folio = DataFolio(tmp_path / "test")
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "create", "v1.0"]
        )

        assert result.exit_code == 1
        assert "Error" in result.output


class TestSnapshotList:
    """Test snapshot list command."""

    def test_list_empty(self, tmp_path):
        """Test listing when no snapshots exist."""
        folio = DataFolio(tmp_path / "test")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "list"]
        )

        assert result.exit_code == 0
        assert "No snapshots found" in result.output

    def test_list_multiple_snapshots(self, tmp_path):
        """Test listing multiple snapshots."""
        folio = DataFolio(tmp_path / "test")
        folio.create_snapshot("v1.0", description="First")
        folio.create_snapshot("v2.0", description="Second")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "list"]
        )

        assert result.exit_code == 0
        assert "v1.0" in result.output
        assert "v2.0" in result.output

    def test_list_filter_by_tag(self, tmp_path):
        """Test filtering snapshots by tag."""
        folio = DataFolio(tmp_path / "test")
        folio.create_snapshot("v1.0", tags=["baseline"])
        folio.create_snapshot("v2.0", tags=["experimental"])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--bundle",
                str(tmp_path / "test"),
                "snapshot",
                "list",
                "--tag",
                "baseline",
            ],
        )

        if result.exit_code != 0:
            print(f"Output: {result.output}")
            if result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception),
                    result.exception,
                    result.exception.__traceback__,
                )
        assert result.exit_code == 0
        assert "v1.0" in result.output
        assert "v2.0" not in result.output


class TestSnapshotShow:
    """Test snapshot show command."""

    def test_show_basic(self, tmp_path):
        """Test showing snapshot details."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.metadata["accuracy"] = 0.89
        folio.create_snapshot("v1.0", description="Test snapshot")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "show", "v1.0"]
        )

        assert result.exit_code == 0
        assert "v1.0" in result.output
        assert "Test snapshot" in result.output

    def test_show_nonexistent(self, tmp_path):
        """Test showing nonexistent snapshot."""
        folio = DataFolio(tmp_path / "test")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--bundle", str(tmp_path / "test"), "snapshot", "show", "nonexistent"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output


class TestSnapshotCompare:
    """Test snapshot compare command."""

    def test_compare_two_snapshots(self, tmp_path):
        """Test comparing two snapshots."""
        folio = DataFolio(tmp_path / "test")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data1", df1)
        folio.create_snapshot("v1.0")

        df2 = pd.DataFrame({"b": [4, 5, 6]})
        folio.add_table("data2", df2)
        folio.create_snapshot("v2.0")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--bundle",
                str(tmp_path / "test"),
                "snapshot",
                "compare",
                "v1.0",
                "v2.0",
            ],
        )

        assert result.exit_code == 0
        assert "Comparing v1.0 → v2.0" in result.output
        assert "data2" in result.output  # Added item

    def test_compare_nonexistent_snapshot(self, tmp_path):
        """Test comparing with nonexistent snapshot."""
        folio = DataFolio(tmp_path / "test")
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--bundle",
                str(tmp_path / "test"),
                "snapshot",
                "compare",
                "v1.0",
                "nonexistent",
            ],
        )

        assert result.exit_code == 1


class TestSnapshotDelete:
    """Test snapshot delete command."""

    def test_delete_with_confirmation(self, tmp_path):
        """Test deleting snapshot with confirmation."""
        folio = DataFolio(tmp_path / "test")
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--bundle", str(tmp_path / "test"), "snapshot", "delete", "v1.0"],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "Deleted snapshot 'v1.0'" in result.output

        # Verify deletion
        folio_reloaded = DataFolio(tmp_path / "test")
        assert "v1.0" not in folio_reloaded._snapshots

    def test_delete_with_yes_flag(self, tmp_path):
        """Test deleting snapshot with -y flag."""
        folio = DataFolio(tmp_path / "test")
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--bundle", str(tmp_path / "test"), "snapshot", "delete", "v1.0", "-y"],
        )

        assert result.exit_code == 0
        assert "Deleted snapshot 'v1.0'" in result.output

    def test_delete_cancelled(self, tmp_path):
        """Test cancelling snapshot deletion."""
        folio = DataFolio(tmp_path / "test")
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--bundle", str(tmp_path / "test"), "snapshot", "delete", "v1.0"],
            input="n\n",
        )

        assert result.exit_code == 0
        assert "Cancelled" in result.output

        # Verify not deleted
        folio_reloaded = DataFolio(tmp_path / "test")
        assert "v1.0" in folio_reloaded._snapshots


class TestSnapshotGC:
    """Test snapshot garbage collection command."""

    def test_gc_dry_run(self, tmp_path):
        """Test garbage collection dry run."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        # Create orphaned version
        folio.add_table("data", pd.DataFrame({"a": [4, 5, 6]}), overwrite=True)
        folio.add_table("data", pd.DataFrame({"a": [7, 8, 9]}), overwrite=True)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--bundle", str(tmp_path / "test"), "snapshot", "gc", "--dry-run"],
        )

        assert result.exit_code == 0

    def test_gc_actual(self, tmp_path):
        """Test actual garbage collection."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "gc"]
        )

        assert result.exit_code == 0


class TestSnapshotReproduce:
    """Test snapshot reproduce command."""

    def test_reproduce_instructions(self, tmp_path):
        """Test showing reproduction instructions."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--bundle", str(tmp_path / "test"), "snapshot", "reproduce", "v1.0"],
        )

        assert result.exit_code == 0
        assert "reproduce" in result.output.lower()


class TestSnapshotStatus:
    """Test snapshot status command."""

    def test_status_no_snapshots(self, tmp_path):
        """Test status when no snapshots exist."""
        folio = DataFolio(tmp_path / "test")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "status"]
        )

        assert result.exit_code == 0
        assert "No snapshots yet" in result.output

    def test_status_no_changes(self, tmp_path):
        """Test status when no changes since last snapshot."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "status"]
        )

        assert result.exit_code == 0
        assert "No changes since last snapshot" in result.output

    def test_status_with_changes(self, tmp_path):
        """Test status showing changes since last snapshot."""
        folio = DataFolio(tmp_path / "test")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data1", df1)
        folio.create_snapshot("v1.0")

        # Add new item
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        folio.add_table("data2", df2)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "status"]
        )

        assert result.exit_code == 0
        assert "Changes since last snapshot" in result.output
        assert "data2" in result.output  # Added item

    def test_status_with_modified_items(self, tmp_path):
        """Test status showing modified items."""
        folio = DataFolio(tmp_path / "test")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df1)
        folio.create_snapshot("v1.0")

        # Modify item
        df2 = pd.DataFrame({"a": [4, 5, 6]})
        folio.add_table("data", df2, overwrite=True)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "status"]
        )

        assert result.exit_code == 0
        assert "Modified" in result.output
        assert "data" in result.output


class TestSnapshotDiff:
    """Test snapshot diff command."""

    def test_diff_to_last_snapshot(self, tmp_path):
        """Test diff to last snapshot (no argument)."""
        folio = DataFolio(tmp_path / "test")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data1", df1)
        folio.create_snapshot("v1.0")

        # Add new item
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        folio.add_table("data2", df2)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "diff"]
        )

        assert result.exit_code == 0
        assert "Comparing current state" in result.output
        assert "data2" in result.output  # Added item

    def test_diff_to_specific_snapshot(self, tmp_path):
        """Test diff to specific snapshot."""
        folio = DataFolio(tmp_path / "test")
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data1", df1)
        folio.create_snapshot("v1.0")

        df2 = pd.DataFrame({"b": [4, 5, 6]})
        folio.add_table("data2", df2)
        folio.create_snapshot("v2.0")

        # Add another item
        df3 = pd.DataFrame({"c": [7, 8, 9]})
        folio.add_table("data3", df3)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "diff", "v1.0"]
        )

        assert result.exit_code == 0
        assert "v1.0" in result.output
        assert "data2" in result.output  # Added since v1.0
        assert "data3" in result.output  # Added since v1.0

    def test_diff_no_changes(self, tmp_path):
        """Test diff when no changes."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "test"), "snapshot", "diff"]
        )

        assert result.exit_code == 0
        assert "No changes" in result.output


class TestInit:
    """Test bundle initialization command."""

    def test_init_new_bundle(self, tmp_path):
        """Test initializing new bundle."""
        bundle_path = tmp_path / "new-bundle"

        runner = CliRunner()
        result = runner.invoke(cli, ["init", str(bundle_path)])

        if result.exit_code != 0:
            print(f"Error output: {result.output}")
        assert result.exit_code == 0
        assert "Initialized DataFolio bundle" in result.output
        assert (bundle_path / "items.json").exists()
        assert (bundle_path / "metadata.json").exists()

    def test_init_with_description(self, tmp_path):
        """Test initializing bundle with description."""
        bundle_path = tmp_path / "new-bundle"

        runner = CliRunner()
        result = runner.invoke(cli, ["init", str(bundle_path), "-d", "Test experiment"])

        assert result.exit_code == 0
        assert "Initialized DataFolio bundle" in result.output

        # Verify metadata
        folio = DataFolio(bundle_path)
        assert folio.metadata.get("description") == "Test experiment"

    def test_init_existing_bundle_with_confirmation(self, tmp_path):
        """Test reinitializing existing bundle."""
        bundle_path = tmp_path / "existing"
        DataFolio(bundle_path)  # Create existing bundle

        runner = CliRunner()
        result = runner.invoke(cli, ["init", str(bundle_path)], input="y\n")

        assert result.exit_code == 0
        assert "Bundle already exists" in result.output

    def test_init_existing_bundle_cancelled(self, tmp_path):
        """Test cancelling reinitialization."""
        bundle_path = tmp_path / "existing"
        DataFolio(bundle_path)  # Create existing bundle

        runner = CliRunner()
        result = runner.invoke(cli, ["init", str(bundle_path)], input="n\n")

        assert result.exit_code == 0
        assert "Cancelled" in result.output


class TestBundlePathResolution:
    """Test bundle path resolution from different sources."""

    def test_explicit_bundle_flag(self, tmp_path):
        """Test --bundle flag."""
        bundle_path = tmp_path / "test"
        folio = DataFolio(bundle_path)
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(cli, ["--bundle", str(bundle_path), "snapshot", "list"])

        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
            if result.exception:
                print(f"Exception: {result.exception}")
        assert result.exit_code == 0
        assert "v1.0" in result.output

    def test_c_flag(self, tmp_path):
        """Test -C flag (git-style)."""
        bundle_path = tmp_path / "test"
        folio = DataFolio(bundle_path)
        folio.create_snapshot("v1.0")

        runner = CliRunner()
        result = runner.invoke(cli, ["-C", str(bundle_path), "snapshot", "list"])

        assert result.exit_code == 0
        assert "v1.0" in result.output


class TestDescribe:
    """Test bundle describe command."""

    def test_describe_basic(self, tmp_path):
        """Test describing a bundle."""
        folio = DataFolio(tmp_path / "test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        runner = CliRunner()
        result = runner.invoke(cli, ["--bundle", str(tmp_path / "test"), "describe"])

        assert result.exit_code == 0
        # The describe method prints directly, so output should contain bundle info


class TestErrorHandling:
    """Test error handling in CLI."""

    def test_nonexistent_bundle(self, tmp_path):
        """Test error when bundle doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--bundle", str(tmp_path / "nonexistent"), "snapshot", "list"]
        )

        assert result.exit_code == 1

    def test_invalid_snapshot_name(self, tmp_path):
        """Test error for invalid snapshot name."""
        folio = DataFolio(tmp_path / "test")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--bundle",
                str(tmp_path / "test"),
                "snapshot",
                "show",
                "nonexistent",
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
