"""Main CLI entry point for DataFolio.

Provides command-line interface for snapshot and bundle management.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union

import click
from rich.console import Console
from rich.table import Table

from datafolio import DataFolio
from datafolio.utils import is_cloud_path

# Global console for Rich output. Disable automatic syntax highlighting so
# status messages render as plain text — otherwise Rich splits tokens like a
# snapshot name "v1.0" with ANSI codes around the highlighted number.
console = Console(highlight=False)


def _alias_path(alias: str) -> str:
    """Look up an alias in the user registry, as a ClickException on failure."""
    from datafolio.folio_registry import FolioRegistry

    try:
        return FolioRegistry().get_alias(alias)
    except KeyError as exc:
        raise click.ClickException(
            f"{exc.args[0]}\nTip: register one with 'datafolio folios alias NAME PATH'."
        ) from None


def resolve_folio_target(
    path: Optional[str] = None, alias: Optional[str] = None
) -> str:
    """Resolve which folio a command targets, as a path string.

    Priority:
    1. Alias (``-a/--alias``), looked up in the user registry
    2. Explicit path (``-f/--folio`` or ``--to``), with ``~`` expanded
    3. ``DATAFOLIO_PATH`` environment variable
    4. Current working directory

    Cloud URIs are returned unchanged; existence is not checked here.

    Args:
        path: Explicit folio path, if given.
        alias: Folio alias, if given.

    Returns:
        Folio path (local path or cloud URI).

    Raises:
        click.UsageError: If both a path and an alias are given.
        click.ClickException: If the alias is not registered.
    """
    if path and alias:
        raise click.UsageError("Pass either a folio path or an alias (-a), not both.")
    if alias:
        return _alias_path(alias)
    if path:
        return path if is_cloud_path(path) else str(Path(path).expanduser())
    env_folio = os.environ.get("DATAFOLIO_PATH")
    if env_folio:
        return (
            env_folio if is_cloud_path(env_folio) else str(Path(env_folio).expanduser())
        )
    return str(Path.cwd())


def find_folio_dir(
    ctx_folio: Optional[str] = None, ctx_alias: Optional[str] = None
) -> Path:
    """Find a local folio directory from multiple sources.

    Priority:
    1. Alias (-a/--alias), looked up in the user registry
    2. Explicit --folio/-f flag
    3. DATAFOLIO_PATH environment variable
    4. Current working directory

    Args:
        ctx_folio: Folio path from CLI context
        ctx_alias: Folio alias from CLI context

    Returns:
        Path to folio directory

    Raises:
        click.ClickException: If folio cannot be found, or resolves to a
            cloud folio (not supported by the commands using this helper)
    """
    target = resolve_folio_target(ctx_folio, ctx_alias)
    if is_cloud_path(target):
        raise click.ClickException(
            f"Cloud folios are not supported by this command: {target}"
        )
    path = Path(target)
    if not path.exists():
        if ctx_alias:
            raise click.ClickException(
                f"Folio not found (from alias '{ctx_alias}'): {target}"
            )
        if ctx_folio:
            raise click.ClickException(
                f"Folio not found: {ctx_folio}{_alias_hint(ctx_folio)}"
            )
        if os.environ.get("DATAFOLIO_PATH"):
            raise click.ClickException(
                f"Folio not found (from DATAFOLIO_PATH): {os.environ['DATAFOLIO_PATH']}"
            )
    return path


def _note_used_folio(path: Union[str, Path]) -> None:
    """Remember the folio this command used, for the recents list.

    Recorded by the root group's result callback only if the command
    finishes successfully.
    """
    ctx = click.get_current_context(silent=True)
    if ctx is not None:
        root = ctx.find_root()
        root.ensure_object(dict)
        root.obj["used_folio"] = str(path)


def _alias_hint(value: Optional[str]) -> str:
    """Suggest ``-a`` when a path argument is actually a registered alias.

    Args:
        value: The path as the user typed it.

    Returns:
        A tip to append to an error message, or ``""``.
    """
    if not value or is_cloud_path(value):
        return ""
    from datafolio.folio_registry import FolioRegistry

    try:
        aliases = FolioRegistry().aliases()
    except Exception:  # noqa: BLE001 - a hint must never mask the real error
        return ""
    if value not in aliases:
        return ""
    return (
        f"\nTip: '{value}' is a registered alias (→ {aliases[value]}). "
        f"Use -a {value}; -f/--to take a path."
    )


def open_existing_folio(target: str, read_only: bool = False) -> DataFolio:
    """Open an existing local or cloud folio, never creating a new one.

    Args:
        target: Folio path or cloud URI.
        read_only: Open read-only.

    Returns:
        The opened DataFolio.

    Raises:
        click.ClickException: If no folio exists at ``target``.
    """
    from datafolio.search import open_existing

    if not is_cloud_path(target):
        try:
            validate_existing_folio(Path(target))
        except click.ClickException as exc:
            hint = _alias_hint(target)
            if hint:
                exc.message = exc.message.split("\nTip:")[0] + hint
            raise
    try:
        folio = open_existing(target, read_only=read_only)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"{exc}\nTip: Use 'datafolio init' to create a new folio."
        ) from None
    _note_used_folio(folio._bundle_dir)
    return folio


def validate_existing_folio(path: Path) -> None:
    """Validate that path is an existing DataFolio bundle.

    Args:
        path: Path to validate

    Raises:
        click.ClickException: If path is not a valid DataFolio bundle
    """
    items_file = path / "items.json"
    metadata_file = path / "metadata.json"

    if not path.exists():
        raise click.ClickException(
            f"Directory does not exist: {path}\n"
            "Tip: Use 'datafolio init' to create a new folio, or use --folio/-f to specify the path."
        )

    if not items_file.exists() and not metadata_file.exists():
        raise click.ClickException(
            f"Not a DataFolio bundle: {path}\n"
            f"Missing required files: items.json and metadata.json\n"
            "Tip: Use 'datafolio init' to create a new folio, or use --folio/-f to specify the correct path."
        )

    _note_used_folio(path.resolve())


def validate_snapshot_name(name: str) -> None:
    """Validate that a snapshot name is safe and valid.

    Args:
        name: Snapshot name to validate

    Raises:
        click.ClickException: If name is invalid
    """
    import re

    # Check for empty name
    if not name or not name.strip():
        raise click.ClickException("Snapshot name cannot be empty")

    # Check for valid characters (alphanumeric, hyphens, underscores, dots)
    if not re.match(r"^[a-zA-Z0-9._-]+$", name):
        raise click.ClickException(
            f"Invalid snapshot name: '{name}'\n"
            "Snapshot names can only contain letters, numbers, hyphens, underscores, and dots."
        )

    # Check for path traversal attempts
    if ".." in name or "/" in name or "\\" in name:
        raise click.ClickException(
            f"Invalid snapshot name: '{name}'\n"
            "Snapshot names cannot contain path separators or '..'."
        )

    # Check length (reasonable limit)
    if len(name) > 100:
        raise click.ClickException(
            f"Snapshot name too long (max 100 characters): '{name}'"
        )


def _get_version():
    """Get DataFolio version."""
    from datafolio import __version__

    return f"DataFolio version {__version__}"


@click.group()
@click.option(
    "--folio",
    "-f",
    type=click.Path(),
    help="Path to DataFolio (default: current directory or DATAFOLIO_PATH env var)",
)
@click.option(
    "--alias",
    "-a",
    help="Alias of a registered DataFolio (see 'datafolio folios list')",
)
@click.version_option(version=None, prog_name="datafolio", message=_get_version())
@click.pass_context
def cli(ctx, folio, alias):
    """DataFolio CLI - Manage data bundles and snapshots.

    Use --folio/-f to specify folio path, -a/--alias for a registered alias,
    or set DATAFOLIO_PATH environment variable.
    """
    ctx.ensure_object(dict)
    if folio and alias:
        raise click.UsageError("Pass either --folio/-f or --alias/-a, not both.")
    ctx.obj["folio"] = folio
    ctx.obj["alias"] = alias


@cli.result_callback()
@click.pass_context
def _record_recent_folio(ctx, result, **kwargs):
    """After a successful command, record the folio it used as recent."""
    used = (ctx.obj or {}).get("used_folio")
    if used:
        from datafolio.folio_registry import FolioRegistry

        FolioRegistry().record_recent(used)
    return result


@cli.group()
@click.pass_context
def snapshot(ctx):
    """Manage snapshots - create, list, compare, delete."""
    pass


# ==================== Snapshot Commands ====================


@snapshot.command("create")
@click.argument("name")
@click.option("--description", "-d", help="Snapshot description")
@click.option(
    "--tag",
    "-t",
    multiple=True,
    help="Tags for the snapshot (can be used multiple times)",
)
@click.option("--no-git", is_flag=True, help="Don't capture git information")
@click.option(
    "--env",
    is_flag=True,
    help="Capture environment information (Python version, packages)",
)
@click.option(
    "--exec", is_flag=True, help="Capture execution context (entry point, working dir)"
)
@click.pass_context
def snapshot_create(ctx, name, description, tag, no_git, env, exec):
    """Create a new snapshot of the current bundle state.

    By default, only git information is captured (with credentials automatically
    removed from remote URLs for security). Environment and execution context
    are opt-in via flags.

    Security: Git remote URLs are automatically sanitized to remove embedded
    credentials (tokens, passwords) before storage.

    Examples:
        # From within folio directory:
        datafolio snapshot create v1.0 -d "Baseline model" -t baseline -t production
        datafolio snapshot create v2.0 --env --exec  # Include environment and execution info

        # From anywhere:
        datafolio --folio /path/to/folio snapshot create v1.0
    """
    try:
        # Validate snapshot name
        validate_snapshot_name(name)

        bundle_path = find_folio_dir(ctx.obj.get("folio"), ctx.obj.get("alias"))
        validate_existing_folio(bundle_path)
        folio = DataFolio(bundle_path)

        tags = list(tag) if tag else None

        folio.create_snapshot(
            name,
            description=description,
            tags=tags,
            capture_git=not no_git,
            capture_environment=env,
            capture_execution=exec,
        )

        console.print(f"[green]✓[/green] Created snapshot '{name}'")

        # Show summary
        info = folio.get_snapshot_info(name)
        console.print(f"  Items: {len(info['item_versions'])}")
        if description:
            console.print(f"  Description: {description}")
        if tags:
            console.print(f"  Tags: {', '.join(tags)}")

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


@snapshot.command("list")
@click.option("--tag", "-t", help="Filter by tag")
@click.pass_context
def snapshot_list(ctx, tag):
    """List all snapshots in the bundle.

    Examples:
        datafolio snapshot list
        datafolio snapshot list --tag baseline
        datafolio --folio /path/to/folio snapshot list
    """
    try:
        bundle_path = find_folio_dir(ctx.obj.get("folio"), ctx.obj.get("alias"))
        validate_existing_folio(bundle_path)
        folio = DataFolio(bundle_path)

        snapshots = folio.list_snapshots()

        # Filter by tag if specified
        if tag:
            snapshots = [s for s in snapshots if tag in (s.get("tags") or [])]

        if not snapshots:
            console.print("[yellow]No snapshots found[/yellow]")
            return

        # Create Rich table
        table = Table(title=f"Snapshots ({len(snapshots)})")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Items", justify="right", style="green")
        table.add_column("Created", style="blue")
        table.add_column("Tags", style="magenta")

        for snap in snapshots:
            from datetime import datetime

            # Format timestamp - convert UTC to local time
            timestamp = snap.get("timestamp", "")
            if timestamp:
                try:
                    # Parse as UTC
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    # Convert to local time
                    local_dt = dt.astimezone()
                    # Format with timezone abbreviation
                    time_str = local_dt.strftime("%Y-%m-%d %H:%M %Z")
                except Exception:
                    time_str = timestamp[:16]
            else:
                time_str = ""

            tags_str = ", ".join(snap.get("tags") or [])
            desc = snap.get("description") or ""

            table.add_row(
                snap["name"],
                (desc[:100] + "...") if (desc and len(desc) > 100) else (desc or ""),
                str(snap.get("num_items", 0)),
                time_str,
                (tags_str[:60] + "...")
                if (tags_str and len(tags_str) > 60)
                else (tags_str or ""),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


@snapshot.command("show")
@click.argument("name")
@click.pass_context
def snapshot_show(ctx, name):
    """Show detailed information about a snapshot.

    Examples:
        datafolio snapshot show v1.0
        datafolio --folio /path/to/folio snapshot show v1.0
    """
    try:
        bundle_path = find_folio_dir(ctx.obj.get("folio"), ctx.obj.get("alias"))
        validate_existing_folio(bundle_path)
        folio = DataFolio(bundle_path)

        info = folio.get_snapshot_info(name)

        console.print(f"\n[bold cyan]Snapshot: {name}[/bold cyan]")
        console.print("=" * 60)

        # Basic info
        if info.get("description"):
            console.print(f"\n[bold]Description:[/bold] {info['description']}")

        if info.get("tags"):
            console.print(f"[bold]Tags:[/bold] {', '.join(info['tags'])}")

        # Format timestamp in local time
        timestamp = info.get("timestamp", "")
        if timestamp:
            from datetime import datetime

            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                local_dt = dt.astimezone()
                timestamp_str = local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                timestamp_str = timestamp
        else:
            timestamp_str = "N/A"
        console.print(f"[bold]Timestamp:[/bold] {timestamp_str}")

        # Items
        item_versions = info.get("item_versions", {})
        console.print(f"\n[bold]Items ({len(item_versions)}):[/bold]")
        for item_name in sorted(item_versions.keys())[:10]:
            console.print(f"  • {item_name}")
        if len(item_versions) > 10:
            console.print(f"  ... and {len(item_versions) - 10} more")

        # Git info
        if "git" in info:
            git = info["git"]
            console.print(f"\n[bold]Git:[/bold]")
            console.print(
                f"  Commit: {git.get('commit_short', git.get('commit', 'N/A')[:7])}"
            )
            console.print(f"  Branch: {git.get('branch', 'N/A')}")
            if git.get("dirty"):
                console.print(f"  Status: [yellow]dirty (uncommitted changes)[/yellow]")

        # Environment
        if "environment" in info:
            env = info["environment"]
            console.print(f"\n[bold]Environment:[/bold]")
            if "python_version" in env:
                console.print(f"  Python: {env['python_version']}")

        # Metadata
        metadata = info.get("metadata_snapshot", {})
        if metadata:
            # Filter internal fields
            user_metadata = {
                k: v
                for k, v in metadata.items()
                if k not in ("created_at", "updated_at", "_datafolio")
            }
            if user_metadata:
                console.print(f"\n[bold]Metadata ({len(user_metadata)} fields):[/bold]")
                for key, value in list(user_metadata.items())[:5]:
                    console.print(f"  • {key}: {value}")
                if len(user_metadata) > 5:
                    console.print(f"  ... and {len(user_metadata) - 5} more fields")

        console.print()

    except KeyError:
        console.print(f"[red]✗[/red] Snapshot '{name}' not found", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


@snapshot.command("compare")
@click.argument("snapshot1")
@click.argument("snapshot2")
@click.pass_context
def snapshot_compare(ctx, snapshot1, snapshot2):
    """Compare two snapshots.

    Examples:
        datafolio snapshot compare v1.0 v2.0
        datafolio --folio /path/to/folio snapshot compare v1.0 v2.0
    """
    try:
        bundle_path = find_folio_dir(ctx.obj.get("folio"), ctx.obj.get("alias"))
        validate_existing_folio(bundle_path)
        folio = DataFolio(bundle_path)

        diff = folio.compare_snapshots(snapshot1, snapshot2)

        console.print(f"\n[bold]Comparing {snapshot1} → {snapshot2}[/bold]")
        console.print("=" * 60)

        # Added items
        if diff["added_items"]:
            console.print(f"\n[green]Added ({len(diff['added_items'])}):[/green]")
            for item in diff["added_items"]:
                console.print(f"  + {item}")

        # Removed items
        if diff["removed_items"]:
            console.print(f"\n[red]Removed ({len(diff['removed_items'])}):[/red]")
            for item in diff["removed_items"]:
                console.print(f"  - {item}")

        # Modified items
        if diff["modified_items"]:
            console.print(
                f"\n[yellow]Modified ({len(diff['modified_items'])}):[/yellow]"
            )
            for item in diff["modified_items"]:
                console.print(f"  ~ {item}")

        # Unchanged items
        if diff["shared_items"]:
            console.print(f"\n[blue]Unchanged ({len(diff['shared_items'])}):[/blue]")
            for item in diff["shared_items"][:5]:
                console.print(f"  = {item}")
            if len(diff["shared_items"]) > 5:
                console.print(f"  ... and {len(diff['shared_items']) - 5} more")

        # Metadata changes
        if diff["metadata_changes"]:
            console.print(
                f"\n[cyan]Metadata Changes ({len(diff['metadata_changes'])}):[/cyan]"
            )
            for key, (old, new) in list(diff["metadata_changes"].items())[:5]:
                console.print(f"  {key}: {old} → {new}")
            if len(diff["metadata_changes"]) > 5:
                console.print(f"  ... and {len(diff['metadata_changes']) - 5} more")

        # Summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Added: {len(diff['added_items'])}")
        console.print(f"  Removed: {len(diff['removed_items'])}")
        console.print(f"  Modified: {len(diff['modified_items'])}")
        console.print(f"  Unchanged: {len(diff['shared_items'])}")
        console.print()

    except KeyError as e:
        console.print(f"[red]✗[/red] {e}", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


@snapshot.command("delete")
@click.argument("name")
@click.option(
    "--cleanup/--no-cleanup",
    default=False,
    help="Cleanup orphaned versions after deletion",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def snapshot_delete(ctx, name, cleanup, yes):
    """Delete a snapshot.

    Examples:
        datafolio snapshot delete v0.1 --cleanup
        datafolio snapshot delete experimental-v5 -y
        datafolio --folio /path/to/folio snapshot delete v1.0 -y
    """
    try:
        bundle_path = find_folio_dir(ctx.obj.get("folio"), ctx.obj.get("alias"))
        validate_existing_folio(bundle_path)
        folio = DataFolio(bundle_path)

        # Confirmation
        if not yes:
            if not click.confirm(f"Delete snapshot '{name}'?"):
                console.print("[yellow]Cancelled[/yellow]")
                return

        folio.delete_snapshot(name, cleanup_orphans=cleanup)

        console.print(f"[green]✓[/green] Deleted snapshot '{name}'")

        if cleanup:
            console.print("  Cleaned up orphaned versions")

    except KeyError:
        console.print(f"[red]✗[/red] Snapshot '{name}' not found", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


@snapshot.command("gc")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be deleted without deleting"
)
@click.pass_context
def snapshot_gc(ctx, dry_run):
    """Cleanup orphaned item versions (garbage collection).

    Example:
        datafolio snapshot gc --dry-run
        datafolio snapshot gc
    """
    try:
        bundle_path = find_folio_dir(ctx.obj.get("folio"), ctx.obj.get("alias"))
        validate_existing_folio(bundle_path)
        folio = DataFolio(bundle_path)

        deleted = folio.cleanup_orphaned_versions(dry_run=dry_run)

        if dry_run:
            if deleted:
                console.print(
                    f"[yellow]Would delete {len(deleted)} orphaned version(s):[/yellow]"
                )
                for filename in deleted[:10]:
                    console.print(f"  • {filename}")
                if len(deleted) > 10:
                    console.print(f"  ... and {len(deleted) - 10} more")
            else:
                console.print("[green]No orphaned versions found[/green]")
        else:
            if deleted:
                console.print(
                    f"[green]✓[/green] Deleted {len(deleted)} orphaned version(s)"
                )
            else:
                console.print("[green]No orphaned versions found[/green]")

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


@snapshot.command("status")
@click.pass_context
def snapshot_status(ctx):
    """Show current bundle state compared to last snapshot.

    Similar to 'git status', shows what has changed since the last snapshot.

    Example:
        datafolio snapshot status
    """
    try:
        bundle_path = find_folio_dir(ctx.obj.get("folio"), ctx.obj.get("alias"))
        validate_existing_folio(bundle_path)
        folio = DataFolio(bundle_path)

        console.print(f"\n[bold]Current bundle:[/bold] {bundle_path}")

        # Check if any snapshots exist
        snapshots = folio.list_snapshots()
        if not snapshots:
            console.print("[yellow]No snapshots yet[/yellow]")
            console.print("\n[cyan]Create your first snapshot:[/cyan]")
            console.print("  datafolio snapshot create v1.0 -d 'Initial snapshot'")
            return

        # Get last snapshot (list_snapshots() sorts newest-first)
        last_snapshot = snapshots[0]

        # Format timestamp
        timestamp = last_snapshot.get("timestamp", "")
        if timestamp:
            from datetime import datetime

            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                local_dt = dt.astimezone()
                date_str = local_dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = timestamp[:10]
        else:
            date_str = "unknown"

        console.print(
            f"[bold]Last snapshot:[/bold] {last_snapshot['name']} ({date_str})"
        )

        # Get diff from last snapshot
        diff = folio.diff_from_snapshot()

        # Show changes
        has_changes = (
            diff["added_items"]
            or diff["removed_items"]
            or diff["modified_items"]
            or diff["metadata_changes"]
        )

        if not has_changes:
            console.print("\n[green]✓ No changes since last snapshot[/green]")
        else:
            console.print("\n[bold]Changes since last snapshot:[/bold]")

            # Added items
            if diff["added_items"]:
                console.print(f"\n[green]Added ({len(diff['added_items'])}):[/green]")
                for item in diff["added_items"]:
                    console.print(f"  [green]+[/green] {item}")

            # Removed items
            if diff["removed_items"]:
                console.print(f"\n[red]Removed ({len(diff['removed_items'])}):[/red]")
                for item in diff["removed_items"]:
                    console.print(f"  [red]-[/red] {item}")

            # Modified items
            if diff["modified_items"]:
                console.print(
                    f"\n[yellow]Modified ({len(diff['modified_items'])}):[/yellow]"
                )
                for item in diff["modified_items"]:
                    console.print(f"  [yellow]~[/yellow] {item}")

            # Metadata changes
            if diff["metadata_changes"]:
                console.print(
                    f"\n[cyan]Metadata changes ({len(diff['metadata_changes'])}):[/cyan]"
                )
                for key, (old, new) in list(diff["metadata_changes"].items())[:5]:
                    console.print(f"  {key}: {old} → {new}")
                if len(diff["metadata_changes"]) > 5:
                    console.print(f"  ... and {len(diff['metadata_changes']) - 5} more")

            # Unchanged items
            if diff["unchanged_items"]:
                console.print(
                    f"\n[dim]Unchanged: {len(diff['unchanged_items'])} items[/dim]"
                )

        console.print()

    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


@snapshot.command("diff")
@click.argument("snapshot", required=False)
@click.pass_context
def snapshot_diff(ctx, snapshot):
    """Show changes between current state and a snapshot.

    If no snapshot is specified, compares to the last snapshot.

    Example:
        datafolio snapshot diff           # Compare to last snapshot
        datafolio snapshot diff v1.0      # Compare to specific snapshot
    """
    try:
        bundle_path = find_folio_dir(ctx.obj.get("folio"), ctx.obj.get("alias"))
        validate_existing_folio(bundle_path)
        folio = DataFolio(bundle_path)

        # Get diff
        diff = folio.diff_from_snapshot(snapshot)

        snapshot_name = diff["snapshot_name"]
        console.print(
            f"\n[bold]Comparing current state to snapshot '{snapshot_name}'[/bold]"
        )
        console.print("=" * 60)

        # Check for changes
        has_changes = (
            diff["added_items"]
            or diff["removed_items"]
            or diff["modified_items"]
            or diff["metadata_changes"]
        )

        if not has_changes:
            console.print("\n[green]✓ No changes[/green]")
        else:
            # Added items
            if diff["added_items"]:
                console.print(f"\n[green]Added ({len(diff['added_items'])}):[/green]")
                for item in diff["added_items"]:
                    console.print(f"  [green]+[/green] {item}")

            # Removed items
            if diff["removed_items"]:
                console.print(f"\n[red]Removed ({len(diff['removed_items'])}):[/red]")
                for item in diff["removed_items"]:
                    console.print(f"  [red]-[/red] {item}")

            # Modified items
            if diff["modified_items"]:
                console.print(
                    f"\n[yellow]Modified ({len(diff['modified_items'])}):[/yellow]"
                )
                for item in diff["modified_items"]:
                    console.print(f"  [yellow]~[/yellow] {item}")

            # Unchanged items
            if diff["unchanged_items"]:
                console.print(
                    f"\n[blue]Unchanged ({len(diff['unchanged_items'])}):[/blue]"
                )
                for item in diff["unchanged_items"][:5]:
                    console.print(f"  [blue]=[/blue] {item}")
                if len(diff["unchanged_items"]) > 5:
                    console.print(f"  ... and {len(diff['unchanged_items']) - 5} more")

            # Metadata changes
            if diff["metadata_changes"]:
                console.print(
                    f"\n[cyan]Metadata changes ({len(diff['metadata_changes'])}):[/cyan]"
                )
                for key, (old, new) in list(diff["metadata_changes"].items())[:5]:
                    console.print(f"  {key}: {old} → {new}")
                if len(diff["metadata_changes"]) > 5:
                    console.print(f"  ... and {len(diff['metadata_changes']) - 5} more")

        # Summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Added: {len(diff['added_items'])}")
        console.print(f"  Removed: {len(diff['removed_items'])}")
        console.print(f"  Modified: {len(diff['modified_items'])}")
        console.print(f"  Unchanged: {len(diff['unchanged_items'])}")
        console.print()

    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        sys.exit(1)
    except KeyError as e:
        console.print(f"[red]✗[/red] {e}", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


# ==================== Bundle Commands ====================


@cli.command("validate")
@click.argument("path", type=click.Path(), required=False)
@click.pass_context
def validate(ctx, path):
    """Validate that a directory is a valid DataFolio bundle.

    If no path is provided, validates the current directory or DATAFOLIO_PATH.

    Examples:
        datafolio validate
        datafolio validate /path/to/folio
        datafolio --folio /path/to/folio validate
    """
    try:
        # Determine which path to validate
        if path:
            folio_path = Path(path)
        else:
            folio_path = find_folio_dir(ctx.obj.get("folio"), ctx.obj.get("alias"))

        # Validate the path structure
        validate_existing_folio(folio_path)

        # Run the real integrity validation: payload existence (and, for
        # local owned items, checksum agreement) for every item.
        folio = DataFolio(folio_path, read_only=True)
        results = folio.validate()
        failures = sorted(name for name, ok in results.items() if not ok)

        contents = folio.list_contents(include_archived=True)
        num_items = sum(len(items) for items in contents.values())
        num_snapshots = len(folio.list_snapshots())

        if failures:
            console.print(f"[red]✗[/red] Invalid DataFolio bundle: {folio_path}")
            console.print(f"  Items: {num_items} ({len(failures)} failing)")
            for name in failures:
                console.print(f"  [red]✗[/red] {name}: missing or corrupt payload")
            sys.exit(1)

        console.print(f"[green]✓[/green] Valid DataFolio bundle: {folio_path}")
        console.print(f"  Items: {num_items} (all payloads verified)")
        console.print(f"  Snapshots: {num_snapshots}")

        if folio.metadata.get("description"):
            console.print(f"  Description: {folio.metadata['description']}")

    except click.ClickException as e:
        # Re-raise Click exceptions (from validate_existing_folio)
        raise
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


@cli.command("describe")
@click.option("--max-metadata", default=10, help="Maximum metadata fields to show")
@click.option(
    "--snapshot", "-s", help="Describe a specific snapshot instead of the full bundle"
)
@click.pass_context
def describe(ctx, max_metadata, snapshot):
    """Show detailed bundle description.

    Examples:
        datafolio describe
        datafolio --folio /path/to/folio describe
        datafolio describe --snapshot v1.0
    """
    try:
        bundle_path = find_folio_dir(ctx.obj.get("folio"), ctx.obj.get("alias"))
        validate_existing_folio(bundle_path)
        folio = DataFolio(bundle_path)

        # Use folio's describe method which prints to stdout
        folio.describe(
            return_string=False, max_metadata_fields=max_metadata, snapshot=snapshot
        )

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


@cli.command("init")
@click.argument("path", type=click.Path(), required=False)
@click.option("--description", "-d", help="Bundle description")
@click.option("--name", "-n", help="Bundle name (default: directory name)")
@click.option(
    "--alias",
    "alias_name",
    help="Register the new folio under this alias in the user registry",
)
@click.pass_context
def init(ctx, path, description, name, alias_name):
    """Initialize a new DataFolio bundle.

    If no path is provided, initializes in the current directory.

    Examples:
        datafolio init experiments/new-exp -d "My experiment"
        datafolio init -d "Current directory experiment"
        datafolio init gs://my-bucket/experiment -d "Cloud experiment"
    """
    try:
        from pathlib import Path

        from datafolio.utils import is_cloud_path

        # Use provided path or current directory. Cloud URIs must stay
        # strings: Path.resolve() would mangle 'gs://bucket/x' into a local
        # './gs:/bucket/x' path.
        is_cloud = bool(path) and is_cloud_path(str(path))
        if is_cloud:
            bundle_path = str(path).rstrip("/")
        elif path:
            bundle_path = Path(path).expanduser().resolve()
        else:
            bundle_path = Path.cwd()

        # Check if bundle already exists (local only; for cloud paths
        # DataFolio detects and opens an existing bundle itself)
        allow_existing = False
        if not is_cloud and (bundle_path / "items.json").exists():
            console.print(
                f"[yellow]⚠ Warning:[/yellow] Bundle already exists at {bundle_path}"
            )
            if not click.confirm("Reinitialize (this won't delete existing data)?"):
                console.print("[yellow]Cancelled[/yellow]")
                return
        elif (
            not is_cloud
            and Path(bundle_path).is_dir()
            and any(Path(bundle_path).iterdir())
        ):
            # An existing NON-folio directory with files in it (including the
            # documented no-argument case: the current directory). Existing
            # files are left alone; init only adds the folio manifests.
            console.print(f"[yellow]⚠[/yellow] {bundle_path} exists and is not empty.")
            if not click.confirm("Initialize a folio in this directory?"):
                console.print("[yellow]Cancelled[/yellow]")
                return
            allow_existing = True

        # Initialize bundle
        console.print(f"[dim]Initializing bundle in:[/dim] {bundle_path}")

        # Determine bundle name
        if name is None:
            name = str(bundle_path).rstrip("/").rsplit("/", 1)[-1]

        # Create the bundle (DataFolio will create the directory)
        folio = DataFolio(
            bundle_path,
            metadata={"description": description} if description else None,
            allow_existing=allow_existing,
        )

        console.print(
            f"\n[green]✓[/green] Initialized DataFolio bundle: [cyan]{name}[/cyan]"
        )
        console.print(f"  Path: {bundle_path}")
        if description:
            console.print(f"  Description: {description}")
        if alias_name:
            folio.set_alias(alias_name)
            console.print(f"  Alias: {alias_name}")
        _note_used_folio(folio._bundle_dir)

        target = f"-a {alias_name}" if alias_name else f"--to {bundle_path}"
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  # Add data to your bundle")
        console.print(f"  datafolio add {target} my_table.parquet -d 'What it is'")
        console.print("\n  # Create a snapshot when ready")
        prefix = f"datafolio -a {alias_name}" if alias_name else "datafolio"
        console.print(f"  {prefix} snapshot create v1.0 -d 'Initial version'")

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


# ==================== Adding Data ====================

# Extensions `datafolio add` imports as tables (converted to parquet).
TABLE_EXTENSIONS = {".parquet", ".pq", ".csv", ".feather", ".arrow", ".ipc"}
# Table formats reference_table() can link without copying.
REFERENCE_FORMATS = {".parquet": "parquet", ".pq": "parquet", ".csv": "csv"}


@cli.command("add")
@click.argument(
    "file_path", metavar="FILE", type=click.Path(exists=True, dir_okay=False)
)
@click.option("--to", "to", help="Path of the folio to add to (local or cloud)")
@click.option(
    "--alias", "-a", "alias", help="Alias of the folio to add to (instead of --to)"
)
@click.option("--name", "-n", help="Item name (default: file name without extension)")
@click.option("--description", "-d", help="Item description")
@click.option(
    "--reference",
    is_flag=True,
    help="Link a parquet/csv file without copying it (referenced table)",
)
@click.option(
    "--as-file",
    is_flag=True,
    help="Store the file unchanged as an artifact (no conversion)",
)
@click.option("--overwrite", is_flag=True, help="Replace an existing item")
@click.pass_context
def add(ctx, file_path, to, alias, name, description, reference, as_file, overwrite):
    """Add FILE to a folio.

    \b
    What gets stored depends on the file:
      .parquet .csv .feather .arrow   table, stored as parquet (streamed,
                                      so files larger than memory work)
      .npy                            numpy array
      .json                           JSON data
      anything else                   the file, unchanged (artifact)

    Use --as-file to store any file unchanged, or --reference to link a
    parquet/csv file in place without copying it.

    The folio is chosen by --to/-a here, else by the global -f/-a,
    DATAFOLIO_PATH, or the current directory.

    \b
    Examples:
        datafolio add --to ~/analysis/exp ~/Downloads/blah.parquet
        datafolio add -a my-folio synapses.parquet -d "Synapse table"
        datafolio add -a my-folio cells.csv --name cells
        datafolio add -a my-folio notes.csv --as-file
        datafolio add -a my-folio /data/huge.parquet --reference
    """
    if reference and as_file:
        raise click.UsageError("--reference and --as-file cannot be combined.")
    try:
        if to or alias:
            target = resolve_folio_target(to, alias)
        else:
            target = resolve_folio_target(ctx.obj.get("folio"), ctx.obj.get("alias"))
        folio = open_existing_folio(target)

        src = Path(file_path).expanduser().resolve()
        ext = src.suffix.lower()
        if name is None:
            name = src.stem
            try:
                from datafolio.utils import validate_item_name

                validate_item_name(name)
            except ValueError as exc:
                raise click.ClickException(
                    f"{exc}. The name was derived from the file name; "
                    f"pass --name to choose a valid one."
                ) from None

        with console.status(f"Adding {src.name}..."):
            if reference:
                if ext not in REFERENCE_FORMATS:
                    raise click.ClickException(
                        f"--reference only supports parquet and csv files, "
                        f"not '{ext or src.name}'. Drop --reference to import "
                        f"the table, or use --as-file."
                    )
                folio.reference_table(
                    name,
                    str(src),
                    table_format=REFERENCE_FORMATS[ext],
                    description=description,
                    overwrite=overwrite,
                )
            elif as_file:
                folio.add_file(src, name, description=description, overwrite=overwrite)
            elif ext in TABLE_EXTENSIONS:
                folio.import_table(
                    name, src, description=description, overwrite=overwrite
                )
            elif ext == ".npy":
                import numpy as np

                folio.add(
                    name,
                    np.load(src, mmap_mode="r"),
                    description=description,
                    overwrite=overwrite,
                )
            elif ext == ".json":
                import orjson

                folio.add(
                    name,
                    orjson.loads(src.read_bytes()),
                    description=description,
                    overwrite=overwrite,
                )
            else:
                folio.add_file(src, name, description=description, overwrite=overwrite)

        item_type = folio.item_info(name)["item_type"]
        where = f"'{alias}'" if alias else folio._bundle_dir
        console.print(
            f"[green]✓[/green] Added [cyan]{name}[/cyan] ({item_type}) to {where}"
        )

    except click.ClickException:
        raise
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


# ==================== Folio Registry ====================


@cli.group()
def folios():
    """Manage folio aliases and the recently used list (~/.datafolio).

    Set DATAFOLIO_HOME to keep the registry somewhere else.
    """


@folios.command("list")
def folios_list():
    """List folio aliases and recently used folios.

    Examples:
        datafolio folios list
    """
    from datafolio.folio_registry import list_folios

    df = list_folios()
    if df.empty:
        console.print("[yellow]No folios registered yet.[/yellow]")
        console.print(
            "Register one with 'datafolio folios alias NAME PATH'; "
            "folios used from the CLI are also remembered here."
        )
        return

    table = Table(title="Folios")
    table.add_column("Alias", style="cyan")
    table.add_column("Path")
    table.add_column("Last used (CLI)", style="dim")
    table.add_column("Exists")
    for row in df.itertuples(index=False):
        exists = "?" if row.exists is None else ("✓" if row.exists else "[red]✗[/red]")
        table.add_row(
            row.alias or "",
            row.path,
            (row.last_accessed or "")[:19].replace("T", " "),
            exists,
        )
    console.print(table)


@folios.command("alias")
@click.argument("alias_name")
@click.argument("path", required=False)
@click.option("--overwrite", is_flag=True, help="Rebind an alias that already exists")
@click.pass_context
def folios_alias(ctx, alias_name, path, overwrite):
    """Register ALIAS_NAME for the folio at PATH.

    PATH defaults to the current folio (-f, DATAFOLIO_PATH, or the current
    directory). Cloud URIs are accepted.

    Examples:
        datafolio folios alias my-folio ~/analysis/my-folio
        datafolio folios alias shared gs://bucket/folios/shared
        datafolio -f ~/analysis/my-folio folios alias my-folio
    """
    from datafolio.folio_registry import FolioRegistry

    target = path or resolve_folio_target(ctx.obj.get("folio"), ctx.obj.get("alias"))
    if not is_cloud_path(target):
        validate_existing_folio(Path(target).expanduser())
    try:
        resolved = FolioRegistry().set_alias(alias_name, target, overwrite=overwrite)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    console.print(f"[green]✓[/green] {alias_name} → {resolved}")


@folios.command("unalias")
@click.argument("alias_name")
def folios_unalias(alias_name):
    """Remove an alias (the folio itself is untouched).

    Examples:
        datafolio folios unalias my-folio
    """
    from datafolio.folio_registry import FolioRegistry

    try:
        FolioRegistry().remove_alias(alias_name)
    except KeyError as exc:
        raise click.ClickException(exc.args[0]) from None
    console.print(f"[green]✓[/green] Removed alias {alias_name}")


@folios.command("forget")
@click.argument("path")
def folios_forget(path):
    """Remove PATH from the recently used list (aliases are kept).

    Examples:
        datafolio folios forget ~/scratch/tmp-folio
    """
    from datafolio.folio_registry import FolioRegistry

    if FolioRegistry().forget(path):
        console.print(f"[green]✓[/green] Forgot {path}")
    else:
        raise click.ClickException(f"Not in the recent list: {path}")


@folios.command("prune")
def folios_prune():
    """Drop aliases and recents whose local folio no longer exists.

    Cloud folios are never pruned.

    Examples:
        datafolio folios prune
    """
    from datafolio.folio_registry import FolioRegistry

    removed = FolioRegistry().prune()
    if not removed:
        console.print("Nothing to prune.")
        return
    for p in removed:
        console.print(f"  [red]✗[/red] {p}")
    console.print(f"[green]✓[/green] Pruned {len(removed)} missing folio(s)")


# ==================== Search ====================


@cli.command("find")
@click.argument("pattern", default="*")
@click.option("--regex", is_flag=True, help="Treat PATTERN as a regular expression")
@click.option(
    "--type",
    "item_types",
    multiple=True,
    type=click.Choice(["table", "model", "artifact", "array", "json", "timestamp"]),
    help="Only items of this type (repeatable)",
)
@click.option(
    "--metadata",
    is_flag=True,
    help="Match folio metadata keys (KEY or KEY=VALUE) instead of item names",
)
@click.option(
    "--in",
    "in_folios",
    multiple=True,
    help="Only search this alias or path (repeatable)",
)
@click.option(
    "--aliases-only", is_flag=True, help="Skip recently used, unaliased folios"
)
@click.option("--local-only", is_flag=True, help="Skip cloud folios")
@click.option("--include-archived", is_flag=True, help="Include archived items")
@click.option("--case-sensitive", is_flag=True, help="Match case-sensitively")
@click.option(
    "--desc",
    "descriptions",
    is_flag=True,
    help="Also match text anywhere in item descriptions",
)
def find_cmd(
    pattern,
    regex,
    item_types,
    metadata,
    in_folios,
    aliases_only,
    local_only,
    include_archived,
    case_sensitive,
    descriptions,
):
    """Find items across registered and recently used folios.

    PATTERN is a glob matched against item names (default: '*'). Only each
    folio's manifest is read, so this is fast. Exits 1 if nothing matches.

    \b
    Examples:
        datafolio find 'cells*'
        datafolio find 'synapse|soma' --regex --type table
        datafolio find '*' --in my-folio --type model
        datafolio find 'dataset=minnie*' --metadata
        datafolio find synapse --desc
    """
    import warnings

    from datafolio.search import find

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            df = find(
                pattern,
                regex=regex,
                item_type=list(item_types) or None,
                metadata=metadata,
                folios=list(in_folios) or None,
                aliases_only=aliases_only,
                local_only=local_only,
                include_archived=include_archived,
                case_sensitive=case_sensitive,
                descriptions=descriptions,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from None
    for w in caught:
        if issubclass(w.category, UserWarning):
            console.print(f"[yellow]⚠[/yellow] {w.message}")

    if df.empty:
        console.print(f"[yellow]No matches for '{pattern}'.[/yellow]")
        if not descriptions and not metadata:
            # The manifests are local and small; a second pass is cheap and
            # only happens when nothing matched.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                in_desc = find(
                    pattern,
                    regex=regex,
                    item_type=list(item_types) or None,
                    folios=list(in_folios) or None,
                    aliases_only=aliases_only,
                    local_only=local_only,
                    include_archived=include_archived,
                    case_sensitive=case_sensitive,
                    descriptions=True,
                )
            if not in_desc.empty:
                n = len(in_desc)
                console.print(
                    f"{n} item{'s' if n != 1 else ''} mention{'' if n != 1 else 's'} "
                    f"it in {'their descriptions' if n != 1 else 'its description'}; "
                    f"add --desc to include {'them' if n != 1 else 'it'}."
                )
        sys.exit(1)

    table = Table(title=f"Matches for '{pattern}'")
    table.add_column("Folio", style="cyan")
    if metadata:
        table.add_column("Key")
        table.add_column("Value")
    else:
        table.add_column("Item")
        table.add_column("Type", style="dim")
        table.add_column("Description")
    for row in df.itertuples(index=False):
        folio_label = row.alias or row.folio_path
        if metadata:
            table.add_row(folio_label, str(row.key), str(row.value))
        else:
            table.add_row(folio_label, row.name, row.item_type, row.description or "")
    console.print(table)


if __name__ == "__main__":
    cli()
