"""Main CLI entry point for DataFolio.

Provides command-line interface for snapshot and bundle management.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from datafolio import DataFolio

# Global console for Rich output
console = Console()


def find_bundle_dir(ctx_bundle: Optional[str] = None) -> Path:
    """Find bundle directory from multiple sources.

    Priority:
    1. Explicit --bundle/-C flag
    2. DATAFOLIO_BUNDLE environment variable
    3. Current working directory

    Args:
        ctx_bundle: Bundle path from CLI context

    Returns:
        Path to bundle directory

    Raises:
        click.ClickException: If bundle cannot be found
    """
    # Check explicit flag
    if ctx_bundle:
        path = Path(ctx_bundle)
        if path.exists():
            return path
        raise click.ClickException(f"Bundle not found: {ctx_bundle}")

    # Check environment variable
    env_bundle = os.environ.get("DATAFOLIO_BUNDLE")
    if env_bundle:
        path = Path(env_bundle)
        if path.exists():
            return path
        raise click.ClickException(
            f"Bundle not found (from DATAFOLIO_BUNDLE): {env_bundle}"
        )

    # Use current directory
    return Path.cwd()


@click.group()
@click.option(
    "--bundle",
    "-C",
    type=click.Path(),
    help="Path to DataFolio bundle (default: current directory or DATAFOLIO_BUNDLE env var)",
)
@click.pass_context
def cli(ctx, bundle):
    """DataFolio CLI - Manage data bundles and snapshots.

    Use --bundle/-C to specify bundle path, or set DATAFOLIO_BUNDLE environment variable.
    """
    ctx.ensure_object(dict)
    ctx.obj["bundle"] = bundle


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

    Example:
        datafolio snapshot create v1.0 -d "Baseline model" -t baseline -t production
        datafolio snapshot create v2.0 --env --exec  # Include environment and execution info
    """
    try:
        bundle_path = find_bundle_dir(ctx.obj.get("bundle"))
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

    Example:
        datafolio snapshot list
        datafolio snapshot list --tag baseline
    """
    try:
        bundle_path = find_bundle_dir(ctx.obj.get("bundle"))
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

            # Format timestamp
            timestamp = snap.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    time_str = timestamp[:16]
            else:
                time_str = ""

            tags_str = ", ".join(snap.get("tags") or [])
            desc = snap.get("description") or ""

            table.add_row(
                snap["name"],
                (desc[:50] + "...") if (desc and len(desc) > 50) else (desc or ""),
                str(snap.get("num_items", 0)),
                time_str,
                (tags_str[:30] + "...")
                if (tags_str and len(tags_str) > 30)
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

    Example:
        datafolio snapshot show v1.0
    """
    try:
        bundle_path = find_bundle_dir(ctx.obj.get("bundle"))
        folio = DataFolio(bundle_path)

        info = folio.get_snapshot_info(name)

        console.print(f"\n[bold cyan]Snapshot: {name}[/bold cyan]")
        console.print("=" * 60)

        # Basic info
        if info.get("description"):
            console.print(f"\n[bold]Description:[/bold] {info['description']}")

        if info.get("tags"):
            console.print(f"[bold]Tags:[/bold] {', '.join(info['tags'])}")

        console.print(f"[bold]Timestamp:[/bold] {info.get('timestamp', 'N/A')}")

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

    Example:
        datafolio snapshot compare v1.0 v2.0
    """
    try:
        bundle_path = find_bundle_dir(ctx.obj.get("bundle"))
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

    Example:
        datafolio snapshot delete v0.1 --cleanup
        datafolio snapshot delete experimental-v5 -y
    """
    try:
        bundle_path = find_bundle_dir(ctx.obj.get("bundle"))
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
        bundle_path = find_bundle_dir(ctx.obj.get("bundle"))
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


@snapshot.command("reproduce")
@click.argument("name")
@click.pass_context
def snapshot_reproduce(ctx, name):
    """Show reproduction instructions for a snapshot.

    Example:
        datafolio snapshot reproduce v1.0
    """
    try:
        bundle_path = find_bundle_dir(ctx.obj.get("bundle"))
        folio = DataFolio(bundle_path)

        instructions = folio.reproduce_instructions(name)

        console.print(f"\n[bold cyan]{instructions}[/bold cyan]")

    except KeyError:
        console.print(f"[red]✗[/red] Snapshot '{name}' not found", style="red")
        sys.exit(1)
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
        bundle_path = find_bundle_dir(ctx.obj.get("bundle"))
        folio = DataFolio(bundle_path)

        console.print(f"\n[bold]Current bundle:[/bold] {bundle_path}")

        # Check if any snapshots exist
        snapshots = folio.list_snapshots()
        if not snapshots:
            console.print("[yellow]No snapshots yet[/yellow]")
            console.print("\n[cyan]Create your first snapshot:[/cyan]")
            console.print("  datafolio snapshot create v1.0 -d 'Initial snapshot'")
            return

        # Get last snapshot
        last_snapshot = snapshots[-1]
        console.print(
            f"[bold]Last snapshot:[/bold] {last_snapshot['name']} "
            f"({last_snapshot['timestamp'][:10]})"
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
        bundle_path = find_bundle_dir(ctx.obj.get("bundle"))
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


@cli.command("describe")
@click.option("--max-metadata", default=10, help="Maximum metadata fields to show")
@click.pass_context
def describe(ctx, max_metadata):
    """Show detailed bundle description.

    Example:
        datafolio describe
    """
    try:
        bundle_path = find_bundle_dir(ctx.obj.get("bundle"))
        folio = DataFolio(bundle_path)

        # Use folio's describe method which prints to stdout
        folio.describe(return_string=False, max_metadata_fields=max_metadata)

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


@cli.command("init")
@click.argument("path", type=click.Path(), required=False)
@click.option("--description", "-d", help="Bundle description")
@click.option("--name", "-n", help="Bundle name (default: directory name)")
@click.pass_context
def init(ctx, path, description, name):
    """Initialize a new DataFolio bundle.

    If no path is provided, initializes in the current directory.

    Example:
        datafolio init experiments/new-exp -d "My experiment"
        datafolio init -d "Current directory experiment"
    """
    try:
        from pathlib import Path

        # Use provided path or current directory
        if path:
            bundle_path = Path(path).resolve()
        else:
            bundle_path = Path.cwd()

        # Check if bundle already exists
        if (bundle_path / "items.json").exists():
            console.print(
                f"[yellow]⚠ Warning:[/yellow] Bundle already exists at {bundle_path}"
            )
            if not click.confirm("Reinitialize (this won't delete existing data)?"):
                console.print("[yellow]Cancelled[/yellow]")
                return

        # Initialize bundle
        console.print(f"[dim]Initializing bundle in:[/dim] {bundle_path}")

        # Determine bundle name
        if name is None:
            name = bundle_path.name

        # Create the bundle (DataFolio will create the directory)
        folio = DataFolio(
            bundle_path, metadata={"description": description} if description else None
        )

        console.print(
            f"\n[green]✓[/green] Initialized DataFolio bundle: [cyan]{name}[/cyan]"
        )
        console.print(f"  Path: {bundle_path}")
        if description:
            console.print(f"  Description: {description}")

        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  # Add data to your bundle")
        console.print("  cd", bundle_path)
        console.print(
            "  python -c \"from datafolio import DataFolio; folio = DataFolio('.'); ...\""
        )
        console.print("\n  # Create a snapshot when ready")
        console.print("  datafolio snapshot create v1.0 -d 'Initial version'")

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    cli()
