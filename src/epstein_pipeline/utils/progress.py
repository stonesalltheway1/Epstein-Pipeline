"""Rich progress bar and summary logging utilities."""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


def create_progress() -> Progress:
    """Return a Rich Progress bar with informative columns.

    Usage::

        with create_progress() as progress:
            task = progress.add_task("Processing docs", total=1000)
            for doc in docs:
                process(doc)
                progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def log_summary(processed: int, errors: int, skipped: int) -> None:
    """Print a summary table of pipeline run results.

    Args:
        processed: Number of items successfully processed.
        errors: Number of items that failed with errors.
        skipped: Number of items that were skipped (e.g. duplicates).
    """
    total = processed + errors + skipped

    table = Table(title="Pipeline Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")

    table.add_row("Processed", f"[green]{processed}[/green]")
    table.add_row("Errors", f"[red]{errors}[/red]" if errors else f"{errors}")
    table.add_row("Skipped", f"[yellow]{skipped}[/yellow]" if skipped else f"{skipped}")
    table.add_row("Total", f"[bold]{total}[/bold]")

    console.print()
    console.print(table)
    console.print()
