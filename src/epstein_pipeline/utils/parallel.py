"""Shared parallel processing utilities using concurrent.futures."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import TypeVar

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = 4,
    label: str = "Processing",
    use_processes: bool = False,
) -> list[R]:
    """Execute *fn* over *items* in parallel with a Rich progress bar.

    Parameters
    ----------
    fn:
        A callable that takes a single item and returns a result.
        When *use_processes* is True, *fn* must be picklable (i.e. a
        module-level function, not a lambda or bound method).
    items:
        The items to process.
    max_workers:
        Maximum number of concurrent workers.
    label:
        Description shown in the progress bar.
    use_processes:
        If True, use ``ProcessPoolExecutor`` for CPU-bound work.
        If False (default), use ``ThreadPoolExecutor`` for I/O-bound work.

    Returns
    -------
    list[R]
        Results in completion order (NOT input order).
    """
    items_list = list(items)
    if not items_list:
        return []

    # Clamp workers to item count
    workers = min(max_workers, len(items_list))

    # Fall back to sequential for single worker
    if workers <= 1:
        return _run_sequential(fn, items_list, label)

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    results: list[R] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task(label, total=len(items_list))

        with executor_cls(max_workers=workers) as executor:
            future_to_item: dict[Future[R], T] = {
                executor.submit(fn, item): item for item in items_list
            }

            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    item = future_to_item[future]
                    logger.error("Failed processing %s: %s", item, exc)
                progress.advance(task)

    return results


def _run_sequential(
    fn: Callable[[T], R],
    items: list[T],
    label: str,
) -> list[R]:
    """Sequential fallback with progress bar."""
    results: list[R] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task(label, total=len(items))
        for item in items:
            try:
                results.append(fn(item))
            except Exception as exc:
                logger.error("Failed processing %s: %s", item, exc)
            progress.advance(task)

    return results
