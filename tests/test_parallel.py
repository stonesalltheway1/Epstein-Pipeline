"""Tests for parallel processing utilities."""

import time

from epstein_pipeline.utils.parallel import run_parallel


def _slow_fn(x: int) -> int:
    """A simple function to test parallel execution."""
    time.sleep(0.01)
    return x * 2


def _failing_fn(x: int) -> int:
    """A function that fails on even inputs."""
    if x % 2 == 0:
        raise ValueError(f"Cannot process {x}")
    return x * 2


def test_run_parallel_basic():
    items = list(range(10))
    results = run_parallel(_slow_fn, items, max_workers=4, label="Test")
    assert len(results) == 10
    assert set(results) == {i * 2 for i in range(10)}


def test_run_parallel_single_worker():
    items = list(range(5))
    results = run_parallel(_slow_fn, items, max_workers=1, label="Sequential")
    assert len(results) == 5


def test_run_parallel_empty():
    results = run_parallel(_slow_fn, [], max_workers=4, label="Empty")
    assert results == []


def test_run_parallel_with_processes():
    items = list(range(5))
    results = run_parallel(_slow_fn, items, max_workers=2, label="Process", use_processes=True)
    assert len(results) == 5


def test_run_parallel_handles_errors():
    items = list(range(6))
    results = run_parallel(_failing_fn, items, max_workers=2, label="Errors")
    # Only odd items should succeed (1, 3, 5)
    assert len(results) == 3
    assert set(results) == {2, 6, 10}
