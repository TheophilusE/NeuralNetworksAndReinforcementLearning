"""Simple multi-threaded task worker utility.

Provides a lightweight thread-pool manager with a convenient API for
submitting callables, tracking in-flight tasks, cancelling, and
graceful shutdown. This is intended for use by backend components
that need to run CPU-bound or blocking tasks concurrently without
spawning full processes.

API:
 - ThreadWorker(max_workers=None)
 - submit(fn, *args, **kwargs) -> Future
 - map(fn, *iterables) -> iterator
 - cancel_all()
 - shutdown(wait=True)

Example usage:
    worker = ThreadWorker(max_workers=4)
    fut = worker.submit(expensive_fn, arg)
    result = fut.result()
    worker.shutdown()

This module uses only the Python standard library.
"""
from __future__ import annotations

import threading
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Iterable, Iterator, Optional
import time

_LOG = logging.getLogger(__name__)


class ThreadWorker:
    """Manage a pool of worker threads and submitted tasks.

    This class wraps `concurrent.futures.ThreadPoolExecutor` and keeps
    a set of in-flight futures so callers can cancel outstanding
    work, query activity, and shutdown cleanly.
    """

    def __init__(self, max_workers: Optional[int] = None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._futures: set[Future] = set()
        self._shutdown = False

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        """Submit a callable to run in the thread pool.

        Returns a `concurrent.futures.Future`.
        """
        if self._shutdown:
            raise RuntimeError("ThreadWorker is shut down")

        fut = self._executor.submit(fn, *args, **kwargs)

        def _remove_done(f: Future) -> None:
            with self._lock:
                self._futures.discard(f)

        with self._lock:
            self._futures.add(fut)
        fut.add_done_callback(_remove_done)
        return fut

    def map(self, fn: Callable[..., Any], *iterables: Iterable[Any], timeout: Optional[float] = None) -> Iterator[Any]:
        """Map `fn` across the provided iterables using the pool.

        This returns an iterator that yields results in the order of the
        inputs (like `executor.map`).
        """
        return self._executor.map(fn, *iterables, timeout=timeout)  # type: ignore[arg-type]

    def cancel_all(self) -> None:
        """Attempt to cancel all in-flight futures.

        Note: cancellation only succeeds for futures that haven't started.
        """
        with self._lock:
            for f in list(self._futures):
                try:
                    f.cancel()
                except Exception:
                    _LOG.exception("Failed to cancel future")

    def in_flight_count(self) -> int:
        """Return number of in-flight (submitted but not finished) tasks."""
        with self._lock:
            return len(self._futures)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the pool and optionally wait for tasks to finish."""
        self._shutdown = True
        self._executor.shutdown(wait=wait)


# Module-level default worker for simple centralized task dispatching.
# Callers can import `default_worker` and use `submit` to dispatch work.
default_worker: ThreadWorker = ThreadWorker()


def submit_task(fn: Callable[..., Any], *args, **kwargs) -> Future:
    """Convenience helper to submit work to the module default worker."""
    return default_worker.submit(fn, *args, **kwargs)


def example_heavy(x: int) -> int:
    """Example CPU-bound or blocking task used in the module self-test."""
    time.sleep(0.1)
    return x * x


if __name__ == "__main__":
    # Quick smoke test when executed directly
    logging.basicConfig(level=logging.INFO)
    w = ThreadWorker(max_workers=4)
    futs = [w.submit(example_heavy, i) for i in range(10)]
    results = [f.result() for f in futs]
    print("results:", results)
    print("in_flight after completion:", w.in_flight_count())
    w.shutdown()
