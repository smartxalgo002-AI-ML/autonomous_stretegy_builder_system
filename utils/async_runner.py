"""
utils/async_runner.py
=====================
Async + multiprocessing helpers used across the system.
Provides:
  - run_in_executor()  : offload CPU-bound work to a ProcessPoolExecutor
  - gather_with_timeout() : asyncio.gather with per-task timeout
  - retry_async()      : exponential-backoff decorator for async coroutines
  - BatchRunner        : concurrent async task batching with throttling
"""

import asyncio
import functools
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Coroutine, Iterable, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# Process-pool helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_cpu_bound(fn: Callable, *args, workers: int = 4, **kwargs) -> Any:
    """
    Run *fn* in a subprocess using ProcessPoolExecutor.
    Useful for vectorised backtests that would block the event loop.
    """
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future = pool.submit(fn, *args, **kwargs)
        return future.result()


def run_batch_cpu_bound(
    fn: Callable,
    items: Iterable,
    workers: int = 4,
    timeout: Optional[float] = None,
) -> List[Any]:
    """
    Apply *fn* to each item in *items* in parallel using ProcessPoolExecutor.
    Returns results in the same order as *items*.
    """
    items = list(items)
    results: List[Any] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for future in as_completed(future_to_idx, timeout=timeout):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Batch CPU task %d failed: %s", idx, exc)
                results[idx] = None
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Async helpers
# ─────────────────────────────────────────────────────────────────────────────

async def run_in_executor(fn: Callable, *args, loop=None, executor=None) -> Any:
    """Await a synchronous *fn* inside an asyncio loop using the default executor."""
    _loop = loop or asyncio.get_event_loop()
    return await _loop.run_in_executor(executor, functools.partial(fn, *args))


async def gather_with_timeout(
    coros: Iterable[Coroutine],
    timeout: float = 60.0,
    return_exceptions: bool = True,
) -> List[Any]:
    """
    asyncio.gather with a global timeout.
    Individual task exceptions are captured (not propagated) when
    *return_exceptions* is True.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=return_exceptions),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("gather_with_timeout: timed out after %.1f s", timeout)
        for t in tasks:
            t.cancel()
        return [None] * len(tasks)


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator: retry an async function up to *max_attempts* times with
    exponential backoff.

    Usage:
        @retry_async(max_attempts=3, base_delay=0.5)
        async def fetch_data(): ...
    """
    def decorator(fn: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "retry_async: %s failed after %d attempts – %s",
                            fn.__name__, max_attempts, exc,
                        )
                        raise
                    logger.warning(
                        "retry_async: %s attempt %d/%d failed (%s). Retrying in %.1f s…",
                        fn.__name__, attempt, max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


class BatchRunner:
    """
    Run async coroutines in throttled batches.

    Parameters
    ----------
    concurrency : int  – maximum number of concurrent tasks
    timeout     : float – per-batch timeout in seconds
    """

    def __init__(self, concurrency: int = 8, timeout: float = 120.0):
        self.concurrency = concurrency
        self.timeout = timeout
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
        return self._semaphore

    async def _run_one(self, coro: Coroutine) -> Any:
        async with self.semaphore:
            try:
                return await asyncio.wait_for(coro, timeout=self.timeout)
            except asyncio.TimeoutError:
                logger.warning("BatchRunner: task timed out after %.1f s", self.timeout)
                return None
            except Exception as exc:
                logger.error("BatchRunner: task error – %s", exc)
                return None

    async def run_all(self, coros: List[Coroutine]) -> List[Any]:
        """Execute all coroutines with bounded concurrency. Returns results list."""
        tasks = [self._run_one(c) for c in coros]
        return await asyncio.gather(*tasks, return_exceptions=True)


# ─────────────────────────────────────────────────────────────────────────────
# Timing utility
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Simple context-manager / decorator for elapsed-time measurement."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        logger.debug("⏱  %s completed in %.3f s", self.label or "task", self.elapsed)
