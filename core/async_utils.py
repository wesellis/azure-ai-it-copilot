"""
Async utilities and optimizations for Azure AI IT Copilot
"""

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, List, Optional, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

T = TypeVar('T')


# Global thread pool for CPU-bound tasks
_thread_pool: Optional[ThreadPoolExecutor] = None


def get_thread_pool() -> ThreadPoolExecutor:
    """Get the global thread pool for CPU-bound tasks"""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="copilot-worker"
        )
    return _thread_pool


def cleanup_thread_pool():
    """Cleanup the global thread pool"""
    global _thread_pool
    if _thread_pool:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None


async def run_in_thread(func: Callable[..., T], *args, **kwargs) -> T:
    """Run a sync function in a thread pool"""
    loop = asyncio.get_event_loop()
    thread_pool = get_thread_pool()

    return await loop.run_in_executor(
        thread_pool,
        functools.partial(func, *args, **kwargs)
    )


async def gather_with_concurrency(
    awaitables: List[Awaitable[T]],
    max_concurrency: int = 10,
    return_exceptions: bool = False
) -> List[Union[T, Exception]]:
    """
    Execute awaitables with limited concurrency
    More memory efficient than asyncio.gather for large lists
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_task(awaitable: Awaitable[T]) -> Union[T, Exception]:
        async with semaphore:
            try:
                return await awaitable
            except Exception as e:
                if return_exceptions:
                    return e
                raise

    limited_awaitables = [limited_task(aw) for aw in awaitables]
    return await asyncio.gather(*limited_awaitables, return_exceptions=return_exceptions)


async def timeout_after(seconds: float, coro: Awaitable[T]) -> Optional[T]:
    """Execute coroutine with timeout, return None on timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {seconds} seconds")
        return None


class AsyncLazy:
    """Lazy async initialization"""

    def __init__(self, factory: Callable[[], Awaitable[T]]):
        self._factory = factory
        self._value: Optional[T] = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def get(self) -> T:
        """Get the lazily initialized value"""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    self._value = await self._factory()
                    self._initialized = True
        return self._value

    def reset(self):
        """Reset the lazy value"""
        self._value = None
        self._initialized = False


class AsyncCache:
    """Simple async cache with TTL"""

    def __init__(self, ttl_seconds: float = 300):
        self._cache: dict = {}
        self._timestamps: dict = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key in self._cache:
                timestamp = self._timestamps[key]
                if asyncio.get_event_loop().time() - timestamp < self._ttl:
                    return self._cache[key]
                else:
                    # Expired
                    del self._cache[key]
                    del self._timestamps[key]
            return None

    async def set(self, key: str, value: Any):
        """Set value in cache"""
        async with self._lock:
            self._cache[key] = value
            self._timestamps[key] = asyncio.get_event_loop().time()

    async def clear(self):
        """Clear all cache"""
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Async retry decorator with exponential backoff"""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}"
                    )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            # All attempts failed
            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            raise last_exception

        return wrapper
    return decorator


class AsyncCircuitBreaker:
    """Async circuit breaker pattern"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Call function through circuit breaker"""
        async with self._lock:
            if self.state == "OPEN":
                if (asyncio.get_event_loop().time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)

            # Success - reset if we were half open
            async with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0

            return result

        except self.expected_exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = asyncio.get_event_loop().time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"

            raise e


async def async_map(
    func: Callable[[T], Awaitable[Any]],
    items: List[T],
    max_concurrency: int = 10
) -> List[Any]:
    """Async map with concurrency control"""
    async_tasks = [func(item) for item in items]
    return await gather_with_concurrency(async_tasks, max_concurrency)


async def async_filter(
    predicate: Callable[[T], Awaitable[bool]],
    items: List[T],
    max_concurrency: int = 10
) -> List[T]:
    """Async filter with concurrency control"""
    async def check_item(item: T) -> tuple:
        result = await predicate(item)
        return item, result

    check_tasks = [check_item(item) for item in items]
    results = await gather_with_concurrency(check_tasks, max_concurrency)

    return [item for item, passed in results if passed]


class AsyncEventAggregator:
    """Aggregate events and process them in batches"""

    def __init__(
        self,
        batch_size: int = 100,
        batch_timeout: float = 5.0,
        processor: Callable[[List], Awaitable[None]] = None
    ):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.processor = processor

        self._queue = asyncio.Queue()
        self._batch = []
        self._last_batch_time = asyncio.get_event_loop().time()
        self._processing_task = None
        self._shutdown = False

    async def add_event(self, event: Any):
        """Add event to be processed"""
        if not self._shutdown:
            await self._queue.put(event)

    async def start_processing(self):
        """Start the batch processing task"""
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._process_batches())

    async def stop_processing(self):
        """Stop processing and process remaining events"""
        self._shutdown = True

        if self._processing_task:
            await self._processing_task

        # Process any remaining events
        if self._batch and self.processor:
            await self.processor(self._batch.copy())
            self._batch.clear()

    async def _process_batches(self):
        """Process events in batches"""
        while not self._shutdown:
            try:
                # Wait for event or timeout
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=self.batch_timeout
                    )
                    self._batch.append(event)
                except asyncio.TimeoutError:
                    pass

                # Check if we should process the batch
                current_time = asyncio.get_event_loop().time()
                should_process = (
                    len(self._batch) >= self.batch_size or
                    (self._batch and
                     current_time - self._last_batch_time >= self.batch_timeout)
                )

                if should_process and self._batch and self.processor:
                    batch_to_process = self._batch.copy()
                    self._batch.clear()
                    self._last_batch_time = current_time

                    try:
                        await self.processor(batch_to_process)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")

            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(1)  # Prevent tight error loop