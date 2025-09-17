"""
Advanced Async Optimization Patterns
Implements sophisticated async patterns for maximum performance
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic, Union, Awaitable
from functools import wraps, partial
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class AsyncMetrics:
    """Metrics for async operations"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_execution_time: float = 0.0
    total_execution_time: float = 0.0
    concurrent_operations: int = 0
    max_concurrent: int = 0
    queue_size: int = 0
    last_operation: Optional[float] = None


class AsyncSemaphorePool:
    """Advanced semaphore pool with priority and backpressure"""

    def __init__(self, max_concurrent: int = 100, priority_levels: int = 3):
        self.max_concurrent = max_concurrent
        self.priority_levels = priority_levels
        self._semaphores = [asyncio.Semaphore(max_concurrent) for _ in range(priority_levels)]
        self._queues = [asyncio.Queue() for _ in range(priority_levels)]
        self._metrics = AsyncMetrics()
        self._active_operations: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, priority: int = 1, operation_id: str = None) -> bool:
        """Acquire semaphore with priority"""
        priority = min(max(0, priority), self.priority_levels - 1)

        # Update metrics
        async with self._lock:
            self._metrics.queue_size += 1
            if operation_id:
                self._active_operations[operation_id] = time.time()

        try:
            await self._semaphores[priority].acquire()
            async with self._lock:
                self._metrics.concurrent_operations += 1
                self._metrics.max_concurrent = max(
                    self._metrics.max_concurrent,
                    self._metrics.concurrent_operations
                )
            return True
        except Exception:
            async with self._lock:
                self._metrics.queue_size -= 1
            return False

    async def release(self, priority: int = 1, operation_id: str = None, success: bool = True):
        """Release semaphore and update metrics"""
        priority = min(max(0, priority), self.priority_levels - 1)

        async with self._lock:
            self._metrics.concurrent_operations -= 1
            self._metrics.queue_size -= 1
            self._metrics.total_operations += 1

            if success:
                self._metrics.successful_operations += 1
            else:
                self._metrics.failed_operations += 1

            if operation_id and operation_id in self._active_operations:
                execution_time = time.time() - self._active_operations[operation_id]
                self._metrics.total_execution_time += execution_time
                self._metrics.avg_execution_time = (
                    self._metrics.total_execution_time / self._metrics.total_operations
                )
                del self._active_operations[operation_id]

            self._metrics.last_operation = time.time()

        self._semaphores[priority].release()


class AsyncCircuitBreaker:
    """Circuit breaker pattern for async operations"""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = 'closed'  # closed, open, half-open
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self._state == 'open':
                if self._should_attempt_reset():
                    self._state = 'half-open'
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.expected_exception as e:
            await self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self._last_failure_time is not None and
            time.time() - self._last_failure_time >= self.recovery_timeout
        )

    async def _on_success(self):
        """Handle successful operation"""
        async with self._lock:
            self._failure_count = 0
            self._state = 'closed'

    async def _on_failure(self):
        """Handle failed operation"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = 'open'


class AsyncTaskManager:
    """Advanced task management with grouping and cancellation"""

    def __init__(self):
        self._task_groups: Dict[str, List[asyncio.Task]] = defaultdict(list)
        self._task_metadata: Dict[asyncio.Task, Dict[str, Any]] = {}
        self._completed_tasks: deque = deque(maxlen=1000)
        self._lock = asyncio.Lock()

    async def create_task(self,
                         coro: Awaitable[T],
                         group: str = "default",
                         name: str = None,
                         priority: int = 1,
                         timeout: float = None) -> asyncio.Task[T]:
        """Create task with metadata and grouping"""
        task = asyncio.create_task(coro, name=name)

        async with self._lock:
            self._task_groups[group].append(task)
            self._task_metadata[task] = {
                'group': group,
                'name': name or f"task_{id(task)}",
                'priority': priority,
                'timeout': timeout,
                'created_at': time.time(),
                'started_at': None,
                'completed_at': None
            }

        # Add completion callback
        task.add_done_callback(self._task_completed)

        return task

    def _task_completed(self, task: asyncio.Task):
        """Handle task completion"""
        asyncio.create_task(self._async_task_completed(task))

    async def _async_task_completed(self, task: asyncio.Task):
        """Async task completion handler"""
        async with self._lock:
            if task in self._task_metadata:
                metadata = self._task_metadata[task]
                metadata['completed_at'] = time.time()
                metadata['duration'] = metadata['completed_at'] - metadata['created_at']
                metadata['exception'] = task.exception() if task.done() else None

                # Move to completed tasks
                self._completed_tasks.append(metadata)

                # Remove from active tracking
                group = metadata['group']
                if task in self._task_groups[group]:
                    self._task_groups[group].remove(task)
                del self._task_metadata[task]

    async def cancel_group(self, group: str) -> int:
        """Cancel all tasks in a group"""
        async with self._lock:
            tasks = self._task_groups.get(group, [])
            cancelled_count = 0

            for task in tasks:
                if not task.done():
                    task.cancel()
                    cancelled_count += 1

            return cancelled_count

    async def wait_for_group(self, group: str, timeout: float = None) -> List[Any]:
        """Wait for all tasks in a group to complete"""
        async with self._lock:
            tasks = self._task_groups.get(group, []).copy()

        if not tasks:
            return []

        done, pending = await asyncio.wait(tasks, timeout=timeout)

        # Cancel pending tasks if timeout occurred
        if pending:
            for task in pending:
                task.cancel()

        return [task.result() for task in done if not task.cancelled() and task.exception() is None]

    def get_group_status(self, group: str) -> Dict[str, Any]:
        """Get status of task group"""
        tasks = self._task_groups.get(group, [])

        running = sum(1 for task in tasks if not task.done())
        completed = sum(1 for task in tasks if task.done() and not task.cancelled())
        cancelled = sum(1 for task in tasks if task.cancelled())
        failed = sum(1 for task in tasks if task.done() and task.exception())

        return {
            'total': len(tasks),
            'running': running,
            'completed': completed,
            'cancelled': cancelled,
            'failed': failed
        }


class AsyncBatchProcessor:
    """Advanced batch processing with adaptive sizing"""

    def __init__(self,
                 initial_batch_size: int = 50,
                 max_batch_size: int = 200,
                 min_batch_size: int = 10,
                 target_latency: float = 1.0):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.target_latency = target_latency

        self._current_batch_size = initial_batch_size
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._processors: Dict[str, Callable] = {}
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._processing = False

    def register_processor(self, batch_type: str, processor: Callable):
        """Register a batch processor"""
        self._processors[batch_type] = processor

    async def add_item(self, batch_type: str, item: Any):
        """Add item to batch queue"""
        await self._batch_queue.put((batch_type, item))

        if not self._processing:
            asyncio.create_task(self._process_batches())

    async def _process_batches(self):
        """Process batches with adaptive sizing"""
        if self._processing:
            return

        self._processing = True

        try:
            batch_items: Dict[str, List[Any]] = defaultdict(list)

            # Collect items for current batch size
            for _ in range(self._current_batch_size):
                try:
                    batch_type, item = await asyncio.wait_for(
                        self._batch_queue.get(),
                        timeout=0.1
                    )
                    batch_items[batch_type].append(item)
                except asyncio.TimeoutError:
                    break

            # Process each batch type
            for batch_type, items in batch_items.items():
                if items and batch_type in self._processors:
                    start_time = time.time()

                    try:
                        await self._processors[batch_type](items)
                        latency = time.time() - start_time

                        # Record metrics and adjust batch size
                        self._metrics[batch_type].append(latency)
                        self._adjust_batch_size(batch_type, latency)

                    except Exception as e:
                        logger.error(f"Batch processing error for {batch_type}: {e}")

        finally:
            self._processing = False

            # Continue processing if queue not empty
            if not self._batch_queue.empty():
                asyncio.create_task(self._process_batches())

    def _adjust_batch_size(self, batch_type: str, latency: float):
        """Adjust batch size based on latency"""
        recent_latencies = self._metrics[batch_type][-10:]  # Last 10 measurements
        avg_latency = sum(recent_latencies) / len(recent_latencies)

        if avg_latency > self.target_latency and self._current_batch_size > self.min_batch_size:
            self._current_batch_size = max(
                self.min_batch_size,
                int(self._current_batch_size * 0.9)
            )
        elif avg_latency < self.target_latency * 0.7 and self._current_batch_size < self.max_batch_size:
            self._current_batch_size = min(
                self.max_batch_size,
                int(self._current_batch_size * 1.1)
            )


class AsyncResourcePool:
    """Generic async resource pool with lifecycle management"""

    def __init__(self,
                 factory: Callable[[], Awaitable[T]],
                 cleanup: Callable[[T], Awaitable[None]] = None,
                 max_size: int = 20,
                 min_size: int = 5,
                 max_lifetime: float = 3600.0,
                 health_check: Callable[[T], Awaitable[bool]] = None):
        self.factory = factory
        self.cleanup = cleanup
        self.max_size = max_size
        self.min_size = min_size
        self.max_lifetime = max_lifetime
        self.health_check = health_check

        self._pool: asyncio.Queue[tuple] = asyncio.Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = asyncio.Lock()
        self._closed = False

    async def initialize(self):
        """Initialize pool with minimum resources"""
        for _ in range(self.min_size):
            resource = await self.factory()
            await self._pool.put((resource, time.time()))
            self._created_count += 1

    @asynccontextmanager
    async def acquire(self):
        """Acquire resource from pool"""
        if self._closed:
            raise RuntimeError("Resource pool is closed")

        resource = None
        resource_time = None

        try:
            # Try to get existing resource
            try:
                resource, resource_time = await asyncio.wait_for(
                    self._pool.get(), timeout=5.0
                )

                # Check resource health and age
                if await self._is_resource_valid(resource, resource_time):
                    yield resource
                    return
                else:
                    # Resource is invalid, create new one
                    if self.cleanup:
                        await self.cleanup(resource)
                    resource = None

            except asyncio.TimeoutError:
                pass

            # Create new resource if needed
            if resource is None:
                if self._created_count < self.max_size:
                    resource = await self.factory()
                    resource_time = time.time()
                    async with self._lock:
                        self._created_count += 1
                else:
                    raise RuntimeError("Resource pool exhausted")

            yield resource

        finally:
            # Return resource to pool if valid
            if resource is not None and not self._closed:
                if await self._is_resource_valid(resource, resource_time):
                    try:
                        await self._pool.put((resource, resource_time))
                    except asyncio.QueueFull:
                        # Pool is full, cleanup resource
                        if self.cleanup:
                            await self.cleanup(resource)
                        async with self._lock:
                            self._created_count -= 1
                else:
                    # Resource is invalid, cleanup
                    if self.cleanup:
                        await self.cleanup(resource)
                    async with self._lock:
                        self._created_count -= 1

    async def _is_resource_valid(self, resource: T, created_time: float) -> bool:
        """Check if resource is still valid"""
        # Check age
        if time.time() - created_time > self.max_lifetime:
            return False

        # Check health
        if self.health_check:
            try:
                return await self.health_check(resource)
            except Exception:
                return False

        return True

    async def close(self):
        """Close pool and cleanup all resources"""
        self._closed = True

        while not self._pool.empty():
            try:
                resource, _ = self._pool.get_nowait()
                if self.cleanup:
                    await self.cleanup(resource)
            except asyncio.QueueEmpty:
                break


# Global instances for common use cases
semaphore_pool = AsyncSemaphorePool()
task_manager = AsyncTaskManager()
batch_processor = AsyncBatchProcessor()


def async_retry(max_attempts: int = 3,
                delay: float = 1.0,
                backoff: float = 2.0,
                exceptions: tuple = (Exception,)):
    """Retry decorator for async functions"""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise last_exception

        return wrapper
    return decorator


def async_timeout(seconds: float):
    """Timeout decorator for async functions"""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
        return wrapper
    return decorator


async def gather_with_concurrency(awaitables: List[Awaitable[T]],
                                 max_concurrency: int = 10) -> List[T]:
    """Gather awaitables with concurrency limit"""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_awaitable(awaitable):
        async with semaphore:
            return await awaitable

    return await asyncio.gather(*[limited_awaitable(aw) for aw in awaitables])


def get_async_metrics() -> Dict[str, Any]:
    """Get comprehensive async operation metrics"""
    return {
        'semaphore_pool': {
            'total_operations': semaphore_pool._metrics.total_operations,
            'successful_operations': semaphore_pool._metrics.successful_operations,
            'failed_operations': semaphore_pool._metrics.failed_operations,
            'avg_execution_time': semaphore_pool._metrics.avg_execution_time,
            'concurrent_operations': semaphore_pool._metrics.concurrent_operations,
            'max_concurrent': semaphore_pool._metrics.max_concurrent,
        },
        'task_manager': {
            'active_groups': len(task_manager._task_groups),
            'active_tasks': sum(len(tasks) for tasks in task_manager._task_groups.values()),
            'completed_tasks': len(task_manager._completed_tasks),
        },
        'batch_processor': {
            'current_batch_size': batch_processor._current_batch_size,
            'queue_size': batch_processor._batch_queue.qsize(),
            'processors_registered': len(batch_processor._processors),
        }
    }