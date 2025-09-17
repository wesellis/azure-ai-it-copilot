"""
Performance optimization utilities and decorators
"""

import asyncio
import time
import functools
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""

    def __init__(self):
        self._metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._counters: Dict[str, int] = defaultdict(int)
        self._timings: Dict[str, list] = defaultdict(list)

    def record_timing(self, operation: str, duration: float):
        """Record operation timing"""
        self._timings[operation].append(duration)
        if len(self._timings[operation]) > 1000:  # Keep only last 1000 measurements
            self._timings[operation] = self._timings[operation][-1000:]

    def increment_counter(self, metric: str, value: int = 1):
        """Increment a counter metric"""
        self._counters[metric] += value

    def get_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for an operation"""
        timings = self._timings.get(operation, [])
        if not timings:
            return {"count": 0, "avg": 0, "min": 0, "max": 0}

        return {
            "count": len(timings),
            "avg": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
            "p95": sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 20 else max(timings)
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all performance statistics"""
        return {
            "timings": {op: self.get_stats(op) for op in self._timings.keys()},
            "counters": dict(self._counters),
            "last_updated": datetime.utcnow().isoformat()
        }


# Global performance monitor
perf_monitor = PerformanceMonitor()


def time_async(operation_name: str = None):
    """Decorator to time async functions"""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                perf_monitor.record_timing(op_name, duration)
                if duration > 1.0:  # Log slow operations
                    logger.warning(f"Slow operation {op_name}: {duration:.3f}s")

        return wrapper
    return decorator


def time_sync(operation_name: str = None):
    """Decorator to time sync functions"""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                perf_monitor.record_timing(op_name, duration)
                if duration > 0.5:  # Log slow sync operations
                    logger.warning(f"Slow sync operation {op_name}: {duration:.3f}s")

        return wrapper
    return decorator


def cache_result(ttl_seconds: int = 300):
    """Simple result caching decorator"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()

            # Check if cached result exists and is still valid
            if key in cache and (now - cache_times[key]) < ttl_seconds:
                perf_monitor.increment_counter(f"cache_hit_{func.__name__}")
                return cache[key]

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = now
            perf_monitor.increment_counter(f"cache_miss_{func.__name__}")

            # Clean old cache entries
            if len(cache) > 1000:  # Prevent unlimited growth
                cutoff = now - ttl_seconds
                to_remove = [k for k, t in cache_times.items() if t < cutoff]
                for k in to_remove:
                    cache.pop(k, None)
                    cache_times.pop(k, None)

            return result

        return wrapper
    return decorator


async def batch_operations(operations: list, batch_size: int = 10, delay: float = 0.1):
    """Execute operations in batches to avoid overwhelming systems"""
    results = []

    for i in range(0, len(operations), batch_size):
        batch = operations[i:i + batch_size]

        # Execute batch concurrently
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        results.extend(batch_results)

        # Small delay between batches
        if i + batch_size < len(operations) and delay > 0:
            await asyncio.sleep(delay)

    return results


class ConnectionPool:
    """Simple connection pool implementation"""

    def __init__(self, factory_func: Callable, max_size: int = 10, timeout: float = 30.0):
        self.factory_func = factory_func
        self.max_size = max_size
        self.timeout = timeout
        self._pool = asyncio.Queue(maxsize=max_size)
        self._created = 0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a connection from the pool"""
        try:
            # Try to get existing connection
            connection = self._pool.get_nowait()
            return connection
        except asyncio.QueueEmpty:
            # Create new connection if under limit
            async with self._lock:
                if self._created < self.max_size:
                    connection = await self.factory_func()
                    self._created += 1
                    return connection
                else:
                    # Wait for available connection
                    return await asyncio.wait_for(self._pool.get(), timeout=self.timeout)

    async def release(self, connection):
        """Release a connection back to the pool"""
        try:
            self._pool.put_nowait(connection)
        except asyncio.QueueFull:
            # Pool is full, close the connection
            if hasattr(connection, 'close'):
                await connection.close()

    async def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            try:
                connection = self._pool.get_nowait()
                if hasattr(connection, 'close'):
                    await connection.close()
            except asyncio.QueueEmpty:
                break


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate: float, burst: int = None):
        self.rate = rate  # tokens per second
        self.burst = burst or int(rate * 2)  # burst capacity
        self.tokens = self.burst
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait_for_tokens(self, tokens: int = 1):
        """Wait until tokens are available"""
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)


def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics"""
    return perf_monitor.get_all_stats()


def reset_performance_stats():
    """Reset all performance statistics"""
    global perf_monitor
    perf_monitor = PerformanceMonitor()


# Async context manager for timing code blocks
class Timer:
    """Context manager for timing code blocks"""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            perf_monitor.record_timing(self.operation_name, duration)