"""
Memory Optimization and Garbage Collection Management
Advanced memory management patterns for optimal performance
"""

import gc
import sys
import weakref
import time
import logging
import threading
import asyncio
from typing import Dict, Any, Optional, List, Set, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import tracemalloc
import psutil
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float
    gc_count: Dict[int, int] = field(default_factory=dict)
    objects_count: int = 0
    largest_objects: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class MemoryTracker:
    """Advanced memory tracking and analysis"""

    def __init__(self):
        self._start_time = time.time()
        self._memory_history: deque = deque(maxlen=1000)
        self._peak_memory = 0.0
        self._gc_stats: Dict[int, List[int]] = defaultdict(list)
        self._tracemalloc_enabled = False
        self._monitoring_enabled = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

    def enable_tracemalloc(self):
        """Enable tracemalloc for detailed memory tracking"""
        if not self._tracemalloc_enabled:
            tracemalloc.start()
            self._tracemalloc_enabled = True
            logger.info("Memory tracing enabled")

    def disable_tracemalloc(self):
        """Disable tracemalloc"""
        if self._tracemalloc_enabled:
            tracemalloc.stop()
            self._tracemalloc_enabled = False
            logger.info("Memory tracing disabled")

    def start_monitoring(self, interval: float = 5.0):
        """Start continuous memory monitoring"""
        if self._monitoring_enabled:
            return

        self._monitoring_enabled = True
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop memory monitoring"""
        if self._monitoring_enabled:
            self._stop_monitoring.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
            self._monitoring_enabled = False
            logger.info("Memory monitoring stopped")

    def _monitor_loop(self, interval: float):
        """Memory monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                stats = self.get_memory_stats()
                self._memory_history.append(stats)

                # Track peak memory
                if stats.rss_mb > self._peak_memory:
                    self._peak_memory = stats.rss_mb

                # Check for memory leaks
                if len(self._memory_history) >= 10:
                    self._check_memory_trends()

                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(interval)

    def _check_memory_trends(self):
        """Check for concerning memory trends"""
        recent_stats = list(self._memory_history)[-10:]
        if len(recent_stats) < 10:
            return

        # Check for consistent memory growth
        memory_values = [stat.rss_mb for stat in recent_stats]
        if all(memory_values[i] <= memory_values[i + 1] for i in range(len(memory_values) - 1)):
            # Consistent growth detected
            growth_rate = (memory_values[-1] - memory_values[0]) / len(memory_values)
            if growth_rate > 5:  # More than 5MB per measurement
                logger.warning(f"Memory leak detected: growth rate {growth_rate:.2f}MB per check")

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        # System memory info
        system_memory = psutil.virtual_memory()

        # GC statistics
        gc_stats = {}
        for generation in range(3):
            gc_stats[generation] = gc.get_count()[generation]

        # Object count
        objects_count = len(gc.get_objects())

        # Get largest objects if tracemalloc is enabled
        largest_objects = []
        if self._tracemalloc_enabled:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            largest_objects = [
                f"{stat.traceback.format()[0]}: {stat.size_diff / 1024 / 1024:.2f}MB"
                for stat in top_stats
            ]

        return MemoryStats(
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=system_memory.available / 1024 / 1024,
            gc_count=gc_stats,
            objects_count=objects_count,
            largest_objects=largest_objects
        )

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report"""
        current_stats = self.get_memory_stats()
        uptime = time.time() - self._start_time

        # Calculate trends
        growth_rate = 0.0
        if len(self._memory_history) >= 2:
            first_memory = self._memory_history[0].rss_mb
            last_memory = self._memory_history[-1].rss_mb
            time_diff = self._memory_history[-1].timestamp - self._memory_history[0].timestamp
            if time_diff > 0:
                growth_rate = (last_memory - first_memory) / time_diff * 3600  # MB per hour

        return {
            'current': {
                'rss_mb': current_stats.rss_mb,
                'vms_mb': current_stats.vms_mb,
                'percent': current_stats.percent,
                'available_mb': current_stats.available_mb,
                'objects_count': current_stats.objects_count
            },
            'peak_memory_mb': self._peak_memory,
            'uptime_hours': uptime / 3600,
            'growth_rate_mb_per_hour': growth_rate,
            'gc_stats': current_stats.gc_count,
            'largest_objects': current_stats.largest_objects,
            'monitoring_enabled': self._monitoring_enabled,
            'tracemalloc_enabled': self._tracemalloc_enabled,
            'history_length': len(self._memory_history)
        }


class ObjectPool(Generic[T]):
    """Generic object pool for memory optimization"""

    def __init__(self, factory: Callable[[], T], max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self._pool: List[T] = []
        self._created_count = 0
        self._reused_count = 0
        self._lock = threading.Lock()

    def acquire(self) -> T:
        """Acquire object from pool"""
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self._reused_count += 1
                return obj
            else:
                obj = self.factory()
                self._created_count += 1
                return obj

    def release(self, obj: T):
        """Release object back to pool"""
        with self._lock:
            if len(self._pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'max_size': self.max_size,
                'created_count': self._created_count,
                'reused_count': self._reused_count,
                'reuse_rate': self._reused_count / max(self._created_count, 1) * 100
            }


class WeakRefCache:
    """Cache using weak references to prevent memory leaks"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, weakref.ref] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key in self._cache:
                ref = self._cache[key]
                obj = ref()
                if obj is not None:
                    self._access_times[key] = time.time()
                    return obj
                else:
                    # Object was garbage collected
                    del self._cache[key]
                    self._access_times.pop(key, None)
            return None

    def set(self, key: str, value: Any):
        """Set item in cache"""
        def cleanup_callback(ref):
            with self._lock:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)

        with self._lock:
            # Cleanup if cache is full
            if len(self._cache) >= self.max_size:
                # Remove oldest accessed item
                oldest_key = min(self._access_times.keys(), key=self._access_times.get)
                self._cache.pop(oldest_key, None)
                self._access_times.pop(oldest_key, None)

            # Store weak reference
            self._cache[key] = weakref.ref(value, cleanup_callback)
            self._access_times[key] = time.time()

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            alive_count = sum(1 for ref in self._cache.values() if ref() is not None)
            return {
                'total_entries': len(self._cache),
                'alive_entries': alive_count,
                'dead_entries': len(self._cache) - alive_count,
                'max_size': self.max_size
            }


class GarbageCollectionManager:
    """Advanced garbage collection management"""

    def __init__(self):
        self._gc_stats = {'collections': 0, 'objects_collected': 0, 'time_spent': 0.0}
        self._thresholds = list(gc.get_threshold())
        self._original_thresholds = self._thresholds.copy()
        self._auto_gc_enabled = True

    def optimize_gc_thresholds(self, workload_type: str = "web_server"):
        """Optimize GC thresholds based on workload type"""
        if workload_type == "web_server":
            # More frequent gen0, less frequent gen1/gen2
            self._thresholds = [700, 15, 15]
        elif workload_type == "batch_processing":
            # Less frequent GC for better throughput
            self._thresholds = [2000, 25, 25]
        elif workload_type == "real_time":
            # Very frequent gen0, avoid gen2
            self._thresholds = [400, 20, 0]
        else:
            # Conservative defaults
            self._thresholds = [700, 10, 10]

        gc.set_threshold(*self._thresholds)
        logger.info(f"GC thresholds optimized for {workload_type}: {self._thresholds}")

    def force_gc_cycle(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics"""
        start_time = time.time()

        # Collect statistics before GC
        before_objects = len(gc.get_objects())
        before_counts = gc.get_count()

        # Force collection for all generations
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))

        # Collect statistics after GC
        after_objects = len(gc.get_objects())
        after_counts = gc.get_count()
        duration = time.time() - start_time

        # Update internal stats
        self._gc_stats['collections'] += 1
        self._gc_stats['objects_collected'] += sum(collected)
        self._gc_stats['time_spent'] += duration

        return {
            'objects_before': before_objects,
            'objects_after': after_objects,
            'objects_collected': sum(collected),
            'collected_by_generation': collected,
            'counts_before': before_counts,
            'counts_after': after_counts,
            'duration_ms': duration * 1000,
            'freed_memory_estimate_mb': (before_objects - after_objects) * 0.001  # Rough estimate
        }

    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        return {
            'enabled': gc.isenabled(),
            'thresholds': gc.get_threshold(),
            'original_thresholds': self._original_thresholds,
            'counts': gc.get_count(),
            'stats': gc.get_stats(),
            'total_collections': self._gc_stats['collections'],
            'total_objects_collected': self._gc_stats['objects_collected'],
            'total_time_spent': self._gc_stats['time_spent']
        }

    def disable_gc(self):
        """Disable automatic garbage collection"""
        gc.disable()
        self._auto_gc_enabled = False
        logger.warning("Automatic garbage collection disabled")

    def enable_gc(self):
        """Enable automatic garbage collection"""
        gc.enable()
        self._auto_gc_enabled = True
        logger.info("Automatic garbage collection enabled")

    def reset_thresholds(self):
        """Reset GC thresholds to original values"""
        gc.set_threshold(*self._original_thresholds)
        self._thresholds = self._original_thresholds.copy()
        logger.info(f"GC thresholds reset to original: {self._original_thresholds}")


# Global instances
memory_tracker = MemoryTracker()
gc_manager = GarbageCollectionManager()


def memory_profile(func):
    """Decorator to profile memory usage of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if memory_tracker._tracemalloc_enabled:
            tracemalloc.start()

        start_memory = memory_tracker.get_memory_stats()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = memory_tracker.get_memory_stats()

            memory_diff = end_memory.rss_mb - start_memory.rss_mb
            duration = end_time - start_time

            if memory_diff > 10:  # Log if function used more than 10MB
                logger.warning(
                    f"High memory usage in {func.__name__}: "
                    f"{memory_diff:.2f}MB in {duration:.3f}s"
                )

    return wrapper


def optimize_for_memory():
    """Apply memory optimizations"""
    # Optimize GC for web server workload
    gc_manager.optimize_gc_thresholds("web_server")

    # Enable memory tracking
    memory_tracker.enable_tracemalloc()
    memory_tracker.start_monitoring()

    # Set smaller recursion limit if very high
    current_limit = sys.getrecursionlimit()
    if current_limit > 3000:
        sys.setrecursionlimit(1500)
        logger.info(f"Recursion limit reduced from {current_limit} to 1500")

    logger.info("Memory optimizations applied")


def get_memory_summary() -> Dict[str, Any]:
    """Get comprehensive memory summary"""
    return {
        'tracker': memory_tracker.get_memory_report(),
        'gc': gc_manager.get_gc_stats(),
        'process': {
            'pid': os.getpid(),
            'threads': threading.active_count(),
            'recursion_limit': sys.getrecursionlimit()
        }
    }


# Context manager for temporary memory optimization
class TemporaryMemoryOptimization:
    """Context manager for temporary memory optimization during heavy operations"""

    def __init__(self, disable_gc: bool = False, force_gc_before: bool = True):
        self.disable_gc = disable_gc
        self.force_gc_before = force_gc_before
        self._gc_was_enabled = gc.isenabled()

    def __enter__(self):
        if self.force_gc_before:
            gc_manager.force_gc_cycle()

        if self.disable_gc:
            gc.disable()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable_gc and self._gc_was_enabled:
            gc.enable()

        # Force GC after heavy operation
        gc_manager.force_gc_cycle()


# Async version of memory optimization context manager
class AsyncTemporaryMemoryOptimization:
    """Async context manager for temporary memory optimization"""

    def __init__(self, disable_gc: bool = False, force_gc_before: bool = True):
        self.disable_gc = disable_gc
        self.force_gc_before = force_gc_before
        self._gc_was_enabled = gc.isenabled()

    async def __aenter__(self):
        if self.force_gc_before:
            # Run GC in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, gc_manager.force_gc_cycle)

        if self.disable_gc:
            gc.disable()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.disable_gc and self._gc_was_enabled:
            gc.enable()

        # Force GC after heavy operation
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, gc_manager.force_gc_cycle)


# Convenience functions
def get_current_memory_usage() -> float:
    """Get current memory usage in MB"""
    return memory_tracker.get_memory_stats().rss_mb


def force_garbage_collection() -> Dict[str, Any]:
    """Force garbage collection and return stats"""
    return gc_manager.force_gc_cycle()


def memory_efficient_operation(func):
    """Decorator for memory-efficient operations"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        async with AsyncTemporaryMemoryOptimization():
            return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        with TemporaryMemoryOptimization():
            return func(*args, **kwargs)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper