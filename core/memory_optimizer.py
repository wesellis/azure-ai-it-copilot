"""
Memory and CPU optimization utilities for maximum performance
Includes object pooling, memory profiling, and resource optimization
"""

import gc
import sys
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Generic
from collections import defaultdict, deque
import psutil
import resource
from functools import wraps
import tracemalloc
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory percentage
    available_mb: float  # Available memory in MB
    gc_counts: tuple  # Garbage collection counts


class ObjectPool(Generic[T]):
    """High-performance object pool to reduce allocation overhead"""

    def __init__(self, factory: callable, max_size: int = 100):
        self._factory = factory
        self._pool: deque = deque(maxlen=max_size)
        self._created = 0
        self._reused = 0
        self._lock = threading.Lock()

    def acquire(self) -> T:
        """Get an object from the pool or create new one"""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._reused += 1
                return obj
            else:
                self._created += 1
                return self._factory()

    def release(self, obj: T):
        """Return object to pool"""
        with self._lock:
            # Reset object state if it has a reset method
            if hasattr(obj, 'reset'):
                obj.reset()
            self._pool.append(obj)

    def stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'created': self._created,
                'reused': self._reused,
                'reuse_rate': (self._reused / max(1, self._created + self._reused)) * 100
            }


class MemoryProfiler:
    """Lightweight memory profiler for performance monitoring"""

    def __init__(self):
        self._snapshots = []
        self._peak_memory = 0
        self._start_memory = 0
        self._enabled = False

    def start(self):
        """Start memory profiling"""
        if not tracemalloc.is_tracing():
            tracemalloc.start(10)  # Keep 10 frames
        self._enabled = True
        self._start_memory = self.get_current_memory().rss_mb

    def stop(self):
        """Stop memory profiling"""
        self._enabled = False
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def snapshot(self, label: str = ""):
        """Take a memory snapshot"""
        if not self._enabled:
            return

        stats = self.get_current_memory()
        if stats.rss_mb > self._peak_memory:
            self._peak_memory = stats.rss_mb

        self._snapshots.append({
            'timestamp': time.time(),
            'label': label,
            'memory': stats,
            'tracemalloc': tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None
        })

    def get_current_memory(self) -> MemoryStats:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        virtual_memory = psutil.virtual_memory()

        return MemoryStats(
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=virtual_memory.available / 1024 / 1024,
            gc_counts=gc.get_count()
        )

    def get_memory_growth(self) -> float:
        """Get memory growth since profiling started"""
        current = self.get_current_memory().rss_mb
        return current - self._start_memory

    def get_peak_memory(self) -> float:
        """Get peak memory usage"""
        return self._peak_memory

    def get_top_allocations(self, limit: int = 10) -> List[Dict]:
        """Get top memory allocations"""
        if not tracemalloc.is_tracing():
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        return [
            {
                'filename': stat.traceback.format()[-1],
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            }
            for stat in top_stats[:limit]
        ]


class ResourceOptimizer:
    """System resource optimization utilities"""

    def __init__(self):
        self._gc_thresholds = gc.get_threshold()
        self._optimized = False

    def optimize_gc(self):
        """Optimize garbage collection settings"""
        if self._optimized:
            return

        # Increase GC thresholds for better performance
        # This reduces GC frequency at cost of slightly higher memory usage
        gc.set_threshold(1000, 15, 15)  # Default is (700, 10, 10)
        self._optimized = True

    def restore_gc(self):
        """Restore original GC settings"""
        if self._optimized:
            gc.set_threshold(*self._gc_thresholds)
            self._optimized = False

    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return stats"""
        before = gc.get_count()
        collected = gc.collect()
        after = gc.get_count()

        return {
            'collected': collected,
            'before': before,
            'after': after,
            'generation_0': before[0] - after[0],
            'generation_1': before[1] - after[1],
            'generation_2': before[2] - after[2]
        }

    def set_memory_limit(self, limit_mb: int):
        """Set memory limit for the process"""
        try:
            # Set memory limit (Linux/Mac only)
            resource.setrlimit(resource.RLIMIT_AS, (limit_mb * 1024 * 1024, -1))
        except (OSError, AttributeError):
            pass  # Not supported on all platforms

    def optimize_for_speed(self):
        """Optimize system settings for maximum speed"""
        # Optimize GC
        self.optimize_gc()

        # Disable debug mode if enabled
        if __debug__:
            import builtins
            builtins.__debug__ = False

        # Set Python optimization flags
        sys.flags.optimize = 2

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=0.1)

        return {
            'cpu_percent': cpu_percent,
            'memory': self._get_memory_stats(),
            'open_files': len(process.open_files()),
            'threads': process.num_threads(),
            'gc_stats': {
                'counts': gc.get_count(),
                'thresholds': gc.get_threshold(),
                'stats': gc.get_stats()
            }
        }

    def _get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'shared_mb': getattr(memory_info, 'shared', 0) / 1024 / 1024,
            'lib_mb': getattr(memory_info, 'lib', 0) / 1024 / 1024,
        }


# Global instances
memory_profiler = MemoryProfiler()
resource_optimizer = ResourceOptimizer()

# Common object pools
string_pool = ObjectPool(str, max_size=200)
list_pool = ObjectPool(list, max_size=100)
dict_pool = ObjectPool(dict, max_size=100)


def memory_monitor(func):
    """Decorator to monitor memory usage of function calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        memory_profiler.snapshot(f"Before {func.__name__}")
        start_memory = memory_profiler.get_current_memory().rss_mb

        try:
            result = func(*args, **kwargs)
        finally:
            end_memory = memory_profiler.get_current_memory().rss_mb
            growth = end_memory - start_memory

            if growth > 10:  # Log if function uses more than 10MB
                print(f"Memory growth in {func.__name__}: {growth:.2f}MB")

            memory_profiler.snapshot(f"After {func.__name__}")

        return result
    return wrapper


def cpu_profile(func):
    """Decorator to profile CPU usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_cpu = process.cpu_percent()
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        end_cpu = process.cpu_percent()
        duration = end_time - start_time

        if duration > 1.0:  # Log slow functions
            print(f"CPU usage in {func.__name__}: {end_cpu:.1f}%, Duration: {duration:.3f}s")

        return result
    return wrapper


class MemoryLeakDetector:
    """Detect potential memory leaks"""

    def __init__(self):
        self._object_counts = defaultdict(int)
        self._last_snapshot = None

    def snapshot(self):
        """Take a snapshot of object counts"""
        self._last_snapshot = time.time()
        self._object_counts.clear()

        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            self._object_counts[obj_type] += 1

    def detect_leaks(self, threshold: int = 1000) -> List[Dict]:
        """Detect potential memory leaks by comparing object counts"""
        current_counts = defaultdict(int)

        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            current_counts[obj_type] += 1

        leaks = []
        for obj_type, current_count in current_counts.items():
            previous_count = self._object_counts.get(obj_type, 0)
            growth = current_count - previous_count

            if growth > threshold:
                leaks.append({
                    'type': obj_type,
                    'previous': previous_count,
                    'current': current_count,
                    'growth': growth
                })

        return sorted(leaks, key=lambda x: x['growth'], reverse=True)


# Global leak detector
leak_detector = MemoryLeakDetector()


def optimize_startup():
    """Optimize application for faster startup"""
    # Optimize garbage collection
    resource_optimizer.optimize_for_speed()

    # Pre-allocate common objects
    for _ in range(50):
        string_pool.release("")
        list_pool.release([])
        dict_pool.release({})

    # Start memory profiling
    memory_profiler.start()

    print(f"Memory optimization enabled. Current usage: {memory_profiler.get_current_memory().rss_mb:.1f}MB")


def cleanup_resources():
    """Clean up resources and restore settings"""
    resource_optimizer.restore_gc()
    memory_profiler.stop()

    # Force garbage collection
    stats = resource_optimizer.force_gc()
    print(f"Cleanup complete. Collected {stats['collected']} objects.")


# Auto-optimize on import
optimize_startup()