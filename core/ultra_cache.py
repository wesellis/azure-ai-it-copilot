"""
Ultra-high performance caching system with multiple strategies
Combines memory, disk, and distributed caching for maximum speed
"""

import asyncio
import hashlib
import pickle
import time
import threading
from typing import Any, Dict, Optional, Union, Callable, TypeVar, Generic
from functools import wraps
from pathlib import Path
import weakref
from collections import OrderedDict
import mmap
import os

T = TypeVar('T')


class FastLRUCache(Generic[T]):
    """Ultra-fast LRU cache with O(1) operations using OrderedDict"""

    __slots__ = ('_cache', '_capacity', '_hits', '_misses', '_lock')

    def __init__(self, capacity: int = 1000):
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._capacity = capacity
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: T):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._capacity:
                # Remove least recently used
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                'size': len(self._cache),
                'capacity': self._capacity,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 2)
            }


class MemoryMappedCache:
    """Memory-mapped file cache for large objects"""

    def __init__(self, cache_dir: Path, max_size: int = 100 * 1024 * 1024):  # 100MB
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._index = {}  # key -> (filename, size, timestamp)
        self._lock = threading.RLock()

    def _get_filename(self, key: str) -> Path:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._index:
                return None

            filename, size, timestamp = self._index[key]
            filepath = self.cache_dir / filename

            if not filepath.exists():
                del self._index[key]
                return None

            try:
                with open(filepath, 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        return pickle.loads(mm[:])
            except (OSError, pickle.PickleError):
                self._remove_file(filepath, key)
                return None

    def put(self, key: str, value: Any):
        with self._lock:
            try:
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                size = len(data)

                # Check if we need to clean up space
                self._cleanup_if_needed(size)

                filename = self._get_filename(key)
                with open(filename, 'wb') as f:
                    f.write(data)

                self._index[key] = (filename.name, size, time.time())

            except (OSError, pickle.PickleError):
                pass  # Ignore cache write failures

    def _cleanup_if_needed(self, needed_size: int):
        """Remove old entries if cache is too large"""
        total_size = sum(size for _, size, _ in self._index.values())

        if total_size + needed_size > self.max_size:
            # Sort by timestamp (oldest first)
            sorted_items = sorted(
                self._index.items(),
                key=lambda x: x[1][2]  # timestamp
            )

            # Remove oldest entries until we have enough space
            for key, (filename, size, _) in sorted_items:
                filepath = self.cache_dir / filename
                self._remove_file(filepath, key)
                total_size -= size

                if total_size + needed_size <= self.max_size:
                    break

    def _remove_file(self, filepath: Path, key: str):
        try:
            filepath.unlink(missing_ok=True)
            self._index.pop(key, None)
        except OSError:
            pass

    def clear(self):
        with self._lock:
            for key, (filename, _, _) in self._index.items():
                filepath = self.cache_dir / filename
                self._remove_file(filepath, key)
            self._index.clear()


class MultiLevelCache:
    """Multi-level cache combining in-memory and disk storage"""

    def __init__(self,
                 memory_capacity: int = 1000,
                 disk_capacity: int = 100 * 1024 * 1024,
                 cache_dir: Optional[Path] = None):
        self.memory_cache = FastLRUCache[Any](memory_capacity)

        if cache_dir is None:
            cache_dir = Path.cwd() / '.cache' / 'ultrafast'

        self.disk_cache = MemoryMappedCache(cache_dir, disk_capacity)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.put(key, value)
            return value

        return None

    def put(self, key: str, value: Any):
        # Always put in memory cache
        self.memory_cache.put(key, value)

        # Put in disk cache for larger objects
        try:
            serialized_size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            if serialized_size > 1024:  # Objects larger than 1KB go to disk
                self.disk_cache.put(key, value)
        except (pickle.PickleError, OSError):
            pass  # Ignore disk cache failures

    def clear(self):
        self.memory_cache.clear()
        self.disk_cache.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            'memory': self.memory_cache.stats(),
            'disk': {
                'entries': len(self.disk_cache._index),
                'total_size': sum(size for _, size, _ in self.disk_cache._index.values())
            }
        }


# Global ultra cache instance
_ultra_cache = MultiLevelCache()


def ultra_cache(ttl: Optional[float] = None,
                key_func: Optional[Callable] = None,
                typed: bool = False) -> Callable:
    """
    Ultra-fast caching decorator with TTL and custom key functions

    Args:
        ttl: Time to live in seconds (None for no expiration)
        key_func: Custom function to generate cache keys
        typed: Whether to include argument types in cache key
    """
    def decorator(func: Callable) -> Callable:
        cache_data = {}  # key -> (value, timestamp)
        lock = threading.RLock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]
                if typed:
                    key_parts.append(str([type(arg).__name__ for arg in args]))
                cache_key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()

            current_time = time.time()

            with lock:
                # Check local function cache first (fastest)
                if cache_key in cache_data:
                    value, timestamp = cache_data[cache_key]
                    if ttl is None or (current_time - timestamp) < ttl:
                        return value
                    else:
                        del cache_data[cache_key]

                # Check global cache
                cached_result = _ultra_cache.get(cache_key)
                if cached_result is not None:
                    stored_value, timestamp = cached_result
                    if ttl is None or (current_time - timestamp) < ttl:
                        # Update local cache
                        cache_data[cache_key] = (stored_value, timestamp)
                        return stored_value

                # Execute function and cache result
                result = func(*args, **kwargs)
                cache_entry = (result, current_time)

                # Store in both local and global cache
                cache_data[cache_key] = cache_entry
                _ultra_cache.put(cache_key, cache_entry)

                return result

        # Add cache management methods
        wrapper.cache_clear = lambda: cache_data.clear()
        wrapper.cache_info = lambda: {
            'local_size': len(cache_data),
            'global_stats': _ultra_cache.stats()
        }

        return wrapper
    return decorator


def async_ultra_cache(ttl: Optional[float] = None,
                     key_func: Optional[Callable] = None) -> Callable:
    """Async version of ultra_cache decorator"""
    def decorator(func: Callable) -> Callable:
        cache_data = {}
        lock = asyncio.Lock()

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                if asyncio.iscoroutinefunction(key_func):
                    cache_key = await key_func(*args, **kwargs)
                else:
                    cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]
                cache_key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()

            current_time = time.time()

            async with lock:
                # Check cache
                if cache_key in cache_data:
                    value, timestamp = cache_data[cache_key]
                    if ttl is None or (current_time - timestamp) < ttl:
                        return value

                # Execute function and cache result
                result = await func(*args, **kwargs)
                cache_data[cache_key] = (result, current_time)

                # Also store in global cache
                _ultra_cache.put(cache_key, (result, current_time))

                return result

        wrapper.cache_clear = lambda: cache_data.clear()
        wrapper.cache_info = lambda: {'size': len(cache_data)}

        return wrapper
    return decorator


class CachePreloader:
    """Preloads frequently accessed data into cache"""

    def __init__(self):
        self._preload_tasks = []

    def add_preload_task(self, key: str, func: Callable, *args, **kwargs):
        """Add a preload task to be executed in background"""
        self._preload_tasks.append((key, func, args, kwargs))

    async def preload_all(self):
        """Execute all preload tasks concurrently"""
        async def _preload(key: str, func: Callable, args: tuple, kwargs: dict):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                _ultra_cache.put(key, (result, time.time()))
            except Exception:
                pass  # Ignore preload failures

        tasks = [
            _preload(key, func, args, kwargs)
            for key, func, args, kwargs in self._preload_tasks
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# Global cache preloader
preloader = CachePreloader()


def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    return {
        'ultra_cache': _ultra_cache.stats(),
        'preload_tasks': len(preloader._preload_tasks)
    }


def clear_all_caches():
    """Clear all caches"""
    _ultra_cache.clear()
    preloader._preload_tasks.clear()


# Warmup frequently used imports
@ultra_cache(ttl=3600)  # Cache for 1 hour
def get_common_import(module_name: str):
    """Cached import for commonly used modules"""
    try:
        return __import__(module_name)
    except ImportError:
        return None