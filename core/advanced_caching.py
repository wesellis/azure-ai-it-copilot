"""
Advanced Caching Strategies
Multi-tier caching with intelligent invalidation and warming
"""

import asyncio
import hashlib
import json
import time
import pickle
import logging
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union, Tuple
from dataclasses import dataclass, field
from functools import wraps
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
import weakref
import threading
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheLevel(str, Enum):
    """Cache level enumeration"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    dependency_keys: List[str] = field(default_factory=list)


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    size: int = 0
    memory_usage: int = 0
    hit_rate: float = 0.0


class CacheStrategy(ABC):
    """Abstract base class for cache strategies"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass


class LRUCache(CacheStrategy):
    """Async LRU cache implementation"""

    def __init__(self, max_size: int = 1000, default_ttl: float = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check if expired
                if entry.ttl and time.time() - entry.created_at > entry.ttl:
                    del self._cache[key]
                    self._stats.expired += 1
                    self._stats.misses += 1
                    return None

                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.last_accessed = time.time()
                entry.access_count += 1

                self._stats.hits += 1
                self._update_hit_rate()
                return entry.value

            self._stats.misses += 1
            self._update_hit_rate()
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        async with self._lock:
            # Calculate size estimate
            size_bytes = len(pickle.dumps(value)) if value is not None else 0

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )

            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Evict if necessary
            while len(self._cache) >= self.max_size:
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._stats.evictions += 1
                self._stats.memory_usage -= oldest_entry.size_bytes

            # Add new entry
            self._cache[key] = entry
            self._stats.size = len(self._cache)
            self._stats.memory_usage += size_bytes

            return True

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._stats.size = len(self._cache)
                self._stats.memory_usage -= entry.size_bytes
                return True
            return False

    async def clear(self) -> bool:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._stats.size = 0
            self._stats.memory_usage = 0
            return True

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._stats

    def _update_hit_rate(self):
        """Update hit rate calculation"""
        total = self._stats.hits + self._stats.misses
        self._stats.hit_rate = (self._stats.hits / total * 100) if total > 0 else 0


class MultiTierCache:
    """Multi-tier cache with L1 (memory), L2 (Redis), L3 (disk) levels"""

    def __init__(self,
                 l1_cache: Optional[CacheStrategy] = None,
                 l2_cache: Optional[CacheStrategy] = None,
                 l3_cache: Optional[CacheStrategy] = None):
        self.l1_cache = l1_cache or LRUCache(max_size=1000, default_ttl=300)
        self.l2_cache = l2_cache  # Redis cache would be implemented here
        self.l3_cache = l3_cache  # Disk cache would be implemented here

        self._cache_warming_tasks: Dict[str, asyncio.Task] = {}
        self._dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self._tag_index: Dict[str, List[str]] = defaultdict(list)

    async def get(self, key: str, promote: bool = True) -> Optional[Any]:
        """Get value from cache with promotion through tiers"""
        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            return value

        # Try L2 cache
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None and promote:
                # Promote to L1
                await self.l1_cache.set(key, value)
                return value

        # Try L3 cache
        if self.l3_cache:
            value = await self.l3_cache.get(key)
            if value is not None and promote:
                # Promote to L1 and L2
                await self.l1_cache.set(key, value)
                if self.l2_cache:
                    await self.l2_cache.set(key, value)
                return value

        return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None,
                  tags: List[str] = None, dependencies: List[str] = None) -> bool:
        """Set value in all cache tiers"""
        # Set in all available cache levels
        tasks = []

        # L1 (always available)
        tasks.append(self.l1_cache.set(key, value, ttl))

        # L2 (if available)
        if self.l2_cache:
            tasks.append(self.l2_cache.set(key, value, ttl))

        # L3 (if available)
        if self.l3_cache:
            tasks.append(self.l3_cache.set(key, value, ttl))

        # Execute all sets concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update tag index
        if tags:
            for tag in tags:
                if key not in self._tag_index[tag]:
                    self._tag_index[tag].append(key)

        # Update dependency graph
        if dependencies:
            for dep_key in dependencies:
                if key not in self._dependency_graph[dep_key]:
                    self._dependency_graph[dep_key].append(key)

        # Return True if at least L1 succeeded
        return not isinstance(results[0], Exception)

    async def delete(self, key: str) -> bool:
        """Delete from all cache tiers"""
        tasks = []

        # Delete from all levels
        tasks.append(self.l1_cache.delete(key))
        if self.l2_cache:
            tasks.append(self.l2_cache.delete(key))
        if self.l3_cache:
            tasks.append(self.l3_cache.delete(key))

        # Delete dependents
        if key in self._dependency_graph:
            for dependent_key in self._dependency_graph[key]:
                tasks.append(self.delete(dependent_key))
            del self._dependency_graph[key]

        # Remove from tag index
        for tag, keys in self._tag_index.items():
            if key in keys:
                keys.remove(key)

        await asyncio.gather(*tasks, return_exceptions=True)
        return True

    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all cache entries with a specific tag"""
        if tag not in self._tag_index:
            return 0

        keys_to_delete = self._tag_index[tag].copy()
        tasks = [self.delete(key) for key in keys_to_delete]
        await asyncio.gather(*tasks, return_exceptions=True)

        del self._tag_index[tag]
        return len(keys_to_delete)

    async def warm_cache(self, key: str, factory_func: Callable[[], Any],
                        ttl: Optional[float] = None, force: bool = False) -> bool:
        """Warm cache with computed value"""
        # Check if already cached and not forcing
        if not force:
            existing = await self.get(key)
            if existing is not None:
                return True

        # Cancel existing warming task for this key
        if key in self._cache_warming_tasks:
            self._cache_warming_tasks[key].cancel()

        # Create warming task
        async def warm_task():
            try:
                if asyncio.iscoroutinefunction(factory_func):
                    value = await factory_func()
                else:
                    value = factory_func()

                await self.set(key, value, ttl)
                return True
            except Exception as e:
                logger.error(f"Cache warming failed for key {key}: {e}")
                return False
            finally:
                self._cache_warming_tasks.pop(key, None)

        self._cache_warming_tasks[key] = asyncio.create_task(warm_task())
        return await self._cache_warming_tasks[key]

    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels"""
        stats = {
            'l1': self.l1_cache.get_stats()
        }

        if self.l2_cache:
            stats['l2'] = self.l2_cache.get_stats()

        if self.l3_cache:
            stats['l3'] = self.l3_cache.get_stats()

        return stats

    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        return {
            'stats': self.get_stats(),
            'dependency_graph_size': len(self._dependency_graph),
            'tag_index_size': len(self._tag_index),
            'warming_tasks': len(self._cache_warming_tasks),
            'total_tags': sum(len(keys) for keys in self._tag_index.values())
        }


class CacheDecorator:
    """Advanced cache decorator with features"""

    def __init__(self, cache: MultiTierCache, ttl: float = 300,
                 key_generator: Optional[Callable] = None,
                 condition: Optional[Callable] = None,
                 tags: Optional[List[str]] = None):
        self.cache = cache
        self.ttl = ttl
        self.key_generator = key_generator or self._default_key_generator
        self.condition = condition
        self.tags = tags or []

    def _default_key_generator(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function signature"""
        func_name = f"{func.__module__}.{func.__name__}"
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        key_data = f"{func_name}:{args_str}:{kwargs_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check condition
            if self.condition and not self.condition(*args, **kwargs):
                return await func(*args, **kwargs)

            # Generate cache key
            cache_key = self.key_generator(func, args, kwargs)

            # Try to get from cache
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await self.cache.set(cache_key, result, self.ttl, tags=self.tags)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle async cache operations
            loop = asyncio.get_event_loop()

            # Check condition
            if self.condition and not self.condition(*args, **kwargs):
                return func(*args, **kwargs)

            # Generate cache key
            cache_key = self.key_generator(func, args, kwargs)

            # Try to get from cache
            cached_result = loop.run_until_complete(self.cache.get(cache_key))
            if cached_result is not None:
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            loop.run_until_complete(
                self.cache.set(cache_key, result, self.ttl, tags=self.tags)
            )

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class CachePreloader:
    """Cache preloading system"""

    def __init__(self, cache: MultiTierCache):
        self.cache = cache
        self._preload_jobs: Dict[str, asyncio.Task] = {}

    async def schedule_preload(self, key: str, factory_func: Callable,
                              schedule_time: float, ttl: Optional[float] = None):
        """Schedule cache preload at specific time"""
        delay = schedule_time - time.time()
        if delay <= 0:
            # Execute immediately
            return await self.cache.warm_cache(key, factory_func, ttl)

        async def preload_task():
            await asyncio.sleep(delay)
            await self.cache.warm_cache(key, factory_func, ttl)

        self._preload_jobs[key] = asyncio.create_task(preload_task())

    async def preload_batch(self, preload_specs: List[Dict[str, Any]],
                           concurrency: int = 10):
        """Preload multiple cache entries concurrently"""
        semaphore = asyncio.Semaphore(concurrency)

        async def preload_one(spec):
            async with semaphore:
                await self.cache.warm_cache(
                    spec['key'],
                    spec['factory'],
                    spec.get('ttl')
                )

        tasks = [preload_one(spec) for spec in preload_specs]
        await asyncio.gather(*tasks, return_exceptions=True)

    def cancel_preload(self, key: str):
        """Cancel scheduled preload"""
        if key in self._preload_jobs:
            self._preload_jobs[key].cancel()
            del self._preload_jobs[key]


# Global cache instance
global_cache = MultiTierCache()


def cached(ttl: float = 300, tags: List[str] = None, condition: Callable = None):
    """Convenient cache decorator"""
    return CacheDecorator(global_cache, ttl=ttl, tags=tags, condition=condition)


def cache_invalidate_tag(tag: str):
    """Invalidate cache by tag"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(global_cache.invalidate_by_tag(tag))


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    return global_cache.get_cache_info()


async def warm_cache_async(key: str, factory_func: Callable, ttl: float = 300):
    """Warm cache asynchronously"""
    return await global_cache.warm_cache(key, factory_func, ttl)


# Example usage and specializations
class DatabaseQueryCache:
    """Specialized cache for database queries"""

    def __init__(self, cache: MultiTierCache):
        self.cache = cache

    @cached(ttl=600, tags=['database'])
    async def get_user_by_id(self, user_id: str):
        """Example cached database query"""
        # This would be replaced with actual database query
        return f"user_data_for_{user_id}"

    async def invalidate_user_cache(self, user_id: str):
        """Invalidate specific user cache"""
        await self.cache.delete(f"user:{user_id}")

    async def invalidate_all_users(self):
        """Invalidate all user cache entries"""
        await self.cache.invalidate_by_tag('database')


class ComputationCache:
    """Cache for expensive computations"""

    def __init__(self, cache: MultiTierCache):
        self.cache = cache

    @cached(ttl=3600, tags=['computation'])
    async def expensive_calculation(self, params: Dict[str, Any]):
        """Example cached expensive calculation"""
        # Simulate expensive computation
        await asyncio.sleep(1)
        return sum(params.values()) * 42

    async def precompute_common_results(self):
        """Precompute common calculation results"""
        common_params = [
            {'a': 1, 'b': 2},
            {'a': 5, 'b': 10},
            {'a': 100, 'b': 200}
        ]

        preload_specs = [
            {
                'key': f"calc:{hash(str(params))}",
                'factory': lambda p=params: self.expensive_calculation(p),
                'ttl': 3600
            }
            for params in common_params
        ]

        preloader = CachePreloader(self.cache)
        await preloader.preload_batch(preload_specs)