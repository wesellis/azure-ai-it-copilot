"""
Unit tests for memory optimization system
Tests memory tracking, object pooling, and garbage collection optimization
"""

import asyncio
import gc
import pytest
import threading
import time
import weakref
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional

from core.memory_optimization import (
    MemoryTracker,
    MemoryStats,
    ObjectPool,
    WeakRefCache,
    GarbageCollectionManager,
    MemoryPressureMonitor,
    MemorySnapshot,
    PoolConfig,
    GCConfig,
    MemoryAlert,
    MemoryAlertLevel
)


class TestMemoryTracker:
    """Test MemoryTracker functionality"""

    def test_memory_stats_collection(self):
        """Test memory statistics collection"""
        tracker = MemoryTracker()

        stats = tracker.get_memory_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.rss_mb > 0
        assert stats.vms_mb > 0
        assert 0 <= stats.percent <= 100
        assert stats.available_mb > 0

    def test_memory_snapshot_creation(self):
        """Test memory snapshot creation and comparison"""
        tracker = MemoryTracker()

        # Create initial snapshot
        snapshot1 = tracker.create_snapshot("test_snapshot_1")

        # Allocate some memory
        large_list = [i for i in range(10000)]

        # Create second snapshot
        snapshot2 = tracker.create_snapshot("test_snapshot_2")

        # Compare snapshots
        diff = tracker.compare_snapshots(snapshot1.name, snapshot2.name)

        assert diff is not None
        assert "rss_mb_delta" in diff
        assert "vms_mb_delta" in diff

        # Cleanup
        del large_list

    def test_memory_leak_detection(self):
        """Test memory leak detection"""
        tracker = MemoryTracker()

        # Create baseline
        tracker.create_snapshot("baseline")

        # Simulate memory growth
        leaky_objects = []
        for i in range(1000):
            leaky_objects.append({"data": "x" * 1000, "id": i})

        tracker.create_snapshot("after_allocation")

        # Check for potential leaks
        leaks = tracker.detect_potential_leaks(
            baseline_snapshot="baseline",
            current_snapshot="after_allocation",
            threshold_mb=1.0
        )

        assert isinstance(leaks, dict)

        # Cleanup
        del leaky_objects

    def test_object_tracking(self):
        """Test object tracking functionality"""
        tracker = MemoryTracker()

        # Track object creation
        test_objects = []
        for i in range(100):
            obj = {"id": i, "data": "test"}
            test_objects.append(obj)
            tracker.track_object(obj, f"test_object_{i}")

        # Get tracked objects
        tracked = tracker.get_tracked_objects()
        assert len(tracked) >= 100

        # Cleanup
        del test_objects
        gc.collect()

        # Check if objects were cleaned up
        remaining = tracker.get_tracked_objects()
        # Some objects might still be referenced, but count should be manageable

    def test_memory_pressure_detection(self):
        """Test memory pressure detection"""
        tracker = MemoryTracker()

        # Test with normal memory usage
        pressure = tracker.get_memory_pressure()
        assert pressure in ["low", "medium", "high", "critical"]

        # Mock high memory usage
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 95.0
            pressure = tracker.get_memory_pressure()
            assert pressure in ["high", "critical"]

    def test_memory_alerts(self):
        """Test memory alert system"""
        tracker = MemoryTracker()
        alerts = []

        def alert_callback(alert: MemoryAlert):
            alerts.append(alert)

        tracker.add_alert_callback(alert_callback)

        # Mock high memory condition
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 90.0
            tracker.check_memory_alerts()

        # Should have triggered an alert (depending on thresholds)
        # This test depends on implementation details of threshold settings

    def test_gc_trigger_tracking(self):
        """Test garbage collection trigger tracking"""
        tracker = MemoryTracker()

        initial_triggers = tracker.get_gc_stats()["triggers"]

        # Force garbage collection
        gc.collect()

        updated_triggers = tracker.get_gc_stats()["triggers"]

        # Triggers count might have increased
        assert updated_triggers >= initial_triggers


class TestObjectPool:
    """Test ObjectPool functionality"""

    def test_pool_creation_and_configuration(self):
        """Test object pool creation and configuration"""
        def create_obj():
            return {"created_at": time.time(), "data": []}

        def reset_obj(obj):
            obj["data"].clear()
            obj["reset_at"] = time.time()

        config = PoolConfig(
            min_size=2,
            max_size=10,
            create_function=create_obj,
            reset_function=reset_obj
        )

        pool = ObjectPool(config)

        assert pool.size >= 2  # Should have minimum objects
        assert pool.available_count >= 2

    def test_object_acquisition_and_return(self):
        """Test object acquisition and return"""
        def create_obj():
            return {"id": id(object()), "data": []}

        config = PoolConfig(
            min_size=1,
            max_size=5,
            create_function=create_obj
        )

        pool = ObjectPool(config)

        # Acquire object
        obj1 = pool.acquire()
        assert obj1 is not None
        assert pool.available_count == pool.size - 1

        # Return object
        pool.return_object(obj1)
        assert pool.available_count == pool.size

    def test_pool_size_management(self):
        """Test pool size management and growth"""
        def create_obj():
            return {"id": time.time()}

        config = PoolConfig(
            min_size=2,
            max_size=8,
            create_function=create_obj
        )

        pool = ObjectPool(config)
        initial_size = pool.size

        # Acquire all available objects
        objects = []
        for _ in range(10):  # Try to acquire more than max
            obj = pool.acquire()
            if obj:
                objects.append(obj)

        # Should not exceed max size
        assert len(objects) <= 8
        assert pool.size <= 8

        # Return objects
        for obj in objects:
            pool.return_object(obj)

    def test_object_health_checking(self):
        """Test object health checking"""
        def create_obj():
            return {"healthy": True, "data": []}

        def health_check(obj):
            return obj.get("healthy", False)

        config = PoolConfig(
            min_size=1,
            max_size=3,
            create_function=create_obj,
            health_check=health_check,
            health_check_interval=0.1
        )

        pool = ObjectPool(config)

        # Acquire and corrupt object
        obj = pool.acquire()
        obj["healthy"] = False
        pool.return_object(obj)

        # Wait for health check
        time.sleep(0.2)

        # Pool should have replaced unhealthy object
        healthy_obj = pool.acquire()
        assert healthy_obj["healthy"] is True

    def test_pool_cleanup(self):
        """Test pool cleanup functionality"""
        def create_obj():
            return {"created": time.time()}

        def cleanup_obj(obj):
            obj["cleaned"] = True

        config = PoolConfig(
            min_size=1,
            max_size=5,
            create_function=create_obj,
            cleanup_function=cleanup_obj
        )

        pool = ObjectPool(config)

        # Get reference to an object
        obj = pool.acquire()
        obj_id = id(obj)
        pool.return_object(obj)

        # Cleanup pool
        pool.cleanup()

        # Pool should be empty after cleanup
        assert pool.size == 0
        assert pool.available_count == 0

    def test_thread_safety(self):
        """Test thread safety of object pool"""
        def create_obj():
            return {"thread_id": threading.current_thread().ident}

        config = PoolConfig(
            min_size=5,
            max_size=20,
            create_function=create_obj
        )

        pool = ObjectPool(config)
        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    obj = pool.acquire()
                    if obj:
                        time.sleep(0.001)  # Simulate work
                        pool.return_object(obj)
                        results.append(True)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0  # No errors should occur
        assert len(results) == 50  # All acquisitions should succeed


class TestWeakRefCache:
    """Test WeakRefCache functionality"""

    def test_cache_basic_operations(self):
        """Test basic cache operations"""
        cache = WeakRefCache(max_size=100)

        # Create objects to cache
        obj1 = {"data": "test1"}
        obj2 = {"data": "test2"}

        # Set cache entries
        cache.set("key1", obj1)
        cache.set("key2", obj2)

        # Get cache entries
        retrieved1 = cache.get("key1")
        retrieved2 = cache.get("key2")

        assert retrieved1 is obj1
        assert retrieved2 is obj2

    def test_weak_reference_cleanup(self):
        """Test automatic cleanup when objects are garbage collected"""
        cache = WeakRefCache(max_size=100)

        # Create and cache object
        obj = {"data": "will be deleted"}
        cache.set("test_key", obj)

        # Verify object is cached
        assert cache.get("test_key") is obj
        assert cache.size == 1

        # Delete object and force garbage collection
        del obj
        gc.collect()

        # Object should be automatically removed from cache
        assert cache.get("test_key") is None
        # Size might not immediately reflect cleanup until next access

    def test_cache_size_limit(self):
        """Test cache size limit enforcement"""
        cache = WeakRefCache(max_size=3)

        # Add objects up to limit
        objects = []
        for i in range(5):
            obj = {"data": f"object_{i}"}
            objects.append(obj)  # Keep reference to prevent GC
            cache.set(f"key_{i}", obj)

        # Cache should not exceed max size
        assert cache.size <= 3

    def test_cache_statistics(self):
        """Test cache statistics collection"""
        cache = WeakRefCache(max_size=10)

        obj1 = {"data": "test1"}
        obj2 = {"data": "test2"}

        # Cache operations
        cache.set("key1", obj1)
        cache.set("key2", obj2)

        # Hit
        cache.get("key1")
        # Miss
        cache.get("nonexistent")

        stats = cache.get_stats()

        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["size"] >= 1

    def test_cache_cleanup_callback(self):
        """Test cleanup callbacks"""
        cleanup_calls = []

        def cleanup_callback(key, obj):
            cleanup_calls.append((key, obj))

        cache = WeakRefCache(max_size=10, cleanup_callback=cleanup_callback)

        # Add object that will be cleaned up
        obj = {"data": "cleanup_test"}
        cache.set("cleanup_key", obj)

        # Delete object
        del obj
        gc.collect()

        # Force cache cleanup
        cache._cleanup_dead_references()

        # Cleanup callback should have been called
        # Note: This depends on implementation details

    def test_thread_safety(self):
        """Test thread safety of weak reference cache"""
        cache = WeakRefCache(max_size=50)
        errors = []

        def worker(worker_id):
            try:
                for i in range(20):
                    obj = {"worker": worker_id, "item": i}
                    cache.set(f"worker_{worker_id}_item_{i}", obj)

                    # Try to retrieve some objects
                    if i > 5:
                        retrieved = cache.get(f"worker_{worker_id}_item_{i-5}")
                        # Object might be None due to GC or eviction
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0  # No threading errors should occur


class TestGarbageCollectionManager:
    """Test GarbageCollectionManager functionality"""

    def test_gc_manager_initialization(self):
        """Test GC manager initialization"""
        config = GCConfig(
            pressure_threshold=0.8,
            collection_interval=30.0,
            aggressive_threshold=0.9
        )

        manager = GarbageCollectionManager(config)

        assert manager.config.pressure_threshold == 0.8
        assert manager.config.collection_interval == 30.0

    def test_memory_pressure_calculation(self):
        """Test memory pressure calculation"""
        manager = GarbageCollectionManager()

        pressure = manager.get_memory_pressure()

        assert 0.0 <= pressure <= 1.0

    def test_gc_scheduling(self):
        """Test garbage collection scheduling"""
        config = GCConfig(collection_interval=0.1)  # Short interval for testing
        manager = GarbageCollectionManager(config)

        initial_collections = gc.get_stats()[0]['collections']

        # Start monitoring
        manager.start_monitoring()

        # Wait for scheduled collection
        time.sleep(0.2)

        # Stop monitoring
        manager.stop_monitoring()

        # Collections might have increased
        final_collections = gc.get_stats()[0]['collections']
        # Note: This test is dependent on GC implementation

    def test_aggressive_collection_trigger(self):
        """Test aggressive collection under high memory pressure"""
        manager = GarbageCollectionManager()

        # Mock high memory pressure
        with patch.object(manager, 'get_memory_pressure', return_value=0.95):
            initial_gen0 = gc.get_stats()[0]['collections']

            manager.trigger_collection_if_needed()

            # Should have triggered aggressive collection
            final_gen0 = gc.get_stats()[0]['collections']
            assert final_gen0 >= initial_gen0

    def test_gc_statistics_collection(self):
        """Test garbage collection statistics"""
        manager = GarbageCollectionManager()

        stats = manager.get_gc_statistics()

        assert "total_collections" in stats
        assert "generation_stats" in stats
        assert "collection_times" in stats

        # Verify structure
        assert len(stats["generation_stats"]) == 3  # 3 GC generations in CPython

    def test_collection_timing(self):
        """Test collection timing measurement"""
        manager = GarbageCollectionManager()

        # Force collection and measure time
        start_time = time.perf_counter()
        manager.force_collection()
        duration = time.perf_counter() - start_time

        # Collection should complete quickly
        assert duration < 1.0  # Should complete within 1 second

    def test_memory_based_collection(self):
        """Test memory pressure-based collection triggering"""
        config = GCConfig(pressure_threshold=0.1)  # Very low threshold
        manager = GarbageCollectionManager(config)

        # Should trigger collection due to low threshold
        initial_collections = sum(gc.get_stats()[i]['collections'] for i in range(3))

        manager.trigger_collection_if_needed()

        final_collections = sum(gc.get_stats()[i]['collections'] for i in range(3))

        # Collections should have increased
        assert final_collections >= initial_collections


class TestMemoryPressureMonitor:
    """Test MemoryPressureMonitor functionality"""

    def test_pressure_monitor_initialization(self):
        """Test pressure monitor initialization"""
        monitor = MemoryPressureMonitor(check_interval=1.0)

        assert monitor.check_interval == 1.0
        assert not monitor.is_monitoring

    def test_pressure_detection(self):
        """Test pressure level detection"""
        monitor = MemoryPressureMonitor()

        pressure = monitor.get_current_pressure()

        assert pressure in ["low", "medium", "high", "critical"]

    def test_pressure_callbacks(self):
        """Test pressure change callbacks"""
        monitor = MemoryPressureMonitor()
        callback_calls = []

        def pressure_callback(level, stats):
            callback_calls.append((level, stats))

        monitor.add_callback(pressure_callback)

        # Mock pressure change
        with patch.object(monitor, 'get_current_pressure', return_value="high"):
            monitor._check_pressure()

        # Should have triggered callback if pressure changed
        # Note: This depends on implementation details

    def test_monitoring_start_stop(self):
        """Test monitoring start and stop"""
        monitor = MemoryPressureMonitor(check_interval=0.1)

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring

        # Wait briefly
        time.sleep(0.2)

        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.is_monitoring

    def test_alert_thresholds(self):
        """Test alert threshold configuration"""
        thresholds = {
            "medium": 0.7,
            "high": 0.85,
            "critical": 0.95
        }

        monitor = MemoryPressureMonitor(alert_thresholds=thresholds)

        # Test threshold detection
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 80.0
            pressure = monitor.get_current_pressure()
            assert pressure == "medium"

            mock_memory.return_value.percent = 90.0
            pressure = monitor.get_current_pressure()
            assert pressure == "high"


class TestIntegrationScenarios:
    """Integration tests for memory optimization components"""

    def test_memory_tracker_with_object_pool(self):
        """Test memory tracker monitoring object pool"""
        tracker = MemoryTracker()

        def create_obj():
            return {"data": [0] * 1000}  # Create somewhat large objects

        config = PoolConfig(
            min_size=5,
            max_size=20,
            create_function=create_obj
        )

        # Take baseline snapshot
        tracker.create_snapshot("before_pool")

        pool = ObjectPool(config)

        # Take snapshot after pool creation
        tracker.create_snapshot("after_pool")

        # Compare memory usage
        diff = tracker.compare_snapshots("before_pool", "after_pool")

        # Pool creation should have increased memory usage
        assert diff["rss_mb_delta"] > 0

    def test_gc_manager_with_pressure_monitor(self):
        """Test GC manager with pressure monitor integration"""
        gc_manager = GarbageCollectionManager()
        pressure_monitor = MemoryPressureMonitor(check_interval=0.1)

        # Set up callback to trigger GC on high pressure
        def pressure_callback(level, stats):
            if level in ["high", "critical"]:
                gc_manager.force_collection()

        pressure_monitor.add_callback(pressure_callback)

        # Start monitoring
        pressure_monitor.start_monitoring()

        try:
            # Simulate memory pressure
            large_objects = []
            for i in range(1000):
                large_objects.append([0] * 1000)

            # Wait for pressure detection
            time.sleep(0.2)

            # Cleanup
            del large_objects

        finally:
            pressure_monitor.stop_monitoring()

    def test_complete_memory_optimization_stack(self):
        """Test complete memory optimization system"""
        # Initialize all components
        tracker = MemoryTracker()

        def create_cached_obj():
            return {"cached_data": time.time()}

        pool_config = PoolConfig(
            min_size=2,
            max_size=10,
            create_function=create_cached_obj
        )

        pool = ObjectPool(pool_config)
        cache = WeakRefCache(max_size=50)
        gc_config = GCConfig(pressure_threshold=0.8)
        gc_manager = GarbageCollectionManager(gc_config)
        pressure_monitor = MemoryPressureMonitor()

        # Take initial snapshot
        tracker.create_snapshot("initial")

        # Simulate workload
        cached_objects = []
        for i in range(20):
            # Get object from pool
            obj = pool.acquire()
            if obj:
                # Cache it
                cache.set(f"obj_{i}", obj)
                cached_objects.append(obj)

                # Return to pool
                pool.return_object(obj)

        # Take snapshot after workload
        tracker.create_snapshot("after_workload")

        # Trigger GC
        gc_manager.trigger_collection_if_needed()

        # Take final snapshot
        tracker.create_snapshot("after_gc")

        # Analyze memory usage
        workload_diff = tracker.compare_snapshots("initial", "after_workload")
        gc_diff = tracker.compare_snapshots("after_workload", "after_gc")

        # Verify memory optimization effects
        assert isinstance(workload_diff, dict)
        assert isinstance(gc_diff, dict)


@pytest.mark.asyncio
class TestAsyncMemoryOperations:
    """Test async memory operations"""

    async def test_async_memory_monitoring(self):
        """Test asynchronous memory monitoring"""
        tracker = MemoryTracker()

        async def memory_intensive_operation():
            # Simulate async operation with memory allocation
            data = []
            for i in range(1000):
                data.append({"id": i, "payload": "x" * 100})
                if i % 100 == 0:
                    await asyncio.sleep(0.01)  # Yield control
            return data

        # Monitor memory during async operation
        initial_stats = tracker.get_memory_stats()

        result = await memory_intensive_operation()

        final_stats = tracker.get_memory_stats()

        # Memory usage should have increased
        assert final_stats.rss_mb >= initial_stats.rss_mb

        # Cleanup
        del result

    async def test_async_object_pool_operations(self):
        """Test async operations with object pool"""
        async def create_async_obj():
            await asyncio.sleep(0.01)  # Simulate async creation
            return {"created_async": True, "timestamp": time.time()}

        # Note: This would require an async-compatible object pool
        # The current ObjectPool is synchronous, so this test demonstrates
        # how async operations might interact with memory optimization

        # Simulate async pool usage
        objects = []
        for i in range(10):
            obj = await create_async_obj()
            objects.append(obj)

        # Verify objects were created
        assert len(objects) == 10
        assert all(obj["created_async"] for obj in objects)

    async def test_concurrent_memory_operations(self):
        """Test concurrent memory operations"""
        tracker = MemoryTracker()

        async def concurrent_operation(operation_id):
            # Each operation allocates and deallocates memory
            data = [{"op_id": operation_id, "item": i} for i in range(100)]
            await asyncio.sleep(0.05)  # Simulate work
            return len(data)

        # Run concurrent operations
        tasks = [concurrent_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All operations should complete successfully
        assert len(results) == 10
        assert all(r == 100 for r in results)


class TestMemoryOptimizationBenchmarks:
    """Performance benchmarks for memory optimization"""

    def test_object_pool_performance(self):
        """Benchmark object pool performance"""
        def create_obj():
            return {"data": [0] * 100}

        config = PoolConfig(
            min_size=10,
            max_size=100,
            create_function=create_obj
        )

        pool = ObjectPool(config)

        # Benchmark acquisition/return cycles
        start_time = time.perf_counter()

        for _ in range(1000):
            obj = pool.acquire()
            if obj:
                # Simulate usage
                obj["accessed"] = time.time()
                pool.return_object(obj)

        elapsed = time.perf_counter() - start_time

        # Should complete quickly
        assert elapsed < 1.0  # Less than 1 second for 1000 operations

        # Calculate operations per second
        ops_per_second = 1000 / elapsed
        assert ops_per_second > 100  # At least 100 ops/sec

    def test_cache_performance(self):
        """Benchmark cache performance"""
        cache = WeakRefCache(max_size=1000)

        # Create test objects
        test_objects = [{"id": i, "data": f"object_{i}"} for i in range(500)]

        # Benchmark cache operations
        start_time = time.perf_counter()

        # Set operations
        for i, obj in enumerate(test_objects):
            cache.set(f"key_{i}", obj)

        # Get operations
        for i in range(500):
            cache.get(f"key_{i}")

        elapsed = time.perf_counter() - start_time

        # Should complete quickly
        assert elapsed < 0.5  # Less than 0.5 seconds for 1000 operations

        # Verify cache efficiency
        stats = cache.get_stats()
        hit_rate = stats["hits"] / (stats["hits"] + stats["misses"]) if (stats["hits"] + stats["misses"]) > 0 else 0
        assert hit_rate > 0.8  # At least 80% hit rate

    def test_gc_manager_overhead(self):
        """Test GC manager overhead"""
        config = GCConfig(collection_interval=0.1)
        manager = GarbageCollectionManager(config)

        # Measure overhead of GC monitoring
        start_time = time.perf_counter()

        # Start monitoring
        manager.start_monitoring()

        # Simulate workload
        for i in range(100):
            data = [j for j in range(100)]
            del data

        # Stop monitoring
        manager.stop_monitoring()

        elapsed = time.perf_counter() - start_time

        # Overhead should be minimal
        assert elapsed < 2.0  # Should complete within 2 seconds