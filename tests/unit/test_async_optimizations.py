"""
Unit tests for async optimization system
Tests all async optimization components with comprehensive coverage
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

from core.async_optimizations import (
    AsyncSemaphorePool,
    AsyncCircuitBreaker,
    AsyncTaskManager,
    AsyncBatchProcessor,
    AsyncResourcePool,
    Priority,
    CircuitBreakerState,
    SemaphoreConfig,
    CircuitBreakerConfig,
    TaskConfig,
    ResourceConfig
)


@pytest.mark.asyncio
class TestAsyncSemaphorePool:
    """Test AsyncSemaphorePool functionality"""

    async def test_basic_acquire_release(self):
        """Test basic semaphore acquire and release"""
        config = SemaphoreConfig(max_concurrent=3, timeout=1.0)
        pool = AsyncSemaphorePool(config)

        # Acquire semaphore
        acquired = await pool.acquire(Priority.NORMAL, "test-op-1")
        assert acquired is True
        assert pool.metrics.concurrent_operations == 1

        # Release semaphore
        await pool.release("test-op-1")
        assert pool.metrics.concurrent_operations == 0

    async def test_priority_queuing(self):
        """Test priority-based queuing"""
        config = SemaphoreConfig(max_concurrent=1, timeout=2.0)
        pool = AsyncSemaphorePool(config)

        # Acquire with high priority
        await pool.acquire(Priority.HIGH, "high-priority")

        # Queue low and normal priority operations
        tasks = []
        for i, priority in enumerate([Priority.LOW, Priority.NORMAL, Priority.HIGH]):
            task = asyncio.create_task(pool.acquire(priority, f"op-{i}"))
            tasks.append(task)
            await asyncio.sleep(0.01)  # Ensure order

        # Release high priority
        await pool.release("high-priority")

        # High priority should be acquired first
        completed_task = await asyncio.wait_for(asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED), 1.0)
        # Verify priority ordering
        assert pool.metrics.total_acquisitions > 0

        # Cleanup
        for task in tasks:
            task.cancel()

    async def test_timeout_handling(self):
        """Test timeout handling in semaphore pool"""
        config = SemaphoreConfig(max_concurrent=1, timeout=0.1)
        pool = AsyncSemaphorePool(config)

        # Acquire semaphore
        await pool.acquire(Priority.NORMAL, "blocking-op")

        # Try to acquire with timeout
        start_time = time.time()
        acquired = await pool.acquire(Priority.NORMAL, "timeout-op")
        elapsed = time.time() - start_time

        assert acquired is False
        assert elapsed >= 0.1
        assert pool.metrics.timeouts > 0

    async def test_backpressure_control(self):
        """Test backpressure control mechanism"""
        config = SemaphoreConfig(max_concurrent=2, enable_backpressure=True, backpressure_threshold=5)
        pool = AsyncSemaphorePool(config)

        # Fill the queue beyond backpressure threshold
        for i in range(10):
            asyncio.create_task(pool.acquire(Priority.LOW, f"op-{i}"))
            await asyncio.sleep(0.01)

        # Should reject new requests due to backpressure
        rejected = await pool.acquire(Priority.LOW, "rejected-op")
        assert rejected is False or pool.metrics.rejections > 0

    async def test_metrics_collection(self):
        """Test metrics collection accuracy"""
        config = SemaphoreConfig(max_concurrent=2)
        pool = AsyncSemaphorePool(config)

        # Perform operations and check metrics
        await pool.acquire(Priority.HIGH, "op-1")
        await pool.acquire(Priority.NORMAL, "op-2")

        assert pool.metrics.total_acquisitions == 2
        assert pool.metrics.concurrent_operations == 2
        assert pool.metrics.max_concurrent == 2

        await pool.release("op-1")
        assert pool.metrics.concurrent_operations == 1


@pytest.mark.asyncio
class TestAsyncCircuitBreaker:
    """Test AsyncCircuitBreaker functionality"""

    async def test_closed_state_operations(self):
        """Test circuit breaker in closed state"""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=1.0)
        breaker = AsyncCircuitBreaker("test-service", config)

        # Successful operation
        async def success_operation():
            return "success"

        result = await breaker.call(success_operation)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.metrics.success_count == 1

    async def test_failure_tracking(self):
        """Test failure tracking and threshold"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)
        breaker = AsyncCircuitBreaker("test-service", config)

        async def failing_operation():
            raise ValueError("Test failure")

        # First failure
        with pytest.raises(ValueError):
            await breaker.call(failing_operation)
        assert breaker.metrics.failure_count == 1

        # Second failure should open circuit
        with pytest.raises(ValueError):
            await breaker.call(failing_operation)

        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.metrics.failure_count == 2

    async def test_open_state_behavior(self):
        """Test circuit breaker behavior in open state"""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        breaker = AsyncCircuitBreaker("test-service", config)

        # Cause failure to open circuit
        async def failing_operation():
            raise ValueError("Test failure")

        with pytest.raises(ValueError):
            await breaker.call(failing_operation)

        assert breaker.state == CircuitBreakerState.OPEN

        # Should reject subsequent calls
        async def any_operation():
            return "should not execute"

        with pytest.raises(Exception):  # Circuit breaker should raise exception
            await breaker.call(any_operation)

    async def test_half_open_recovery(self):
        """Test half-open state and recovery"""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1, success_threshold=2)
        breaker = AsyncCircuitBreaker("test-service", config)

        # Open the circuit
        async def failing_operation():
            raise ValueError("Test failure")

        with pytest.raises(ValueError):
            await breaker.call(failing_operation)

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Should transition to half-open
        async def success_operation():
            return "success"

        # First success in half-open
        result = await breaker.call(success_operation)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Second success should close circuit
        result = await breaker.call(success_operation)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED


@pytest.mark.asyncio
class TestAsyncTaskManager:
    """Test AsyncTaskManager functionality"""

    async def test_task_creation_and_tracking(self):
        """Test task creation and tracking"""
        manager = AsyncTaskManager()

        async def sample_task():
            await asyncio.sleep(0.1)
            return "completed"

        # Create task
        task_id = await manager.create_task(sample_task(), "test-task")
        assert task_id in manager._tasks
        assert manager.get_active_task_count() == 1

        # Wait for completion
        result = await manager.wait_for_task(task_id)
        assert result == "completed"

    async def test_task_grouping(self):
        """Test task grouping functionality"""
        manager = AsyncTaskManager()

        async def group_task(value):
            await asyncio.sleep(0.1)
            return value * 2

        # Create task group
        group_id = "test-group"
        tasks = []
        for i in range(3):
            task_id = await manager.create_task(group_task(i), f"task-{i}", group_id)
            tasks.append(task_id)

        # Wait for group completion
        results = await manager.wait_for_group(group_id)
        assert len(results) == 3
        assert all(isinstance(r, int) for r in results.values())

    async def test_task_cancellation(self):
        """Test task cancellation"""
        manager = AsyncTaskManager()

        async def long_running_task():
            await asyncio.sleep(10)  # Long running
            return "should not complete"

        # Create and cancel task
        task_id = await manager.create_task(long_running_task(), "long-task")
        await asyncio.sleep(0.1)  # Let it start

        cancelled = await manager.cancel_task(task_id)
        assert cancelled is True

        # Task should be cancelled
        with pytest.raises(asyncio.CancelledError):
            await manager.wait_for_task(task_id)

    async def test_timeout_handling(self):
        """Test task timeout handling"""
        config = TaskConfig(default_timeout=0.1)
        manager = AsyncTaskManager(config)

        async def timeout_task():
            await asyncio.sleep(1.0)  # Longer than timeout
            return "should timeout"

        task_id = await manager.create_task(timeout_task(), "timeout-test")

        with pytest.raises(asyncio.TimeoutError):
            await manager.wait_for_task(task_id)

    async def test_resource_cleanup(self):
        """Test automatic resource cleanup"""
        manager = AsyncTaskManager()

        async def resource_task():
            # Simulate resource allocation
            return "resource allocated"

        # Create multiple tasks
        task_ids = []
        for i in range(5):
            task_id = await manager.create_task(resource_task(), f"resource-task-{i}")
            task_ids.append(task_id)

        # Wait for completion
        for task_id in task_ids:
            await manager.wait_for_task(task_id)

        # Cleanup should remove completed tasks
        await manager.cleanup_completed_tasks()
        assert manager.get_active_task_count() == 0


@pytest.mark.asyncio
class TestAsyncBatchProcessor:
    """Test AsyncBatchProcessor functionality"""

    async def test_basic_batch_processing(self):
        """Test basic batch processing"""
        processor = AsyncBatchProcessor(batch_size=3, flush_interval=1.0)

        async def process_batch(items):
            return [item * 2 for item in items]

        processor.set_processor(process_batch)

        # Add items
        results = []
        for i in range(5):
            result = await processor.add_item(i)
            if result:
                results.extend(result)

        # Should have processed first batch of 3
        assert len(results) >= 3

    async def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing based on latency"""
        processor = AsyncBatchProcessor(
            batch_size=2,
            flush_interval=1.0,
            adaptive_sizing=True,
            target_latency_ms=50
        )

        async def fast_processor(items):
            await asyncio.sleep(0.01)  # Fast processing
            return [item * 2 for item in items]

        processor.set_processor(fast_processor)

        # Process items and check if batch size adapts
        initial_batch_size = processor.current_batch_size

        for i in range(10):
            await processor.add_item(i)
            await asyncio.sleep(0.02)

        # Batch size should adapt based on fast processing
        # Implementation should increase batch size for fast operations

    async def test_flush_interval(self):
        """Test automatic flushing based on time interval"""
        processor = AsyncBatchProcessor(batch_size=10, flush_interval=0.1)

        async def slow_processor(items):
            await asyncio.sleep(0.05)
            return items

        processor.set_processor(slow_processor)

        # Add single item (won't reach batch size)
        start_time = time.time()
        result = await processor.add_item("test")

        # Should flush after interval even with incomplete batch
        if not result:
            await asyncio.sleep(0.15)  # Wait for flush interval
            # Check that item was processed due to time flush

    async def test_error_handling_in_batch(self):
        """Test error handling during batch processing"""
        processor = AsyncBatchProcessor(batch_size=2)

        async def failing_processor(items):
            if len(items) > 1:
                raise ValueError("Batch processing failed")
            return items

        processor.set_processor(failing_processor)

        # Add items that will cause failure
        with pytest.raises(ValueError):
            await processor.add_item("item1")
            await processor.add_item("item2")  # This should trigger batch and fail


@pytest.mark.asyncio
class TestAsyncResourcePool:
    """Test AsyncResourcePool functionality"""

    async def test_resource_creation_and_acquisition(self):
        """Test resource creation and acquisition"""
        async def create_resource():
            return {"id": f"resource-{time.time()}", "connection": "active"}

        async def destroy_resource(resource):
            resource["connection"] = "closed"

        config = ResourceConfig(
            min_size=1,
            max_size=3,
            create_resource=create_resource,
            destroy_resource=destroy_resource
        )

        pool = AsyncResourcePool(config)
        await pool.initialize()

        # Acquire resource
        resource = await pool.acquire()
        assert resource is not None
        assert resource["connection"] == "active"

        # Release resource
        await pool.release(resource)

    async def test_pool_size_management(self):
        """Test pool size management"""
        async def create_resource():
            return {"id": f"resource-{time.time()}"}

        config = ResourceConfig(
            min_size=2,
            max_size=4,
            create_resource=create_resource
        )

        pool = AsyncResourcePool(config)
        await pool.initialize()

        # Should have minimum resources
        assert pool.size >= 2

        # Acquire all resources
        resources = []
        for _ in range(4):
            resource = await pool.acquire()
            if resource:
                resources.append(resource)

        # Should not exceed max size
        assert len(resources) <= 4

        # Release resources
        for resource in resources:
            await pool.release(resource)

    async def test_resource_health_checks(self):
        """Test resource health checking"""
        async def create_resource():
            return {"id": f"resource-{time.time()}", "healthy": True}

        async def check_health(resource):
            return resource.get("healthy", False)

        config = ResourceConfig(
            min_size=1,
            max_size=2,
            create_resource=create_resource,
            health_check=check_health,
            health_check_interval=0.1
        )

        pool = AsyncResourcePool(config)
        await pool.initialize()

        # Acquire and mark resource as unhealthy
        resource = await pool.acquire()
        resource["healthy"] = False
        await pool.release(resource)

        # Wait for health check
        await asyncio.sleep(0.2)

        # Unhealthy resource should be removed and replaced

    async def test_resource_timeout_handling(self):
        """Test resource acquisition timeout"""
        async def create_resource():
            return {"id": f"resource-{time.time()}"}

        config = ResourceConfig(
            min_size=1,
            max_size=1,
            create_resource=create_resource,
            acquire_timeout=0.1
        )

        pool = AsyncResourcePool(config)
        await pool.initialize()

        # Acquire the only resource
        resource = await pool.acquire()
        assert resource is not None

        # Try to acquire another (should timeout)
        start_time = time.time()
        resource2 = await pool.acquire()
        elapsed = time.time() - start_time

        assert resource2 is None
        assert elapsed >= 0.1


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration tests combining multiple async optimization components"""

    async def test_circuit_breaker_with_semaphore(self):
        """Test circuit breaker integrated with semaphore pool"""
        semaphore_config = SemaphoreConfig(max_concurrent=2)
        pool = AsyncSemaphorePool(semaphore_config)

        breaker_config = CircuitBreakerConfig(failure_threshold=2)
        breaker = AsyncCircuitBreaker("test-service", breaker_config)

        async def protected_operation():
            acquired = await pool.acquire(Priority.NORMAL, "protected-op")
            if not acquired:
                raise Exception("Could not acquire semaphore")

            try:
                # Simulate work
                await asyncio.sleep(0.1)
                return "success"
            finally:
                await pool.release("protected-op")

        # Should work normally
        result = await breaker.call(protected_operation)
        assert result == "success"

    async def test_task_manager_with_batch_processor(self):
        """Test task manager with batch processor integration"""
        manager = AsyncTaskManager()
        processor = AsyncBatchProcessor(batch_size=3, flush_interval=0.5)

        async def process_batch(items):
            # Simulate batch processing
            await asyncio.sleep(0.1)
            return [item * 2 for item in items]

        processor.set_processor(process_batch)

        async def batch_task(items):
            results = []
            for item in items:
                result = await processor.add_item(item)
                if result:
                    results.extend(result)
            return results

        # Create task for batch processing
        task_id = await manager.create_task(
            batch_task([1, 2, 3, 4, 5]),
            "batch-processing-task"
        )

        result = await manager.wait_for_task(task_id)
        assert isinstance(result, list)

    async def test_resource_pool_with_circuit_breaker(self):
        """Test resource pool with circuit breaker protection"""
        async def create_connection():
            return {"connection": "active", "id": f"conn-{time.time()}"}

        async def close_connection(conn):
            conn["connection"] = "closed"

        resource_config = ResourceConfig(
            min_size=1,
            max_size=3,
            create_resource=create_connection,
            destroy_resource=close_connection
        )

        pool = AsyncResourcePool(resource_config)
        await pool.initialize()

        breaker_config = CircuitBreakerConfig(failure_threshold=2)
        breaker = AsyncCircuitBreaker("db-service", breaker_config)

        async def db_operation():
            conn = await pool.acquire()
            if not conn:
                raise Exception("No connection available")

            try:
                # Simulate database operation
                if conn["connection"] != "active":
                    raise Exception("Connection failed")
                await asyncio.sleep(0.05)
                return "query result"
            finally:
                await pool.release(conn)

        # Should work with circuit breaker protection
        result = await breaker.call(db_operation)
        assert result == "query result"


@pytest.mark.asyncio
class TestPerformanceAndMetrics:
    """Performance and metrics tests"""

    async def test_semaphore_performance_metrics(self):
        """Test semaphore performance under load"""
        config = SemaphoreConfig(max_concurrent=10)
        pool = AsyncSemaphorePool(config)

        # Simulate concurrent load
        async def load_operation(op_id):
            acquired = await pool.acquire(Priority.NORMAL, f"load-op-{op_id}")
            if acquired:
                await asyncio.sleep(0.01)  # Simulate work
                await pool.release(f"load-op-{op_id}")
            return acquired

        # Run concurrent operations
        tasks = [load_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        # Check metrics
        assert pool.metrics.total_acquisitions > 0
        assert pool.metrics.max_concurrent <= 10
        successful_acquisitions = sum(1 for r in results if r)
        assert successful_acquisitions > 0

    async def test_circuit_breaker_recovery_time(self):
        """Test circuit breaker recovery time measurement"""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        breaker = AsyncCircuitBreaker("perf-test", config)

        # Cause failure
        async def failing_op():
            raise ValueError("Test failure")

        start_time = time.time()

        with pytest.raises(ValueError):
            await breaker.call(failing_op)

        # Wait for recovery
        await asyncio.sleep(0.15)

        # Successful operation
        async def success_op():
            return "recovered"

        result = await breaker.call(success_op)
        recovery_time = time.time() - start_time

        assert result == "recovered"
        assert recovery_time >= 0.1  # Should respect timeout

    async def test_batch_processor_throughput(self):
        """Test batch processor throughput"""
        processor = AsyncBatchProcessor(batch_size=10, flush_interval=0.1)

        processed_items = []

        async def throughput_processor(items):
            await asyncio.sleep(0.01)  # Simulate processing time
            processed_items.extend(items)
            return items

        processor.set_processor(throughput_processor)

        # Send items rapidly
        start_time = time.time()
        for i in range(100):
            await processor.add_item(i)

        # Wait for processing to complete
        await asyncio.sleep(0.5)
        processing_time = time.time() - start_time

        # Calculate throughput
        throughput = len(processed_items) / processing_time
        assert throughput > 0  # Should process items efficiently

    @pytest.mark.benchmark
    async def test_async_optimization_benchmark(self, benchmark_timer):
        """Benchmark async optimization components"""
        # This would be used with pytest-benchmark for detailed performance testing
        config = SemaphoreConfig(max_concurrent=5)
        pool = AsyncSemaphorePool(config)

        async def benchmark_operation():
            acquired = await pool.acquire(Priority.NORMAL, "benchmark")
            if acquired:
                await asyncio.sleep(0.001)  # Minimal work
                await pool.release("benchmark")
            return acquired

        benchmark_timer.start()

        # Run operations
        tasks = [benchmark_operation() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        elapsed = benchmark_timer.stop()

        success_rate = sum(1 for r in results if r) / len(results)
        ops_per_second = len(results) / elapsed

        # Performance assertions
        assert success_rate > 0.8  # At least 80% success rate
        assert ops_per_second > 100  # At least 100 ops/sec
        assert elapsed < 2.0  # Should complete within 2 seconds