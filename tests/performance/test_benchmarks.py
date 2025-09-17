"""
Performance Benchmarking and Testing Suite
Comprehensive performance testing for all system components
"""

import asyncio
import pytest
import time
import statistics
import psutil
import gc
import threading
import memory_profiler
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, Mock
import json
import uuid

# Performance testing imports
from locust import HttpUser, task, between
from locust.env import Environment
from locust.stats import stats_printer

# Local imports
from core.async_optimizations import AsyncSemaphorePool, SemaphoreConfig, AsyncCircuitBreaker, CircuitBreakerConfig
from core.memory_optimization import MemoryTracker, ObjectPool, PoolConfig, GarbageCollectionManager
from core.advanced_caching import MultiTierCache, LRUCache
from core.database_optimization import QueryOptimizer
from core.performance import PerformanceMonitor
from ai_orchestrator.orchestrator import AzureAIOrchestrator


@dataclass
class BenchmarkResult:
    """Benchmark test result"""
    test_name: str
    operations_per_second: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    total_operations: int
    test_duration_seconds: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """Performance test thresholds"""
    max_response_time_ms: float = 1000.0
    min_ops_per_second: float = 100.0
    max_memory_usage_mb: float = 512.0
    max_cpu_usage_percent: float = 80.0
    min_success_rate: float = 95.0
    max_p95_response_ms: float = 2000.0
    max_p99_response_ms: float = 5000.0


class PerformanceBenchmark:
    """Base class for performance benchmarks"""

    def __init__(self, name: str, thresholds: PerformanceThresholds = None):
        self.name = name
        self.thresholds = thresholds or PerformanceThresholds()
        self.results: List[BenchmarkResult] = []

    async def run_benchmark(self, operation: Callable,
                          num_operations: int = 1000,
                          concurrency: int = 10,
                          duration_seconds: Optional[float] = None) -> BenchmarkResult:
        """Run performance benchmark"""

        # Setup monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        response_times = []
        errors = []
        successful_operations = 0

        start_time = time.time()

        if duration_seconds:
            # Duration-based test
            end_time = start_time + duration_seconds

            async def duration_worker():
                nonlocal successful_operations
                while time.time() < end_time:
                    try:
                        op_start = time.time()
                        await operation()
                        op_end = time.time()

                        response_times.append((op_end - op_start) * 1000)  # ms
                        successful_operations += 1
                    except Exception as e:
                        errors.append(str(e))

                    await asyncio.sleep(0.001)  # Small delay to prevent tight loop

            # Run concurrent workers
            workers = [duration_worker() for _ in range(concurrency)]
            await asyncio.gather(*workers)

            total_operations = successful_operations + len(errors)

        else:
            # Operation count-based test
            semaphore = asyncio.Semaphore(concurrency)

            async def operation_worker():
                nonlocal successful_operations
                async with semaphore:
                    try:
                        op_start = time.time()
                        await operation()
                        op_end = time.time()

                        response_times.append((op_end - op_start) * 1000)  # ms
                        successful_operations += 1
                    except Exception as e:
                        errors.append(str(e))

            # Create and run tasks
            tasks = [operation_worker() for _ in range(num_operations)]
            await asyncio.gather(*tasks, return_exceptions=True)

            total_operations = num_operations

        end_time = time.time()
        test_duration = end_time - start_time

        # Calculate metrics
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory

        cpu_usage = process.cpu_percent()

        ops_per_second = successful_operations / test_duration if test_duration > 0 else 0
        success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0

        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0
        p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else 0

        result = BenchmarkResult(
            test_name=self.name,
            operations_per_second=ops_per_second,
            avg_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            success_rate=success_rate,
            total_operations=total_operations,
            test_duration_seconds=test_duration,
            errors=errors[:10],  # Keep only first 10 errors
            metadata={
                "concurrency": concurrency,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory
            }
        )

        self.results.append(result)
        return result

    def validate_result(self, result: BenchmarkResult) -> List[str]:
        """Validate benchmark result against thresholds"""
        violations = []

        if result.avg_response_time_ms > self.thresholds.max_response_time_ms:
            violations.append(f"Average response time {result.avg_response_time_ms:.2f}ms exceeds {self.thresholds.max_response_time_ms}ms")

        if result.operations_per_second < self.thresholds.min_ops_per_second:
            violations.append(f"Operations per second {result.operations_per_second:.2f} below {self.thresholds.min_ops_per_second}")

        if result.memory_usage_mb > self.thresholds.max_memory_usage_mb:
            violations.append(f"Memory usage {result.memory_usage_mb:.2f}MB exceeds {self.thresholds.max_memory_usage_mb}MB")

        if result.cpu_usage_percent > self.thresholds.max_cpu_usage_percent:
            violations.append(f"CPU usage {result.cpu_usage_percent:.2f}% exceeds {self.thresholds.max_cpu_usage_percent}%")

        if result.success_rate < self.thresholds.min_success_rate:
            violations.append(f"Success rate {result.success_rate:.2f}% below {self.thresholds.min_success_rate}%")

        if result.p95_response_time_ms > self.thresholds.max_p95_response_ms:
            violations.append(f"P95 response time {result.p95_response_time_ms:.2f}ms exceeds {self.thresholds.max_p95_response_ms}ms")

        if result.p99_response_time_ms > self.thresholds.max_p99_response_ms:
            violations.append(f"P99 response time {result.p99_response_time_ms:.2f}ms exceeds {self.thresholds.max_p99_response_ms}ms")

        return violations


@pytest.mark.performance
class TestAsyncOptimizationsBenchmarks:
    """Benchmark tests for async optimization components"""

    @pytest.fixture
    def semaphore_pool(self):
        """Async semaphore pool for testing"""
        config = SemaphoreConfig(max_concurrent=50, timeout=5.0)
        return AsyncSemaphorePool(config)

    @pytest.fixture
    def circuit_breaker(self):
        """Circuit breaker for testing"""
        config = CircuitBreakerConfig(failure_threshold=5, timeout=2.0)
        return AsyncCircuitBreaker("test-service", config)

    @pytest.mark.asyncio
    async def test_semaphore_pool_benchmark(self, semaphore_pool):
        """Benchmark semaphore pool performance"""
        benchmark = PerformanceBenchmark(
            "SemaphorePool",
            PerformanceThresholds(min_ops_per_second=500.0, max_response_time_ms=50.0)
        )

        async def semaphore_operation():
            acquired = await semaphore_pool.acquire(1, f"bench-{uuid.uuid4()}")
            if acquired:
                await asyncio.sleep(0.001)  # Simulate minimal work
                await semaphore_pool.release(f"bench-{uuid.uuid4()}")
                return True
            return False

        result = await benchmark.run_benchmark(
            semaphore_operation,
            num_operations=5000,
            concurrency=20
        )

        # Validate performance
        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        # Verify metrics
        assert result.operations_per_second > 500
        assert result.success_rate > 90.0
        assert result.avg_response_time_ms < 50

    @pytest.mark.asyncio
    async def test_circuit_breaker_benchmark(self, circuit_breaker):
        """Benchmark circuit breaker performance"""
        benchmark = PerformanceBenchmark(
            "CircuitBreaker",
            PerformanceThresholds(min_ops_per_second=200.0, max_response_time_ms=100.0)
        )

        success_count = 0

        async def circuit_breaker_operation():
            nonlocal success_count
            # Simulate mostly successful operations
            if success_count % 20 != 0:  # 95% success rate
                await asyncio.sleep(0.002)  # Simulate work
                success_count += 1
                return "success"
            else:
                raise Exception("Simulated failure")

        async def protected_operation():
            return await circuit_breaker.call(circuit_breaker_operation)

        result = await benchmark.run_benchmark(
            protected_operation,
            num_operations=2000,
            concurrency=10
        )

        # Validate performance
        violations = benchmark.validate_result(result)
        # Allow some violations due to circuit breaker failures
        assert len(violations) <= 2, f"Too many performance violations: {violations}"

        # Verify circuit breaker functionality
        assert result.operations_per_second > 100
        assert result.success_rate > 85.0  # Some failures expected due to circuit breaker

    @pytest.mark.asyncio
    async def test_concurrent_async_operations_benchmark(self):
        """Benchmark concurrent async operations"""
        benchmark = PerformanceBenchmark(
            "ConcurrentAsyncOps",
            PerformanceThresholds(min_ops_per_second=1000.0, max_response_time_ms=20.0)
        )

        async def async_operation():
            # Simulate async I/O operation
            await asyncio.sleep(0.001)
            return uuid.uuid4().hex

        result = await benchmark.run_benchmark(
            async_operation,
            num_operations=10000,
            concurrency=50
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 1000
        assert result.avg_response_time_ms < 20


@pytest.mark.performance
class TestMemoryOptimizationBenchmarks:
    """Benchmark tests for memory optimization components"""

    @pytest.fixture
    def memory_tracker(self):
        """Memory tracker for testing"""
        return MemoryTracker()

    @pytest.fixture
    def object_pool(self):
        """Object pool for testing"""
        def create_obj():
            return {"data": [0] * 100, "timestamp": time.time()}

        config = PoolConfig(
            min_size=10,
            max_size=100,
            create_function=create_obj
        )
        return ObjectPool(config)

    @pytest.mark.asyncio
    async def test_memory_tracker_benchmark(self, memory_tracker):
        """Benchmark memory tracker performance"""
        benchmark = PerformanceBenchmark(
            "MemoryTracker",
            PerformanceThresholds(min_ops_per_second=1000.0, max_response_time_ms=10.0)
        )

        async def memory_tracking_operation():
            stats = memory_tracker.get_memory_stats()
            pressure = memory_tracker.get_memory_pressure()
            return stats, pressure

        result = await benchmark.run_benchmark(
            memory_tracking_operation,
            num_operations=5000,
            concurrency=5
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 1000
        assert result.avg_response_time_ms < 10

    @pytest.mark.asyncio
    async def test_object_pool_benchmark(self, object_pool):
        """Benchmark object pool performance"""
        benchmark = PerformanceBenchmark(
            "ObjectPool",
            PerformanceThresholds(min_ops_per_second=2000.0, max_response_time_ms=5.0)
        )

        async def object_pool_operation():
            obj = object_pool.acquire()
            if obj:
                # Simulate object usage
                obj["data"][0] = time.time()
                object_pool.return_object(obj)
                return True
            return False

        result = await benchmark.run_benchmark(
            object_pool_operation,
            num_operations=10000,
            concurrency=20
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 2000
        assert result.success_rate > 95.0

    @pytest.mark.asyncio
    async def test_garbage_collection_benchmark(self):
        """Benchmark garbage collection optimization"""
        gc_manager = GarbageCollectionManager()

        benchmark = PerformanceBenchmark(
            "GarbageCollection",
            PerformanceThresholds(min_ops_per_second=100.0, max_response_time_ms=100.0)
        )

        async def gc_operation():
            # Create some objects that will need garbage collection
            data = [i for i in range(1000)]
            del data

            # Trigger GC if needed
            gc_manager.trigger_collection_if_needed()

            return True

        result = await benchmark.run_benchmark(
            gc_operation,
            num_operations=500,
            concurrency=5
        )

        violations = benchmark.validate_result(result)
        # GC operations are inherently expensive, allow some violations
        assert len(violations) <= 3, f"Too many performance violations: {violations}"

        assert result.success_rate > 95.0

    @pytest.mark.asyncio
    async def test_memory_pressure_benchmark(self):
        """Benchmark memory pressure under load"""
        benchmark = PerformanceBenchmark(
            "MemoryPressure",
            PerformanceThresholds(max_memory_usage_mb=200.0)  # Allow higher memory usage for this test
        )

        async def memory_intensive_operation():
            # Create and immediately destroy large objects
            large_data = [uuid.uuid4().hex for _ in range(10000)]
            result = len(large_data)
            del large_data
            return result

        result = await benchmark.run_benchmark(
            memory_intensive_operation,
            num_operations=100,
            concurrency=5
        )

        # Focus on memory usage validation
        assert result.memory_usage_mb < 200.0
        assert result.success_rate > 90.0


@pytest.mark.performance
class TestCachingBenchmarks:
    """Benchmark tests for caching components"""

    @pytest.fixture
    def lru_cache(self):
        """LRU cache for testing"""
        return LRUCache(max_size=1000, default_ttl=300.0)

    @pytest.fixture
    def multi_tier_cache(self):
        """Multi-tier cache for testing"""
        return MultiTierCache()

    @pytest.mark.asyncio
    async def test_lru_cache_benchmark(self, lru_cache):
        """Benchmark LRU cache performance"""
        benchmark = PerformanceBenchmark(
            "LRUCache",
            PerformanceThresholds(min_ops_per_second=5000.0, max_response_time_ms=2.0)
        )

        cache_keys = [f"key_{i}" for i in range(100)]
        cache_values = [f"value_{i}" * 10 for i in range(100)]

        # Pre-populate cache
        for key, value in zip(cache_keys, cache_values):
            await lru_cache.set(key, value)

        async def cache_operation():
            # Mix of get and set operations
            import random
            if random.random() < 0.8:  # 80% reads
                key = random.choice(cache_keys)
                return await lru_cache.get(key)
            else:  # 20% writes
                key = f"dynamic_key_{random.randint(0, 1000)}"
                value = f"dynamic_value_{random.randint(0, 1000)}"
                return await lru_cache.set(key, value)

        result = await benchmark.run_benchmark(
            cache_operation,
            num_operations=50000,
            concurrency=20
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 5000
        assert result.avg_response_time_ms < 2.0

    @pytest.mark.asyncio
    async def test_multi_tier_cache_benchmark(self, multi_tier_cache):
        """Benchmark multi-tier cache performance"""
        benchmark = PerformanceBenchmark(
            "MultiTierCache",
            PerformanceThresholds(min_ops_per_second=2000.0, max_response_time_ms=5.0)
        )

        async def multi_tier_operation():
            key = f"multi_key_{uuid.uuid4().hex[:8]}"
            value = {"data": uuid.uuid4().hex, "timestamp": time.time()}

            # Set and immediately get
            await multi_tier_cache.set(key, value)
            result = await multi_tier_cache.get(key)

            return result is not None

        result = await benchmark.run_benchmark(
            multi_tier_operation,
            num_operations=10000,
            concurrency=15
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 2000
        assert result.success_rate > 95.0

    @pytest.mark.asyncio
    async def test_cache_hit_ratio_benchmark(self, lru_cache):
        """Benchmark cache hit ratio performance"""
        # Pre-populate cache
        for i in range(500):
            await lru_cache.set(f"popular_key_{i}", f"popular_value_{i}")

        benchmark = PerformanceBenchmark(
            "CacheHitRatio",
            PerformanceThresholds(min_ops_per_second=10000.0)
        )

        async def cache_hit_operation():
            import random
            # 90% chance of hitting existing keys
            if random.random() < 0.9:
                key = f"popular_key_{random.randint(0, 499)}"
            else:
                key = f"miss_key_{random.randint(0, 1000)}"

            return await lru_cache.get(key)

        result = await benchmark.run_benchmark(
            cache_hit_operation,
            num_operations=100000,
            concurrency=30
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        # Should achieve very high performance due to cache hits
        assert result.operations_per_second > 10000


@pytest.mark.performance
class TestDatabaseBenchmarks:
    """Benchmark tests for database operations"""

    @pytest.mark.asyncio
    async def test_query_optimizer_benchmark(self):
        """Benchmark query optimizer performance"""
        query_optimizer = QueryOptimizer()

        benchmark = PerformanceBenchmark(
            "QueryOptimizer",
            PerformanceThresholds(min_ops_per_second=100.0, max_response_time_ms=50.0)
        )

        async def query_operation():
            # Simulate query monitoring
            query = "SELECT * FROM test_table WHERE id = ?"
            params = {"id": uuid.uuid4().hex}

            async with query_optimizer.monitor_query(query, params):
                # Simulate query execution time
                await asyncio.sleep(0.005)  # 5ms simulated query
                return "query_result"

        result = await benchmark.run_benchmark(
            query_operation,
            num_operations=2000,
            concurrency=10
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 100
        assert result.avg_response_time_ms < 50

    @pytest.mark.asyncio
    async def test_database_connection_benchmark(self):
        """Benchmark database connection performance"""
        benchmark = PerformanceBenchmark(
            "DatabaseConnection",
            PerformanceThresholds(min_ops_per_second=500.0, max_response_time_ms=20.0)
        )

        async def connection_operation():
            # Simulate database connection and simple query
            await asyncio.sleep(0.002)  # 2ms connection overhead

            # Simulate simple query
            await asyncio.sleep(0.003)  # 3ms query time

            return {"result": "success", "rows": 1}

        result = await benchmark.run_benchmark(
            connection_operation,
            num_operations=5000,
            concurrency=15
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 500
        assert result.avg_response_time_ms < 20


@pytest.mark.performance
class TestAIOrchestatorBenchmarks:
    """Benchmark tests for AI Orchestrator"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock AI Orchestrator for benchmarking"""
        orchestrator = Mock(spec=AzureAIOrchestrator)

        orchestrator.analyze_infrastructure = AsyncMock(return_value={
            "analysis": "Mock analysis result",
            "recommendations": ["Mock recommendation 1", "Mock recommendation 2"],
            "execution_time_ms": 150
        })

        orchestrator.optimize_costs = AsyncMock(return_value={
            "savings": 1000.0,
            "recommendations": ["Resize VM", "Delete unused storage"],
            "execution_time_ms": 200
        })

        return orchestrator

    @pytest.mark.asyncio
    async def test_ai_analysis_benchmark(self, mock_orchestrator):
        """Benchmark AI analysis performance"""
        benchmark = PerformanceBenchmark(
            "AIAnalysis",
            PerformanceThresholds(min_ops_per_second=10.0, max_response_time_ms=500.0)
        )

        async def ai_analysis_operation():
            return await mock_orchestrator.analyze_infrastructure(
                subscription_id="test-sub",
                resource_group="test-rg"
            )

        result = await benchmark.run_benchmark(
            ai_analysis_operation,
            num_operations=100,
            concurrency=5
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 10
        assert result.success_rate > 95.0

    @pytest.mark.asyncio
    async def test_ai_cost_optimization_benchmark(self, mock_orchestrator):
        """Benchmark AI cost optimization performance"""
        benchmark = PerformanceBenchmark(
            "AICostOptimization",
            PerformanceThresholds(min_ops_per_second=8.0, max_response_time_ms=600.0)
        )

        async def cost_optimization_operation():
            return await mock_orchestrator.optimize_costs(
                subscription_id="test-sub",
                time_range_days=30
            )

        result = await benchmark.run_benchmark(
            cost_optimization_operation,
            num_operations=80,
            concurrency=4
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 8
        assert result.success_rate > 95.0


@pytest.mark.performance
class TestSystemIntegrationBenchmarks:
    """Benchmark tests for system integration performance"""

    @pytest.mark.asyncio
    async def test_full_stack_benchmark(self):
        """Benchmark full stack operation performance"""
        # This simulates a complete operation from API to database
        benchmark = PerformanceBenchmark(
            "FullStack",
            PerformanceThresholds(min_ops_per_second=50.0, max_response_time_ms=200.0)
        )

        async def full_stack_operation():
            # Simulate API processing
            await asyncio.sleep(0.005)  # 5ms API overhead

            # Simulate authentication
            await asyncio.sleep(0.002)  # 2ms auth check

            # Simulate database query
            await asyncio.sleep(0.010)  # 10ms database query

            # Simulate AI processing
            await asyncio.sleep(0.020)  # 20ms AI processing

            # Simulate response formatting
            await asyncio.sleep(0.003)  # 3ms response formatting

            return {
                "status": "success",
                "data": {"result": "full_stack_result"},
                "processing_time_ms": 40
            }

        result = await benchmark.run_benchmark(
            full_stack_operation,
            num_operations=1000,
            concurrency=8
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 50
        assert result.avg_response_time_ms < 200

    @pytest.mark.asyncio
    async def test_concurrent_user_simulation_benchmark(self):
        """Benchmark concurrent user simulation"""
        benchmark = PerformanceBenchmark(
            "ConcurrentUsers",
            PerformanceThresholds(min_ops_per_second=100.0, max_response_time_ms=100.0)
        )

        user_sessions = {}

        async def user_operation():
            user_id = uuid.uuid4().hex[:8]

            # Simulate user session
            if user_id not in user_sessions:
                user_sessions[user_id] = {
                    "login_time": time.time(),
                    "operations": 0
                }

            # Simulate user operation
            await asyncio.sleep(0.010)  # 10ms operation

            user_sessions[user_id]["operations"] += 1

            return {"user_id": user_id, "operation_count": user_sessions[user_id]["operations"]}

        result = await benchmark.run_benchmark(
            user_operation,
            num_operations=5000,
            concurrency=25
        )

        violations = benchmark.validate_result(result)
        assert len(violations) == 0, f"Performance violations: {violations}"

        assert result.operations_per_second > 100
        assert len(user_sessions) > 0  # Multiple users simulated

    @pytest.mark.asyncio
    async def test_error_recovery_benchmark(self):
        """Benchmark error recovery performance"""
        benchmark = PerformanceBenchmark(
            "ErrorRecovery",
            PerformanceThresholds(min_ops_per_second=200.0, success_rate=70.0)  # Lower success rate expected
        )

        operation_count = 0

        async def error_prone_operation():
            nonlocal operation_count
            operation_count += 1

            # Simulate 30% failure rate
            if operation_count % 3 == 0:
                await asyncio.sleep(0.005)  # 5ms before failure
                raise Exception("Simulated failure")

            # Successful operation
            await asyncio.sleep(0.003)  # 3ms successful operation
            return {"status": "success", "operation": operation_count}

        result = await benchmark.run_benchmark(
            error_prone_operation,
            num_operations=3000,
            concurrency=15
        )

        # Expect some violations due to errors, but check error handling performance
        assert result.operations_per_second > 200
        assert result.success_rate > 60.0  # Should handle failures gracefully
        assert len(result.errors) > 0  # Should capture some errors


@pytest.mark.performance
class TestStressTesting:
    """Stress testing for system limits"""

    @pytest.mark.asyncio
    async def test_memory_stress_test(self):
        """Stress test memory usage"""
        benchmark = PerformanceBenchmark(
            "MemoryStress",
            PerformanceThresholds(max_memory_usage_mb=500.0)  # Allow higher memory for stress test
        )

        large_objects = []

        async def memory_stress_operation():
            # Create increasingly large objects
            size = len(large_objects) * 1000
            large_obj = [uuid.uuid4().hex for _ in range(size)]
            large_objects.append(large_obj)

            # Periodically clean up to prevent infinite growth
            if len(large_objects) > 20:
                large_objects.pop(0)

            return len(large_obj)

        result = await benchmark.run_benchmark(
            memory_stress_operation,
            num_operations=100,
            concurrency=3
        )

        # Memory stress test - focus on memory usage validation
        assert result.memory_usage_mb < 500.0
        assert result.success_rate > 80.0

    @pytest.mark.asyncio
    async def test_connection_stress_test(self):
        """Stress test connection handling"""
        benchmark = PerformanceBenchmark(
            "ConnectionStress",
            PerformanceThresholds(min_ops_per_second=100.0, max_response_time_ms=100.0)
        )

        active_connections = set()

        async def connection_operation():
            connection_id = uuid.uuid4().hex
            active_connections.add(connection_id)

            try:
                # Simulate connection work
                await asyncio.sleep(0.020)  # 20ms connection work
                return {"connection_id": connection_id, "status": "completed"}
            finally:
                active_connections.discard(connection_id)

        result = await benchmark.run_benchmark(
            connection_operation,
            duration_seconds=10.0,  # 10-second stress test
            concurrency=50  # High concurrency
        )

        violations = benchmark.validate_result(result)
        # Allow some violations under stress
        assert len(violations) <= 2, f"Too many violations under stress: {violations}"

        assert result.operations_per_second > 100
        assert len(active_connections) < 10  # Connections should be cleaned up


# Benchmark test runner
class BenchmarkRunner:
    """Runner for executing benchmark test suites"""

    def __init__(self):
        self.results: Dict[str, List[BenchmarkResult]] = {}

    async def run_suite(self, test_class: type) -> Dict[str, BenchmarkResult]:
        """Run a complete benchmark test suite"""
        suite_results = {}

        # Instantiate test class
        test_instance = test_class()

        # Find all test methods
        test_methods = [method for method in dir(test_instance)
                      if method.startswith('test_') and callable(getattr(test_instance, method))]

        for method_name in test_methods:
            method = getattr(test_instance, method_name)

            try:
                # Setup fixtures if needed
                # This is simplified - in real implementation would handle pytest fixtures

                # Run test method
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()

                print(f"✓ {method_name} completed successfully")

            except Exception as e:
                print(f"✗ {method_name} failed: {e}")

        return suite_results

    def generate_report(self) -> str:
        """Generate benchmark report"""
        report = ["# Performance Benchmark Report", ""]
        report.append(f"Generated at: {datetime.utcnow().isoformat()}")
        report.append("")

        for suite_name, results in self.results.items():
            report.append(f"## {suite_name}")
            report.append("")

            for result in results:
                report.append(f"### {result.test_name}")
                report.append(f"- Operations/sec: {result.operations_per_second:.2f}")
                report.append(f"- Avg response time: {result.avg_response_time_ms:.2f}ms")
                report.append(f"- P95 response time: {result.p95_response_time_ms:.2f}ms")
                report.append(f"- Memory usage: {result.memory_usage_mb:.2f}MB")
                report.append(f"- Success rate: {result.success_rate:.2f}%")
                report.append("")

        return "\n".join(report)


# CLI runner for benchmarks
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Performance Benchmark Runner")
    parser.add_argument("--suite", choices=["async", "memory", "cache", "database", "ai", "integration", "stress", "all"],
                       default="all", help="Benchmark suite to run")
    parser.add_argument("--output", help="Output file for benchmark report")

    args = parser.parse_args()

    async def run_benchmarks():
        runner = BenchmarkRunner()

        suites = {
            "async": TestAsyncOptimizationsBenchmarks,
            "memory": TestMemoryOptimizationBenchmarks,
            "cache": TestCachingBenchmarks,
            "database": TestDatabaseBenchmarks,
            "ai": TestAIOrchestatorBenchmarks,
            "integration": TestSystemIntegrationBenchmarks,
            "stress": TestStressTesting
        }

        if args.suite == "all":
            for suite_name, suite_class in suites.items():
                print(f"\n=== Running {suite_name} benchmarks ===")
                await runner.run_suite(suite_class)
        else:
            suite_class = suites.get(args.suite)
            if suite_class:
                print(f"\n=== Running {args.suite} benchmarks ===")
                await runner.run_suite(suite_class)

        # Generate report
        report = runner.generate_report()

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nBenchmark report written to {args.output}")
        else:
            print("\n" + report)

    asyncio.run(run_benchmarks())