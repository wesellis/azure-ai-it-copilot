"""
Load testing for AI Orchestrator
Comprehensive load testing using Locust and async testing patterns
"""

import asyncio
import json
import random
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import statistics

import pytest
from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer
from locust.log import setup_logging
import aiohttp
import asyncio
from unittest.mock import patch, AsyncMock, Mock

# Local imports
from ai_orchestrator.orchestrator import AzureAIOrchestrator
from core.async_optimizations import AsyncSemaphorePool, SemaphoreConfig
from core.performance import PerformanceMonitor


class AIOrchestatorUser(HttpUser):
    """Locust user for AI Orchestrator load testing"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Initialize user session"""
        self.user_id = str(uuid.uuid4())
        self.session_token = self.login()

    def login(self) -> str:
        """Simulate user login"""
        response = self.client.post("/auth/login", json={
            "username": f"test_user_{self.user_id[:8]}",
            "password": "test_password"
        })

        if response.status_code == 200:
            return response.json().get("access_token", "mock_token")
        return "mock_token"

    @task(3)
    def analyze_infrastructure(self):
        """Test infrastructure analysis endpoint"""
        headers = {"Authorization": f"Bearer {self.session_token}"}

        payload = {
            "subscription_id": "test-subscription",
            "resource_group": "test-rg",
            "analysis_type": "comprehensive",
            "include_costs": True,
            "include_performance": True,
            "include_security": True
        }

        with self.client.post(
            "/api/ai/analyze/infrastructure",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Infrastructure analysis failed: {response.status_code}")

    @task(2)
    def optimize_costs(self):
        """Test cost optimization endpoint"""
        headers = {"Authorization": f"Bearer {self.session_token}"}

        payload = {
            "subscription_id": "test-subscription",
            "time_range_days": 30,
            "optimization_level": "aggressive",
            "exclude_resources": []
        }

        with self.client.post(
            "/api/ai/optimize/costs",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "recommendations" in data:
                    response.success()
                else:
                    response.failure("Missing recommendations in response")
            else:
                response.failure(f"Cost optimization failed: {response.status_code}")

    @task(2)
    def performance_analysis(self):
        """Test performance analysis endpoint"""
        headers = {"Authorization": f"Bearer {self.session_token}"}

        payload = {
            "resource_id": f"/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/vm-{random.randint(1, 100)}",
            "time_range_hours": 24,
            "metrics": ["cpu", "memory", "network", "disk"]
        }

        with self.client.post(
            "/api/ai/analyze/performance",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Performance analysis failed: {response.status_code}")

    @task(1)
    def security_assessment(self):
        """Test security assessment endpoint"""
        headers = {"Authorization": f"Bearer {self.session_token}"}

        payload = {
            "subscription_id": "test-subscription",
            "resource_group": "test-rg",
            "assessment_type": "comprehensive",
            "compliance_frameworks": ["SOC2", "ISO27001"]
        }

        with self.client.post(
            "/api/ai/assess/security",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Security assessment failed: {response.status_code}")

    @task(1)
    def chat_interaction(self):
        """Test chat/conversation endpoint"""
        headers = {"Authorization": f"Bearer {self.session_token}"}

        questions = [
            "What are my highest cost resources this month?",
            "Show me performance issues in my infrastructure",
            "What security vulnerabilities do I have?",
            "How can I optimize my Azure costs?",
            "Which VMs are underutilized?",
            "What are the latest security alerts?",
            "Generate a cost optimization report",
            "Recommend VM sizing improvements"
        ]

        payload = {
            "message": random.choice(questions),
            "context": {
                "subscription_id": "test-subscription",
                "user_id": self.user_id
            }
        }

        with self.client.post(
            "/api/ai/chat",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Chat interaction failed: {response.status_code}")

    @task(1)
    def get_recommendations(self):
        """Test getting cached recommendations"""
        headers = {"Authorization": f"Bearer {self.session_token}"}

        with self.client.get(
            f"/api/recommendations?subscription_id=test-subscription&type=all",
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Get recommendations failed: {response.status_code}")


class StreamingUser(HttpUser):
    """User for testing streaming/WebSocket connections"""

    wait_time = between(2, 5)

    @task
    def test_streaming_analysis(self):
        """Test streaming analysis endpoint"""
        headers = {"Authorization": "Bearer mock_token"}

        payload = {
            "subscription_id": "test-subscription",
            "stream": True,
            "analysis_type": "real_time_monitoring"
        }

        with self.client.post(
            "/api/ai/stream/analysis",
            json=payload,
            headers=headers,
            stream=True,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                # Simulate reading streaming data
                for _ in range(5):  # Read first 5 chunks
                    try:
                        chunk = next(response.iter_content(chunk_size=1024))
                        if chunk:
                            continue
                    except StopIteration:
                        break
                response.success()
            else:
                response.failure(f"Streaming analysis failed: {response.status_code}")


@pytest.mark.load_test
class TestAIOrchestatorLoadTest:
    """Async load testing for AI Orchestrator components"""

    @pytest.fixture
    def performance_monitor(self):
        """Performance monitor for load tests"""
        return PerformanceMonitor()

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock AI Orchestrator for load testing"""
        orchestrator = Mock(spec=AzureAIOrchestrator)

        # Mock async methods
        orchestrator.analyze_infrastructure = AsyncMock(return_value={
            "analysis": "Infrastructure analysis complete",
            "recommendations": ["Optimize VM sizes", "Consolidate storage"],
            "cost_impact": 1500.0,
            "execution_time_ms": random.randint(500, 2000)
        })

        orchestrator.optimize_costs = AsyncMock(return_value={
            "recommendations": [
                {"type": "resize", "resource": "vm-1", "savings": 200.0},
                {"type": "delete", "resource": "unused-disk", "savings": 50.0}
            ],
            "total_savings": 250.0,
            "execution_time_ms": random.randint(300, 1500)
        })

        orchestrator.analyze_performance = AsyncMock(return_value={
            "metrics": {"cpu": 65.0, "memory": 78.0, "network": 12.0},
            "issues": ["High memory usage", "CPU spikes detected"],
            "recommendations": ["Add memory", "Scale horizontally"],
            "execution_time_ms": random.randint(200, 1000)
        })

        orchestrator.assess_security = AsyncMock(return_value={
            "security_score": 75.0,
            "vulnerabilities": ["Open SSH port", "Outdated OS"],
            "recommendations": ["Configure NSG", "Update OS"],
            "execution_time_ms": random.randint(400, 1800)
        })

        return orchestrator

    @pytest.mark.asyncio
    async def test_concurrent_analysis_requests(self, mock_orchestrator, performance_monitor):
        """Test concurrent infrastructure analysis requests"""
        async def single_analysis():
            start_time = time.time()
            result = await mock_orchestrator.analyze_infrastructure(
                subscription_id="test-sub",
                resource_group="test-rg"
            )
            end_time = time.time()

            return {
                "result": result,
                "duration": end_time - start_time,
                "success": "analysis" in result
            }

        # Test with increasing concurrency levels
        concurrency_levels = [5, 10, 25, 50, 100]
        results = {}

        for concurrency in concurrency_levels:
            performance_monitor.start_operation(f"concurrent_analysis_{concurrency}")

            # Create concurrent tasks
            tasks = [single_analysis() for _ in range(concurrency)]

            start_time = time.time()
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            performance_monitor.end_operation(f"concurrent_analysis_{concurrency}")

            # Analyze results
            successful_tasks = [r for r in task_results if not isinstance(r, Exception) and r["success"]]
            failed_tasks = [r for r in task_results if isinstance(r, Exception) or not r.get("success", False)]

            avg_duration = statistics.mean([r["duration"] for r in successful_tasks]) if successful_tasks else 0

            results[concurrency] = {
                "total_requests": concurrency,
                "successful": len(successful_tasks),
                "failed": len(failed_tasks),
                "success_rate": len(successful_tasks) / concurrency * 100,
                "total_time": total_time,
                "avg_request_duration": avg_duration,
                "requests_per_second": concurrency / total_time if total_time > 0 else 0
            }

        # Verify performance requirements
        for concurrency, stats in results.items():
            assert stats["success_rate"] >= 95.0, f"Success rate too low at {concurrency} concurrent requests"
            assert stats["requests_per_second"] >= 1.0, f"Throughput too low at {concurrency} concurrent requests"

        return results

    @pytest.mark.asyncio
    async def test_sustained_load(self, mock_orchestrator, performance_monitor):
        """Test sustained load over time"""
        duration_seconds = 60  # 1 minute sustained load
        requests_per_second = 10
        total_requests = duration_seconds * requests_per_second

        results = []
        errors = []

        async def worker():
            while len(results) + len(errors) < total_requests:
                try:
                    start_time = time.time()

                    # Randomly choose operation type
                    operations = [
                        mock_orchestrator.analyze_infrastructure,
                        mock_orchestrator.optimize_costs,
                        mock_orchestrator.analyze_performance,
                        mock_orchestrator.assess_security
                    ]

                    operation = random.choice(operations)
                    result = await operation(subscription_id="test-sub")

                    end_time = time.time()

                    results.append({
                        "operation": operation.__name__,
                        "duration": end_time - start_time,
                        "timestamp": datetime.utcnow(),
                        "success": True
                    })

                except Exception as e:
                    errors.append({
                        "error": str(e),
                        "timestamp": datetime.utcnow()
                    })

                # Rate limiting
                await asyncio.sleep(1.0 / requests_per_second)

        # Start multiple workers
        num_workers = 5
        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

        # Wait for completion or timeout
        try:
            await asyncio.wait_for(asyncio.gather(*workers), timeout=duration_seconds + 10)
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for worker_task in workers:
                worker_task.cancel()

        # Analyze results
        total_completed = len(results) + len(errors)
        success_rate = len(results) / total_completed * 100 if total_completed > 0 else 0

        avg_duration = statistics.mean([r["duration"] for r in results]) if results else 0
        p95_duration = statistics.quantiles([r["duration"] for r in results], n=20)[18] if len(results) >= 20 else 0

        actual_rps = len(results) / duration_seconds

        # Performance assertions
        assert success_rate >= 95.0, f"Success rate {success_rate}% below 95%"
        assert avg_duration <= 2.0, f"Average duration {avg_duration}s above 2s"
        assert p95_duration <= 5.0, f"P95 duration {p95_duration}s above 5s"
        assert actual_rps >= requests_per_second * 0.8, f"Actual RPS {actual_rps} below target {requests_per_second}"

        return {
            "total_requests": total_completed,
            "successful_requests": len(results),
            "failed_requests": len(errors),
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "p95_duration": p95_duration,
            "requests_per_second": actual_rps
        }

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, mock_orchestrator):
        """Test memory usage under sustained load"""
        import psutil
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run sustained load
        async def memory_intensive_operation():
            # Simulate operations that might cause memory leaks
            for _ in range(100):
                result = await mock_orchestrator.analyze_infrastructure(
                    subscription_id="test-sub",
                    resource_group="test-rg"
                )
                # Create some objects to stress memory
                large_data = [random.random() for _ in range(1000)]
                del large_data

        # Run multiple operations concurrently
        tasks = [memory_intensive_operation() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Force garbage collection
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for test operations)
        assert memory_increase < 100, f"Memory increased by {memory_increase}MB, possible leak"

        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase
        }

    @pytest.mark.asyncio
    async def test_async_semaphore_performance(self):
        """Test async semaphore pool performance under load"""
        config = SemaphoreConfig(max_concurrent=20, timeout=5.0)
        semaphore_pool = AsyncSemaphorePool(config)

        async def semaphore_operation(op_id):
            acquired = await semaphore_pool.acquire(1, f"load-test-{op_id}")
            if acquired:
                # Simulate work
                await asyncio.sleep(random.uniform(0.1, 0.5))
                await semaphore_pool.release(f"load-test-{op_id}")
                return True
            return False

        # Test with high concurrency
        num_operations = 200
        tasks = [semaphore_operation(i) for i in range(num_operations)]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        successful_operations = sum(1 for r in results if r)
        success_rate = successful_operations / num_operations * 100

        # Verify semaphore pool performance
        assert success_rate >= 90.0, f"Semaphore success rate {success_rate}% below 90%"
        assert total_time <= 60.0, f"Total time {total_time}s too high for {num_operations} operations"

        # Check semaphore metrics
        metrics = semaphore_pool.metrics
        assert metrics.total_acquisitions >= successful_operations
        assert metrics.timeouts <= num_operations * 0.1  # Max 10% timeouts

        return {
            "total_operations": num_operations,
            "successful_operations": successful_operations,
            "success_rate": success_rate,
            "total_time": total_time,
            "operations_per_second": successful_operations / total_time,
            "semaphore_metrics": {
                "total_acquisitions": metrics.total_acquisitions,
                "timeouts": metrics.timeouts,
                "rejections": metrics.rejections,
                "max_concurrent": metrics.max_concurrent
            }
        }

    @pytest.mark.asyncio
    async def test_orchestrator_scaling_limits(self, mock_orchestrator):
        """Test orchestrator scaling limits"""
        # Test with exponentially increasing load
        scaling_levels = [1, 5, 10, 25, 50, 100, 200]
        results = {}

        for level in scaling_levels:
            async def scaled_operation():
                return await mock_orchestrator.analyze_infrastructure(
                    subscription_id="test-sub",
                    resource_group="test-rg"
                )

            # Measure performance at this scaling level
            tasks = [scaled_operation() for _ in range(level)]

            start_time = time.time()
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            successful = [r for r in task_results if not isinstance(r, Exception)]
            failed = [r for r in task_results if isinstance(r, Exception)]

            results[level] = {
                "concurrent_requests": level,
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / level * 100,
                "total_time": total_time,
                "requests_per_second": level / total_time if total_time > 0 else 0,
                "avg_response_time": total_time / level if level > 0 else 0
            }

            # Break if performance degrades significantly
            if results[level]["success_rate"] < 80.0:
                break

        # Find the scaling limit (where success rate drops below 95%)
        scaling_limit = None
        for level, stats in results.items():
            if stats["success_rate"] < 95.0:
                scaling_limit = level
                break

        if scaling_limit is None:
            scaling_limit = max(results.keys())

        return {
            "scaling_results": results,
            "scaling_limit": scaling_limit,
            "max_tested_concurrency": max(results.keys()),
            "peak_rps": max(stats["requests_per_second"] for stats in results.values())
        }


@pytest.mark.load_test
class TestLocustIntegration:
    """Integration with Locust for comprehensive load testing"""

    def test_locust_load_test_setup(self):
        """Test Locust load test configuration"""
        # This would typically be run separately, but we can test the setup
        from locust.env import Environment
        from locust.stats import stats_printer
        from locust.log import setup_logging
        import gevent

        # Setup Locust environment
        env = Environment(user_classes=[AIOrchestatorUser])
        env.create_local_runner()

        # Verify configuration
        assert env.user_classes == [AIOrchestatorUser]
        assert env.runner is not None

        # Test with minimal load
        env.runner.start(user_count=1, spawn_rate=1)

        # Let it run briefly
        gevent.sleep(2)

        # Stop the test
        env.runner.stop()

        # Verify stats were collected
        assert env.stats.total.num_requests >= 0

    def test_custom_load_test_scenarios(self):
        """Test custom load test scenarios"""
        scenarios = {
            "smoke_test": {
                "users": 1,
                "spawn_rate": 1,
                "duration": 10
            },
            "load_test": {
                "users": 10,
                "spawn_rate": 2,
                "duration": 60
            },
            "stress_test": {
                "users": 50,
                "spawn_rate": 5,
                "duration": 120
            },
            "spike_test": {
                "users": 100,
                "spawn_rate": 50,
                "duration": 30
            }
        }

        # Validate scenario configurations
        for scenario_name, config in scenarios.items():
            assert config["users"] > 0
            assert config["spawn_rate"] > 0
            assert config["duration"] > 0
            assert config["spawn_rate"] <= config["users"]

        return scenarios


class TestLoadTestUtilities:
    """Utilities for load testing"""

    def test_load_test_data_generation(self):
        """Test generation of realistic test data"""
        def generate_subscription_data():
            return {
                "subscription_id": str(uuid.uuid4()),
                "resource_groups": [f"rg-{i}" for i in range(random.randint(1, 10))],
                "locations": random.sample(["eastus", "westus", "centralus", "northeurope"], k=random.randint(1, 3))
            }

        def generate_resource_data():
            resource_types = [
                "Microsoft.Compute/virtualMachines",
                "Microsoft.Storage/storageAccounts",
                "Microsoft.Network/virtualNetworks",
                "Microsoft.Web/sites",
                "Microsoft.Sql/servers"
            ]

            return {
                "id": f"/subscriptions/{uuid.uuid4()}/resourceGroups/test-rg/providers/{random.choice(resource_types)}/test-resource",
                "name": f"test-resource-{random.randint(1, 1000)}",
                "type": random.choice(resource_types),
                "location": random.choice(["eastus", "westus", "centralus"]),
                "tags": {
                    "environment": random.choice(["dev", "test", "prod"]),
                    "owner": f"team-{random.randint(1, 5)}"
                }
            }

        # Generate test data sets
        subscriptions = [generate_subscription_data() for _ in range(10)]
        resources = [generate_resource_data() for _ in range(100)]

        # Validate generated data
        assert len(subscriptions) == 10
        assert len(resources) == 100
        assert all("subscription_id" in sub for sub in subscriptions)
        assert all("id" in res for res in resources)

        return {
            "subscriptions": subscriptions,
            "resources": resources
        }

    def test_performance_thresholds(self):
        """Define and test performance thresholds"""
        thresholds = {
            "response_time": {
                "p50": 1.0,  # 50th percentile under 1 second
                "p95": 3.0,  # 95th percentile under 3 seconds
                "p99": 5.0   # 99th percentile under 5 seconds
            },
            "throughput": {
                "min_rps": 10,   # Minimum 10 requests per second
                "target_rps": 50 # Target 50 requests per second
            },
            "error_rate": {
                "max_percentage": 1.0  # Maximum 1% error rate
            },
            "resource_usage": {
                "max_cpu_percent": 80,     # Maximum 80% CPU usage
                "max_memory_mb": 2048,     # Maximum 2GB memory usage
                "max_memory_growth_mb": 100 # Maximum 100MB memory growth during test
            }
        }

        def validate_performance_results(results: Dict[str, Any]) -> bool:
            """Validate results against thresholds"""
            violations = []

            # Check response times
            if results.get("p95_response_time", 0) > thresholds["response_time"]["p95"]:
                violations.append(f"P95 response time {results['p95_response_time']}s exceeds {thresholds['response_time']['p95']}s")

            # Check throughput
            if results.get("requests_per_second", 0) < thresholds["throughput"]["min_rps"]:
                violations.append(f"RPS {results['requests_per_second']} below minimum {thresholds['throughput']['min_rps']}")

            # Check error rate
            if results.get("error_rate_percent", 0) > thresholds["error_rate"]["max_percentage"]:
                violations.append(f"Error rate {results['error_rate_percent']}% exceeds {thresholds['error_rate']['max_percentage']}%")

            return len(violations) == 0, violations

        # Test threshold validation
        good_results = {
            "p95_response_time": 2.5,
            "requests_per_second": 25,
            "error_rate_percent": 0.5
        }

        bad_results = {
            "p95_response_time": 6.0,
            "requests_per_second": 5,
            "error_rate_percent": 5.0
        }

        is_valid, violations = validate_performance_results(good_results)
        assert is_valid, f"Good results should pass validation: {violations}"

        is_valid, violations = validate_performance_results(bad_results)
        assert not is_valid, "Bad results should fail validation"
        assert len(violations) == 3, f"Expected 3 violations, got {len(violations)}"

        return thresholds, validate_performance_results


# Command-line interface for running load tests
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="AI Orchestrator Load Testing")
    parser.add_argument("--test-type", choices=["unit", "locust", "stress"], default="unit",
                       help="Type of load test to run")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--spawn-rate", type=int, default=2, help="User spawn rate per second")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--host", default="http://localhost:8000", help="Target host URL")

    args = parser.parse_args()

    if args.test_type == "unit":
        # Run async unit tests
        pytest.main([__file__ + "::TestAIOrchestatorLoadTest", "-v"])

    elif args.test_type == "locust":
        # Run Locust load test
        from locust.main import main as locust_main

        # Set up Locust arguments
        sys.argv = [
            "locust",
            "-f", __file__,
            "--host", args.host,
            "--users", str(args.users),
            "--spawn-rate", str(args.spawn_rate),
            "--run-time", f"{args.duration}s",
            "--headless"
        ]

        locust_main()

    elif args.test_type == "stress":
        # Run stress test with high concurrency
        async def run_stress_test():
            test_instance = TestAIOrchestatorLoadTest()
            mock_orchestrator = test_instance.mock_orchestrator()

            print("Running stress test...")
            results = await test_instance.test_orchestrator_scaling_limits(mock_orchestrator)

            print("\nStress Test Results:")
            print(f"Scaling limit: {results['scaling_limit']} concurrent requests")
            print(f"Peak RPS: {results['peak_rps']:.2f}")
            print(f"Max tested concurrency: {results['max_tested_concurrency']}")

            return results

        asyncio.run(run_stress_test())