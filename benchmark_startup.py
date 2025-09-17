#!/usr/bin/env python3
"""
Startup Performance Benchmark Tool
Measures and compares startup times between original and ultra-optimized versions
"""

import subprocess
import time
import sys
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Dict, Any


class StartupBenchmark:
    """Benchmark tool for measuring application startup performance"""

    def __init__(self):
        self.results = {}
        self.iterations = 5

    def measure_import_time(self, script_content: str) -> float:
        """Measure time to import critical modules"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            f.flush()

            start_time = time.perf_counter()
            result = subprocess.run([sys.executable, f.name],
                                  capture_output=True, text=True)
            end_time = time.perf_counter()

            Path(f.name).unlink()  # Clean up

            if result.returncode != 0:
                raise RuntimeError(f"Script failed: {result.stderr}")

            return end_time - start_time

    def benchmark_original_imports(self) -> List[float]:
        """Benchmark original import pattern"""
        script = """
import sys
import time
start = time.perf_counter()

# Original imports
import fastapi
import uvicorn
import redis
import pydantic
import azure.identity
import azure.mgmt.resource
import openai
import langchain

end = time.perf_counter()
print(f"Import time: {end - start:.6f}")
"""

        times = []
        for i in range(self.iterations):
            try:
                runtime = self.measure_import_time(script)
                times.append(runtime)
                print(f"Original imports #{i+1}: {runtime:.3f}s")
            except Exception as e:
                print(f"Error in original benchmark #{i+1}: {e}")

        return times

    def benchmark_ultra_imports(self) -> List[float]:
        """Benchmark ultra-optimized import pattern"""
        script = """
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import time
start = time.perf_counter()

# Ultra-fast imports
from core.fast_imports import fastapi, uvicorn, redis, azure_identity
from core.ultra_cache import ultra_cache
from core.memory_optimizer import optimize_startup

# Initialize optimizations
optimize_startup()

end = time.perf_counter()
print(f"Ultra import time: {end - start:.6f}")
"""

        times = []
        for i in range(self.iterations):
            try:
                runtime = self.measure_import_time(script)
                times.append(runtime)
                print(f"Ultra imports #{i+1}: {runtime:.3f}s")
            except Exception as e:
                print(f"Error in ultra benchmark #{i+1}: {e}")

        return times

    def benchmark_server_startup(self, script_path: str, timeout: int = 30) -> float:
        """Benchmark server startup time until health check responds"""
        import requests
        import threading

        # Start server process
        process = subprocess.Popen([sys.executable, script_path],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

        start_time = time.perf_counter()

        # Wait for server to be ready
        max_wait = timeout
        health_url = "http://localhost:8000/health"

        while time.perf_counter() - start_time < max_wait:
            try:
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    ready_time = time.perf_counter() - start_time
                    process.terminate()
                    process.wait(timeout=5)
                    return ready_time
            except requests.exceptions.RequestException:
                pass

            time.sleep(0.1)

        # Server didn't start in time
        process.terminate()
        process.wait(timeout=5)
        raise TimeoutError(f"Server didn't start within {timeout} seconds")

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive startup benchmark"""
        print("🚀 Starting comprehensive startup benchmark...")
        print(f"Running {self.iterations} iterations each\n")

        results = {}

        # Test 1: Import performance
        print("📦 Testing import performance...")
        try:
            original_times = self.benchmark_original_imports()
            results['original_imports'] = {
                'times': original_times,
                'mean': mean(original_times),
                'median': median(original_times),
                'min': min(original_times),
                'max': max(original_times),
                'stdev': stdev(original_times) if len(original_times) > 1 else 0
            }
        except Exception as e:
            print(f"❌ Original imports benchmark failed: {e}")
            results['original_imports'] = None

        try:
            ultra_times = self.benchmark_ultra_imports()
            results['ultra_imports'] = {
                'times': ultra_times,
                'mean': mean(ultra_times),
                'median': median(ultra_times),
                'min': min(ultra_times),
                'max': max(ultra_times),
                'stdev': stdev(ultra_times) if len(ultra_times) > 1 else 0
            }
        except Exception as e:
            print(f"❌ Ultra imports benchmark failed: {e}")
            results['ultra_imports'] = None

        # Calculate improvement
        if results['original_imports'] and results['ultra_imports']:
            original_mean = results['original_imports']['mean']
            ultra_mean = results['ultra_imports']['mean']
            improvement = ((original_mean - ultra_mean) / original_mean) * 100
            results['import_improvement'] = improvement
            print(f"📈 Import performance improvement: {improvement:.1f}%")

        print("\n" + "="*60)
        return results

    def print_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results"""
        print("\n🎯 BENCHMARK RESULTS")
        print("="*60)

        if results.get('original_imports'):
            orig = results['original_imports']
            print(f"📦 Original Imports:")
            print(f"   Mean: {orig['mean']:.3f}s")
            print(f"   Min:  {orig['min']:.3f}s")
            print(f"   Max:  {orig['max']:.3f}s")
            print(f"   StdDev: {orig['stdev']:.3f}s")

        if results.get('ultra_imports'):
            ultra = results['ultra_imports']
            print(f"\n⚡ Ultra-Fast Imports:")
            print(f"   Mean: {ultra['mean']:.3f}s")
            print(f"   Min:  {ultra['min']:.3f}s")
            print(f"   Max:  {ultra['max']:.3f}s")
            print(f"   StdDev: {ultra['stdev']:.3f}s")

        if 'import_improvement' in results:
            improvement = results['import_improvement']
            print(f"\n🚀 Performance Improvement: {improvement:.1f}%")

            if improvement > 50:
                print("✅ EXCELLENT: >50% improvement achieved!")
            elif improvement > 25:
                print("✅ GOOD: >25% improvement achieved!")
            elif improvement > 0:
                print("✅ POSITIVE: Some improvement achieved!")
            else:
                print("❌ No improvement or regression detected")

        print("\n" + "="*60)

    def save_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {filename}")


def main():
    """Run startup performance benchmark"""
    benchmark = StartupBenchmark()

    try:
        results = benchmark.run_full_benchmark()
        benchmark.print_results(results)
        benchmark.save_results(results)

    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()