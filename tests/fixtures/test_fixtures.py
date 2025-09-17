"""
Test Fixtures and Utilities
Comprehensive test fixtures for all test categories
"""

import asyncio
import pytest
import tempfile
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import json

# Database fixtures
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

# Azure SDK mocks
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.monitor import MonitorManagementClient

# Local imports
from database.models import Base, User, Incident, Resource, SystemLog, Configuration
from core.async_optimizations import AsyncSemaphorePool, SemaphoreConfig
from core.memory_optimization import MemoryTracker, ObjectPool, PoolConfig
from core.advanced_caching import MultiTierCache
from ai_orchestrator.orchestrator import AzureAIOrchestrator


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_database_url():
    """Test database URL for in-memory SQLite"""
    return "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
async def test_database_engine(test_database_url):
    """Test database engine with schema creation"""
    engine = create_async_engine(
        test_database_url,
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture
async def test_db_session(test_database_engine) -> AsyncGenerator[AsyncSession, None]:
    """Test database session with automatic rollback"""
    async_session = async_sessionmaker(
        test_database_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session() as session:
        # Start a transaction
        transaction = await session.begin()

        yield session

        # Always rollback to keep tests isolated
        await transaction.rollback()


@pytest.fixture
async def db_user(test_db_session: AsyncSession) -> User:
    """Create a test user in database"""
    user = User(
        email="test@example.com",
        username="testuser",
        full_name="Test User",
        role="user",
        azure_user_id="azure-123",
        is_active=True
    )

    test_db_session.add(user)
    await test_db_session.flush()  # Get ID without committing
    await test_db_session.refresh(user)

    return user


@pytest.fixture
async def db_resource(test_db_session: AsyncSession) -> Resource:
    """Create a test resource in database"""
    resource = Resource(
        resource_id="/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm",
        name="test-vm",
        resource_type="virtual_machine",
        subscription_id="test-subscription",
        resource_group="test-rg",
        location="eastus",
        status="Running",
        configuration={"vm_size": "Standard_D2s_v3", "os": "Linux"},
        tags={"environment": "test", "team": "dev"},
        cost_monthly=150.0
    )

    test_db_session.add(resource)
    await test_db_session.flush()
    await test_db_session.refresh(resource)

    return resource


@pytest.fixture
async def db_incident(test_db_session: AsyncSession, db_user: User, db_resource: Resource) -> Incident:
    """Create a test incident in database"""
    incident = Incident(
        title="Test Incident",
        description="Test incident description",
        severity="high",
        status="open",
        resource_id=db_resource.resource_id,
        assignee_id=db_user.id,
        metadata_={"alert_rule": "test-rule", "threshold": 90.0}
    )

    test_db_session.add(incident)
    await test_db_session.flush()
    await test_db_session.refresh(incident)

    return incident


# ============================================================================
# Azure Service Mocks
# ============================================================================

@pytest.fixture
def mock_azure_credential():
    """Mock Azure credential"""
    with patch('azure.identity.DefaultAzureCredential') as mock:
        mock_cred = Mock()
        mock_cred.get_token.return_value = Mock(token="mock_token", expires_on=9999999999)
        mock.return_value = mock_cred
        yield mock


@pytest.fixture
def mock_resource_client():
    """Mock Azure Resource Management client"""
    client = Mock(spec=ResourceManagementClient)

    # Mock resource groups
    mock_rg = Mock()
    mock_rg.name = "test-rg"
    mock_rg.location = "eastus"
    mock_rg.id = "/subscriptions/test/resourceGroups/test-rg"
    mock_rg.tags = {"environment": "test"}

    client.resource_groups.list.return_value = [mock_rg]
    client.resource_groups.get.return_value = mock_rg

    # Mock resources
    mock_vm = Mock()
    mock_vm.id = "/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm"
    mock_vm.name = "test-vm"
    mock_vm.type = "Microsoft.Compute/virtualMachines"
    mock_vm.location = "eastus"

    client.resources.list.return_value = [mock_vm]
    client.resources.get_by_id.return_value = mock_vm

    return client


@pytest.fixture
def mock_compute_client():
    """Mock Azure Compute Management client"""
    client = Mock(spec=ComputeManagementClient)

    # Mock VM
    mock_vm = Mock()
    mock_vm.name = "test-vm"
    mock_vm.id = "/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm"
    mock_vm.location = "eastus"
    mock_vm.provisioning_state = "Succeeded"
    mock_vm.hardware_profile.vm_size = "Standard_D2s_v3"
    mock_vm.os_profile.computer_name = "test-vm"
    mock_vm.os_profile.admin_username = "azureuser"

    client.virtual_machines.list_all.return_value = [mock_vm]
    client.virtual_machines.get.return_value = mock_vm

    return client


@pytest.fixture
def mock_monitor_client():
    """Mock Azure Monitor client"""
    client = Mock(spec=MonitorManagementClient)

    # Mock metrics
    mock_metric = Mock()
    mock_metric.name.value = "Percentage CPU"
    mock_metric.unit = "Percent"

    mock_data_point = Mock()
    mock_data_point.time_stamp = datetime.utcnow()
    mock_data_point.average = 65.5
    mock_data_point.maximum = 85.0
    mock_data_point.minimum = 45.0

    mock_timeseries = Mock()
    mock_timeseries.data = [mock_data_point]
    mock_metric.timeseries = [mock_timeseries]

    mock_response = Mock()
    mock_response.value = [mock_metric]

    client.metrics.list.return_value = mock_response

    return client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    client = Mock()

    # Mock chat completion
    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content="Mock AI response with recommendations"))
    ]
    mock_response.usage = Mock(total_tokens=150)

    client.chat.completions.create.return_value = mock_response

    return client


# ============================================================================
# AI Orchestrator Fixtures
# ============================================================================

@pytest.fixture
def mock_ai_orchestrator():
    """Mock AI Orchestrator with realistic responses"""
    orchestrator = Mock(spec=AzureAIOrchestrator)

    # Mock analyze_infrastructure
    orchestrator.analyze_infrastructure = AsyncMock(return_value={
        "analysis_id": str(uuid.uuid4()),
        "analysis": "Infrastructure analysis completed successfully",
        "recommendations": [
            {
                "type": "cost_optimization",
                "priority": "high",
                "description": "Resize oversized VMs",
                "potential_savings": 800.0,
                "resources": ["vm-1", "vm-2"]
            },
            {
                "type": "performance",
                "priority": "medium",
                "description": "Add load balancer",
                "estimated_cost": 200.0,
                "resources": ["web-app-1"]
            }
        ],
        "cost_analysis": {
            "current_monthly_cost": 3000.0,
            "projected_savings": 800.0,
            "optimization_percentage": 26.7
        },
        "performance_metrics": {
            "overall_health_score": 85.0,
            "cpu_utilization": 65.0,
            "memory_utilization": 78.0
        },
        "security_assessment": {
            "security_score": 75.0,
            "vulnerabilities": ["Open SSH port", "Unencrypted storage"],
            "compliance_status": "Partially Compliant"
        },
        "execution_time_ms": 2450
    })

    # Mock optimize_costs
    orchestrator.optimize_costs = AsyncMock(return_value={
        "optimization_id": str(uuid.uuid4()),
        "recommendations": [
            {
                "type": "resize",
                "resource_id": "/subscriptions/test/vm-1",
                "current_size": "Standard_D4s_v3",
                "recommended_size": "Standard_D2s_v3",
                "monthly_savings": 150.0,
                "confidence": 95.0
            }
        ],
        "total_monthly_savings": 150.0,
        "implementation_risk": "low"
    })

    # Mock analyze_performance
    orchestrator.analyze_performance = AsyncMock(return_value={
        "analysis_id": str(uuid.uuid4()),
        "metrics": {
            "cpu": {"average": 65.0, "p95": 85.0},
            "memory": {"average": 78.0, "p95": 92.0},
            "network": {"bytes_per_sec": 1048576},
            "disk": {"iops": 150, "latency_ms": 5.2}
        },
        "performance_issues": [
            {
                "type": "high_cpu_usage",
                "severity": "medium",
                "description": "CPU usage frequently above 80%"
            }
        ],
        "overall_health_score": 72.0,
        "recommendations": ["Scale up VM", "Add auto-scaling"]
    })

    # Mock assess_security
    orchestrator.assess_security = AsyncMock(return_value={
        "assessment_id": str(uuid.uuid4()),
        "security_score": 75.0,
        "vulnerabilities": [
            {
                "id": "vuln-001",
                "severity": "high",
                "title": "Open SSH port",
                "description": "SSH port 22 is open to internet",
                "remediation": "Restrict SSH access"
            }
        ],
        "compliance_status": {
            "overall_compliance": 78.0,
            "frameworks": {"SOC2": 85.0, "ISO27001": 72.0}
        }
    })

    # Mock chat_interaction
    orchestrator.chat_interaction = AsyncMock(return_value={
        "response_id": str(uuid.uuid4()),
        "message": "Based on your infrastructure, I recommend optimizing VM sizes to save approximately $800 per month.",
        "suggestions": [
            "Show VM rightsizing details",
            "Calculate cost savings",
            "Generate optimization report"
        ],
        "confidence": 95.0
    })

    return orchestrator


# ============================================================================
# Core Component Fixtures
# ============================================================================

@pytest.fixture
def semaphore_pool():
    """Async semaphore pool for testing"""
    config = SemaphoreConfig(max_concurrent=10, timeout=5.0)
    return AsyncSemaphorePool(config)


@pytest.fixture
def memory_tracker():
    """Memory tracker for testing"""
    return MemoryTracker()


@pytest.fixture
def object_pool():
    """Object pool for testing"""
    def create_obj():
        return {"id": uuid.uuid4().hex, "data": [], "created_at": datetime.utcnow()}

    config = PoolConfig(
        min_size=5,
        max_size=20,
        create_function=create_obj
    )
    return ObjectPool(config)


@pytest.fixture
def multi_tier_cache():
    """Multi-tier cache for testing"""
    return MultiTierCache()


# ============================================================================
# API Test Fixtures
# ============================================================================

@pytest.fixture
def api_client():
    """FastAPI test client"""
    from fastapi.testclient import TestClient
    from api.server import app

    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers for API requests"""
    return {
        "Authorization": "Bearer test_token_12345",
        "Content-Type": "application/json",
        "X-Request-ID": str(uuid.uuid4())
    }


@pytest.fixture
def sample_subscription_data():
    """Sample Azure subscription data"""
    return {
        "subscription_id": "test-subscription-12345",
        "display_name": "Test Subscription",
        "state": "Enabled",
        "tenant_id": "test-tenant-12345",
        "resource_groups": [
            {
                "name": "test-rg-1",
                "location": "eastus",
                "tags": {"environment": "test", "team": "dev"}
            },
            {
                "name": "test-rg-2",
                "location": "westus",
                "tags": {"environment": "prod", "team": "ops"}
            }
        ]
    }


@pytest.fixture
def sample_vm_data():
    """Sample VM data for testing"""
    return {
        "id": "/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm",
        "name": "test-vm",
        "location": "eastus",
        "vm_size": "Standard_D2s_v3",
        "os_type": "Linux",
        "provisioning_state": "Succeeded",
        "power_state": "VM running",
        "network_interfaces": [
            {
                "id": "/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Network/networkInterfaces/test-vm-nic",
                "primary": True
            }
        ],
        "disks": [
            {
                "name": "test-vm-osdisk",
                "disk_size_gb": 128,
                "storage_account_type": "Premium_LRS"
            }
        ],
        "tags": {
            "environment": "test",
            "application": "web-server",
            "owner": "dev-team"
        }
    }


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing"""
    base_time = datetime.utcnow() - timedelta(hours=1)

    return [
        {
            "metric_name": "Percentage CPU",
            "unit": "Percent",
            "time_series": [
                {
                    "timestamp": base_time + timedelta(minutes=i),
                    "average": 50 + (i % 20),
                    "maximum": 70 + (i % 30),
                    "minimum": 30 + (i % 15)
                }
                for i in range(60)  # 60 data points (1 hour)
            ]
        },
        {
            "metric_name": "Available Memory Bytes",
            "unit": "Bytes",
            "time_series": [
                {
                    "timestamp": base_time + timedelta(minutes=i),
                    "average": 2147483648 - (i * 1048576),  # 2GB decreasing
                    "maximum": 2684354560,  # 2.5GB
                    "minimum": 1073741824   # 1GB
                }
                for i in range(60)
            ]
        }
    ]


# ============================================================================
# Performance Test Fixtures
# ============================================================================

@pytest.fixture
def performance_thresholds():
    """Performance test thresholds"""
    from tests.performance.test_benchmarks import PerformanceThresholds

    return PerformanceThresholds(
        max_response_time_ms=1000.0,
        min_ops_per_second=100.0,
        max_memory_usage_mb=512.0,
        max_cpu_usage_percent=80.0,
        min_success_rate=95.0
    )


@pytest.fixture
def benchmark_runner():
    """Benchmark test runner"""
    from tests.performance.test_benchmarks import PerformanceBenchmark

    def create_benchmark(name: str, thresholds=None):
        return PerformanceBenchmark(name, thresholds)

    return create_benchmark


# ============================================================================
# Load Test Fixtures
# ============================================================================

@pytest.fixture
def load_test_config():
    """Load test configuration"""
    return {
        "users": 10,
        "spawn_rate": 2,
        "duration": 30,
        "host": "http://localhost:8000",
        "scenarios": {
            "smoke": {"users": 1, "duration": 10},
            "load": {"users": 10, "duration": 60},
            "stress": {"users": 50, "duration": 120}
        }
    }


# ============================================================================
# File and Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_directory():
    """Temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_file():
    """Temporary file for tests"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        yield temp_file.name

    # Cleanup
    try:
        os.unlink(temp_file.name)
    except FileNotFoundError:
        pass


@pytest.fixture
def sample_config_file(temp_file):
    """Sample configuration file"""
    config_data = {
        "database": {
            "url": "postgresql://user:pass@localhost/testdb",
            "pool_size": 10
        },
        "azure": {
            "subscription_id": "test-subscription",
            "tenant_id": "test-tenant",
            "client_id": "test-client"
        },
        "ai": {
            "openai_endpoint": "https://test.openai.azure.com",
            "deployment_name": "gpt-4"
        }
    }

    with open(temp_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    return temp_file


# ============================================================================
# Environment and Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_environment():
    """Mock environment variables"""
    env_vars = {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "AZURE_SUBSCRIPTION_ID": "test-subscription-123",
        "AZURE_TENANT_ID": "test-tenant-123",
        "AZURE_CLIENT_ID": "test-client-123",
        "AZURE_CLIENT_SECRET": "test-secret",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_KEY": "test-openai-key",
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/1",
        "JWT_SECRET_KEY": "test-jwt-secret-key"
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis_mock = Mock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=False)
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.flushdb = AsyncMock(return_value=True)

    return redis_mock


@pytest.fixture
def mock_time():
    """Mock time functions for deterministic testing"""
    fixed_time = datetime(2024, 1, 15, 12, 0, 0)

    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.utcnow.return_value = fixed_time
        mock_datetime.now.return_value = fixed_time
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        with patch('time.time', return_value=fixed_time.timestamp()):
            yield fixed_time


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test"""
    yield

    # Cleanup any test artifacts
    import gc
    gc.collect()

    # Reset any global state if needed
    # This would be customized based on your application's global state


@pytest.fixture
def isolated_test():
    """Fixture for tests that need complete isolation"""
    # Setup isolated environment
    original_modules = list(sys.modules.keys())

    yield

    # Cleanup: remove any modules imported during test
    current_modules = list(sys.modules.keys())
    for module in current_modules:
        if module not in original_modules and module.startswith('tests.'):
            del sys.modules[module]


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def test_data_generator():
    """Generator for creating test data"""
    class TestDataGenerator:
        @staticmethod
        def create_user_data(count: int = 1) -> List[Dict[str, Any]]:
            return [
                {
                    "email": f"user{i}@example.com",
                    "username": f"user{i}",
                    "full_name": f"User {i}",
                    "role": "user" if i % 2 == 0 else "admin",
                    "is_active": True
                }
                for i in range(count)
            ]

        @staticmethod
        def create_resource_data(count: int = 1) -> List[Dict[str, Any]]:
            return [
                {
                    "resource_id": f"/subscriptions/test/resource-{i}",
                    "name": f"resource-{i}",
                    "resource_type": "virtual_machine",
                    "location": "eastus" if i % 2 == 0 else "westus",
                    "cost_monthly": 100.0 + (i * 10)
                }
                for i in range(count)
            ]

        @staticmethod
        def create_incident_data(count: int = 1) -> List[Dict[str, Any]]:
            severities = ["low", "medium", "high", "critical"]
            statuses = ["open", "in_progress", "resolved", "closed"]

            return [
                {
                    "title": f"Test Incident {i}",
                    "description": f"Description for incident {i}",
                    "severity": severities[i % len(severities)],
                    "status": statuses[i % len(statuses)],
                    "resource_id": f"/subscriptions/test/resource-{i}"
                }
                for i in range(count)
            ]

    return TestDataGenerator()


@pytest.fixture
def assertion_helpers():
    """Helper functions for test assertions"""
    class AssertionHelpers:
        @staticmethod
        def assert_response_structure(response_data: Dict[str, Any], required_fields: List[str]):
            """Assert response has required fields"""
            for field in required_fields:
                assert field in response_data, f"Missing required field: {field}"

        @staticmethod
        def assert_performance_within_bounds(duration: float, max_duration: float):
            """Assert performance is within acceptable bounds"""
            assert duration <= max_duration, f"Duration {duration}s exceeds maximum {max_duration}s"

        @staticmethod
        def assert_error_format(error_response: Dict[str, Any]):
            """Assert error response has correct format"""
            required_fields = ["error", "message", "timestamp"]
            for field in required_fields:
                assert field in error_response, f"Error response missing {field}"

    return AssertionHelpers()