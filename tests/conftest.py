"""
Pytest configuration and shared fixtures
"""

import os
import sys
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch

import pytest
import pytest_asyncio
from faker import Faker

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["AZURE_SUBSCRIPTION_ID"] = "test-subscription-id"
os.environ["AZURE_TENANT_ID"] = "test-tenant-id"
os.environ["AZURE_CLIENT_ID"] = "test-client-id"
os.environ["AZURE_CLIENT_SECRET"] = "test-secret"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
os.environ["AZURE_OPENAI_KEY"] = "test-key"
os.environ["AZURE_LOG_ANALYTICS_WORKSPACE_ID"] = "test-workspace"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"

fake = Faker()


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_azure_credential():
    """Mock Azure credential"""
    with patch("azure.identity.DefaultAzureCredential") as mock:
        yield mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    mock = Mock()
    mock.ainvoke = AsyncMock(return_value=Mock(content="Test response"))
    return mock


@pytest.fixture
def mock_redis_client():
    """Mock Redis client"""
    mock = Mock()
    mock.get = Mock(return_value=None)
    mock.setex = Mock(return_value=True)
    mock.delete = Mock(return_value=1)
    mock.ping = Mock(return_value=True)
    return mock


@pytest.fixture
def mock_resource_client():
    """Mock Azure Resource Management client"""
    mock = Mock()
    mock.resource_groups.list.return_value = [
        Mock(name="test-rg-1", location="eastus"),
        Mock(name="test-rg-2", location="westus")
    ]
    return mock


@pytest.fixture
def mock_compute_client():
    """Mock Azure Compute client"""
    mock = Mock()

    # Mock VM
    mock_vm = Mock()
    mock_vm.id = "/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm"
    mock_vm.name = "test-vm"
    mock_vm.location = "eastus"
    mock_vm.hardware_profile.vm_size = "Standard_B2s"
    mock_vm.provisioning_state = "Succeeded"

    mock.virtual_machines.get.return_value = mock_vm
    mock.virtual_machines.list_all.return_value = [mock_vm]

    return mock


@pytest.fixture
def sample_vm_config():
    """Sample VM configuration"""
    return {
        "name": "test-vm",
        "resource_group": "test-rg",
        "location": "eastus",
        "size": "Standard_B2s",
        "admin_username": "azureuser",
        "admin_password": "TestP@ssw0rd123!"
    }


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for ML models"""
    import pandas as pd
    import numpy as np

    n_samples = 100
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
        'cpu_usage_percent': np.random.normal(50, 10, n_samples),
        'memory_usage_percent': np.random.normal(60, 15, n_samples),
        'disk_read_bytes_per_sec': np.random.normal(1000, 200, n_samples),
        'disk_write_bytes_per_sec': np.random.normal(800, 150, n_samples),
        'network_in_bytes_per_sec': np.random.normal(5000, 1000, n_samples),
        'network_out_bytes_per_sec': np.random.normal(4000, 800, n_samples),
        'request_rate': np.random.normal(100, 20, n_samples),
        'error_rate': np.random.normal(2, 0.5, n_samples)
    })


@pytest.fixture
async def mock_orchestrator(mock_openai_client, mock_redis_client):
    """Mock AI Orchestrator"""
    with patch("ai_orchestrator.orchestrator.AzureChatOpenAI", return_value=mock_openai_client):
        with patch("ai_orchestrator.orchestrator.redis.Redis", return_value=mock_redis_client):
            from ai_orchestrator.orchestrator import AzureAIOrchestrator
            orchestrator = AzureAIOrchestrator()
            yield orchestrator


@pytest_asyncio.fixture
async def test_client():
    """FastAPI test client"""
    from fastapi.testclient import TestClient
    from api.server import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Authorization headers for API requests"""
    return {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json"
    }


@pytest.fixture
def mock_graph_api():
    """Mock Microsoft Graph API responses"""
    mock = Mock()

    mock.get_user.return_value = {
        "id": "user-123",
        "displayName": "Test User",
        "mail": "test@example.com",
        "userPrincipalName": "test@example.com"
    }

    mock.list_users.return_value = [
        {"id": "user-1", "displayName": "User 1"},
        {"id": "user-2", "displayName": "User 2"}
    ]

    return mock


@pytest.fixture
def mock_teams_webhook():
    """Mock Teams webhook for notifications"""
    mock = Mock()
    mock.send_message = AsyncMock(return_value={"status": "sent"})
    return mock


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after tests"""
    yield

    # Clean up any test files created
    test_dirs = [
        "ml-models/saved",
        "logs",
        "test_output"
    ]

    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set environment variables for testing"""
    test_vars = {
        "AZURE_SUBSCRIPTION_ID": "test-sub-123",
        "AZURE_TENANT_ID": "test-tenant-123",
        "AZURE_CLIENT_ID": "test-client-123",
        "AZURE_CLIENT_SECRET": "test-secret",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_KEY": "test-key-123",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4-test",
        "REDIS_HOST": "test-redis",
        "REDIS_PORT": "6379",
        "JWT_SECRET_KEY": "test-jwt-secret",
        "API_KEY": "test-api-key"
    }

    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)

    return test_vars


# Performance testing fixtures
@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer"""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()
            return self.end_time - self.start_time

    return Timer()


# Database fixtures
@pytest.fixture
async def mock_db_session():
    """Mock database session"""
    mock = AsyncMock()
    mock.add = AsyncMock()
    mock.commit = AsyncMock()
    mock.rollback = AsyncMock()
    mock.close = AsyncMock()
    yield mock