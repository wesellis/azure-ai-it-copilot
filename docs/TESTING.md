# Testing Documentation

## Overview

This document provides comprehensive information about the testing framework, verification procedures, and quality assurance for the Azure AI IT Copilot project.

## Test Framework Setup

### Prerequisites

The project uses pytest as the primary testing framework with the following dependencies:

```bash
# Core testing packages
pytest>=8.4.2
pytest-asyncio>=1.2.0
pytest-mock>=3.15.1
pytest-cov>=7.0.0
pytest-xdist>=3.8.0

# Mocking and fixtures
faker>=37.8.0
responses>=0.25.8
aioresponses>=0.7.8

# Application dependencies
fastapi
azure-identity
azure-mgmt-*
langchain-openai
redis
```

### Installation

```bash
# Install testing dependencies
pip install pytest pytest-asyncio pytest-mock pytest-cov pytest-xdist
pip install faker responses aioresponses

# Install application dependencies
pip install -r requirements.txt
```

## Test Structure

### Directory Layout

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── fixtures/                   # Test data and mock objects
├── integration/                # Integration tests
├── unit/                      # Unit tests
├── test_api.py                # API endpoint tests
└── test_orchestrator.py       # Core orchestrator tests
```

### Test Categories

#### 1. API Tests (`test_api.py`)
- **Health Check Tests**: Verify service availability
- **Authentication Tests**: JWT token validation
- **Resource Management**: CRUD operations
- **Permission Tests**: Role-based access control

#### 2. Orchestrator Tests (`test_orchestrator.py`)
- **Intent Classification**: Natural language understanding
- **Permission Validation**: Security checks
- **Command Processing**: Core business logic
- **Status Reporting**: System health monitoring

#### 3. Unit Tests (`unit/test_orchestrator_unit.py`)
- **Component Initialization**: Service setup
- **Agent Functionality**: Individual agent tests
- **Approval Workflows**: Decision logic
- **Caching Mechanisms**: Performance optimization

## Running Tests

### Basic Test Execution

```bash
# Run all tests
cd /path/to/azure-ai-it-copilot
PYTHONPATH=. python3 -m pytest tests/ -v

# Run specific test file
PYTHONPATH=. python3 -m pytest tests/test_api.py -v

# Run single test
PYTHONPATH=. python3 -m pytest tests/test_api.py::TestAPI::test_health_check -v
```

### Coverage Reports

```bash
# Generate coverage report
PYTHONPATH=. python3 -m pytest tests/ --cov=ai_orchestrator --cov=api --cov=integrations --cov-report=html

# View coverage in browser
open htmlcov/index.html
```

### Parallel Execution

```bash
# Run tests in parallel
PYTHONPATH=. python3 -m pytest tests/ -n auto
```

## Test Results Summary

### Current Test Suite

**Total Tests**: 30
- API Tests: 8
- Orchestrator Tests: 7
- Unit Tests: 15

### Test Categories by Functionality

#### Authentication & Security
- JWT token validation
- Role-based permissions
- Resource access control
- Security headers verification

#### Core Functionality
- Intent classification accuracy
- Command processing logic
- Resource management operations
- Error handling mechanisms

#### Integration Points
- Azure service connectivity
- External API integrations
- Database operations
- Cache functionality

## Test Coverage Analysis

### Current Coverage Metrics

```
Total Coverage: ~35% (API and Orchestrator core)
```

#### Detailed Coverage by Module

| Module | Coverage | Status |
|--------|----------|---------|
| `api/server.py` | 35% | ✅ Core endpoints tested |
| `ai_orchestrator/orchestrator.py` | 36% | ✅ Main flows tested |
| `integrations/*` | 0% | 🔄 Requires mock setup |
| `ml_models/*` | 0% | 🔄 Requires test data |
| `automation_engine/*` | 0% | 🔄 Requires Azure mocks |

### Areas Requiring Additional Testing

1. **Integration Services**
   - Microsoft Graph API calls
   - Teams/Slack notifications
   - Azure Sentinel integration

2. **Machine Learning Models**
   - Anomaly detection algorithms
   - Predictive analytics
   - Model training/inference

3. **Automation Engine**
   - PowerShell script execution
   - Resource provisioning
   - Error recovery mechanisms

## Mock Configuration

### Azure Services Mocking

```python
# Example Azure service mock
@patch('azure.identity.DefaultAzureCredential')
@patch('azure.mgmt.resource.ResourceManagementClient')
def test_azure_resource_creation(mock_client, mock_credential):
    mock_client.return_value.resource_groups.create_or_update.return_value = Mock()
    # Test implementation
```

### External API Mocking

```python
# Example external API mock
@responses.activate
def test_teams_notification():
    responses.add(
        responses.POST,
        'https://outlook.office.com/webhook/...',
        json={'status': 'success'},
        status=200
    )
    # Test implementation
```

## Performance Testing

### Load Testing Setup

```bash
# Install performance testing tools
pip install locust pytest-benchmark

# Run load tests
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

### Benchmark Tests

```python
def test_intent_classification_performance(benchmark):
    """Benchmark intent classification speed"""
    orchestrator = AzureAIOrchestrator()
    result = benchmark(orchestrator.classify_intent, "create a new VM")
    assert result.intent_type == IntentType.RESOURCE_CREATE
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          PYTHONPATH=. pytest tests/ --cov=. --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Test Environment Setup

### Environment Variables

```bash
# Required for testing
export AZURE_SUBSCRIPTION_ID="test-subscription-id"
export AZURE_TENANT_ID="test-tenant-id"
export AZURE_CLIENT_ID="test-client-id"
export AZURE_CLIENT_SECRET="test-client-secret"
export JWT_SECRET_KEY="test-jwt-secret"
```

### Test Configuration

```python
# tests/conftest.py
import pytest
import os
from fastapi.testclient import TestClient

@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables"""
    os.environ.update({
        'AZURE_SUBSCRIPTION_ID': 'test-subscription-id',
        'AZURE_TENANT_ID': 'test-tenant-id',
        'AZURE_CLIENT_ID': 'test-client-id',
        'AZURE_CLIENT_SECRET': 'test-client-secret'
    })

@pytest.fixture
def client():
    """FastAPI test client"""
    from api.server import app
    return TestClient(app)
```

## Debugging Tests

### Common Issues and Solutions

#### 1. Module Import Errors
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH=/path/to/azure-ai-it-copilot
```

#### 2. Azure Authentication in Tests
```python
# Mock Azure credentials
@patch('azure.identity.DefaultAzureCredential')
def test_with_mocked_azure(mock_credential):
    mock_credential.return_value = Mock()
    # Test implementation
```

#### 3. Async Test Failures
```python
# Use pytest-asyncio for async tests
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### Debug Mode

```bash
# Run tests with debug output
PYTHONPATH=. pytest tests/ -v -s --tb=long

# Run specific test with pdb
PYTHONPATH=. pytest tests/test_api.py::TestAPI::test_health_check -v -s --pdb
```

## Quality Gates

### Pre-commit Checks

```bash
# Run before committing
make test
make lint
make type-check
```

### Coverage Requirements

- **Minimum Coverage**: 70% (configurable in pytest.ini)
- **Critical Paths**: 90% coverage required
- **New Code**: 80% coverage required

### Performance Thresholds

- **API Response Time**: < 200ms for standard operations
- **Intent Classification**: < 100ms per request
- **Resource Operations**: < 5s for creation/deletion

## Troubleshooting

### Common Test Failures

1. **Authentication Errors**: Check environment variables and mocking
2. **Import Errors**: Verify PYTHONPATH and module structure
3. **Async Timeouts**: Increase timeout values or check mock responses
4. **Coverage Failures**: Add tests for uncovered code paths

### Test Data Management

```python
# Use factories for consistent test data
from faker import Faker

fake = Faker()

def create_test_user():
    return {
        'username': fake.user_name(),
        'email': fake.email(),
        'role': 'reader'
    }
```

## Best Practices

### Test Writing Guidelines

1. **Arrange-Act-Assert**: Structure tests clearly
2. **One Assertion Per Test**: Focus on single behaviors
3. **Descriptive Names**: Use clear, descriptive test names
4. **Independent Tests**: Avoid test dependencies
5. **Mock External Services**: Isolate units under test

### Test Maintenance

1. **Regular Updates**: Keep tests current with code changes
2. **Performance Monitoring**: Track test execution times
3. **Coverage Analysis**: Regularly review coverage reports
4. **Refactor Tests**: Maintain test code quality

## Future Enhancements

### Planned Improvements

1. **Visual Testing**: UI component testing
2. **Contract Testing**: API contract validation
3. **Security Testing**: Vulnerability scanning
4. **Chaos Engineering**: Failure injection testing
5. **A/B Testing**: Feature flag testing

### Test Infrastructure

1. **Test Containers**: Isolated test environments
2. **Test Data Factories**: Automated test data generation
3. **Parallel Execution**: Faster test runs
4. **Cross-browser Testing**: Multi-environment validation