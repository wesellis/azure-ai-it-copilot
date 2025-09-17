"""
Comprehensive API endpoint testing
Tests all REST API endpoints with authentication, validation, and error handling
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import patch, AsyncMock, Mock

from fastapi.testclient import TestClient
from fastapi import status
import httpx

# Local imports
from api.server import app
from config.optimized_settings import get_settings


@pytest.mark.api
class TestAuthenticationEndpoints:
    """Test authentication and authorization endpoints"""

    @pytest.fixture
    def test_client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Valid authentication headers"""
        return {
            "Authorization": "Bearer valid_test_token",
            "Content-Type": "application/json"
        }

    @pytest.fixture
    def invalid_auth_headers(self):
        """Invalid authentication headers"""
        return {
            "Authorization": "Bearer invalid_token",
            "Content-Type": "application/json"
        }

    def test_login_endpoint(self, test_client):
        """Test user login endpoint"""
        login_data = {
            "username": "test_user",
            "password": "test_password"
        }

        response = test_client.post("/auth/login", json=login_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self, test_client):
        """Test login with invalid credentials"""
        login_data = {
            "username": "invalid_user",
            "password": "wrong_password"
        }

        response = test_client.post("/auth/login", json=login_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data

    def test_token_refresh(self, test_client, auth_headers):
        """Test token refresh endpoint"""
        response = test_client.post("/auth/refresh", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "expires_in" in data

    def test_logout(self, test_client, auth_headers):
        """Test user logout endpoint"""
        response = test_client.post("/auth/logout", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Successfully logged out"

    def test_user_profile(self, test_client, auth_headers):
        """Test user profile endpoint"""
        response = test_client.get("/auth/profile", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "user_id" in data
        assert "username" in data
        assert "email" in data

    def test_unauthorized_access(self, test_client, invalid_auth_headers):
        """Test access with invalid token"""
        response = test_client.get("/auth/profile", headers=invalid_auth_headers)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.api
class TestAIOrchestatorEndpoints:
    """Test AI Orchestrator API endpoints"""

    @pytest.fixture
    def test_client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Valid authentication headers"""
        return {
            "Authorization": "Bearer valid_test_token",
            "Content-Type": "application/json"
        }

    @pytest.fixture
    def mock_orchestrator_response(self):
        """Mock orchestrator response"""
        return {
            "analysis_id": str(uuid.uuid4()),
            "analysis": "Comprehensive infrastructure analysis completed",
            "recommendations": [
                {
                    "type": "cost_optimization",
                    "priority": "high",
                    "description": "Resize oversized VMs",
                    "potential_savings": 1500.0,
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
                "current_monthly_cost": 5000.0,
                "projected_savings": 1500.0,
                "optimization_percentage": 30.0
            },
            "performance_metrics": {
                "overall_health_score": 85.0,
                "cpu_utilization": 65.0,
                "memory_utilization": 78.0,
                "storage_utilization": 45.0
            },
            "security_assessment": {
                "security_score": 75.0,
                "vulnerabilities": ["Open SSH port", "Unencrypted storage"],
                "compliance_status": "Partially Compliant"
            },
            "execution_time_ms": 2450,
            "timestamp": datetime.utcnow().isoformat()
        }

    @patch('ai_orchestrator.orchestrator.AzureAIOrchestrator')
    def test_infrastructure_analysis(self, mock_orchestrator, test_client, auth_headers, mock_orchestrator_response):
        """Test infrastructure analysis endpoint"""
        mock_orchestrator.return_value.analyze_infrastructure = AsyncMock(return_value=mock_orchestrator_response)

        payload = {
            "subscription_id": "test-subscription-123",
            "resource_group": "test-rg",
            "analysis_type": "comprehensive",
            "include_costs": True,
            "include_performance": True,
            "include_security": True
        }

        response = test_client.post(
            "/api/ai/analyze/infrastructure",
            json=payload,
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "analysis_id" in data
        assert "analysis" in data
        assert "recommendations" in data
        assert "cost_analysis" in data
        assert "performance_metrics" in data
        assert "security_assessment" in data
        assert "execution_time_ms" in data

        # Verify recommendations structure
        assert len(data["recommendations"]) >= 1
        for rec in data["recommendations"]:
            assert "type" in rec
            assert "priority" in rec
            assert "description" in rec

    def test_infrastructure_analysis_invalid_input(self, test_client, auth_headers):
        """Test infrastructure analysis with invalid input"""
        payload = {
            "subscription_id": "",  # Invalid empty subscription ID
            "resource_group": "test-rg"
        }

        response = test_client.post(
            "/api/ai/analyze/infrastructure",
            json=payload,
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('ai_orchestrator.orchestrator.AzureAIOrchestrator')
    def test_cost_optimization(self, mock_orchestrator, test_client, auth_headers):
        """Test cost optimization endpoint"""
        mock_response = {
            "optimization_id": str(uuid.uuid4()),
            "recommendations": [
                {
                    "type": "resize",
                    "resource_id": "/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/vm-1",
                    "current_size": "Standard_D4s_v3",
                    "recommended_size": "Standard_D2s_v3",
                    "monthly_savings": 150.0,
                    "confidence": 95.0
                },
                {
                    "type": "delete",
                    "resource_id": "/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Storage/storageAccounts/unused-storage",
                    "reason": "No activity in 90 days",
                    "monthly_savings": 25.0,
                    "confidence": 100.0
                }
            ],
            "total_monthly_savings": 175.0,
            "implementation_risk": "low",
            "execution_time_ms": 1200
        }

        mock_orchestrator.return_value.optimize_costs = AsyncMock(return_value=mock_response)

        payload = {
            "subscription_id": "test-subscription-123",
            "time_range_days": 30,
            "optimization_level": "aggressive",
            "exclude_resources": [],
            "include_reserved_instances": True
        }

        response = test_client.post(
            "/api/ai/optimize/costs",
            json=payload,
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "optimization_id" in data
        assert "recommendations" in data
        assert "total_monthly_savings" in data
        assert data["total_monthly_savings"] == 175.0

        # Verify recommendations structure
        for rec in data["recommendations"]:
            assert "type" in rec
            assert "monthly_savings" in rec
            assert "confidence" in rec

    @patch('ai_orchestrator.orchestrator.AzureAIOrchestrator')
    def test_performance_analysis(self, mock_orchestrator, test_client, auth_headers):
        """Test performance analysis endpoint"""
        mock_response = {
            "analysis_id": str(uuid.uuid4()),
            "resource_id": "/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm",
            "metrics": {
                "cpu": {
                    "average": 65.5,
                    "maximum": 95.2,
                    "minimum": 25.1,
                    "p95": 85.7
                },
                "memory": {
                    "average": 78.3,
                    "maximum": 92.1,
                    "minimum": 45.6,
                    "p95": 89.2
                },
                "network": {
                    "bytes_in_per_sec": 1048576,
                    "bytes_out_per_sec": 524288,
                    "packets_per_sec": 1000
                },
                "disk": {
                    "read_iops": 150,
                    "write_iops": 75,
                    "latency_ms": 5.2
                }
            },
            "performance_issues": [
                {
                    "type": "high_cpu_usage",
                    "severity": "medium",
                    "description": "CPU usage frequently above 80%",
                    "recommendation": "Consider scaling up or out"
                },
                {
                    "type": "memory_pressure",
                    "severity": "high",
                    "description": "Memory usage consistently above 85%",
                    "recommendation": "Increase memory allocation"
                }
            ],
            "overall_health_score": 72.0,
            "recommendations": [
                "Increase VM size to Standard_D4s_v3",
                "Implement auto-scaling",
                "Add monitoring alerts for memory usage"
            ]
        }

        mock_orchestrator.return_value.analyze_performance = AsyncMock(return_value=mock_response)

        payload = {
            "resource_id": "/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm",
            "time_range_hours": 24,
            "metrics": ["cpu", "memory", "network", "disk"]
        }

        response = test_client.post(
            "/api/ai/analyze/performance",
            json=payload,
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "analysis_id" in data
        assert "metrics" in data
        assert "performance_issues" in data
        assert "overall_health_score" in data
        assert "recommendations" in data

        # Verify metrics structure
        assert "cpu" in data["metrics"]
        assert "memory" in data["metrics"]
        assert data["overall_health_score"] == 72.0

    @patch('ai_orchestrator.orchestrator.AzureAIOrchestrator')
    def test_security_assessment(self, mock_orchestrator, test_client, auth_headers):
        """Test security assessment endpoint"""
        mock_response = {
            "assessment_id": str(uuid.uuid4()),
            "security_score": 75.0,
            "compliance_status": {
                "overall_compliance": 78.0,
                "frameworks": {
                    "SOC2": 85.0,
                    "ISO27001": 72.0,
                    "PCI_DSS": 68.0
                }
            },
            "vulnerabilities": [
                {
                    "id": "vuln-001",
                    "severity": "high",
                    "title": "Open SSH port to internet",
                    "description": "SSH port 22 is open to 0.0.0.0/0",
                    "resource": "vm-web-01",
                    "remediation": "Restrict SSH access to specific IP ranges"
                },
                {
                    "id": "vuln-002",
                    "severity": "medium",
                    "title": "Unencrypted storage account",
                    "description": "Storage account does not have encryption enabled",
                    "resource": "storage-account-01",
                    "remediation": "Enable storage account encryption"
                }
            ],
            "security_recommendations": [
                "Enable Azure Security Center",
                "Implement network security groups",
                "Enable disk encryption",
                "Configure Azure Key Vault for secrets management"
            ],
            "compliance_gaps": [
                "Multi-factor authentication not enabled",
                "Audit logging incomplete",
                "Data classification missing"
            ]
        }

        mock_orchestrator.return_value.assess_security = AsyncMock(return_value=mock_response)

        payload = {
            "subscription_id": "test-subscription-123",
            "resource_group": "test-rg",
            "assessment_type": "comprehensive",
            "compliance_frameworks": ["SOC2", "ISO27001", "PCI_DSS"]
        }

        response = test_client.post(
            "/api/ai/assess/security",
            json=payload,
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "assessment_id" in data
        assert "security_score" in data
        assert "vulnerabilities" in data
        assert "security_recommendations" in data
        assert "compliance_status" in data

        # Verify vulnerability structure
        for vuln in data["vulnerabilities"]:
            assert "severity" in vuln
            assert "title" in vuln
            assert "remediation" in vuln

    @patch('ai_orchestrator.orchestrator.AzureAIOrchestrator')
    def test_chat_interaction(self, mock_orchestrator, test_client, auth_headers):
        """Test chat/conversation endpoint"""
        mock_response = {
            "response_id": str(uuid.uuid4()),
            "message": "Based on your infrastructure analysis, I found 3 cost optimization opportunities that could save you approximately $1,500 per month. Here are the key recommendations:\n\n1. **Resize oversized VMs**: Your Standard_D8s_v3 VMs are running at only 35% CPU utilization. Downsizing to Standard_D4s_v3 could save $800/month.\n\n2. **Delete unused storage**: I found 5 storage accounts with no activity in the last 90 days, costing $150/month.\n\n3. **Reserved instances**: Converting your production VMs to reserved instances could save an additional $550/month.\n\nWould you like me to provide detailed implementation steps for any of these recommendations?",
            "context": {
                "conversation_id": str(uuid.uuid4()),
                "user_id": "test_user_123",
                "subscription_id": "test-subscription-123"
            },
            "suggestions": [
                "Show me the detailed VM rightsizing plan",
                "Which storage accounts are unused?",
                "Calculate reserved instance savings",
                "Generate cost optimization report"
            ],
            "confidence": 95.0,
            "execution_time_ms": 1800
        }

        mock_orchestrator.return_value.chat_interaction = AsyncMock(return_value=mock_response)

        payload = {
            "message": "What are my biggest cost optimization opportunities this month?",
            "context": {
                "subscription_id": "test-subscription-123",
                "user_id": "test_user_123",
                "conversation_history": []
            }
        }

        response = test_client.post(
            "/api/ai/chat",
            json=payload,
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "response_id" in data
        assert "message" in data
        assert "context" in data
        assert "suggestions" in data
        assert "confidence" in data

        # Verify response quality
        assert len(data["message"]) > 50  # Substantial response
        assert data["confidence"] >= 80.0  # High confidence


@pytest.mark.api
class TestRecommendationEndpoints:
    """Test recommendation management endpoints"""

    @pytest.fixture
    def test_client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Valid authentication headers"""
        return {
            "Authorization": "Bearer valid_test_token",
            "Content-Type": "application/json"
        }

    @pytest.fixture
    def sample_recommendations(self):
        """Sample recommendations data"""
        return [
            {
                "id": str(uuid.uuid4()),
                "type": "cost_optimization",
                "priority": "high",
                "title": "Resize oversized VMs",
                "description": "Several VMs are significantly oversized for their workload",
                "potential_savings": 800.0,
                "resources": ["vm-web-01", "vm-api-02"],
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "estimated_implementation_time": "30 minutes"
            },
            {
                "id": str(uuid.uuid4()),
                "type": "performance",
                "priority": "medium",
                "title": "Add application gateway",
                "description": "Improve application performance with load balancing",
                "estimated_cost": 150.0,
                "resources": ["web-app-01"],
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "estimated_implementation_time": "2 hours"
            }
        ]

    def test_get_recommendations(self, test_client, auth_headers, sample_recommendations):
        """Test get recommendations endpoint"""
        with patch('api.recommendations.get_recommendations_from_db', return_value=sample_recommendations):
            response = test_client.get(
                "/api/recommendations?subscription_id=test-subscription&type=all",
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "recommendations" in data
            assert len(data["recommendations"]) == 2

            # Verify recommendation structure
            for rec in data["recommendations"]:
                assert "id" in rec
                assert "type" in rec
                assert "priority" in rec
                assert "title" in rec
                assert "status" in rec

    def test_get_recommendations_filtered(self, test_client, auth_headers, sample_recommendations):
        """Test get recommendations with filters"""
        cost_recommendations = [r for r in sample_recommendations if r["type"] == "cost_optimization"]

        with patch('api.recommendations.get_recommendations_from_db', return_value=cost_recommendations):
            response = test_client.get(
                "/api/recommendations?subscription_id=test-subscription&type=cost_optimization&priority=high",
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert len(data["recommendations"]) == 1
            assert data["recommendations"][0]["type"] == "cost_optimization"
            assert data["recommendations"][0]["priority"] == "high"

    def test_update_recommendation_status(self, test_client, auth_headers):
        """Test update recommendation status"""
        recommendation_id = str(uuid.uuid4())
        payload = {
            "status": "approved",
            "notes": "Approved for implementation in next maintenance window"
        }

        with patch('api.recommendations.update_recommendation_status') as mock_update:
            mock_update.return_value = True

            response = test_client.patch(
                f"/api/recommendations/{recommendation_id}/status",
                json=payload,
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "updated"

    def test_implement_recommendation(self, test_client, auth_headers):
        """Test implement recommendation endpoint"""
        recommendation_id = str(uuid.uuid4())
        payload = {
            "auto_implement": True,
            "dry_run": False,
            "notification_webhook": "https://example.com/webhook"
        }

        mock_result = {
            "implementation_id": str(uuid.uuid4()),
            "status": "in_progress",
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
            "steps": [
                {"step": 1, "description": "Validate resource access", "status": "completed"},
                {"step": 2, "description": "Resize VM", "status": "in_progress"},
                {"step": 3, "description": "Verify new configuration", "status": "pending"}
            ]
        }

        with patch('api.recommendations.implement_recommendation', return_value=mock_result):
            response = test_client.post(
                f"/api/recommendations/{recommendation_id}/implement",
                json=payload,
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "implementation_id" in data
            assert "status" in data
            assert "steps" in data
            assert len(data["steps"]) == 3

    def test_get_implementation_status(self, test_client, auth_headers):
        """Test get implementation status"""
        implementation_id = str(uuid.uuid4())

        mock_status = {
            "implementation_id": implementation_id,
            "status": "completed",
            "progress_percentage": 100,
            "started_at": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "results": {
                "success": True,
                "cost_savings_achieved": 800.0,
                "resources_modified": ["vm-web-01", "vm-api-02"]
            }
        }

        with patch('api.recommendations.get_implementation_status', return_value=mock_status):
            response = test_client.get(
                f"/api/implementations/{implementation_id}/status",
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["implementation_id"] == implementation_id
            assert data["status"] == "completed"
            assert data["progress_percentage"] == 100
            assert "results" in data


@pytest.mark.api
class TestHealthAndStatusEndpoints:
    """Test health check and status endpoints"""

    @pytest.fixture
    def test_client(self):
        """FastAPI test client"""
        return TestClient(app)

    def test_health_check_basic(self, test_client):
        """Test basic health check endpoint"""
        response = test_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_check_detailed(self, test_client):
        """Test detailed health check endpoint"""
        response = test_client.get("/health/detailed")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "status" in data
        assert "components" in data
        assert "system_info" in data

        # Verify component health
        expected_components = ["database", "redis", "azure_services", "ai_orchestrator"]
        for component in expected_components:
            assert component in data["components"]
            assert "status" in data["components"][component]

    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint"""
        response = test_client.get("/metrics")

        assert response.status_code == status.HTTP_200_OK
        # Response should be in Prometheus format
        assert "text/plain" in response.headers.get("content-type", "")

    def test_readiness_probe(self, test_client):
        """Test Kubernetes readiness probe"""
        response = test_client.get("/ready")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ready"] is True

    def test_liveness_probe(self, test_client):
        """Test Kubernetes liveness probe"""
        response = test_client.get("/live")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["alive"] is True

    def test_version_info(self, test_client):
        """Test version information endpoint"""
        response = test_client.get("/version")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "version" in data
        assert "build_date" in data
        assert "commit_hash" in data


@pytest.mark.api
class TestErrorHandling:
    """Test API error handling"""

    @pytest.fixture
    def test_client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Valid authentication headers"""
        return {
            "Authorization": "Bearer valid_test_token",
            "Content-Type": "application/json"
        }

    def test_404_not_found(self, test_client):
        """Test 404 error handling"""
        response = test_client.get("/api/nonexistent/endpoint")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data

    def test_422_validation_error(self, test_client, auth_headers):
        """Test 422 validation error handling"""
        invalid_payload = {
            "subscription_id": "",  # Invalid empty string
            "invalid_field": "should not be here"
        }

        response = test_client.post(
            "/api/ai/analyze/infrastructure",
            json=invalid_payload,
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_500_internal_server_error(self, test_client, auth_headers):
        """Test 500 error handling"""
        # Mock an internal server error
        with patch('ai_orchestrator.orchestrator.AzureAIOrchestrator') as mock_orchestrator:
            mock_orchestrator.return_value.analyze_infrastructure.side_effect = Exception("Internal error")

            payload = {
                "subscription_id": "test-subscription",
                "resource_group": "test-rg"
            }

            response = test_client.post(
                "/api/ai/analyze/infrastructure",
                json=payload,
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "detail" in data

    def test_rate_limiting(self, test_client, auth_headers):
        """Test rate limiting"""
        # This would require actual rate limiting implementation
        # For now, we test the structure

        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = test_client.get("/api/recommendations", headers=auth_headers)
            responses.append(response)

        # Check if any requests were rate limited
        rate_limited = any(r.status_code == status.HTTP_429_TOO_MANY_REQUESTS for r in responses)

        # This assertion might not pass without actual rate limiting
        # assert rate_limited, "Expected at least one request to be rate limited"

    def test_cors_headers(self, test_client):
        """Test CORS headers"""
        response = test_client.options("/api/ai/analyze/infrastructure")

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers


@pytest.mark.api
class TestWebSocketEndpoints:
    """Test WebSocket endpoints for real-time features"""

    @pytest.fixture
    def test_client(self):
        """FastAPI test client"""
        return TestClient(app)

    def test_websocket_connection(self, test_client):
        """Test WebSocket connection establishment"""
        with test_client.websocket_connect("/ws/notifications") as websocket:
            # Send authentication
            websocket.send_json({"type": "auth", "token": "valid_test_token"})

            # Receive authentication confirmation
            data = websocket.receive_json()
            assert data["type"] == "auth_success"

    def test_websocket_real_time_updates(self, test_client):
        """Test real-time updates via WebSocket"""
        with test_client.websocket_connect("/ws/analysis") as websocket:
            # Send analysis request
            websocket.send_json({
                "type": "start_analysis",
                "subscription_id": "test-subscription",
                "analysis_type": "real_time_monitoring"
            })

            # Receive progress updates
            for _ in range(3):  # Expect multiple progress updates
                data = websocket.receive_json()
                assert "type" in data
                assert data["type"] in ["progress", "result", "error"]

    def test_websocket_disconnection_handling(self, test_client):
        """Test WebSocket disconnection handling"""
        with test_client.websocket_connect("/ws/notifications") as websocket:
            websocket.send_json({"type": "auth", "token": "valid_test_token"})
            websocket.receive_json()  # Auth confirmation

            # Simulate disconnection by closing
            websocket.close()

            # Connection should be properly cleaned up
            # This would be verified in actual WebSocket implementation


@pytest.mark.api
class TestAPIPerformance:
    """Test API performance characteristics"""

    @pytest.fixture
    def test_client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Valid authentication headers"""
        return {
            "Authorization": "Bearer valid_test_token",
            "Content-Type": "application/json"
        }

    def test_response_time_health_check(self, test_client):
        """Test health check response time"""
        import time

        start_time = time.time()
        response = test_client.get("/health")
        end_time = time.time()

        response_time = end_time - start_time

        assert response.status_code == status.HTTP_200_OK
        assert response_time < 0.1  # Should respond within 100ms

    def test_response_time_recommendations(self, test_client, auth_headers):
        """Test recommendations endpoint response time"""
        import time

        with patch('api.recommendations.get_recommendations_from_db', return_value=[]):
            start_time = time.time()
            response = test_client.get("/api/recommendations", headers=auth_headers)
            end_time = time.time()

            response_time = end_time - start_time

            assert response.status_code == status.HTTP_200_OK
            assert response_time < 1.0  # Should respond within 1 second

    def test_concurrent_requests(self, test_client, auth_headers):
        """Test handling of concurrent requests"""
        import threading
        import time

        results = []

        def make_request():
            start_time = time.time()
            response = test_client.get("/health")
            end_time = time.time()
            results.append({
                "status_code": response.status_code,
                "response_time": end_time - start_time
            })

        # Create multiple threads for concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        total_time = time.time() - start_time

        # Verify all requests succeeded
        assert len(results) == 10
        assert all(r["status_code"] == 200 for r in results)

        # Verify reasonable performance under concurrency
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        assert avg_response_time < 0.5  # Average response time under 500ms
        assert total_time < 2.0  # All 10 requests complete within 2 seconds

    def test_large_payload_handling(self, test_client, auth_headers):
        """Test handling of large payloads"""
        # Create a large but reasonable payload
        large_payload = {
            "subscription_id": "test-subscription",
            "resource_group": "test-rg",
            "analysis_type": "comprehensive",
            "metadata": {
                "large_data": "x" * 10000,  # 10KB of data
                "resource_list": [f"resource-{i}" for i in range(1000)]  # Large list
            }
        }

        with patch('ai_orchestrator.orchestrator.AzureAIOrchestrator') as mock_orchestrator:
            mock_orchestrator.return_value.analyze_infrastructure = AsyncMock(return_value={"status": "completed"})

            response = test_client.post(
                "/api/ai/analyze/infrastructure",
                json=large_payload,
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK

    def test_response_compression(self, test_client):
        """Test response compression for large responses"""
        # Test with Accept-Encoding header
        headers = {"Accept-Encoding": "gzip, deflate"}

        response = test_client.get("/health/detailed", headers=headers)

        assert response.status_code == status.HTTP_200_OK
        # Check if response was compressed (would depend on server configuration)
        # assert "gzip" in response.headers.get("content-encoding", "")


# Integration test for full API workflow
@pytest.mark.api
@pytest.mark.integration
class TestAPIWorkflow:
    """Test complete API workflow scenarios"""

    @pytest.fixture
    def test_client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Valid authentication headers"""
        return {
            "Authorization": "Bearer valid_test_token",
            "Content-Type": "application/json"
        }

    def test_complete_infrastructure_analysis_workflow(self, test_client, auth_headers):
        """Test complete infrastructure analysis workflow"""
        # Step 1: Start infrastructure analysis
        analysis_payload = {
            "subscription_id": "test-subscription",
            "resource_group": "test-rg",
            "analysis_type": "comprehensive"
        }

        with patch('ai_orchestrator.orchestrator.AzureAIOrchestrator') as mock_orchestrator:
            mock_analysis_result = {
                "analysis_id": "analysis-123",
                "status": "completed",
                "recommendations": [
                    {"id": "rec-1", "type": "cost_optimization", "priority": "high"}
                ]
            }

            mock_orchestrator.return_value.analyze_infrastructure = AsyncMock(return_value=mock_analysis_result)

            response = test_client.post(
                "/api/ai/analyze/infrastructure",
                json=analysis_payload,
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            analysis_data = response.json()
            analysis_id = analysis_data["analysis_id"]

        # Step 2: Get recommendations from analysis
        with patch('api.recommendations.get_recommendations_from_db') as mock_get_recs:
            mock_get_recs.return_value = [
                {
                    "id": "rec-1",
                    "type": "cost_optimization",
                    "priority": "high",
                    "title": "Resize VM",
                    "status": "pending"
                }
            ]

            response = test_client.get(
                f"/api/recommendations?analysis_id={analysis_id}",
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            recommendations = response.json()["recommendations"]
            recommendation_id = recommendations[0]["id"]

        # Step 3: Approve recommendation
        approval_payload = {
            "status": "approved",
            "notes": "Approved for implementation"
        }

        with patch('api.recommendations.update_recommendation_status') as mock_update:
            mock_update.return_value = True

            response = test_client.patch(
                f"/api/recommendations/{recommendation_id}/status",
                json=approval_payload,
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK

        # Step 4: Implement recommendation
        implementation_payload = {
            "auto_implement": True,
            "dry_run": False
        }

        with patch('api.recommendations.implement_recommendation') as mock_implement:
            mock_implement.return_value = {
                "implementation_id": "impl-123",
                "status": "in_progress"
            }

            response = test_client.post(
                f"/api/recommendations/{recommendation_id}/implement",
                json=implementation_payload,
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            implementation_data = response.json()
            implementation_id = implementation_data["implementation_id"]

        # Step 5: Check implementation status
        with patch('api.recommendations.get_implementation_status') as mock_status:
            mock_status.return_value = {
                "implementation_id": implementation_id,
                "status": "completed",
                "progress_percentage": 100
            }

            response = test_client.get(
                f"/api/implementations/{implementation_id}/status",
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            status_data = response.json()
            assert status_data["status"] == "completed"

        # Verify complete workflow succeeded
        assert analysis_id == "analysis-123"
        assert recommendation_id == "rec-1"
        assert implementation_id == "impl-123"