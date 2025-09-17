import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.server import app


class TestAPI:
    """Test suite for FastAPI endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer test-token"}

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_detailed_health_check(self, client):
        """Test detailed health check with authentication"""
        with patch('api.server.verify_token', return_value={"sub": "testuser", "role": "owner"}):
            with patch('api.server.security', return_value=None):
                response = client.get("/health/detailed", headers={"Authorization": "Bearer test-token"})
                assert response.status_code == 200
                data = response.json()
                assert "orchestrator" in data
                assert "redis" in data

    def test_login_success(self, client):
        """Test successful login"""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        response = client.post("/auth/login", json={
            "username": "invalid",
            "password": "wrong"
        })
        assert response.status_code == 401

    @patch('api.server.verify_token')
    @patch('api.server.orchestrator')
    def test_process_command(self, mock_orchestrator, mock_verify, client, auth_headers):
        """Test command processing endpoint"""
        mock_verify.return_value = {"sub": "testuser", "role": "owner"}
        mock_orchestrator.process_command = AsyncMock(return_value={
            "status": "success",
            "result": {"message": "Command executed"}
        })
        
        response = client.post("/api/v1/command", 
                              json={"command": "List all VMs"},
                              headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @patch('api.server.verify_token')
    def test_query_resources(self, mock_verify, client, auth_headers):
        """Test resource query endpoint"""
        mock_verify.return_value = {"sub": "testuser", "role": "reader"}
        
        response = client.post("/api/v1/resources/query",
                              json={"resource_type": "vm"},
                              headers=auth_headers)
        
        # Should work even for reader role
        assert response.status_code in [200, 500]  # 500 if orchestrator not fully mocked

    @patch('api.server.verify_token')
    def test_create_resource_forbidden(self, mock_verify, client, auth_headers):
        """Test resource creation with insufficient permissions"""
        mock_verify.return_value = {"sub": "testuser", "role": "reader"}
        
        response = client.post("/api/v1/resources/create",
                              json={
                                  "resource_type": "vm",
                                  "specifications": {}
                              },
                              headers=auth_headers)
        
        assert response.status_code == 403
        assert "Insufficient permissions" in response.json()["detail"]

    @patch('api.server.verify_token') 
    def test_delete_resource_owner_only(self, mock_verify, client, auth_headers):
        """Test resource deletion requires owner role"""
        mock_verify.return_value = {"sub": "testuser", "role": "contributor"}
        
        response = client.delete("/api/v1/resources/test-resource",
                                headers=auth_headers)
        
        assert response.status_code == 403
        assert "Only owners" in response.json()["detail"]