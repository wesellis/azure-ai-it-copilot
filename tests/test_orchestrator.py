import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_orchestrator.orchestrator import AzureAIOrchestrator, IntentType


class TestAzureAIOrchestrator:
    """Test suite for Azure AI Orchestrator"""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables"""
        monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "test-subscription")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "6379")

    @pytest.fixture
    @patch('ai_orchestrator.orchestrator.DefaultAzureCredential')
    @patch('ai_orchestrator.orchestrator.redis.Redis')
    @patch('ai_orchestrator.orchestrator.AzureChatOpenAI')
    def orchestrator(self, mock_llm, mock_redis, mock_credential, mock_env):
        """Create orchestrator instance with mocked dependencies"""
        mock_redis.return_value.ping.return_value = True
        return AzureAIOrchestrator()

    @pytest.mark.asyncio
    async def test_classify_intent_resource_create(self, orchestrator):
        """Test intent classification for resource creation"""
        command = "Create a Linux VM with 8GB RAM in East US"
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "resource_create"
        orchestrator.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        intent = await orchestrator.classify_intent(command)
        assert intent == IntentType.RESOURCE_CREATE

    @pytest.mark.asyncio
    async def test_classify_intent_incident_diagnosis(self, orchestrator):
        """Test intent classification for incident diagnosis"""
        command = "Diagnose high CPU usage on vm-prod-001"
        
        mock_response = Mock()
        mock_response.content = "incident_diagnosis"
        orchestrator.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        intent = await orchestrator.classify_intent(command)
        assert intent == IntentType.INCIDENT_DIAGNOSIS

    @pytest.mark.asyncio
    async def test_validate_permissions_owner(self, orchestrator):
        """Test permission validation for owner role"""
        context = {"user_role": "owner"}
        
        # Owner should have all permissions
        for intent in IntentType:
            if intent != IntentType.UNKNOWN:
                assert await orchestrator.validate_permissions(intent, context) == True

    @pytest.mark.asyncio
    async def test_validate_permissions_reader(self, orchestrator):
        """Test permission validation for reader role"""
        context = {"user_role": "reader"}
        
        # Reader should only have query permissions
        assert await orchestrator.validate_permissions(IntentType.RESOURCE_QUERY, context) == True
        assert await orchestrator.validate_permissions(IntentType.RESOURCE_CREATE, context) == False
        assert await orchestrator.validate_permissions(IntentType.RESOURCE_DELETE, context) == False

    @pytest.mark.asyncio
    async def test_process_command_success(self, orchestrator):
        """Test successful command processing"""
        command = "List all VMs"
        
        # Mock intent classification
        mock_response = Mock()
        mock_response.content = "resource_query"
        orchestrator.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.create_plan = AsyncMock(return_value={"requires_approval": False})
        mock_agent.execute = AsyncMock(return_value={"status": "success", "data": []})
        orchestrator.agents[IntentType.RESOURCE_QUERY] = mock_agent
        
        result = await orchestrator.process_command(command)
        
        assert result["status"] == "success"
        mock_agent.create_plan.assert_called_once()
        mock_agent.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_command_insufficient_permissions(self, orchestrator):
        """Test command processing with insufficient permissions"""
        command = "Delete all VMs"
        context = {"user_role": "reader"}
        
        mock_response = Mock()
        mock_response.content = "resource_delete"
        orchestrator.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await orchestrator.process_command(command, context)
        
        assert result["status"] == "error"
        assert "Insufficient permissions" in result["message"]

    def test_get_status(self, orchestrator):
        """Test orchestrator status retrieval"""
        status = orchestrator.get_status()
        
        assert status["status"] == "healthy"
        assert "agents_loaded" in status
        assert status["agents_loaded"] > 0
        assert "redis_connected" in status