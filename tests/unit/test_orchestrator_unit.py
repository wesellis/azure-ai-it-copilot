"""
Unit tests for AI Orchestrator
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ai_orchestrator.orchestrator import AzureAIOrchestrator, IntentType, BaseAgent


class TestAzureAIOrchestrator:
    """Test suite for Azure AI Orchestrator"""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_openai_client, mock_redis_client):
        """Test orchestrator initialization"""
        with patch("ai_orchestrator.orchestrator.AzureChatOpenAI", return_value=mock_openai_client):
            with patch("ai_orchestrator.orchestrator.redis.Redis", return_value=mock_redis_client):
                orchestrator = AzureAIOrchestrator()

                assert orchestrator.llm is not None
                assert orchestrator.redis_client is not None
                assert orchestrator.memory is not None
                assert len(orchestrator.agents) > 0

    @pytest.mark.asyncio
    async def test_classify_intent_resource_create(self, mock_orchestrator):
        """Test intent classification for resource creation"""
        command = "Create a new virtual machine with 8GB RAM"

        mock_orchestrator.llm.ainvoke = AsyncMock(
            return_value=Mock(content="resource_create")
        )

        intent = await mock_orchestrator.classify_intent(command)
        assert intent == IntentType.RESOURCE_CREATE

    @pytest.mark.asyncio
    async def test_classify_intent_cost_optimization(self, mock_orchestrator):
        """Test intent classification for cost optimization"""
        command = "Optimize our Azure costs and find savings"

        mock_orchestrator.llm.ainvoke = AsyncMock(
            return_value=Mock(content="cost_optimization")
        )

        intent = await mock_orchestrator.classify_intent(command)
        assert intent == IntentType.COST_OPTIMIZATION

    @pytest.mark.asyncio
    async def test_classify_intent_incident_diagnosis(self, mock_orchestrator):
        """Test intent classification for incident diagnosis"""
        command = "Diagnose high CPU usage on production server"

        mock_orchestrator.llm.ainvoke = AsyncMock(
            return_value=Mock(content="incident_diagnosis")
        )

        intent = await mock_orchestrator.classify_intent(command)
        assert intent == IntentType.INCIDENT_DIAGNOSIS

    @pytest.mark.asyncio
    async def test_validate_permissions_reader_role(self, mock_orchestrator):
        """Test permission validation for reader role"""
        context = {"user_role": "reader"}

        # Reader can query
        can_query = await mock_orchestrator.validate_permissions(
            IntentType.RESOURCE_QUERY, context
        )
        assert can_query is True

        # Reader cannot create
        can_create = await mock_orchestrator.validate_permissions(
            IntentType.RESOURCE_CREATE, context
        )
        assert can_create is False

    @pytest.mark.asyncio
    async def test_validate_permissions_owner_role(self, mock_orchestrator):
        """Test permission validation for owner role"""
        context = {"user_role": "owner"}

        # Owner can do everything
        for intent_type in IntentType:
            can_perform = await mock_orchestrator.validate_permissions(
                intent_type, context
            )
            assert can_perform is True

    @pytest.mark.asyncio
    async def test_process_command_success(self, mock_orchestrator):
        """Test successful command processing"""
        command = "List all virtual machines"
        context = {"user_role": "contributor"}

        # Mock intent classification
        mock_orchestrator.classify_intent = AsyncMock(
            return_value=IntentType.RESOURCE_QUERY
        )

        # Mock agent
        mock_agent = Mock()
        mock_agent.create_plan = AsyncMock(return_value={
            "operation_type": "query",
            "requires_approval": False
        })
        mock_agent.execute = AsyncMock(return_value={
            "status": "success",
            "vms": ["vm1", "vm2"]
        })

        mock_orchestrator.agents[IntentType.RESOURCE_QUERY] = mock_agent

        result = await mock_orchestrator.process_command(command, context)

        assert result["status"] == "success"
        assert "vms" in result
        mock_agent.create_plan.assert_called_once()
        mock_agent.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_command_requires_approval(self, mock_orchestrator):
        """Test command processing that requires approval"""
        command = "Delete all VMs in test resource group"
        context = {"user_role": "owner"}

        mock_orchestrator.classify_intent = AsyncMock(
            return_value=IntentType.RESOURCE_DELETE
        )

        # Mock agent with plan requiring approval
        mock_agent = Mock()
        mock_agent.create_plan = AsyncMock(return_value={
            "operation_type": "delete",
            "requires_approval": True
        })

        mock_orchestrator.agents[IntentType.RESOURCE_DELETE] = mock_agent
        mock_orchestrator.request_approval = AsyncMock(return_value=False)

        result = await mock_orchestrator.process_command(command, context)

        assert result["status"] == "cancelled"
        assert "cancelled by user" in result["message"]
        mock_orchestrator.request_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_command_permission_denied(self, mock_orchestrator):
        """Test command processing with insufficient permissions"""
        command = "Delete resource group"
        context = {"user_role": "reader"}

        mock_orchestrator.classify_intent = AsyncMock(
            return_value=IntentType.RESOURCE_DELETE
        )

        result = await mock_orchestrator.process_command(command, context)

        assert result["status"] == "error"
        assert "Insufficient permissions" in result["message"]

    @pytest.mark.asyncio
    async def test_process_command_with_caching(self, mock_orchestrator):
        """Test that command results are cached"""
        command = "Get VM status"
        context = {"user_role": "contributor"}

        mock_orchestrator.classify_intent = AsyncMock(
            return_value=IntentType.RESOURCE_QUERY
        )

        mock_agent = Mock()
        mock_agent.create_plan = AsyncMock(return_value={
            "operation_type": "query",
            "requires_approval": False
        })
        mock_agent.execute = AsyncMock(return_value={
            "status": "success",
            "data": "test"
        })

        mock_orchestrator.agents[IntentType.RESOURCE_QUERY] = mock_agent

        result = await mock_orchestrator.process_command(command, context)

        # Verify Redis cache was called
        mock_orchestrator.redis_client.setex.assert_called_once()
        cache_call = mock_orchestrator.redis_client.setex.call_args
        assert cache_call[0][1] == 3600  # TTL
        assert "command" in cache_call[0][2]  # Cached data

    @pytest.mark.asyncio
    async def test_request_approval_auto_approve_safe(self, mock_orchestrator):
        """Test auto-approval for safe operations"""
        plan = {
            "operation_type": "query",
            "action": "list_vms"
        }

        approved = await mock_orchestrator.request_approval(plan)
        assert approved is True

    @pytest.mark.asyncio
    async def test_request_approval_deny_delete(self, mock_orchestrator):
        """Test that delete operations are not auto-approved"""
        plan = {
            "operation_type": "delete",
            "action": "delete_resource_group",
            "target": "production-rg"
        }

        approved = await mock_orchestrator.request_approval(plan)
        assert approved is False

    def test_get_status(self, mock_orchestrator):
        """Test getting orchestrator status"""
        status = mock_orchestrator.get_status()

        assert status["status"] == "healthy"
        assert status["agents_loaded"] > 0
        assert "memory_size" in status
        assert status["redis_connected"] is True


class TestBaseAgent:
    """Test suite for Base Agent"""

    @pytest.mark.asyncio
    async def test_base_agent_initialization(self, mock_orchestrator):
        """Test base agent initialization"""
        agent = BaseAgent(mock_orchestrator)

        assert agent.orchestrator == mock_orchestrator
        assert agent.llm == mock_orchestrator.llm

    @pytest.mark.asyncio
    async def test_base_agent_abstract_methods(self, mock_orchestrator):
        """Test that abstract methods raise NotImplementedError"""
        agent = BaseAgent(mock_orchestrator)

        with pytest.raises(NotImplementedError):
            await agent.create_plan("test command", {})

        with pytest.raises(NotImplementedError):
            await agent.execute({})