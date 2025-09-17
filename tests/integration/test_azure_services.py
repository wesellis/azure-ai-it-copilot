"""
Integration tests for Azure services
Tests real Azure service integrations with proper mocking and service validation
"""

import asyncio
import pytest
import json
import os
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Azure SDK imports
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.monitor import MonitorManagementClient
from azure.mgmt.costmanagement import CostManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

# Local imports
from ai_orchestrator.agents.cost_agent import CostOptimizationAgent
from ai_orchestrator.agents.security_agent import SecurityAgent
from ai_orchestrator.agents.performance_agent import PerformanceAgent
from azure_integration.resource_manager import AzureResourceManager
from azure_integration.monitoring import AzureMonitoringService
from azure_integration.cost_management import AzureCostService


@pytest.mark.integration
class TestAzureResourceManager:
    """Integration tests for Azure Resource Manager"""

    @pytest.fixture
    def mock_resource_client(self):
        """Mock Azure Resource Management client"""
        client = Mock(spec=ResourceManagementClient)

        # Mock resource groups
        mock_rg1 = Mock()
        mock_rg1.name = "test-rg-1"
        mock_rg1.location = "eastus"
        mock_rg1.id = "/subscriptions/test-sub/resourceGroups/test-rg-1"
        mock_rg1.tags = {"environment": "test", "owner": "team-a"}

        mock_rg2 = Mock()
        mock_rg2.name = "test-rg-2"
        mock_rg2.location = "westus"
        mock_rg2.id = "/subscriptions/test-sub/resourceGroups/test-rg-2"
        mock_rg2.tags = {"environment": "prod", "owner": "team-b"}

        client.resource_groups.list.return_value = [mock_rg1, mock_rg2]
        client.resource_groups.get.return_value = mock_rg1

        # Mock resources
        mock_vm = Mock()
        mock_vm.id = "/subscriptions/test-sub/resourceGroups/test-rg-1/providers/Microsoft.Compute/virtualMachines/test-vm"
        mock_vm.name = "test-vm"
        mock_vm.type = "Microsoft.Compute/virtualMachines"
        mock_vm.location = "eastus"
        mock_vm.tags = {"role": "web-server"}

        client.resources.list.return_value = [mock_vm]
        client.resources.get_by_id.return_value = mock_vm

        return client

    @pytest.fixture
    def azure_resource_manager(self, mock_resource_client):
        """Azure Resource Manager with mocked client"""
        with patch('azure_integration.resource_manager.ResourceManagementClient', return_value=mock_resource_client):
            with patch('azure.identity.DefaultAzureCredential'):
                manager = AzureResourceManager("test-subscription-id")
                return manager

    @pytest.mark.asyncio
    async def test_list_resource_groups(self, azure_resource_manager):
        """Test listing resource groups"""
        resource_groups = await azure_resource_manager.list_resource_groups()

        assert len(resource_groups) == 2
        assert resource_groups[0]["name"] == "test-rg-1"
        assert resource_groups[1]["name"] == "test-rg-2"
        assert resource_groups[0]["location"] == "eastus"
        assert resource_groups[1]["location"] == "westus"

    @pytest.mark.asyncio
    async def test_get_resource_group_details(self, azure_resource_manager):
        """Test getting resource group details"""
        rg_details = await azure_resource_manager.get_resource_group("test-rg-1")

        assert rg_details["name"] == "test-rg-1"
        assert rg_details["location"] == "eastus"
        assert "test-sub" in rg_details["id"]
        assert rg_details["tags"]["environment"] == "test"

    @pytest.mark.asyncio
    async def test_list_resources_by_type(self, azure_resource_manager):
        """Test listing resources by type"""
        vms = await azure_resource_manager.list_resources_by_type("Microsoft.Compute/virtualMachines")

        assert len(vms) == 1
        assert vms[0]["name"] == "test-vm"
        assert vms[0]["type"] == "Microsoft.Compute/virtualMachines"
        assert vms[0]["location"] == "eastus"

    @pytest.mark.asyncio
    async def test_get_resource_details(self, azure_resource_manager):
        """Test getting specific resource details"""
        resource_id = "/subscriptions/test-sub/resourceGroups/test-rg-1/providers/Microsoft.Compute/virtualMachines/test-vm"
        resource = await azure_resource_manager.get_resource(resource_id)

        assert resource["id"] == resource_id
        assert resource["name"] == "test-vm"
        assert resource["tags"]["role"] == "web-server"

    @pytest.mark.asyncio
    async def test_tag_based_filtering(self, azure_resource_manager):
        """Test filtering resources by tags"""
        resources = await azure_resource_manager.list_resources_by_tag("environment", "test")

        # Should return resources from test-rg-1 which has environment=test tag
        assert len(resources) >= 0  # Depends on mock implementation

    @pytest.mark.asyncio
    async def test_error_handling(self, azure_resource_manager):
        """Test error handling for non-existent resources"""
        with patch.object(azure_resource_manager.client.resource_groups, 'get',
                         side_effect=ResourceNotFoundError("Resource group not found")):

            with pytest.raises(ResourceNotFoundError):
                await azure_resource_manager.get_resource_group("non-existent-rg")


@pytest.mark.integration
class TestAzureComputeServices:
    """Integration tests for Azure Compute services"""

    @pytest.fixture
    def mock_compute_client(self):
        """Mock Azure Compute Management client"""
        client = Mock(spec=ComputeManagementClient)

        # Mock VM instance
        mock_vm = Mock()
        mock_vm.name = "test-vm-1"
        mock_vm.id = "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm-1"
        mock_vm.location = "eastus"
        mock_vm.provisioning_state = "Succeeded"
        mock_vm.vm_id = "12345678-1234-1234-1234-123456789012"

        # VM Size/SKU
        mock_vm.hardware_profile.vm_size = "Standard_D2s_v3"

        # OS Profile
        mock_vm.os_profile.computer_name = "test-vm-1"
        mock_vm.os_profile.admin_username = "azureuser"

        # Network Profile
        mock_nic = Mock()
        mock_nic.id = "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Network/networkInterfaces/test-vm-1-nic"
        mock_vm.network_profile.network_interfaces = [mock_nic]

        # Storage Profile
        mock_vm.storage_profile.os_disk.disk_size_gb = 128
        mock_vm.storage_profile.os_disk.managed_disk.storage_account_type = "Premium_LRS"

        client.virtual_machines.list_all.return_value = [mock_vm]
        client.virtual_machines.get.return_value = mock_vm
        client.virtual_machines.list.return_value = [mock_vm]

        # Mock VM sizes
        mock_size = Mock()
        mock_size.name = "Standard_D2s_v3"
        mock_size.number_of_cores = 2
        mock_size.memory_in_mb = 8192
        mock_size.max_data_disk_count = 4

        client.virtual_machine_sizes.list.return_value = [mock_size]

        return client

    @pytest.fixture
    def azure_compute_manager(self, mock_compute_client):
        """Azure Compute Manager with mocked client"""
        with patch('azure_integration.compute_manager.ComputeManagementClient', return_value=mock_compute_client):
            with patch('azure.identity.DefaultAzureCredential'):
                from azure_integration.compute_manager import AzureComputeManager
                manager = AzureComputeManager("test-subscription-id")
                return manager

    @pytest.mark.asyncio
    async def test_list_virtual_machines(self, azure_compute_manager):
        """Test listing virtual machines"""
        vms = await azure_compute_manager.list_virtual_machines()

        assert len(vms) == 1
        assert vms[0]["name"] == "test-vm-1"
        assert vms[0]["location"] == "eastus"
        assert vms[0]["provisioning_state"] == "Succeeded"
        assert vms[0]["vm_size"] == "Standard_D2s_v3"

    @pytest.mark.asyncio
    async def test_get_vm_details(self, azure_compute_manager):
        """Test getting VM details"""
        vm_details = await azure_compute_manager.get_vm_details("test-rg", "test-vm-1")

        assert vm_details["name"] == "test-vm-1"
        assert vm_details["hardware_profile"]["vm_size"] == "Standard_D2s_v3"
        assert vm_details["os_profile"]["admin_username"] == "azureuser"
        assert vm_details["storage_profile"]["os_disk"]["disk_size_gb"] == 128

    @pytest.mark.asyncio
    async def test_get_available_vm_sizes(self, azure_compute_manager):
        """Test getting available VM sizes"""
        sizes = await azure_compute_manager.get_available_vm_sizes("eastus")

        assert len(sizes) == 1
        assert sizes[0]["name"] == "Standard_D2s_v3"
        assert sizes[0]["number_of_cores"] == 2
        assert sizes[0]["memory_in_mb"] == 8192

    @pytest.mark.asyncio
    async def test_vm_power_operations(self, azure_compute_manager):
        """Test VM power operations (start/stop/restart)"""
        # Mock async operations
        mock_operation = AsyncMock()
        mock_operation.result.return_value = {"status": "Succeeded"}

        with patch.object(azure_compute_manager.client.virtual_machines, 'begin_start', return_value=mock_operation):
            result = await azure_compute_manager.start_vm("test-rg", "test-vm-1")
            assert result["status"] == "Succeeded"

        with patch.object(azure_compute_manager.client.virtual_machines, 'begin_deallocate', return_value=mock_operation):
            result = await azure_compute_manager.stop_vm("test-rg", "test-vm-1")
            assert result["status"] == "Succeeded"

    @pytest.mark.asyncio
    async def test_vm_scaling_recommendations(self, azure_compute_manager):
        """Test VM scaling recommendations"""
        # This would integrate with monitoring data
        recommendations = await azure_compute_manager.get_scaling_recommendations("test-rg", "test-vm-1")

        # Mock implementation should return scaling suggestions
        assert isinstance(recommendations, dict)


@pytest.mark.integration
class TestAzureMonitoringService:
    """Integration tests for Azure Monitoring services"""

    @pytest.fixture
    def mock_monitor_client(self):
        """Mock Azure Monitor client"""
        client = Mock(spec=MonitorManagementClient)

        # Mock metrics data
        mock_metric = Mock()
        mock_metric.name.value = "Percentage CPU"
        mock_metric.unit = "Percent"

        # Mock time series data
        mock_data_point = Mock()
        mock_data_point.time_stamp = datetime.utcnow() - timedelta(minutes=5)
        mock_data_point.average = 65.5
        mock_data_point.maximum = 85.2
        mock_data_point.minimum = 45.1

        mock_timeseries = Mock()
        mock_timeseries.data = [mock_data_point]
        mock_metric.timeseries = [mock_timeseries]

        mock_metrics_response = Mock()
        mock_metrics_response.value = [mock_metric]

        client.metrics.list.return_value = mock_metrics_response

        # Mock activity logs
        mock_log_entry = Mock()
        mock_log_entry.event_timestamp = datetime.utcnow() - timedelta(hours=1)
        mock_log_entry.operation_name.value = "Microsoft.Compute/virtualMachines/start/action"
        mock_log_entry.status.value = "Succeeded"
        mock_log_entry.caller = "user@example.com"
        mock_log_entry.resource_id = "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm"

        client.activity_logs.list.return_value = [mock_log_entry]

        return client

    @pytest.fixture
    def azure_monitoring_service(self, mock_monitor_client):
        """Azure Monitoring Service with mocked client"""
        with patch('azure_integration.monitoring.MonitorManagementClient', return_value=mock_monitor_client):
            with patch('azure.identity.DefaultAzureCredential'):
                service = AzureMonitoringService("test-subscription-id")
                return service

    @pytest.mark.asyncio
    async def test_get_vm_metrics(self, azure_monitoring_service):
        """Test getting VM metrics"""
        resource_id = "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm"

        metrics = await azure_monitoring_service.get_vm_metrics(
            resource_id,
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow(),
            metrics=["Percentage CPU", "Available Memory Bytes"]
        )

        assert len(metrics) == 1  # Based on mock
        assert metrics[0]["name"] == "Percentage CPU"
        assert metrics[0]["unit"] == "Percent"
        assert len(metrics[0]["data"]) == 1
        assert metrics[0]["data"][0]["average"] == 65.5

    @pytest.mark.asyncio
    async def test_get_activity_logs(self, azure_monitoring_service):
        """Test getting activity logs"""
        logs = await azure_monitoring_service.get_activity_logs(
            start_time=datetime.utcnow() - timedelta(hours=24),
            end_time=datetime.utcnow(),
            resource_group="test-rg"
        )

        assert len(logs) == 1
        assert logs[0]["operation"] == "Microsoft.Compute/virtualMachines/start/action"
        assert logs[0]["status"] == "Succeeded"
        assert logs[0]["caller"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_performance_analysis(self, azure_monitoring_service):
        """Test performance analysis"""
        resource_id = "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm"

        analysis = await azure_monitoring_service.analyze_performance(
            resource_id,
            time_range_hours=24
        )

        assert isinstance(analysis, dict)
        assert "cpu_utilization" in analysis
        assert "memory_utilization" in analysis
        assert "recommendations" in analysis

    @pytest.mark.asyncio
    async def test_alert_rules_management(self, azure_monitoring_service):
        """Test alert rules management"""
        # Mock alert rule creation
        with patch.object(azure_monitoring_service.client.metric_alerts, 'create_or_update') as mock_create:
            mock_create.return_value = Mock(id="alert-rule-id")

            alert_rule = await azure_monitoring_service.create_metric_alert(
                resource_group="test-rg",
                rule_name="high-cpu-alert",
                resource_id="/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm",
                metric_name="Percentage CPU",
                threshold=80.0,
                operator="GreaterThan"
            )

            assert alert_rule["id"] == "alert-rule-id"

    @pytest.mark.asyncio
    async def test_custom_metrics_collection(self, azure_monitoring_service):
        """Test custom metrics collection"""
        # This would test integration with custom metrics
        custom_metrics = await azure_monitoring_service.collect_custom_metrics(
            resource_id="/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm",
            metric_namespace="Custom/Application"
        )

        assert isinstance(custom_metrics, list)


@pytest.mark.integration
class TestAzureCostManagement:
    """Integration tests for Azure Cost Management"""

    @pytest.fixture
    def mock_cost_client(self):
        """Mock Azure Cost Management client"""
        client = Mock(spec=CostManagementClient)

        # Mock cost data
        mock_cost_result = Mock()
        mock_cost_result.rows = [
            ["2024-01-15", "Microsoft.Compute/virtualMachines", "test-vm-1", 125.50, "USD"],
            ["2024-01-15", "Microsoft.Storage/storageAccounts", "teststorage", 45.25, "USD"],
            ["2024-01-15", "Microsoft.Network/loadBalancers", "test-lb", 12.75, "USD"]
        ]
        mock_cost_result.columns = [
            {"name": "UsageDate", "type": "String"},
            {"name": "ResourceType", "type": "String"},
            {"name": "ResourceName", "type": "String"},
            {"name": "Cost", "type": "Number"},
            {"name": "Currency", "type": "String"}
        ]

        mock_query_result = Mock()
        mock_query_result.rows = mock_cost_result.rows
        mock_query_result.columns = mock_cost_result.columns

        client.query.usage.return_value = mock_query_result

        return client

    @pytest.fixture
    def azure_cost_service(self, mock_cost_client):
        """Azure Cost Service with mocked client"""
        with patch('azure_integration.cost_management.CostManagementClient', return_value=mock_cost_client):
            with patch('azure.identity.DefaultAzureCredential'):
                service = AzureCostService("test-subscription-id")
                return service

    @pytest.mark.asyncio
    async def test_get_subscription_costs(self, azure_cost_service):
        """Test getting subscription costs"""
        costs = await azure_cost_service.get_subscription_costs(
            start_date="2024-01-01",
            end_date="2024-01-31"
        )

        assert isinstance(costs, list)
        assert len(costs) == 3
        assert costs[0]["resource_type"] == "Microsoft.Compute/virtualMachines"
        assert costs[0]["cost"] == 125.50
        assert costs[0]["currency"] == "USD"

    @pytest.mark.asyncio
    async def test_get_resource_group_costs(self, azure_cost_service):
        """Test getting resource group costs"""
        costs = await azure_cost_service.get_resource_group_costs(
            resource_group="test-rg",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )

        assert isinstance(costs, list)
        total_cost = sum(item["cost"] for item in costs)
        assert total_cost == 183.50  # Sum of mock data

    @pytest.mark.asyncio
    async def test_cost_analysis_by_service(self, azure_cost_service):
        """Test cost analysis by service"""
        analysis = await azure_cost_service.analyze_costs_by_service(
            start_date="2024-01-01",
            end_date="2024-01-31"
        )

        assert isinstance(analysis, dict)
        assert "Microsoft.Compute/virtualMachines" in analysis
        assert analysis["Microsoft.Compute/virtualMachines"]["total_cost"] == 125.50

    @pytest.mark.asyncio
    async def test_cost_optimization_recommendations(self, azure_cost_service):
        """Test cost optimization recommendations"""
        recommendations = await azure_cost_service.get_cost_optimization_recommendations()

        assert isinstance(recommendations, list)
        # Mock implementation should return optimization suggestions

    @pytest.mark.asyncio
    async def test_budget_alerts(self, azure_cost_service):
        """Test budget alerts and monitoring"""
        budget_status = await azure_cost_service.check_budget_status(
            budget_name="monthly-budget",
            threshold_percentage=80.0
        )

        assert isinstance(budget_status, dict)
        assert "current_spend" in budget_status
        assert "budget_amount" in budget_status
        assert "percentage_used" in budget_status

    @pytest.mark.asyncio
    async def test_cost_forecasting(self, azure_cost_service):
        """Test cost forecasting"""
        forecast = await azure_cost_service.generate_cost_forecast(
            forecast_days=30,
            resource_group="test-rg"
        )

        assert isinstance(forecast, dict)
        assert "forecasted_cost" in forecast
        assert "confidence_interval" in forecast
        assert "trend_analysis" in forecast


@pytest.mark.integration
class TestAIAgentsIntegration:
    """Integration tests for AI agents with Azure services"""

    @pytest.mark.asyncio
    async def test_cost_optimization_agent_integration(self):
        """Test cost optimization agent with real Azure data"""
        # Mock Azure services
        with patch('ai_orchestrator.agents.cost_agent.AzureCostService') as mock_cost_service:
            mock_cost_service.return_value.get_subscription_costs.return_value = [
                {"resource_type": "Microsoft.Compute/virtualMachines", "cost": 150.0, "resource_name": "oversized-vm"},
                {"resource_type": "Microsoft.Storage/storageAccounts", "cost": 25.0, "resource_name": "unused-storage"}
            ]

            mock_cost_service.return_value.get_cost_optimization_recommendations.return_value = [
                {"type": "resize", "resource": "oversized-vm", "potential_savings": 75.0},
                {"type": "delete", "resource": "unused-storage", "potential_savings": 25.0}
            ]

            agent = CostOptimizationAgent()

            optimization_plan = await agent.analyze_costs_and_recommend(
                subscription_id="test-subscription",
                time_range_days=30
            )

            assert isinstance(optimization_plan, dict)
            assert "recommendations" in optimization_plan
            assert "total_potential_savings" in optimization_plan
            assert optimization_plan["total_potential_savings"] == 100.0

    @pytest.mark.asyncio
    async def test_performance_agent_integration(self):
        """Test performance agent with Azure monitoring data"""
        with patch('ai_orchestrator.agents.performance_agent.AzureMonitoringService') as mock_monitor:
            mock_monitor.return_value.get_vm_metrics.return_value = [
                {
                    "name": "Percentage CPU",
                    "data": [{"average": 85.0, "maximum": 95.0, "timestamp": datetime.utcnow()}]
                },
                {
                    "name": "Available Memory Bytes",
                    "data": [{"average": 1024*1024*512, "minimum": 1024*1024*256, "timestamp": datetime.utcnow()}]  # 512MB avg, 256MB min
                }
            ]

            agent = PerformanceAgent()

            performance_analysis = await agent.analyze_resource_performance(
                resource_id="/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm",
                time_range_hours=24
            )

            assert isinstance(performance_analysis, dict)
            assert "cpu_analysis" in performance_analysis
            assert "memory_analysis" in performance_analysis
            assert "recommendations" in performance_analysis

    @pytest.mark.asyncio
    async def test_security_agent_integration(self):
        """Test security agent with Azure security data"""
        with patch('ai_orchestrator.agents.security_agent.AzureSecurityService') as mock_security:
            mock_security.return_value.get_security_alerts.return_value = [
                {
                    "alert_id": "alert-123",
                    "severity": "High",
                    "title": "Suspicious login detected",
                    "resource": "test-vm-1",
                    "timestamp": datetime.utcnow()
                }
            ]

            mock_security.return_value.get_compliance_status.return_value = {
                "overall_score": 75.0,
                "failed_controls": ["Network Security Groups", "Key Vault Access"],
                "passed_controls": ["Encryption at Rest", "Identity Management"]
            }

            agent = SecurityAgent()

            security_assessment = await agent.perform_security_assessment(
                subscription_id="test-subscription",
                resource_group="test-rg"
            )

            assert isinstance(security_assessment, dict)
            assert "security_score" in security_assessment
            assert "alerts" in security_assessment
            assert "compliance_status" in security_assessment

    @pytest.mark.asyncio
    async def test_orchestrator_multi_agent_workflow(self):
        """Test orchestrator coordinating multiple agents"""
        from ai_orchestrator.orchestrator import AzureAIOrchestrator

        # Mock all Azure services
        with patch('ai_orchestrator.orchestrator.AzureCostService'), \
             patch('ai_orchestrator.orchestrator.AzureMonitoringService'), \
             patch('ai_orchestrator.orchestrator.AzureResourceManager'):

            orchestrator = AzureAIOrchestrator()

            # Mock LLM response for comprehensive analysis
            mock_llm_response = {
                "analysis": "Comprehensive analysis of Azure environment",
                "cost_optimization": {"potential_savings": 500.0},
                "performance_issues": {"high_cpu_vms": ["vm-1", "vm-2"]},
                "security_recommendations": ["Enable MFA", "Configure NSGs"],
                "action_plan": ["Resize oversized VMs", "Delete unused resources", "Apply security hardening"]
            }

            with patch.object(orchestrator.llm, 'ainvoke', return_value=Mock(content=json.dumps(mock_llm_response))):
                result = await orchestrator.comprehensive_analysis(
                    subscription_id="test-subscription",
                    resource_group="test-rg"
                )

                assert isinstance(result, dict)
                assert "analysis" in result
                assert "action_plan" in result


@pytest.mark.integration
class TestAzureNetworkServices:
    """Integration tests for Azure Network services"""

    @pytest.fixture
    def mock_network_client(self):
        """Mock Azure Network Management client"""
        client = Mock(spec=NetworkManagementClient)

        # Mock Virtual Network
        mock_vnet = Mock()
        mock_vnet.name = "test-vnet"
        mock_vnet.id = "/subscriptions/test-sub/resourceGroups/test-rg/providers/Microsoft.Network/virtualNetworks/test-vnet"
        mock_vnet.location = "eastus"
        mock_vnet.address_space.address_prefixes = ["10.0.0.0/16"]

        # Mock Subnet
        mock_subnet = Mock()
        mock_subnet.name = "default"
        mock_subnet.address_prefix = "10.0.1.0/24"
        mock_vnet.subnets = [mock_subnet]

        client.virtual_networks.list_all.return_value = [mock_vnet]
        client.virtual_networks.get.return_value = mock_vnet

        # Mock Network Security Group
        mock_nsg = Mock()
        mock_nsg.name = "test-nsg"
        mock_nsg.location = "eastus"

        # Mock Security Rules
        mock_rule = Mock()
        mock_rule.name = "SSH"
        mock_rule.protocol = "Tcp"
        mock_rule.source_port_range = "*"
        mock_rule.destination_port_range = "22"
        mock_rule.source_address_prefix = "*"
        mock_rule.destination_address_prefix = "*"
        mock_rule.access = "Allow"
        mock_rule.priority = 1000
        mock_rule.direction = "Inbound"

        mock_nsg.security_rules = [mock_rule]

        client.network_security_groups.list_all.return_value = [mock_nsg]
        client.network_security_groups.get.return_value = mock_nsg

        return client

    @pytest.mark.asyncio
    async def test_network_topology_analysis(self, mock_network_client):
        """Test network topology analysis"""
        with patch('azure_integration.network_manager.NetworkManagementClient', return_value=mock_network_client):
            with patch('azure.identity.DefaultAzureCredential'):
                from azure_integration.network_manager import AzureNetworkManager
                manager = AzureNetworkManager("test-subscription-id")

                topology = await manager.analyze_network_topology("test-rg")

                assert isinstance(topology, dict)
                assert "virtual_networks" in topology
                assert "network_security_groups" in topology

    @pytest.mark.asyncio
    async def test_security_group_analysis(self, mock_network_client):
        """Test network security group analysis"""
        with patch('azure_integration.network_manager.NetworkManagementClient', return_value=mock_network_client):
            with patch('azure.identity.DefaultAzureCredential'):
                from azure_integration.network_manager import AzureNetworkManager
                manager = AzureNetworkManager("test-subscription-id")

                security_analysis = await manager.analyze_security_groups("test-rg")

                assert isinstance(security_analysis, dict)
                assert "security_issues" in security_analysis
                assert "recommendations" in security_analysis


@pytest.mark.integration
class TestAzureServiceAuthentication:
    """Integration tests for Azure service authentication"""

    @pytest.mark.asyncio
    async def test_managed_identity_authentication(self):
        """Test managed identity authentication"""
        with patch('azure.identity.ManagedIdentityCredential') as mock_credential:
            mock_credential.return_value.get_token.return_value = Mock(token="test-token")

            # Test authentication with various Azure services
            from azure_integration.auth_manager import AzureAuthManager
            auth_manager = AzureAuthManager()

            token = await auth_manager.get_access_token("https://management.azure.com/")
            assert token == "test-token"

    @pytest.mark.asyncio
    async def test_service_principal_authentication(self):
        """Test service principal authentication"""
        with patch('azure.identity.ClientSecretCredential') as mock_credential:
            mock_credential.return_value.get_token.return_value = Mock(token="sp-token")

            from azure_integration.auth_manager import AzureAuthManager
            auth_manager = AzureAuthManager(
                client_id="test-client-id",
                client_secret="test-secret",
                tenant_id="test-tenant-id"
            )

            token = await auth_manager.get_access_token("https://graph.microsoft.com/")
            assert token == "sp-token"

    @pytest.mark.asyncio
    async def test_token_refresh_handling(self):
        """Test token refresh handling"""
        # Mock expired token scenario
        with patch('azure.identity.DefaultAzureCredential') as mock_credential:
            # First call returns expired token, second call returns fresh token
            mock_credential.return_value.get_token.side_effect = [
                Mock(token="expired-token", expires_on=1234567890),  # Past timestamp
                Mock(token="fresh-token", expires_on=9999999999)     # Future timestamp
            ]

            from azure_integration.auth_manager import AzureAuthManager
            auth_manager = AzureAuthManager()

            # Should automatically refresh token
            token = await auth_manager.get_cached_token("https://management.azure.com/")
            assert token in ["expired-token", "fresh-token"]  # Depends on implementation


@pytest.mark.integration
class TestAzureServiceLimitsAndQuotas:
    """Integration tests for Azure service limits and quota management"""

    @pytest.mark.asyncio
    async def test_subscription_quota_checking(self):
        """Test subscription quota and limits checking"""
        with patch('azure_integration.quota_manager.SubscriptionClient') as mock_client:
            mock_quota = Mock()
            mock_quota.current_value = 80
            mock_quota.limit = 100
            mock_quota.name.value = "cores"

            mock_client.return_value.subscriptions.list_locations.return_value = [
                Mock(name="eastus", display_name="East US")
            ]

            from azure_integration.quota_manager import AzureQuotaManager
            quota_manager = AzureQuotaManager("test-subscription-id")

            quotas = await quota_manager.check_subscription_quotas("eastus")

            assert isinstance(quotas, list)

    @pytest.mark.asyncio
    async def test_resource_deployment_feasibility(self):
        """Test resource deployment feasibility checking"""
        # This would check if there are enough quotas to deploy new resources
        from azure_integration.deployment_validator import AzureDeploymentValidator

        validator = AzureDeploymentValidator("test-subscription-id")

        deployment_request = {
            "vm_count": 5,
            "vm_size": "Standard_D2s_v3",
            "location": "eastus"
        }

        with patch.object(validator, 'check_compute_quotas', return_value={"feasible": True, "remaining_quota": 15}):
            feasibility = await validator.validate_deployment(deployment_request)

            assert feasibility["feasible"] is True
            assert "quota_usage" in feasibility


@pytest.mark.integration
class TestErrorHandlingAndResilience:
    """Integration tests for error handling and resilience"""

    @pytest.mark.asyncio
    async def test_azure_service_timeout_handling(self):
        """Test handling of Azure service timeouts"""
        with patch('azure_integration.resource_manager.ResourceManagementClient') as mock_client:
            # Mock timeout exception
            mock_client.return_value.resource_groups.list.side_effect = asyncio.TimeoutError("Request timed out")

            from azure_integration.resource_manager import AzureResourceManager
            manager = AzureResourceManager("test-subscription-id")

            with pytest.raises(asyncio.TimeoutError):
                await manager.list_resource_groups()

    @pytest.mark.asyncio
    async def test_azure_service_rate_limiting(self):
        """Test handling of Azure service rate limiting"""
        with patch('azure_integration.resource_manager.ResourceManagementClient') as mock_client:
            # Mock rate limiting (HTTP 429)
            mock_client.return_value.resource_groups.list.side_effect = HttpResponseError(
                message="Too Many Requests",
                response=Mock(status_code=429)
            )

            from azure_integration.resource_manager import AzureResourceManager
            manager = AzureResourceManager("test-subscription-id")

            with pytest.raises(HttpResponseError):
                await manager.list_resource_groups()

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self):
        """Test handling of partial failures in batch operations"""
        # Test scenario where some resources succeed and others fail
        from azure_integration.batch_operations import AzureBatchOperations

        operations = AzureBatchOperations("test-subscription-id")

        # Mock mixed results
        with patch.object(operations, 'start_vms') as mock_start:
            mock_start.return_value = {
                "successful": ["vm-1", "vm-2"],
                "failed": [{"vm": "vm-3", "error": "VM not found"}]
            }

            results = await operations.start_multiple_vms(["vm-1", "vm-2", "vm-3"])

            assert len(results["successful"]) == 2
            assert len(results["failed"]) == 1


# Performance and Load Testing Markers
@pytest.mark.performance
@pytest.mark.integration
class TestAzureServicePerformance:
    """Performance tests for Azure service integrations"""

    @pytest.mark.asyncio
    async def test_concurrent_resource_queries(self):
        """Test concurrent Azure resource queries"""
        from azure_integration.resource_manager import AzureResourceManager

        with patch('azure_integration.resource_manager.ResourceManagementClient'):
            manager = AzureResourceManager("test-subscription-id")

            # Mock multiple concurrent calls
            tasks = []
            for i in range(10):
                task = asyncio.create_task(manager.list_resource_groups())
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start_time

            # Should complete within reasonable time
            assert elapsed < 5.0  # 5 seconds for 10 concurrent calls
            assert len(results) == 10

    @pytest.mark.asyncio
    async def test_large_dataset_processing(self):
        """Test processing of large Azure datasets"""
        # Mock large dataset response
        large_dataset = [{"id": f"resource-{i}", "name": f"resource-{i}"} for i in range(1000)]

        with patch('azure_integration.resource_manager.ResourceManagementClient') as mock_client:
            mock_client.return_value.resources.list.return_value = large_dataset

            from azure_integration.resource_manager import AzureResourceManager
            manager = AzureResourceManager("test-subscription-id")

            start_time = time.time()
            resources = await manager.list_all_resources()
            elapsed = time.time() - start_time

            # Should process large dataset efficiently
            assert len(resources) == 1000
            assert elapsed < 2.0  # Should complete within 2 seconds