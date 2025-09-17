"""
Resource Management Agent
Handles Azure resource creation, modification, and monitoring operations
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.core.exceptions import AzureError

from ..orchestrator import BaseAgent

logger = logging.getLogger(__name__)


class ResourceAgent(BaseAgent):
    """Agent for managing Azure resources"""

    def __init__(self, orchestrator):
        super().__init__(orchestrator)
        self.resource_client = orchestrator.resource_client
        self.compute_client = orchestrator.compute_client
        self.network_client = orchestrator.network_client
        self.storage_client = orchestrator.storage_client

    async def create_plan(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create execution plan for resource operations"""
        context = context or {}

        # Parse command to determine resource type and operation
        command_lower = command.lower()

        if "create" in command_lower:
            if "vm" in command_lower or "virtual machine" in command_lower:
                return await self._plan_vm_creation(command, context)
            elif "storage" in command_lower:
                return await self._plan_storage_creation(command, context)
            elif "network" in command_lower or "vnet" in command_lower:
                return await self._plan_network_creation(command, context)

        elif "list" in command_lower or "show" in command_lower:
            return await self._plan_resource_query(command, context)

        elif "delete" in command_lower:
            return await self._plan_resource_deletion(command, context)

        elif "scale" in command_lower or "resize" in command_lower:
            return await self._plan_resource_scaling(command, context)

        return {
            "operation": "resource_management",
            "command": command,
            "context": context,
            "steps": [{"action": "analyze_command", "description": "Analyze resource operation"}],
            "requires_approval": True,
            "estimated_time": "2-5 minutes",
            "agent_type": "ResourceAgent"
        }

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the resource management plan"""
        start_time = datetime.now()

        try:
            operation = plan.get("operation", "unknown")

            if operation == "vm_creation":
                result = await self._execute_vm_creation(plan)
            elif operation == "storage_creation":
                result = await self._execute_storage_creation(plan)
            elif operation == "network_creation":
                result = await self._execute_network_creation(plan)
            elif operation == "resource_query":
                result = await self._execute_resource_query(plan)
            elif operation == "resource_deletion":
                result = await self._execute_resource_deletion(plan)
            elif operation == "resource_scaling":
                result = await self._execute_resource_scaling(plan)
            else:
                result = {
                    "status": "error",
                    "message": f"Unknown operation: {operation}"
                }

            return {
                **result,
                "plan": plan,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }

        except Exception as e:
            logger.error(f"Resource operation failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Resource operation failed: {str(e)}",
                "plan": plan,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }

    async def _plan_vm_creation(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan VM creation operation"""
        # Extract VM specifications from command
        vm_specs = self._parse_vm_specs(command)

        steps = [
            {"action": "validate_specs", "description": "Validate VM specifications"},
            {"action": "check_quotas", "description": "Check Azure quotas"},
            {"action": "create_resource_group", "description": "Ensure resource group exists"},
            {"action": "create_network", "description": "Create or verify network"},
            {"action": "create_vm", "description": "Create virtual machine"},
            {"action": "configure_monitoring", "description": "Setup monitoring"}
        ]

        return {
            "operation": "vm_creation",
            "command": command,
            "context": context,
            "vm_specs": vm_specs,
            "steps": steps,
            "requires_approval": True,
            "estimated_time": "5-10 minutes",
            "agent_type": "ResourceAgent"
        }

    async def _plan_storage_creation(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan storage account creation"""
        storage_specs = self._parse_storage_specs(command)

        steps = [
            {"action": "validate_name", "description": "Validate storage account name"},
            {"action": "check_availability", "description": "Check name availability"},
            {"action": "create_storage", "description": "Create storage account"},
            {"action": "configure_access", "description": "Configure access policies"}
        ]

        return {
            "operation": "storage_creation",
            "command": command,
            "context": context,
            "storage_specs": storage_specs,
            "steps": steps,
            "requires_approval": True,
            "estimated_time": "2-3 minutes",
            "agent_type": "ResourceAgent"
        }

    async def _plan_network_creation(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan network creation"""
        network_specs = self._parse_network_specs(command)

        steps = [
            {"action": "validate_cidrs", "description": "Validate CIDR blocks"},
            {"action": "create_vnet", "description": "Create virtual network"},
            {"action": "create_subnets", "description": "Create subnets"},
            {"action": "configure_security", "description": "Setup security groups"}
        ]

        return {
            "operation": "network_creation",
            "command": command,
            "context": context,
            "network_specs": network_specs,
            "steps": steps,
            "requires_approval": True,
            "estimated_time": "3-5 minutes",
            "agent_type": "ResourceAgent"
        }

    async def _plan_resource_query(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan resource query operation"""
        query_specs = self._parse_query_specs(command)

        steps = [
            {"action": "query_resources", "description": "Query Azure resources"},
            {"action": "format_results", "description": "Format results for display"}
        ]

        return {
            "operation": "resource_query",
            "command": command,
            "context": context,
            "query_specs": query_specs,
            "steps": steps,
            "requires_approval": False,
            "estimated_time": "30 seconds",
            "agent_type": "ResourceAgent"
        }

    async def _plan_resource_deletion(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan resource deletion"""
        deletion_specs = self._parse_deletion_specs(command)

        steps = [
            {"action": "validate_resource", "description": "Validate resource exists"},
            {"action": "check_dependencies", "description": "Check for dependencies"},
            {"action": "backup_config", "description": "Backup configuration"},
            {"action": "delete_resource", "description": "Delete the resource"}
        ]

        return {
            "operation": "resource_deletion",
            "command": command,
            "context": context,
            "deletion_specs": deletion_specs,
            "steps": steps,
            "requires_approval": True,
            "estimated_time": "2-5 minutes",
            "agent_type": "ResourceAgent"
        }

    async def _plan_resource_scaling(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan resource scaling operation"""
        scaling_specs = self._parse_scaling_specs(command)

        steps = [
            {"action": "validate_target", "description": "Validate scaling target"},
            {"action": "check_limits", "description": "Check scaling limits"},
            {"action": "scale_resource", "description": "Perform scaling operation"},
            {"action": "verify_scaling", "description": "Verify scaling completed"}
        ]

        return {
            "operation": "resource_scaling",
            "command": command,
            "context": context,
            "scaling_specs": scaling_specs,
            "steps": steps,
            "requires_approval": True,
            "estimated_time": "3-8 minutes",
            "agent_type": "ResourceAgent"
        }

    async def _execute_vm_creation(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute VM creation"""
        try:
            vm_specs = plan.get("vm_specs", {})

            # For now, return a detailed plan rather than actual creation
            # In production, this would make real Azure API calls

            vm_config = {
                "name": vm_specs.get("name", "ai-copilot-vm"),
                "size": vm_specs.get("size", "Standard_D2s_v3"),
                "location": vm_specs.get("location", "East US"),
                "os_type": vm_specs.get("os_type", "Linux"),
                "resource_group": vm_specs.get("resource_group", "rg-ai-copilot")
            }

            # Simulate VM creation process
            logger.info(f"Creating VM with config: {vm_config}")

            return {
                "status": "success",
                "message": "VM creation plan generated successfully",
                "vm_config": vm_config,
                "next_steps": [
                    "Review VM specifications",
                    "Approve creation",
                    "Monitor creation progress"
                ],
                "estimated_cost": "$50-100/month"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"VM creation failed: {str(e)}"
            }

    async def _execute_storage_creation(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute storage account creation"""
        try:
            storage_specs = plan.get("storage_specs", {})

            storage_config = {
                "name": storage_specs.get("name", "aicopilotstorage"),
                "tier": storage_specs.get("tier", "Standard"),
                "replication": storage_specs.get("replication", "LRS"),
                "location": storage_specs.get("location", "East US")
            }

            logger.info(f"Creating storage account with config: {storage_config}")

            return {
                "status": "success",
                "message": "Storage account creation plan generated",
                "storage_config": storage_config,
                "estimated_cost": "$10-50/month"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Storage creation failed: {str(e)}"
            }

    async def _execute_network_creation(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute network creation"""
        try:
            network_specs = plan.get("network_specs", {})

            network_config = {
                "vnet_name": network_specs.get("vnet_name", "ai-copilot-vnet"),
                "address_space": network_specs.get("address_space", "10.0.0.0/16"),
                "subnets": network_specs.get("subnets", [
                    {"name": "default", "address_prefix": "10.0.1.0/24"}
                ]),
                "location": network_specs.get("location", "East US")
            }

            logger.info(f"Creating network with config: {network_config}")

            return {
                "status": "success",
                "message": "Network creation plan generated",
                "network_config": network_config,
                "estimated_cost": "$5-20/month"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Network creation failed: {str(e)}"
            }

    async def _execute_resource_query(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute resource query"""
        try:
            query_specs = plan.get("query_specs", {})
            resource_type = query_specs.get("resource_type", "all")

            # Simulate querying resources
            resources = []

            if resource_type in ["all", "vm", "virtual_machines"]:
                resources.extend([
                    {
                        "type": "Virtual Machine",
                        "name": "web-vm-01",
                        "location": "East US",
                        "status": "Running",
                        "size": "Standard_D2s_v3"
                    }
                ])

            if resource_type in ["all", "storage"]:
                resources.extend([
                    {
                        "type": "Storage Account",
                        "name": "webappstorage001",
                        "location": "East US",
                        "tier": "Standard",
                        "replication": "LRS"
                    }
                ])

            return {
                "status": "success",
                "message": f"Found {len(resources)} resources",
                "resources": resources,
                "query_specs": query_specs
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Resource query failed: {str(e)}"
            }

    async def _execute_resource_deletion(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute resource deletion"""
        try:
            deletion_specs = plan.get("deletion_specs", {})

            return {
                "status": "success",
                "message": "Resource deletion plan generated",
                "deletion_specs": deletion_specs,
                "warning": "This operation cannot be undone"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Resource deletion failed: {str(e)}"
            }

    async def _execute_resource_scaling(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute resource scaling"""
        try:
            scaling_specs = plan.get("scaling_specs", {})

            return {
                "status": "success",
                "message": "Resource scaling plan generated",
                "scaling_specs": scaling_specs,
                "estimated_downtime": "2-5 minutes"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Resource scaling failed: {str(e)}"
            }

    def _parse_vm_specs(self, command: str) -> Dict[str, Any]:
        """Parse VM specifications from command"""
        specs = {}

        # Extract VM size
        if "8gb" in command.lower() or "8 gb" in command.lower():
            specs["size"] = "Standard_D2s_v3"
        elif "16gb" in command.lower() or "16 gb" in command.lower():
            specs["size"] = "Standard_D4s_v3"

        # Extract location
        if "east us" in command.lower():
            specs["location"] = "East US"
        elif "west us" in command.lower():
            specs["location"] = "West US"

        # Extract OS type
        if "linux" in command.lower():
            specs["os_type"] = "Linux"
        elif "windows" in command.lower():
            specs["os_type"] = "Windows"

        return specs

    def _parse_storage_specs(self, command: str) -> Dict[str, Any]:
        """Parse storage specifications from command"""
        specs = {}

        if "premium" in command.lower():
            specs["tier"] = "Premium"
        elif "standard" in command.lower():
            specs["tier"] = "Standard"

        return specs

    def _parse_network_specs(self, command: str) -> Dict[str, Any]:
        """Parse network specifications from command"""
        specs = {}

        # Extract CIDR blocks if mentioned
        # This is a simplified parser - production would be more robust

        return specs

    def _parse_query_specs(self, command: str) -> Dict[str, Any]:
        """Parse query specifications from command"""
        specs = {}

        if "vm" in command.lower() or "virtual machine" in command.lower():
            specs["resource_type"] = "vm"
        elif "storage" in command.lower():
            specs["resource_type"] = "storage"
        elif "network" in command.lower():
            specs["resource_type"] = "network"
        else:
            specs["resource_type"] = "all"

        return specs

    def _parse_deletion_specs(self, command: str) -> Dict[str, Any]:
        """Parse deletion specifications from command"""
        specs = {}

        # Extract resource name/ID from command
        # This would be more sophisticated in production

        return specs

    def _parse_scaling_specs(self, command: str) -> Dict[str, Any]:
        """Parse scaling specifications from command"""
        specs = {}

        # Extract scaling target and parameters
        # This would be more sophisticated in production

        return specs