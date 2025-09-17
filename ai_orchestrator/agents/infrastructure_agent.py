"""
Infrastructure Agent - Manages Azure resources via natural language
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential

from langchain.tools import Tool
from langchain.schema import HumanMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)


class InfrastructureAgent:
    """Agent for managing Azure infrastructure resources"""

    def __init__(self, orchestrator):
        """Initialize the infrastructure agent"""
        self.orchestrator = orchestrator
        self.llm = orchestrator.llm
        self.credential = DefaultAzureCredential()
        self.subscription_id = orchestrator.subscription_id

        # Initialize Azure clients
        self.resource_client = ResourceManagementClient(
            self.credential,
            self.subscription_id
        )
        self.compute_client = ComputeManagementClient(
            self.credential,
            self.subscription_id
        )
        self.network_client = NetworkManagementClient(
            self.credential,
            self.subscription_id
        )
        self.storage_client = StorageManagementClient(
            self.credential,
            self.subscription_id
        )

        self.tools = self._create_tools()

    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for infrastructure operations"""
        return [
            Tool(
                name="list_resources",
                func=self.list_resources,
                description="List Azure resources with optional filters"
            ),
            Tool(
                name="create_vm",
                func=self.create_virtual_machine,
                description="Create a new virtual machine"
            ),
            Tool(
                name="delete_resource",
                func=self.delete_resource,
                description="Delete an Azure resource"
            ),
            Tool(
                name="get_resource_details",
                func=self.get_resource_details,
                description="Get detailed information about a resource"
            ),
            Tool(
                name="modify_resource",
                func=self.modify_resource,
                description="Modify an existing resource"
            )
        ]

    async def create_plan(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an execution plan from natural language command

        Args:
            command: Natural language command
            context: Optional context

        Returns:
            Execution plan dictionary
        """
        # Parse the command to understand the intent
        prompt = f"""
        Analyze this infrastructure command and create an execution plan:

        Command: {command}

        Extract the following:
        1. Operation type (create, delete, modify, query)
        2. Resource type (vm, storage, network, etc.)
        3. Resource specifications (size, location, name, etc.)
        4. Any special requirements

        Respond in JSON format with these fields:
        - operation: the operation to perform
        - resource_type: type of Azure resource
        - specifications: dict of specifications
        - validation_required: boolean
        - estimated_cost: estimated monthly cost in USD
        - requires_approval: boolean
        """

        response = await self.llm.ainvoke(prompt)
        plan = json.loads(response.content)

        # Add execution steps
        plan['steps'] = self._generate_execution_steps(plan)
        plan['rollback_plan'] = self._generate_rollback_plan(plan)

        logger.info(f"Created execution plan: {plan}")
        return plan

    def _generate_execution_steps(self, plan: Dict[str, Any]) -> List[Dict]:
        """Generate detailed execution steps"""
        steps = []

        if plan['operation'] == 'create':
            if plan['resource_type'] == 'vm':
                steps = [
                    {
                        'order': 1,
                        'action': 'validate_quota',
                        'description': 'Check if quota allows new VM'
                    },
                    {
                        'order': 2,
                        'action': 'create_resource_group',
                        'description': 'Ensure resource group exists'
                    },
                    {
                        'order': 3,
                        'action': 'create_network',
                        'description': 'Create or verify network components'
                    },
                    {
                        'order': 4,
                        'action': 'create_vm',
                        'description': 'Create the virtual machine'
                    },
                    {
                        'order': 5,
                        'action': 'configure_monitoring',
                        'description': 'Set up monitoring and alerts'
                    }
                ]
        elif plan['operation'] == 'delete':
            steps = [
                {
                    'order': 1,
                    'action': 'backup_data',
                    'description': 'Backup any important data'
                },
                {
                    'order': 2,
                    'action': 'remove_dependencies',
                    'description': 'Remove dependent resources'
                },
                {
                    'order': 3,
                    'action': 'delete_resource',
                    'description': f"Delete the {plan['resource_type']}"
                }
            ]

        return steps

    def _generate_rollback_plan(self, plan: Dict[str, Any]) -> Dict:
        """Generate rollback plan in case of failure"""
        rollback = {
            'strategy': 'automatic',
            'steps': []
        }

        if plan['operation'] == 'create':
            rollback['steps'] = [
                'Delete created resources in reverse order',
                'Restore original configuration',
                'Clean up any partial deployments'
            ]
        elif plan['operation'] == 'modify':
            rollback['steps'] = [
                'Restore from backup configuration',
                'Revert changes',
                'Verify original state'
            ]

        return rollback

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the infrastructure plan

        Args:
            plan: Execution plan

        Returns:
            Execution result
        """
        result = {
            'status': 'started',
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'resources_created': [],
            'resources_modified': [],
            'resources_deleted': []
        }

        try:
            # Execute each step in the plan
            for step in plan.get('steps', []):
                logger.info(f"Executing step {step['order']}: {step['action']}")

                if step['action'] == 'create_resource_group':
                    rg_result = await self._ensure_resource_group(plan)
                    result['steps_completed'].append(step)

                elif step['action'] == 'create_vm':
                    vm_result = await self._create_vm(plan['specifications'])
                    result['resources_created'].append(vm_result)
                    result['steps_completed'].append(step)

                elif step['action'] == 'delete_resource':
                    delete_result = await self._delete_resource(plan['specifications'])
                    result['resources_deleted'].append(delete_result)
                    result['steps_completed'].append(step)

            result['status'] = 'completed'
            result['end_time'] = datetime.now().isoformat()
            result['message'] = f"Successfully executed {plan['operation']} operation"

        except Exception as e:
            logger.error(f"Execution failed: {str(e)}")
            result['status'] = 'failed'
            result['error'] = str(e)

            # Execute rollback if needed
            if plan.get('rollback_plan'):
                logger.info("Executing rollback plan")
                await self._execute_rollback(plan['rollback_plan'])

        return result

    async def _ensure_resource_group(self, plan: Dict) -> Dict:
        """Ensure resource group exists"""
        rg_name = plan['specifications'].get('resource_group', 'rg-ai-copilot-default')
        location = plan['specifications'].get('location', 'eastus')

        try:
            rg = self.resource_client.resource_groups.get(rg_name)
            logger.info(f"Resource group {rg_name} already exists")
        except:
            # Create resource group
            rg_params = {
                'location': location,
                'tags': {
                    'created_by': 'ai-copilot',
                    'created_at': datetime.now().isoformat()
                }
            }
            rg = self.resource_client.resource_groups.create_or_update(
                rg_name,
                rg_params
            )
            logger.info(f"Created resource group {rg_name}")

        return {'name': rg_name, 'location': location}

    async def _create_vm(self, specs: Dict) -> Dict:
        """Create a virtual machine"""
        vm_name = specs.get('name', f"vm-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        location = specs.get('location', 'eastus')
        size = self._map_size_description_to_sku(specs.get('size', '8GB RAM'))
        os_type = specs.get('os', 'linux').lower()

        # VM parameters
        vm_params = {
            'location': location,
            'hardware_profile': {
                'vm_size': size
            },
            'storage_profile': {
                'image_reference': self._get_image_reference(os_type),
                'os_disk': {
                    'name': f"{vm_name}-osdisk",
                    'caching': 'ReadWrite',
                    'create_option': 'FromImage',
                    'managed_disk': {
                        'storage_account_type': 'Premium_LRS'
                    }
                }
            },
            'os_profile': {
                'computer_name': vm_name,
                'admin_username': 'azureuser',
                'admin_password': 'P@ssw0rd123!'  # In production, use Key Vault
            },
            'network_profile': {
                'network_interfaces': [{
                    'id': f"/subscriptions/{self.subscription_id}/resourceGroups/{specs.get('resource_group')}/providers/Microsoft.Network/networkInterfaces/{vm_name}-nic",
                    'primary': True
                }]
            },
            'tags': {
                'created_by': 'ai-copilot',
                'purpose': specs.get('purpose', 'general'),
                'environment': specs.get('environment', 'development')
            }
        }

        # Create VM (simplified for demo)
        logger.info(f"Creating VM {vm_name} with size {size}")

        # In production, this would make actual Azure API calls
        return {
            'name': vm_name,
            'type': 'Microsoft.Compute/virtualMachines',
            'location': location,
            'size': size,
            'status': 'created'
        }

    def _map_size_description_to_sku(self, description: str) -> str:
        """Map natural language size description to Azure VM SKU"""
        # Extract memory from description
        memory_match = re.search(r'(\d+)\s*GB', description, re.IGNORECASE)

        if memory_match:
            memory_gb = int(memory_match.group(1))

            # Simple mapping (in production, use more sophisticated logic)
            if memory_gb <= 4:
                return "Standard_B2s"
            elif memory_gb <= 8:
                return "Standard_D2s_v3"
            elif memory_gb <= 16:
                return "Standard_D4s_v3"
            elif memory_gb <= 32:
                return "Standard_D8s_v3"
            else:
                return "Standard_D16s_v3"

        return "Standard_B2s"  # Default

    def _get_image_reference(self, os_type: str) -> Dict:
        """Get image reference for OS type"""
        if 'windows' in os_type:
            return {
                'publisher': 'MicrosoftWindowsServer',
                'offer': 'WindowsServer',
                'sku': '2022-Datacenter',
                'version': 'latest'
            }
        else:
            return {
                'publisher': 'Canonical',
                'offer': 'UbuntuServer',
                'sku': '20.04-LTS',
                'version': 'latest'
            }

    async def _delete_resource(self, specs: Dict) -> Dict:
        """Delete a resource"""
        resource_id = specs.get('resource_id')
        resource_name = specs.get('name')

        logger.info(f"Deleting resource: {resource_name or resource_id}")

        # In production, make actual deletion call
        return {
            'resource': resource_name or resource_id,
            'status': 'deleted',
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_rollback(self, rollback_plan: Dict):
        """Execute rollback plan"""
        logger.info("Executing rollback plan")
        for step in rollback_plan.get('steps', []):
            logger.info(f"Rollback: {step}")
            # Execute rollback step
            await asyncio.sleep(1)  # Simulate rollback action

    async def list_resources(
        self,
        resource_type: Optional[str] = None,
        resource_group: Optional[str] = None
    ) -> List[Dict]:
        """
        List Azure resources

        Args:
            resource_type: Optional filter by resource type
            resource_group: Optional filter by resource group

        Returns:
            List of resources
        """
        resources = []

        try:
            # List all resources or filter by resource group
            if resource_group:
                resource_list = self.resource_client.resources.list_by_resource_group(
                    resource_group
                )
            else:
                resource_list = self.resource_client.resources.list()

            for resource in resource_list:
                if not resource_type or resource_type in resource.type:
                    resources.append({
                        'name': resource.name,
                        'type': resource.type,
                        'location': resource.location,
                        'resource_group': resource.id.split('/')[4],
                        'tags': resource.tags
                    })

        except Exception as e:
            logger.error(f"Error listing resources: {str(e)}")

        return resources

    async def create_virtual_machine(
        self,
        name: str,
        location: str,
        size: str,
        resource_group: str,
        **kwargs
    ) -> Dict:
        """Create a new virtual machine"""
        specs = {
            'name': name,
            'location': location,
            'size': size,
            'resource_group': resource_group,
            **kwargs
        }

        plan = {
            'operation': 'create',
            'resource_type': 'vm',
            'specifications': specs,
            'steps': self._generate_execution_steps({
                'operation': 'create',
                'resource_type': 'vm'
            })
        }

        return await self.execute(plan)

    async def delete_resource(
        self,
        resource_id: str = None,
        resource_name: str = None
    ) -> Dict:
        """Delete an Azure resource"""
        specs = {
            'resource_id': resource_id,
            'name': resource_name
        }

        plan = {
            'operation': 'delete',
            'specifications': specs,
            'steps': self._generate_execution_steps({
                'operation': 'delete',
                'resource_type': 'generic'
            })
        }

        return await self.execute(plan)

    async def get_resource_details(self, resource_id: str) -> Dict:
        """Get detailed information about a resource"""
        try:
            resource = self.resource_client.resources.get_by_id(
                resource_id,
                api_version='2021-04-01'
            )

            return {
                'name': resource.name,
                'type': resource.type,
                'location': resource.location,
                'properties': resource.properties,
                'tags': resource.tags,
                'sku': resource.sku
            }
        except Exception as e:
            logger.error(f"Error getting resource details: {str(e)}")
            return {'error': str(e)}

    async def modify_resource(
        self,
        resource_id: str,
        modifications: Dict
    ) -> Dict:
        """Modify an existing resource"""
        # This would implement resource modification logic
        return {
            'resource_id': resource_id,
            'modifications': modifications,
            'status': 'modified'
        }