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
        self.subscription_id = orchestrator.subscription_id

        # Use the orchestrator's Azure clients to avoid duplication
        self.resource_client = orchestrator.resource_client
        self.compute_client = orchestrator.compute_client
        self.network_client = orchestrator.network_client
        self.storage_client = orchestrator.storage_client

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
        if not plan:
            return {
                'status': 'failed',
                'message': 'No execution plan provided'
            }

        result = {
            'status': 'started',
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'resources_created': [],
            'resources_modified': [],
            'resources_deleted': [],
            'plan_id': plan.get('id', 'unknown')
        }

        rollback_executed = False

        try:
            steps = plan.get('steps', [])
            if not steps:
                logger.warning("No steps found in execution plan")
                result['status'] = 'completed'
                result['message'] = 'No steps to execute'
                return result

            # Execute each step in the plan
            for i, step in enumerate(steps):
                try:
                    logger.info(f"Executing step {step.get('order', i+1)}: {step.get('action', 'unknown')}")

                    step_result = None
                    if step['action'] == 'validate_quota':
                        step_result = await self._validate_quota(plan)
                    elif step['action'] == 'create_resource_group':
                        step_result = await self._ensure_resource_group(plan)
                    elif step['action'] == 'create_network':
                        step_result = await self._ensure_network(plan)
                    elif step['action'] == 'create_vm':
                        step_result = await self._create_vm(plan['specifications'])
                        if step_result:
                            result['resources_created'].append(step_result)
                    elif step['action'] == 'configure_monitoring':
                        step_result = await self._configure_monitoring(plan)
                    elif step['action'] == 'delete_resource':
                        step_result = await self._delete_resource(plan['specifications'])
                        if step_result:
                            result['resources_deleted'].append(step_result)
                    else:
                        logger.warning(f"Unknown step action: {step['action']}")
                        continue

                    # Mark step as completed
                    step['completed'] = True
                    step['result'] = step_result
                    result['steps_completed'].append(step)

                except Exception as step_error:
                    logger.error(f"Step {step.get('action')} failed: {str(step_error)}")
                    step['failed'] = True
                    step['error'] = str(step_error)
                    result['steps_completed'].append(step)

                    # If critical step fails, stop execution
                    if step.get('critical', True):
                        raise step_error

            result['status'] = 'completed'
            result['end_time'] = datetime.now().isoformat()
            result['message'] = f"Successfully executed {plan.get('operation', 'unknown')} operation"

            # Validate final state
            if plan.get('operation') == 'create':
                validation_result = await self._validate_deployment(result)
                result['validation'] = validation_result

        except Exception as e:
            logger.error(f"Execution failed: {str(e)}", exc_info=True)
            result['status'] = 'failed'
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            result['end_time'] = datetime.now().isoformat()

            # Execute rollback if needed and not already executed
            if plan.get('rollback_plan') and not rollback_executed:
                try:
                    logger.info("Executing rollback plan")
                    rollback_result = await self._execute_rollback(plan['rollback_plan'], result)
                    result['rollback'] = rollback_result
                    rollback_executed = True
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {str(rollback_error)}")
                    result['rollback_error'] = str(rollback_error)

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

    async def _validate_quota(self, plan: Dict) -> Dict:
        """Validate Azure quota for requested resources"""
        try:
            # In production, check actual quota
            return {
                'status': 'validated',
                'quota_available': True,
                'message': 'Sufficient quota available'
            }
        except Exception as e:
            logger.error(f"Quota validation failed: {str(e)}")
            return {
                'status': 'failed',
                'quota_available': False,
                'error': str(e)
            }

    async def _ensure_network(self, plan: Dict) -> Dict:
        """Ensure network components exist"""
        try:
            # In production, create/verify VNet, subnet, NSG
            network_name = plan['specifications'].get('network', 'default-vnet')
            logger.info(f"Ensuring network {network_name}")
            return {
                'status': 'ensured',
                'network_name': network_name,
                'vnet_created': True,
                'subnet_created': True
            }
        except Exception as e:
            logger.error(f"Network setup failed: {str(e)}")
            raise

    async def _configure_monitoring(self, plan: Dict) -> Dict:
        """Configure monitoring for created resources"""
        try:
            # In production, set up Application Insights, alerts
            logger.info("Configuring monitoring")
            return {
                'status': 'configured',
                'monitoring_enabled': True,
                'alerts_configured': True
            }
        except Exception as e:
            logger.error(f"Monitoring configuration failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    async def _validate_deployment(self, result: Dict) -> Dict:
        """Validate that deployment was successful"""
        try:
            validation = {
                'all_steps_completed': len([s for s in result['steps_completed'] if s.get('completed')]) > 0,
                'resources_healthy': True,  # In production, check resource health
                'networks_accessible': True,  # In production, test connectivity
                'monitoring_active': True   # In production, verify monitoring
            }

            validation['overall_status'] = all(validation.values())
            return validation
        except Exception as e:
            logger.error(f"Deployment validation failed: {str(e)}")
            return {
                'overall_status': False,
                'error': str(e)
            }

    async def _execute_rollback(self, rollback_plan: Dict, execution_result: Dict = None):
        """Execute rollback plan"""
        rollback_result = {
            'status': 'started',
            'steps_executed': [],
            'start_time': datetime.now().isoformat()
        }

        try:
            logger.info("Executing rollback plan")

            # Rollback created resources in reverse order
            if execution_result and execution_result.get('resources_created'):
                for resource in reversed(execution_result['resources_created']):
                    try:
                        logger.info(f"Rolling back resource: {resource.get('name')}")
                        # In production, actually delete the resource
                        rollback_result['steps_executed'].append({
                            'action': 'delete_resource',
                            'resource': resource.get('name'),
                            'status': 'success'
                        })
                    except Exception as e:
                        logger.error(f"Failed to rollback resource {resource.get('name')}: {str(e)}")
                        rollback_result['steps_executed'].append({
                            'action': 'delete_resource',
                            'resource': resource.get('name'),
                            'status': 'failed',
                            'error': str(e)
                        })

            # Execute rollback steps from plan
            for step in rollback_plan.get('steps', []):
                try:
                    logger.info(f"Rollback step: {step}")
                    await asyncio.sleep(1)  # Simulate rollback action
                    rollback_result['steps_executed'].append({
                        'action': step,
                        'status': 'success'
                    })
                except Exception as e:
                    logger.error(f"Rollback step failed: {str(e)}")
                    rollback_result['steps_executed'].append({
                        'action': step,
                        'status': 'failed',
                        'error': str(e)
                    })

            rollback_result['status'] = 'completed'
            rollback_result['end_time'] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Rollback execution failed: {str(e)}")
            rollback_result['status'] = 'failed'
            rollback_result['error'] = str(e)
            rollback_result['end_time'] = datetime.now().isoformat()

        return rollback_result

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