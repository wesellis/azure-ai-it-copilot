"""
Azure Resource Management Client
Handles Azure resource operations with proper error handling and async support
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

from config.settings import get_settings

logger = logging.getLogger(__name__)


class AzureResourceClient:
    """Azure Resource Management Client with async support"""

    def __init__(self):
        """Initialize Azure clients"""
        self.settings = get_settings()
        self.credential = None
        self.resource_client = None
        self.compute_client = None
        self.network_client = None
        self.storage_client = None
        self._initialized = False

    async def initialize(self):
        """Initialize Azure clients asynchronously"""
        if self._initialized:
            return

        try:
            self.credential = DefaultAzureCredential()

            # Initialize clients
            self.resource_client = ResourceManagementClient(
                self.credential,
                self.settings.azure_subscription_id
            )

            self.compute_client = ComputeManagementClient(
                self.credential,
                self.settings.azure_subscription_id
            )

            self.network_client = NetworkManagementClient(
                self.credential,
                self.settings.azure_subscription_id
            )

            self.storage_client = StorageManagementClient(
                self.credential,
                self.settings.azure_subscription_id
            )

            self._initialized = True
            logger.info("Azure clients initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure clients: {str(e)}")
            raise

    def close(self):
        """Close Azure clients"""
        # For sync clients, just mark as uninitialized
        self._initialized = False
        logger.info("Azure clients closed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.close()

    async def list_resource_groups(self) -> List[Dict[str, Any]]:
        """List all resource groups"""
        if not self._initialized:
            await self.initialize()

        try:
            resource_groups = []
            async for rg in self.resource_client.resource_groups.list():
                resource_groups.append({
                    "name": rg.name,
                    "location": rg.location,
                    "id": rg.id,
                    "tags": rg.tags or {},
                    "provisioning_state": rg.provisioning_state
                })

            logger.info(f"Retrieved {len(resource_groups)} resource groups")
            return resource_groups

        except Exception as e:
            logger.error(f"Error listing resource groups: {str(e)}")
            raise

    async def get_resource_group(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific resource group"""
        if not self._initialized:
            await self.initialize()

        try:
            rg = await self.resource_client.resource_groups.get(name)
            return {
                "name": rg.name,
                "location": rg.location,
                "id": rg.id,
                "tags": rg.tags or {},
                "provisioning_state": rg.provisioning_state
            }

        except ResourceNotFoundError:
            logger.warning(f"Resource group '{name}' not found")
            return None
        except Exception as e:
            logger.error(f"Error getting resource group '{name}': {str(e)}")
            raise

    async def list_virtual_machines(self, resource_group: Optional[str] = None) -> List[Dict[str, Any]]:
        """List virtual machines"""
        if not self._initialized:
            await self.initialize()

        try:
            vms = []

            if resource_group:
                vm_list = self.compute_client.virtual_machines.list(resource_group)
            else:
                vm_list = self.compute_client.virtual_machines.list_all()

            async for vm in vm_list:
                # Get instance view for runtime status
                try:
                    instance_view = await self.compute_client.virtual_machines.instance_view(
                        vm.id.split('/')[4],  # resource group from ID
                        vm.name
                    )
                    statuses = [status.display_status for status in instance_view.statuses]
                    power_state = next((s for s in statuses if "PowerState" in s), "Unknown")
                except:
                    power_state = "Unknown"

                vms.append({
                    "name": vm.name,
                    "id": vm.id,
                    "location": vm.location,
                    "size": vm.hardware_profile.vm_size if vm.hardware_profile else "Unknown",
                    "os_type": vm.storage_profile.os_disk.os_type if vm.storage_profile.os_disk else "Unknown",
                    "power_state": power_state,
                    "tags": vm.tags or {},
                    "resource_group": vm.id.split('/')[4]
                })

            logger.info(f"Retrieved {len(vms)} virtual machines")
            return vms

        except Exception as e:
            logger.error(f"Error listing virtual machines: {str(e)}")
            raise

    async def get_virtual_machine(self, resource_group: str, name: str) -> Optional[Dict[str, Any]]:
        """Get specific virtual machine details"""
        if not self._initialized:
            await self.initialize()

        try:
            vm = await self.compute_client.virtual_machines.get(resource_group, name)

            # Get instance view
            try:
                instance_view = await self.compute_client.virtual_machines.instance_view(resource_group, name)
                statuses = [status.display_status for status in instance_view.statuses]
                power_state = next((s for s in statuses if "PowerState" in s), "Unknown")
            except:
                power_state = "Unknown"

            return {
                "name": vm.name,
                "id": vm.id,
                "location": vm.location,
                "size": vm.hardware_profile.vm_size if vm.hardware_profile else "Unknown",
                "os_type": vm.storage_profile.os_disk.os_type if vm.storage_profile.os_disk else "Unknown",
                "power_state": power_state,
                "tags": vm.tags or {},
                "resource_group": resource_group,
                "network_interfaces": [nic.id for nic in vm.network_profile.network_interfaces] if vm.network_profile else [],
                "provisioning_state": vm.provisioning_state
            }

        except ResourceNotFoundError:
            logger.warning(f"Virtual machine '{name}' not found in resource group '{resource_group}'")
            return None
        except Exception as e:
            logger.error(f"Error getting virtual machine '{name}': {str(e)}")
            raise

    async def start_virtual_machine(self, resource_group: str, name: str) -> bool:
        """Start a virtual machine"""
        if not self._initialized:
            await self.initialize()

        try:
            operation = await self.compute_client.virtual_machines.begin_start(resource_group, name)
            await operation.wait()
            logger.info(f"Started virtual machine '{name}' in resource group '{resource_group}'")
            return True

        except Exception as e:
            logger.error(f"Error starting virtual machine '{name}': {str(e)}")
            raise

    async def stop_virtual_machine(self, resource_group: str, name: str, deallocate: bool = True) -> bool:
        """Stop a virtual machine"""
        if not self._initialized:
            await self.initialize()

        try:
            if deallocate:
                operation = await self.compute_client.virtual_machines.begin_deallocate(resource_group, name)
            else:
                operation = await self.compute_client.virtual_machines.begin_power_off(resource_group, name)

            await operation.wait()
            action = "deallocated" if deallocate else "powered off"
            logger.info(f"Virtual machine '{name}' {action} in resource group '{resource_group}'")
            return True

        except Exception as e:
            logger.error(f"Error stopping virtual machine '{name}': {str(e)}")
            raise

    async def restart_virtual_machine(self, resource_group: str, name: str) -> bool:
        """Restart a virtual machine"""
        if not self._initialized:
            await self.initialize()

        try:
            operation = await self.compute_client.virtual_machines.begin_restart(resource_group, name)
            await operation.wait()
            logger.info(f"Restarted virtual machine '{name}' in resource group '{resource_group}'")
            return True

        except Exception as e:
            logger.error(f"Error restarting virtual machine '{name}': {str(e)}")
            raise

    async def list_storage_accounts(self, resource_group: Optional[str] = None) -> List[Dict[str, Any]]:
        """List storage accounts"""
        if not self._initialized:
            await self.initialize()

        try:
            storage_accounts = []

            if resource_group:
                account_list = self.storage_client.storage_accounts.list_by_resource_group(resource_group)
            else:
                account_list = self.storage_client.storage_accounts.list()

            async for account in account_list:
                storage_accounts.append({
                    "name": account.name,
                    "id": account.id,
                    "location": account.location,
                    "kind": account.kind,
                    "tier": account.sku.tier if account.sku else "Unknown",
                    "replication": account.sku.name if account.sku else "Unknown",
                    "tags": account.tags or {},
                    "resource_group": account.id.split('/')[4],
                    "provisioning_state": account.provisioning_state,
                    "primary_endpoints": {
                        "blob": account.primary_endpoints.blob if account.primary_endpoints else None,
                        "file": account.primary_endpoints.file if account.primary_endpoints else None,
                        "queue": account.primary_endpoints.queue if account.primary_endpoints else None,
                        "table": account.primary_endpoints.table if account.primary_endpoints else None
                    }
                })

            logger.info(f"Retrieved {len(storage_accounts)} storage accounts")
            return storage_accounts

        except Exception as e:
            logger.error(f"Error listing storage accounts: {str(e)}")
            raise

    async def list_resources(self, resource_group: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all resources in subscription or resource group"""
        if not self._initialized:
            await self.initialize()

        try:
            resources = []

            if resource_group:
                resource_list = self.resource_client.resources.list_by_resource_group(resource_group)
            else:
                resource_list = self.resource_client.resources.list()

            async for resource in resource_list:
                resources.append({
                    "name": resource.name,
                    "id": resource.id,
                    "type": resource.type,
                    "location": resource.location,
                    "tags": resource.tags or {},
                    "resource_group": resource.id.split('/')[4],
                    "kind": getattr(resource, 'kind', None),
                    "sku": getattr(resource, 'sku', None)
                })

            logger.info(f"Retrieved {len(resources)} resources")
            return resources

        except Exception as e:
            logger.error(f"Error listing resources: {str(e)}")
            raise

    async def get_resource_usage(self, location: str) -> Dict[str, Any]:
        """Get resource usage and quotas for a location"""
        if not self._initialized:
            await self.initialize()

        try:
            # Get compute usage
            compute_usage = []
            async for usage in self.compute_client.usage.list(location):
                compute_usage.append({
                    "name": usage.name.localized_value,
                    "current_value": usage.current_value,
                    "limit": usage.limit,
                    "unit": usage.unit
                })

            return {
                "location": location,
                "compute_usage": compute_usage,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting resource usage for location '{location}': {str(e)}")
            raise

    async def tag_resource(self, resource_id: str, tags: Dict[str, str]) -> bool:
        """Add or update tags on a resource"""
        if not self._initialized:
            await self.initialize()

        try:
            # Get current resource
            resource = await self.resource_client.resources.get_by_id(
                resource_id,
                api_version="2021-04-01"
            )

            # Merge tags
            current_tags = resource.tags or {}
            current_tags.update(tags)

            # Update resource with new tags
            resource.tags = current_tags
            await self.resource_client.resources.begin_create_or_update_by_id(
                resource_id,
                api_version="2021-04-01",
                parameters=resource
            )

            logger.info(f"Updated tags for resource '{resource_id}'")
            return True

        except Exception as e:
            logger.error(f"Error tagging resource '{resource_id}': {str(e)}")
            raise