"""
Azure Automation Module - Python implementation for Azure resource management
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum

from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.web import WebSiteManagementClient
from azure.mgmt.sql import SqlManagementClient
from azure.mgmt.keyvault import KeyVaultManagementClient
from azure.mgmt.monitor import MonitorManagementClient
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.mgmt.resource.resources.models import ResourceGroup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceAction(Enum):
    """Enum for resource actions"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"
    SCALE = "scale"
    RESTART = "restart"
    STOP = "stop"
    START = "start"


class AzureAutomation:
    """Main class for Azure automation operations"""

    def __init__(self, subscription_id: Optional[str] = None, credential=None):
        """
        Initialize Azure automation client

        Args:
            subscription_id: Azure subscription ID
            credential: Azure credential object
        """
        self.subscription_id = subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID")

        # Initialize credential
        self.credential = credential or self._get_credential()

        # Initialize Azure clients
        self._initialize_clients()

        # Resource type mapping
        self.resource_handlers = {
            "virtual_machine": self._handle_vm_operation,
            "storage_account": self._handle_storage_operation,
            "web_app": self._handle_webapp_operation,
            "sql_database": self._handle_sql_operation,
            "key_vault": self._handle_keyvault_operation,
            "network": self._handle_network_operation,
            "resource_group": self._handle_resource_group_operation
        }

    def _get_credential(self):
        """Get Azure credential based on environment"""
        # Try service principal first
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        tenant_id = os.getenv("AZURE_TENANT_ID")

        if all([client_id, client_secret, tenant_id]):
            logger.info("Using service principal authentication")
            return ClientSecretCredential(
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id
            )

        # Fall back to default credential
        logger.info("Using default Azure credential")
        return DefaultAzureCredential()

    def _initialize_clients(self):
        """Initialize Azure management clients"""
        try:
            self.resource_client = ResourceManagementClient(
                self.credential, self.subscription_id
            )
            self.compute_client = ComputeManagementClient(
                self.credential, self.subscription_id
            )
            self.network_client = NetworkManagementClient(
                self.credential, self.subscription_id
            )
            self.storage_client = StorageManagementClient(
                self.credential, self.subscription_id
            )
            self.web_client = WebSiteManagementClient(
                self.credential, self.subscription_id
            )
            self.sql_client = SqlManagementClient(
                self.credential, self.subscription_id
            )
            self.keyvault_client = KeyVaultManagementClient(
                self.credential, self.subscription_id
            )
            self.monitor_client = MonitorManagementClient(
                self.credential, self.subscription_id
            )

            logger.info("Azure clients initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure clients: {str(e)}")
            raise

    async def execute_operation(
        self,
        resource_type: str,
        action: ResourceAction,
        config: Dict[str, Any],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute an Azure resource operation

        Args:
            resource_type: Type of resource to operate on
            action: Action to perform
            config: Configuration for the operation
            dry_run: If True, simulate the operation

        Returns:
            Operation result
        """
        logger.info(f"Executing {action.value} on {resource_type}")

        if dry_run:
            logger.info("DRY RUN mode - no changes will be made")
            return {
                "status": "dry_run",
                "resource_type": resource_type,
                "action": action.value,
                "config": config,
                "message": "Operation simulated successfully"
            }

        try:
            # Get appropriate handler
            handler = self.resource_handlers.get(resource_type)

            if not handler:
                raise ValueError(f"Unsupported resource type: {resource_type}")

            # Execute operation
            result = await handler(action, config)

            return {
                "status": "success",
                "resource_type": resource_type,
                "action": action.value,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }

        except AzureError as e:
            logger.error(f"Azure error during operation: {str(e)}")
            return {
                "status": "error",
                "resource_type": resource_type,
                "action": action.value,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "status": "error",
                "resource_type": resource_type,
                "action": action.value,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _handle_vm_operation(
        self,
        action: ResourceAction,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle virtual machine operations"""

        resource_group = config.get("resource_group")
        vm_name = config.get("name")

        if action == ResourceAction.CREATE:
            return await self._create_vm(config)

        elif action == ResourceAction.UPDATE:
            return await self._update_vm(resource_group, vm_name, config)

        elif action == ResourceAction.DELETE:
            return await self._delete_vm(resource_group, vm_name)

        elif action == ResourceAction.QUERY:
            return await self._query_vms(resource_group, vm_name)

        elif action == ResourceAction.START:
            return await self._start_vm(resource_group, vm_name)

        elif action == ResourceAction.STOP:
            return await self._stop_vm(resource_group, vm_name)

        elif action == ResourceAction.RESTART:
            return await self._restart_vm(resource_group, vm_name)

        elif action == ResourceAction.SCALE:
            return await self._scale_vm(resource_group, vm_name, config)

        else:
            raise ValueError(f"Unsupported VM action: {action}")

    async def _create_vm(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new virtual machine"""

        resource_group = config.get("resource_group")
        vm_name = config.get("name")
        location = config.get("location", "eastus")
        vm_size = config.get("size", "Standard_B2s")

        # Ensure resource group exists
        await self._ensure_resource_group(resource_group, location)

        # Create network interface if not provided
        nic = await self._ensure_network_interface(resource_group, vm_name, location)

        # Create VM
        vm_parameters = {
            "location": location,
            "hardware_profile": {"vm_size": vm_size},
            "storage_profile": {
                "image_reference": {
                    "publisher": config.get("publisher", "Canonical"),
                    "offer": config.get("offer", "UbuntuServer"),
                    "sku": config.get("sku", "18.04-LTS"),
                    "version": "latest"
                },
                "os_disk": {
                    "name": f"{vm_name}-osdisk",
                    "caching": "ReadWrite",
                    "create_option": "FromImage",
                    "managed_disk": {"storage_account_type": "Standard_LRS"}
                }
            },
            "os_profile": {
                "computer_name": vm_name,
                "admin_username": config.get("admin_username", "azureuser"),
                "admin_password": config.get("admin_password", "P@ssw0rd123!"),
                "linux_configuration": {
                    "disable_password_authentication": False
                }
            },
            "network_profile": {
                "network_interfaces": [{"id": nic.id}]
            }
        }

        # Start VM creation
        async_vm_creation = self.compute_client.virtual_machines.begin_create_or_update(
            resource_group, vm_name, vm_parameters
        )

        # Wait for creation to complete
        vm = async_vm_creation.result()

        return {
            "id": vm.id,
            "name": vm.name,
            "location": vm.location,
            "provisioning_state": vm.provisioning_state,
            "vm_id": vm.vm_id
        }

    async def _update_vm(
        self,
        resource_group: str,
        vm_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an existing VM"""

        # Get existing VM
        vm = self.compute_client.virtual_machines.get(resource_group, vm_name)

        # Update properties
        if "size" in config:
            vm.hardware_profile.vm_size = config["size"]

        if "tags" in config:
            vm.tags = config["tags"]

        # Apply updates
        async_update = self.compute_client.virtual_machines.begin_create_or_update(
            resource_group, vm_name, vm
        )

        updated_vm = async_update.result()

        return {
            "id": updated_vm.id,
            "name": updated_vm.name,
            "status": "updated",
            "provisioning_state": updated_vm.provisioning_state
        }

    async def _delete_vm(self, resource_group: str, vm_name: str) -> Dict[str, Any]:
        """Delete a VM and its associated resources"""

        # Start deletion
        async_delete = self.compute_client.virtual_machines.begin_delete(
            resource_group, vm_name
        )

        # Wait for deletion
        async_delete.wait()

        # Also delete associated resources (NIC, disk, etc.)
        # This is simplified - in production, you'd want more control

        return {
            "status": "deleted",
            "vm_name": vm_name,
            "resource_group": resource_group
        }

    async def _query_vms(
        self,
        resource_group: Optional[str] = None,
        vm_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query VMs"""

        if vm_name and resource_group:
            # Get specific VM
            vm = self.compute_client.virtual_machines.get(
                resource_group, vm_name, expand='instanceView'
            )

            return {
                "id": vm.id,
                "name": vm.name,
                "location": vm.location,
                "size": vm.hardware_profile.vm_size,
                "provisioning_state": vm.provisioning_state,
                "power_state": self._get_power_state(vm)
            }

        elif resource_group:
            # List VMs in resource group
            vms = list(self.compute_client.virtual_machines.list(resource_group))
        else:
            # List all VMs
            vms = list(self.compute_client.virtual_machines.list_all())

        return {
            "count": len(vms),
            "vms": [
                {
                    "id": vm.id,
                    "name": vm.name,
                    "location": vm.location,
                    "size": vm.hardware_profile.vm_size
                }
                for vm in vms
            ]
        }

    async def _start_vm(self, resource_group: str, vm_name: str) -> Dict[str, Any]:
        """Start a stopped VM"""

        async_start = self.compute_client.virtual_machines.begin_start(
            resource_group, vm_name
        )

        async_start.wait()

        return {
            "status": "started",
            "vm_name": vm_name,
            "resource_group": resource_group
        }

    async def _stop_vm(self, resource_group: str, vm_name: str) -> Dict[str, Any]:
        """Stop a running VM"""

        async_stop = self.compute_client.virtual_machines.begin_deallocate(
            resource_group, vm_name
        )

        async_stop.wait()

        return {
            "status": "stopped",
            "vm_name": vm_name,
            "resource_group": resource_group
        }

    async def _restart_vm(self, resource_group: str, vm_name: str) -> Dict[str, Any]:
        """Restart a VM"""

        async_restart = self.compute_client.virtual_machines.begin_restart(
            resource_group, vm_name
        )

        async_restart.wait()

        return {
            "status": "restarted",
            "vm_name": vm_name,
            "resource_group": resource_group
        }

    async def _scale_vm(
        self,
        resource_group: str,
        vm_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Scale a VM to a different size"""

        new_size = config.get("new_size")

        if not new_size:
            raise ValueError("new_size is required for scaling")

        # Get VM
        vm = self.compute_client.virtual_machines.get(resource_group, vm_name)

        # Update size
        vm.hardware_profile.vm_size = new_size

        # Apply change
        async_update = self.compute_client.virtual_machines.begin_create_or_update(
            resource_group, vm_name, vm
        )

        updated_vm = async_update.result()

        return {
            "status": "scaled",
            "vm_name": vm_name,
            "old_size": config.get("old_size", "unknown"),
            "new_size": new_size,
            "provisioning_state": updated_vm.provisioning_state
        }

    async def _handle_storage_operation(
        self,
        action: ResourceAction,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle storage account operations"""

        resource_group = config.get("resource_group")
        account_name = config.get("name")

        if action == ResourceAction.CREATE:
            return await self._create_storage_account(config)

        elif action == ResourceAction.UPDATE:
            return await self._update_storage_account(resource_group, account_name, config)

        elif action == ResourceAction.DELETE:
            return await self._delete_storage_account(resource_group, account_name)

        elif action == ResourceAction.QUERY:
            return await self._query_storage_accounts(resource_group, account_name)

        else:
            raise ValueError(f"Unsupported storage action: {action}")

    async def _create_storage_account(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new storage account"""

        resource_group = config.get("resource_group")
        account_name = config.get("name")
        location = config.get("location", "eastus")
        sku = config.get("sku", "Standard_LRS")
        kind = config.get("kind", "StorageV2")

        # Ensure resource group exists
        await self._ensure_resource_group(resource_group, location)

        # Storage account parameters
        parameters = {
            "sku": {"name": sku},
            "kind": kind,
            "location": location,
            "enable_https_traffic_only": True,
            "minimum_tls_version": "TLS1_2",
            "tags": config.get("tags", {})
        }

        # Create storage account
        async_creation = self.storage_client.storage_accounts.begin_create(
            resource_group, account_name, parameters
        )

        storage_account = async_creation.result()

        return {
            "id": storage_account.id,
            "name": storage_account.name,
            "location": storage_account.location,
            "provisioning_state": storage_account.provisioning_state,
            "primary_endpoints": storage_account.primary_endpoints
        }

    async def _update_storage_account(
        self,
        resource_group: str,
        account_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update storage account properties"""

        update_params = {}

        if "tags" in config:
            update_params["tags"] = config["tags"]

        if "enable_https_only" in config:
            update_params["enable_https_traffic_only"] = config["enable_https_only"]

        if "min_tls_version" in config:
            update_params["minimum_tls_version"] = config["min_tls_version"]

        # Apply updates
        storage_account = self.storage_client.storage_accounts.update(
            resource_group, account_name, update_params
        )

        return {
            "id": storage_account.id,
            "name": storage_account.name,
            "status": "updated"
        }

    async def _delete_storage_account(
        self,
        resource_group: str,
        account_name: str
    ) -> Dict[str, Any]:
        """Delete a storage account"""

        self.storage_client.storage_accounts.delete(resource_group, account_name)

        return {
            "status": "deleted",
            "account_name": account_name,
            "resource_group": resource_group
        }

    async def _query_storage_accounts(
        self,
        resource_group: Optional[str] = None,
        account_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query storage accounts"""

        if account_name and resource_group:
            # Get specific account
            account = self.storage_client.storage_accounts.get_properties(
                resource_group, account_name
            )

            return {
                "id": account.id,
                "name": account.name,
                "location": account.location,
                "sku": account.sku.name,
                "kind": account.kind,
                "provisioning_state": account.provisioning_state
            }

        elif resource_group:
            # List accounts in resource group
            accounts = list(self.storage_client.storage_accounts.list_by_resource_group(resource_group))
        else:
            # List all accounts
            accounts = list(self.storage_client.storage_accounts.list())

        return {
            "count": len(accounts),
            "storage_accounts": [
                {
                    "id": acc.id,
                    "name": acc.name,
                    "location": acc.location,
                    "sku": acc.sku.name
                }
                for acc in accounts
            ]
        }

    async def _handle_webapp_operation(
        self,
        action: ResourceAction,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle web app operations"""

        # Simplified implementation - extend as needed
        if action == ResourceAction.CREATE:
            return await self._create_webapp(config)
        elif action == ResourceAction.QUERY:
            return await self._query_webapps(config.get("resource_group"))
        else:
            raise ValueError(f"Unsupported web app action: {action}")

    async def _create_webapp(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new web app"""

        resource_group = config.get("resource_group")
        app_name = config.get("name")
        location = config.get("location", "eastus")

        # Create App Service Plan first
        plan_name = f"{app_name}-plan"
        app_service_plan = {
            "location": location,
            "sku": {"name": "B1", "tier": "Basic"},
            "reserved": config.get("is_linux", False)
        }

        plan = self.web_client.app_service_plans.begin_create_or_update(
            resource_group, plan_name, app_service_plan
        ).result()

        # Create Web App
        site_config = {
            "location": location,
            "server_farm_id": plan.id,
            "site_config": {
                "linux_fx_version": config.get("runtime", "PYTHON|3.9") if config.get("is_linux") else None,
                "always_on": True
            }
        }

        webapp = self.web_client.web_apps.begin_create_or_update(
            resource_group, app_name, site_config
        ).result()

        return {
            "id": webapp.id,
            "name": webapp.name,
            "url": f"https://{webapp.default_host_name}",
            "state": webapp.state
        }

    async def _query_webapps(self, resource_group: Optional[str] = None) -> Dict[str, Any]:
        """Query web apps"""

        if resource_group:
            apps = list(self.web_client.web_apps.list_by_resource_group(resource_group))
        else:
            apps = list(self.web_client.web_apps.list())

        return {
            "count": len(apps),
            "web_apps": [
                {
                    "id": app.id,
                    "name": app.name,
                    "url": f"https://{app.default_host_name}",
                    "state": app.state
                }
                for app in apps
            ]
        }

    async def _handle_sql_operation(
        self,
        action: ResourceAction,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle SQL database operations"""

        # Placeholder - implement as needed
        return {"status": "not_implemented", "resource_type": "sql_database"}

    async def _handle_keyvault_operation(
        self,
        action: ResourceAction,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Key Vault operations"""

        # Placeholder - implement as needed
        return {"status": "not_implemented", "resource_type": "key_vault"}

    async def _handle_network_operation(
        self,
        action: ResourceAction,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle network operations"""

        # Placeholder - implement as needed
        return {"status": "not_implemented", "resource_type": "network"}

    async def _handle_resource_group_operation(
        self,
        action: ResourceAction,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle resource group operations"""

        if action == ResourceAction.CREATE:
            return await self._create_resource_group(config)
        elif action == ResourceAction.DELETE:
            return await self._delete_resource_group(config.get("name"))
        elif action == ResourceAction.QUERY:
            return await self._query_resource_groups()
        else:
            raise ValueError(f"Unsupported resource group action: {action}")

    async def _create_resource_group(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a resource group"""

        name = config.get("name")
        location = config.get("location", "eastus")

        rg = self.resource_client.resource_groups.create_or_update(
            name,
            {"location": location, "tags": config.get("tags", {})}
        )

        return {
            "id": rg.id,
            "name": rg.name,
            "location": rg.location,
            "provisioning_state": rg.properties.provisioning_state
        }

    async def _delete_resource_group(self, name: str) -> Dict[str, Any]:
        """Delete a resource group"""

        async_delete = self.resource_client.resource_groups.begin_delete(name)
        async_delete.wait()

        return {
            "status": "deleted",
            "resource_group": name
        }

    async def _query_resource_groups(self) -> Dict[str, Any]:
        """List all resource groups"""

        groups = list(self.resource_client.resource_groups.list())

        return {
            "count": len(groups),
            "resource_groups": [
                {
                    "id": rg.id,
                    "name": rg.name,
                    "location": rg.location
                }
                for rg in groups
            ]
        }

    # Helper methods

    async def _ensure_resource_group(self, name: str, location: str) -> ResourceGroup:
        """Ensure resource group exists"""

        try:
            rg = self.resource_client.resource_groups.get(name)
            logger.info(f"Resource group {name} already exists")
            return rg
        except ResourceNotFoundError:
            logger.info(f"Creating resource group {name}")
            return self.resource_client.resource_groups.create_or_update(
                name, {"location": location}
            )

    async def _ensure_network_interface(
        self,
        resource_group: str,
        vm_name: str,
        location: str
    ) -> Any:
        """Ensure network interface exists for VM"""

        nic_name = f"{vm_name}-nic"

        # Check if NIC exists
        try:
            nic = self.network_client.network_interfaces.get(resource_group, nic_name)
            return nic
        except ResourceNotFoundError:
            pass

        # Create VNet if needed
        vnet_name = f"{resource_group}-vnet"
        subnet_name = "default"

        try:
            vnet = self.network_client.virtual_networks.get(resource_group, vnet_name)
        except ResourceNotFoundError:
            # Create VNet
            vnet_params = {
                "location": location,
                "address_space": {"address_prefixes": ["10.0.0.0/16"]},
                "subnets": [{
                    "name": subnet_name,
                    "address_prefix": "10.0.0.0/24"
                }]
            }

            async_vnet = self.network_client.virtual_networks.begin_create_or_update(
                resource_group, vnet_name, vnet_params
            )
            vnet = async_vnet.result()

        # Get subnet
        subnet = self.network_client.subnets.get(resource_group, vnet_name, subnet_name)

        # Create public IP
        public_ip_name = f"{vm_name}-ip"
        public_ip_params = {
            "location": location,
            "public_ip_allocation_method": "Dynamic"
        }

        async_public_ip = self.network_client.public_ip_addresses.begin_create_or_update(
            resource_group, public_ip_name, public_ip_params
        )
        public_ip = async_public_ip.result()

        # Create NIC
        nic_params = {
            "location": location,
            "ip_configurations": [{
                "name": "ipconfig1",
                "subnet": {"id": subnet.id},
                "public_ip_address": {"id": public_ip.id}
            }]
        }

        async_nic = self.network_client.network_interfaces.begin_create_or_update(
            resource_group, nic_name, nic_params
        )

        return async_nic.result()

    def _get_power_state(self, vm) -> str:
        """Get VM power state from instance view"""

        if hasattr(vm, 'instance_view') and vm.instance_view:
            for status in vm.instance_view.statuses:
                if status.code.startswith('PowerState/'):
                    return status.code.split('/')[-1]

        return "unknown"


# Example usage
async def main():
    """Example usage of Azure automation"""

    automation = AzureAutomation()

    # Create a VM
    vm_config = {
        "resource_group": "test-rg",
        "name": "test-vm-001",
        "location": "eastus",
        "size": "Standard_B2s",
        "admin_username": "azureuser",
        "admin_password": "SecureP@ssw0rd123!"
    }

    result = await automation.execute_operation(
        resource_type="virtual_machine",
        action=ResourceAction.CREATE,
        config=vm_config,
        dry_run=True  # Set to False to actually create
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())