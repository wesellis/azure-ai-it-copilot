"""
Azure Client Factory Service
Provides centralized Azure client management with proper configuration
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache

from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.monitor.query import LogsQueryClient

from core.interfaces import IAzureClientFactory, IConfigurationProvider
from core.base_classes import BaseService
from core.exceptions import AzureClientError, ConfigurationError

logger = logging.getLogger(__name__)


class AzureClientFactory(BaseService, IAzureClientFactory):
    """Factory for creating and managing Azure service clients"""

    def __init__(self, config_provider: IConfigurationProvider):
        super().__init__(config_provider)
        self._credential: Optional[DefaultAzureCredential] = None
        self._clients: Dict[str, Any] = {}
        self._subscription_id: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize the Azure client factory"""
        try:
            # Get Azure configuration
            azure_config = self.config_provider.get_azure_credentials()
            self._subscription_id = azure_config.get("subscription_id")

            if not self._subscription_id:
                raise ConfigurationError("Azure subscription ID not configured")

            # Initialize credential
            self._credential = DefaultAzureCredential()

            # Test credential by getting a token
            await self._test_credential()

            logger.info("Azure client factory initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure client factory: {e}")
            raise AzureClientError(f"Azure client factory initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the Azure client factory"""
        self._clients.clear()
        self._credential = None
        logger.info("Azure client factory shutdown")

    async def _test_credential(self) -> None:
        """Test Azure credential"""
        try:
            # Try to get a token for Azure Resource Manager
            token = self._credential.get_token("https://management.azure.com/.default")
            if not token:
                raise AzureClientError("Failed to obtain Azure authentication token")
        except Exception as e:
            raise AzureClientError(f"Azure credential test failed: {e}")

    @lru_cache(maxsize=1)
    def get_resource_client(self) -> ResourceManagementClient:
        """Get Azure Resource Management client"""
        if "resource" not in self._clients:
            try:
                if not self._credential or not self._subscription_id:
                    raise AzureClientError("Azure client factory not properly initialized")

                client = ResourceManagementClient(
                    credential=self._credential,
                    subscription_id=self._subscription_id
                )
                self._clients["resource"] = client
                logger.debug("Created Azure Resource Management client")

            except Exception as e:
                raise AzureClientError(f"Failed to create Resource Management client: {e}")

        return self._clients["resource"]

    @lru_cache(maxsize=1)
    def get_compute_client(self) -> ComputeManagementClient:
        """Get Azure Compute Management client"""
        if "compute" not in self._clients:
            try:
                if not self._credential or not self._subscription_id:
                    raise AzureClientError("Azure client factory not properly initialized")

                client = ComputeManagementClient(
                    credential=self._credential,
                    subscription_id=self._subscription_id
                )
                self._clients["compute"] = client
                logger.debug("Created Azure Compute Management client")

            except Exception as e:
                raise AzureClientError(f"Failed to create Compute Management client: {e}")

        return self._clients["compute"]

    @lru_cache(maxsize=1)
    def get_network_client(self) -> NetworkManagementClient:
        """Get Azure Network Management client"""
        if "network" not in self._clients:
            try:
                if not self._credential or not self._subscription_id:
                    raise AzureClientError("Azure client factory not properly initialized")

                client = NetworkManagementClient(
                    credential=self._credential,
                    subscription_id=self._subscription_id
                )
                self._clients["network"] = client
                logger.debug("Created Azure Network Management client")

            except Exception as e:
                raise AzureClientError(f"Failed to create Network Management client: {e}")

        return self._clients["network"]

    @lru_cache(maxsize=1)
    def get_storage_client(self) -> StorageManagementClient:
        """Get Azure Storage Management client"""
        if "storage" not in self._clients:
            try:
                if not self._credential or not self._subscription_id:
                    raise AzureClientError("Azure client factory not properly initialized")

                client = StorageManagementClient(
                    credential=self._credential,
                    subscription_id=self._subscription_id
                )
                self._clients["storage"] = client
                logger.debug("Created Azure Storage Management client")

            except Exception as e:
                raise AzureClientError(f"Failed to create Storage Management client: {e}")

        return self._clients["storage"]

    @lru_cache(maxsize=1)
    def get_monitor_client(self) -> LogsQueryClient:
        """Get Azure Monitor client"""
        if "monitor" not in self._clients:
            try:
                if not self._credential:
                    raise AzureClientError("Azure client factory not properly initialized")

                client = LogsQueryClient(credential=self._credential)
                self._clients["monitor"] = client
                logger.debug("Created Azure Monitor client")

            except Exception as e:
                raise AzureClientError(f"Failed to create Monitor client: {e}")

        return self._clients["monitor"]

    def get_cost_management_client(self):
        """Get Azure Cost Management client (optional)"""
        if "cost_management" not in self._clients:
            try:
                # Import here since it might not be available
                from azure.mgmt.costmanagement import CostManagementClient

                if not self._credential:
                    raise AzureClientError("Azure client factory not properly initialized")

                client = CostManagementClient(credential=self._credential)
                self._clients["cost_management"] = client
                logger.debug("Created Azure Cost Management client")

            except ImportError:
                logger.warning("Azure Cost Management client not available")
                return None
            except Exception as e:
                raise AzureClientError(f"Failed to create Cost Management client: {e}")

        return self._clients["cost_management"]

    def get_advisor_client(self):
        """Get Azure Advisor client (optional)"""
        if "advisor" not in self._clients:
            try:
                # Import here since it might not be available
                from azure.mgmt.advisor import AdvisorManagementClient

                if not self._credential or not self._subscription_id:
                    raise AzureClientError("Azure client factory not properly initialized")

                client = AdvisorManagementClient(
                    credential=self._credential,
                    subscription_id=self._subscription_id
                )
                self._clients["advisor"] = client
                logger.debug("Created Azure Advisor client")

            except ImportError:
                logger.warning("Azure Advisor client not available")
                return None
            except Exception as e:
                raise AzureClientError(f"Failed to create Advisor client: {e}")

        return self._clients["advisor"]

    def get_security_client(self):
        """Get Azure Security Center client (optional)"""
        if "security" not in self._clients:
            try:
                # Import here since it might not be available
                from azure.mgmt.security import SecurityCenter

                if not self._credential or not self._subscription_id:
                    raise AzureClientError("Azure client factory not properly initialized")

                client = SecurityCenter(
                    credential=self._credential,
                    subscription_id=self._subscription_id
                )
                self._clients["security"] = client
                logger.debug("Created Azure Security Center client")

            except ImportError:
                logger.warning("Azure Security Center client not available")
                return None
            except Exception as e:
                raise AzureClientError(f"Failed to create Security Center client: {e}")

        return self._clients["security"]

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Azure clients"""
        health_status = await super().health_check()

        try:
            # Test resource client
            resource_client = self.get_resource_client()
            resource_groups = list(resource_client.resource_groups.list())
            health_status["resource_client"] = {
                "status": "healthy",
                "resource_groups_count": len(resource_groups)
            }

        except Exception as e:
            health_status["resource_client"] = {
                "status": "unhealthy",
                "error": str(e)
            }

        return health_status

    def get_credential(self) -> DefaultAzureCredential:
        """Get the Azure credential"""
        return self._credential

    def get_subscription_id(self) -> str:
        """Get the subscription ID"""
        return self._subscription_id