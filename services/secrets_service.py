"""
Secrets Service Implementation
Provides secure secrets management with multiple backends
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime

from core.interfaces import ISecretsManager, IConfigurationProvider
from core.base_classes import BaseService
from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class SecretsService(BaseService, ISecretsManager):
    """Multi-backend secrets management service"""

    def __init__(self, config_provider: IConfigurationProvider):
        super().__init__(config_provider)
        self._providers: Dict[str, Any] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes

    async def initialize(self) -> None:
        """Initialize secrets service with available providers"""
        # Initialize Azure Key Vault if configured
        try:
            vault_url = self.config_provider.get_setting("azure_key_vault_url")
            if vault_url:
                self._providers["azure_keyvault"] = AzureKeyVaultProvider(vault_url)
                logger.info("Azure Key Vault provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure Key Vault: {e}")

        # Initialize environment provider (always available)
        self._providers["environment"] = EnvironmentProvider()
        logger.info("Environment secrets provider initialized")

        # Initialize file provider if configured
        try:
            secrets_file = self.config_provider.get_setting("secrets_file_path")
            if secrets_file and os.path.exists(secrets_file):
                self._providers["file"] = FileSecretsProvider(secrets_file)
                logger.info("File secrets provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize file secrets provider: {e}")

        logger.info(f"Secrets service initialized with {len(self._providers)} providers")

    async def shutdown(self) -> None:
        """Shutdown secrets service"""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                await provider.close()

        self._providers.clear()
        self._cache.clear()
        logger.info("Secrets service shutdown")

    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret value from available providers"""
        try:
            # Check cache first
            if self._is_cached(key):
                cached_value = self._cache[key]
                if self._is_cache_valid(cached_value):
                    return cached_value["value"]
                else:
                    # Remove expired cache entry
                    del self._cache[key]

            # Try providers in order of preference
            provider_order = ["azure_keyvault", "environment", "file"]

            for provider_name in provider_order:
                provider = self._providers.get(provider_name)
                if provider:
                    try:
                        value = await provider.get_secret(key)
                        if value is not None:
                            # Cache the value
                            self._cache[key] = {
                                "value": value,
                                "retrieved_at": datetime.utcnow(),
                                "provider": provider_name
                            }
                            logger.debug(f"Secret '{key}' retrieved from {provider_name}")
                            return value
                    except Exception as e:
                        logger.warning(f"Failed to get secret '{key}' from {provider_name}: {e}")

            logger.warning(f"Secret '{key}' not found in any provider")
            return None

        except Exception as e:
            logger.error(f"Error retrieving secret '{key}': {e}")
            return None

    async def set_secret(self, key: str, value: str) -> bool:
        """Set secret value in primary provider"""
        try:
            # Use Azure Key Vault as primary if available, otherwise environment
            primary_provider = self._providers.get("azure_keyvault") or self._providers.get("environment")

            if primary_provider:
                success = await primary_provider.set_secret(key, value)
                if success:
                    # Update cache
                    self._cache[key] = {
                        "value": value,
                        "retrieved_at": datetime.utcnow(),
                        "provider": "azure_keyvault" if "azure_keyvault" in self._providers else "environment"
                    }
                    logger.info(f"Secret '{key}' set successfully")
                    return True

            logger.error(f"No suitable provider available to set secret '{key}'")
            return False

        except Exception as e:
            logger.error(f"Error setting secret '{key}': {e}")
            return False

    async def delete_secret(self, key: str) -> bool:
        """Delete secret from all providers"""
        try:
            success = True

            for provider_name, provider in self._providers.items():
                try:
                    if hasattr(provider, 'delete_secret'):
                        result = await provider.delete_secret(key)
                        if not result:
                            success = False
                            logger.warning(f"Failed to delete secret '{key}' from {provider_name}")
                except Exception as e:
                    logger.warning(f"Error deleting secret '{key}' from {provider_name}: {e}")
                    success = False

            # Remove from cache
            if key in self._cache:
                del self._cache[key]

            if success:
                logger.info(f"Secret '{key}' deleted successfully")

            return success

        except Exception as e:
            logger.error(f"Error deleting secret '{key}': {e}")
            return False

    def _is_cached(self, key: str) -> bool:
        """Check if secret is in cache"""
        return key in self._cache

    def _is_cache_valid(self, cached_entry: Dict[str, Any]) -> bool:
        """Check if cached entry is still valid"""
        retrieved_at = cached_entry["retrieved_at"]
        age = (datetime.utcnow() - retrieved_at).total_seconds()
        return age < self._cache_ttl

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on secrets service"""
        health_status = await super().health_check()

        provider_status = {}
        for provider_name, provider in self._providers.items():
            try:
                if hasattr(provider, 'health_check'):
                    provider_status[provider_name] = await provider.health_check()
                else:
                    provider_status[provider_name] = {"status": "available"}
            except Exception as e:
                provider_status[provider_name] = {"status": "error", "error": str(e)}

        health_status.update({
            "providers": provider_status,
            "cache_entries": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl
        })

        return health_status


class AzureKeyVaultProvider:
    """Azure Key Vault secrets provider"""

    def __init__(self, vault_url: str):
        self.vault_url = vault_url
        self._client = None

    async def _get_client(self):
        """Get Key Vault client"""
        if not self._client:
            try:
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()
                self._client = SecretClient(vault_url=self.vault_url, credential=credential)
            except ImportError:
                raise ConfigurationError("Azure Key Vault SDK not available")

        return self._client

    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret from Azure Key Vault"""
        try:
            client = await self._get_client()
            secret = client.get_secret(key)
            return secret.value
        except Exception as e:
            logger.debug(f"Failed to get secret '{key}' from Key Vault: {e}")
            return None

    async def set_secret(self, key: str, value: str) -> bool:
        """Set secret in Azure Key Vault"""
        try:
            client = await self._get_client()
            client.set_secret(key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to set secret '{key}' in Key Vault: {e}")
            return False

    async def delete_secret(self, key: str) -> bool:
        """Delete secret from Azure Key Vault"""
        try:
            client = await self._get_client()
            client.begin_delete_secret(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret '{key}' from Key Vault: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check Key Vault connectivity"""
        try:
            client = await self._get_client()
            # Try to list secrets to verify connectivity
            list(client.list_properties_of_secrets(max_page_size=1))
            return {"status": "healthy", "vault_url": self.vault_url}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class EnvironmentProvider:
    """Environment variables secrets provider"""

    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret from environment variables"""
        return os.getenv(key)

    async def set_secret(self, key: str, value: str) -> bool:
        """Set secret in environment (runtime only)"""
        try:
            os.environ[key] = value
            return True
        except Exception as e:
            logger.error(f"Failed to set environment variable '{key}': {e}")
            return False

    async def delete_secret(self, key: str) -> bool:
        """Delete secret from environment"""
        try:
            if key in os.environ:
                del os.environ[key]
            return True
        except Exception as e:
            logger.error(f"Failed to delete environment variable '{key}': {e}")
            return False


class FileSecretsProvider:
    """File-based secrets provider"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._secrets: Dict[str, str] = {}
        self._load_secrets()

    def _load_secrets(self):
        """Load secrets from file"""
        try:
            import json
            with open(self.file_path, 'r') as f:
                self._secrets = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load secrets from {self.file_path}: {e}")
            self._secrets = {}

    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret from file"""
        return self._secrets.get(key)

    async def set_secret(self, key: str, value: str) -> bool:
        """Set secret in file"""
        try:
            self._secrets[key] = value
            return self._save_secrets()
        except Exception as e:
            logger.error(f"Failed to set secret '{key}' in file: {e}")
            return False

    async def delete_secret(self, key: str) -> bool:
        """Delete secret from file"""
        try:
            if key in self._secrets:
                del self._secrets[key]
                return self._save_secrets()
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret '{key}' from file: {e}")
            return False

    def _save_secrets(self) -> bool:
        """Save secrets to file"""
        try:
            import json
            with open(self.file_path, 'w') as f:
                json.dump(self._secrets, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save secrets to {self.file_path}: {e}")
            return False