"""
Secrets management integration for Azure AI IT Copilot
Supports Azure Key Vault, environment variables, and local secrets
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Optional Azure Key Vault imports
try:
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential
    KEYVAULT_AVAILABLE = True
except ImportError:
    SecretClient = None
    DefaultAzureCredential = None
    KEYVAULT_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecretSource(Enum):
    """Sources for secrets"""
    ENVIRONMENT = "environment"
    AZURE_KEYVAULT = "azure_keyvault"
    LOCAL_FILE = "local_file"


@dataclass
class SecretInfo:
    """Information about a secret"""
    key: str
    source: SecretSource
    exists: bool
    last_updated: Optional[str] = None
    expires: Optional[str] = None


class SecretProvider(ABC):
    """Abstract base class for secret providers"""

    @abstractmethod
    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret by key"""
        pass

    @abstractmethod
    async def set_secret(self, key: str, value: str) -> bool:
        """Set a secret value"""
        pass

    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List available secret keys"""
        pass

    @abstractmethod
    async def delete_secret(self, key: str) -> bool:
        """Delete a secret"""
        pass

    @abstractmethod
    def get_info(self, key: str) -> Optional[SecretInfo]:
        """Get information about a secret"""
        pass


class EnvironmentSecretProvider(SecretProvider):
    """Provider for environment variable secrets"""

    def __init__(self):
        self.prefix = "SECRET_"

    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret from environment variables"""
        env_key = f"{self.prefix}{key.upper()}"
        return os.getenv(env_key) or os.getenv(key)

    async def set_secret(self, key: str, value: str) -> bool:
        """Set secret in environment (not persistent)"""
        env_key = f"{self.prefix}{key.upper()}"
        os.environ[env_key] = value
        return True

    async def list_secrets(self) -> List[str]:
        """List environment variable secrets"""
        secrets = []
        for key in os.environ:
            if key.startswith(self.prefix):
                secrets.append(key[len(self.prefix):].lower())
        return secrets

    async def delete_secret(self, key: str) -> bool:
        """Delete secret from environment"""
        env_key = f"{self.prefix}{key.upper()}"
        if env_key in os.environ:
            del os.environ[env_key]
            return True
        return False

    def get_info(self, key: str) -> Optional[SecretInfo]:
        """Get information about environment secret"""
        env_key = f"{self.prefix}{key.upper()}"
        exists = env_key in os.environ or key in os.environ

        return SecretInfo(
            key=key,
            source=SecretSource.ENVIRONMENT,
            exists=exists
        )


class AzureKeyVaultProvider(SecretProvider):
    """Provider for Azure Key Vault secrets"""

    def __init__(self, vault_url: str):
        if not KEYVAULT_AVAILABLE:
            raise ImportError("Azure Key Vault libraries not available")

        self.vault_url = vault_url
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=self.credential)

    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret from Azure Key Vault"""
        try:
            secret = self.client.get_secret(key)
            return secret.value
        except Exception as e:
            logger.warning(f"Failed to get secret {key} from Key Vault: {e}")
            return None

    async def set_secret(self, key: str, value: str) -> bool:
        """Set secret in Azure Key Vault"""
        try:
            self.client.set_secret(key, value)
            logger.info(f"Secret {key} stored in Key Vault")
            return True
        except Exception as e:
            logger.error(f"Failed to set secret {key} in Key Vault: {e}")
            return False

    async def list_secrets(self) -> List[str]:
        """List secrets in Azure Key Vault"""
        try:
            secrets = []
            for secret in self.client.list_properties_of_secrets():
                secrets.append(secret.name)
            return secrets
        except Exception as e:
            logger.error(f"Failed to list secrets from Key Vault: {e}")
            return []

    async def delete_secret(self, key: str) -> bool:
        """Delete secret from Azure Key Vault"""
        try:
            self.client.begin_delete_secret(key)
            logger.info(f"Secret {key} deleted from Key Vault")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {key} from Key Vault: {e}")
            return False

    def get_info(self, key: str) -> Optional[SecretInfo]:
        """Get information about Key Vault secret"""
        try:
            secret = self.client.get_secret(key)
            return SecretInfo(
                key=key,
                source=SecretSource.AZURE_KEYVAULT,
                exists=True,
                last_updated=secret.properties.updated_on.isoformat() if secret.properties.updated_on else None,
                expires=secret.properties.expires_on.isoformat() if secret.properties.expires_on else None
            )
        except Exception:
            return SecretInfo(
                key=key,
                source=SecretSource.AZURE_KEYVAULT,
                exists=False
            )


class LocalFileProvider(SecretProvider):
    """Provider for local file-based secrets (development only)"""

    def __init__(self, secrets_file: str = ".secrets.json"):
        self.secrets_file = secrets_file
        self._load_secrets()

    def _load_secrets(self) -> Dict[str, str]:
        """Load secrets from local file"""
        if os.path.exists(self.secrets_file):
            try:
                with open(self.secrets_file, 'r') as f:
                    self._secrets = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load secrets file: {e}")
                self._secrets = {}
        else:
            self._secrets = {}
        return self._secrets

    def _save_secrets(self) -> bool:
        """Save secrets to local file"""
        try:
            with open(self.secrets_file, 'w') as f:
                json.dump(self._secrets, f, indent=2)
            # Set restrictive permissions
            os.chmod(self.secrets_file, 0o600)
            return True
        except Exception as e:
            logger.error(f"Failed to save secrets file: {e}")
            return False

    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret from local file"""
        return self._secrets.get(key)

    async def set_secret(self, key: str, value: str) -> bool:
        """Set secret in local file"""
        self._secrets[key] = value
        return self._save_secrets()

    async def list_secrets(self) -> List[str]:
        """List secrets in local file"""
        return list(self._secrets.keys())

    async def delete_secret(self, key: str) -> bool:
        """Delete secret from local file"""
        if key in self._secrets:
            del self._secrets[key]
            return self._save_secrets()
        return False

    def get_info(self, key: str) -> Optional[SecretInfo]:
        """Get information about local file secret"""
        exists = key in self._secrets

        return SecretInfo(
            key=key,
            source=SecretSource.LOCAL_FILE,
            exists=exists
        )


class SecretsManager:
    """Manages secrets from multiple providers with fallback"""

    def __init__(self):
        self.providers: List[SecretProvider] = []
        self.cache: Dict[str, str] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, float] = {}

        # Initialize providers based on availability and configuration
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available secret providers"""
        # Always add environment provider
        self.providers.append(EnvironmentSecretProvider())

        # Add Azure Key Vault if configured and available
        vault_url = os.getenv('AZURE_KEYVAULT_URL')
        if vault_url and KEYVAULT_AVAILABLE:
            try:
                provider = AzureKeyVaultProvider(vault_url)
                self.providers.append(provider)
                logger.info("Azure Key Vault provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure Key Vault provider: {e}")

        # Add local file provider for development
        environment = os.getenv('ENVIRONMENT', 'development')
        if environment == 'development':
            self.providers.append(LocalFileProvider())
            logger.info("Local file provider initialized for development")

        logger.info(f"Initialized {len(self.providers)} secret providers")

    async def get_secret(self, key: str, use_cache: bool = True) -> Optional[str]:
        """
        Get secret from providers with fallback

        Args:
            key: Secret key
            use_cache: Whether to use cached value

        Returns:
            Secret value or None if not found
        """
        # Check cache first if enabled
        if use_cache and self._is_cached(key):
            return self.cache[key]

        # Try each provider in order
        for provider in self.providers:
            try:
                value = await provider.get_secret(key)
                if value:
                    # Cache the value
                    if use_cache:
                        self._cache_secret(key, value)
                    return value
            except Exception as e:
                logger.warning(f"Provider {type(provider).__name__} failed to get secret {key}: {e}")

        logger.warning(f"Secret {key} not found in any provider")
        return None

    async def set_secret(self, key: str, value: str, provider_type: SecretSource = None) -> bool:
        """
        Set secret in specified or default provider

        Args:
            key: Secret key
            value: Secret value
            provider_type: Specific provider to use

        Returns:
            True if successful
        """
        # Clear cache for this key
        self._clear_cache(key)

        # If provider type specified, use that provider
        if provider_type:
            for provider in self.providers:
                if self._get_provider_type(provider) == provider_type:
                    return await provider.set_secret(key, value)

        # Otherwise, use the first writable provider (skip environment for persistence)
        for provider in self.providers:
            if not isinstance(provider, EnvironmentSecretProvider):
                try:
                    success = await provider.set_secret(key, value)
                    if success:
                        return True
                except Exception as e:
                    logger.error(f"Failed to set secret in {type(provider).__name__}: {e}")

        return False

    async def list_secrets(self) -> Dict[SecretSource, List[str]]:
        """List secrets from all providers"""
        all_secrets = {}

        for provider in self.providers:
            try:
                provider_type = self._get_provider_type(provider)
                secrets = await provider.list_secrets()
                all_secrets[provider_type] = secrets
            except Exception as e:
                logger.error(f"Failed to list secrets from {type(provider).__name__}: {e}")

        return all_secrets

    async def get_secret_info(self, key: str) -> List[SecretInfo]:
        """Get information about a secret from all providers"""
        info_list = []

        for provider in self.providers:
            try:
                info = provider.get_info(key)
                if info:
                    info_list.append(info)
            except Exception as e:
                logger.error(f"Failed to get secret info from {type(provider).__name__}: {e}")

        return info_list

    def _get_provider_type(self, provider: SecretProvider) -> SecretSource:
        """Get the type of a provider"""
        if isinstance(provider, EnvironmentSecretProvider):
            return SecretSource.ENVIRONMENT
        elif isinstance(provider, AzureKeyVaultProvider):
            return SecretSource.AZURE_KEYVAULT
        elif isinstance(provider, LocalFileProvider):
            return SecretSource.LOCAL_FILE
        else:
            return SecretSource.ENVIRONMENT

    def _is_cached(self, key: str) -> bool:
        """Check if secret is cached and not expired"""
        if key not in self.cache:
            return False

        import time
        if key in self.cache_timestamps:
            age = time.time() - self.cache_timestamps[key]
            if age > self.cache_ttl:
                self._clear_cache(key)
                return False

        return True

    def _cache_secret(self, key: str, value: str):
        """Cache a secret value"""
        import time
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()

    def _clear_cache(self, key: str):
        """Clear cached secret"""
        self.cache.pop(key, None)
        self.cache_timestamps.pop(key, None)

    def clear_all_cache(self):
        """Clear all cached secrets"""
        self.cache.clear()
        self.cache_timestamps.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get status of secrets manager"""
        provider_status = []
        for provider in self.providers:
            provider_status.append({
                'type': self._get_provider_type(provider).value,
                'class': type(provider).__name__,
                'available': True
            })

        return {
            'providers_count': len(self.providers),
            'providers': provider_status,
            'cache_size': len(self.cache),
            'keyvault_available': KEYVAULT_AVAILABLE
        }


# Global instance
_secrets_manager = None


def get_secrets_manager() -> SecretsManager:
    """Get or create global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


async def get_secret(key: str) -> Optional[str]:
    """Convenience function to get a secret"""
    manager = get_secrets_manager()
    return await manager.get_secret(key)


async def set_secret(key: str, value: str, provider: SecretSource = None) -> bool:
    """Convenience function to set a secret"""
    manager = get_secrets_manager()
    return await manager.set_secret(key, value, provider)


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        manager = SecretsManager()

        # Get status
        status = manager.get_status()
        print(f"Secrets manager status: {status}")

        # Test getting a secret
        secret = await manager.get_secret("AZURE_OPENAI_KEY")
        if secret:
            print(f"Found secret (length: {len(secret)})")
        else:
            print("Secret not found")

        # List all secrets
        all_secrets = await manager.list_secrets()
        for provider, secrets in all_secrets.items():
            print(f"{provider.value}: {len(secrets)} secrets")

    asyncio.run(main())