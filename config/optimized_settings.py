"""
Optimized Configuration Management
High-performance configuration loading with caching and lazy evaluation
"""

import os
import threading
from typing import Optional, Dict, Any, List
from functools import lru_cache, cached_property
from pathlib import Path
import warnings

from pydantic import Field, validator, root_validator
from pydantic_settings import BaseSettings
from enum import Enum

# Import performance utilities
from core.performance import cache_result, time_sync


class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OptimizedSettings(BaseSettings):
    """
    High-performance settings with optimized loading and caching
    """

    model_config = {
        'env_file': ['.env.local', '.env'],
        'env_file_encoding': 'utf-8',
        'extra': 'ignore',
        'case_sensitive': False,
        'validate_assignment': True,
        'use_enum_values': True,
        'frozen': True  # Make immutable for thread safety
    }

    # Core Configuration (loaded immediately)
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        env="ENVIRONMENT",
        description="Application environment"
    )

    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )

    # Application Configuration
    app_name: str = Field(
        default="Azure AI IT Copilot",
        env="APP_NAME",
        min_length=1,
        max_length=100
    )

    app_version: str = Field(
        default="1.0.0",
        env="APP_VERSION",
        regex=r'^\d+\.\d+\.\d+$'
    )

    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT", ge=1, le=65535)

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )

    # CORS
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")

    # Azure Configuration (lazy loaded for performance)
    azure_subscription_id: str = Field(default="", env="AZURE_SUBSCRIPTION_ID")
    azure_tenant_id: str = Field(default="", env="AZURE_TENANT_ID")
    azure_client_id: str = Field(default="", env="AZURE_CLIENT_ID")
    azure_client_secret: str = Field(default="", env="AZURE_CLIENT_SECRET")
    azure_resource_group: str = Field(default="rg-ai-copilot", env="AZURE_RESOURCE_GROUP")
    azure_location: str = Field(default="eastus", env="AZURE_LOCATION")

    # Azure OpenAI (lazy loaded)
    azure_openai_endpoint: str = Field(default="", env="AZURE_OPENAI_ENDPOINT")
    azure_openai_key: str = Field(default="", env="AZURE_OPENAI_KEY")
    azure_openai_model: str = Field(default="gpt-4", env="AZURE_OPENAI_MODEL")
    azure_openai_temperature: float = Field(default=0.1, env="AZURE_OPENAI_TEMPERATURE")

    # Database Configuration (lazy loaded)
    cosmos_db_endpoint: str = Field(default="", env="COSMOS_DB_ENDPOINT")
    cosmos_db_key: str = Field(default="", env="COSMOS_DB_KEY")
    cosmos_db_name: str = Field(default="aiitcopilot", env="COSMOS_DB_NAME")

    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_ssl: bool = Field(default=False, env="REDIS_SSL")

    # Security
    jwt_secret_key: str = Field(default="change-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")

    # Performance Settings
    worker_processes: int = Field(default=1, env="WORKER_PROCESSES")
    max_connections: int = Field(default=1000, env="MAX_CONNECTIONS")
    request_timeout: int = Field(default=60, env="REQUEST_TIMEOUT")

    # Feature Flags
    enable_auto_remediation: bool = Field(default=True, env="ENABLE_AUTO_REMEDIATION")
    enable_predictive_analytics: bool = Field(default=True, env="ENABLE_PREDICTIVE_ANALYTICS")
    enable_cost_optimization: bool = Field(default=True, env="ENABLE_COST_OPTIMIZATION")
    enable_compliance_checks: bool = Field(default=True, env="ENABLE_COMPLIANCE_CHECKS")

    # Validation
    @validator('cors_origins')
    def validate_cors_origins(cls, v):
        if isinstance(v, str):
            if v.strip() == "*":
                return "*"
            origins = [origin.strip() for origin in v.split(',') if origin.strip()]
            return ','.join(origins)
        return v

    @validator('azure_openai_temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return round(v, 2)

    @validator('jwt_expiration_hours')
    def validate_jwt_expiration(cls, v):
        if not 1 <= v <= 168:
            raise ValueError("JWT expiration must be between 1 and 168 hours")
        return v

    @root_validator
    def validate_production_requirements(cls, values):
        env = values.get('environment')
        if env == Environment.PRODUCTION:
            # Warn about production settings
            if values.get('jwt_secret_key') == 'change-in-production':
                warnings.warn("Using default JWT secret in production!", UserWarning)

        return values

    # Cached Properties for Performance
    @cached_property
    @cache_result(ttl_seconds=300)
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @cached_property
    @cache_result(ttl_seconds=300)
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @cached_property
    @cache_result(ttl_seconds=300)
    def is_testing(self) -> bool:
        return self.environment == Environment.TESTING

    @cached_property
    @cache_result(ttl_seconds=300)
    def is_staging(self) -> bool:
        return self.environment == Environment.STAGING

    @cached_property
    @cache_result(ttl_seconds=60)
    def get_cors_origins_list(self) -> List[str]:
        """Get CORS origins as optimized list with caching"""
        if self.cors_origins == "*":
            return ["*"]
        origins = [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]
        return origins if origins else ["*"]

    @cached_property
    @cache_result(ttl_seconds=60)
    def get_redis_url(self) -> str:
        """Get optimized Redis connection URL with caching"""
        scheme = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{scheme}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @cached_property
    @cache_result(ttl_seconds=300)
    def get_cors_config(self) -> Dict[str, Any]:
        """Get optimized CORS configuration with caching"""
        return {
            "allow_origins": self.get_cors_origins_list,
            "allow_credentials": self.cors_allow_credentials,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Authorization", "Content-Type", "X-Request-ID"],
            "expose_headers": ["X-Request-ID", "X-Response-Time"]
        }

    @cached_property
    @cache_result(ttl_seconds=300)
    def get_azure_credentials(self) -> Dict[str, str]:
        """Get Azure credentials with lazy loading and caching"""
        return {
            "subscription_id": self.azure_subscription_id,
            "tenant_id": self.azure_tenant_id,
            "client_id": self.azure_client_id,
            "client_secret": self.azure_client_secret,
        }

    @cached_property
    @cache_result(ttl_seconds=300)
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration with lazy loading and caching"""
        return {
            "azure_endpoint": self.azure_openai_endpoint,
            "api_key": self.azure_openai_key,
            "api_version": "2024-02-01",
            "model": self.azure_openai_model,
            "temperature": self.azure_openai_temperature,
            "max_tokens": 4000,
            "timeout": 30
        }

    @cache_result(ttl_seconds=300)
    def is_azure_configured(self) -> bool:
        """Check if Azure credentials are properly configured"""
        creds = self.get_azure_credentials
        return all([
            creds["subscription_id"] and creds["subscription_id"] != "development-subscription",
            creds["tenant_id"] and creds["tenant_id"] != "development-tenant",
            creds["client_id"] and creds["client_id"] != "development-client",
            creds["client_secret"] and creds["client_secret"] != "development-secret"
        ])

    @cache_result(ttl_seconds=300)
    def is_openai_configured(self) -> bool:
        """Check if OpenAI configuration is properly set up"""
        config = self.get_openai_config
        return all([
            config["azure_endpoint"] and config["azure_endpoint"] != "https://development.openai.azure.com",
            config["api_key"] and config["api_key"] != "development-openai-key"
        ])


class SettingsManager:
    """
    Thread-safe settings manager with optimized loading
    """

    def __init__(self):
        self._settings: Optional[OptimizedSettings] = None
        self._lock = threading.RLock()
        self._config_file_mtime: Optional[float] = None

    @time_sync("settings_load")
    def get_settings(self, force_reload: bool = False) -> OptimizedSettings:
        """Get settings with optimized loading and caching"""
        with self._lock:
            # Check if we need to reload based on file modification time
            if self._should_reload() or force_reload:
                self._settings = OptimizedSettings()
                self._update_config_file_mtime()

            return self._settings

    def _should_reload(self) -> bool:
        """Check if settings should be reloaded based on file changes"""
        if self._settings is None:
            return True

        # Check .env file modification time
        env_files = ['.env.local', '.env']
        current_mtime = 0

        for env_file in env_files:
            if os.path.exists(env_file):
                current_mtime = max(current_mtime, os.path.getmtime(env_file))

        if self._config_file_mtime is None:
            return True

        return current_mtime > self._config_file_mtime

    def _update_config_file_mtime(self):
        """Update the configuration file modification time"""
        env_files = ['.env.local', '.env']
        current_mtime = 0

        for env_file in env_files:
            if os.path.exists(env_file):
                current_mtime = max(current_mtime, os.path.getmtime(env_file))

        self._config_file_mtime = current_mtime

    def reload_settings(self) -> OptimizedSettings:
        """Force reload settings"""
        return self.get_settings(force_reload=True)

    @cache_result(ttl_seconds=60)
    def get_environment_info(self) -> Dict[str, Any]:
        """Get cached environment information for health checks"""
        settings = self.get_settings()
        return {
            "environment": settings.environment.value,
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "debug": settings.debug,
            "api_host": settings.api_host,
            "api_port": settings.api_port,
        }

    @cache_result(ttl_seconds=300)
    def get_security_config(self) -> Dict[str, Any]:
        """Get security-related configuration summary"""
        settings = self.get_settings()
        return {
            "jwt_algorithm": settings.jwt_algorithm,
            "jwt_expiration_hours": settings.jwt_expiration_hours,
            "cors_configured": len(settings.get_cors_origins_list) > 0,
            "azure_configured": settings.is_azure_configured(),
            "openai_configured": settings.is_openai_configured(),
            "production_ready": settings.is_production and settings.is_azure_configured()
        }


# Global settings manager instance
_settings_manager = SettingsManager()


def get_settings() -> OptimizedSettings:
    """Get optimized settings instance"""
    return _settings_manager.get_settings()


def reload_settings() -> OptimizedSettings:
    """Force reload settings"""
    return _settings_manager.reload_settings()


def get_environment_info() -> Dict[str, Any]:
    """Get environment information"""
    return _settings_manager.get_environment_info()


def get_security_config() -> Dict[str, Any]:
    """Get security configuration"""
    return _settings_manager.get_security_config()