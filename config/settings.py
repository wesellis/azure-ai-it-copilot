"""
Optimized Application settings and configuration management
Uses pydantic for validation, type safety, and performance optimization
"""

import os
from typing import Optional, Dict, Any, List, Union
from functools import lru_cache, cached_property
from pydantic import Field, validator, root_validator
from pydantic_settings import BaseSettings
from enum import Enum
from pathlib import Path
import warnings


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


class Settings(BaseSettings):
    """Optimized main application settings with validation and caching"""

    model_config = {
        'env_file': ['.env.local', '.env'],  # Support multiple env files
        'env_file_encoding': 'utf-8',
        'extra': 'ignore',  # Ignore extra fields in .env
        'case_sensitive': False,  # Allow case-insensitive env vars
        'validate_assignment': True,  # Validate on assignment
        'use_enum_values': True  # Use enum values instead of names
    }

    # Environment Configuration
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
        max_length=100,
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        env="APP_VERSION",
        pattern=r'^\d+\.\d+\.\d+$',
        description="Application version (semver)"
    )
    api_host: str = Field(
        default="0.0.0.0",
        env="API_HOST",
        description="API server host"
    )
    api_port: int = Field(
        default=8000,
        env="API_PORT",
        ge=1,
        le=65535,
        description="API server port"
    )

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")

    # CORS
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")

    @cached_property
    def get_cors_origins_list(self) -> List[str]:
        """Get CORS origins as optimized list with caching"""
        if self.cors_origins == "*":
            return ["*"]
        # Filter out empty origins and normalize
        origins = [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]
        return origins if origins else ["*"]  # Fallback to allow all if no valid origins

    # Azure Configuration
    azure_subscription_id: str = Field(default="development-subscription", env="AZURE_SUBSCRIPTION_ID")
    azure_tenant_id: str = Field(default="development-tenant", env="AZURE_TENANT_ID")
    azure_client_id: str = Field(default="development-client", env="AZURE_CLIENT_ID")
    azure_client_secret: str = Field(default="development-secret", env="AZURE_CLIENT_SECRET")
    azure_resource_group: str = Field(default="rg-ai-copilot", env="AZURE_RESOURCE_GROUP")
    azure_location: str = Field(default="eastus", env="AZURE_LOCATION")

    # Azure OpenAI
    azure_openai_endpoint: str = Field(default="https://development.openai.azure.com", env="AZURE_OPENAI_ENDPOINT")
    azure_openai_key: str = Field(default="development-openai-key", env="AZURE_OPENAI_KEY")
    azure_openai_model: str = Field(default="gpt-4", env="AZURE_OPENAI_MODEL")
    azure_openai_temperature: float = Field(default=0.1, env="AZURE_OPENAI_TEMPERATURE")

    # Database
    cosmos_db_endpoint: str = Field(default="https://development.documents.azure.com", env="COSMOS_DB_ENDPOINT")
    cosmos_db_key: str = Field(default="development-cosmos-key", env="COSMOS_DB_KEY")
    cosmos_db_name: str = Field(default="aiitcopilot", env="COSMOS_DB_NAME")

    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_ssl: bool = Field(default=False, env="REDIS_SSL")

    # Security
    jwt_secret_key: str = Field(default="development-jwt-secret-change-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")

    # Authentication
    admin_username: str = Field(default="admin", env="ADMIN_USERNAME")
    admin_password: Optional[str] = Field(default=None, env="ADMIN_PASSWORD")
    admin_password_hash: Optional[str] = Field(default=None, env="ADMIN_PASSWORD_HASH")

    # Feature Flags
    enable_auto_remediation: bool = Field(default=True, env="ENABLE_AUTO_REMEDIATION")
    enable_predictive_analytics: bool = Field(default=True, env="ENABLE_PREDICTIVE_ANALYTICS")
    enable_cost_optimization: bool = Field(default=True, env="ENABLE_COST_OPTIMIZATION")
    enable_compliance_checks: bool = Field(default=True, env="ENABLE_COMPLIANCE_CHECKS")

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")

    # Monitoring
    application_insights_key: Optional[str] = Field(default=None, env="APPLICATION_INSIGHTS_KEY")
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3001, env="GRAFANA_PORT")
    grafana_user: str = Field(default="admin", env="GRAFANA_USER")
    grafana_password: str = Field(default="admin", env="GRAFANA_PASSWORD")

    # Deployment Configuration
    resource_group: str = Field(default="rg-ai-copilot", env="RESOURCE_GROUP")
    location: str = Field(default="eastus", env="LOCATION")
    acr_name: str = Field(default="aicopilotacr", env="ACR_NAME")
    aks_cluster_name: str = Field(default="aks-ai-copilot", env="AKS_CLUSTER_NAME")
    redis_name: str = Field(default="redis-ai-copilot", env="REDIS_NAME")
    app_insights_name: str = Field(default="ai-copilot-insights", env="APP_INSIGHTS_NAME")
    log_analytics_workspace: str = Field(default="ai-copilot-logs", env="LOG_ANALYTICS_WORKSPACE")

    # External Integrations
    teams_webhook_url: Optional[str] = Field(default=None, env="TEAMS_WEBHOOK_URL")
    slack_bot_token: Optional[str] = Field(default=None, env="SLACK_BOT_TOKEN")
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")

    # Microsoft Graph
    graph_client_id: Optional[str] = Field(default=None, env="GRAPH_CLIENT_ID")
    graph_client_secret: Optional[str] = Field(default=None, env="GRAPH_CLIENT_SECRET")
    graph_tenant_id: Optional[str] = Field(default=None, env="GRAPH_TENANT_ID")

    # Agent Configuration
    max_agents: int = Field(default=10, env="MAX_AGENTS")
    agent_timeout: int = Field(default=300, env="AGENT_TIMEOUT")
    auto_approve_safe_operations: bool = Field(default=True, env="AUTO_APPROVE_SAFE_OPERATIONS")

    # Performance
    worker_processes: int = Field(default=1, env="WORKER_PROCESSES")
    max_connections: int = Field(default=1000, env="MAX_CONNECTIONS")
    request_timeout: int = Field(default=60, env="REQUEST_TIMEOUT")

    # Feature Flags
    enable_websockets: bool = Field(default=True, env="ENABLE_WEBSOCKETS")
    enable_background_tasks: bool = Field(default=True, env="ENABLE_BACKGROUND_TASKS")
    enable_cost_optimization: bool = Field(default=True, env="ENABLE_COST_OPTIMIZATION")
    enable_predictive_analytics: bool = Field(default=True, env="ENABLE_PREDICTIVE_ANALYTICS")

    @validator('cors_origins')
    def validate_cors_origins(cls, v):
        """Validate and normalize CORS origins"""
        if isinstance(v, str):
            # Clean up origins - remove empty strings and whitespace
            if v.strip() == "*":
                return "*"
            origins = [origin.strip() for origin in v.split(',') if origin.strip()]
            return ','.join(origins)
        return v

    @validator('api_host')
    def validate_api_host(cls, v):
        """Validate API host format"""
        if v in ['localhost', '127.0.0.1', '0.0.0.0']:
            return v
        # Basic validation for IP or hostname
        if not v.replace('.', '').replace('-', '').replace('_', '').isalnum():
            raise ValueError(f"Invalid API host format: {v}")
        return v

    @validator('azure_openai_temperature')
    def validate_temperature(cls, v):
        """Validate OpenAI temperature range"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return round(v, 2)  # Round to 2 decimal places

    @validator('jwt_expiration_hours')
    def validate_jwt_expiration(cls, v):
        """Validate JWT expiration with reasonable limits"""
        if not 1 <= v <= 168:  # 1 hour to 1 week max
            raise ValueError("JWT expiration must be between 1 and 168 hours")
        return v

    @root_validator
    def validate_production_requirements(cls, values):
        """Validate production-specific requirements"""
        env = values.get('environment')
        if env == Environment.PRODUCTION:
            # Check critical production settings
            jwt_secret = values.get('jwt_secret_key', '')
            if jwt_secret == 'development-jwt-secret-change-in-production':
                warnings.warn(
                    "Using default JWT secret in production! Please set JWT_SECRET_KEY",
                    UserWarning
                )

            azure_tenant = values.get('azure_tenant_id', '')
            if azure_tenant == 'development-tenant':
                warnings.warn(
                    "Using default Azure tenant in production! Please set AZURE_TENANT_ID",
                    UserWarning
                )

        return values

    @cached_property
    def is_development(self) -> bool:
        """Check if running in development mode (cached)"""
        return self.environment == Environment.DEVELOPMENT

    @cached_property
    def is_production(self) -> bool:
        """Check if running in production mode (cached)"""
        return self.environment == Environment.PRODUCTION

    @cached_property
    def is_testing(self) -> bool:
        """Check if running in testing mode (cached)"""
        return self.environment == Environment.TESTING

    @cached_property
    def is_staging(self) -> bool:
        """Check if running in staging mode (cached)"""
        return self.environment == Environment.STAGING

    @cached_property
    def get_redis_url(self) -> str:
        """Get optimized Redis connection URL with caching"""
        scheme = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{scheme}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @cached_property
    def get_cors_config(self) -> Dict[str, Any]:
        """Get optimized CORS configuration with caching"""
        return {
            "allow_origins": self.get_cors_origins_list(),
            "allow_credentials": self.cors_allow_credentials,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Authorization", "Content-Type", "X-Request-ID"],
            "expose_headers": ["X-Request-ID", "X-Response-Time"]
        }

    @cached_property
    def get_azure_credentials(self) -> Dict[str, str]:
        """Get Azure credentials as optimized dictionary with caching"""
        return {
            "subscription_id": self.azure_subscription_id,
            "tenant_id": self.azure_tenant_id,
            "client_id": self.azure_client_id,
            "client_secret": self.azure_client_secret,
        }

    def is_azure_configured(self) -> bool:
        """Check if Azure credentials are properly configured"""
        creds = self.get_azure_credentials
        return all([
            creds["subscription_id"] != "development-subscription",
            creds["tenant_id"] != "development-tenant",
            creds["client_id"] != "development-client",
            creds["client_secret"] != "development-secret"
        ])

    @cached_property
    def get_openai_config(self) -> Dict[str, Any]:
        """Get optimized OpenAI configuration with caching"""
        return {
            "azure_endpoint": self.azure_openai_endpoint,
            "api_key": self.azure_openai_key,
            "api_version": "2024-02-01",
            "model": self.azure_openai_model,
            "temperature": self.azure_openai_temperature,
            "max_tokens": 4000,  # Reasonable default
            "timeout": 30  # Request timeout
        }

    def is_openai_configured(self) -> bool:
        """Check if OpenAI configuration is properly set up"""
        config = self.get_openai_config
        return all([
            config["azure_endpoint"] != "https://development.openai.azure.com",
            config["api_key"] != "development-openai-key",
            config["model"] != "gpt-4"  # Ensure specific model is configured
        ])



# Global settings instance with proper caching
_settings_instance: Optional[Settings] = None

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance with optimized loading"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

def reload_settings() -> Settings:
    """Force reload settings (useful for configuration updates)"""
    global _settings_instance
    _settings_instance = None
    get_settings.cache_clear()
    return get_settings()

@lru_cache(maxsize=1)
def get_environment_info() -> Dict[str, Any]:
    """Get cached environment information for health checks"""
    settings = get_settings()
    return {
        "environment": settings.environment.value,
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "debug": settings.debug,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "uptime_start": settings.model_config.get('created_at', 'unknown')
    }

def get_security_config() -> Dict[str, Any]:
    """Get security-related configuration summary"""
    settings = get_settings()
    return {
        "jwt_algorithm": settings.jwt_algorithm,
        "jwt_expiration_hours": settings.jwt_expiration_hours,
        "cors_configured": len(settings.get_cors_origins_list) > 0,
        "azure_configured": settings.is_azure_configured(),
        "openai_configured": settings.is_openai_configured(),
        "production_ready": settings.is_production and settings.is_azure_configured()
    }