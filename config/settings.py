"""
Application settings and configuration management
Uses pydantic for validation and type safety
"""

import os
from typing import Optional, Dict, Any, List
from functools import lru_cache
from pydantic import BaseSettings, Field, validator
from enum import Enum


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
    """Main application settings"""

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Application
    app_name: str = Field(default="Azure AI IT Copilot", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")

    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")

    # Azure Configuration
    azure_subscription_id: str = Field(..., env="AZURE_SUBSCRIPTION_ID")
    azure_tenant_id: str = Field(..., env="AZURE_TENANT_ID")
    azure_client_id: str = Field(..., env="AZURE_CLIENT_ID")
    azure_client_secret: str = Field(..., env="AZURE_CLIENT_SECRET")
    azure_resource_group: str = Field(default="rg-ai-copilot", env="AZURE_RESOURCE_GROUP")
    azure_location: str = Field(default="eastus", env="AZURE_LOCATION")

    # Azure OpenAI
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_key: str = Field(..., env="AZURE_OPENAI_KEY")
    azure_openai_model: str = Field(default="gpt-4", env="AZURE_OPENAI_MODEL")
    azure_openai_temperature: float = Field(default=0.1, env="AZURE_OPENAI_TEMPERATURE")

    # Database
    cosmos_db_endpoint: str = Field(..., env="COSMOS_DB_ENDPOINT")
    cosmos_db_key: str = Field(..., env="COSMOS_DB_KEY")
    cosmos_db_name: str = Field(default="aiitcopilot", env="COSMOS_DB_NAME")

    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_ssl: bool = Field(default=False, env="REDIS_SSL")

    # Security
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")

    # Monitoring
    application_insights_key: Optional[str] = Field(default=None, env="APPLICATION_INSIGHTS_KEY")
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")

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

    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment value"""
        if v not in Environment:
            raise ValueError(f"Invalid environment: {v}")
        return v

    @validator('api_port')
    def validate_port(cls, v):
        """Validate API port"""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @validator('azure_openai_temperature')
    def validate_temperature(cls, v):
        """Validate OpenAI temperature"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @validator('jwt_expiration_hours')
    def validate_jwt_expiration(cls, v):
        """Validate JWT expiration"""
        if v <= 0:
            raise ValueError("JWT expiration must be positive")
        return v

    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION

    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING

    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        scheme = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{scheme}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": self.cors_origins,
            "allow_credentials": self.cors_allow_credentials,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

    def get_azure_credentials(self) -> Dict[str, str]:
        """Get Azure credentials as dictionary"""
        return {
            "subscription_id": self.azure_subscription_id,
            "tenant_id": self.azure_tenant_id,
            "client_id": self.azure_client_id,
            "client_secret": self.azure_client_secret,
        }

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration"""
        return {
            "azure_endpoint": self.azure_openai_endpoint,
            "api_key": self.azure_openai_key,
            "api_version": "2024-02-01",
            "model": self.azure_openai_model,
            "temperature": self.azure_openai_temperature,
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def get_environment_info() -> Dict[str, Any]:
    """Get environment information for health checks"""
    settings = get_settings()
    return {
        "environment": settings.environment,
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "debug": settings.debug,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
    }