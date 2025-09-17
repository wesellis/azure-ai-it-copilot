"""
Core module for Azure AI IT Copilot
Contains base interfaces, abstractions, and shared utilities
"""

# Explicit imports for better performance and clarity
from .interfaces import (
    IAgentOrchestrator,
    IAgent,
    IAzureClientFactory,
    IConfigurationProvider,
    INotificationService,
    ITaskQueue,
    IRepository,
    ISecretsManager,
    IHealthChecker,
    IMetricsCollector,
    ICacheProvider,
    IEventBus
)

from .base_classes import (
    BaseAgent,
    BaseService,
    BaseRepository,
    BaseClient,
    BaseEventHandler
)

from .exceptions import (
    CopilotException,
    AgentExecutionError,
    ConfigurationError,
    AzureClientError,
    AuthenticationError,
    ValidationError,
    NotificationError,
    TaskExecutionError,
    ResourceNotFoundError,
    QuotaExceededError,
    CacheError,
    IntegrationError,
    SecurityError,
    RateLimitError
)

__all__ = [
    # Interfaces
    "IAgentOrchestrator",
    "IAgent",
    "IAzureClientFactory",
    "IConfigurationProvider",
    "INotificationService",
    "ITaskQueue",
    "IRepository",
    "ISecretsManager",
    "IHealthChecker",
    "IMetricsCollector",
    "ICacheProvider",
    "IEventBus",

    # Base Classes
    "BaseAgent",
    "BaseService",
    "BaseRepository",
    "BaseClient",
    "BaseEventHandler",

    # Exceptions
    "CopilotException",
    "AgentExecutionError",
    "ConfigurationError",
    "AzureClientError",
    "AuthenticationError",
    "ValidationError",
    "NotificationError",
    "TaskExecutionError",
    "ResourceNotFoundError",
    "QuotaExceededError",
    "CacheError",
    "IntegrationError",
    "SecurityError",
    "RateLimitError"
]