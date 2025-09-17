"""
Core interfaces and abstractions for Azure AI IT Copilot
Defines contracts for all major components to ensure modularity
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from datetime import datetime


@runtime_checkable
class IAgentOrchestrator(Protocol):
    """Interface for AI agent orchestrators"""

    async def process_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a natural language command"""
        ...

    async def create_plan(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create an execution plan for a command"""
        ...

    async def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a previously created plan"""
        ...


@runtime_checkable
class IAgent(Protocol):
    """Interface for AI agents"""

    agent_type: str
    capabilities: List[str]

    async def create_plan(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create execution plan from command"""
        ...

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan"""
        ...

    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if agent can handle the command"""
        ...


@runtime_checkable
class IAzureClientFactory(Protocol):
    """Interface for Azure client factory"""

    def get_resource_client(self) -> Any:
        """Get Azure Resource Management client"""
        ...

    def get_compute_client(self) -> Any:
        """Get Azure Compute Management client"""
        ...

    def get_network_client(self) -> Any:
        """Get Azure Network Management client"""
        ...

    def get_storage_client(self) -> Any:
        """Get Azure Storage Management client"""
        ...

    def get_monitor_client(self) -> Any:
        """Get Azure Monitor client"""
        ...


@runtime_checkable
class IConfigurationProvider(Protocol):
    """Interface for configuration providers"""

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting"""
        ...

    def get_azure_credentials(self) -> Dict[str, str]:
        """Get Azure credentials"""
        ...

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration"""
        ...

    def is_development(self) -> bool:
        """Check if running in development mode"""
        ...


@runtime_checkable
class INotificationService(Protocol):
    """Interface for notification services"""

    async def send_alert(self, alert_type: str, severity: str, message: str,
                        recipients: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send alert notification"""
        ...

    async def send_report(self, report_type: str, content: Dict[str, Any],
                         recipients: List[str]) -> Dict[str, Any]:
        """Send report notification"""
        ...


@runtime_checkable
class ITaskQueue(Protocol):
    """Interface for task queue systems"""

    async def enqueue_task(self, task_name: str, args: tuple = (), kwargs: Dict[str, Any] = None) -> str:
        """Enqueue a background task"""
        ...

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task"""
        ...

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        ...


@runtime_checkable
class IRepository(Protocol):
    """Interface for data repositories"""

    async def create(self, entity: Dict[str, Any]) -> str:
        """Create a new entity"""
        ...

    async def get_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID"""
        ...

    async def update(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity"""
        ...

    async def delete(self, entity_id: str) -> bool:
        """Delete entity"""
        ...

    async def find(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Find entities by filters"""
        ...


@runtime_checkable
class ISecretsManager(Protocol):
    """Interface for secrets management"""

    async def get_secret(self, key: str) -> Optional[str]:
        """Get secret value"""
        ...

    async def set_secret(self, key: str, value: str) -> bool:
        """Set secret value"""
        ...

    async def delete_secret(self, key: str) -> bool:
        """Delete secret"""
        ...


@runtime_checkable
class IHealthChecker(Protocol):
    """Interface for health checking"""

    async def check_health(self) -> Dict[str, Any]:
        """Perform health check"""
        ...

    async def check_dependencies(self) -> Dict[str, Any]:
        """Check health of dependencies"""
        ...


@runtime_checkable
class IMetricsCollector(Protocol):
    """Interface for metrics collection"""

    def increment_counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        ...

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value"""
        ...

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value"""
        ...


@runtime_checkable
class ICacheProvider(Protocol):
    """Interface for cache providers"""

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        ...

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        ...

    async def clear(self) -> bool:
        """Clear all cache"""
        ...


@runtime_checkable
class IEventBus(Protocol):
    """Interface for event bus systems"""

    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        ...

    async def subscribe(self, event_type: str, handler: callable) -> str:
        """Subscribe to events"""
        ...

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        ...