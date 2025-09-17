"""
Dependency Injection Container for Azure AI IT Copilot
Manages service registration and resolution for improved modularity
"""

from typing import Dict, Any, TypeVar, Type, Callable, Optional
import inspect
from functools import wraps
import logging

from .interfaces import (
    IConfigurationProvider, IAzureClientFactory, INotificationService,
    ITaskQueue, IRepository, ISecretsManager, IHealthChecker,
    IMetricsCollector, ICacheProvider, IEventBus
)
from .exceptions import ConfigurationError

T = TypeVar('T')

logger = logging.getLogger(__name__)


class ServiceLifetime:
    """Service lifetime options"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceDescriptor:
    """Describes a service registration"""

    def __init__(self, service_type: Type, implementation: Type = None,
                 factory: Callable = None, instance: Any = None,
                 lifetime: str = ServiceLifetime.TRANSIENT):
        self.service_type = service_type
        self.implementation = implementation
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime

        if not any([implementation, factory, instance]):
            raise ConfigurationError("Service descriptor must have implementation, factory, or instance")


class DependencyContainer:
    """Dependency injection container"""

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._building = set()

    def register_singleton(self, service_type: Type[T], implementation: Type[T] = None,
                          factory: Callable[[], T] = None, instance: T = None) -> 'DependencyContainer':
        """Register a singleton service"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            factory=factory,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
        self._services[service_type] = descriptor
        logger.debug(f"Registered singleton service: {service_type.__name__}")
        return self

    def register_transient(self, service_type: Type[T], implementation: Type[T] = None,
                          factory: Callable[[], T] = None) -> 'DependencyContainer':
        """Register a transient service"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            factory=factory,
            lifetime=ServiceLifetime.TRANSIENT
        )
        self._services[service_type] = descriptor
        logger.debug(f"Registered transient service: {service_type.__name__}")
        return self

    def register_scoped(self, service_type: Type[T], implementation: Type[T] = None,
                       factory: Callable[[], T] = None) -> 'DependencyContainer':
        """Register a scoped service"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            factory=factory,
            lifetime=ServiceLifetime.SCOPED
        )
        self._services[service_type] = descriptor
        logger.debug(f"Registered scoped service: {service_type.__name__}")
        return self

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service instance"""
        if service_type in self._building:
            raise ConfigurationError(f"Circular dependency detected for {service_type.__name__}")

        descriptor = self._services.get(service_type)
        if not descriptor:
            raise ConfigurationError(f"Service {service_type.__name__} is not registered")

        # Check if singleton instance already exists
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._instances:
                return self._instances[service_type]

        try:
            self._building.add(service_type)
            instance = self._create_instance(descriptor)

            # Cache singleton instances
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                self._instances[service_type] = instance

            return instance
        finally:
            self._building.discard(service_type)

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create an instance from service descriptor"""
        # Use existing instance if provided
        if descriptor.instance is not None:
            return descriptor.instance

        # Use factory if provided
        if descriptor.factory is not None:
            return descriptor._inject_dependencies(descriptor.factory)

        # Use implementation type
        if descriptor.implementation is not None:
            return self._create_with_dependencies(descriptor.implementation)

        raise ConfigurationError(f"Cannot create instance for {descriptor.service_type.__name__}")

    def _create_with_dependencies(self, implementation_type: Type) -> Any:
        """Create instance with dependency injection"""
        constructor = implementation_type.__init__
        signature = inspect.signature(constructor)

        # Get constructor parameters (excluding 'self')
        parameters = [param for name, param in signature.parameters.items() if name != 'self']

        # Resolve dependencies
        dependencies = []
        for param in parameters:
            if param.annotation != inspect.Parameter.empty:
                dependency = self.resolve(param.annotation)
                dependencies.append(dependency)
            else:
                logger.warning(f"Parameter {param.name} in {implementation_type.__name__} has no type annotation")

        return implementation_type(*dependencies)

    def _inject_dependencies(self, factory: Callable) -> Any:
        """Inject dependencies into factory function"""
        signature = inspect.signature(factory)
        parameters = list(signature.parameters.values())

        # Resolve dependencies
        dependencies = []
        for param in parameters:
            if param.annotation != inspect.Parameter.empty:
                dependency = self.resolve(param.annotation)
                dependencies.append(dependency)

        return factory(*dependencies)

    def is_registered(self, service_type: Type) -> bool:
        """Check if service type is registered"""
        return service_type in self._services

    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """Get all registered services"""
        return self._services.copy()

    def clear(self) -> None:
        """Clear all registrations and instances"""
        self._services.clear()
        self._instances.clear()
        self._building.clear()


# Global container instance
_container = DependencyContainer()


def get_container() -> DependencyContainer:
    """Get the global dependency container"""
    return _container


def inject(service_type: Type[T]) -> T:
    """Decorator for method parameter injection"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Inject service if not provided in kwargs
            param_name = None
            signature = inspect.signature(func)
            for name, param in signature.parameters.items():
                if param.annotation == service_type:
                    param_name = name
                    break

            if param_name and param_name not in kwargs:
                kwargs[param_name] = _container.resolve(service_type)

            return func(*args, **kwargs)
        return wrapper
    return decorator


def configure_services() -> DependencyContainer:
    """Configure default services"""
    container = get_container()

    # Import here to avoid circular imports
    from config.settings import get_settings
    from services.azure_client_factory import AzureClientFactory
    from services.notification_service import NotificationService
    from services.task_queue_service import TaskQueueService
    from services.secrets_service import SecretsService
    from services.health_service import HealthService
    from services.metrics_service import MetricsService
    from services.cache_service import CacheService
    from services.event_bus_service import EventBusService

    # Configuration
    container.register_singleton(
        IConfigurationProvider,
        factory=lambda: get_settings()
    )

    # Azure clients
    container.register_singleton(
        IAzureClientFactory,
        AzureClientFactory
    )

    # Core services
    container.register_singleton(INotificationService, NotificationService)
    container.register_singleton(ITaskQueue, TaskQueueService)
    container.register_singleton(ISecretsManager, SecretsService)
    container.register_singleton(IHealthChecker, HealthService)
    container.register_singleton(IMetricsCollector, MetricsService)
    container.register_singleton(ICacheProvider, CacheService)
    container.register_singleton(IEventBus, EventBusService)

    logger.info("Services configured successfully")
    return container


def create_scope() -> 'ServiceScope':
    """Create a new service scope"""
    return ServiceScope(_container)


class ServiceScope:
    """Represents a service scope for scoped dependencies"""

    def __init__(self, container: DependencyContainer):
        self._container = container
        self._scoped_instances: Dict[Type, Any] = {}

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve service within this scope"""
        descriptor = self._container._services.get(service_type)
        if not descriptor:
            raise ConfigurationError(f"Service {service_type.__name__} is not registered")

        # For scoped services, use scoped instance if exists
        if descriptor.lifetime == ServiceLifetime.SCOPED:
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]

            instance = self._container._create_instance(descriptor)
            self._scoped_instances[service_type] = instance
            return instance

        # For singleton and transient, delegate to container
        return self._container.resolve(service_type)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()

    def dispose(self):
        """Dispose scoped instances"""
        for instance in self._scoped_instances.values():
            if hasattr(instance, 'dispose'):
                try:
                    instance.dispose()
                except Exception as e:
                    logger.error(f"Error disposing scoped instance: {e}")

        self._scoped_instances.clear()