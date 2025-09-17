"""
Application Factory for Azure AI IT Copilot
Provides modular application setup with dependency injection
"""

import logging
from typing import Optional
from pathlib import Path

from .dependency_injection import DependencyContainer, configure_services
from .plugin_manager import get_plugin_manager, register_agent_plugin, register_service_plugin
from .interfaces import IConfigurationProvider, IAzureClientFactory, INotificationService
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class Application:
    """Main application class with modular architecture"""

    def __init__(self):
        self.container: Optional[DependencyContainer] = None
        self.plugin_manager = get_plugin_manager()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the application"""
        try:
            logger.info("🚀 Initializing Azure AI IT Copilot Application")

            # Setup dependency injection
            await self._setup_dependency_injection()

            # Register built-in plugins
            await self._register_builtin_plugins()

            # Discover external plugins
            self._discover_external_plugins()

            # Initialize plugins
            await self.plugin_manager.initialize_plugins(self.container)

            # Initialize core services
            await self._initialize_core_services()

            self._initialized = True
            logger.info("✅ Application initialized successfully")

        except Exception as e:
            logger.error(f"❌ Application initialization failed: {e}")
            raise ConfigurationError(f"Application initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the application"""
        try:
            logger.info("🛑 Shutting down Azure AI IT Copilot Application")

            # Shutdown plugins
            await self.plugin_manager.shutdown_plugins()

            # Shutdown core services
            await self._shutdown_core_services()

            self._initialized = False
            logger.info("✅ Application shutdown completed")

        except Exception as e:
            logger.error(f"❌ Application shutdown failed: {e}")

    async def _setup_dependency_injection(self) -> None:
        """Setup dependency injection container"""
        logger.info("🔧 Setting up dependency injection")
        self.container = configure_services()

    async def _register_builtin_plugins(self) -> None:
        """Register built-in plugins"""
        logger.info("📦 Registering built-in plugins")

        # Register agent plugins
        try:
            from ai_orchestrator.agents import (
                ResourceAgent, IncidentAgent, CostOptimizationAgent,
                ComplianceAgent, PredictiveAgent, InfrastructureAgent
            )

            register_agent_plugin(ResourceAgent)
            register_agent_plugin(IncidentAgent)
            register_agent_plugin(CostOptimizationAgent)
            register_agent_plugin(ComplianceAgent)
            register_agent_plugin(PredictiveAgent)
            register_agent_plugin(InfrastructureAgent)

            logger.info("✅ Built-in agent plugins registered")

        except ImportError as e:
            logger.warning(f"⚠️ Some built-in agents not available: {e}")

        # Register service plugins would be done here
        # They're already registered in configure_services()

    def _discover_external_plugins(self) -> None:
        """Discover external plugins"""
        logger.info("🔍 Discovering external plugins")

        # Add plugin search paths
        plugin_paths = [
            Path("plugins"),
            Path("custom_agents"),
            Path("extensions")
        ]

        for path in plugin_paths:
            if path.exists():
                self.plugin_manager.add_plugin_path(path)

        # Discover plugins
        self.plugin_manager.discover_plugins()

    async def _initialize_core_services(self) -> None:
        """Initialize core services"""
        logger.info("🔧 Initializing core services")

        try:
            # Initialize Azure client factory
            azure_factory = self.container.resolve(IAzureClientFactory)
            await azure_factory.initialize()

            # Initialize notification service
            notification_service = self.container.resolve(INotificationService)
            await notification_service.initialize()

            logger.info("✅ Core services initialized")

        except Exception as e:
            logger.error(f"❌ Core services initialization failed: {e}")
            raise

    async def _shutdown_core_services(self) -> None:
        """Shutdown core services"""
        logger.info("🛑 Shutting down core services")

        try:
            # Shutdown services
            if self.container:
                azure_factory = self.container.resolve(IAzureClientFactory)
                await azure_factory.shutdown()

                notification_service = self.container.resolve(INotificationService)
                await notification_service.shutdown()

            logger.info("✅ Core services shutdown completed")

        except Exception as e:
            logger.error(f"❌ Core services shutdown failed: {e}")

    def get_service(self, service_type):
        """Get a service from the container"""
        if not self._initialized:
            raise RuntimeError("Application not initialized")

        return self.container.resolve(service_type)

    def is_initialized(self) -> bool:
        """Check if application is initialized"""
        return self._initialized

    def get_plugin_info(self) -> dict:
        """Get information about loaded plugins"""
        return {
            "plugins": self.plugin_manager.get_plugin_info(),
            "total_plugins": len(self.plugin_manager.get_all_plugins()),
            "plugin_manager_initialized": self.plugin_manager.is_initialized
        }


# Global application instance
_application: Optional[Application] = None


def get_application() -> Application:
    """Get the global application instance"""
    global _application
    if _application is None:
        _application = Application()
    return _application


async def initialize_application() -> Application:
    """Initialize the global application"""
    app = get_application()
    if not app.is_initialized():
        await app.initialize()
    return app


async def shutdown_application() -> None:
    """Shutdown the global application"""
    global _application
    if _application and _application.is_initialized():
        await _application.shutdown()
        _application = None