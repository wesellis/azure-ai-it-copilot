"""
Plugin Manager for Azure AI IT Copilot
Provides modular plugin architecture for extending functionality
"""

import logging
import importlib
import inspect
from typing import Dict, Any, List, Type, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from .interfaces import IAgent
from .base_classes import BaseAgent
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class IPlugin(ABC):
    """Interface for plugins"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description"""
        pass

    @abstractmethod
    async def initialize(self, container) -> None:
        """Initialize the plugin"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin"""
        pass


class AgentPlugin(IPlugin):
    """Base class for agent plugins"""

    def __init__(self, agent_class: Type[BaseAgent]):
        self.agent_class = agent_class
        self._initialized = False

    @property
    def name(self) -> str:
        return getattr(self.agent_class, 'agent_type', self.agent_class.__name__)

    @property
    def version(self) -> str:
        return getattr(self.agent_class, 'version', '1.0.0')

    @property
    def description(self) -> str:
        return getattr(self.agent_class, '__doc__', 'No description available')

    async def initialize(self, container) -> None:
        """Initialize agent plugin"""
        # Register agent class in dependency container
        container.register_transient(IAgent, self.agent_class)
        self._initialized = True
        logger.info(f"Agent plugin '{self.name}' initialized")

    async def shutdown(self) -> None:
        """Shutdown agent plugin"""
        self._initialized = False
        logger.info(f"Agent plugin '{self.name}' shutdown")


class ServicePlugin(IPlugin):
    """Base class for service plugins"""

    def __init__(self, service_class: Type, interface_type: Type = None):
        self.service_class = service_class
        self.interface_type = interface_type
        self._initialized = False

    @property
    def name(self) -> str:
        return getattr(self.service_class, 'service_name', self.service_class.__name__)

    @property
    def version(self) -> str:
        return getattr(self.service_class, 'version', '1.0.0')

    @property
    def description(self) -> str:
        return getattr(self.service_class, '__doc__', 'No description available')

    async def initialize(self, container) -> None:
        """Initialize service plugin"""
        if self.interface_type:
            container.register_singleton(self.interface_type, self.service_class)
        else:
            container.register_singleton(self.service_class, self.service_class)

        self._initialized = True
        logger.info(f"Service plugin '{self.name}' initialized")

    async def shutdown(self) -> None:
        """Shutdown service plugin"""
        self._initialized = False
        logger.info(f"Service plugin '{self.name}' shutdown")


class PluginManager:
    """Manages plugin lifecycle and registration"""

    def __init__(self):
        self._plugins: Dict[str, IPlugin] = {}
        self._plugin_paths: List[Path] = []
        self._initialized = False

    def add_plugin_path(self, path: Path) -> None:
        """Add a path to search for plugins"""
        if path.exists() and path.is_dir():
            self._plugin_paths.append(path)
            logger.debug(f"Added plugin path: {path}")
        else:
            logger.warning(f"Plugin path does not exist: {path}")

    def register_plugin(self, plugin: IPlugin) -> None:
        """Register a plugin manually"""
        if plugin.name in self._plugins:
            logger.warning(f"Plugin '{plugin.name}' is already registered")
            return

        self._plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")

    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin"""
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
        return False

    async def initialize_plugins(self, container) -> None:
        """Initialize all registered plugins"""
        logger.info("Initializing plugins...")

        for plugin_name, plugin in self._plugins.items():
            try:
                await plugin.initialize(container)
                logger.info(f"Plugin '{plugin_name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize plugin '{plugin_name}': {e}")

        self._initialized = True
        logger.info(f"Initialized {len(self._plugins)} plugins")

    async def shutdown_plugins(self) -> None:
        """Shutdown all plugins"""
        logger.info("Shutting down plugins...")

        for plugin_name, plugin in self._plugins.items():
            try:
                await plugin.shutdown()
                logger.info(f"Plugin '{plugin_name}' shutdown successfully")
            except Exception as e:
                logger.error(f"Failed to shutdown plugin '{plugin_name}': {e}")

        self._initialized = False
        logger.info("All plugins shutdown")

    def discover_plugins(self) -> None:
        """Discover plugins from registered paths"""
        logger.info("Discovering plugins...")

        for plugin_path in self._plugin_paths:
            try:
                self._discover_plugins_in_path(plugin_path)
            except Exception as e:
                logger.error(f"Error discovering plugins in {plugin_path}: {e}")

    def _discover_plugins_in_path(self, path: Path) -> None:
        """Discover plugins in a specific path"""
        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for plugin classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (hasattr(obj, '__plugin__') and
                        getattr(obj, '__plugin__') is True and
                        issubclass(obj, IPlugin)):

                        plugin = obj()
                        self.register_plugin(plugin)

            except Exception as e:
                logger.warning(f"Failed to load plugin from {py_file}: {e}")

    def get_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """Get a plugin by name"""
        return self._plugins.get(plugin_name)

    def get_all_plugins(self) -> Dict[str, IPlugin]:
        """Get all registered plugins"""
        return self._plugins.copy()

    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """Get information about all plugins"""
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "type": type(plugin).__name__
            }
            for plugin in self._plugins.values()
        ]

    @property
    def is_initialized(self) -> bool:
        """Check if plugin manager is initialized"""
        return self._initialized


# Global plugin manager instance
_plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager"""
    return _plugin_manager


def register_agent_plugin(agent_class: Type[BaseAgent]) -> None:
    """Convenience function to register an agent plugin"""
    plugin = AgentPlugin(agent_class)
    _plugin_manager.register_plugin(plugin)


def register_service_plugin(service_class: Type, interface_type: Type = None) -> None:
    """Convenience function to register a service plugin"""
    plugin = ServicePlugin(service_class, interface_type)
    _plugin_manager.register_plugin(plugin)


# Decorator for marking plugin classes
def plugin(cls):
    """Decorator to mark a class as a plugin"""
    cls.__plugin__ = True
    return cls