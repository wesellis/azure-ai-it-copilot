"""
Hot-reloading configuration management for Azure AI IT Copilot
Monitors .env file changes and reloads configuration without restart
"""

import os
import time
import threading
from typing import Dict, Callable, Optional, Any
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigChangeHandler(FileSystemEventHandler):
    """Handles configuration file change events"""

    def __init__(self, reload_callback: Callable):
        self.reload_callback = reload_callback
        self.last_reload = 0
        self.debounce_delay = 1.0  # Prevent rapid reloads

    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return

        if event.src_path.endswith('.env'):
            current_time = time.time()
            if current_time - self.last_reload > self.debounce_delay:
                logger.info(f"Configuration file changed: {event.src_path}")
                self.reload_callback()
                self.last_reload = current_time


class HotReloadConfig:
    """Manages hot-reloading of configuration"""

    def __init__(self, config_path: str = None, watch_enabled: bool = True):
        self.config_path = config_path or self._find_env_file()
        self.watch_enabled = watch_enabled
        self.observer = None
        self.config_cache = {}
        self.change_callbacks = []
        self.last_modified = 0
        self._lock = threading.Lock()

        # Load initial configuration
        self.reload_config()

        # Start file watcher if enabled
        if self.watch_enabled and self.config_path:
            self.start_watching()

    def _find_env_file(self) -> Optional[str]:
        """Find the .env file in the project"""
        # Check current directory and parent directories
        current_dir = Path.cwd()
        for path in [current_dir] + list(current_dir.parents):
            env_file = path / '.env'
            if env_file.exists():
                return str(env_file)
        return None

    def reload_config(self) -> bool:
        """
        Reload configuration from file

        Returns:
            True if configuration was reloaded, False otherwise
        """
        if not self.config_path or not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found: {self.config_path}")
            return False

        try:
            # Check if file was modified
            current_modified = os.path.getmtime(self.config_path)
            if current_modified <= self.last_modified:
                return False  # No changes

            with self._lock:
                old_config = self.config_cache.copy()
                new_config = {}

                # Read .env file
                with open(self.config_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()

                        # Skip empty lines and comments
                        if not line or line.startswith('#'):
                            continue

                        # Parse key=value pairs
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()

                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]

                            new_config[key] = value

                # Update cache and environment
                self.config_cache = new_config
                self.last_modified = current_modified

                # Update os.environ
                for key, value in new_config.items():
                    os.environ[key] = value

                # Notify callbacks of changes
                changes = self._detect_changes(old_config, new_config)
                if changes:
                    self._notify_callbacks(changes)

                logger.info(f"Configuration reloaded from {self.config_path}")
                logger.debug(f"Loaded {len(new_config)} configuration values")

                return True

        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return False

    def _detect_changes(self, old_config: Dict, new_config: Dict) -> Dict[str, Dict]:
        """Detect changes between old and new configuration"""
        changes = {
            'added': {},
            'modified': {},
            'removed': {}
        }

        # Find added and modified keys
        for key, value in new_config.items():
            if key not in old_config:
                changes['added'][key] = value
            elif old_config[key] != value:
                changes['modified'][key] = {
                    'old': old_config[key],
                    'new': value
                }

        # Find removed keys
        for key in old_config:
            if key not in new_config:
                changes['removed'][key] = old_config[key]

        return changes

    def _notify_callbacks(self, changes: Dict):
        """Notify registered callbacks of configuration changes"""
        for callback in self.change_callbacks:
            try:
                callback(changes)
            except Exception as e:
                logger.error(f"Error in configuration change callback: {e}")

    def start_watching(self):
        """Start watching configuration file for changes"""
        if not self.config_path:
            logger.warning("No configuration file to watch")
            return

        try:
            self.observer = Observer()
            handler = ConfigChangeHandler(self.reload_config)

            # Watch the directory containing the config file
            watch_dir = os.path.dirname(self.config_path)
            self.observer.schedule(handler, watch_dir, recursive=False)
            self.observer.start()

            logger.info(f"Started watching configuration file: {self.config_path}")

        except Exception as e:
            logger.error(f"Error starting file watcher: {e}")

    def stop_watching(self):
        """Stop watching configuration file"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped watching configuration file")

    def add_change_callback(self, callback: Callable[[Dict], None]):
        """
        Add callback to be notified of configuration changes

        Args:
            callback: Function that takes changes dict as parameter
        """
        self.change_callbacks.append(callback)

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback to environment

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config_cache.get(key, os.getenv(key, default))

    def get_all_config(self) -> Dict[str, str]:
        """Get all configuration values"""
        return self.config_cache.copy()

    def validate_required_keys(self, required_keys: list) -> Dict[str, bool]:
        """
        Validate that required configuration keys are present

        Args:
            required_keys: List of required configuration keys

        Returns:
            Dict mapping keys to whether they are present
        """
        validation = {}
        for key in required_keys:
            validation[key] = key in self.config_cache and bool(self.config_cache[key])
        return validation

    def get_status(self) -> Dict[str, Any]:
        """Get status information about configuration management"""
        return {
            'config_file': self.config_path,
            'file_exists': os.path.exists(self.config_path) if self.config_path else False,
            'watching_enabled': self.watch_enabled,
            'is_watching': self.observer is not None and self.observer.is_alive(),
            'last_modified': datetime.fromtimestamp(self.last_modified).isoformat() if self.last_modified else None,
            'config_count': len(self.config_cache),
            'callbacks_registered': len(self.change_callbacks)
        }


# Global instance for easy access
_hot_reload_config = None


def get_hot_reload_config(config_path: str = None, watch_enabled: bool = True) -> HotReloadConfig:
    """Get or create global hot-reload configuration instance"""
    global _hot_reload_config
    if _hot_reload_config is None:
        _hot_reload_config = HotReloadConfig(config_path, watch_enabled)
    return _hot_reload_config


def get_config(key: str, default: Any = None) -> Any:
    """Convenience function to get configuration value"""
    config = get_hot_reload_config()
    return config.get_config(key, default)


def add_config_change_callback(callback: Callable[[Dict], None]):
    """Convenience function to add configuration change callback"""
    config = get_hot_reload_config()
    config.add_change_callback(callback)


# Example callback for logging configuration changes
def log_config_changes(changes: Dict):
    """Example callback that logs configuration changes"""
    if changes['added']:
        logger.info(f"Configuration added: {list(changes['added'].keys())}")

    if changes['modified']:
        for key, change in changes['modified'].items():
            # Don't log sensitive values
            if any(word in key.lower() for word in ['password', 'secret', 'key', 'token']):
                logger.info(f"Configuration modified: {key} = [REDACTED]")
            else:
                logger.info(f"Configuration modified: {key} = {change['new']} (was {change['old']})")

    if changes['removed']:
        logger.info(f"Configuration removed: {list(changes['removed'].keys())}")


if __name__ == "__main__":
    # Example usage
    import time

    # Create hot-reload config with logging callback
    config = HotReloadConfig()
    config.add_change_callback(log_config_changes)

    print("Hot-reload configuration started. Modify .env file to see changes.")
    print(f"Watching: {config.config_path}")
    print(f"Current config count: {len(config.get_all_config())}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        config.stop_watching()
        print("\nStopped watching configuration.")