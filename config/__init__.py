"""
Configuration management for Azure AI IT Copilot
Centralized configuration loading and validation
"""

from .settings import Settings, get_settings

__all__ = [
    'Settings',
    'get_settings'
]