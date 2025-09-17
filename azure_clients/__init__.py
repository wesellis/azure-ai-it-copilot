"""
Azure client implementations
Centralized Azure service clients with proper error handling and authentication
"""

from .resource_client_simple import AzureResourceClient
from .cosmos_client_simple import CosmosDBClient

__all__ = [
    'AzureResourceClient',
    'CosmosDBClient'
]