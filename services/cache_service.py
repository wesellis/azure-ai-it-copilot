"""
Cache Service Implementation
"""

import logging
import json
from typing import Any, Optional
from datetime import datetime, timedelta

from core.interfaces import ICacheProvider, IConfigurationProvider
from core.base_classes import BaseService

logger = logging.getLogger(__name__)


class CacheService(BaseService, ICacheProvider):
    """In-memory cache service with TTL support"""

    def __init__(self, config_provider: IConfigurationProvider):
        super().__init__(config_provider)
        self._cache = {}
        self._default_ttl = 300  # 5 minutes

    async def initialize(self) -> None:
        """Initialize cache service"""
        logger.info("Cache service initialized")

    async def shutdown(self) -> None:
        """Shutdown cache service"""
        await self.clear()
        logger.info("Cache service shutdown")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self._cache:
            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                return None
            return entry["value"]
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            expiry = datetime.utcnow() + timedelta(seconds=ttl or self._default_ttl)
            self._cache[key] = {
                "value": value,
                "expires_at": expiry
            }
            return True
        except Exception as e:
            logger.error(f"Failed to set cache key '{key}': {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def clear(self) -> bool:
        """Clear all cache"""
        self._cache.clear()
        return True

    def _is_expired(self, entry: dict) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > entry["expires_at"]