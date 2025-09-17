"""
Event Bus Service Implementation
"""

import logging
import asyncio
from typing import Dict, Any, List, Callable
from datetime import datetime

from core.interfaces import IEventBus, IConfigurationProvider
from core.base_classes import BaseService

logger = logging.getLogger(__name__)


class EventBusService(BaseService, IEventBus):
    """In-memory event bus service"""

    def __init__(self, config_provider: IConfigurationProvider):
        super().__init__(config_provider)
        self._subscribers: Dict[str, List[Dict[str, Any]]] = {}

    async def initialize(self) -> None:
        """Initialize event bus service"""
        logger.info("Event bus service initialized")

    async def shutdown(self) -> None:
        """Shutdown event bus service"""
        self._subscribers.clear()
        logger.info("Event bus service shutdown")

    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        if event_type in self._subscribers:
            event = {
                "type": event_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }

            for subscriber in self._subscribers[event_type]:
                try:
                    handler = subscriber["handler"]
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed for {event_type}: {e}")

    async def subscribe(self, event_type: str, handler: Callable) -> str:
        """Subscribe to events"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        subscription_id = f"{event_type}_{len(self._subscribers[event_type])}"
        self._subscribers[event_type].append({
            "id": subscription_id,
            "handler": handler
        })

        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        for event_type, subscribers in self._subscribers.items():
            for i, subscriber in enumerate(subscribers):
                if subscriber["id"] == subscription_id:
                    del subscribers[i]
                    return True
        return False