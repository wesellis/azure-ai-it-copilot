"""
Metrics Service Implementation
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, REGISTRY

from core.interfaces import IMetricsCollector, IConfigurationProvider
from core.base_classes import BaseService

logger = logging.getLogger(__name__)


class MetricsService(BaseService, IMetricsCollector):
    """Prometheus-based metrics collection service"""

    def __init__(self, config_provider: IConfigurationProvider):
        super().__init__(config_provider)
        self._metrics = {}

    async def initialize(self) -> None:
        """Initialize metrics service"""
        self._register_default_metrics()
        logger.info("Metrics service initialized")

    async def shutdown(self) -> None:
        """Shutdown metrics service"""
        logger.info("Metrics service shutdown")

    def increment_counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        metric = self._get_or_create_counter(name, labels)
        if labels:
            metric.labels(**labels).inc(value)
        else:
            metric.inc(value)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value"""
        metric = self._get_or_create_histogram(name, labels)
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value"""
        metric = self._get_or_create_gauge(name, labels)
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)

    def _get_or_create_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> Counter:
        """Get or create a counter metric"""
        if name not in self._metrics:
            label_names = list(labels.keys()) if labels else []
            self._metrics[name] = Counter(name, f"Counter metric: {name}", label_names)
        return self._metrics[name]

    def _get_or_create_histogram(self, name: str, labels: Optional[Dict[str, str]] = None) -> Histogram:
        """Get or create a histogram metric"""
        if name not in self._metrics:
            label_names = list(labels.keys()) if labels else []
            self._metrics[name] = Histogram(name, f"Histogram metric: {name}", label_names)
        return self._metrics[name]

    def _get_or_create_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Gauge:
        """Get or create a gauge metric"""
        if name not in self._metrics:
            label_names = list(labels.keys()) if labels else []
            self._metrics[name] = Gauge(name, f"Gauge metric: {name}", label_names)
        return self._metrics[name]

    def _register_default_metrics(self):
        """Register default application metrics"""
        # Application metrics
        self._metrics["requests_total"] = Counter(
            "requests_total", "Total requests", ["method", "endpoint", "status"]
        )
        self._metrics["request_duration"] = Histogram(
            "request_duration_seconds", "Request duration"
        )
        self._metrics["active_connections"] = Gauge(
            "active_connections", "Active connections"
        )