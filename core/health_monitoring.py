"""
Comprehensive Health Monitoring System
Provides real-time health checks, metrics collection, and system observability
"""

import asyncio
import time
import logging
import psutil
import os
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import json

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: Union[float, int, str, bool]
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    description: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)
    trend: Optional[str] = None  # "increasing", "decreasing", "stable"


@dataclass
class ComponentHealth:
    """Health status for a system component"""
    component: str
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.utcnow)
    check_duration_ms: float = 0.0
    error_message: Optional[str] = None
    uptime_seconds: float = 0.0


class HealthMonitor:
    """Comprehensive health monitoring system"""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._health_checks: Dict[str, Callable] = {}
        self._component_health: Dict[str, ComponentHealth] = {}
        self._system_metrics: Dict[str, HealthMetric] = {}
        self._start_time = time.time()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._alert_callbacks: List[Callable] = []
        self._health_history: Dict[str, List[ComponentHealth]] = {}
        self._running = False

    def register_health_check(self, component: str, check_func: Callable, dependencies: List[str] = None):
        """Register a health check function for a component"""
        self._health_checks[component] = check_func
        self._component_health[component] = ComponentHealth(
            component=component,
            status=HealthStatus.UNKNOWN,
            dependencies=dependencies or []
        )
        logger.info(f"Registered health check for component: {component}")

    def register_alert_callback(self, callback: Callable):
        """Register callback for health alerts"""
        self._alert_callbacks.append(callback)

    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")

    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self.perform_health_checks()
                await self._update_system_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    async def perform_health_checks(self) -> Dict[str, ComponentHealth]:
        """Perform all registered health checks"""
        tasks = []

        for component in self._health_checks:
            task = asyncio.create_task(self._check_component_health(component))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return self._component_health

    async def _check_component_health(self, component: str):
        """Check health of a specific component"""
        check_func = self._health_checks[component]
        start_time = time.time()

        try:
            # Run health check
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            # Parse result
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                metrics = []
                error_message = None
            elif isinstance(result, dict):
                status = HealthStatus(result.get('status', HealthStatus.UNKNOWN))
                metrics = result.get('metrics', [])
                error_message = result.get('error')
            else:
                status = HealthStatus.UNKNOWN
                metrics = []
                error_message = f"Invalid health check result type: {type(result)}"

            # Update component health
            duration_ms = (time.time() - start_time) * 1000

            component_health = ComponentHealth(
                component=component,
                status=status,
                metrics=metrics,
                dependencies=self._component_health[component].dependencies,
                last_check=datetime.utcnow(),
                check_duration_ms=duration_ms,
                error_message=error_message,
                uptime_seconds=time.time() - self._start_time
            )

            self._component_health[component] = component_health

            # Store history
            if component not in self._health_history:
                self._health_history[component] = []
            self._health_history[component].append(component_health)

            # Keep only last 100 records per component
            if len(self._health_history[component]) > 100:
                self._health_history[component] = self._health_history[component][-100:]

        except Exception as e:
            logger.error(f"Health check failed for {component}: {e}")

            self._component_health[component] = ComponentHealth(
                component=component,
                status=HealthStatus.CRITICAL,
                last_check=datetime.utcnow(),
                check_duration_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
                uptime_seconds=time.time() - self._start_time
            )

    async def _update_system_metrics(self):
        """Update system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._system_metrics['cpu_usage'] = HealthMetric(
                name='cpu_usage',
                value=cpu_percent,
                status=self._get_status_from_thresholds(cpu_percent, 70, 90),
                threshold_warning=70,
                threshold_critical=90,
                unit='%',
                description='CPU usage percentage'
            )

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._system_metrics['memory_usage'] = HealthMetric(
                name='memory_usage',
                value=memory_percent,
                status=self._get_status_from_thresholds(memory_percent, 80, 95),
                threshold_warning=80,
                threshold_critical=95,
                unit='%',
                description='Memory usage percentage'
            )

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            self._system_metrics['disk_usage'] = HealthMetric(
                name='disk_usage',
                value=disk_percent,
                status=self._get_status_from_thresholds(disk_percent, 80, 95),
                threshold_warning=80,
                threshold_critical=95,
                unit='%',
                description='Disk usage percentage'
            )

            # Network metrics
            network = psutil.net_io_counters()
            self._system_metrics['network_bytes_sent'] = HealthMetric(
                name='network_bytes_sent',
                value=network.bytes_sent,
                status=HealthStatus.HEALTHY,
                unit='bytes',
                description='Total bytes sent'
            )

            # Process metrics
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            self._system_metrics['process_memory'] = HealthMetric(
                name='process_memory',
                value=process_memory,
                status=self._get_status_from_thresholds(process_memory, 500, 1000),
                threshold_warning=500,
                threshold_critical=1000,
                unit='MB',
                description='Process memory usage'
            )

            # File descriptors
            num_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            self._system_metrics['file_descriptors'] = HealthMetric(
                name='file_descriptors',
                value=num_fds,
                status=self._get_status_from_thresholds(num_fds, 500, 900),
                threshold_warning=500,
                threshold_critical=900,
                unit='count',
                description='Number of open file descriptors'
            )

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def _get_status_from_thresholds(self, value: float, warning: float, critical: float) -> HealthStatus:
        """Determine health status based on thresholds"""
        if value >= critical:
            return HealthStatus.CRITICAL
        elif value >= warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    async def _check_alerts(self):
        """Check for alert conditions and trigger callbacks"""
        alerts = []

        # Check component health
        for component, health in self._component_health.items():
            if health.status == HealthStatus.CRITICAL:
                alerts.append({
                    'type': 'component_critical',
                    'component': component,
                    'message': f"Component {component} is in critical state",
                    'error': health.error_message,
                    'timestamp': datetime.utcnow()
                })
            elif health.status == HealthStatus.WARNING:
                alerts.append({
                    'type': 'component_warning',
                    'component': component,
                    'message': f"Component {component} is in warning state",
                    'timestamp': datetime.utcnow()
                })

        # Check system metrics
        for metric_name, metric in self._system_metrics.items():
            if metric.status == HealthStatus.CRITICAL:
                alerts.append({
                    'type': 'metric_critical',
                    'metric': metric_name,
                    'value': metric.value,
                    'threshold': metric.threshold_critical,
                    'message': f"Metric {metric_name} is critical: {metric.value}{metric.unit}",
                    'timestamp': datetime.utcnow()
                })

        # Trigger alert callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        overall_status = HealthStatus.HEALTHY
        critical_components = []
        warning_components = []

        # Determine overall status
        for component, health in self._component_health.items():
            if health.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                critical_components.append(component)
            elif health.status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.WARNING
                warning_components.append(component)

        # Check system metrics
        critical_metrics = []
        warning_metrics = []

        for metric_name, metric in self._system_metrics.items():
            if metric.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                critical_metrics.append(metric_name)
            elif metric.status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.WARNING
                warning_metrics.append(metric_name)

        return {
            'status': overall_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': time.time() - self._start_time,
            'components': {
                'total': len(self._component_health),
                'healthy': len([h for h in self._component_health.values() if h.status == HealthStatus.HEALTHY]),
                'warning': len(warning_components),
                'critical': len(critical_components),
                'critical_components': critical_components,
                'warning_components': warning_components
            },
            'system_metrics': {
                'critical_metrics': critical_metrics,
                'warning_metrics': warning_metrics
            }
        }

    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information"""
        return {
            'overall': self.get_overall_health(),
            'components': {
                name: {
                    'status': health.status.value,
                    'last_check': health.last_check.isoformat(),
                    'check_duration_ms': health.check_duration_ms,
                    'error_message': health.error_message,
                    'uptime_seconds': health.uptime_seconds,
                    'metrics': [
                        {
                            'name': m.name,
                            'value': m.value,
                            'status': m.status.value,
                            'unit': m.unit,
                            'description': m.description
                        } for m in health.metrics
                    ] if health.metrics else []
                }
                for name, health in self._component_health.items()
            },
            'system_metrics': {
                name: {
                    'value': metric.value,
                    'status': metric.status.value,
                    'threshold_warning': metric.threshold_warning,
                    'threshold_critical': metric.threshold_critical,
                    'unit': metric.unit,
                    'description': metric.description,
                    'last_updated': metric.last_updated.isoformat()
                }
                for name, metric in self._system_metrics.items()
            }
        }

    def get_health_history(self, component: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get health history for components"""
        if component:
            history = self._health_history.get(component, [])
            return {
                component: [
                    {
                        'status': h.status.value,
                        'timestamp': h.last_check.isoformat(),
                        'check_duration_ms': h.check_duration_ms,
                        'error_message': h.error_message
                    }
                    for h in history[-limit:]
                ]
            }
        else:
            return {
                comp: [
                    {
                        'status': h.status.value,
                        'timestamp': h.last_check.isoformat(),
                        'check_duration_ms': h.check_duration_ms,
                        'error_message': h.error_message
                    }
                    for h in history[-limit:]
                ]
                for comp, history in self._health_history.items()
            }


# Global health monitor instance
health_monitor = HealthMonitor()


def health_check(component: str, dependencies: List[str] = None):
    """Decorator to register a function as a health check"""
    def decorator(func):
        health_monitor.register_health_check(component, func, dependencies)
        return func
    return decorator


async def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    return health_monitor.get_overall_health()


async def get_detailed_health_status() -> Dict[str, Any]:
    """Get detailed health status"""
    return health_monitor.get_detailed_health()


# Example health checks for common components
@health_check("database")
async def check_database_health():
    """Database health check"""
    try:
        # This would be replaced with actual database check
        # For example: await db.execute("SELECT 1")
        return {
            'status': HealthStatus.HEALTHY,
            'metrics': [
                HealthMetric(
                    name='connection_pool_size',
                    value=10,
                    status=HealthStatus.HEALTHY,
                    unit='connections',
                    description='Active database connections'
                )
            ]
        }
    except Exception as e:
        return {
            'status': HealthStatus.CRITICAL,
            'error': str(e)
        }


@health_check("redis")
async def check_redis_health():
    """Redis health check"""
    try:
        # This would be replaced with actual Redis check
        # For example: await redis.ping()
        return {
            'status': HealthStatus.HEALTHY,
            'metrics': [
                HealthMetric(
                    name='ping_response_time',
                    value=1.2,
                    status=HealthStatus.HEALTHY,
                    unit='ms',
                    description='Redis ping response time'
                )
            ]
        }
    except Exception as e:
        return {
            'status': HealthStatus.CRITICAL,
            'error': str(e)
        }


@health_check("external_apis")
async def check_external_apis_health():
    """External APIs health check"""
    try:
        # This would check Azure APIs, OpenAI, etc.
        return {
            'status': HealthStatus.HEALTHY,
            'metrics': [
                HealthMetric(
                    name='azure_api_response_time',
                    value=150,
                    status=HealthStatus.HEALTHY,
                    threshold_warning=500,
                    threshold_critical=1000,
                    unit='ms',
                    description='Azure API response time'
                )
            ]
        }
    except Exception as e:
        return {
            'status': HealthStatus.CRITICAL,
            'error': str(e)
        }