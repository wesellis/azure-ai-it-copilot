"""
Comprehensive Logging and Observability System
Advanced logging, tracing, and observability with structured data and correlation
"""

import asyncio
import time
import logging
import json
import os
import sys
import uuid
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from contextlib import contextmanager, asynccontextmanager
import traceback
import psutil
from collections import defaultdict, deque
import hashlib
import weakref

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TraceLevel(str, Enum):
    """Trace level enumeration"""
    SPAN = "span"
    EVENT = "event"
    METRIC = "metric"
    LOG = "log"


@dataclass
class CorrelationContext:
    """Request correlation context"""
    correlation_id: str
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    trace_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    correlation_context: Optional[CorrelationContext] = None
    exception_info: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, Union[int, float]] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Distributed trace span"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "started"  # started, success, error
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Union[int, float]] = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricPoint:
    """Metric data point"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    metric_type: str  # counter, gauge, histogram, timer
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


class ContextManager:
    """Thread-safe correlation context manager"""

    def __init__(self):
        self._contexts: Dict[str, CorrelationContext] = {}
        self._thread_contexts: threading.local = threading.local()
        self._lock = threading.RLock()

    def set_context(self, context: CorrelationContext):
        """Set correlation context for current thread"""
        with self._lock:
            self._contexts[context.correlation_id] = context
            self._thread_contexts.current = context

    def get_context(self) -> Optional[CorrelationContext]:
        """Get current thread correlation context"""
        return getattr(self._thread_contexts, 'current', None)

    def get_context_by_id(self, correlation_id: str) -> Optional[CorrelationContext]:
        """Get context by correlation ID"""
        return self._contexts.get(correlation_id)

    def create_context(self, operation_name: str = "", user_id: str = None,
                      parent_context: CorrelationContext = None) -> CorrelationContext:
        """Create new correlation context"""
        correlation_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        context = CorrelationContext(
            correlation_id=correlation_id,
            request_id=request_id,
            user_id=user_id,
            operation_id=operation_name,
            parent_span_id=parent_context.correlation_id if parent_context else None,
            trace_id=parent_context.trace_id if parent_context else correlation_id
        )

        self.set_context(context)
        return context

    def clear_context(self):
        """Clear current thread context"""
        if hasattr(self._thread_contexts, 'current'):
            delattr(self._thread_contexts, 'current')

    @contextmanager
    def operation_context(self, operation_name: str, user_id: str = None):
        """Context manager for operations"""
        old_context = self.get_context()
        new_context = self.create_context(operation_name, user_id, old_context)

        try:
            yield new_context
        finally:
            if old_context:
                self.set_context(old_context)
            else:
                self.clear_context()


class StructuredLogger:
    """Advanced structured logger with correlation support"""

    def __init__(self, name: str, context_manager: ContextManager):
        self.name = name
        self.context_manager = context_manager
        self._python_logger = logging.getLogger(name)
        self._handlers: List[Callable] = []
        self._filters: List[Callable] = []

    def add_handler(self, handler: Callable):
        """Add custom log handler"""
        self._handlers.append(handler)

    def add_filter(self, filter_func: Callable) -> None:
        """Add log filter function"""
        self._filters.append(filter_func)

    def _should_log(self, level: LogLevel, entry: LogEntry) -> bool:
        """Check if log entry should be processed"""
        for filter_func in self._filters:
            if not filter_func(level, entry):
                return False
        return True

    def _create_log_entry(self, level: LogLevel, message: str,
                         extra_data: Dict[str, Any] = None,
                         exception_info: str = None,
                         tags: List[str] = None,
                         metrics: Dict[str, Union[int, float]] = None) -> LogEntry:
        """Create structured log entry"""
        frame = sys._getframe(3)  # Go up the call stack to find actual caller

        return LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            logger_name=self.name,
            module=frame.f_globals.get('__name__', ''),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            correlation_context=self.context_manager.get_context(),
            exception_info=exception_info,
            extra_data=extra_data or {},
            tags=tags or [],
            metrics=metrics or {}
        )

    def _format_log_entry(self, entry: LogEntry) -> str:
        """Format log entry as JSON"""
        data = {
            'timestamp': entry.timestamp.isoformat(),
            'level': entry.level.value,
            'message': entry.message,
            'logger': entry.logger_name,
            'module': entry.module,
            'function': entry.function,
            'line': entry.line_number
        }

        # Add correlation context
        if entry.correlation_context:
            data['correlation'] = {
                'correlation_id': entry.correlation_context.correlation_id,
                'request_id': entry.correlation_context.request_id,
                'user_id': entry.correlation_context.user_id,
                'session_id': entry.correlation_context.session_id,
                'operation_id': entry.correlation_context.operation_id,
                'trace_id': entry.correlation_context.trace_id,
                'parent_span_id': entry.correlation_context.parent_span_id
            }

        # Add extra data
        if entry.extra_data:
            data['extra'] = entry.extra_data

        # Add tags
        if entry.tags:
            data['tags'] = entry.tags

        # Add metrics
        if entry.metrics:
            data['metrics'] = entry.metrics

        # Add exception info
        if entry.exception_info:
            data['exception'] = entry.exception_info

        return json.dumps(data, default=str)

    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method"""
        entry = self._create_log_entry(level, message, **kwargs)

        if not self._should_log(level, entry):
            return

        # Format and send to Python logger
        formatted_message = self._format_log_entry(entry)
        getattr(self._python_logger, level.value.lower())(formatted_message)

        # Send to custom handlers
        for handler in self._handlers:
            try:
                handler(entry)
            except Exception as e:
                # Avoid infinite recursion by using basic logging
                logging.error(f"Error in log handler: {e}")

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        if 'exception_info' not in kwargs:
            kwargs['exception_info'] = traceback.format_exc()
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        if 'exception_info' not in kwargs:
            kwargs['exception_info'] = traceback.format_exc()
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def metric(self, name: str, value: Union[int, float], **kwargs):
        """Log metric with structured data"""
        metrics = {name: value}
        self.info(f"Metric: {name}={value}", metrics=metrics, **kwargs)


class TraceManager:
    """Distributed tracing manager"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._active_spans: Dict[str, TraceSpan] = {}
        self._completed_spans: deque = deque(maxlen=10000)
        self._span_relationships: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()

    def start_span(self, operation_name: str, parent_span_id: str = None,
                  tags: Dict[str, str] = None) -> TraceSpan:
        """Start new trace span"""
        span_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4()) if not parent_span_id else self._get_trace_id(parent_span_id)

        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.utcnow(),
            tags=tags or {}
        )

        with self._lock:
            self._active_spans[span_id] = span
            if parent_span_id:
                self._span_relationships[parent_span_id].append(span_id)

        return span

    def finish_span(self, span_id: str, status: str = "success",
                   tags: Dict[str, str] = None, metrics: Dict[str, Union[int, float]] = None):
        """Finish trace span"""
        with self._lock:
            if span_id not in self._active_spans:
                return

            span = self._active_spans[span_id]
            span.end_time = datetime.utcnow()
            span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
            span.status = status

            if tags:
                span.tags.update(tags)
            if metrics:
                span.metrics.update(metrics)

            # Move to completed spans
            del self._active_spans[span_id]
            self._completed_spans.append(span)

    def add_span_log(self, span_id: str, message: str, data: Dict[str, Any] = None):
        """Add log to span"""
        with self._lock:
            if span_id in self._active_spans:
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'message': message,
                    'data': data or {}
                }
                self._active_spans[span_id].logs.append(log_entry)

    def _get_trace_id(self, span_id: str) -> str:
        """Get trace ID for span"""
        if span_id in self._active_spans:
            return self._active_spans[span_id].trace_id

        # Look in completed spans
        for span in reversed(self._completed_spans):
            if span.span_id == span_id:
                return span.trace_id

        return str(uuid.uuid4())  # Fallback

    @asynccontextmanager
    async def trace_operation(self, operation_name: str, parent_span_id: str = None,
                             tags: Dict[str, str] = None):
        """Async context manager for tracing operations"""
        span = self.start_span(operation_name, parent_span_id, tags)

        try:
            yield span
            self.finish_span(span.span_id, "success")
        except Exception as e:
            self.finish_span(span.span_id, "error", tags={'error': str(e)})
            raise

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary of trace"""
        spans = []

        # Check active spans
        for span in self._active_spans.values():
            if span.trace_id == trace_id:
                spans.append(span)

        # Check completed spans
        for span in self._completed_spans:
            if span.trace_id == trace_id:
                spans.append(span)

        if not spans:
            return {'trace_id': trace_id, 'spans': []}

        # Sort by start time
        spans.sort(key=lambda s: s.start_time)

        return {
            'trace_id': trace_id,
            'total_spans': len(spans),
            'total_duration_ms': sum(s.duration_ms for s in spans if s.duration_ms),
            'start_time': spans[0].start_time.isoformat(),
            'end_time': max(s.end_time for s in spans if s.end_time).isoformat() if any(s.end_time for s in spans) else None,
            'spans': [
                {
                    'span_id': s.span_id,
                    'operation_name': s.operation_name,
                    'duration_ms': s.duration_ms,
                    'status': s.status,
                    'tags': s.tags
                }
                for s in spans
            ]
        }


class MetricsCollector:
    """System and application metrics collector"""

    def __init__(self):
        self._metrics: deque = deque(maxlen=100000)
        self._counters: defaultdict = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: defaultdict = defaultdict(list)
        self._timers: defaultdict = defaultdict(list)
        self._lock = threading.RLock()

    def counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment counter metric"""
        with self._lock:
            self._counters[name] += value
            self._record_metric('counter', name, value, tags)

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set gauge metric"""
        with self._lock:
            self._gauges[name] = value
            self._record_metric('gauge', name, value, tags)

    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record histogram value"""
        with self._lock:
            self._histograms[name].append(value)
            # Keep only last 1000 values
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
            self._record_metric('histogram', name, value, tags)

    def timer(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record timer metric"""
        with self._lock:
            self._timers[name].append(duration_ms)
            # Keep only last 1000 values
            if len(self._timers[name]) > 1000:
                self._timers[name] = self._timers[name][-1000:]
            self._record_metric('timer', name, duration_ms, tags)

    @contextmanager
    def time_operation(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.timer(name, duration_ms, tags)

    def _record_metric(self, metric_type: str, name: str, value: Union[int, float],
                      tags: Dict[str, str] = None):
        """Record metric point"""
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            metric_type=metric_type,
            tags=tags or {}
        )
        self._metrics.append(metric_point)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {
                    name: {
                        'count': len(values),
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0,
                        'avg': sum(values) / len(values) if values else 0
                    }
                    for name, values in self._histograms.items()
                },
                'timers': {
                    name: {
                        'count': len(values),
                        'min_ms': min(values) if values else 0,
                        'max_ms': max(values) if values else 0,
                        'avg_ms': sum(values) / len(values) if values else 0
                    }
                    for name, values in self._timers.items()
                }
            }

    def collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.gauge('system.cpu.usage_percent', cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.gauge('system.memory.usage_percent', memory.percent)
            self.gauge('system.memory.available_mb', memory.available / 1024 / 1024)

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.gauge('system.disk.usage_percent', disk.percent)
            self.gauge('system.disk.free_gb', disk.free / 1024 / 1024 / 1024)

            # Network metrics
            network = psutil.net_io_counters()
            self.counter('system.network.bytes_sent', network.bytes_sent)
            self.counter('system.network.bytes_recv', network.bytes_recv)

            # Process metrics
            process = psutil.Process(os.getpid())
            self.gauge('process.memory.rss_mb', process.memory_info().rss / 1024 / 1024)
            self.gauge('process.cpu.percent', process.cpu_percent())

            if hasattr(process, 'num_fds'):
                self.gauge('process.file_descriptors', process.num_fds())

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class ObservabilityManager:
    """Central observability manager"""

    def __init__(self, service_name: str = "azure-ai-copilot"):
        self.service_name = service_name
        self.context_manager = ContextManager()
        self.trace_manager = TraceManager(service_name)
        self.metrics_collector = MetricsCollector()

        # Create loggers
        self._loggers: Dict[str, StructuredLogger] = {}
        self._system_metrics_task: Optional[asyncio.Task] = None

    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create structured logger"""
        if name not in self._loggers:
            self._loggers[name] = StructuredLogger(name, self.context_manager)
        return self._loggers[name]

    async def start_system_monitoring(self, interval: float = 30.0):
        """Start system metrics collection"""
        async def collect_metrics():
            while True:
                try:
                    self.metrics_collector.collect_system_metrics()
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in system metrics collection: {e}")
                    await asyncio.sleep(5)

        self._system_metrics_task = asyncio.create_task(collect_metrics())

    async def stop_system_monitoring(self):
        """Stop system metrics collection"""
        if self._system_metrics_task:
            self._system_metrics_task.cancel()
            try:
                await self._system_metrics_task
            except asyncio.CancelledError:
                pass

    def get_observability_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary"""
        return {
            'service_name': self.service_name,
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': self.metrics_collector.get_metrics_summary(),
            'active_spans': len(self.trace_manager._active_spans),
            'completed_spans': len(self.trace_manager._completed_spans),
            'loggers': list(self._loggers.keys()),
            'context_count': len(self.context_manager._contexts)
        }


# Global observability manager
observability = ObservabilityManager()


# Decorators
def trace_operation(operation_name: str = None, tags: Dict[str, str] = None):
    """Decorator to trace function execution"""
    def decorator(func):
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with observability.trace_manager.trace_operation(op_name, tags=tags) as span:
                observability.trace_manager.add_span_log(
                    span.span_id, f"Starting {op_name}",
                    {'args': str(args), 'kwargs': str(kwargs)}
                )
                result = await func(*args, **kwargs)
                observability.trace_manager.add_span_log(
                    span.span_id, f"Completed {op_name}"
                )
                return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span = observability.trace_manager.start_span(op_name, tags=tags)
            try:
                observability.trace_manager.add_span_log(
                    span.span_id, f"Starting {op_name}",
                    {'args': str(args), 'kwargs': str(kwargs)}
                )
                result = func(*args, **kwargs)
                observability.trace_manager.add_span_log(
                    span.span_id, f"Completed {op_name}"
                )
                observability.trace_manager.finish_span(span.span_id, "success")
                return result
            except Exception as e:
                observability.trace_manager.finish_span(span.span_id, "error", tags={'error': str(e)})
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def measure_time(metric_name: str = None, tags: Dict[str, str] = None):
    """Decorator to measure function execution time"""
    def decorator(func):
        name = metric_name or f"function.{func.__name__}.duration"

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with observability.metrics_collector.time_operation(name, tags):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with observability.metrics_collector.time_operation(name, tags):
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def log_calls(logger_name: str = None, level: LogLevel = LogLevel.INFO):
    """Decorator to log function calls"""
    def decorator(func):
        logger_inst = observability.get_logger(logger_name or func.__module__)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger_inst._log(
                level, f"Calling {func.__name__}",
                extra_data={'args': str(args), 'kwargs': str(kwargs)}
            )
            try:
                result = await func(*args, **kwargs)
                logger_inst._log(level, f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger_inst.error(f"Error in {func.__name__}: {str(e)}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger_inst._log(
                level, f"Calling {func.__name__}",
                extra_data={'args': str(args), 'kwargs': str(kwargs)}
            )
            try:
                result = func(*args, **kwargs)
                logger_inst._log(level, f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger_inst.error(f"Error in {func.__name__}: {str(e)}")
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Utility functions
def get_logger(name: str = None) -> StructuredLogger:
    """Get structured logger instance"""
    return observability.get_logger(name or __name__)


def create_correlation_context(operation_name: str = "", user_id: str = None) -> CorrelationContext:
    """Create new correlation context"""
    return observability.context_manager.create_context(operation_name, user_id)


async def get_observability_status() -> Dict[str, Any]:
    """Get current observability status"""
    return observability.get_observability_summary()


def record_metric(name: str, value: Union[int, float], metric_type: str = "gauge",
                 tags: Dict[str, str] = None):
    """Record metric value"""
    if metric_type == "counter":
        observability.metrics_collector.counter(name, int(value), tags)
    elif metric_type == "gauge":
        observability.metrics_collector.gauge(name, float(value), tags)
    elif metric_type == "histogram":
        observability.metrics_collector.histogram(name, float(value), tags)
    elif metric_type == "timer":
        observability.metrics_collector.timer(name, float(value), tags)


# Initialize system monitoring
async def start_observability():
    """Start observability system"""
    await observability.start_system_monitoring()


async def stop_observability():
    """Stop observability system"""
    await observability.stop_system_monitoring()