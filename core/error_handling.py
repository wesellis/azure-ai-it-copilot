"""
Advanced Error Handling and Circuit Breaker Implementation
Comprehensive error handling patterns with circuit breakers and resilience
"""

import asyncio
import time
import logging
import traceback
from typing import Dict, Any, List, Optional, Callable, Union, Type, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import json
import threading
from collections import deque, defaultdict
import hashlib
import weakref

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(str, Enum):
    """Retry strategy types"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    FIXED = "fixed"


@dataclass
class ErrorContext:
    """Comprehensive error context information"""
    error_id: str
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    service_name: str = ""
    operation_name: str = ""
    retry_count: int = 0
    is_retryable: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    request_timeout: float = 30.0   # seconds
    success_threshold: int = 3      # successes needed to close circuit
    half_open_max_requests: int = 5
    error_threshold_percentage: float = 50.0
    minimum_requests: int = 10      # minimum requests before circuit can open


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    request_count: int = 0
    last_failure_time: Optional[datetime] = None
    state_changed_time: datetime = field(default_factory=datetime.utcnow)
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    avg_response_time: float = 0.0


class CircuitBreaker:
    """Advanced circuit breaker implementation"""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        self._request_times: deque = deque(maxlen=100)
        self._half_open_request_count = 0

    async def __aenter__(self):
        """Async context manager entry"""
        await self._check_circuit_state()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if exc_type is None:
            await self._record_success()
        else:
            await self._record_failure()

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self._lock:
            await self._check_circuit_state()

        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.request_timeout
                )
            else:
                result = func(*args, **kwargs)

            execution_time = time.time() - start_time
            await self._record_success(execution_time)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            await self._record_failure(execution_time)
            raise

    async def _check_circuit_state(self):
        """Check and update circuit breaker state"""
        current_time = datetime.utcnow()

        if self.stats.state == CircuitState.OPEN:
            time_since_failure = (current_time - self.stats.last_failure_time).total_seconds()
            if time_since_failure >= self.config.recovery_timeout:
                await self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Next retry in {self.config.recovery_timeout - time_since_failure:.1f}s"
                )

        elif self.stats.state == CircuitState.HALF_OPEN:
            if self._half_open_request_count >= self.config.half_open_max_requests:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is HALF_OPEN with max requests reached"
                )

    async def _record_success(self, execution_time: float = 0.0):
        """Record successful operation"""
        async with self._lock:
            self.stats.success_count += 1
            self.stats.total_successes += 1
            self.stats.request_count += 1
            self.stats.total_requests += 1

            # Update average response time
            self._request_times.append(execution_time)
            if self._request_times:
                self.stats.avg_response_time = sum(self._request_times) / len(self._request_times)

            if self.stats.state == CircuitState.HALF_OPEN:
                self._half_open_request_count += 1
                if self.stats.success_count >= self.config.success_threshold:
                    await self._transition_to_closed()

    async def _record_failure(self, execution_time: float = 0.0):
        """Record failed operation"""
        async with self._lock:
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.request_count += 1
            self.stats.total_requests += 1
            self.stats.last_failure_time = datetime.utcnow()

            self._request_times.append(execution_time)
            if self._request_times:
                self.stats.avg_response_time = sum(self._request_times) / len(self._request_times)

            # Check if circuit should open
            if self.stats.state == CircuitState.CLOSED:
                if self._should_open_circuit():
                    await self._transition_to_open()
            elif self.stats.state == CircuitState.HALF_OPEN:
                await self._transition_to_open()

    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on failure rate"""
        if self.stats.request_count < self.config.minimum_requests:
            return False

        failure_rate = (self.stats.failure_count / self.stats.request_count) * 100
        return (
            self.stats.failure_count >= self.config.failure_threshold or
            failure_rate >= self.config.error_threshold_percentage
        )

    async def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        self.stats.state = CircuitState.OPEN
        self.stats.state_changed_time = datetime.utcnow()
        self._reset_counters()
        logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN")

    async def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        self.stats.state = CircuitState.HALF_OPEN
        self.stats.state_changed_time = datetime.utcnow()
        self._half_open_request_count = 0
        self._reset_counters()
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")

    async def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        self.stats.state = CircuitState.CLOSED
        self.stats.state_changed_time = datetime.utcnow()
        self._half_open_request_count = 0
        self._reset_counters()
        logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")

    def _reset_counters(self):
        """Reset failure and success counters"""
        self.stats.failure_count = 0
        self.stats.success_count = 0
        self.stats.request_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'name': self.name,
            'state': self.stats.state.value,
            'failure_count': self.stats.failure_count,
            'success_count': self.stats.success_count,
            'request_count': self.stats.request_count,
            'total_requests': self.stats.total_requests,
            'total_failures': self.stats.total_failures,
            'total_successes': self.stats.total_successes,
            'failure_rate': (self.stats.total_failures / self.stats.total_requests * 100)
                           if self.stats.total_requests > 0 else 0,
            'avg_response_time': self.stats.avg_response_time,
            'last_failure_time': self.stats.last_failure_time.isoformat()
                                if self.stats.last_failure_time else None,
            'state_changed_time': self.stats.state_changed_time.isoformat(),
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'request_timeout': self.config.request_timeout,
                'success_threshold': self.config.success_threshold
            }
        }


class RetryHandler:
    """Advanced retry handler with multiple strategies"""

    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()

    async def execute_with_retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    raise

                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break

                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        raise RetryExhaustedError(
            f"All {self.config.max_attempts} retry attempts failed. "
            f"Last error: {str(last_exception)}"
        ) from last_exception

    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        # Check non-retryable exceptions first
        for exc_type in self.config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False

        # If retryable exceptions specified, check those
        if self.config.retryable_exceptions:
            for exc_type in self.config.retryable_exceptions:
                if isinstance(exception, exc_type):
                    return True
            return False

        # Default retryable exceptions
        non_retryable = (
            ValueError, TypeError, AttributeError,
            KeyError, IndexError, PermissionError
        )
        return not isinstance(exception, non_retryable)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy"""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay

        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)

        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)

        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._fibonacci(attempt + 1)

        else:
            delay = self.config.base_delay

        # Apply jitter if enabled
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)

        return min(delay, self.config.max_delay)

    def _fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number"""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)


class ErrorTracker:
    """Comprehensive error tracking and analysis"""

    def __init__(self, max_errors: int = 10000):
        self.max_errors = max_errors
        self._errors: deque = deque(maxlen=max_errors)
        self._error_counts: defaultdict = defaultdict(int)
        self._error_patterns: defaultdict = defaultdict(list)
        self._lock = threading.Lock()

    def record_error(self, error_context: ErrorContext):
        """Record error with context"""
        with self._lock:
            self._errors.append(error_context)
            self._error_counts[error_context.error_type] += 1

            # Track error patterns
            pattern_key = f"{error_context.service_name}:{error_context.operation_name}"
            self._error_patterns[pattern_key].append(error_context)

            # Keep only last 100 errors per pattern
            if len(self._error_patterns[pattern_key]) > 100:
                self._error_patterns[pattern_key] = self._error_patterns[pattern_key][-100:]

    def get_error_summary(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Get error summary for specified time window"""
        if time_window is None:
            time_window = timedelta(hours=1)

        cutoff_time = datetime.utcnow() - time_window
        recent_errors = [e for e in self._errors if e.timestamp >= cutoff_time]

        if not recent_errors:
            return {
                'total_errors': 0,
                'time_window_hours': time_window.total_seconds() / 3600
            }

        # Count by type
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        service_errors = defaultdict(int)

        for error in recent_errors:
            error_types[error.error_type] += 1
            severity_counts[error.severity.value] += 1
            service_errors[error.service_name] += 1

        return {
            'total_errors': len(recent_errors),
            'time_window_hours': time_window.total_seconds() / 3600,
            'error_rate_per_hour': len(recent_errors) / (time_window.total_seconds() / 3600),
            'by_type': dict(error_types),
            'by_severity': dict(severity_counts),
            'by_service': dict(service_errors),
            'top_errors': sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]
        }

    def get_error_trends(self) -> Dict[str, Any]:
        """Analyze error trends over time"""
        if len(self._errors) < 2:
            return {'status': 'insufficient_data'}

        # Compare last hour vs previous hour
        now = datetime.utcnow()
        last_hour = [e for e in self._errors if now - timedelta(hours=1) <= e.timestamp <= now]
        prev_hour = [e for e in self._errors if now - timedelta(hours=2) <= e.timestamp <= now - timedelta(hours=1)]

        if not last_hour and not prev_hour:
            return {'status': 'no_recent_errors'}

        last_hour_count = len(last_hour)
        prev_hour_count = len(prev_hour)

        if prev_hour_count == 0:
            trend = 'new_errors' if last_hour_count > 0 else 'stable'
            change_percent = 100 if last_hour_count > 0 else 0
        else:
            change_percent = ((last_hour_count - prev_hour_count) / prev_hour_count) * 100
            if change_percent > 10:
                trend = 'increasing'
            elif change_percent < -10:
                trend = 'decreasing'
            else:
                trend = 'stable'

        return {
            'status': 'analyzed',
            'trend': trend,
            'change_percent': change_percent,
            'last_hour_errors': last_hour_count,
            'prev_hour_errors': prev_hour_count
        }


class ErrorManager:
    """Central error management system"""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_tracker = ErrorTracker()
        self.retry_handler = RetryHandler()
        self._error_callbacks: List[Callable] = []
        self._correlation_map: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]

    def register_error_callback(self, callback: Callable):
        """Register callback for error notifications"""
        self._error_callbacks.append(callback)

    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Handle error with comprehensive context"""
        error_context = self._create_error_context(error, context or {})
        self.error_tracker.record_error(error_context)

        # Trigger callbacks
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_context)
                else:
                    callback(error_context)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

        return error_context

    def _create_error_context(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Create comprehensive error context"""
        error_id = hashlib.sha256(
            f"{type(error).__name__}{str(error)}{time.time()}".encode()
        ).hexdigest()[:16]

        severity = self._determine_severity(error)

        return ErrorContext(
            error_id=error_id,
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            stack_trace=traceback.format_exc(),
            context_data=context,
            correlation_id=context.get('correlation_id'),
            user_id=context.get('user_id'),
            request_id=context.get('request_id'),
            service_name=context.get('service_name', ''),
            operation_name=context.get('operation_name', ''),
            retry_count=context.get('retry_count', 0)
        )

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type"""
        critical_exceptions = (
            SystemExit, KeyboardInterrupt, MemoryError,
            ConnectionError, OSError
        )

        high_severity_exceptions = (
            RuntimeError, IOError, TimeoutError,
            PermissionError, FileNotFoundError
        )

        if isinstance(error, critical_exceptions):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, high_severity_exceptions):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on errors and circuit breakers"""
        circuit_stats = {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}
        error_summary = self.error_tracker.get_error_summary()
        error_trends = self.error_tracker.get_error_trends()

        # Calculate health score
        health_score = 100
        open_circuits = sum(1 for cb in self.circuit_breakers.values()
                           if cb.stats.state == CircuitState.OPEN)
        health_score -= open_circuits * 20

        critical_errors = error_summary.get('by_severity', {}).get('critical', 0)
        health_score -= critical_errors * 10

        return {
            'health_score': max(health_score, 0),
            'circuit_breakers': circuit_stats,
            'error_summary': error_summary,
            'error_trends': error_trends,
            'timestamp': datetime.utcnow().isoformat()
        }


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted"""
    pass


# Global error manager instance
error_manager = ErrorManager()


# Decorators
def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator to add circuit breaker protection"""
    def decorator(func):
        cb = error_manager.get_circuit_breaker(name, config)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(cb.call(func, *args, **kwargs))

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def with_retry(config: RetryConfig = None):
    """Decorator to add retry logic"""
    retry_handler = RetryHandler(config or RetryConfig())

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await retry_handler.execute_with_retry(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                retry_handler.execute_with_retry(func, *args, **kwargs)
            )

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def error_handler(service_name: str = "", operation_name: str = ""):
    """Decorator to automatically handle and track errors"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    'service_name': service_name,
                    'operation_name': operation_name or func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                await error_manager.handle_error(e, context)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'service_name': service_name,
                    'operation_name': operation_name or func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                loop = asyncio.get_event_loop()
                loop.run_until_complete(error_manager.handle_error(e, context))
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Utility functions
async def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health"""
    return error_manager.get_system_health()


async def get_error_summary(hours: int = 1) -> Dict[str, Any]:
    """Get error summary for specified hours"""
    return error_manager.error_tracker.get_error_summary(timedelta(hours=hours))


def register_error_callback(callback: Callable):
    """Register global error callback"""
    error_manager.register_error_callback(callback)