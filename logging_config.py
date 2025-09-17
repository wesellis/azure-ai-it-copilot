"""
Centralized logging configuration for Azure AI IT Copilot
Provides structured logging with proper formatting and handlers
"""

import logging
import logging.config
import sys
import os
from typing import Dict, Any
from datetime import datetime
import json

from config.settings import get_settings


class CustomJSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record):
        """Format log record as JSON"""
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_obj[key] = value

        return json.dumps(log_obj)


def setup_logging() -> None:
    """Configure logging for the application"""
    settings = get_settings()

    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Determine log format based on environment
    if settings.environment.value == "production":
        formatter_class = CustomJSONFormatter
        formatter_format = None
    else:
        formatter_class = logging.Formatter
        formatter_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'class': 'logging.Formatter' if not settings.is_production() else 'logging_config.CustomJSONFormatter',
                'format': formatter_format if formatter_format else '%(message)s'
            },
            'detailed': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
            },
            'json': {
                'class': 'logging_config.CustomJSONFormatter'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': settings.log_level.value,
                'formatter': 'default',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json' if settings.is_production() else 'detailed',
                'filename': f'{log_dir}/ai-copilot.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json' if settings.is_production() else 'detailed',
                'filename': f'{log_dir}/ai-copilot-errors.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            }
        },
        'loggers': {
            'ai_orchestrator': {
                'level': settings.log_level.value,
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'api': {
                'level': settings.log_level.value,
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'azure_clients': {
                'level': settings.log_level.value,
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'integrations': {
                'level': settings.log_level.value,
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'automation_engine': {
                'level': settings.log_level.value,
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'ml_models': {
                'level': settings.log_level.value,
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'INFO',
                'handlers': ['file'],
                'propagate': False
            },
            'azure': {
                'level': 'WARNING',
                'handlers': ['file'],
                'propagate': False
            },
            'azure.core': {
                'level': 'WARNING',
                'handlers': ['file'],
                'propagate': False
            }
        },
        'root': {
            'level': settings.log_level.value,
            'handlers': ['console', 'file']
        }
    }

    # Apply configuration
    logging.config.dictConfig(config)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured for {settings.environment.value} environment")
    logger.info(f"Log level: {settings.log_level.value}")
    logger.info(f"Log directory: {log_dir}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with proper configuration"""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class"""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


class StructuredLogger:
    """Structured logger for consistent logging across the application"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        extra = kwargs
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        extra = kwargs
        self.logger.warning(message, extra=extra)

    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with structured data"""
        extra = kwargs
        if error:
            extra['error_type'] = type(error).__name__
            extra['error_message'] = str(error)
        self.logger.error(message, extra=extra, exc_info=error is not None)

    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        extra = kwargs
        self.logger.debug(message, extra=extra)

    def critical(self, message: str, **kwargs):
        """Log critical message with structured data"""
        extra = kwargs
        self.logger.critical(message, extra=extra)


def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}", exc_info=True)
            raise
    return wrapper


def log_async_function_call(func):
    """Decorator to log async function calls"""
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling async {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Async {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Async {func.__name__} failed with error: {str(e)}", exc_info=True)
            raise
    return wrapper


def get_request_logger(request_id: str) -> StructuredLogger:
    """Get a logger with request context"""
    logger = StructuredLogger("request")
    # Add request ID to all log messages
    logger.request_id = request_id
    return logger