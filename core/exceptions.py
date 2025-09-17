"""
Custom exceptions for Azure AI IT Copilot
Provides specific error types for better error handling
"""


class CopilotException(Exception):
    """Base exception for Azure AI IT Copilot"""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert exception to dictionary"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class AgentExecutionError(CopilotException):
    """Exception raised when agent execution fails"""

    def __init__(self, message: str, agent_type: str = None, plan_id: str = None):
        super().__init__(message, "AGENT_EXECUTION_FAILED")
        self.agent_type = agent_type
        self.plan_id = plan_id
        self.details.update({
            "agent_type": agent_type,
            "plan_id": plan_id
        })


class ConfigurationError(CopilotException):
    """Exception raised for configuration-related errors"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key
        self.details.update({
            "config_key": config_key
        })


class AzureClientError(CopilotException):
    """Exception raised for Azure client errors"""

    def __init__(self, message: str, service: str = None, operation: str = None):
        super().__init__(message, "AZURE_CLIENT_ERROR")
        self.service = service
        self.operation = operation
        self.details.update({
            "service": service,
            "operation": operation
        })


class AuthenticationError(CopilotException):
    """Exception raised for authentication failures"""

    def __init__(self, message: str, auth_type: str = None):
        super().__init__(message, "AUTHENTICATION_ERROR")
        self.auth_type = auth_type
        self.details.update({
            "auth_type": auth_type
        })


class ValidationError(CopilotException):
    """Exception raised for validation failures"""

    def __init__(self, message: str, field: str = None, value: str = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field
        self.value = value
        self.details.update({
            "field": field,
            "value": value
        })


class NotificationError(CopilotException):
    """Exception raised for notification failures"""

    def __init__(self, message: str, notification_type: str = None, recipient: str = None):
        super().__init__(message, "NOTIFICATION_ERROR")
        self.notification_type = notification_type
        self.recipient = recipient
        self.details.update({
            "notification_type": notification_type,
            "recipient": recipient
        })


class TaskExecutionError(CopilotException):
    """Exception raised for background task execution failures"""

    def __init__(self, message: str, task_name: str = None, task_id: str = None):
        super().__init__(message, "TASK_EXECUTION_ERROR")
        self.task_name = task_name
        self.task_id = task_id
        self.details.update({
            "task_name": task_name,
            "task_id": task_id
        })


class ResourceNotFoundError(CopilotException):
    """Exception raised when a resource is not found"""

    def __init__(self, message: str, resource_type: str = None, resource_id: str = None):
        super().__init__(message, "RESOURCE_NOT_FOUND")
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.details.update({
            "resource_type": resource_type,
            "resource_id": resource_id
        })


class QuotaExceededError(CopilotException):
    """Exception raised when Azure quota is exceeded"""

    def __init__(self, message: str, quota_type: str = None, current_usage: int = None, limit: int = None):
        super().__init__(message, "QUOTA_EXCEEDED")
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.limit = limit
        self.details.update({
            "quota_type": quota_type,
            "current_usage": current_usage,
            "limit": limit
        })


class CacheError(CopilotException):
    """Exception raised for cache-related errors"""

    def __init__(self, message: str, cache_key: str = None, operation: str = None):
        super().__init__(message, "CACHE_ERROR")
        self.cache_key = cache_key
        self.operation = operation
        self.details.update({
            "cache_key": cache_key,
            "operation": operation
        })


class IntegrationError(CopilotException):
    """Exception raised for external integration failures"""

    def __init__(self, message: str, integration: str = None, endpoint: str = None):
        super().__init__(message, "INTEGRATION_ERROR")
        self.integration = integration
        self.endpoint = endpoint
        self.details.update({
            "integration": integration,
            "endpoint": endpoint
        })


class SecurityError(CopilotException):
    """Exception raised for security-related issues"""

    def __init__(self, message: str, security_context: str = None):
        super().__init__(message, "SECURITY_ERROR")
        self.security_context = security_context
        self.details.update({
            "security_context": security_context
        })


class RateLimitError(CopilotException):
    """Exception raised when rate limits are exceeded"""

    def __init__(self, message: str, limit_type: str = None, retry_after: int = None):
        super().__init__(message, "RATE_LIMIT_EXCEEDED")
        self.limit_type = limit_type
        self.retry_after = retry_after
        self.details.update({
            "limit_type": limit_type,
            "retry_after": retry_after
        })