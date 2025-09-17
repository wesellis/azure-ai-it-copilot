"""
Enterprise Audit Logging System for Compliance
Comprehensive logging of all user actions, API calls, and system events
SOC2, HIPAA, and ISO 27001 compliant
"""

import asyncio
import json
import time
import hashlib
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
import aiofiles

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_CALL = "api_call"
    AI_REQUEST = "ai_request"
    AZURE_OPERATION = "azure_operation"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    ERROR_EVENT = "error_event"
    COMPLIANCE_CHECK = "compliance_check"
    RATE_LIMIT_EVENT = "rate_limit_event"
    ADMIN_ACTION = "admin_action"


class EventSeverity(Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStandard(Enum):
    """Compliance standards"""
    SOC2 = "soc2"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"


@dataclass
class AuditEvent:
    """Comprehensive audit event structure"""

    # Core event information
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: EventType = EventType.API_CALL
    severity: EventSeverity = EventSeverity.LOW

    # User and session information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    user_role: Optional[str] = None
    user_ip: Optional[str] = None
    user_agent: Optional[str] = None

    # Request information
    endpoint: Optional[str] = None
    method: Optional[str] = None
    request_id: Optional[str] = None

    # Event details
    action: str = ""
    resource: Optional[str] = None
    resource_type: Optional[str] = None
    result: str = "unknown"  # success, failure, error

    # Data and parameters (sanitized)
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Compliance and security
    compliance_tags: List[ComplianceStandard] = field(default_factory=list)
    sensitive_data_accessed: bool = False
    data_classification: Optional[str] = None

    # Performance information
    duration_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost_cents: Optional[int] = None

    # Error information
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

    # Integrity
    event_hash: Optional[str] = field(default=None)

    def __post_init__(self):
        """Generate event hash for integrity verification"""
        if not self.event_hash:
            self.event_hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate SHA-256 hash of core event data for integrity"""
        core_data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "action": self.action,
            "result": self.result
        }
        data_str = json.dumps(core_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify event integrity using hash"""
        expected_hash = self._generate_hash()
        return self.event_hash == expected_hash


class AuditLogger:
    """Enterprise audit logging system"""

    def __init__(self,
                 log_directory: str = "audit_logs",
                 max_file_size_mb: int = 100,
                 retention_days: int = 2555):  # 7 years for compliance
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)

        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.retention_days = retention_days

        self._current_file: Optional[Path] = None
        self._file_handle = None
        self._lock = asyncio.Lock()

        # In-memory event buffer for high-frequency logging
        self._event_buffer: List[AuditEvent] = []
        self._buffer_size = 100

        # Compliance mapping
        self._compliance_mappings = {
            ComplianceStandard.SOC2: {
                "required_fields": ["user_id", "timestamp", "action", "result"],
                "retention_days": 365
            },
            ComplianceStandard.HIPAA: {
                "required_fields": ["user_id", "timestamp", "action", "resource", "result"],
                "retention_days": 2555  # 7 years
            },
            ComplianceStandard.ISO27001: {
                "required_fields": ["user_id", "timestamp", "action", "result", "user_ip"],
                "retention_days": 2555
            }
        }

    async def log_event(self, event: AuditEvent):
        """Log an audit event"""
        async with self._lock:
            # Sanitize sensitive data
            event = self._sanitize_event(event)

            # Add to buffer
            self._event_buffer.append(event)

            # Flush buffer if full
            if len(self._event_buffer) >= self._buffer_size:
                await self._flush_buffer()

    async def log_user_action(self,
                             user_id: str,
                             action: str,
                             resource: Optional[str] = None,
                             result: str = "success",
                             metadata: Dict[str, Any] = None,
                             severity: EventSeverity = EventSeverity.MEDIUM,
                             compliance_tags: List[ComplianceStandard] = None):
        """Log a user action with compliance tagging"""

        event = AuditEvent(
            event_type=EventType.USER_LOGIN if "login" in action.lower() else EventType.API_CALL,
            severity=severity,
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            metadata=metadata or {},
            compliance_tags=compliance_tags or [ComplianceStandard.SOC2]
        )

        await self.log_event(event)

    async def log_ai_request(self,
                            user_id: str,
                            prompt: str,
                            response: str,
                            tokens_used: int,
                            cost_cents: int,
                            duration_ms: float,
                            metadata: Dict[str, Any] = None):
        """Log AI/LLM requests with token usage and cost tracking"""

        # Sanitize prompt and response
        sanitized_prompt = self._sanitize_ai_content(prompt)
        sanitized_response = self._sanitize_ai_content(response)

        event = AuditEvent(
            event_type=EventType.AI_REQUEST,
            severity=EventSeverity.MEDIUM,
            user_id=user_id,
            action="ai_request",
            request_data={"prompt_length": len(prompt), "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8]},
            response_data={"response_length": len(response), "response_hash": hashlib.md5(response.encode()).hexdigest()[:8]},
            tokens_used=tokens_used,
            cost_cents=cost_cents,
            duration_ms=duration_ms,
            metadata=metadata or {},
            compliance_tags=[ComplianceStandard.SOC2, ComplianceStandard.ISO27001]
        )

        await self.log_event(event)

    async def log_azure_operation(self,
                                 user_id: str,
                                 operation: str,
                                 resource_group: str,
                                 resource_name: str,
                                 result: str,
                                 metadata: Dict[str, Any] = None):
        """Log Azure infrastructure operations"""

        event = AuditEvent(
            event_type=EventType.AZURE_OPERATION,
            severity=EventSeverity.HIGH,
            user_id=user_id,
            action=operation,
            resource=f"{resource_group}/{resource_name}",
            resource_type="azure_resource",
            result=result,
            metadata=metadata or {},
            compliance_tags=[ComplianceStandard.SOC2, ComplianceStandard.ISO27001]
        )

        await self.log_event(event)

    async def log_security_event(self,
                                user_id: Optional[str],
                                event_description: str,
                                severity: EventSeverity,
                                user_ip: Optional[str] = None,
                                metadata: Dict[str, Any] = None):
        """Log security-related events"""

        event = AuditEvent(
            event_type=EventType.SECURITY_EVENT,
            severity=severity,
            user_id=user_id,
            user_ip=user_ip,
            action=event_description,
            result="detected",
            metadata=metadata or {},
            compliance_tags=[ComplianceStandard.SOC2, ComplianceStandard.ISO27001, ComplianceStandard.HIPAA]
        )

        await self.log_event(event)

    async def log_error(self,
                       user_id: Optional[str],
                       error_message: str,
                       error_code: str,
                       stack_trace: Optional[str] = None,
                       metadata: Dict[str, Any] = None):
        """Log error events"""

        event = AuditEvent(
            event_type=EventType.ERROR_EVENT,
            severity=EventSeverity.MEDIUM,
            user_id=user_id,
            action="error_occurred",
            result="error",
            error_code=error_code,
            error_message=error_message,
            stack_trace=stack_trace,
            metadata=metadata or {}
        )

        await self.log_event(event)

    def _sanitize_event(self, event: AuditEvent) -> AuditEvent:
        """Sanitize event data to remove sensitive information"""

        # List of fields that might contain sensitive data
        sensitive_patterns = [
            "password", "token", "secret", "key", "credential",
            "ssn", "social_security", "credit_card", "bank_account"
        ]

        # Sanitize request and response data
        if event.request_data:
            event.request_data = self._sanitize_dict(event.request_data, sensitive_patterns)

        if event.response_data:
            event.response_data = self._sanitize_dict(event.response_data, sensitive_patterns)

        if event.metadata:
            event.metadata = self._sanitize_dict(event.metadata, sensitive_patterns)

        # Truncate long error messages
        if event.error_message and len(event.error_message) > 1000:
            event.error_message = event.error_message[:1000] + "... [truncated]"

        # Remove stack traces in production
        if event.stack_trace and len(event.stack_trace) > 5000:
            event.stack_trace = event.stack_trace[:5000] + "... [truncated]"

        return event

    def _sanitize_dict(self, data: Dict[str, Any], sensitive_patterns: List[str]) -> Dict[str, Any]:
        """Sanitize dictionary data"""
        sanitized = {}

        for key, value in data.items():
            key_lower = key.lower()

            # Check if key contains sensitive pattern
            is_sensitive = any(pattern in key_lower for pattern in sensitive_patterns)

            if is_sensitive:
                if isinstance(value, str):
                    sanitized[key] = f"[REDACTED:{len(value)} chars]"
                else:
                    sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value, sensitive_patterns)
            else:
                sanitized[key] = value

        return sanitized

    def _sanitize_ai_content(self, content: str) -> str:
        """Sanitize AI prompt/response content"""
        # For audit purposes, we store hash and metadata, not full content
        return f"[CONTENT:{len(content)} chars, hash:{hashlib.md5(content.encode()).hexdigest()[:8]}]"

    async def _flush_buffer(self):
        """Flush event buffer to disk"""
        if not self._event_buffer:
            return

        # Get current log file
        log_file = await self._get_current_log_file()

        # Write events
        async with aiofiles.open(log_file, 'a', encoding='utf-8') as f:
            for event in self._event_buffer:
                event_json = json.dumps(asdict(event), default=str)
                await f.write(event_json + '\n')

        # Clear buffer
        self._event_buffer.clear()

        logger.info(f"Flushed {len(self._event_buffer)} audit events to {log_file}")

    async def _get_current_log_file(self) -> Path:
        """Get current log file, rotating if necessary"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_directory / f"audit_{today}.jsonl"

        # Check if we need to rotate based on file size
        if log_file.exists() and log_file.stat().st_size > self.max_file_size:
            # Create new file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_directory / f"audit_{today}_{timestamp}.jsonl"

        return log_file

    async def search_events(self,
                           user_id: Optional[str] = None,
                           event_type: Optional[EventType] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: int = 1000) -> List[AuditEvent]:
        """Search audit events with filters"""

        events = []

        # Get all log files in date range
        log_files = self._get_log_files_in_range(start_date, end_date)

        for log_file in log_files:
            if not log_file.exists():
                continue

            async with aiofiles.open(log_file, 'r', encoding='utf-8') as f:
                async for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        event = AuditEvent(**event_data)

                        # Apply filters
                        if user_id and event.user_id != user_id:
                            continue

                        if event_type and event.event_type != event_type:
                            continue

                        if start_date:
                            event_time = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
                            if event_time < start_date:
                                continue

                        if end_date:
                            event_time = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
                            if event_time > end_date:
                                continue

                        events.append(event)

                        if len(events) >= limit:
                            break

                    except json.JSONDecodeError:
                        continue

            if len(events) >= limit:
                break

        return events[:limit]

    def _get_log_files_in_range(self,
                               start_date: Optional[datetime],
                               end_date: Optional[datetime]) -> List[Path]:
        """Get log files within date range"""

        if not start_date:
            start_date = datetime.now() - timedelta(days=30)

        if not end_date:
            end_date = datetime.now()

        log_files = []
        current_date = start_date.date()

        while current_date <= end_date.date():
            pattern = f"audit_{current_date.strftime('%Y-%m-%d')}*.jsonl"
            matching_files = list(self.log_directory.glob(pattern))
            log_files.extend(matching_files)
            current_date += timedelta(days=1)

        return sorted(log_files)

    async def generate_compliance_report(self,
                                       standard: ComplianceStandard,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specified standard"""

        # Get events with compliance tags
        events = await self.search_events(start_date=start_date, end_date=end_date)
        compliance_events = [e for e in events if standard in e.compliance_tags]

        # Generate report
        report = {
            "compliance_standard": standard.value,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(compliance_events),
            "event_breakdown": {},
            "user_activity": {},
            "security_events": [],
            "integrity_check": {"passed": 0, "failed": 0}
        }

        # Analyze events
        for event in compliance_events:
            # Event type breakdown
            event_type = event.event_type.value
            if event_type not in report["event_breakdown"]:
                report["event_breakdown"][event_type] = 0
            report["event_breakdown"][event_type] += 1

            # User activity
            if event.user_id:
                if event.user_id not in report["user_activity"]:
                    report["user_activity"][event.user_id] = 0
                report["user_activity"][event.user_id] += 1

            # Security events
            if event.event_type == EventType.SECURITY_EVENT:
                report["security_events"].append({
                    "timestamp": event.timestamp,
                    "severity": event.severity.value,
                    "action": event.action,
                    "user_id": event.user_id
                })

            # Integrity check
            if event.verify_integrity():
                report["integrity_check"]["passed"] += 1
            else:
                report["integrity_check"]["failed"] += 1

        return report

    async def cleanup_old_logs(self):
        """Clean up old log files based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for log_file in self.log_directory.glob("audit_*.jsonl"):
            try:
                file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info(f"Deleted old audit log: {log_file}")
            except Exception as e:
                logger.error(f"Error deleting old log file {log_file}: {e}")


# Global audit logger instance
audit_logger = AuditLogger()


# Decorator for automatic audit logging
def audit_api_call(action: str,
                  resource_type: Optional[str] = None,
                  compliance_tags: List[ComplianceStandard] = None):
    """Decorator for automatic API call audit logging"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            user_id = kwargs.get('user_id', 'system')
            request_id = kwargs.get('request_id', str(uuid.uuid4()))

            try:
                result = await func(*args, **kwargs)

                duration_ms = (time.time() - start_time) * 1000

                await audit_logger.log_user_action(
                    user_id=user_id,
                    action=action,
                    resource=resource_type,
                    result="success",
                    metadata={
                        "function": f"{func.__module__}.{func.__name__}",
                        "request_id": request_id,
                        "duration_ms": duration_ms
                    },
                    compliance_tags=compliance_tags or [ComplianceStandard.SOC2]
                )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                await audit_logger.log_error(
                    user_id=user_id,
                    error_message=str(e),
                    error_code=type(e).__name__,
                    metadata={
                        "function": f"{func.__module__}.{func.__name__}",
                        "request_id": request_id,
                        "duration_ms": duration_ms
                    }
                )

                raise

        return wrapper
    return decorator