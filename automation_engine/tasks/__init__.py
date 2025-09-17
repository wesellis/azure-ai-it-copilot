"""
Celery tasks for Azure AI IT Copilot background processing
Optimized imports for better performance
"""

# Explicit imports for better performance and IDE support
from .monitoring import (
    monitor_azure_resources,
    compliance_scan,
    health_check
)

from .remediation import (
    auto_remediate_incident,
    scale_resources,
    restart_service
)

from .analytics import (
    detect_cost_anomalies,
    generate_cost_forecast,
    analyze_resource_utilization,
    generate_insights_report,
    trend_analysis
)

from .maintenance import (
    cleanup_unused_resources,
    optimize_storage_tiers,
    update_resource_tags,
    vm_rightsizing_analysis,
    database_maintenance
)

from .notifications import (
    send_alert,
    send_cost_report,
    send_security_notification,
    send_maintenance_notification,
    send_health_summary,
    process_webhook_notification
)

__all__ = [
    # Monitoring tasks
    "monitor_azure_resources",
    "compliance_scan",
    "health_check",

    # Remediation tasks
    "auto_remediate_incident",
    "scale_resources",
    "restart_service",

    # Analytics tasks
    "detect_cost_anomalies",
    "generate_cost_forecast",
    "analyze_resource_utilization",
    "generate_insights_report",
    "trend_analysis",

    # Maintenance tasks
    "cleanup_unused_resources",
    "optimize_storage_tiers",
    "update_resource_tags",
    "vm_rightsizing_analysis",
    "database_maintenance",

    # Notification tasks
    "send_alert",
    "send_cost_report",
    "send_security_notification",
    "send_maintenance_notification",
    "send_health_summary",
    "process_webhook_notification"
]