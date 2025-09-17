"""
Celery tasks for Azure AI IT Copilot background processing
"""

from .monitoring import *
from .remediation import *
from .analytics import *
from .maintenance import *
from .notifications import *

__all__ = [
    # Monitoring tasks
    'monitor_azure_resources',
    'compliance_scan',
    'health_check',

    # Remediation tasks
    'auto_remediate_incident',
    'scale_resources',
    'restart_service',

    # Analytics tasks
    'detect_cost_anomalies',
    'run_predictive_analysis',
    'generate_insights',

    # Maintenance tasks
    'cleanup_old_logs',
    'update_ml_models',
    'backup_configuration',

    # Notification tasks
    'send_alert',
    'send_critical_alert',
    'send_daily_report',
]