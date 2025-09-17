"""
Celery application and task configuration for Azure AI IT Copilot
Handles background processing, scheduled tasks, and async operations
"""

import os
from celery import Celery
from celery.schedules import crontab
from kombu import Queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
app = Celery('azure_ai_copilot')

# Configuration
app.conf.update(
    # Broker settings
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2'),

    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task routing
    task_routes={
        'automation_engine.tasks.monitoring.*': {'queue': 'monitoring'},
        'automation_engine.tasks.remediation.*': {'queue': 'remediation'},
        'automation_engine.tasks.analytics.*': {'queue': 'analytics'},
        'automation_engine.tasks.maintenance.*': {'queue': 'maintenance'},
    },

    # Queue definitions
    task_default_queue='default',
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('monitoring', routing_key='monitoring'),
        Queue('remediation', routing_key='remediation'),
        Queue('analytics', routing_key='analytics'),
        Queue('maintenance', routing_key='maintenance'),
        Queue('high_priority', routing_key='high_priority'),
    ),

    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,

    # Task execution settings
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes
    task_reject_on_worker_lost=True,

    # Result settings
    result_expires=3600,  # 1 hour

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Security
    worker_hijack_root_logger=False,
    worker_log_color=False,
)

# Scheduled tasks configuration
app.conf.beat_schedule = {
    # System monitoring tasks
    'monitor-azure-resources': {
        'task': 'automation_engine.tasks.monitoring.monitor_azure_resources',
        'schedule': crontab(minute='*/5'),  # Every 5 minutes
        'options': {'queue': 'monitoring'}
    },

    'check-cost-anomalies': {
        'task': 'automation_engine.tasks.analytics.detect_cost_anomalies',
        'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
        'options': {'queue': 'analytics'}
    },

    'compliance-scan': {
        'task': 'automation_engine.tasks.monitoring.compliance_scan',
        'schedule': crontab(minute=0, hour=2),  # Daily at 2 AM
        'options': {'queue': 'monitoring'}
    },

    'predictive-analysis': {
        'task': 'automation_engine.tasks.analytics.run_predictive_analysis',
        'schedule': crontab(minute=0, hour='*/12'),  # Every 12 hours
        'options': {'queue': 'analytics'}
    },

    # Maintenance tasks
    'cleanup-old-logs': {
        'task': 'automation_engine.tasks.maintenance.cleanup_old_logs',
        'schedule': crontab(minute=0, hour=1),  # Daily at 1 AM
        'options': {'queue': 'maintenance'}
    },

    'update-ml-models': {
        'task': 'automation_engine.tasks.maintenance.update_ml_models',
        'schedule': crontab(minute=0, hour=3, day_of_week=0),  # Weekly on Sunday at 3 AM
        'options': {'queue': 'maintenance'}
    },

    'backup-configuration': {
        'task': 'automation_engine.tasks.maintenance.backup_configuration',
        'schedule': crontab(minute=0, hour=4),  # Daily at 4 AM
        'options': {'queue': 'maintenance'}
    },

    # Health checks
    'health-check': {
        'task': 'automation_engine.tasks.monitoring.health_check',
        'schedule': crontab(minute='*/10'),  # Every 10 minutes
        'options': {'queue': 'monitoring'}
    },
}

# Auto-discover tasks
app.autodiscover_tasks([
    'automation_engine.tasks.monitoring',
    'automation_engine.tasks.remediation',
    'automation_engine.tasks.analytics',
    'automation_engine.tasks.maintenance',
    'automation_engine.tasks.notifications',
])

# Task failure handling
@app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery setup"""
    print(f'Request: {self.request!r}')
    return {'status': 'debug_task_completed', 'worker_id': self.request.id}

# Error handling
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None):
    """Handle task failures"""
    logger.error(f"Task {task_id} failed: {exception}")
    logger.error(f"Traceback: {traceback}")

    # Send notification for critical task failures
    if sender and hasattr(sender, 'request') and sender.request.get('queue') == 'high_priority':
        from automation_engine.tasks.notifications import send_critical_alert
        send_critical_alert.delay(
            title="Critical Task Failure",
            message=f"High priority task {task_id} failed: {exception}",
            task_id=task_id
        )

# Register failure handler
app.signals.task_failure.connect(task_failure_handler)

# Task retry configuration
app.conf.task_annotations = {
    'automation_engine.tasks.monitoring.*': {
        'rate_limit': '10/m',
        'autoretry_for': (Exception,),
        'retry_kwargs': {'max_retries': 3, 'countdown': 60},
    },
    'automation_engine.tasks.remediation.*': {
        'rate_limit': '5/m',
        'autoretry_for': (Exception,),
        'retry_kwargs': {'max_retries': 2, 'countdown': 120},
    },
    'automation_engine.tasks.analytics.*': {
        'rate_limit': '2/m',
        'autoretry_for': (Exception,),
        'retry_kwargs': {'max_retries': 1, 'countdown': 300},
    },
}

if __name__ == '__main__':
    app.start()