"""
Notifications Tasks Module
Handles alerts, notifications, and communication with external systems
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from celery import Task
from celery.exceptions import Retry

from automation_engine.celery_app import celery_app
from config.settings import get_settings

logger = logging.getLogger(__name__)


class BaseNotificationTask(Task):
    """Base class for notification tasks with common functionality"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Notification task {task_id} failed: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Notification task {task_id} retrying: {exc}")


@celery_app.task(bind=True, base=BaseNotificationTask, max_retries=3)
def send_alert(self, alert_type: str, severity: str, message: str,
               recipients: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Send alert notifications to specified recipients

    Args:
        alert_type: Type of alert (cost, security, performance, compliance)
        severity: Alert severity (low, medium, high, critical)
        message: Alert message content
        recipients: List of recipient identifiers
        metadata: Additional alert metadata

    Returns:
        Dict containing notification results
    """
    try:
        settings = get_settings()
        metadata = metadata or {}

        # Mock notification sending
        # In production, this would integrate with actual notification services
        notification_channels = {
            "email": True,
            "teams": bool(settings.teams_webhook_url),
            "slack": bool(settings.slack_webhook_url),
            "sms": False  # Mock SMS capability
        }

        delivery_results = []

        for recipient in recipients:
            # Mock delivery to different channels
            for channel, enabled in notification_channels.items():
                if enabled:
                    delivery_result = {
                        "recipient": recipient,
                        "channel": channel,
                        "status": "delivered",
                        "delivery_time": datetime.utcnow().isoformat(),
                        "message_id": f"msg_{channel}_{hash(f'{recipient}{datetime.utcnow()}')}"
                    }
                    delivery_results.append(delivery_result)

        # Create alert record
        alert_record = {
            "alert_id": f"alert_{hash(f'{alert_type}{message}{datetime.utcnow()}')}",
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata,
            "recipients_count": len(recipients),
            "channels_used": [ch for ch, enabled in notification_channels.items() if enabled],
            "created_at": datetime.utcnow().isoformat(),
            "status": "sent"
        }

        result = {
            "task_id": self.request.id,
            "alert_record": alert_record,
            "delivery_results": delivery_results,
            "summary": {
                "total_deliveries": len(delivery_results),
                "successful_deliveries": len([d for d in delivery_results if d["status"] == "delivered"]),
                "failed_deliveries": len([d for d in delivery_results if d["status"] == "failed"])
            },
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Alert sent: {alert_type} ({severity}) to {len(recipients)} recipients")
        return result

    except Exception as exc:
        logger.error(f"Alert sending failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseNotificationTask, max_retries=3)
def send_cost_report(self, subscription_id: str, report_period: str,
                    recipients: List[str], include_forecast: bool = True) -> Dict[str, Any]:
    """
    Send cost analysis report

    Args:
        subscription_id: Azure subscription ID
        report_period: Report period (daily, weekly, monthly)
        recipients: List of report recipients
        include_forecast: Whether to include cost forecast

    Returns:
        Dict containing report delivery results
    """
    try:
        settings = get_settings()

        # Mock cost report generation
        current_costs = {
            "total_cost": 1245.67,
            "compute_cost": 678.90,
            "storage_cost": 234.56,
            "network_cost": 123.45,
            "other_cost": 208.76,
            "currency": "USD",
            "period": report_period
        }

        if include_forecast:
            forecast = {
                "next_month_forecast": 1389.45,
                "trend": "increasing",
                "confidence": 0.87
            }
            current_costs["forecast"] = forecast

        # Mock report content
        report_content = {
            "subscription_id": subscription_id,
            "report_period": report_period,
            "generated_at": datetime.utcnow().isoformat(),
            "cost_breakdown": current_costs,
            "top_resources": [
                {"name": "vm-prod-cluster", "cost": 234.56, "type": "Virtual Machine"},
                {"name": "sql-database-prod", "cost": 189.23, "type": "SQL Database"},
                {"name": "storage-prod-logs", "cost": 145.67, "type": "Storage Account"}
            ],
            "recommendations": [
                "Consider rightsizing VM cluster - potential savings: $78/month",
                "Optimize storage tiers for log data - potential savings: $45/month"
            ]
        }

        # Mock report delivery
        delivery_results = []
        for recipient in recipients:
            delivery_result = {
                "recipient": recipient,
                "delivery_method": "email",
                "status": "delivered",
                "delivered_at": datetime.utcnow().isoformat(),
                "report_size_kb": 156
            }
            delivery_results.append(delivery_result)

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "report_content": report_content,
            "delivery_results": delivery_results,
            "summary": {
                "recipients_count": len(recipients),
                "successful_deliveries": len([d for d in delivery_results if d["status"] == "delivered"]),
                "total_cost_reported": current_costs["total_cost"]
            },
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Cost report sent for {subscription_id}: ${current_costs['total_cost']:.2f} to {len(recipients)} recipients")
        return result

    except Exception as exc:
        logger.error(f"Cost report sending failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseNotificationTask, max_retries=3)
def send_security_notification(self, security_event: Dict[str, Any],
                              recipients: List[str], escalation_level: str = "normal") -> Dict[str, Any]:
    """
    Send security-related notifications

    Args:
        security_event: Security event details
        recipients: List of recipients
        escalation_level: Escalation level (normal, urgent, critical)

    Returns:
        Dict containing notification results
    """
    try:
        settings = get_settings()

        # Determine notification urgency based on event and escalation level
        urgency_mapping = {
            "normal": {"priority": "medium", "delivery_timeout": 300},
            "urgent": {"priority": "high", "delivery_timeout": 60},
            "critical": {"priority": "critical", "delivery_timeout": 30}
        }

        urgency = urgency_mapping.get(escalation_level, urgency_mapping["normal"])

        # Mock security notification content
        notification_content = {
            "event_id": security_event.get("event_id", f"sec_{hash(str(security_event))}"),
            "event_type": security_event.get("type", "security_alert"),
            "severity": security_event.get("severity", "medium"),
            "description": security_event.get("description", "Security event detected"),
            "affected_resources": security_event.get("resources", []),
            "recommended_actions": security_event.get("actions", [
                "Review security logs",
                "Verify resource configurations",
                "Contact security team if needed"
            ]),
            "escalation_level": escalation_level,
            "priority": urgency["priority"],
            "detected_at": security_event.get("detected_at", datetime.utcnow().isoformat())
        }

        # Mock notification delivery
        delivery_results = []
        for recipient in recipients:
            # For critical events, try multiple channels
            channels = ["email"]
            if escalation_level == "critical":
                channels.extend(["teams", "slack"])

            for channel in channels:
                delivery_result = {
                    "recipient": recipient,
                    "channel": channel,
                    "status": "delivered",
                    "priority": urgency["priority"],
                    "delivered_at": datetime.utcnow().isoformat(),
                    "delivery_time_seconds": 2  # Mock delivery time
                }
                delivery_results.append(delivery_result)

        result = {
            "task_id": self.request.id,
            "security_event": notification_content,
            "escalation_level": escalation_level,
            "delivery_results": delivery_results,
            "summary": {
                "recipients_notified": len(recipients),
                "channels_used": len(set(d["channel"] for d in delivery_results)),
                "priority_level": urgency["priority"],
                "event_severity": notification_content["severity"]
            },
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Security notification sent: {notification_content['event_type']} ({escalation_level}) to {len(recipients)} recipients")
        return result

    except Exception as exc:
        logger.error(f"Security notification failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseNotificationTask, max_retries=3)
def send_maintenance_notification(self, maintenance_type: str, affected_resources: List[str],
                                 start_time: str, duration_minutes: int,
                                 recipients: List[str]) -> Dict[str, Any]:
    """
    Send maintenance window notifications

    Args:
        maintenance_type: Type of maintenance
        affected_resources: List of affected resource names
        start_time: Maintenance start time (ISO format)
        duration_minutes: Expected duration in minutes
        recipients: List of recipients

    Returns:
        Dict containing notification results
    """
    try:
        settings = get_settings()

        # Calculate maintenance window details
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        # Mock maintenance notification content
        maintenance_info = {
            "maintenance_id": f"maint_{hash(f'{maintenance_type}{start_time}')}",
            "type": maintenance_type,
            "affected_resources": affected_resources,
            "start_time": start_time,
            "end_time": end_dt.isoformat(),
            "duration_minutes": duration_minutes,
            "impact_level": "medium" if duration_minutes < 60 else "high",
            "description": f"Scheduled {maintenance_type} maintenance",
            "preparation_steps": [
                "Ensure backups are current",
                "Review service dependencies",
                "Prepare rollback procedures"
            ],
            "contact_info": "support@company.com"
        }

        # Mock notification delivery with different lead times
        delivery_results = []
        notification_types = ["advance_notice", "reminder", "start_notification"]

        for recipient in recipients:
            for notif_type in notification_types:
                delivery_result = {
                    "recipient": recipient,
                    "notification_type": notif_type,
                    "channel": "email",
                    "status": "scheduled" if notif_type != "advance_notice" else "delivered",
                    "scheduled_for": start_time if notif_type == "start_notification" else datetime.utcnow().isoformat(),
                    "content_type": "maintenance_notification"
                }
                delivery_results.append(delivery_result)

        result = {
            "task_id": self.request.id,
            "maintenance_info": maintenance_info,
            "delivery_results": delivery_results,
            "summary": {
                "affected_resources_count": len(affected_resources),
                "recipients_count": len(recipients),
                "notifications_scheduled": len(delivery_results),
                "maintenance_duration_hours": round(duration_minutes / 60, 2)
            },
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Maintenance notification sent: {maintenance_type} affecting {len(affected_resources)} resources")
        return result

    except Exception as exc:
        logger.error(f"Maintenance notification failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseNotificationTask, max_retries=3)
def send_health_summary(self, subscription_id: str, health_data: Dict[str, Any],
                       recipients: List[str], summary_type: str = "daily") -> Dict[str, Any]:
    """
    Send system health summary notifications

    Args:
        subscription_id: Azure subscription ID
        health_data: System health metrics and status
        recipients: List of recipients
        summary_type: Type of summary (daily, weekly, monthly)

    Returns:
        Dict containing notification results
    """
    try:
        settings = get_settings()

        # Mock health summary compilation
        health_summary = {
            "subscription_id": subscription_id,
            "summary_type": summary_type,
            "period_start": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "period_end": datetime.utcnow().isoformat(),
            "overall_health_score": health_data.get("overall_score", 85.2),
            "services_status": {
                "compute": {"status": "healthy", "availability": 99.95},
                "storage": {"status": "healthy", "availability": 99.99},
                "network": {"status": "degraded", "availability": 98.50},
                "database": {"status": "healthy", "availability": 99.85}
            },
            "key_metrics": {
                "total_resources": health_data.get("total_resources", 156),
                "active_alerts": health_data.get("active_alerts", 3),
                "resolved_incidents": health_data.get("resolved_incidents", 2),
                "cost_variance": health_data.get("cost_variance", "+2.3%")
            },
            "recommendations": [
                "Monitor network performance in East US region",
                "Review and optimize VM usage patterns",
                "Update security policies for storage accounts"
            ],
            "next_actions": [
                "Scheduled maintenance window: Saturday 2:00 AM",
                "Security review: Due next week",
                "Cost optimization review: In progress"
            ]
        }

        # Mock notification delivery
        delivery_results = []
        for recipient in recipients:
            delivery_result = {
                "recipient": recipient,
                "delivery_method": "email",
                "content_type": f"{summary_type}_health_summary",
                "status": "delivered",
                "delivered_at": datetime.utcnow().isoformat(),
                "summary_score": health_summary["overall_health_score"]
            }
            delivery_results.append(delivery_result)

        result = {
            "task_id": self.request.id,
            "health_summary": health_summary,
            "delivery_results": delivery_results,
            "summary": {
                "recipients_count": len(recipients),
                "overall_health_score": health_summary["overall_health_score"],
                "active_alerts": health_summary["key_metrics"]["active_alerts"],
                "summary_period": summary_type
            },
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Health summary sent for {subscription_id}: score {health_summary['overall_health_score']:.1f} to {len(recipients)} recipients")
        return result

    except Exception as exc:
        logger.error(f"Health summary sending failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseNotificationTask, max_retries=3)
def process_webhook_notification(self, webhook_url: str, payload: Dict[str, Any],
                                headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Process outbound webhook notifications

    Args:
        webhook_url: Destination webhook URL
        payload: Notification payload
        headers: Optional custom headers

    Returns:
        Dict containing webhook delivery results
    """
    try:
        settings = get_settings()
        headers = headers or {}

        # Mock webhook delivery
        # In production, this would make actual HTTP requests
        delivery_attempt = {
            "webhook_url": webhook_url,
            "payload_size_bytes": len(str(payload)),
            "headers": headers,
            "attempt_time": datetime.utcnow().isoformat(),
            "status_code": 200,
            "response_time_ms": 125,
            "status": "success"
        }

        # Mock response
        webhook_response = {
            "status": "received",
            "message": "Webhook processed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }

        result = {
            "task_id": self.request.id,
            "webhook_url": webhook_url,
            "payload": payload,
            "delivery_attempt": delivery_attempt,
            "webhook_response": webhook_response,
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Webhook notification sent to {webhook_url}: {delivery_attempt['status_code']}")
        return result

    except Exception as exc:
        logger.error(f"Webhook notification failed: {exc}")
        raise self.retry(exc=exc, countdown=60)