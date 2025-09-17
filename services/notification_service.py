"""
Notification Service Implementation
Provides centralized notification management
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from core.interfaces import INotificationService, IConfigurationProvider
from core.base_classes import BaseService
from core.exceptions import NotificationError

logger = logging.getLogger(__name__)


class NotificationService(BaseService, INotificationService):
    """Centralized notification service"""

    def __init__(self, config_provider: IConfigurationProvider):
        super().__init__(config_provider)
        self._notification_handlers = {}
        self._delivery_queue = []

    async def initialize(self) -> None:
        """Initialize notification service"""
        # Initialize notification channels
        await self._initialize_email_handler()
        await self._initialize_teams_handler()
        await self._initialize_slack_handler()
        await self._initialize_webhook_handler()

        logger.info("Notification service initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown notification service"""
        # Clean up handlers
        for handler in self._notification_handlers.values():
            if hasattr(handler, 'close'):
                await handler.close()

        self._notification_handlers.clear()
        logger.info("Notification service shutdown")

    async def send_alert(self, alert_type: str, severity: str, message: str,
                        recipients: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send alert notification"""
        try:
            metadata = metadata or {}

            # Create alert payload
            alert_payload = {
                "type": alert_type,
                "severity": severity,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata
            }

            # Determine notification channels based on severity
            channels = self._get_channels_for_severity(severity)

            # Send to each channel
            delivery_results = []
            for recipient in recipients:
                for channel in channels:
                    try:
                        result = await self._send_to_channel(
                            channel, recipient, alert_payload
                        )
                        delivery_results.append({
                            "recipient": recipient,
                            "channel": channel,
                            "status": result.get("status", "unknown"),
                            "message_id": result.get("message_id")
                        })
                    except Exception as e:
                        logger.error(f"Failed to send alert via {channel}: {e}")
                        delivery_results.append({
                            "recipient": recipient,
                            "channel": channel,
                            "status": "failed",
                            "error": str(e)
                        })

            return {
                "alert_id": f"alert_{hash(f'{alert_type}{message}{datetime.utcnow()}')}",
                "delivery_results": delivery_results,
                "summary": {
                    "total_attempts": len(delivery_results),
                    "successful": len([r for r in delivery_results if r["status"] == "sent"]),
                    "failed": len([r for r in delivery_results if r["status"] == "failed"])
                }
            }

        except Exception as e:
            logger.error(f"Alert sending failed: {e}")
            raise NotificationError(f"Failed to send alert: {e}", notification_type=alert_type)

    async def send_report(self, report_type: str, content: Dict[str, Any],
                         recipients: List[str]) -> Dict[str, Any]:
        """Send report notification"""
        try:
            # Format report content
            report_payload = {
                "type": report_type,
                "content": content,
                "generated_at": datetime.utcnow().isoformat(),
                "format": "json"
            }

            # Send via email primarily for reports
            delivery_results = []
            for recipient in recipients:
                try:
                    result = await self._send_to_channel(
                        "email", recipient, report_payload
                    )
                    delivery_results.append({
                        "recipient": recipient,
                        "channel": "email",
                        "status": result.get("status", "unknown"),
                        "message_id": result.get("message_id")
                    })
                except Exception as e:
                    logger.error(f"Failed to send report to {recipient}: {e}")
                    delivery_results.append({
                        "recipient": recipient,
                        "channel": "email",
                        "status": "failed",
                        "error": str(e)
                    })

            return {
                "report_id": f"report_{hash(f'{report_type}{datetime.utcnow()}')}",
                "delivery_results": delivery_results,
                "summary": {
                    "total_attempts": len(delivery_results),
                    "successful": len([r for r in delivery_results if r["status"] == "sent"]),
                    "failed": len([r for r in delivery_results if r["status"] == "failed"])
                }
            }

        except Exception as e:
            logger.error(f"Report sending failed: {e}")
            raise NotificationError(f"Failed to send report: {e}", notification_type=report_type)

    def _get_channels_for_severity(self, severity: str) -> List[str]:
        """Get notification channels based on severity"""
        severity_channels = {
            "low": ["email"],
            "medium": ["email", "teams"],
            "high": ["email", "teams", "slack"],
            "critical": ["email", "teams", "slack", "webhook"]
        }
        return severity_channels.get(severity, ["email"])

    async def _send_to_channel(self, channel: str, recipient: str,
                              payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification to specific channel"""
        handler = self._notification_handlers.get(channel)
        if not handler:
            raise NotificationError(f"Handler for channel {channel} not available")

        return await handler.send(recipient, payload)

    async def _initialize_email_handler(self):
        """Initialize email notification handler"""
        self._notification_handlers["email"] = EmailNotificationHandler(
            self.config_provider
        )

    async def _initialize_teams_handler(self):
        """Initialize Microsoft Teams notification handler"""
        teams_webhook = self.config_provider.get_setting("teams_webhook_url")
        if teams_webhook:
            self._notification_handlers["teams"] = TeamsNotificationHandler(
                self.config_provider
            )

    async def _initialize_slack_handler(self):
        """Initialize Slack notification handler"""
        slack_webhook = self.config_provider.get_setting("slack_webhook_url")
        if slack_webhook:
            self._notification_handlers["slack"] = SlackNotificationHandler(
                self.config_provider
            )

    async def _initialize_webhook_handler(self):
        """Initialize webhook notification handler"""
        self._notification_handlers["webhook"] = WebhookNotificationHandler(
            self.config_provider
        )


class EmailNotificationHandler:
    """Email notification handler"""

    def __init__(self, config_provider: IConfigurationProvider):
        self.config_provider = config_provider

    async def send(self, recipient: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification"""
        # Mock email sending
        # In production, use SendGrid, SMTP, or Azure Communication Services
        logger.info(f"Sending email to {recipient}: {payload['type']}")

        return {
            "status": "sent",
            "message_id": f"email_{hash(f'{recipient}{datetime.utcnow()}')}",
            "timestamp": datetime.utcnow().isoformat()
        }


class TeamsNotificationHandler:
    """Microsoft Teams notification handler"""

    def __init__(self, config_provider: IConfigurationProvider):
        self.config_provider = config_provider
        self.webhook_url = config_provider.get_setting("teams_webhook_url")

    async def send(self, recipient: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send Teams notification"""
        # Mock Teams webhook sending
        # In production, use actual Teams webhook API
        logger.info(f"Sending Teams notification: {payload['type']}")

        # Format Teams card
        teams_card = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": payload.get("message", "Notification"),
            "themeColor": self._get_color_for_type(payload.get("type", "info")),
            "sections": [{
                "activityTitle": f"Azure AI Copilot - {payload.get('type', 'Notification')}",
                "activitySubtitle": payload.get("message", ""),
                "facts": [
                    {"name": "Severity", "value": payload.get("severity", "medium")},
                    {"name": "Time", "value": payload.get("timestamp", "")}
                ]
            }]
        }

        return {
            "status": "sent",
            "message_id": f"teams_{hash(f'{payload}{datetime.utcnow()}')}",
            "timestamp": datetime.utcnow().isoformat(),
            "webhook_response": "200"
        }

    def _get_color_for_type(self, notification_type: str) -> str:
        """Get color theme for notification type"""
        colors = {
            "info": "0078d4",
            "warning": "ff8c00",
            "error": "d13438",
            "critical": "8B0000",
            "success": "107c10"
        }
        return colors.get(notification_type, "0078d4")


class SlackNotificationHandler:
    """Slack notification handler"""

    def __init__(self, config_provider: IConfigurationProvider):
        self.config_provider = config_provider
        self.webhook_url = config_provider.get_setting("slack_webhook_url")

    async def send(self, recipient: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send Slack notification"""
        # Mock Slack webhook sending
        # In production, use actual Slack webhook API
        logger.info(f"Sending Slack notification: {payload['type']}")

        # Format Slack message
        slack_message = {
            "text": payload.get("message", "Notification"),
            "attachments": [{
                "color": self._get_color_for_severity(payload.get("severity", "medium")),
                "fields": [
                    {"title": "Type", "value": payload.get("type", "notification"), "short": True},
                    {"title": "Severity", "value": payload.get("severity", "medium"), "short": True},
                    {"title": "Time", "value": payload.get("timestamp", ""), "short": False}
                ]
            }]
        }

        return {
            "status": "sent",
            "message_id": f"slack_{hash(f'{payload}{datetime.utcnow()}')}",
            "timestamp": datetime.utcnow().isoformat(),
            "webhook_response": "ok"
        }

    def _get_color_for_severity(self, severity: str) -> str:
        """Get color for severity level"""
        colors = {
            "low": "good",
            "medium": "warning",
            "high": "danger",
            "critical": "#8B0000"
        }
        return colors.get(severity, "good")


class WebhookNotificationHandler:
    """Generic webhook notification handler"""

    def __init__(self, config_provider: IConfigurationProvider):
        self.config_provider = config_provider

    async def send(self, recipient: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook notification"""
        # Mock webhook sending
        # In production, use httpx or aiohttp to send actual webhooks
        logger.info(f"Sending webhook notification to {recipient}")

        webhook_payload = {
            "event": "notification",
            "data": payload,
            "recipient": recipient,
            "source": "azure-ai-copilot"
        }

        return {
            "status": "sent",
            "message_id": f"webhook_{hash(f'{payload}{datetime.utcnow()}')}",
            "timestamp": datetime.utcnow().isoformat(),
            "response_code": 200
        }