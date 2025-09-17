"""
Microsoft Teams Integration
Send notifications and messages to Teams channels
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class TeamsConnector:
    """Microsoft Teams webhook connector for notifications"""

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Teams connector

        Args:
            webhook_url: Teams incoming webhook URL
        """
        self.webhook_url = webhook_url or os.getenv("TEAMS_WEBHOOK_URL")

        if not self.webhook_url:
            logger.warning("Teams webhook URL not configured")

    async def send_message(
        self,
        title: str,
        text: str,
        color: str = "0076D7",
        facts: Optional[List[Dict[str, str]]] = None,
        actions: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Send a message to Teams channel

        Args:
            title: Message title
            text: Message body
            color: Theme color (hex)
            facts: List of fact key-value pairs
            actions: List of action buttons

        Returns:
            Success status
        """
        if not self.webhook_url:
            logger.error("Teams webhook URL not configured")
            return False

        # Build message card
        message = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": title,
            "sections": [{
                "activityTitle": title,
                "text": text,
                "markdown": True
            }]
        }

        # Add facts if provided
        if facts:
            message["sections"][0]["facts"] = facts

        # Add actions if provided
        if actions:
            message["potentialAction"] = actions

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=message,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully sent Teams message: {title}")
                        return True
                    else:
                        logger.error(f"Failed to send Teams message: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error sending Teams message: {str(e)}")
            return False

    async def send_incident_alert(
        self,
        incident_title: str,
        description: str,
        severity: str,
        affected_resources: List[str],
        incident_id: str
    ) -> bool:
        """Send incident alert to Teams"""

        # Determine color based on severity
        color_map = {
            "critical": "FF0000",
            "high": "FF6600",
            "medium": "FFCC00",
            "low": "00CC00"
        }
        color = color_map.get(severity.lower(), "0076D7")

        facts = [
            {"name": "Incident ID", "value": incident_id},
            {"name": "Severity", "value": severity.upper()},
            {"name": "Affected Resources", "value": ", ".join(affected_resources[:5])},
            {"name": "Time", "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
        ]

        actions = [
            {
                "@type": "OpenUri",
                "name": "View in Portal",
                "targets": [{
                    "os": "default",
                    "uri": f"{os.getenv('PORTAL_URL', 'https://portal.azure.com')}/incidents/{incident_id}"
                }]
            }
        ]

        return await self.send_message(
            title=f"🚨 Incident Alert: {incident_title}",
            text=description,
            color=color,
            facts=facts,
            actions=actions
        )

    async def send_deployment_notification(
        self,
        deployment_name: str,
        environment: str,
        status: str,
        version: str,
        deployed_by: str
    ) -> bool:
        """Send deployment notification to Teams"""

        # Determine color based on status
        if status.lower() == "success":
            color = "00CC00"
            emoji = "✅"
        elif status.lower() == "failed":
            color = "FF0000"
            emoji = "❌"
        else:
            color = "0076D7"
            emoji = "🔄"

        facts = [
            {"name": "Environment", "value": environment},
            {"name": "Version", "value": version},
            {"name": "Deployed By", "value": deployed_by},
            {"name": "Status", "value": status},
            {"name": "Time", "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
        ]

        return await self.send_message(
            title=f"{emoji} Deployment: {deployment_name}",
            text=f"Deployment of **{deployment_name}** to **{environment}** environment",
            color=color,
            facts=facts
        )

    async def send_cost_alert(
        self,
        alert_type: str,
        current_spend: float,
        budget_limit: float,
        percentage: float,
        recommendations: List[str]
    ) -> bool:
        """Send cost alert to Teams"""

        if percentage >= 90:
            color = "FF0000"
            emoji = "🚨"
        elif percentage >= 75:
            color = "FF6600"
            emoji = "⚠️"
        else:
            color = "FFCC00"
            emoji = "📊"

        text = f"**{alert_type}**\n\n"
        text += f"Current spend: **${current_spend:,.2f}**\n"
        text += f"Budget limit: **${budget_limit:,.2f}**\n"
        text += f"Usage: **{percentage:.1f}%**\n\n"

        if recommendations:
            text += "**Recommendations:**\n"
            for rec in recommendations[:3]:
                text += f"• {rec}\n"

        facts = [
            {"name": "Current Spend", "value": f"${current_spend:,.2f}"},
            {"name": "Budget Limit", "value": f"${budget_limit:,.2f}"},
            {"name": "Usage", "value": f"{percentage:.1f}%"},
            {"name": "Alert Time", "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
        ]

        return await self.send_message(
            title=f"{emoji} Cost Alert: {alert_type}",
            text=text,
            color=color,
            facts=facts
        )

    async def send_security_alert(
        self,
        alert_title: str,
        threat_level: str,
        description: str,
        affected_resources: List[str],
        recommended_actions: List[str]
    ) -> bool:
        """Send security alert to Teams"""

        # Always use red for security alerts
        color = "FF0000"

        text = f"**{description}**\n\n"

        if recommended_actions:
            text += "**Recommended Actions:**\n"
            for action in recommended_actions[:3]:
                text += f"• {action}\n"

        facts = [
            {"name": "Threat Level", "value": threat_level.upper()},
            {"name": "Affected Resources", "value": str(len(affected_resources))},
            {"name": "First Resource", "value": affected_resources[0] if affected_resources else "N/A"},
            {"name": "Detection Time", "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
        ]

        actions = [
            {
                "@type": "OpenUri",
                "name": "View in Security Center",
                "targets": [{
                    "os": "default",
                    "uri": f"{os.getenv('PORTAL_URL', 'https://portal.azure.com')}/security"
                }]
            }
        ]

        return await self.send_message(
            title=f"🔐 Security Alert: {alert_title}",
            text=text,
            color=color,
            facts=facts,
            actions=actions
        )

    async def send_performance_alert(
        self,
        resource_name: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        unit: str
    ) -> bool:
        """Send performance alert to Teams"""

        percentage = (current_value / threshold) * 100 if threshold > 0 else 0

        if percentage >= 100:
            color = "FF0000"
            emoji = "🔴"
        elif percentage >= 80:
            color = "FF6600"
            emoji = "🟠"
        else:
            color = "FFCC00"
            emoji = "🟡"

        facts = [
            {"name": "Resource", "value": resource_name},
            {"name": "Metric", "value": metric_name},
            {"name": "Current Value", "value": f"{current_value:.2f} {unit}"},
            {"name": "Threshold", "value": f"{threshold:.2f} {unit}"},
            {"name": "Usage", "value": f"{percentage:.1f}%"}
        ]

        return await self.send_message(
            title=f"{emoji} Performance Alert: {resource_name}",
            text=f"Resource **{resource_name}** has exceeded performance threshold for **{metric_name}**",
            color=color,
            facts=facts
        )