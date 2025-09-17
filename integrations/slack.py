"""
Slack Integration
Send notifications and interact with Slack workspaces
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class SlackConnector:
    """Slack API connector for notifications and interactions"""

    def __init__(self, bot_token: Optional[str] = None, webhook_url: Optional[str] = None):
        """
        Initialize Slack connector

        Args:
            bot_token: Slack bot token for API calls
            webhook_url: Slack incoming webhook URL for simple notifications
        """
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.api_url = "https://slack.com/api"

        if not self.bot_token and not self.webhook_url:
            logger.warning("Slack credentials not configured")

    async def send_webhook_message(self, text: str, channel: Optional[str] = None) -> bool:
        """
        Send a simple message via webhook

        Args:
            text: Message text
            channel: Optional channel override

        Returns:
            Success status
        """
        if not self.webhook_url:
            logger.error("Slack webhook URL not configured")
            return False

        payload = {"text": text}
        if channel:
            payload["channel"] = channel

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        logger.info("Successfully sent Slack webhook message")
                        return True
                    else:
                        logger.error(f"Failed to send Slack webhook message: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error sending Slack webhook message: {str(e)}")
            return False

    async def send_api_message(
        self,
        channel: str,
        text: Optional[str] = None,
        blocks: Optional[List[Dict]] = None,
        attachments: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Send a message via Slack API with rich formatting

        Args:
            channel: Channel ID or name
            text: Message text (fallback)
            blocks: Block Kit blocks for rich formatting
            attachments: Legacy attachments

        Returns:
            API response
        """
        if not self.bot_token:
            logger.error("Slack bot token not configured")
            return {"ok": False, "error": "not_configured"}

        payload = {"channel": channel}

        if text:
            payload["text"] = text
        if blocks:
            payload["blocks"] = blocks
        if attachments:
            payload["attachments"] = attachments

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/chat.postMessage",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.bot_token}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    result = await response.json()

                    if result.get("ok"):
                        logger.info(f"Successfully sent Slack message to {channel}")
                    else:
                        logger.error(f"Failed to send Slack message: {result.get('error')}")

                    return result

        except Exception as e:
            logger.error(f"Error sending Slack API message: {str(e)}")
            return {"ok": False, "error": str(e)}

    async def send_incident_alert(
        self,
        channel: str,
        incident_title: str,
        description: str,
        severity: str,
        affected_resources: List[str],
        incident_id: str
    ) -> Dict[str, Any]:
        """Send incident alert to Slack"""

        # Determine emoji based on severity
        emoji_map = {
            "critical": ":rotating_light:",
            "high": ":warning:",
            "medium": ":orange_circle:",
            "low": ":large_blue_circle:"
        }
        emoji = emoji_map.get(severity.lower(), ":grey_question:")

        # Build blocks for rich formatting
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Incident: {incident_title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": description
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Incident ID:*\n{incident_id}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{severity.upper()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Affected Resources:*\n{', '.join(affected_resources[:3])}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                    }
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View in Portal"
                        },
                        "url": f"{os.getenv('PORTAL_URL', 'https://portal.azure.com')}/incidents/{incident_id}",
                        "action_id": "view_incident"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Acknowledge"
                        },
                        "style": "primary",
                        "action_id": f"ack_incident_{incident_id}"
                    }
                ]
            }
        ]

        return await self.send_api_message(
            channel=channel,
            text=f"Incident Alert: {incident_title}",
            blocks=blocks
        )

    async def send_deployment_notification(
        self,
        channel: str,
        deployment_name: str,
        environment: str,
        status: str,
        version: str,
        deployed_by: str
    ) -> Dict[str, Any]:
        """Send deployment notification to Slack"""

        # Determine emoji based on status
        if status.lower() == "success":
            emoji = ":white_check_mark:"
            color = "good"
        elif status.lower() == "failed":
            emoji = ":x:"
            color = "danger"
        else:
            emoji = ":arrows_counterclockwise:"
            color = "warning"

        attachments = [{
            "color": color,
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} Deployment: {deployment_name}",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Environment:* {environment}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Version:* {version}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Status:* {status}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Deployed By:* {deployed_by}"
                        }
                    ]
                },
                {
                    "type": "context",
                    "elements": [{
                        "type": "mrkdwn",
                        "text": f"Deployed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                    }]
                }
            ]
        }]

        return await self.send_api_message(
            channel=channel,
            text=f"Deployment {status}: {deployment_name}",
            attachments=attachments
        )

    async def send_cost_alert(
        self,
        channel: str,
        alert_type: str,
        current_spend: float,
        budget_limit: float,
        percentage: float,
        recommendations: List[str]
    ) -> Dict[str, Any]:
        """Send cost alert to Slack"""

        if percentage >= 90:
            emoji = ":rotating_light:"
            color = "danger"
        elif percentage >= 75:
            emoji = ":warning:"
            color = "warning"
        else:
            emoji = ":chart_with_upwards_trend:"
            color = "good"

        # Build recommendation text
        rec_text = ""
        if recommendations:
            rec_text = "\n*Recommendations:*\n"
            for rec in recommendations[:3]:
                rec_text += f"• {rec}\n"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Cost Alert: {alert_type}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Current spend: *${current_spend:,.2f}*\nBudget limit: *${budget_limit:,.2f}*\nUsage: *{percentage:.1f}%*{rec_text}"
                }
            },
            {
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": f"Alert generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
                }]
            }
        ]

        attachments = [{
            "color": color,
            "blocks": blocks
        }]

        return await self.send_api_message(
            channel=channel,
            text=f"Cost Alert: {alert_type} - {percentage:.1f}% of budget",
            attachments=attachments
        )

    async def send_interactive_message(
        self,
        channel: str,
        text: str,
        actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Send an interactive message with buttons"""

        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text
                }
            },
            {
                "type": "actions",
                "elements": actions
            }
        ]

        return await self.send_api_message(
            channel=channel,
            text=text,
            blocks=blocks
        )

    async def update_message(
        self,
        channel: str,
        timestamp: str,
        text: Optional[str] = None,
        blocks: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Update an existing message"""

        if not self.bot_token:
            logger.error("Slack bot token not configured")
            return {"ok": False, "error": "not_configured"}

        payload = {
            "channel": channel,
            "ts": timestamp
        }

        if text:
            payload["text"] = text
        if blocks:
            payload["blocks"] = blocks

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/chat.update",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.bot_token}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    result = await response.json()

                    if result.get("ok"):
                        logger.info(f"Successfully updated Slack message")
                    else:
                        logger.error(f"Failed to update Slack message: {result.get('error')}")

                    return result

        except Exception as e:
            logger.error(f"Error updating Slack message: {str(e)}")
            return {"ok": False, "error": str(e)}

    async def get_channel_list(self) -> List[Dict[str, Any]]:
        """Get list of channels the bot has access to"""

        if not self.bot_token:
            logger.error("Slack bot token not configured")
            return []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/conversations.list",
                    headers={
                        "Authorization": f"Bearer {self.bot_token}"
                    }
                ) as response:
                    result = await response.json()

                    if result.get("ok"):
                        return result.get("channels", [])
                    else:
                        logger.error(f"Failed to get Slack channels: {result.get('error')}")
                        return []

        except Exception as e:
            logger.error(f"Error getting Slack channels: {str(e)}")
            return []