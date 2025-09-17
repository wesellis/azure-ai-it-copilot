"""
Advanced monitoring and alerting configuration for Azure AI IT Copilot
Real-time monitoring with Prometheus, Grafana, and custom metrics
"""

import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import psutil
import aiohttp

logger = logging.getLogger(__name__)


class AdvancedMonitoring:
    """Advanced monitoring and metrics collection"""

    def __init__(self):
        self.registry = CollectorRegistry()
        self.setup_metrics()
        self.alerts = []

    def setup_metrics(self):
        """Setup custom Prometheus metrics"""
        
        # API Metrics
        self.api_requests_total = Counter(
            'ai_copilot_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code', 'user_role'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'ai_copilot_api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # AI Operations Metrics
        self.command_processing_time = Histogram(
            'ai_copilot_command_processing_seconds',
            'Time to process natural language commands',
            ['intent_type', 'agent'],
            registry=self.registry
        )
        
        self.agent_execution_success = Counter(
            'ai_copilot_agent_executions_total',
            'Total agent executions',
            ['agent', 'status'],
            registry=self.registry
        )
        
        self.prediction_accuracy = Gauge(
            'ai_copilot_prediction_accuracy',
            'Accuracy of predictive models',
            ['model_type'],
            registry=self.registry
        )
        
        # Infrastructure Metrics
        self.azure_resources_monitored = Gauge(
            'ai_copilot_azure_resources_total',
            'Total Azure resources being monitored',
            ['resource_type', 'region'],
            registry=self.registry
        )
        
        self.incident_response_time = Histogram(
            'ai_copilot_incident_response_seconds',
            'Time from incident detection to resolution',
            ['severity', 'auto_resolved'],
            registry=self.registry
        )
        
        # Cost Metrics
        self.cost_optimization_savings = Counter(
            'ai_copilot_cost_savings_dollars',
            'Total cost savings achieved',
            ['optimization_type'],
            registry=self.registry
        )
        
        # System Health Metrics
        self.system_health_score = Gauge(
            'ai_copilot_system_health_score',
            'Overall system health score (0-100)',
            registry=self.registry
        )
        
        self.active_alerts = Gauge(
            'ai_copilot_active_alerts',
            'Number of active alerts',
            ['severity'],
            registry=self.registry
        )

    def record_api_request(self, method: str, endpoint: str, status_code: int, 
                          duration: float, user_role: str = "unknown"):
        """Record API request metrics"""
        self.api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            user_role=user_role
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def record_command_processing(self, intent_type: str, agent: str, 
                                 duration: float, success: bool):
        """Record command processing metrics"""
        self.command_processing_time.labels(
            intent_type=intent_type,
            agent=agent
        ).observe(duration)
        
        self.agent_execution_success.labels(
            agent=agent,
            status="success" if success else "error"
        ).inc()

    def update_prediction_accuracy(self, model_type: str, accuracy: float):
        """Update prediction model accuracy"""
        self.prediction_accuracy.labels(model_type=model_type).set(accuracy)

    def record_cost_savings(self, optimization_type: str, savings_amount: float):
        """Record cost optimization savings"""
        self.cost_optimization_savings.labels(
            optimization_type=optimization_type
        ).inc(savings_amount)

    def update_system_health(self, score: float):
        """Update overall system health score"""
        self.system_health_score.set(score)

    def calculate_system_health(self) -> float:
        """Calculate comprehensive system health score"""
        try:
            # CPU and Memory health
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Component health scores (0-100)
            cpu_health = max(0, 100 - cpu_percent)
            memory_health = max(0, 100 - memory_percent)  
            disk_health = max(0, 100 - disk_percent)
            
            # API health (based on recent error rates)
            api_health = 100  # Would calculate from recent metrics
            
            # Agent health (based on success rates)
            agent_health = 95  # Would calculate from recent executions
            
            # Weighted average
            health_score = (
                cpu_health * 0.2 +
                memory_health * 0.2 +
                disk_health * 0.15 +
                api_health * 0.25 +
                agent_health * 0.2
            )
            
            return min(100, max(0, health_score))
            
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return 50  # Default moderate health

    async def collect_azure_metrics(self):
        """Collect metrics from Azure resources"""
        # This would integrate with Azure Monitor API
        # Mock implementation for now
        
        mock_metrics = {
            "vm_count": 15,
            "storage_accounts": 8,
            "app_services": 12,
            "databases": 5,
            "total_cost_monthly": 15420.50
        }
        
        # Update resource counters
        self.azure_resources_monitored.labels(
            resource_type="virtual_machines",
            region="eastus"
        ).set(mock_metrics["vm_count"])
        
        return mock_metrics

    def generate_alert(self, severity: str, title: str, description: str, 
                      resource: str = None) -> Dict[str, Any]:
        """Generate monitoring alert"""
        alert = {
            "id": f"alert-{int(time.time())}",
            "severity": severity,
            "title": title,
            "description": description,
            "resource": resource,
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        self.active_alerts.labels(severity=severity).inc()
        
        logger.warning(f"Alert generated: {title} ({severity})")
        return alert

    def check_alert_conditions(self):
        """Check for alert conditions"""
        alerts_generated = []
        
        # System resource alerts
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > 85:
            alerts_generated.append(
                self.generate_alert(
                    "high",
                    "High CPU Usage",
                    f"CPU usage is {cpu_percent:.1f}%",
                    "system"
                )
            )
        
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            alerts_generated.append(
                self.generate_alert(
                    "critical",
                    "High Memory Usage", 
                    f"Memory usage is {memory_percent:.1f}%",
                    "system"
                )
            )
        
        # AI model performance alerts
        # This would check actual model metrics
        
        return alerts_generated

    def get_metrics(self) -> str:
        """Get Prometheus metrics output"""
        return generate_latest(self.registry)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "system_health": {
                "score": self.calculate_system_health(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "active_alerts": [
                alert for alert in self.alerts 
                if alert["status"] == "active"
            ],
            "recent_metrics": {
                "api_requests_last_hour": 1247,  # Would calculate from metrics
                "commands_processed": 89,
                "incidents_resolved": 12,
                "cost_savings": 2340.50
            },
            "agent_status": {
                "infrastructure_agent": "healthy",
                "incident_agent": "healthy", 
                "cost_agent": "healthy",
                "compliance_agent": "healthy",
                "predictive_agent": "healthy"
            }
        }


class AlertManager:
    """Advanced alert management and notifications"""

    def __init__(self, monitoring: AdvancedMonitoring):
        self.monitoring = monitoring
        self.notification_channels = []
        self.escalation_rules = []

    def add_notification_channel(self, channel_type: str, config: Dict[str, Any]):
        """Add notification channel (email, Slack, Teams, etc.)"""
        self.notification_channels.append({
            "type": channel_type,
            "config": config,
            "enabled": True
        })

    def add_escalation_rule(self, severity: str, delay_minutes: int, 
                           notify_channels: List[str]):
        """Add alert escalation rule"""
        self.escalation_rules.append({
            "severity": severity,
            "delay_minutes": delay_minutes,
            "notify_channels": notify_channels,
            "enabled": True
        })

    async def process_alerts(self):
        """Process and route alerts"""
        active_alerts = [
            alert for alert in self.monitoring.alerts 
            if alert["status"] == "active" and not alert["acknowledged"]
        ]
        
        for alert in active_alerts:
            await self.handle_alert(alert)

    async def handle_alert(self, alert: Dict[str, Any]):
        """Handle individual alert"""
        severity = alert["severity"]
        
        # Immediate notification for critical alerts
        if severity == "critical":
            await self.send_notifications(alert, ["email", "slack", "teams"])
        elif severity == "high":
            await self.send_notifications(alert, ["slack", "teams"])
        else:
            await self.send_notifications(alert, ["slack"])

    async def send_notifications(self, alert: Dict[str, Any], 
                               channel_types: List[str]):
        """Send notifications through specified channels"""
        for channel in self.notification_channels:
            if channel["type"] in channel_types and channel["enabled"]:
                try:
                    await self.send_notification(alert, channel)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel['type']}: {e}")

    async def send_notification(self, alert: Dict[str, Any], 
                              channel: Dict[str, Any]):
        """Send notification through specific channel"""
        if channel["type"] == "slack":
            await self.send_slack_notification(alert, channel["config"])
        elif channel["type"] == "teams":
            await self.send_teams_notification(alert, channel["config"])
        elif channel["type"] == "email":
            await self.send_email_notification(alert, channel["config"])

    async def send_slack_notification(self, alert: Dict[str, Any], 
                                    config: Dict[str, Any]):
        """Send Slack notification"""
        webhook_url = config.get("webhook_url")
        if not webhook_url:
            return

        color_map = {
            "low": "#36a64f",      # Green
            "medium": "#ff9900",   # Orange  
            "high": "#ff6600",     # Red-orange
            "critical": "#ff0000"  # Red
        }

        payload = {
            "text": f"🚨 Azure AI Copilot Alert: {alert['title']}",
            "attachments": [{
                "color": color_map.get(alert["severity"], "#cccccc"),
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert["severity"].upper(),
                        "short": True
                    },
                    {
                        "title": "Resource", 
                        "value": alert.get("resource", "Unknown"),
                        "short": True
                    },
                    {
                        "title": "Description",
                        "value": alert["description"],
                        "short": False
                    },
                    {
                        "title": "Time",
                        "value": alert["timestamp"],
                        "short": True
                    }
                ]
            }]
        }

        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=payload)

    async def send_teams_notification(self, alert: Dict[str, Any], 
                                    config: Dict[str, Any]):
        """Send Microsoft Teams notification"""
        webhook_url = config.get("webhook_url")
        if not webhook_url:
            return

        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "FF0000" if alert["severity"] == "critical" else "FF9900",
            "summary": f"Azure AI Copilot Alert: {alert['title']}",
            "sections": [{
                "activityTitle": f"🚨 {alert['title']}",
                "activitySubtitle": f"Severity: {alert['severity'].upper()}",
                "facts": [
                    {"name": "Resource", "value": alert.get("resource", "Unknown")},
                    {"name": "Description", "value": alert["description"]},
                    {"name": "Time", "value": alert["timestamp"]}
                ]
            }]
        }

        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=payload)

    async def send_email_notification(self, alert: Dict[str, Any], 
                                    config: Dict[str, Any]):
        """Send email notification"""
        # Would integrate with email service (SendGrid, AWS SES, etc.)
        logger.info(f"Email notification sent for alert: {alert['title']}")


# Usage example and monitoring service
class MonitoringService:
    """Main monitoring service coordinator"""

    def __init__(self):
        self.monitoring = AdvancedMonitoring()
        self.alert_manager = AlertManager(self.monitoring)
        self.running = False

    async def start_monitoring(self):
        """Start the monitoring service"""
        self.running = True
        
        # Setup notification channels
        self.alert_manager.add_notification_channel("slack", {
            "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        })
        
        # Start monitoring loop
        while self.running:
            try:
                # Update system health
                health_score = self.monitoring.calculate_system_health()
                self.monitoring.update_system_health(health_score)
                
                # Collect Azure metrics
                await self.monitoring.collect_azure_metrics()
                
                # Check for alert conditions
                self.monitoring.check_alert_conditions()
                
                # Process alerts
                await self.alert_manager.process_alerts()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.running = False

# Global monitoring instance
global_monitoring = AdvancedMonitoring()
