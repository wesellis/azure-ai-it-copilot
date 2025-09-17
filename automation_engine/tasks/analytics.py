"""
Analytics Tasks Module
Handles cost anomaly detection, predictive analysis, and insights generation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from celery import Task
from celery.exceptions import Retry

from automation_engine.celery_app import celery_app
from config.settings import get_settings

logger = logging.getLogger(__name__)


class BaseAnalyticsTask(Task):
    """Base class for analytics tasks with common functionality"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Analytics task {task_id} failed: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Analytics task {task_id} retrying: {exc}")


@celery_app.task(bind=True, base=BaseAnalyticsTask, max_retries=3)
def detect_cost_anomalies(self, subscription_id: str, time_range_hours: int = 24) -> Dict[str, Any]:
    """
    Detect cost anomalies in Azure resources

    Args:
        subscription_id: Azure subscription ID
        time_range_hours: Time range to analyze in hours

    Returns:
        Dict containing anomaly detection results
    """
    try:
        settings = get_settings()

        # Mock anomaly detection logic
        # In production, this would use Azure Cost Management API
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_range_hours)

        anomalies = []

        # Simulate cost spike detection
        mock_resources = [
            {"name": "vm-prod-01", "cost_increase": 45.2, "threshold": 30.0},
            {"name": "storage-account-logs", "cost_increase": 15.8, "threshold": 20.0},
        ]

        for resource in mock_resources:
            if resource["cost_increase"] > resource["threshold"]:
                anomalies.append({
                    "resource_name": resource["name"],
                    "anomaly_type": "cost_spike",
                    "severity": "high" if resource["cost_increase"] > 40 else "medium",
                    "cost_increase_percent": resource["cost_increase"],
                    "threshold_percent": resource["threshold"],
                    "detected_at": datetime.utcnow().isoformat(),
                    "subscription_id": subscription_id
                })

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "analysis_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": time_range_hours
            },
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Cost anomaly detection completed for {subscription_id}: {len(anomalies)} anomalies found")
        return result

    except Exception as exc:
        logger.error(f"Cost anomaly detection failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseAnalyticsTask, max_retries=3)
def generate_cost_forecast(self, subscription_id: str, forecast_days: int = 30) -> Dict[str, Any]:
    """
    Generate cost forecast for Azure resources

    Args:
        subscription_id: Azure subscription ID
        forecast_days: Number of days to forecast

    Returns:
        Dict containing cost forecast
    """
    try:
        settings = get_settings()

        # Mock forecast generation
        # In production, this would use historical cost data and ML models
        current_daily_cost = 125.50
        growth_rate = 0.02  # 2% monthly growth

        forecast_data = []
        for day in range(1, forecast_days + 1):
            date = datetime.utcnow() + timedelta(days=day)
            projected_cost = current_daily_cost * (1 + (growth_rate * day / 30))

            forecast_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "projected_daily_cost": round(projected_cost, 2),
                "confidence": 0.85 - (day * 0.01)  # Decreasing confidence over time
            })

        total_forecast = sum(item["projected_daily_cost"] for item in forecast_data)

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "forecast_period_days": forecast_days,
            "total_projected_cost": round(total_forecast, 2),
            "daily_forecasts": forecast_data,
            "model_info": {
                "type": "linear_regression",
                "confidence_avg": 0.75,
                "last_trained": datetime.utcnow().isoformat()
            },
            "status": "completed",
            "generated_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Cost forecast generated for {subscription_id}: ${total_forecast:.2f} for {forecast_days} days")
        return result

    except Exception as exc:
        logger.error(f"Cost forecast generation failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseAnalyticsTask, max_retries=3)
def analyze_resource_utilization(self, subscription_id: str, resource_group: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze resource utilization patterns

    Args:
        subscription_id: Azure subscription ID
        resource_group: Optional resource group filter

    Returns:
        Dict containing utilization analysis
    """
    try:
        settings = get_settings()

        # Mock utilization analysis
        # In production, this would query Azure Monitor metrics
        resources = [
            {
                "resource_id": "/subscriptions/sub1/resourceGroups/rg1/providers/Microsoft.Compute/virtualMachines/vm-prod-01",
                "resource_type": "Microsoft.Compute/virtualMachines",
                "name": "vm-prod-01",
                "cpu_utilization_avg": 35.2,
                "memory_utilization_avg": 67.8,
                "disk_utilization_avg": 45.1,
                "network_utilization_avg": 23.4,
                "utilization_score": 42.9,
                "recommendation": "Consider downsizing VM size"
            },
            {
                "resource_id": "/subscriptions/sub1/resourceGroups/rg1/providers/Microsoft.Storage/storageAccounts/storage01",
                "resource_type": "Microsoft.Storage/storageAccounts",
                "name": "storage01",
                "capacity_utilization": 78.5,
                "transaction_rate": 1250,
                "utilization_score": 78.5,
                "recommendation": "Monitor capacity growth"
            }
        ]

        # Calculate summary statistics
        total_resources = len(resources)
        avg_utilization = sum(r["utilization_score"] for r in resources) / total_resources

        underutilized = [r for r in resources if r["utilization_score"] < 30]
        overutilized = [r for r in resources if r["utilization_score"] > 80]

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "analysis_summary": {
                "total_resources_analyzed": total_resources,
                "average_utilization": round(avg_utilization, 2),
                "underutilized_count": len(underutilized),
                "overutilized_count": len(overutilized)
            },
            "resource_details": resources,
            "recommendations": {
                "cost_savings_potential": "$245.50/month",
                "optimization_actions": len(underutilized) + len(overutilized),
                "priority_resources": [r["name"] for r in underutilized + overutilized]
            },
            "status": "completed",
            "analyzed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Resource utilization analyzed for {subscription_id}: {total_resources} resources")
        return result

    except Exception as exc:
        logger.error(f"Resource utilization analysis failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseAnalyticsTask, max_retries=3)
def generate_insights_report(self, subscription_id: str, report_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Generate comprehensive insights report

    Args:
        subscription_id: Azure subscription ID
        report_type: Type of report (comprehensive, cost, security, performance)

    Returns:
        Dict containing insights report
    """
    try:
        settings = get_settings()

        # Mock insights generation
        insights = {
            "cost_insights": [
                {
                    "type": "savings_opportunity",
                    "title": "Unused Virtual Machines",
                    "description": "3 VMs have been idle for over 7 days",
                    "potential_savings": "$180.00/month",
                    "priority": "high"
                },
                {
                    "type": "optimization",
                    "title": "Storage Tier Optimization",
                    "description": "Move infrequently accessed data to cool storage",
                    "potential_savings": "$65.00/month",
                    "priority": "medium"
                }
            ],
            "security_insights": [
                {
                    "type": "vulnerability",
                    "title": "Public IP Exposure",
                    "description": "2 VMs have unnecessary public IP addresses",
                    "risk_level": "medium",
                    "priority": "high"
                }
            ],
            "performance_insights": [
                {
                    "type": "bottleneck",
                    "title": "Database Performance",
                    "description": "SQL Database showing high DTU utilization",
                    "impact": "medium",
                    "priority": "medium"
                }
            ]
        }

        # Filter insights based on report type
        if report_type != "comprehensive":
            insights = {f"{report_type}_insights": insights.get(f"{report_type}_insights", [])}

        total_insights = sum(len(category) for category in insights.values())

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "report_type": report_type,
            "insights": insights,
            "summary": {
                "total_insights": total_insights,
                "high_priority": sum(1 for category in insights.values()
                                   for insight in category if insight.get("priority") == "high"),
                "potential_monthly_savings": "$245.00"
            },
            "status": "completed",
            "generated_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Insights report generated for {subscription_id}: {total_insights} insights")
        return result

    except Exception as exc:
        logger.error(f"Insights report generation failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseAnalyticsTask, max_retries=3)
def trend_analysis(self, subscription_id: str, metric_type: str, days: int = 30) -> Dict[str, Any]:
    """
    Perform trend analysis on various metrics

    Args:
        subscription_id: Azure subscription ID
        metric_type: Type of metric to analyze (cost, usage, performance)
        days: Number of days to analyze

    Returns:
        Dict containing trend analysis
    """
    try:
        settings = get_settings()

        # Mock trend analysis
        # In production, this would analyze historical data
        base_value = 100.0
        trend_data = []

        for day in range(days):
            date = datetime.utcnow() - timedelta(days=(days - day - 1))
            # Simulate some variation in the data
            variation = (day % 7) * 0.1 + (day / days) * 0.2
            value = base_value * (1 + variation)

            trend_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(value, 2),
                "metric_type": metric_type
            })

        # Calculate trend
        first_week_avg = sum(d["value"] for d in trend_data[:7]) / 7
        last_week_avg = sum(d["value"] for d in trend_data[-7:]) / 7
        trend_percentage = ((last_week_avg - first_week_avg) / first_week_avg) * 100

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "metric_type": metric_type,
            "analysis_period_days": days,
            "trend_data": trend_data,
            "trend_analysis": {
                "trend_direction": "increasing" if trend_percentage > 0 else "decreasing",
                "trend_percentage": round(trend_percentage, 2),
                "first_week_average": round(first_week_avg, 2),
                "last_week_average": round(last_week_avg, 2),
                "volatility": "low"  # Mock volatility calculation
            },
            "status": "completed",
            "analyzed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Trend analysis completed for {subscription_id}: {metric_type} trend is {trend_percentage:.2f}%")
        return result

    except Exception as exc:
        logger.error(f"Trend analysis failed: {exc}")
        raise self.retry(exc=exc, countdown=60)