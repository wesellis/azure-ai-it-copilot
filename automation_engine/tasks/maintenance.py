"""
Maintenance Tasks Module
Handles system maintenance, cleanup, optimization, and routine operations
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from celery import Task
from celery.exceptions import Retry

from automation_engine.celery_app import celery_app
from config.settings import get_settings

logger = logging.getLogger(__name__)


class BaseMaintenanceTask(Task):
    """Base class for maintenance tasks with common functionality"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Maintenance task {task_id} failed: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Maintenance task {task_id} retrying: {exc}")


@celery_app.task(bind=True, base=BaseMaintenanceTask, max_retries=3)
def cleanup_unused_resources(self, subscription_id: str, dry_run: bool = True) -> Dict[str, Any]:
    """
    Clean up unused Azure resources

    Args:
        subscription_id: Azure subscription ID
        dry_run: If True, only identify resources without deleting

    Returns:
        Dict containing cleanup results
    """
    try:
        settings = get_settings()

        # Mock unused resource detection
        # In production, this would query Azure Resource Manager
        unused_resources = [
            {
                "resource_id": "/subscriptions/sub1/resourceGroups/rg1/providers/Microsoft.Compute/virtualMachines/vm-test-01",
                "name": "vm-test-01",
                "type": "Microsoft.Compute/virtualMachines",
                "location": "eastus",
                "cost_per_month": 75.50,
                "last_activity": "2024-01-01T00:00:00Z",
                "tags": {"environment": "test", "project": "archived"},
                "reason": "No activity for 30 days"
            },
            {
                "resource_id": "/subscriptions/sub1/resourceGroups/rg1/providers/Microsoft.Storage/storageAccounts/tempstorage123",
                "name": "tempstorage123",
                "type": "Microsoft.Storage/storageAccounts",
                "location": "eastus",
                "cost_per_month": 12.30,
                "last_activity": "2024-01-15T00:00:00Z",
                "tags": {"temporary": "true"},
                "reason": "Marked as temporary, unused for 15 days"
            }
        ]

        actions_taken = []
        total_savings = 0

        for resource in unused_resources:
            action = {
                "resource_id": resource["resource_id"],
                "resource_name": resource["name"],
                "action": "identified" if dry_run else "deleted",
                "cost_savings": resource["cost_per_month"],
                "timestamp": datetime.utcnow().isoformat()
            }

            if not dry_run:
                # Mock deletion logic
                # In production, this would use Azure SDK to delete resources
                logger.info(f"Would delete resource: {resource['name']}")
                action["action"] = "deleted"

            actions_taken.append(action)
            total_savings += resource["cost_per_month"]

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "dry_run": dry_run,
            "resources_found": len(unused_resources),
            "resources_processed": len(actions_taken),
            "total_potential_savings": round(total_savings, 2),
            "actions_taken": actions_taken,
            "unused_resources": unused_resources,
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Cleanup completed for {subscription_id}: {len(unused_resources)} resources, ${total_savings:.2f} savings")
        return result

    except Exception as exc:
        logger.error(f"Resource cleanup failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseMaintenanceTask, max_retries=3)
def optimize_storage_tiers(self, subscription_id: str, storage_account: Optional[str] = None) -> Dict[str, Any]:
    """
    Optimize storage tiers based on access patterns

    Args:
        subscription_id: Azure subscription ID
        storage_account: Optional specific storage account to optimize

    Returns:
        Dict containing optimization results
    """
    try:
        settings = get_settings()

        # Mock storage analysis
        # In production, this would analyze blob access patterns
        storage_accounts = [
            {
                "account_name": "prodstorageacct",
                "resource_group": "rg-production",
                "current_tier": "Hot",
                "recommended_tier": "Cool",
                "blob_count": 1245,
                "total_size_gb": 567.8,
                "last_access_days": 45,
                "monthly_savings": 85.20,
                "optimization_confidence": 0.92
            },
            {
                "account_name": "logsstorageacct",
                "resource_group": "rg-logging",
                "current_tier": "Hot",
                "recommended_tier": "Archive",
                "blob_count": 3421,
                "total_size_gb": 1234.5,
                "last_access_days": 120,
                "monthly_savings": 245.60,
                "optimization_confidence": 0.98
            }
        ]

        if storage_account:
            storage_accounts = [sa for sa in storage_accounts if sa["account_name"] == storage_account]

        optimizations = []
        total_savings = 0

        for account in storage_accounts:
            if account["current_tier"] != account["recommended_tier"]:
                optimization = {
                    "account_name": account["account_name"],
                    "resource_group": account["resource_group"],
                    "current_tier": account["current_tier"],
                    "recommended_tier": account["recommended_tier"],
                    "blob_count": account["blob_count"],
                    "size_gb": account["total_size_gb"],
                    "monthly_savings": account["monthly_savings"],
                    "confidence": account["optimization_confidence"],
                    "action": "tier_change_recommended",
                    "processed_at": datetime.utcnow().isoformat()
                }
                optimizations.append(optimization)
                total_savings += account["monthly_savings"]

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "storage_account_filter": storage_account,
            "accounts_analyzed": len(storage_accounts),
            "optimizations_identified": len(optimizations),
            "total_monthly_savings": round(total_savings, 2),
            "optimizations": optimizations,
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Storage optimization completed for {subscription_id}: ${total_savings:.2f} potential savings")
        return result

    except Exception as exc:
        logger.error(f"Storage optimization failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseMaintenanceTask, max_retries=3)
def update_resource_tags(self, subscription_id: str, tag_policy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update resource tags based on policy

    Args:
        subscription_id: Azure subscription ID
        tag_policy: Tag policy configuration

    Returns:
        Dict containing tag update results
    """
    try:
        settings = get_settings()

        # Mock resource tagging
        # In production, this would use Azure Resource Manager
        resources = [
            {
                "resource_id": "/subscriptions/sub1/resourceGroups/rg1/providers/Microsoft.Compute/virtualMachines/vm-prod-01",
                "name": "vm-prod-01",
                "current_tags": {"environment": "prod"},
                "missing_tags": ["owner", "project", "cost-center"],
                "invalid_tags": []
            },
            {
                "resource_id": "/subscriptions/sub1/resourceGroups/rg1/providers/Microsoft.Storage/storageAccounts/storage01",
                "name": "storage01",
                "current_tags": {"environment": "production", "owner": "team-a"},
                "missing_tags": ["project", "cost-center"],
                "invalid_tags": ["environment"]  # Should be "prod" not "production"
            }
        ]

        required_tags = tag_policy.get("required_tags", ["environment", "owner", "project", "cost-center"])
        tag_values = tag_policy.get("tag_values", {})

        updates = []
        total_resources_updated = 0

        for resource in resources:
            updates_needed = []

            # Check for missing required tags
            for tag in required_tags:
                if tag not in resource["current_tags"]:
                    updates_needed.append({
                        "action": "add",
                        "tag": tag,
                        "value": tag_values.get(tag, f"update-required-{tag}")
                    })

            # Check for invalid tag values
            for tag, value in resource["current_tags"].items():
                if tag in tag_values and value != tag_values[tag]:
                    updates_needed.append({
                        "action": "update",
                        "tag": tag,
                        "old_value": value,
                        "new_value": tag_values[tag]
                    })

            if updates_needed:
                update_record = {
                    "resource_id": resource["resource_id"],
                    "resource_name": resource["name"],
                    "updates": updates_needed,
                    "status": "completed",
                    "updated_at": datetime.utcnow().isoformat()
                }
                updates.append(update_record)
                total_resources_updated += 1

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "tag_policy": tag_policy,
            "resources_analyzed": len(resources),
            "resources_updated": total_resources_updated,
            "updates": updates,
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Tag updates completed for {subscription_id}: {total_resources_updated} resources updated")
        return result

    except Exception as exc:
        logger.error(f"Tag update failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseMaintenanceTask, max_retries=3)
def vm_rightsizing_analysis(self, subscription_id: str, resource_group: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze VMs for rightsizing opportunities

    Args:
        subscription_id: Azure subscription ID
        resource_group: Optional resource group filter

    Returns:
        Dict containing rightsizing analysis
    """
    try:
        settings = get_settings()

        # Mock VM rightsizing analysis
        # In production, this would analyze VM metrics from Azure Monitor
        vms = [
            {
                "vm_name": "vm-prod-01",
                "resource_group": "rg-production",
                "current_size": "Standard_D4s_v3",
                "current_cost": 140.16,
                "cpu_utilization_avg": 25.3,
                "memory_utilization_avg": 45.2,
                "recommended_size": "Standard_D2s_v3",
                "recommended_cost": 70.08,
                "monthly_savings": 70.08,
                "confidence": 0.85,
                "analysis_period_days": 30
            },
            {
                "vm_name": "vm-prod-02",
                "resource_group": "rg-production",
                "current_size": "Standard_B2s",
                "current_cost": 30.37,
                "cpu_utilization_avg": 75.8,
                "memory_utilization_avg": 82.1,
                "recommended_size": "Standard_B4ms",
                "recommended_cost": 60.74,
                "monthly_savings": -30.37,  # Negative = cost increase
                "confidence": 0.90,
                "analysis_period_days": 30
            }
        ]

        if resource_group:
            vms = [vm for vm in vms if vm["resource_group"] == resource_group]

        recommendations = []
        total_potential_savings = 0

        for vm in vms:
            recommendation = {
                "vm_name": vm["vm_name"],
                "resource_group": vm["resource_group"],
                "current_size": vm["current_size"],
                "recommended_size": vm["recommended_size"],
                "current_monthly_cost": vm["current_cost"],
                "recommended_monthly_cost": vm["recommended_cost"],
                "monthly_savings": vm["monthly_savings"],
                "cpu_utilization": vm["cpu_utilization_avg"],
                "memory_utilization": vm["memory_utilization_avg"],
                "confidence": vm["confidence"],
                "recommendation_type": "downsize" if vm["monthly_savings"] > 0 else "upsize",
                "priority": "high" if abs(vm["monthly_savings"]) > 50 else "medium",
                "analyzed_at": datetime.utcnow().isoformat()
            }
            recommendations.append(recommendation)
            total_potential_savings += vm["monthly_savings"]

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "resource_group_filter": resource_group,
            "vms_analyzed": len(vms),
            "recommendations": recommendations,
            "summary": {
                "total_monthly_savings": round(total_potential_savings, 2),
                "downsize_opportunities": len([r for r in recommendations if r["recommendation_type"] == "downsize"]),
                "upsize_recommendations": len([r for r in recommendations if r["recommendation_type"] == "upsize"]),
                "high_priority_count": len([r for r in recommendations if r["priority"] == "high"])
            },
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"VM rightsizing analysis completed for {subscription_id}: ${total_potential_savings:.2f} potential savings")
        return result

    except Exception as exc:
        logger.error(f"VM rightsizing analysis failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, base=BaseMaintenanceTask, max_retries=3)
def database_maintenance(self, subscription_id: str, maintenance_type: str = "optimization") -> Dict[str, Any]:
    """
    Perform database maintenance operations

    Args:
        subscription_id: Azure subscription ID
        maintenance_type: Type of maintenance (optimization, backup, cleanup)

    Returns:
        Dict containing maintenance results
    """
    try:
        settings = get_settings()

        # Mock database maintenance
        databases = [
            {
                "database_name": "prod-sql-01",
                "server_name": "sql-server-prod",
                "resource_group": "rg-databases",
                "type": "Azure SQL Database",
                "size_gb": 125.5,
                "current_tier": "S2",
                "dtu_utilization": 65.2
            },
            {
                "database_name": "logs-cosmosdb",
                "server_name": "cosmos-prod",
                "resource_group": "rg-databases",
                "type": "Cosmos DB",
                "size_gb": 45.8,
                "current_tier": "400 RU/s",
                "ru_utilization": 35.1
            }
        ]

        maintenance_results = []

        for db in databases:
            if maintenance_type == "optimization":
                result = {
                    "database_name": db["database_name"],
                    "server_name": db["server_name"],
                    "maintenance_type": "optimization",
                    "actions_performed": [
                        "Index analysis completed",
                        "Query performance reviewed",
                        "Statistics updated"
                    ],
                    "performance_improvement": "15% query performance increase",
                    "recommendations": [
                        "Consider adding index on frequently queried columns",
                        "Review and optimize top 10 slowest queries"
                    ]
                }
            elif maintenance_type == "backup":
                result = {
                    "database_name": db["database_name"],
                    "server_name": db["server_name"],
                    "maintenance_type": "backup",
                    "actions_performed": [
                        "Full backup completed",
                        "Backup integrity verified",
                        "Backup retention policy applied"
                    ],
                    "backup_size_gb": db["size_gb"] * 0.7,  # Compressed size
                    "backup_duration_minutes": 12
                }
            else:  # cleanup
                result = {
                    "database_name": db["database_name"],
                    "server_name": db["server_name"],
                    "maintenance_type": "cleanup",
                    "actions_performed": [
                        "Old backup files cleaned",
                        "Temporary tables removed",
                        "Log files truncated"
                    ],
                    "space_freed_gb": 8.5,
                    "cleanup_duration_minutes": 5
                }

            result.update({
                "status": "completed",
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat()
            })
            maintenance_results.append(result)

        result = {
            "task_id": self.request.id,
            "subscription_id": subscription_id,
            "maintenance_type": maintenance_type,
            "databases_processed": len(databases),
            "maintenance_results": maintenance_results,
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Database maintenance completed for {subscription_id}: {len(databases)} databases processed")
        return result

    except Exception as exc:
        logger.error(f"Database maintenance failed: {exc}")
        raise self.retry(exc=exc, countdown=60)