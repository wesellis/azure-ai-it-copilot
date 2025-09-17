"""
Task Queue Service Implementation
Provides background task management
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import uuid

from core.interfaces import ITaskQueue, IConfigurationProvider
from core.base_classes import BaseService
from core.exceptions import TaskExecutionError

logger = logging.getLogger(__name__)


class TaskQueueService(BaseService, ITaskQueue):
    """Background task queue service"""

    def __init__(self, config_provider: IConfigurationProvider):
        super().__init__(config_provider)
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_results: Dict[str, Dict[str, Any]] = {}
        self._task_queue = asyncio.Queue()
        self._worker_tasks: list = []

    async def initialize(self) -> None:
        """Initialize task queue service"""
        # Start worker tasks
        worker_count = self.config_provider.get_setting("worker_processes", 2)
        for i in range(worker_count):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._worker_tasks.append(worker)

        logger.info(f"Task queue service initialized with {worker_count} workers")

    async def shutdown(self) -> None:
        """Shutdown task queue service"""
        # Cancel all running tasks
        for task_id, task in self._running_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled task {task_id}")

        # Cancel worker tasks
        for worker in self._worker_tasks:
            worker.cancel()

        # Wait for all tasks to complete
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        self._running_tasks.clear()
        self._task_results.clear()
        logger.info("Task queue service shutdown")

    async def enqueue_task(self, task_name: str, args: tuple = (), kwargs: Dict[str, Any] = None) -> str:
        """Enqueue a background task"""
        try:
            task_id = str(uuid.uuid4())
            kwargs = kwargs or {}

            task_item = {
                "task_id": task_id,
                "task_name": task_name,
                "args": args,
                "kwargs": kwargs,
                "enqueued_at": datetime.utcnow().isoformat(),
                "status": "queued"
            }

            await self._task_queue.put(task_item)

            # Store initial status
            self._task_results[task_id] = {
                "task_id": task_id,
                "task_name": task_name,
                "status": "queued",
                "enqueued_at": task_item["enqueued_at"],
                "result": None,
                "error": None
            }

            logger.info(f"Task {task_name} enqueued with ID {task_id}")
            return task_id

        except Exception as e:
            logger.error(f"Failed to enqueue task {task_name}: {e}")
            raise TaskExecutionError(f"Failed to enqueue task: {e}", task_name=task_name)

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task"""
        if task_id not in self._task_results:
            return {
                "task_id": task_id,
                "status": "not_found",
                "error": "Task not found"
            }

        return self._task_results[task_id].copy()

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        try:
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                if not task.done():
                    task.cancel()

                    # Update status
                    if task_id in self._task_results:
                        self._task_results[task_id].update({
                            "status": "cancelled",
                            "cancelled_at": datetime.utcnow().isoformat()
                        })

                    logger.info(f"Task {task_id} cancelled")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    async def _worker(self, worker_name: str):
        """Worker coroutine to process tasks"""
        logger.info(f"Worker {worker_name} started")

        try:
            while True:
                try:
                    # Get task from queue
                    task_item = await self._task_queue.get()
                    task_id = task_item["task_id"]

                    logger.info(f"Worker {worker_name} processing task {task_id}")

                    # Update status to running
                    self._task_results[task_id].update({
                        "status": "running",
                        "started_at": datetime.utcnow().isoformat(),
                        "worker": worker_name
                    })

                    # Execute task
                    task_coro = self._execute_task(task_item)
                    task = asyncio.create_task(task_coro)
                    self._running_tasks[task_id] = task

                    try:
                        result = await task

                        # Update status to completed
                        self._task_results[task_id].update({
                            "status": "completed",
                            "completed_at": datetime.utcnow().isoformat(),
                            "result": result,
                            "error": None
                        })

                        logger.info(f"Task {task_id} completed successfully")

                    except asyncio.CancelledError:
                        logger.info(f"Task {task_id} was cancelled")
                        self._task_results[task_id].update({
                            "status": "cancelled",
                            "cancelled_at": datetime.utcnow().isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Task {task_id} failed: {e}")
                        self._task_results[task_id].update({
                            "status": "failed",
                            "failed_at": datetime.utcnow().isoformat(),
                            "error": str(e)
                        })
                    finally:
                        # Clean up
                        if task_id in self._running_tasks:
                            del self._running_tasks[task_id]

                    # Mark task as done in queue
                    self._task_queue.task_done()

                except asyncio.CancelledError:
                    logger.info(f"Worker {worker_name} cancelled")
                    break
                except Exception as e:
                    logger.error(f"Worker {worker_name} error: {e}")
                    # Continue processing other tasks

        except asyncio.CancelledError:
            logger.info(f"Worker {worker_name} stopped")

    async def _execute_task(self, task_item: Dict[str, Any]) -> Any:
        """Execute a specific task"""
        task_name = task_item["task_name"]
        args = task_item["args"]
        kwargs = task_item["kwargs"]

        # Map task names to actual functions
        task_functions = {
            "analyze_costs": self._analyze_costs_task,
            "generate_report": self._generate_report_task,
            "backup_data": self._backup_data_task,
            "cleanup_resources": self._cleanup_resources_task,
            "send_notification": self._send_notification_task,
            "health_check": self._health_check_task
        }

        task_func = task_functions.get(task_name)
        if not task_func:
            raise TaskExecutionError(f"Unknown task: {task_name}", task_name=task_name)

        return await task_func(*args, **kwargs)

    async def _analyze_costs_task(self, subscription_id: str, **kwargs) -> Dict[str, Any]:
        """Background cost analysis task"""
        # Simulate cost analysis
        await asyncio.sleep(2)  # Simulate processing time

        return {
            "subscription_id": subscription_id,
            "total_cost": 1245.67,
            "cost_breakdown": {
                "compute": 678.90,
                "storage": 234.56,
                "network": 123.45,
                "other": 208.76
            },
            "optimizations": [
                {"type": "rightsizing", "savings": 156.78},
                {"type": "storage_tier", "savings": 89.23}
            ],
            "analysis_date": datetime.utcnow().isoformat()
        }

    async def _generate_report_task(self, report_type: str, **kwargs) -> Dict[str, Any]:
        """Background report generation task"""
        # Simulate report generation
        await asyncio.sleep(3)  # Simulate processing time

        return {
            "report_type": report_type,
            "generated_at": datetime.utcnow().isoformat(),
            "data": {
                "summary": "Report generated successfully",
                "metrics": {"total_resources": 45, "healthy": 42, "warning": 3},
                "recommendations": ["Optimize VM sizes", "Review storage policies"]
            }
        }

    async def _backup_data_task(self, resource_id: str, **kwargs) -> Dict[str, Any]:
        """Background backup task"""
        # Simulate backup operation
        await asyncio.sleep(5)  # Simulate backup time

        return {
            "resource_id": resource_id,
            "backup_id": f"backup_{uuid.uuid4()}",
            "backup_size_gb": 12.5,
            "backup_location": f"/backups/{resource_id}/backup.vhd",
            "backup_completed_at": datetime.utcnow().isoformat()
        }

    async def _cleanup_resources_task(self, dry_run: bool = True, **kwargs) -> Dict[str, Any]:
        """Background resource cleanup task"""
        # Simulate cleanup operation
        await asyncio.sleep(4)  # Simulate cleanup time

        cleaned_resources = [
            {"name": "unused-vm-01", "type": "vm", "cost_savings": 75.50},
            {"name": "old-storage-account", "type": "storage", "cost_savings": 23.40}
        ]

        return {
            "dry_run": dry_run,
            "resources_cleaned": len(cleaned_resources),
            "total_savings": sum(r["cost_savings"] for r in cleaned_resources),
            "cleaned_resources": cleaned_resources,
            "cleanup_completed_at": datetime.utcnow().isoformat()
        }

    async def _send_notification_task(self, message: str, recipients: list, **kwargs) -> Dict[str, Any]:
        """Background notification task"""
        # Simulate notification sending
        await asyncio.sleep(1)  # Simulate sending time

        return {
            "message": message,
            "recipients_count": len(recipients),
            "delivery_status": "sent",
            "sent_at": datetime.utcnow().isoformat()
        }

    async def _health_check_task(self, **kwargs) -> Dict[str, Any]:
        """Background health check task"""
        # Simulate health check
        await asyncio.sleep(2)  # Simulate check time

        return {
            "overall_health": "healthy",
            "checks_passed": 8,
            "checks_failed": 1,
            "services": {
                "api": "healthy",
                "database": "healthy",
                "redis": "healthy",
                "azure_clients": "warning"
            },
            "checked_at": datetime.utcnow().isoformat()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on task queue service"""
        health_status = await super().health_check()

        health_status.update({
            "queue_size": self._task_queue.qsize(),
            "running_tasks": len(self._running_tasks),
            "total_tasks_processed": len(self._task_results),
            "workers_active": len([w for w in self._worker_tasks if not w.done()])
        })

        return health_status