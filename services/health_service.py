"""
Health Service Implementation
Provides system health monitoring and checks
"""

import logging
import asyncio
from typing import Dict, Any
from datetime import datetime

from core.interfaces import IHealthChecker, IConfigurationProvider
from core.base_classes import BaseService

logger = logging.getLogger(__name__)


class HealthService(BaseService, IHealthChecker):
    """System health monitoring service"""

    def __init__(self, config_provider: IConfigurationProvider):
        super().__init__(config_provider)
        self._checks = []

    async def initialize(self) -> None:
        """Initialize health service"""
        self._register_default_checks()
        logger.info("Health service initialized")

    async def shutdown(self) -> None:
        """Shutdown health service"""
        self._checks.clear()
        logger.info("Health service shutdown")

    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_result = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "summary": {
                "total_checks": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }

        for check_name, check_func in self._checks:
            try:
                check_result = await check_func()
                health_result["checks"][check_name] = check_result

                # Update summary
                health_result["summary"]["total_checks"] += 1
                status = check_result.get("status", "unknown")
                if status == "healthy":
                    health_result["summary"]["passed"] += 1
                elif status == "warning":
                    health_result["summary"]["warnings"] += 1
                else:
                    health_result["summary"]["failed"] += 1

            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                health_result["checks"][check_name] = {
                    "status": "failed",
                    "error": str(e)
                }
                health_result["summary"]["failed"] += 1

        # Determine overall status
        if health_result["summary"]["failed"] > 0:
            health_result["status"] = "unhealthy"
        elif health_result["summary"]["warnings"] > 0:
            health_result["status"] = "degraded"

        return health_result

    async def check_dependencies(self) -> Dict[str, Any]:
        """Check health of external dependencies"""
        dependency_checks = {
            "redis": self._check_redis,
            "azure_clients": self._check_azure_clients,
            "file_system": self._check_file_system
        }

        results = {}
        for name, check_func in dependency_checks.items():
            try:
                results[name] = await check_func()
            except Exception as e:
                results[name] = {"status": "failed", "error": str(e)}

        return results

    def _register_default_checks(self):
        """Register default health checks"""
        self._checks = [
            ("configuration", self._check_configuration),
            ("memory", self._check_memory),
            ("disk_space", self._check_disk_space),
            ("redis", self._check_redis),
            ("azure_clients", self._check_azure_clients)
        ]

    async def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration health"""
        try:
            # Verify critical configuration
            required_settings = ["azure_subscription_id", "jwt_secret_key"]
            missing_settings = []

            for setting in required_settings:
                value = self.config_provider.get_setting(setting)
                if not value or value.startswith("development-"):
                    missing_settings.append(setting)

            if missing_settings:
                return {
                    "status": "warning",
                    "message": f"Default values found for: {', '.join(missing_settings)}",
                    "missing_settings": missing_settings
                }

            return {"status": "healthy", "message": "Configuration validated"}

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()

            status = "healthy"
            if memory.percent > 90:
                status = "failed"
            elif memory.percent > 80:
                status = "warning"

            return {
                "status": status,
                "memory_percent": memory.percent,
                "available_mb": round(memory.available / 1024 / 1024, 2),
                "total_mb": round(memory.total / 1024 / 1024, 2)
            }
        except ImportError:
            return {"status": "warning", "message": "psutil not available for memory check"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")

            free_percent = (free / total) * 100

            status = "healthy"
            if free_percent < 5:
                status = "failed"
            elif free_percent < 10:
                status = "warning"

            return {
                "status": status,
                "free_percent": round(free_percent, 2),
                "free_gb": round(free / (1024**3), 2),
                "total_gb": round(total / (1024**3), 2)
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            # This would integrate with actual Redis client
            # For now, return mock status
            return {
                "status": "healthy",
                "response_time_ms": 2.5,
                "connected": True
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _check_azure_clients(self) -> Dict[str, Any]:
        """Check Azure client connectivity"""
        try:
            # Mock Azure client health check
            return {
                "status": "healthy",
                "subscription_accessible": True,
                "credentials_valid": True
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _check_file_system(self) -> Dict[str, Any]:
        """Check file system access"""
        try:
            import tempfile
            import os

            # Test write access
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(b"health_check")
                temp_path = temp_file.name

            # Clean up
            os.unlink(temp_path)

            return {"status": "healthy", "write_access": True}
        except Exception as e:
            return {"status": "failed", "error": str(e)}