#!/usr/bin/env python3
"""
Azure AI IT Copilot - Main Application Entry Point
Production-ready modular application launcher with dependency injection
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Import configuration and logging
from config.settings import get_settings
from logging_config import setup_logging

# Import modular application
from core.application import get_application, initialize_application, shutdown_application
from core.dependency_injection import get_container

# Import database
from database.connection import init_database, db_manager

# Import API server
from api.server import app as api_app

# Import authentication
from auth import azure_ad_auth

logger = logging.getLogger(__name__)


class ApplicationManager:
    """Enhanced application lifecycle manager with modular architecture"""

    def __init__(self):
        self.settings = get_settings()
        self.shutdown_event = asyncio.Event()
        self.application = get_application()
        self._components_started = False

    async def startup(self):
        """Initialize all application components with modular architecture"""
        try:
            logger.info("🚀 Starting Azure AI IT Copilot with modular architecture...")

            # Initialize modular application
            logger.info("🔧 Initializing modular application...")
            await initialize_application()

            # Initialize database
            logger.info("📊 Initializing database...")
            if not init_database():
                raise RuntimeError("Database initialization failed")

            # Initialize Azure AD if configured
            if azure_ad_auth.enabled:
                logger.info("🔐 Azure AD authentication enabled")
            else:
                logger.warning("⚠️ Azure AD authentication disabled - using basic auth")

            # Initialize background tasks with DI container
            await self._start_background_tasks()

            self._components_started = True
            logger.info("✅ Modular application startup completed successfully")

            # Log plugin information
            plugin_info = self.application.get_plugin_info()
            logger.info(f"📦 Loaded {plugin_info['total_plugins']} plugins")

        except Exception as e:
            logger.error(f"❌ Application startup failed: {str(e)}")
            raise

    async def shutdown(self):
        """Cleanup application components with modular shutdown"""
        try:
            logger.info("🛑 Shutting down Azure AI IT Copilot...")

            # Stop background tasks
            await self._stop_background_tasks()

            # Shutdown modular application
            await shutdown_application()

            # Close database connections
            db_manager.close()

            # Set shutdown event
            self.shutdown_event.set()

            logger.info("✅ Modular application shutdown completed")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {str(e)}")

    async def _start_background_tasks(self):
        """Start background tasks using dependency injection"""
        try:
            container = get_container()

            # Initialize task queue service if available
            from core.interfaces import ITaskQueue
            if container.is_registered(ITaskQueue):
                task_queue = container.resolve(ITaskQueue)
                await task_queue.initialize()
                logger.info("🔄 Task queue service initialized")

            # Initialize other background services
            logger.info("🔄 Background tasks initialized with DI")

        except Exception as e:
            logger.warning(f"⚠️ Background tasks initialization warning: {e}")

    async def _stop_background_tasks(self):
        """Stop background tasks gracefully"""
        try:
            container = get_container()

            # Shutdown task queue service if available
            from core.interfaces import ITaskQueue
            if container.is_registered(ITaskQueue):
                task_queue = container.resolve(ITaskQueue)
                await task_queue.shutdown()

            logger.info("🔄 Background tasks stopped")

        except Exception as e:
            logger.warning(f"⚠️ Background tasks shutdown warning: {e}")


# Global application manager
app_manager = ApplicationManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await app_manager.startup()
    yield
    # Shutdown
    await app_manager.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""

    # Create main app with lifespan
    app = FastAPI(
        title="Azure AI IT Copilot",
        description="AI-powered Azure infrastructure management and automation platform",
        version="1.0.0",
        docs_url="/docs" if app_manager.settings.environment.value != "production" else None,
        redoc_url="/redoc" if app_manager.settings.environment.value != "production" else None,
        lifespan=lifespan
    )

    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure properly for production
    )

    # Add CORS middleware
    origins = app_manager.settings.get_cors_origins_list()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the API app
    app.mount("/api", api_app)

    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint"""
        return {
            "status": "healthy",
            "service": "azure-ai-it-copilot",
            "version": "1.0.0",
            "environment": app_manager.settings.environment.value
        }

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Azure AI IT Copilot API",
            "version": "1.0.0",
            "docs": "/docs" if app_manager.settings.environment.value != "production" else "disabled",
            "health": "/health"
        }

    return app


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(app_manager.shutdown())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main application entry point"""
    try:
        # Setup logging first
        setup_logging()
        logger.info("🔧 Logging configured")

        # Setup signal handlers
        setup_signal_handlers()

        # Get settings
        settings = get_settings()

        # Create application
        app = create_app()

        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=getattr(settings, 'api_host', '0.0.0.0'),
            port=getattr(settings, 'api_port', 8000),
            log_level=getattr(settings, 'log_level', 'info').lower(),
            access_log=True,
            reload=settings.environment.value == "development",
            workers=1 if settings.environment.value == "development" else 4,
        )

        # Start server
        server = uvicorn.Server(config)

        logger.info(f"🌐 Starting server on {config.host}:{config.port}")
        logger.info(f"📋 Environment: {settings.environment.value}")
        logger.info(f"📖 API docs: http://{config.host}:{config.port}/docs")

        server.run()

    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"❌ Application failed to start: {str(e)}")
        sys.exit(1)


def run_development():
    """Run in development mode with auto-reload"""
    os.environ["ENVIRONMENT"] = "development"
    main()


def run_production():
    """Run in production mode"""
    os.environ["ENVIRONMENT"] = "production"
    main()


if __name__ == "__main__":
    main()