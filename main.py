#!/usr/bin/env python3
"""
Azure AI IT Copilot - Main Application Entry Point
Production-ready application launcher with proper initialization
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

# Import database
from database.connection import init_database, db_manager

# Import API server
from api.server import app as api_app

# Import authentication
from auth import azure_ad_auth

logger = logging.getLogger(__name__)


class ApplicationManager:
    """Manages application lifecycle and components"""

    def __init__(self):
        self.settings = get_settings()
        self.shutdown_event = asyncio.Event()
        self._components_started = False

    async def startup(self):
        """Initialize all application components"""
        try:
            logger.info("🚀 Starting Azure AI IT Copilot...")

            # Initialize database
            logger.info("📊 Initializing database...")
            if not init_database():
                raise RuntimeError("Database initialization failed")

            # Initialize Azure AD if configured
            if azure_ad_auth.enabled:
                logger.info("🔐 Azure AD authentication enabled")
            else:
                logger.warning("⚠️ Azure AD authentication disabled - using basic auth")

            # Initialize background tasks (if needed)
            await self._start_background_tasks()

            self._components_started = True
            logger.info("✅ Application startup completed successfully")

        except Exception as e:
            logger.error(f"❌ Application startup failed: {str(e)}")
            raise

    async def shutdown(self):
        """Cleanup application components"""
        try:
            logger.info("🛑 Shutting down Azure AI IT Copilot...")

            # Stop background tasks
            await self._stop_background_tasks()

            # Close database connections
            db_manager.close()

            # Set shutdown event
            self.shutdown_event.set()

            logger.info("✅ Application shutdown completed")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {str(e)}")

    async def _start_background_tasks(self):
        """Start background tasks like data cleanup, monitoring, etc."""
        # This would start any background tasks
        # For now, we'll just log that background tasks are ready
        logger.info("🔄 Background tasks initialized")

    async def _stop_background_tasks(self):
        """Stop background tasks gracefully"""
        logger.info("🔄 Stopping background tasks...")


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
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    if cors_origins == "*":
        origins = ["*"]
    else:
        origins = [origin.strip() for origin in cors_origins.split(",")]

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