"""
Database Connection Management
Handles database connections, sessions, and lifecycle
"""

import logging
import os
from typing import Generator, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from config.settings import get_settings
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection and session manager"""

    def __init__(self):
        """Initialize database manager"""
        self.settings = get_settings()
        self._engine = None
        self._session_factory = None
        self._initialized = False

    def initialize(self):
        """Initialize database connection"""
        if self._initialized:
            return

        try:
            # Determine database URL
            if hasattr(self.settings, 'database_url') and self.settings.database_url:
                database_url = self.settings.database_url
            else:
                # Fallback to SQLite for development
                database_url = "sqlite:///./azure_ai_copilot.db"
                logger.warning("No database URL configured, using SQLite fallback")

            # Create engine with appropriate settings
            if database_url.startswith("sqlite"):
                self._engine = create_engine(
                    database_url,
                    echo=self.settings.environment.value == "development",
                    connect_args={"check_same_thread": False}
                )
            else:
                # PostgreSQL or other databases
                self._engine = create_engine(
                    database_url,
                    echo=self.settings.environment.value == "development",
                    poolclass=QueuePool,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )

            # Create session factory
            self._session_factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self._engine
            )

            # Create tables if they don't exist
            Base.metadata.create_all(bind=self._engine)

            self._initialized = True
            logger.info("Database connection initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    @property
    def engine(self):
        """Get database engine"""
        if not self._initialized:
            self.initialize()
        return self._engine

    @property
    def session_factory(self):
        """Get session factory"""
        if not self._initialized:
            self.initialize()
        return self._session_factory

    def get_session(self) -> Session:
        """Get database session"""
        return self.session_factory()

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()

    def create_all_tables(self):
        """Create all database tables"""
        if not self._initialized:
            self.initialize()
        Base.metadata.create_all(bind=self._engine)
        logger.info("All database tables created")

    def drop_all_tables(self):
        """Drop all database tables"""
        if not self._initialized:
            self.initialize()
        Base.metadata.drop_all(bind=self._engine)
        logger.info("All database tables dropped")

    def reset_database(self):
        """Reset database by dropping and recreating all tables"""
        logger.warning("Resetting database - all data will be lost!")
        self.drop_all_tables()
        self.create_all_tables()

    def check_connection(self) -> bool:
        """Check if database connection is healthy"""
        try:
            with self.session_scope() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return False

    def get_table_info(self) -> dict:
        """Get information about database tables"""
        try:
            metadata = MetaData()
            metadata.reflect(bind=self.engine)

            table_info = {}
            for table_name, table in metadata.tables.items():
                table_info[table_name] = {
                    "columns": [col.name for col in table.columns],
                    "primary_keys": [col.name for col in table.primary_key.columns],
                    "foreign_keys": [
                        f"{fk.parent.name} -> {fk.column.table.name}.{fk.column.name}"
                        for fk in table.foreign_keys
                    ]
                }

            return table_info
        except Exception as e:
            logger.error(f"Failed to get table info: {str(e)}")
            return {}

    def close(self):
        """Close database connections"""
        if self._engine:
            self._engine.dispose()
            self._initialized = False
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get database session

    Usage:
        @app.get("/users/")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    with db_manager.session_scope() as session:
        yield session


def init_database():
    """Initialize database for application startup"""
    try:
        db_manager.initialize()

        # Check if database needs initial data
        with db_manager.session_scope() as session:
            from .models import User
            user_count = session.query(User).count()

            if user_count == 0:
                logger.info("Database appears to be empty, consider running initial data setup")

        logger.info("Database initialization completed")
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False


def create_admin_user(email: str, name: str, azure_id: str = None) -> bool:
    """Create initial admin user"""
    try:
        with db_manager.session_scope() as session:
            from .models import User

            # Check if admin user already exists
            existing_user = session.query(User).filter(User.email == email).first()
            if existing_user:
                logger.info(f"Admin user {email} already exists")
                return True

            # Create admin user
            admin_user = User(
                email=email,
                name=name,
                azure_id=azure_id,
                role="owner",
                is_active=True
            )

            session.add(admin_user)
            session.commit()

            logger.info(f"Admin user {email} created successfully")
            return True

    except Exception as e:
        logger.error(f"Failed to create admin user: {str(e)}")
        return False


def setup_default_configurations():
    """Setup default system configurations"""
    try:
        with db_manager.session_scope() as session:
            from .models import Configuration

            default_configs = [
                {
                    "key": "system.max_concurrent_commands",
                    "value": 10,
                    "description": "Maximum number of concurrent command executions",
                    "category": "system"
                },
                {
                    "key": "monitoring.metric_retention_days",
                    "value": 90,
                    "description": "Number of days to retain metric data",
                    "category": "monitoring"
                },
                {
                    "key": "cost.alert_threshold_percentage",
                    "value": 80,
                    "description": "Percentage threshold for cost alerts",
                    "category": "cost"
                },
                {
                    "key": "incidents.auto_resolve_after_hours",
                    "value": 24,
                    "description": "Auto-resolve incidents after specified hours",
                    "category": "incidents"
                },
                {
                    "key": "compliance.check_frequency_hours",
                    "value": 24,
                    "description": "Frequency of compliance checks in hours",
                    "category": "compliance"
                }
            ]

            for config_data in default_configs:
                existing_config = session.query(Configuration).filter(
                    Configuration.key == config_data["key"]
                ).first()

                if not existing_config:
                    config = Configuration(**config_data)
                    session.add(config)

            session.commit()
            logger.info("Default configurations setup completed")
            return True

    except Exception as e:
        logger.error(f"Failed to setup default configurations: {str(e)}")
        return False


# Utility functions for database operations
async def log_system_event(
    level: str,
    category: str,
    action: str,
    message: str,
    user_id: str = None,
    metadata: dict = None
):
    """Log system event to database"""
    try:
        with db_manager.session_scope() as session:
            from .models import SystemLog

            log_entry = SystemLog(
                log_level=level.upper(),
                category=category,
                action=action,
                message=message,
                user_id=user_id,
                metadata=metadata
            )

            session.add(log_entry)
            session.commit()

    except Exception as e:
        logger.error(f"Failed to log system event: {str(e)}")


async def record_command_execution(
    user_id: str,
    command: str,
    intent: str,
    result: dict,
    status: str,
    execution_time: float = None,
    error_message: str = None,
    context: dict = None
):
    """Record command execution in database"""
    try:
        with db_manager.session_scope() as session:
            from .models import CommandHistory

            history_entry = CommandHistory(
                user_id=user_id,
                command=command,
                intent=intent,
                context=context,
                result=result,
                status=status,
                execution_time=execution_time,
                error_message=error_message
            )

            session.add(history_entry)
            session.commit()

    except Exception as e:
        logger.error(f"Failed to record command execution: {str(e)}")


async def cleanup_old_data():
    """Cleanup old data based on retention policies"""
    try:
        with db_manager.session_scope() as session:
            from .models import ResourceMetric, SystemLog, CommandHistory
            from datetime import datetime, timedelta

            # Get retention settings
            metric_retention_days = 90  # Could be fetched from Configuration table
            log_retention_days = 30
            history_retention_days = 180

            cutoff_date_metrics = datetime.utcnow() - timedelta(days=metric_retention_days)
            cutoff_date_logs = datetime.utcnow() - timedelta(days=log_retention_days)
            cutoff_date_history = datetime.utcnow() - timedelta(days=history_retention_days)

            # Cleanup old metrics
            deleted_metrics = session.query(ResourceMetric).filter(
                ResourceMetric.created_at < cutoff_date_metrics
            ).delete()

            # Cleanup old logs
            deleted_logs = session.query(SystemLog).filter(
                SystemLog.created_at < cutoff_date_logs
            ).delete()

            # Cleanup old command history
            deleted_history = session.query(CommandHistory).filter(
                CommandHistory.created_at < cutoff_date_history
            ).delete()

            session.commit()

            logger.info(f"Data cleanup completed: {deleted_metrics} metrics, {deleted_logs} logs, {deleted_history} history records deleted")

    except Exception as e:
        logger.error(f"Data cleanup failed: {str(e)}")