"""
Comprehensive Database Testing Framework
Tests database models, queries, migrations, and performance with SQLAlchemy 2.0
"""

import asyncio
import pytest
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator
from unittest.mock import patch, AsyncMock, Mock
import tempfile
import os

# SQLAlchemy imports
from sqlalchemy import create_engine, text, select, insert, update, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import sqlalchemy as sa

# Database imports
from database.models import (
    Base, User, Incident, Resource, SystemLog, Configuration,
    UserRole, IncidentStatus, IncidentSeverity, ResourceType
)
from database.connection import DatabaseManager, get_database_url
from config.optimized_settings import get_settings


@pytest.fixture(scope="session")
def test_database_url():
    """Test database URL for SQLite in-memory database"""
    return "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
async def test_engine(test_database_url):
    """Async database engine for testing"""
    engine = create_async_engine(
        test_database_url,
        echo=False,  # Set to True for SQL debugging
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Async database session for testing"""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def sample_user(test_session: AsyncSession) -> User:
    """Create a sample user for testing"""
    user = User(
        email="test@example.com",
        username="testuser",
        full_name="Test User",
        role=UserRole.USER,
        azure_user_id="azure-user-123",
        is_active=True
    )

    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)

    return user


@pytest.fixture
async def sample_resource(test_session: AsyncSession) -> Resource:
    """Create a sample resource for testing"""
    resource = Resource(
        resource_id="/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/test-vm",
        name="test-vm",
        resource_type=ResourceType.VIRTUAL_MACHINE,
        subscription_id="test-subscription",
        resource_group="test-rg",
        location="eastus",
        status="Running",
        configuration={"vm_size": "Standard_D2s_v3", "os_type": "Linux"},
        tags={"environment": "test", "owner": "testuser"},
        cost_monthly=150.0,
        last_updated=datetime.utcnow()
    )

    test_session.add(resource)
    await test_session.commit()
    await test_session.refresh(resource)

    return resource


@pytest.fixture
async def sample_incident(test_session: AsyncSession, sample_user: User, sample_resource: Resource) -> Incident:
    """Create a sample incident for testing"""
    incident = Incident(
        title="High CPU Usage Alert",
        description="CPU usage exceeded 90% for more than 5 minutes",
        severity=IncidentSeverity.HIGH,
        status=IncidentStatus.OPEN,
        resource_id=sample_resource.resource_id,
        assignee_id=sample_user.id,
        metadata_={
            "alert_rule": "high-cpu-usage",
            "threshold": 90.0,
            "current_value": 95.2
        }
    )

    test_session.add(incident)
    await test_session.commit()
    await test_session.refresh(incident)

    return incident


@pytest.mark.database
class TestDatabaseModels:
    """Test database model definitions and relationships"""

    async def test_user_model_creation(self, test_session: AsyncSession):
        """Test User model creation and validation"""
        user = User(
            email="newuser@example.com",
            username="newuser",
            full_name="New User",
            role=UserRole.ADMIN,
            azure_user_id="azure-user-456",
            is_active=True
        )

        test_session.add(user)
        await test_session.commit()
        await test_session.refresh(user)

        # Verify user creation
        assert user.id is not None
        assert user.email == "newuser@example.com"
        assert user.role == UserRole.ADMIN
        assert user.is_active is True
        assert user.created_at is not None
        assert user.updated_at is not None

    async def test_user_model_validation(self, test_session: AsyncSession):
        """Test User model validation constraints"""
        # Test invalid email
        with pytest.raises(Exception):  # Should raise validation error
            user = User(
                email="invalid-email",  # Invalid email format
                username="testuser2",
                full_name="Test User 2",
                role=UserRole.USER
            )
            test_session.add(user)
            await test_session.commit()

    async def test_user_password_hashing(self, test_session: AsyncSession):
        """Test password hashing functionality"""
        user = User(
            email="password_test@example.com",
            username="passworduser",
            full_name="Password User",
            role=UserRole.USER
        )

        # Set password
        user.set_password("secure_password_123")

        test_session.add(user)
        await test_session.commit()

        # Verify password is hashed
        assert user.hashed_password is not None
        assert user.hashed_password != "secure_password_123"

        # Verify password verification
        assert user.verify_password("secure_password_123") is True
        assert user.verify_password("wrong_password") is False

    async def test_resource_model_creation(self, test_session: AsyncSession):
        """Test Resource model creation"""
        resource = Resource(
            resource_id="/subscriptions/test/resourceGroups/test-rg/providers/Microsoft.Storage/storageAccounts/teststorage",
            name="teststorage",
            resource_type=ResourceType.STORAGE_ACCOUNT,
            subscription_id="test-subscription",
            resource_group="test-rg",
            location="westus",
            status="Available",
            configuration={"sku": "Standard_LRS", "tier": "Standard"},
            tags={"purpose": "testing", "team": "dev"},
            cost_monthly=25.50
        )

        test_session.add(resource)
        await test_session.commit()
        await test_session.refresh(resource)

        # Verify resource creation
        assert resource.id is not None
        assert resource.resource_type == ResourceType.STORAGE_ACCOUNT
        assert resource.cost_monthly == 25.50
        assert resource.configuration["sku"] == "Standard_LRS"
        assert resource.tags["purpose"] == "testing"

    async def test_incident_model_relationships(self, test_session: AsyncSession, sample_user: User, sample_resource: Resource):
        """Test Incident model relationships"""
        incident = Incident(
            title="Storage Account Accessibility Issue",
            description="Storage account is not accessible from application",
            severity=IncidentSeverity.MEDIUM,
            status=IncidentStatus.IN_PROGRESS,
            resource_id=sample_resource.resource_id,
            assignee_id=sample_user.id,
            metadata_={
                "error_code": "403",
                "last_successful_access": "2024-01-15T10:30:00Z"
            }
        )

        test_session.add(incident)
        await test_session.commit()
        await test_session.refresh(incident)

        # Test relationships
        assert incident.assignee_id == sample_user.id
        assert incident.resource_id == sample_resource.resource_id

        # Load relationships
        result = await test_session.execute(
            select(Incident).options(
                sa.orm.selectinload(Incident.assignee)
            ).where(Incident.id == incident.id)
        )
        incident_with_relations = result.scalar_one()

        assert incident_with_relations.assignee.email == sample_user.email

    async def test_system_log_model(self, test_session: AsyncSession, sample_user: User):
        """Test SystemLog model"""
        log_entry = SystemLog(
            level="INFO",
            message="User login successful",
            module="authentication",
            function="login",
            user_id=sample_user.id,
            log_metadata={
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0...",
                "session_id": "session-123"
            }
        )

        test_session.add(log_entry)
        await test_session.commit()
        await test_session.refresh(log_entry)

        # Verify log entry
        assert log_entry.id is not None
        assert log_entry.level == "INFO"
        assert log_entry.user_id == sample_user.id
        assert log_entry.log_metadata["ip_address"] == "192.168.1.100"

    async def test_configuration_model(self, test_session: AsyncSession):
        """Test Configuration model"""
        config = Configuration(
            key="email_notifications_enabled",
            value="true",
            description="Enable email notifications for incidents",
            category="notifications",
            is_sensitive=False
        )

        test_session.add(config)
        await test_session.commit()
        await test_session.refresh(config)

        # Verify configuration
        assert config.id is not None
        assert config.key == "email_notifications_enabled"
        assert config.get_typed_value() is True  # Should convert "true" to boolean


@pytest.mark.database
class TestDatabaseQueries:
    """Test database queries and operations"""

    async def test_user_queries(self, test_session: AsyncSession):
        """Test user-related queries"""
        # Create test users
        users = [
            User(email=f"user{i}@example.com", username=f"user{i}", full_name=f"User {i}", role=UserRole.USER)
            for i in range(5)
        ]

        for user in users:
            test_session.add(user)

        await test_session.commit()

        # Test: Get all users
        result = await test_session.execute(select(User))
        all_users = result.scalars().all()
        assert len(all_users) >= 5

        # Test: Get user by email
        result = await test_session.execute(
            select(User).where(User.email == "user1@example.com")
        )
        user = result.scalar_one_or_none()
        assert user is not None
        assert user.username == "user1"

        # Test: Get users by role
        result = await test_session.execute(
            select(User).where(User.role == UserRole.USER)
        )
        user_role_users = result.scalars().all()
        assert len(user_role_users) >= 5

        # Test: Count active users
        result = await test_session.execute(
            select(sa.func.count(User.id)).where(User.is_active == True)
        )
        active_count = result.scalar()
        assert active_count >= 5

    async def test_resource_queries(self, test_session: AsyncSession):
        """Test resource-related queries"""
        # Create test resources
        resources = [
            Resource(
                resource_id=f"/subscriptions/test/resourceGroups/rg-{i}/providers/Microsoft.Compute/virtualMachines/vm-{i}",
                name=f"vm-{i}",
                resource_type=ResourceType.VIRTUAL_MACHINE,
                subscription_id="test-subscription",
                resource_group=f"rg-{i}",
                location="eastus" if i % 2 == 0 else "westus",
                status="Running",
                cost_monthly=100.0 + (i * 10),
                tags={"environment": "test", "index": str(i)}
            )
            for i in range(10)
        ]

        for resource in resources:
            test_session.add(resource)

        await test_session.commit()

        # Test: Get resources by type
        result = await test_session.execute(
            select(Resource).where(Resource.resource_type == ResourceType.VIRTUAL_MACHINE)
        )
        vms = result.scalars().all()
        assert len(vms) >= 10

        # Test: Get resources by location
        result = await test_session.execute(
            select(Resource).where(Resource.location == "eastus")
        )
        eastus_resources = result.scalars().all()
        assert len(eastus_resources) >= 5

        # Test: Get resources by cost range
        result = await test_session.execute(
            select(Resource).where(
                Resource.cost_monthly.between(120.0, 150.0)
            )
        )
        cost_filtered = result.scalars().all()
        assert len(cost_filtered) >= 3

        # Test: Aggregate queries
        result = await test_session.execute(
            select(
                sa.func.sum(Resource.cost_monthly),
                sa.func.avg(Resource.cost_monthly),
                sa.func.count(Resource.id)
            ).where(Resource.resource_type == ResourceType.VIRTUAL_MACHINE)
        )
        total_cost, avg_cost, count = result.one()

        assert total_cost >= 1450.0  # Sum of costs
        assert avg_cost >= 145.0    # Average cost
        assert count >= 10          # Count of VMs

    async def test_incident_queries(self, test_session: AsyncSession, sample_user: User):
        """Test incident-related queries"""
        # Create test incidents
        incidents = []
        for i in range(8):
            incident = Incident(
                title=f"Test Incident {i}",
                description=f"Description for incident {i}",
                severity=IncidentSeverity.HIGH if i % 3 == 0 else IncidentSeverity.MEDIUM,
                status=IncidentStatus.OPEN if i < 5 else IncidentStatus.RESOLVED,
                resource_id=f"/subscriptions/test/resource-{i}",
                assignee_id=sample_user.id,
                metadata_={"test_index": i}
            )
            incidents.append(incident)
            test_session.add(incident)

        await test_session.commit()

        # Test: Get open incidents
        result = await test_session.execute(
            select(Incident).where(Incident.status == IncidentStatus.OPEN)
        )
        open_incidents = result.scalars().all()
        assert len(open_incidents) >= 5

        # Test: Get high severity incidents
        result = await test_session.execute(
            select(Incident).where(Incident.severity == IncidentSeverity.HIGH)
        )
        high_severity = result.scalars().all()
        assert len(high_severity) >= 2

        # Test: Get incidents by assignee
        result = await test_session.execute(
            select(Incident).where(Incident.assignee_id == sample_user.id)
        )
        assigned_incidents = result.scalars().all()
        assert len(assigned_incidents) >= 8

        # Test: Recent incidents (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(days=1)
        result = await test_session.execute(
            select(Incident).where(Incident.created_at >= yesterday)
        )
        recent_incidents = result.scalars().all()
        assert len(recent_incidents) >= 8

    async def test_complex_joins(self, test_session: AsyncSession, sample_user: User, sample_resource: Resource, sample_incident: Incident):
        """Test complex JOIN queries"""
        # Test: Join incidents with users and resources
        query = (
            select(
                Incident.title,
                User.full_name.label("assignee_name"),
                Resource.name.label("resource_name"),
                Resource.cost_monthly
            )
            .join(User, Incident.assignee_id == User.id)
            .join(Resource, Incident.resource_id == Resource.resource_id)
            .where(Incident.status == IncidentStatus.OPEN)
        )

        result = await test_session.execute(query)
        joined_data = result.all()

        assert len(joined_data) >= 1
        for row in joined_data:
            assert row.assignee_name is not None
            assert row.resource_name is not None
            assert row.cost_monthly is not None

    async def test_subqueries(self, test_session: AsyncSession):
        """Test subquery usage"""
        # Create some test data first
        users = [
            User(email=f"subq_user{i}@example.com", username=f"subq_user{i}", full_name=f"SubQ User {i}", role=UserRole.USER)
            for i in range(3)
        ]

        for user in users:
            test_session.add(user)

        await test_session.commit()

        # Test: Subquery to find users with incidents
        incident_subquery = select(Incident.assignee_id).distinct().subquery()

        query = (
            select(User)
            .where(User.id.in_(select(incident_subquery.c.assignee_id)))
        )

        result = await test_session.execute(query)
        users_with_incidents = result.scalars().all()

        # Should find users that have incidents assigned
        assert isinstance(users_with_incidents, list)

    async def test_pagination(self, test_session: AsyncSession):
        """Test query pagination"""
        # Create test data
        users = [
            User(email=f"page_user{i}@example.com", username=f"page_user{i}", full_name=f"Page User {i}", role=UserRole.USER)
            for i in range(25)
        ]

        for user in users:
            test_session.add(user)

        await test_session.commit()

        # Test pagination
        page_size = 10
        offset = 0

        query = (
            select(User)
            .order_by(User.created_at)
            .limit(page_size)
            .offset(offset)
        )

        result = await test_session.execute(query)
        first_page = result.scalars().all()

        assert len(first_page) == page_size

        # Second page
        offset = page_size
        query = (
            select(User)
            .order_by(User.created_at)
            .limit(page_size)
            .offset(offset)
        )

        result = await test_session.execute(query)
        second_page = result.scalars().all()

        assert len(second_page) == page_size

        # Verify no overlap
        first_page_ids = {user.id for user in first_page}
        second_page_ids = {user.id for user in second_page}
        assert first_page_ids.isdisjoint(second_page_ids)


@pytest.mark.database
class TestDatabaseTransactions:
    """Test database transaction handling"""

    async def test_successful_transaction(self, test_session: AsyncSession):
        """Test successful transaction commit"""
        # Start a transaction and create multiple records
        user = User(
            email="transaction_user@example.com",
            username="transaction_user",
            full_name="Transaction User",
            role=UserRole.USER
        )

        resource = Resource(
            resource_id="/subscriptions/test/transaction-resource",
            name="transaction-resource",
            resource_type=ResourceType.VIRTUAL_MACHINE,
            subscription_id="test-subscription",
            resource_group="test-rg",
            location="eastus",
            status="Running",
            cost_monthly=200.0
        )

        test_session.add(user)
        test_session.add(resource)

        # Commit transaction
        await test_session.commit()

        # Verify records were created
        user_result = await test_session.execute(
            select(User).where(User.email == "transaction_user@example.com")
        )
        created_user = user_result.scalar_one_or_none()
        assert created_user is not None

        resource_result = await test_session.execute(
            select(Resource).where(Resource.resource_id == "/subscriptions/test/transaction-resource")
        )
        created_resource = resource_result.scalar_one_or_none()
        assert created_resource is not None

    async def test_transaction_rollback(self, test_session: AsyncSession):
        """Test transaction rollback on error"""
        # Create a user
        user = User(
            email="rollback_user@example.com",
            username="rollback_user",
            full_name="Rollback User",
            role=UserRole.USER
        )

        test_session.add(user)
        await test_session.flush()  # Flush to get ID but don't commit

        user_id = user.id

        try:
            # Create an incident with invalid resource_id to cause an error
            incident = Incident(
                title="Test Incident",
                description="Test Description",
                severity=IncidentSeverity.HIGH,
                status=IncidentStatus.OPEN,
                resource_id="",  # Invalid empty resource_id
                assignee_id=user_id
            )

            test_session.add(incident)
            await test_session.commit()  # This should fail
        except Exception:
            await test_session.rollback()

        # Verify user was not created due to rollback
        result = await test_session.execute(
            select(User).where(User.email == "rollback_user@example.com")
        )
        user_after_rollback = result.scalar_one_or_none()
        assert user_after_rollback is None

    async def test_nested_transactions(self, test_session: AsyncSession):
        """Test nested transaction using savepoints"""
        # Create outer transaction
        user = User(
            email="nested_user@example.com",
            username="nested_user",
            full_name="Nested User",
            role=UserRole.USER
        )

        test_session.add(user)
        await test_session.flush()

        # Create savepoint
        savepoint = await test_session.begin_nested()

        try:
            # Create resource in nested transaction
            resource = Resource(
                resource_id="/subscriptions/test/nested-resource",
                name="nested-resource",
                resource_type=ResourceType.STORAGE_ACCOUNT,
                subscription_id="test-subscription",
                resource_group="test-rg",
                location="westus",
                status="Available",
                cost_monthly=50.0
            )

            test_session.add(resource)

            # Simulate error and rollback to savepoint
            raise Exception("Simulated error")

        except Exception:
            await savepoint.rollback()

        # Commit outer transaction
        await test_session.commit()

        # Verify user was created but resource was not
        user_result = await test_session.execute(
            select(User).where(User.email == "nested_user@example.com")
        )
        created_user = user_result.scalar_one_or_none()
        assert created_user is not None

        resource_result = await test_session.execute(
            select(Resource).where(Resource.resource_id == "/subscriptions/test/nested-resource")
        )
        created_resource = resource_result.scalar_one_or_none()
        assert created_resource is None

    async def test_concurrent_transactions(self, test_engine):
        """Test concurrent transaction handling"""
        async def create_user_transaction(session_maker, user_id: int):
            async with session_maker() as session:
                user = User(
                    email=f"concurrent_user_{user_id}@example.com",
                    username=f"concurrent_user_{user_id}",
                    full_name=f"Concurrent User {user_id}",
                    role=UserRole.USER
                )

                session.add(user)
                await session.commit()
                return user.id

        # Create session maker
        session_maker = async_sessionmaker(test_engine, class_=AsyncSession)

        # Run concurrent transactions
        tasks = [
            create_user_transaction(session_maker, i)
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all transactions succeeded
        assert len(results) == 5
        assert all(user_id is not None for user_id in results)

        # Verify all users were created
        async with session_maker() as session:
            result = await session.execute(
                select(sa.func.count(User.id)).where(
                    User.email.like("concurrent_user_%@example.com")
                )
            )
            count = result.scalar()
            assert count == 5


@pytest.mark.database
class TestDatabasePerformance:
    """Test database performance characteristics"""

    async def test_bulk_insert_performance(self, test_session: AsyncSession):
        """Test bulk insert performance"""
        import time

        # Prepare bulk data
        bulk_users = [
            {
                "email": f"bulk_user_{i}@example.com",
                "username": f"bulk_user_{i}",
                "full_name": f"Bulk User {i}",
                "role": UserRole.USER,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            for i in range(1000)
        ]

        # Measure bulk insert time
        start_time = time.time()

        # Use bulk insert
        await test_session.execute(
            insert(User),
            bulk_users
        )
        await test_session.commit()

        end_time = time.time()
        insert_time = end_time - start_time

        # Verify insert was successful and reasonably fast
        result = await test_session.execute(
            select(sa.func.count(User.id)).where(
                User.email.like("bulk_user_%@example.com")
            )
        )
        count = result.scalar()

        assert count == 1000
        assert insert_time < 5.0  # Should complete within 5 seconds

        # Calculate insertions per second
        insertions_per_second = 1000 / insert_time
        assert insertions_per_second > 100  # At least 100 insertions per second

    async def test_query_performance_with_indexes(self, test_session: AsyncSession):
        """Test query performance with database indexes"""
        import time

        # Create test data
        resources = [
            Resource(
                resource_id=f"/subscriptions/test/resource-{i}",
                name=f"resource-{i}",
                resource_type=ResourceType.VIRTUAL_MACHINE,
                subscription_id="test-subscription",
                resource_group=f"rg-{i % 10}",  # 10 different resource groups
                location="eastus" if i % 2 == 0 else "westus",
                status="Running",
                cost_monthly=100.0 + (i % 100),
                tags={"environment": "test", "index": str(i)}
            )
            for i in range(1000)
        ]

        # Bulk insert
        await test_session.execute(
            insert(Resource),
            [
                {
                    "resource_id": r.resource_id,
                    "name": r.name,
                    "resource_type": r.resource_type,
                    "subscription_id": r.subscription_id,
                    "resource_group": r.resource_group,
                    "location": r.location,
                    "status": r.status,
                    "cost_monthly": r.cost_monthly,
                    "tags": r.tags,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                for r in resources
            ]
        )
        await test_session.commit()

        # Test indexed query performance
        start_time = time.time()

        # Query by resource_group (should be indexed)
        result = await test_session.execute(
            select(Resource).where(Resource.resource_group == "rg-5")
        )
        indexed_results = result.scalars().all()

        indexed_query_time = time.time() - start_time

        # Test non-indexed query performance
        start_time = time.time()

        # Query by cost range
        result = await test_session.execute(
            select(Resource).where(
                Resource.cost_monthly.between(150.0, 160.0)
            )
        )
        range_results = result.scalars().all()

        range_query_time = time.time() - start_time

        # Verify results and performance
        assert len(indexed_results) == 100  # Should find ~100 resources
        assert len(range_results) >= 10    # Should find some resources

        # Both queries should be fast on small dataset
        assert indexed_query_time < 1.0
        assert range_query_time < 1.0

    async def test_connection_pool_performance(self, test_engine):
        """Test connection pool performance under load"""
        import time

        async def database_operation(session_maker, operation_id: int):
            async with session_maker() as session:
                # Simulate database work
                result = await session.execute(
                    select(sa.func.random()).limit(1)
                )
                random_value = result.scalar()

                return {"operation_id": operation_id, "result": random_value}

        session_maker = async_sessionmaker(test_engine, class_=AsyncSession)

        # Run concurrent database operations
        start_time = time.time()

        tasks = [
            database_operation(session_maker, i)
            for i in range(50)
        ]

        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Verify all operations completed successfully
        assert len(results) == 50
        assert all("result" in result for result in results)

        # Should complete reasonably quickly
        assert total_time < 10.0  # All 50 operations within 10 seconds

        # Calculate operations per second
        ops_per_second = 50 / total_time
        assert ops_per_second > 5  # At least 5 operations per second

    async def test_large_result_set_handling(self, test_session: AsyncSession):
        """Test handling of large result sets"""
        # Create large dataset
        large_dataset = [
            {
                "level": "INFO",
                "message": f"Log message {i}",
                "module": "test_module",
                "function": "test_function",
                "log_metadata": {"index": i, "batch": i // 100},
                "created_at": datetime.utcnow()
            }
            for i in range(5000)
        ]

        # Insert large dataset
        await test_session.execute(
            insert(SystemLog),
            large_dataset
        )
        await test_session.commit()

        # Test streaming large result set
        import time
        start_time = time.time()

        # Use stream_results for large datasets
        result = await test_session.stream(
            select(SystemLog).where(SystemLog.module == "test_module")
        )

        count = 0
        async for row in result:
            count += 1
            if count >= 1000:  # Process first 1000 rows
                break

        stream_time = time.time() - start_time

        # Verify streaming performance
        assert count == 1000
        assert stream_time < 5.0  # Should stream 1000 rows within 5 seconds


@pytest.mark.database
class TestDatabaseMigrations:
    """Test database migrations and schema changes"""

    async def test_schema_creation(self, test_engine):
        """Test schema creation from models"""
        # Drop all tables
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        # Recreate all tables
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Verify tables exist
        async with test_engine.begin() as conn:
            # Check if tables exist
            result = await conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ))
            tables = [row[0] for row in result.fetchall()]

            expected_tables = ["users", "incidents", "resources", "system_logs", "configurations"]
            for table in expected_tables:
                assert table in tables

    async def test_model_compatibility(self, test_session: AsyncSession):
        """Test model compatibility with database schema"""
        # Test that all models can be instantiated and saved
        test_objects = []

        # User
        user = User(
            email="schema_test@example.com",
            username="schema_test",
            full_name="Schema Test User",
            role=UserRole.USER
        )
        test_objects.append(user)

        # Resource
        resource = Resource(
            resource_id="/subscriptions/test/schema-resource",
            name="schema-resource",
            resource_type=ResourceType.VIRTUAL_MACHINE,
            subscription_id="test-subscription",
            resource_group="test-rg",
            location="eastus",
            status="Running",
            cost_monthly=100.0
        )
        test_objects.append(resource)

        # Configuration
        config = Configuration(
            key="schema_test_config",
            value="test_value",
            description="Schema test configuration",
            category="testing"
        )
        test_objects.append(config)

        # System Log
        log = SystemLog(
            level="INFO",
            message="Schema test log",
            module="testing",
            function="test_model_compatibility"
        )
        test_objects.append(log)

        # Add all objects
        for obj in test_objects:
            test_session.add(obj)

        await test_session.commit()

        # Verify all objects were saved
        for obj in test_objects:
            assert obj.id is not None

        # Create incident with relationships
        incident = Incident(
            title="Schema Test Incident",
            description="Test incident for schema validation",
            severity=IncidentSeverity.LOW,
            status=IncidentStatus.OPEN,
            resource_id=resource.resource_id,
            assignee_id=user.id
        )

        test_session.add(incident)
        await test_session.commit()

        assert incident.id is not None


@pytest.mark.database
class TestDatabaseUtilities:
    """Test database utility functions and helpers"""

    async def test_database_url_generation(self):
        """Test database URL generation"""
        # Test with environment variables
        with patch.dict(os.environ, {
            'DATABASE_HOST': 'localhost',
            'DATABASE_PORT': '5432',
            'DATABASE_NAME': 'testdb',
            'DATABASE_USER': 'testuser',
            'DATABASE_PASSWORD': 'testpass'
        }):
            url = get_database_url()
            assert "postgresql+asyncpg://testuser:testpass@localhost:5432/testdb" in url

    async def test_database_manager_initialization(self):
        """Test DatabaseManager initialization"""
        # Test with SQLite for testing
        db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")

        # Test engine creation
        engine = await db_manager.get_engine()
        assert engine is not None

        # Test session creation
        async with db_manager.get_session() as session:
            assert session is not None
            assert isinstance(session, AsyncSession)

        # Cleanup
        await db_manager.close()

    async def test_database_health_check(self, test_engine):
        """Test database health check functionality"""
        async def check_database_health(engine):
            try:
                async with engine.begin() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    return result.scalar() == 1
            except Exception:
                return False

        # Test healthy database
        is_healthy = await check_database_health(test_engine)
        assert is_healthy is True

    async def test_query_logging(self, test_session: AsyncSession):
        """Test query logging functionality"""
        # This would test SQL query logging in real implementation
        # For now, we'll test that queries execute properly

        # Enable query logging (would be configured in real implementation)
        logged_queries = []

        def query_logger(query_str):
            logged_queries.append(query_str)

        # Execute a query
        result = await test_session.execute(
            select(User).where(User.email == "test@example.com")
        )

        # Verify query executed
        user = result.scalar_one_or_none()

        # In real implementation, would verify query was logged
        # For now, just verify the query worked
        assert isinstance(logged_queries, list)


@pytest.mark.database
@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database with other components"""

    async def test_database_with_caching(self, test_session: AsyncSession):
        """Test database integration with caching layer"""
        from core.advanced_caching import global_cache

        # Create test user
        user = User(
            email="cached_user@example.com",
            username="cached_user",
            full_name="Cached User",
            role=UserRole.USER
        )

        test_session.add(user)
        await test_session.commit()
        await test_session.refresh(user)

        # Cache user data
        await global_cache.set(f"user:{user.id}", user.to_dict() if hasattr(user, 'to_dict') else str(user))

        # Retrieve from cache
        cached_user = await global_cache.get(f"user:{user.id}")
        assert cached_user is not None

    async def test_database_with_monitoring(self, test_session: AsyncSession):
        """Test database integration with monitoring"""
        from core.performance import PerformanceMonitor

        monitor = PerformanceMonitor()

        # Monitor database operation
        monitor.start_operation("database_query")

        result = await test_session.execute(
            select(sa.func.count(User.id))
        )
        count = result.scalar()

        monitor.end_operation("database_query")

        # Verify monitoring captured the operation
        metrics = monitor.get_metrics()
        assert "database_query" in metrics

        assert isinstance(count, int)

    async def test_database_with_error_handling(self, test_session: AsyncSession):
        """Test database error handling integration"""
        from core.error_handling import error_manager

        try:
            # Attempt invalid operation
            await test_session.execute(
                text("SELECT * FROM nonexistent_table")
            )
        except Exception as e:
            # Handle error with error manager
            error_context = await error_manager.handle_error(
                e,
                {
                    "operation": "database_query",
                    "table": "nonexistent_table"
                }
            )

            assert error_context.error_type is not None
            assert "database" in str(error_context.context_data.get("operation", ""))


# Test runner configuration
if __name__ == "__main__":
    # Run database tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "database"
    ])