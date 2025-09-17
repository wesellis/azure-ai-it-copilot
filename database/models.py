"""
Optimized Database Models for Azure AI IT Copilot
SQLAlchemy models with performance optimizations, proper indexing, and constraints
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, String, DateTime, JSON, Text, Boolean, Float, ForeignKey,
    Index, CheckConstraint, UniqueConstraint, func
)
from sqlalchemy.orm import declarative_base, relationship, validates
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.hybrid import hybrid_property
import uuid

Base = declarative_base()


# Optimized timestamp function for consistency
def utc_now():
    """Return current UTC timestamp"""
    return datetime.now(timezone.utc)


class TimestampMixin:
    """Mixin for models with timestamp fields"""
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class OptimizedBase(Base):
    """Base class with common optimizations"""
    __abstract__ = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def update_from_dict(self, data: Dict[str, Any]):
        """Update model from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class User(OptimizedBase, TimestampMixin):
    """Optimized user model for authentication and authorization"""
    __tablename__ = "users"
    __table_args__ = (
        Index('idx_users_email', 'email'),
        Index('idx_users_azure_id', 'azure_id'),
        Index('idx_users_active', 'is_active'),
        Index('idx_users_role', 'role'),
        CheckConstraint(
            "role IN ('admin', 'owner', 'contributor', 'reader')",
            name='check_user_role'
        ),
        CheckConstraint(
            "email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'",
            name='check_email_format'
        )
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    azure_id = Column(String(255), unique=True, nullable=True, index=True)
    email = Column(String(320), unique=True, nullable=False, index=True)  # RFC 5321 max length
    name = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    role = Column(String(20), nullable=False, default="reader", index=True)
    department = Column(String(100), nullable=True)
    job_title = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_login = Column(DateTime(timezone=True), nullable=True)
    login_count = Column(Integer, default=0, nullable=False)

    # Optimized relationships with lazy loading
    command_history = relationship(
        "CommandHistory",
        back_populates="user",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )
    incidents_created = relationship(
        "Incident",
        back_populates="created_by",
        foreign_keys="Incident.created_by_id",
        lazy="dynamic"
    )
    incidents_assigned = relationship(
        "Incident",
        back_populates="assigned_to",
        foreign_keys="Incident.assigned_to_id",
        lazy="dynamic"
    )
    cost_alerts = relationship(
        "CostAlert",
        back_populates="user",
        foreign_keys="CostAlert.user_id",
        lazy="dynamic"
    )

    @hybrid_property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.name

    @validates('email')
    def validate_email(self, key, email):
        """Validate email format"""
        import re
        if not re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$', email):
            raise ValueError("Invalid email format")
        return email.lower()

    @validates('role')
    def validate_role(self, key, role):
        """Validate user role"""
        valid_roles = ['admin', 'owner', 'contributor', 'reader']
        if role not in valid_roles:
            raise ValueError(f"Role must be one of: {valid_roles}")
        return role

    def update_last_login(self):
        """Update last login timestamp and increment counter"""
        self.last_login = utc_now()
        self.login_count += 1


class CommandHistory(OptimizedBase, TimestampMixin):
    """Optimized command execution history with indexing"""
    __tablename__ = "command_history"
    __table_args__ = (
        Index('idx_command_history_user_id', 'user_id'),
        Index('idx_command_history_status', 'status'),
        Index('idx_command_history_intent', 'intent'),
        Index('idx_command_history_created_at', 'created_at'),
        Index('idx_command_history_user_status', 'user_id', 'status'),
        CheckConstraint(
            "status IN ('pending', 'processing', 'success', 'error', 'cancelled')",
            name='check_command_status'
        ),
        CheckConstraint(
            "execution_time >= 0",
            name='check_execution_time_positive'
        )
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    command = Column(Text, nullable=False)
    intent = Column(String(100), nullable=False, index=True)
    context = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False, default="pending", index=True)
    execution_time = Column(Float, nullable=True)  # seconds
    error_message = Column(Text, nullable=True)
    request_id = Column(String(36), nullable=True, index=True)  # For request tracing
    retry_count = Column(Integer, default=0, nullable=False)

    # Optimized relationships
    user = relationship("User", back_populates="command_history")

    @validates('status')
    def validate_status(self, key, status):
        """Validate command status"""
        valid_statuses = ['pending', 'processing', 'success', 'error', 'cancelled']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return status

    @hybrid_property
    def is_completed(self) -> bool:
        """Check if command is completed"""
        return self.status in ['success', 'error', 'cancelled']

    @hybrid_property
    def duration_ms(self) -> Optional[int]:
        """Get execution time in milliseconds"""
        return int(self.execution_time * 1000) if self.execution_time else None


class Incident(OptimizedBase, TimestampMixin):
    """Optimized incident tracking and management with comprehensive indexing"""
    __tablename__ = "incidents"
    __table_args__ = (
        Index('idx_incidents_incident_id', 'incident_id'),
        Index('idx_incidents_severity', 'severity'),
        Index('idx_incidents_status', 'status'),
        Index('idx_incidents_category', 'category'),
        Index('idx_incidents_created_by', 'created_by_id'),
        Index('idx_incidents_assigned_to', 'assigned_to_id'),
        Index('idx_incidents_detected_at', 'detected_at'),
        Index('idx_incidents_priority_score', 'priority_score'),
        Index('idx_incidents_status_severity', 'status', 'severity'),
        Index('idx_incidents_active', 'status', 'detected_at'),
        CheckConstraint(
            "severity IN ('critical', 'high', 'medium', 'low')",
            name='check_incident_severity'
        ),
        CheckConstraint(
            "status IN ('open', 'investigating', 'resolving', 'resolved', 'closed')",
            name='check_incident_status'
        ),
        CheckConstraint(
            "priority_score >= 1 AND priority_score <= 10",
            name='check_incident_priority_range'
        ),
        CheckConstraint(
            "root_cause_confidence >= 0.0 AND root_cause_confidence <= 1.0",
            name='check_root_cause_confidence_range'
        )
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)  # Increased length for better descriptions
    description = Column(Text, nullable=False)
    severity = Column(String(10), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="open", index=True)
    category = Column(String(50), nullable=True, index=True)

    # Affected resources with better structure
    affected_resources = Column(JSON, nullable=True)
    symptoms = Column(JSON, nullable=True)

    # Enhanced root cause analysis
    root_cause = Column(Text, nullable=True)
    root_cause_confidence = Column(Float, nullable=True)  # 0.0 to 1.0
    ai_analysis_summary = Column(Text, nullable=True)

    # Resolution tracking
    resolution_summary = Column(Text, nullable=True)
    resolution_actions = Column(JSON, nullable=True)
    resolution_time_minutes = Column(Integer, nullable=True)

    # Timing with better precision
    detected_at = Column(DateTime(timezone=True), nullable=False, index=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Assignment with better tracking
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    assigned_to_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    # Enhanced metadata
    tags = Column(JSON, nullable=True)
    priority_score = Column(Integer, nullable=True, index=True)  # 1-10 scale
    business_impact = Column(Text, nullable=True)
    customer_impact_level = Column(String(20), nullable=True)  # none, low, medium, high, critical
    estimated_cost_impact = Column(Float, nullable=True)

    # Optimized relationships with proper foreign keys
    created_by = relationship(
        "User",
        back_populates="incidents_created",
        foreign_keys=[created_by_id]
    )
    assigned_to = relationship(
        "User",
        back_populates="incidents_assigned",
        foreign_keys=[assigned_to_id]
    )
    diagnostics = relationship(
        "IncidentDiagnostic",
        back_populates="incident",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )
    timeline = relationship(
        "IncidentTimeline",
        back_populates="incident",
        lazy="dynamic",
        cascade="all, delete-orphan",
        order_by="IncidentTimeline.created_at"
    )

    @validates('severity')
    def validate_severity(self, key, severity):
        """Validate incident severity"""
        valid_severities = ['critical', 'high', 'medium', 'low']
        if severity not in valid_severities:
            raise ValueError(f"Severity must be one of: {valid_severities}")
        return severity

    @validates('status')
    def validate_status(self, key, status):
        """Validate incident status"""
        valid_statuses = ['open', 'investigating', 'resolving', 'resolved', 'closed']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return status

    @hybrid_property
    def is_resolved(self) -> bool:
        """Check if incident is resolved"""
        return self.status in ['resolved', 'closed']

    @hybrid_property
    def is_active(self) -> bool:
        """Check if incident is active"""
        return self.status in ['open', 'investigating', 'resolving']

    @hybrid_property
    def duration_minutes(self) -> Optional[int]:
        """Get incident duration in minutes"""
        if self.resolved_at and self.detected_at:
            delta = self.resolved_at - self.detected_at
            return int(delta.total_seconds() / 60)
        return None

    def assign_to_user(self, user_id: str):
        """Assign incident to a user"""
        self.assigned_to_id = user_id
        self.acknowledged_at = utc_now()

    def resolve_incident(self, resolution_summary: str, resolution_actions: Optional[Dict] = None):
        """Mark incident as resolved"""
        self.status = 'resolved'
        self.resolved_at = utc_now()
        self.resolution_summary = resolution_summary
        if resolution_actions:
            self.resolution_actions = resolution_actions

        # Calculate resolution time
        if self.detected_at:
            delta = self.resolved_at - self.detected_at
            self.resolution_time_minutes = int(delta.total_seconds() / 60)


class IncidentDiagnostic(OptimizedBase, TimestampMixin):
    """Optimized diagnostic data for incidents with indexing"""
    __tablename__ = "incident_diagnostics"
    __table_args__ = (
        Index('idx_incident_diagnostics_incident_id', 'incident_id'),
        Index('idx_incident_diagnostics_type', 'diagnostic_type'),
        Index('idx_incident_diagnostics_confidence', 'confidence_score'),
        Index('idx_incident_diagnostics_created_at', 'created_at'),
        CheckConstraint(
            "diagnostic_type IN ('metrics', 'logs', 'correlation', 'anomaly_detection', 'ai_analysis')",
            name='check_diagnostic_type'
        ),
        CheckConstraint(
            "confidence_score >= 0.0 AND confidence_score <= 1.0",
            name='check_confidence_score_range'
        )
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False)
    diagnostic_type = Column(String(50), nullable=False)  # metrics, logs, correlation
    query = Column(Text, nullable=False)
    result = Column(JSON, nullable=False)
    anomalies_detected = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    incident = relationship("Incident", back_populates="diagnostics")


class IncidentTimeline(OptimizedBase, TimestampMixin):
    """Optimized timeline of incident events with indexing"""
    __tablename__ = "incident_timeline"
    __table_args__ = (
        Index('idx_incident_timeline_incident_id', 'incident_id'),
        Index('idx_incident_timeline_event_type', 'event_type'),
        Index('idx_incident_timeline_user_id', 'user_id'),
        Index('idx_incident_timeline_created_at', 'created_at'),
        Index('idx_incident_timeline_incident_event', 'incident_id', 'event_type'),
        CheckConstraint(
            "event_type IN ('detected', 'acknowledged', 'investigating', 'action_taken', 'escalated', 'resolved', 'closed', 'reopened')",
            name='check_timeline_event_type'
        )
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False)
    event_type = Column(String(50), nullable=False)  # detected, investigating, action_taken, resolved
    description = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    incident = relationship("Incident", back_populates="timeline")
    user = relationship("User")


class AzureResource(OptimizedBase, TimestampMixin):
    """Optimized Azure resource inventory with comprehensive indexing for performance"""
    __tablename__ = "azure_resources"
    __table_args__ = (
        Index('idx_azure_resources_azure_id', 'azure_resource_id'),
        Index('idx_azure_resources_type', 'resource_type'),
        Index('idx_azure_resources_resource_group', 'resource_group'),
        Index('idx_azure_resources_subscription', 'subscription_id'),
        Index('idx_azure_resources_location', 'location'),
        Index('idx_azure_resources_status', 'status'),
        Index('idx_azure_resources_health', 'health_status'),
        Index('idx_azure_resources_last_monitored', 'last_monitored'),
        Index('idx_azure_resources_cost', 'estimated_monthly_cost'),
        Index('idx_azure_resources_composite', 'subscription_id', 'resource_group', 'resource_type'),
        CheckConstraint(
            "health_status IN ('healthy', 'warning', 'critical', 'unknown')",
            name='check_resource_health_status'
        ),
        CheckConstraint(
            "estimated_monthly_cost >= 0",
            name='check_positive_cost'
        )
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    azure_resource_id = Column(String(500), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    resource_type = Column(String(100), nullable=False, index=True)
    resource_group = Column(String(90), nullable=False, index=True)  # Azure limit
    subscription_id = Column(String(36), nullable=False, index=True)  # UUID format
    location = Column(String(50), nullable=False, index=True)

    # Enhanced configuration
    configuration = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)
    azure_api_version = Column(String(20), nullable=True)

    # Enhanced status tracking
    status = Column(String(30), nullable=True, index=True)
    provisioning_state = Column(String(30), nullable=True)
    availability_state = Column(String(20), nullable=True)

    # Enhanced cost tracking
    estimated_monthly_cost = Column(Float, nullable=True, index=True)
    currency = Column(String(3), nullable=True, default='USD')
    cost_last_updated = Column(DateTime(timezone=True), nullable=True)

    # Enhanced monitoring
    last_monitored = Column(DateTime(timezone=True), nullable=True, index=True)
    health_status = Column(String(10), nullable=True, default='unknown', index=True)
    health_check_count = Column(Integer, default=0, nullable=False)
    performance_score = Column(Float, nullable=True)  # 0-100 scale

    # Compliance and security
    compliance_status = Column(String(20), nullable=True)
    security_score = Column(Float, nullable=True)  # 0-100 scale
    last_security_scan = Column(DateTime(timezone=True), nullable=True)

    # Optimized relationships with lazy loading
    metrics = relationship(
        "ResourceMetric",
        back_populates="resource",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )
    cost_data = relationship(
        "CostData",
        back_populates="resource",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )

    @validates('health_status')
    def validate_health_status(self, key, health_status):
        """Validate resource health status"""
        if health_status is None:
            return 'unknown'
        valid_statuses = ['healthy', 'warning', 'critical', 'unknown']
        if health_status not in valid_statuses:
            raise ValueError(f"Health status must be one of: {valid_statuses}")
        return health_status

    @hybrid_property
    def is_healthy(self) -> bool:
        """Check if resource is healthy"""
        return self.health_status == 'healthy'

    @hybrid_property
    def needs_attention(self) -> bool:
        """Check if resource needs attention"""
        return self.health_status in ['warning', 'critical']

    @hybrid_property
    def full_resource_name(self) -> str:
        """Get full Azure resource name"""
        return f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/{self.resource_type}/{self.name}"

    def update_health_status(self, status: str, performance_score: Optional[float] = None):
        """Update resource health status"""
        self.health_status = status
        self.last_monitored = utc_now()
        self.health_check_count += 1
        if performance_score is not None:
            self.performance_score = performance_score

    def update_cost_estimate(self, cost: float, currency: str = 'USD'):
        """Update cost estimate"""
        self.estimated_monthly_cost = cost
        self.currency = currency
        self.cost_last_updated = utc_now()


class ResourceMetric(Base):
    """Time-series metrics for resources"""
    __tablename__ = "resource_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_id = Column(UUID(as_uuid=True), ForeignKey("azure_resources.id"), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)
    timestamp = Column(DateTime, nullable=False)
    aggregation = Column(String(20), nullable=True)  # average, maximum, minimum, total

    # Anomaly detection
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    resource = relationship("AzureResource", back_populates="metrics")


class CostData(Base):
    """Cost tracking and optimization data"""
    __tablename__ = "cost_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_id = Column(UUID(as_uuid=True), ForeignKey("azure_resources.id"), nullable=True)
    subscription_id = Column(String(255), nullable=False)
    resource_group = Column(String(255), nullable=True)
    service_name = Column(String(255), nullable=False)

    # Cost details
    cost_amount = Column(Float, nullable=False)
    currency = Column(String(10), nullable=False, default="USD")
    billing_period = Column(String(20), nullable=False)  # daily, monthly
    usage_date = Column(DateTime, nullable=False)

    # Usage details
    quantity = Column(Float, nullable=True)
    unit_of_measure = Column(String(50), nullable=True)

    # Optimization
    optimization_potential = Column(Float, nullable=True)
    recommendations = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    resource = relationship("AzureResource", back_populates="cost_data")


class CostAlert(Base):
    """Cost monitoring and alerting"""
    __tablename__ = "cost_alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_name = Column(String(255), nullable=False)
    alert_type = Column(String(50), nullable=False)  # budget_exceeded, spike_detected, forecast_high
    threshold_amount = Column(Float, nullable=False)
    current_amount = Column(Float, nullable=False)
    scope = Column(String(100), nullable=False)  # subscription, resource_group, resource
    scope_id = Column(String(500), nullable=False)

    # Alert details
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    recommendations = Column(JSON, nullable=True)

    # Status
    status = Column(String(20), default="active")  # active, resolved, dismissed
    acknowledged = Column(Boolean, default=False)
    acknowledged_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)

    # Assignment
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="cost_alerts", foreign_keys=[user_id])
    acknowledged_by = relationship("User", foreign_keys=[acknowledged_by_id])


class ComplianceCheck(Base):
    """Compliance assessment results"""
    __tablename__ = "compliance_checks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    check_id = Column(String(100), nullable=False)  # e.g., CIS-1.1, SOC2-CC6.1
    framework = Column(String(50), nullable=False)  # CIS, SOC2, ISO27001, etc.
    category = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)

    # Assessment
    status = Column(String(20), nullable=False)  # pass, fail, warning, not_applicable
    score = Column(Float, nullable=True)  # 0-100
    evidence = Column(JSON, nullable=True)
    findings = Column(JSON, nullable=True)

    # Remediation
    remediation_guidance = Column(Text, nullable=True)
    remediation_effort = Column(String(20), nullable=True)  # low, medium, high

    # Scope
    resource_id = Column(UUID(as_uuid=True), ForeignKey("azure_resources.id"), nullable=True)
    subscription_id = Column(String(255), nullable=False)

    # Assessment metadata
    assessed_at = Column(DateTime, nullable=False)
    assessor = Column(String(100), nullable=True)  # system, user, third_party

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    resource = relationship("AzureResource")


class Prediction(Base):
    """ML model predictions and forecasts"""
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_type = Column(String(50), nullable=False)  # cost_forecast, capacity_planning, anomaly_detection
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)

    # Scope
    scope_type = Column(String(50), nullable=False)  # resource, resource_group, subscription
    scope_id = Column(String(500), nullable=False)

    # Prediction data
    prediction_data = Column(JSON, nullable=False)
    confidence_score = Column(Float, nullable=True)
    accuracy_metrics = Column(JSON, nullable=True)

    # Time range
    prediction_date = Column(DateTime, nullable=False)
    forecast_horizon = Column(String(20), nullable=True)  # 7d, 30d, 90d

    # Model metadata
    training_data_size = Column(Integer, nullable=True)
    feature_importance = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class SystemLog(Base):
    """System audit and activity logs"""
    __tablename__ = "system_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    log_level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    category = Column(String(50), nullable=False)  # auth, api, agent, system, security
    action = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)

    # Context
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    session_id = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)

    # Additional data
    log_metadata = Column(JSON, nullable=True)
    request_id = Column(String(255), nullable=True)
    correlation_id = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")


class Configuration(Base):
    """System configuration and settings"""
    __tablename__ = "configurations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(255), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=False)  # system, monitoring, authentication, etc.
    is_sensitive = Column(Boolean, default=False)

    # Change tracking
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    updated_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    created_by = relationship("User", foreign_keys=[created_by_id])
    updated_by = relationship("User", foreign_keys=[updated_by_id])


# Optimized Database utility functions
def create_tables(engine, checkfirst: bool = True):
    """Create all database tables with optimization"""
    try:
        Base.metadata.create_all(bind=engine, checkfirst=checkfirst)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create tables: {e}")
        return False


def drop_tables(engine, checkfirst: bool = True):
    """Drop all database tables safely"""
    try:
        Base.metadata.drop_all(bind=engine, checkfirst=checkfirst)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to drop tables: {e}")
        return False


def get_table_info(engine) -> Dict[str, Dict[str, Any]]:
    """Get information about all tables"""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables_info = {}

    for table_name in inspector.get_table_names():
        tables_info[table_name] = {
            'columns': inspector.get_columns(table_name),
            'indexes': inspector.get_indexes(table_name),
            'foreign_keys': inspector.get_foreign_keys(table_name),
            'primary_key': inspector.get_pk_constraint(table_name)
        }

    return tables_info


def optimize_database(engine):
    """Run database optimization commands"""
    try:
        # PostgreSQL specific optimizations
        if 'postgresql' in str(engine.url):
            with engine.connect() as conn:
                conn.execute('ANALYZE;')  # Update table statistics
                conn.execute('VACUUM ANALYZE;')  # Vacuum and analyze

        # SQLite specific optimizations
        elif 'sqlite' in str(engine.url):
            with engine.connect() as conn:
                conn.execute('ANALYZE;')
                conn.execute('PRAGMA optimize;')

        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Database optimization failed: {e}")
        return False


def validate_database_schema(engine) -> bool:
    """Validate that all expected tables and indexes exist"""
    try:
        from sqlalchemy import inspect
        inspector = inspect(engine)

        # Check that all model tables exist
        expected_tables = {
            'users', 'command_history', 'incidents', 'incident_diagnostics',
            'incident_timeline', 'azure_resources', 'resource_metrics',
            'cost_data', 'cost_alerts', 'compliance_checks', 'predictions',
            'system_logs', 'configurations'
        }

        existing_tables = set(inspector.get_table_names())
        missing_tables = expected_tables - existing_tables

        if missing_tables:
            logging.getLogger(__name__).warning(f"Missing tables: {missing_tables}")
            return False

        return True

    except Exception as e:
        logging.getLogger(__name__).error(f"Schema validation failed: {e}")
        return False