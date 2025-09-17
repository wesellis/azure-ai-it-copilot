"""
Database Models for Azure AI IT Copilot
SQLAlchemy models for data persistence
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Boolean, Float, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    azure_id = Column(String(255), unique=True, nullable=True)  # Azure AD object ID
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    role = Column(String(50), nullable=False, default="reader")
    department = Column(String(255), nullable=True)
    job_title = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    command_history = relationship("CommandHistory", back_populates="user")
    incidents = relationship("Incident", back_populates="created_by")
    cost_alerts = relationship("CostAlert", back_populates="user")


class CommandHistory(Base):
    """Command execution history"""
    __tablename__ = "command_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    command = Column(Text, nullable=False)
    intent = Column(String(100), nullable=False)
    context = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    status = Column(String(50), nullable=False)  # success, error, cancelled
    execution_time = Column(Float, nullable=True)  # seconds
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="command_history")


class Incident(Base):
    """Incident tracking and management"""
    __tablename__ = "incidents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(String(100), unique=True, nullable=False)  # Human-readable ID
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False)  # critical, high, medium, low
    status = Column(String(50), nullable=False, default="open")  # open, investigating, resolving, resolved, closed
    category = Column(String(100), nullable=True)  # performance, availability, security, etc.

    # Affected resources
    affected_resources = Column(JSON, nullable=True)
    symptoms = Column(JSON, nullable=True)

    # Root cause analysis
    root_cause = Column(Text, nullable=True)
    root_cause_confidence = Column(Float, nullable=True)

    # Resolution
    resolution_summary = Column(Text, nullable=True)
    resolution_actions = Column(JSON, nullable=True)

    # Timing
    detected_at = Column(DateTime, nullable=False)
    resolved_at = Column(DateTime, nullable=True)

    # Assignment
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    assigned_to_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # Metadata
    tags = Column(JSON, nullable=True)
    priority_score = Column(Integer, nullable=True)
    business_impact = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    created_by = relationship("User", back_populates="incidents", foreign_keys=[created_by_id])
    assigned_to = relationship("User", foreign_keys=[assigned_to_id])
    diagnostics = relationship("IncidentDiagnostic", back_populates="incident")
    timeline = relationship("IncidentTimeline", back_populates="incident")


class IncidentDiagnostic(Base):
    """Diagnostic data for incidents"""
    __tablename__ = "incident_diagnostics"

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


class IncidentTimeline(Base):
    """Timeline of incident events"""
    __tablename__ = "incident_timeline"

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


class AzureResource(Base):
    """Azure resource inventory and metadata"""
    __tablename__ = "azure_resources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    azure_resource_id = Column(String(500), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_group = Column(String(255), nullable=False)
    subscription_id = Column(String(255), nullable=False)
    location = Column(String(100), nullable=False)

    # Configuration
    configuration = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)

    # Status
    status = Column(String(50), nullable=True)
    provisioning_state = Column(String(50), nullable=True)

    # Cost tracking
    estimated_monthly_cost = Column(Float, nullable=True)

    # Monitoring
    last_monitored = Column(DateTime, nullable=True)
    health_status = Column(String(20), nullable=True)  # healthy, warning, critical, unknown

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    metrics = relationship("ResourceMetric", back_populates="resource")
    cost_data = relationship("CostData", back_populates="resource")


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


# Database utility functions
def create_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def drop_tables(engine):
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)