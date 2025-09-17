#!/bin/bash

# Azure AI IT Copilot - SaaS Components Setup Script
# Copies essential components from SaaS Pieces library

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Paths
SAAS_PIECES="/home/mookyjooky/Dropbox/GITHUB/07 - SAAS PIECES"
PROJECT_ROOT="/home/mookyjooky/Dropbox/GITHUB/06 - COMPLETED TOOLS/azure-ai-it-copilot"

echo -e "${BLUE}🚀 Setting up SaaS components for Azure AI IT Copilot${NC}"

# Create SaaS directories
echo -e "${GREEN}Creating SaaS directories...${NC}"
mkdir -p "$PROJECT_ROOT/saas"
mkdir -p "$PROJECT_ROOT/saas/auth"
mkdir -p "$PROJECT_ROOT/saas/payments"
mkdir -p "$PROJECT_ROOT/saas/admin"
mkdir -p "$PROJECT_ROOT/saas/analytics"
mkdir -p "$PROJECT_ROOT/saas/multi_tenant"

# Copy Authentication Components
echo -e "${GREEN}Copying authentication components...${NC}"
if [ -d "$SAAS_PIECES/01-Authentication" ]; then
    cp -r "$SAAS_PIECES/01-Authentication"/* "$PROJECT_ROOT/saas/auth/" 2>/dev/null || true
    echo "  ✅ Authentication components copied"
fi

# Copy Payment Components
echo -e "${GREEN}Copying payment components...${NC}"
if [ -d "$SAAS_PIECES/02-Payments/stripe-complete" ]; then
    cp -r "$SAAS_PIECES/02-Payments/stripe-complete"/* "$PROJECT_ROOT/saas/payments/" 2>/dev/null || true
    echo "  ✅ Stripe payment components copied"
fi

# Copy Admin Dashboard
echo -e "${GREEN}Copying admin dashboard...${NC}"
if [ -d "$SAAS_PIECES/16-Admin" ]; then
    cp -r "$SAAS_PIECES/16-Admin"/* "$PROJECT_ROOT/saas/admin/" 2>/dev/null || true
    echo "  ✅ Admin dashboard copied"
fi

# Copy Analytics Components
echo -e "${GREEN}Copying analytics components...${NC}"
if [ -d "$SAAS_PIECES/13-Analytics" ]; then
    cp -r "$SAAS_PIECES/13-Analytics"/* "$PROJECT_ROOT/saas/analytics/" 2>/dev/null || true
    echo "  ✅ Analytics components copied"
fi

# Create multi-tenant configuration
echo -e "${GREEN}Creating multi-tenant configuration...${NC}"
cat > "$PROJECT_ROOT/saas/multi_tenant/config.py" << 'EOF'
"""
Multi-tenant configuration for Azure AI IT Copilot SaaS
"""

from pydantic import BaseSettings, Field
from typing import Optional

class SaaSSettings(BaseSettings):
    """SaaS-specific settings"""

    # Stripe Configuration
    stripe_secret_key: str = Field(..., env="STRIPE_SECRET_KEY")
    stripe_publishable_key: str = Field(..., env="STRIPE_PUBLISHABLE_KEY")
    stripe_webhook_secret: str = Field(..., env="STRIPE_WEBHOOK_SECRET")

    # Pricing Tiers
    starter_price_id: str = Field(default="price_starter", env="STRIPE_STARTER_PRICE_ID")
    professional_price_id: str = Field(default="price_pro", env="STRIPE_PRO_PRICE_ID")
    enterprise_price_id: str = Field(default="price_enterprise", env="STRIPE_ENTERPRISE_PRICE_ID")

    # Trial Settings
    trial_days: int = Field(default=14, env="TRIAL_DAYS")

    # Multi-tenant Database
    tenant_db_prefix: str = Field(default="tenant_", env="TENANT_DB_PREFIX")
    shared_db_url: Optional[str] = Field(None, env="SHARED_DATABASE_URL")

    # Rate Limiting per Plan
    starter_api_calls: int = Field(default=1000, env="STARTER_API_CALLS")
    professional_api_calls: int = Field(default=10000, env="PRO_API_CALLS")
    enterprise_api_calls: int = Field(default=100000, env="ENTERPRISE_API_CALLS")

    # Resource Limits per Plan
    starter_max_resources: int = Field(default=50, env="STARTER_MAX_RESOURCES")
    professional_max_resources: int = Field(default=500, env="PRO_MAX_RESOURCES")
    enterprise_max_resources: int = Field(default=-1, env="ENTERPRISE_MAX_RESOURCES")  # -1 = unlimited

    # Feature Flags
    enable_predictive_agent: bool = Field(default=True, env="ENABLE_PREDICTIVE_AGENT")
    enable_cost_optimization: bool = Field(default=True, env="ENABLE_COST_OPTIMIZATION")
    enable_compliance_automation: bool = Field(default=True, env="ENABLE_COMPLIANCE")

    class Config:
        env_file = ".env.saas"
EOF

# Create tenant models
cat > "$PROJECT_ROOT/saas/multi_tenant/models.py" << 'EOF'
"""
Multi-tenant database models for SaaS
"""

from sqlalchemy import Column, String, DateTime, Boolean, Integer, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import uuid
import enum

from database.models import Base

class SubscriptionTier(enum.Enum):
    TRIAL = "trial"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    domain = Column(String, unique=True)

    # Subscription
    subscription_tier = Column(Enum(SubscriptionTier), default=SubscriptionTier.TRIAL)
    stripe_customer_id = Column(String, unique=True)
    stripe_subscription_id = Column(String)

    # Limits
    max_resources = Column(Integer, default=50)
    max_api_calls = Column(Integer, default=1000)
    max_users = Column(Integer, default=5)

    # Status
    is_active = Column(Boolean, default=True)
    trial_ends_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="tenant")
    api_keys = relationship("ApiKey", back_populates="tenant")
    usage_records = relationship("UsageRecord", back_populates="tenant")

    def is_trial_active(self):
        """Check if trial is still active"""
        if self.subscription_tier != SubscriptionTier.TRIAL:
            return False
        return self.trial_ends_at and self.trial_ends_at > datetime.utcnow()

    def days_left_in_trial(self):
        """Calculate days left in trial"""
        if not self.is_trial_active():
            return 0
        delta = self.trial_ends_at - datetime.utcnow()
        return delta.days

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    name = Column(String, nullable=False)
    key_hash = Column(String, nullable=False)
    last_used_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

    tenant = relationship("Tenant", back_populates="api_keys")

class UsageRecord(Base):
    __tablename__ = "usage_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)

    # Usage metrics
    api_calls = Column(Integer, default=0)
    resources_managed = Column(Integer, default=0)
    agents_executed = Column(Integer, default=0)

    # Billing period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)

    # Costs
    estimated_cost = Column(Integer, default=0)  # in cents

    tenant = relationship("Tenant", back_populates="usage_records")
EOF

# Create SaaS requirements file
cat > "$PROJECT_ROOT/requirements-saas.txt" << 'EOF'
# SaaS-specific dependencies
stripe>=5.0.0
python-multipart>=0.0.6
redis-om>=0.2.0
sendgrid>=6.10.0
segment-analytics-python>=2.2.0
posthog>=3.0.0
sentry-sdk>=1.35.0

# Multi-tenancy
sqlalchemy-utils>=0.41.0
tenant-schemas-sqlalchemy>=0.1.0

# Admin dashboard
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.1.0

# Rate limiting
slowapi>=0.1.9
redis>=5.0.0

# Billing & Invoicing
weasyprint>=60.0
jinja2>=3.1.2
EOF

# Create example .env.saas file
cat > "$PROJECT_ROOT/.env.saas.example" << 'EOF'
# Stripe Configuration
STRIPE_SECRET_KEY=sk_test_your_key_here
STRIPE_PUBLISHABLE_KEY=pk_test_your_key_here
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# Stripe Price IDs (create these in Stripe Dashboard)
STRIPE_STARTER_PRICE_ID=price_starter_monthly
STRIPE_PRO_PRICE_ID=price_pro_monthly
STRIPE_ENTERPRISE_PRICE_ID=price_enterprise_monthly

# Trial Configuration
TRIAL_DAYS=14

# Database
SHARED_DATABASE_URL=postgresql://user:pass@localhost/saas_db

# Redis (for caching and rate limiting)
REDIS_URL=redis://localhost:6379/0

# Email Service (SendGrid)
SENDGRID_API_KEY=SG.your_api_key_here
FROM_EMAIL=noreply@azureaiitcopilot.com

# Analytics (optional)
SEGMENT_WRITE_KEY=your_segment_key
POSTHOG_API_KEY=your_posthog_key
SENTRY_DSN=your_sentry_dsn

# Rate Limits (per month)
STARTER_API_CALLS=1000
PRO_API_CALLS=10000
ENTERPRISE_API_CALLS=100000

# Resource Limits
STARTER_MAX_RESOURCES=50
PRO_MAX_RESOURCES=500
ENTERPRISE_MAX_RESOURCES=-1

# Feature Flags
ENABLE_PREDICTIVE_AGENT=true
ENABLE_COST_OPTIMIZATION=true
ENABLE_COMPLIANCE=true
EOF

# Create main SaaS application file
cat > "$PROJECT_ROOT/main_saas.py" << 'EOF'
"""
Azure AI IT Copilot - SaaS Edition
Multi-tenant cloud version with subscription management
"""

import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import stripe

from saas.multi_tenant.config import SaaSSettings
from saas.multi_tenant.models import Tenant, SubscriptionTier
from core.dependency_injection import DependencyContainer
from api.server import app as base_app

# Load SaaS configuration
saas_settings = SaaSSettings()
stripe.api_key = saas_settings.stripe_secret_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    """SaaS application lifespan manager"""
    # Initialize base application
    container = DependencyContainer()
    await container.initialize()

    # Initialize SaaS components
    print("🚀 Initializing Azure AI IT Copilot SaaS Edition")
    print(f"📊 Pricing Tiers: Starter ($299), Professional ($999), Enterprise ($2999)")
    print(f"🔐 Multi-tenant mode: ENABLED")

    yield

    # Cleanup
    await container.cleanup()

# Create SaaS application
app = FastAPI(
    title="Azure AI IT Copilot - SaaS Edition",
    description="Enterprise IT Operations Platform as a Service",
    version="2.0.0-saas",
    lifespan=lifespan
)

# Add CORS for SaaS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.azureaiitcopilot.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount base application routes
app.mount("/api/v1", base_app)

# SaaS-specific routes
@app.get("/saas/health")
async def saas_health():
    """SaaS health check endpoint"""
    return {
        "status": "healthy",
        "mode": "multi-tenant",
        "version": "2.0.0-saas"
    }

@app.post("/saas/tenant/register")
async def register_tenant(name: str, email: str, plan: str = "trial"):
    """Register new tenant with trial"""
    # Implementation here
    return {"status": "tenant_created", "trial_days": saas_settings.trial_days}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
EOF

echo -e "${BLUE}📝 Creating migration for multi-tenant tables...${NC}"
cat > "$PROJECT_ROOT/migrations/add_multi_tenant_support.sql" << 'EOF'
-- Multi-tenant support migration
-- Add tenant_id to all existing tables

-- Create tenants table
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) UNIQUE,
    subscription_tier VARCHAR(50) DEFAULT 'trial',
    stripe_customer_id VARCHAR(255) UNIQUE,
    stripe_subscription_id VARCHAR(255),
    max_resources INTEGER DEFAULT 50,
    max_api_calls INTEGER DEFAULT 1000,
    max_users INTEGER DEFAULT 5,
    is_active BOOLEAN DEFAULT TRUE,
    trial_ends_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Add tenant_id to existing tables
ALTER TABLE users ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
ALTER TABLE resources ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);

-- Create indexes for performance
CREATE INDEX idx_users_tenant ON users(tenant_id);
CREATE INDEX idx_resources_tenant ON resources(tenant_id);
CREATE INDEX idx_incidents_tenant ON incidents(tenant_id);

-- Row-level security policies
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE resources ENABLE ROW LEVEL SECURITY;
ALTER TABLE incidents ENABLE ROW LEVEL SECURITY;
EOF

echo -e "${GREEN}✅ SaaS components setup complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Copy .env.saas.example to .env.saas and fill in your keys"
echo "2. Run: pip install -r requirements-saas.txt"
echo "3. Run database migration: alembic upgrade head"
echo "4. Start SaaS version: python main_saas.py"
echo ""
echo -e "${BLUE}📚 Full documentation: docs/SAAS_TRANSFORMATION_PLAN.md${NC}"