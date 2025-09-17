#!/bin/bash

# Azure AI IT Copilot - MINIMAL COST Deployment
# Start at $5-10/month, scale as you grow!

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}💰 Azure AI IT Copilot - Ultra Low Cost Deployment${NC}"
echo -e "${GREEN}Starting at just $5-10/month!${NC}"
echo ""

# Deployment options
echo -e "${YELLOW}Choose your deployment strategy:${NC}"
echo ""
echo "1) 🆓 FREE TIER - Vercel + Supabase ($0/month)"
echo "   - Vercel: Free hosting for API"
echo "   - Supabase: Free PostgreSQL + Auth"
echo "   - Perfect for 0-10 customers"
echo ""
echo "2) 💵 MINIMAL TIER - Shared Hosting ($5/month)"
echo "   - Deploy on existing ShiftSwap server (24.144.84.9)"
echo "   - Uses only 1GB RAM droplet"
echo "   - Good for 10-50 customers"
echo ""
echo "3) 📦 CONTAINERIZED - Docker on existing server ($0 extra)"
echo "   - Deploy alongside QuickRecon or BidMaster"
echo "   - Share resources with existing apps"
echo "   - Good for 50-100 customers"
echo ""
echo "4) 🚀 SCALE TIER - Dedicated small droplet ($12/month)"
echo "   - 1 vCPU, 2GB RAM droplet"
echo "   - Dedicated resources"
echo "   - Good for 100-500 customers"
echo ""
read -p "Choice (1-4): " choice

case $choice in
    1)
        echo -e "${GREEN}Setting up FREE TIER deployment...${NC}"

        # Create Vercel deployment configuration
        cat > vercel.json << 'EOF'
{
  "functions": {
    "api/*.py": {
      "runtime": "python3.9"
    }
  },
  "rewrites": [
    { "source": "/(.*)", "destination": "/api/main" }
  ],
  "env": {
    "SUPABASE_URL": "@supabase-url",
    "SUPABASE_KEY": "@supabase-key",
    "STRIPE_SECRET_KEY": "@stripe-secret-key"
  }
}
EOF

        # Create Supabase setup script
        cat > setup-supabase.sql << 'EOF'
-- Free Supabase setup
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    subscription_tier TEXT DEFAULT 'trial',
    stripe_customer_id TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE usage_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    api_calls INTEGER DEFAULT 0,
    month DATE DEFAULT date_trunc('month', NOW()),
    UNIQUE(tenant_id, month)
);

-- Row Level Security
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_tracking ENABLE ROW LEVEL SECURITY;
EOF

        echo -e "${GREEN}Instructions for FREE deployment:${NC}"
        echo "1. Create free Supabase account at: https://supabase.com"
        echo "2. Run the SQL script above in Supabase SQL editor"
        echo "3. Install Vercel CLI: npm i -g vercel"
        echo "4. Run: vercel deploy"
        echo "5. Add environment variables in Vercel dashboard"
        echo ""
        echo -e "${YELLOW}Monthly cost: $0${NC}"
        echo -e "${GREEN}Can handle: ~1000 API calls/month free${NC}"
        ;;

    2)
        echo -e "${GREEN}Setting up MINIMAL TIER on ShiftSwap server...${NC}"

        MINIMAL_IP="24.144.84.9"

        # Create minimal Docker compose
        cat > docker-compose.minimal.yml << 'EOF'
version: '3.8'

services:
  azure-ai-api:
    build:
      context: .
      dockerfile: Dockerfile.minimal
    ports:
      - "8001:8000"  # Different port to avoid conflicts
    environment:
      - DATABASE_URL=sqlite:///./azure_ai.db  # SQLite for minimal cost
      - REDIS_URL=  # No Redis in minimal mode
      - CACHE_TYPE=memory  # In-memory caching
    volumes:
      - ./data:/app/data
    restart: always
    mem_limit: 512m  # Limit memory usage
    cpus: 0.5  # Limit CPU usage
EOF

        # Create minimal Dockerfile
        cat > Dockerfile.minimal << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install only essential dependencies
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

COPY . .

# Use single worker for minimal memory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
EOF

        # Create minimal requirements
        cat > requirements-minimal.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
stripe==7.0.0
pydantic==2.5.0
python-jose[cryptography]==3.3.0
httpx==0.25.2
# Remove heavy dependencies like pandas, numpy, etc.
EOF

        echo "Deploying to ShiftSwap server..."
        scp docker-compose.minimal.yml root@${MINIMAL_IP}:/opt/azure-ai/
        ssh root@${MINIMAL_IP} "cd /opt/azure-ai && docker-compose -f docker-compose.minimal.yml up -d"

        echo -e "${GREEN}Deployed to minimal server!${NC}"
        echo -e "${YELLOW}Monthly cost: $5 (using existing droplet)${NC}"
        ;;

    3)
        echo -e "${GREEN}Setting up CONTAINERIZED on existing server...${NC}"

        echo "Which server to use?"
        echo "1) QuickRecon (165.227.182.12) - 8GB RAM"
        echo "2) BidMaster (68.183.155.4) - 4GB RAM"
        read -p "Choice: " server_choice

        if [ "$server_choice" == "1" ]; then
            SHARED_IP="165.227.182.12"
            PORT="8002"
        else
            SHARED_IP="68.183.155.4"
            PORT="8003"
        fi

        # Create shared deployment config
        cat > docker-compose.shared.yml << EOF
version: '3.8'

services:
  azure-ai:
    container_name: azure-ai-copilot
    build: .
    ports:
      - "${PORT}:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@shared-db:5432/azure_ai
    networks:
      - shared-network
    restart: always
    mem_limit: 1g  # Limit to 1GB
    cpus: 1.0  # Limit to 1 CPU

networks:
  shared-network:
    external: true  # Use existing network from other apps
EOF

        echo "Deploying to shared server at ${SHARED_IP}..."
        scp docker-compose.shared.yml root@${SHARED_IP}:/opt/azure-ai/
        ssh root@${SHARED_IP} "cd /opt/azure-ai && docker-compose -f docker-compose.shared.yml up -d"

        echo -e "${GREEN}Deployed to shared server!${NC}"
        echo -e "${YELLOW}Monthly cost: $0 (using existing server)${NC}"
        echo -e "${BLUE}Access at: http://${SHARED_IP}:${PORT}${NC}"
        ;;

    4)
        echo -e "${GREEN}Setting up SCALE TIER...${NC}"

        # Create new small droplet
        cat > create-small-droplet.sh << 'EOF'
#!/bin/bash
doctl compute droplet create azure-ai-copilot-small \
    --region nyc3 \
    --size s-1vcpu-2gb \
    --image docker-20-04 \
    --ssh-keys $(doctl compute ssh-key list --format ID --no-header) \
    --user-data-file cloud-init.yml \
    --wait
EOF

        # Create cloud-init for automated setup
        cat > cloud-init.yml << 'EOF'
#cloud-config
package_update: true
packages:
  - docker.io
  - docker-compose

write_files:
  - path: /opt/start.sh
    content: |
      #!/bin/bash
      cd /opt/azure-ai
      docker-compose up -d

runcmd:
  - mkdir -p /opt/azure-ai
  - systemctl enable docker
  - systemctl start docker
EOF

        echo -e "${YELLOW}This will create a $12/month droplet${NC}"
        read -p "Continue? (y/n): " confirm

        if [ "$confirm" == "y" ]; then
            bash create-small-droplet.sh
            echo -e "${GREEN}Small droplet created!${NC}"
        fi
        ;;
esac

# Create cost scaling guide
cat > SCALING-GUIDE.md << 'EOF'
# Azure AI IT Copilot - Cost Scaling Guide

## Phase 1: FREE ($0/month)
- **Customers**: 0-10
- **Infrastructure**: Vercel + Supabase free tier
- **Limits**: 100GB bandwidth, 500MB database
- **Upgrade when**: Hitting free tier limits

## Phase 2: MINIMAL ($5-10/month)
- **Customers**: 10-50
- **Infrastructure**: Shared 1GB droplet or existing server
- **Database**: SQLite or shared PostgreSQL
- **Upgrade when**: >1000 API calls/day

## Phase 3: SHARED ($0 extra)
- **Customers**: 50-100
- **Infrastructure**: Containerized on existing apps
- **Database**: Shared PostgreSQL
- **Upgrade when**: Performance impacts other apps

## Phase 4: SMALL ($12-24/month)
- **Customers**: 100-500
- **Infrastructure**: 1vCPU/2GB dedicated droplet
- **Database**: Managed PostgreSQL ($15/month)
- **Upgrade when**: >10000 API calls/day

## Phase 5: STANDARD ($48/month)
- **Customers**: 500-1000
- **Infrastructure**: 2vCPU/4GB droplet
- **Database**: Managed PostgreSQL with replicas
- **Upgrade when**: Revenue > $50k/month

## Revenue vs Cost Analysis

| Customers | Revenue/mo | Infra Cost | Profit  | Margin |
|-----------|------------|------------|---------|--------|
| 1         | $299       | $0         | $299    | 100%   |
| 5         | $1,495     | $5         | $1,490  | 99.7%  |
| 10        | $2,990     | $10        | $2,980  | 99.7%  |
| 25        | $7,475     | $12        | $7,463  | 99.8%  |
| 50        | $14,950    | $24        | $14,926 | 99.8%  |
| 100       | $29,900    | $48        | $29,852 | 99.8%  |

## Automatic Scaling Triggers

```javascript
// Auto-scale logic
if (monthlyRevenue < 1000) {
    infrastructure = "FREE_TIER";
} else if (monthlyRevenue < 5000) {
    infrastructure = "MINIMAL";
} else if (monthlyRevenue < 15000) {
    infrastructure = "SHARED";
} else if (monthlyRevenue < 30000) {
    infrastructure = "SMALL";
} else {
    infrastructure = "STANDARD";
}
```
EOF

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Minimal cost deployment configured!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Cost Summary:${NC}"
echo "• Start: $0-5/month"
echo "• 10 customers: Still under $10/month"
echo "• 50 customers: Still under $25/month"
echo "• 100 customers: Still under $50/month"
echo ""
echo -e "${YELLOW}Your profit margins:${NC}"
echo "• 1 customer ($299) = 100% profit"
echo "• 10 customers ($2,990) = 99.7% profit"
echo "• 50 customers ($14,950) = 99.8% profit"
echo ""
echo -e "${GREEN}Scale only when you have paying customers!${NC}"