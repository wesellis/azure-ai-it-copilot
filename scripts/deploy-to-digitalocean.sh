#!/bin/bash

# Azure AI IT Copilot - Digital Ocean Deployment Script
# Deploys to your existing Digital Ocean infrastructure

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
CREDENTIALS_PATH="/home/mookyjooky/Dropbox/SIDEGIGS/05 - Scripts & Tools/SECURE-DO-NOT-COMMIT"
PROJECT_NAME="azure-ai-it-copilot"
DOMAIN="azureaiitcopilot.com"

echo -e "${BLUE}🚀 Azure AI IT Copilot - Digital Ocean Deployment${NC}"
echo ""

# Check for credentials
if [ ! -f "$CREDENTIALS_PATH/digitalocean-config.env" ]; then
    echo -e "${RED}❌ Digital Ocean credentials not found!${NC}"
    exit 1
fi

if [ ! -f "$CREDENTIALS_PATH/stripe-live-config.env" ]; then
    echo -e "${RED}❌ Stripe credentials not found!${NC}"
    exit 1
fi

# Source credentials
source "$CREDENTIALS_PATH/digitalocean-config.env"
source "$CREDENTIALS_PATH/stripe-live-config.env"

echo -e "${GREEN}✅ Credentials loaded${NC}"

# Choose deployment server
echo -e "${YELLOW}Select deployment target:${NC}"
echo "1) New Droplet (recommended for production)"
echo "2) QuickRecon AI App server (165.227.182.12)"
echo "3) BidMaster Pro server (68.183.155.4)"
echo "4) Custom IP"
read -p "Choice (1-4): " choice

case $choice in
    1)
        TARGET_IP="NEW"
        echo -e "${GREEN}Will create new droplet${NC}"
        ;;
    2)
        TARGET_IP="$QUICKRECON_AI_APP_IP"
        TARGET_NAME="QuickRecon AI App server"
        ;;
    3)
        TARGET_IP="$BIDMASTER_PRO_APP_IP"
        TARGET_NAME="BidMaster Pro server"
        ;;
    4)
        read -p "Enter IP address: " TARGET_IP
        TARGET_NAME="Custom server"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Create deployment package
echo -e "${GREEN}Creating deployment package...${NC}"

# Create temp directory
TEMP_DIR="/tmp/${PROJECT_NAME}-deploy-$(date +%s)"
mkdir -p "$TEMP_DIR"

# Copy necessary files
cp -r . "$TEMP_DIR/" 2>/dev/null || true

# Create production environment file
cat > "$TEMP_DIR/.env.production" << EOF
# Production Environment Configuration
APP_NAME=Azure AI IT Copilot
APP_ENV=production
APP_VERSION=2.0.0-saas

# Database
DATABASE_URL=postgresql://azureai:secure_password@localhost:5432/azureai_prod

# Redis
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Azure Configuration (will be set by customers)
AZURE_TENANT_ID=customer_specific
AZURE_CLIENT_ID=customer_specific
AZURE_CLIENT_SECRET=customer_specific
AZURE_SUBSCRIPTION_ID=customer_specific

# Stripe Configuration (Live Keys)
STRIPE_PUBLISHABLE_KEY=${STRIPE_PUBLISHABLE_KEY}
STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
STRIPE_WEBHOOK_SECRET=whsec_will_be_generated

# Stripe Product IDs for Azure AI IT Copilot
STRIPE_PRODUCT_ID=azure_ai_copilot
STRIPE_STARTER_PRICE_ID=price_azure_starter
STRIPE_PRO_PRICE_ID=price_azure_pro
STRIPE_ENTERPRISE_PRICE_ID=price_azure_enterprise

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)

# Monitoring
SENTRY_DSN=
ENABLE_MONITORING=true

# Feature Flags
ENABLE_PREDICTIVE_AGENT=true
ENABLE_COST_OPTIMIZATION=true
ENABLE_COMPLIANCE=true
ENABLE_MULTI_TENANT=true

# Rate Limiting (per month)
STARTER_API_CALLS=1000
PRO_API_CALLS=10000
ENTERPRISE_API_CALLS=100000

# Trial Configuration
TRIAL_DAYS=14
EOF

# Create Docker Compose for production
cat > "$TEMP_DIR/docker-compose.prod.yml" << 'EOF'
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.optimized
    ports:
      - "8000:8000"
    environment:
      - ENV_FILE=.env.production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: always
    networks:
      - azure-ai-network

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: azureai_prod
      POSTGRES_USER: azureai
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - azure-ai-network
    restart: always

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - azure-ai-network
    restart: always

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    networks:
      - azure-ai-network
    restart: always

volumes:
  postgres_data:
  redis_data:

networks:
  azure-ai-network:
    driver: bridge
EOF

# Create nginx configuration
mkdir -p "$TEMP_DIR/nginx"
cat > "$TEMP_DIR/nginx/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    server {
        listen 80;
        server_name azureaiitcopilot.com www.azureaiitcopilot.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name azureaiitcopilot.com www.azureaiitcopilot.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        client_max_body_size 50M;

        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
EOF

# Create deployment script for server
cat > "$TEMP_DIR/setup-server.sh" << 'EOF'
#!/bin/bash

# Server setup script
echo "Setting up Azure AI IT Copilot on server..."

# Update system
apt-get update
apt-get upgrade -y

# Install Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

if ! command -v docker-compose &> /dev/null; then
    apt-get install -y docker-compose-plugin
fi

# Install other dependencies
apt-get install -y nginx certbot python3-certbot-nginx git

# Create application directory
mkdir -p /opt/azure-ai-copilot
cd /opt/azure-ai-copilot

# Start services
docker compose -f docker-compose.prod.yml up -d

# Set up SSL with Let's Encrypt
certbot --nginx -d azureaiitcopilot.com -d www.azureaiitcopilot.com --non-interactive --agree-tos -m wes@wesellis.com

# Set up cron for SSL renewal
(crontab -l 2>/dev/null; echo "0 0,12 * * * certbot renew --quiet") | crontab -

echo "✅ Server setup complete!"
EOF

chmod +x "$TEMP_DIR/setup-server.sh"

# Create tarball
echo -e "${GREEN}Creating deployment archive...${NC}"
cd "$TEMP_DIR"
tar -czf "${PROJECT_NAME}.tar.gz" .

if [ "$TARGET_IP" == "NEW" ]; then
    echo -e "${BLUE}Creating new Digital Ocean droplet...${NC}"

    # Create droplet using doctl (if installed)
    if command -v doctl &> /dev/null; then
        DROPLET_ID=$(doctl compute droplet create azure-ai-copilot \
            --region nyc3 \
            --size s-2vcpu-4gb \
            --image ubuntu-22-04-x64 \
            --ssh-keys $(doctl compute ssh-key list --format ID --no-header) \
            --wait \
            --format ID \
            --no-header)

        TARGET_IP=$(doctl compute droplet get $DROPLET_ID --format PublicIPv4 --no-header)
        echo -e "${GREEN}✅ New droplet created: $TARGET_IP${NC}"
    else
        echo -e "${YELLOW}Please create a droplet manually in Digital Ocean dashboard${NC}"
        echo "Recommended specs: 2 vCPU, 4GB RAM, 80GB SSD"
        read -p "Enter the new droplet IP when ready: " TARGET_IP
    fi
fi

# Deploy to server
echo -e "${BLUE}Deploying to $TARGET_IP...${NC}"

# Copy files to server
scp "${PROJECT_NAME}.tar.gz" root@${TARGET_IP}:/tmp/

# Execute setup on server
ssh root@${TARGET_IP} << ENDSSH
    cd /opt
    mkdir -p azure-ai-copilot
    cd azure-ai-copilot
    tar -xzf /tmp/${PROJECT_NAME}.tar.gz
    ./setup-server.sh
ENDSSH

# Cleanup
rm -rf "$TEMP_DIR"

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo -e "${BLUE}Access your application at:${NC}"
echo "  https://azureaiitcopilot.com (after DNS configuration)"
echo "  http://${TARGET_IP}:8000 (direct access)"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Configure DNS to point to: ${TARGET_IP}"
echo "2. Set up Stripe webhook endpoint"
echo "3. Configure customer Azure credentials"
echo "4. Monitor application logs"
echo ""
echo -e "${GREEN}🎉 Azure AI IT Copilot SaaS is live!${NC}"