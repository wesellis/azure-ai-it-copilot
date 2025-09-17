# 🚀 Quick Start Guide

Get Azure AI IT Copilot running in 15 minutes!

## Prerequisites

### Required Azure Resources
- Azure Subscription (Owner or Contributor role)
- Azure OpenAI Service access (GPT-4 deployment)
- Azure Active Directory tenant
- Resource Group for deployment

### Local Development Tools
```bash
# Check your versions
node --version      # Requires 18.0+
python --version    # Requires 3.11+
docker --version    # Requires 20.10+
az --version        # Azure CLI 2.50+
pwsh --version      # PowerShell 7.0+
```

## 🎯 Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/azure-ai-it-copilot.git
cd azure-ai-it-copilot

# Run the interactive setup wizard
./setup.sh

# Or manually set environment variables
cp .env.example .env
nano .env
```

### Environment Variables
```env
# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-service-principal-id
AZURE_CLIENT_SECRET=your-service-principal-secret
AZURE_RESOURCE_GROUP=rg-ai-copilot

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=gpt-4-turbo
AZURE_OPENAI_API_VERSION=2024-02-01

# Database Configuration
COSMOS_DB_ENDPOINT=https://your-cosmos.documents.azure.com:443/
COSMOS_DB_KEY=your-cosmos-key
REDIS_CONNECTION_STRING=your-redis-connection

# Authentication
JWT_SECRET=generate-a-strong-secret-key
AZURE_AD_CLIENT_ID=your-app-registration-id
AZURE_AD_TENANT_ID=your-tenant-id

# Optional: Advanced Features
ENABLE_PREDICTIVE_MAINTENANCE=true
ENABLE_COST_OPTIMIZATION=true
ENABLE_AUTO_REMEDIATION=false
```

## 🎯 Step 2: Install Dependencies

```bash
# Install Python dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Node dependencies
npm install

# Install PowerShell modules
pwsh -Command "Install-Module -Name Az -Force"
```

## 🎯 Step 3: Deploy Azure Resources

### Option A: Automated Deployment
```bash
# Deploy all required Azure resources
./deploy/azure-setup.sh

# This creates:
# - Resource Group
# - Cosmos DB Account
# - Redis Cache
# - Key Vault
# - Application Insights
# - Storage Account
```

### Option B: Terraform Deployment
```bash
cd infrastructure/terraform
terraform init
terraform plan -var-file="environments/dev.tfvars"
terraform apply -auto-approve
```

### Option C: Manual Setup
```bash
# Create Resource Group
az group create --name rg-ai-copilot --location eastus2

# Create Cosmos DB
az cosmosdb create \
  --name cosmos-ai-copilot \
  --resource-group rg-ai-copilot \
  --default-consistency-level Session

# Create Redis Cache
az redis create \
  --name redis-ai-copilot \
  --resource-group rg-ai-copilot \
  --sku Basic \
  --vm-size C0
```

## 🎯 Step 4: Initialize Database

```bash
# Run database migrations
python scripts/init_database.py

# Seed with sample data (optional)
python scripts/seed_data.py --sample-data
```

## 🎯 Step 5: Start the Platform

### Development Mode
```bash
# Start all services with Docker Compose
docker-compose up -d

# Or start individually:
# Start API server
npm run dev:api

# Start AI orchestrator
python ai-orchestrator/main.py

# Start dashboard
npm run dev:dashboard
```

### Production Mode
```bash
# Build and start production containers
docker-compose -f docker-compose.prod.yml up -d

# Or deploy to Kubernetes
kubectl apply -f kubernetes/
```

## 🎯 Step 6: Access the Platform

### Web Dashboard
```
http://localhost:3000
```
Default credentials:
- Username: `admin@yourdomain.com`
- Password: `Welcome123!`

### CLI Interface
```bash
# Install CLI globally
npm install -g @azure-ai-copilot/cli

# Configure CLI
ai-copilot configure

# Test your first command
ai-copilot> "Show me all virtual machines in the production resource group"
```

### API Endpoint
```bash
# Test API health
curl http://localhost:8000/health

# Get access token
TOKEN=$(curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"Welcome123!"}' \
  | jq -r '.access_token')

# Make your first API call
curl http://localhost:8000/api/v1/command \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"command":"List all resources in subscription"}'
```

## 🎨 Your First Commands

### Infrastructure Management
```bash
# Natural language commands
"Create a Ubuntu VM with 8GB RAM in East US"
"List all stopped VMs older than 30 days"
"Delete all resources tagged as 'temporary'"
"Scale the production app service to 3 instances"
```

### Cost Optimization
```bash
"Show me the top 10 most expensive resources"
"Identify unused resources costing more than $100/month"
"Recommend reserved instances that would save money"
"Turn off all dev/test resources outside business hours"
```

### Incident Response
```bash
"Diagnose high CPU usage on VM-PROD-001"
"Show me all errors in the last hour"
"Fix the failing health checks on the web app"
"Restart all services in the backend resource group"
```

### Compliance & Security
```bash
"Check if all storage accounts have encryption enabled"
"List resources not compliant with CIS benchmarks"
"Enable diagnostic logs for all databases"
"Find all publicly accessible resources"
```

## 🔧 Configuration Options

### Basic Configuration
```yaml
# config/settings.yaml
ai_settings:
  model: gpt-4-turbo
  temperature: 0.3
  max_tokens: 2000

execution:
  require_approval: true
  dry_run: false
  max_parallel_operations: 5

monitoring:
  enable_metrics: true
  enable_tracing: true
  log_level: INFO
```

### Agent Configuration
```yaml
# config/agents.yaml
agents:
  infrastructure:
    enabled: true
    auto_remediation: false
    approval_required_for:
      - delete
      - scale_down

  incident:
    enabled: true
    auto_diagnose: true
    auto_fix_known_issues: true

  cost:
    enabled: true
    optimization_threshold: 100  # dollars
    auto_implement: false
```

## 🧪 Testing Your Setup

### Run Health Checks
```bash
# Check all components
./scripts/health_check.sh

# Expected output:
✅ Azure Connection: OK
✅ OpenAI Service: OK
✅ Database: OK
✅ Redis Cache: OK
✅ API Server: OK
✅ AI Orchestrator: OK
✅ Dashboard: OK
```

### Run Test Commands
```bash
# Test infrastructure query (read-only)
ai-copilot test "List all resource groups"

# Test cost analysis
ai-copilot test "What's our current monthly Azure spend?"

# Test incident diagnosis (simulation)
ai-copilot test --simulate "Diagnose high memory usage on VM-TEST"
```

## 🐛 Troubleshooting

### Common Issues

#### Issue: "Azure OpenAI rate limit exceeded"
```bash
# Solution: Implement retry logic or increase limits
az cognitive-services account update \
  --name your-openai \
  --resource-group rg-ai-copilot \
  --custom-domain your-openai
```

#### Issue: "Cannot connect to Redis"
```bash
# Check Redis connection
redis-cli -h your-redis.redis.cache.windows.net \
  -p 6380 \
  --tls \
  -a your-redis-key \
  ping
```

#### Issue: "Authentication failed"
```bash
# Verify Azure AD configuration
az ad app show --id $AZURE_AD_CLIENT_ID

# Test service principal
az login --service-principal \
  -u $AZURE_CLIENT_ID \
  -p $AZURE_CLIENT_SECRET \
  --tenant $AZURE_TENANT_ID
```

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
export TRACE_ENABLED=true

# Start with debug output
npm run dev:api -- --debug
```

## 📚 Next Steps

1. **Explore Natural Language Commands**
   - See [Command Examples](COMMAND_EXAMPLES.md) for inspiration
   - Learn about [Intent Patterns](INTENT_PATTERNS.md)

2. **Configure Agents**
   - Customize [Agent Behavior](AGENT_CONFIGURATION.md)
   - Build [Custom Agents](CUSTOM_AGENTS.md)

3. **Set Up Monitoring**
   - Configure [Dashboards](MONITORING.md)
   - Set up [Alerts](ALERTING.md)

4. **Production Deployment**
   - Follow [Production Guide](PRODUCTION_DEPLOYMENT.md)
   - Review [Security Checklist](SECURITY_CHECKLIST.md)

## 🆘 Getting Help

- 📖 [Full Documentation](https://docs.azure-ai-copilot.com)
- 💬 [Discord Community](https://discord.gg/azure-ai-copilot)
- 🐛 [Report Issues](https://github.com/yourusername/azure-ai-it-copilot/issues)
- 📧 [Email Support](mailto:support@azure-ai-copilot.com)

## 🎉 Congratulations!

You now have Azure AI IT Copilot running! Try these commands to see the power:

```bash
"Deploy a complete 3-tier application with database and monitoring"
"Optimize our Azure costs without affecting production"
"Set up disaster recovery for all critical resources"
```

Welcome to the future of IT operations! 🚀