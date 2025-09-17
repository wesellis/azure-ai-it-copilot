# Deployment Verification Guide

## Overview

This guide provides step-by-step verification procedures for deploying the Azure AI IT Copilot to production environments, ensuring all components are functional and properly configured.

## Pre-Deployment Checklist

### Environment Requirements ✅

**Azure Subscription**:
- [x] Azure subscription with Owner or Contributor access
- [x] Azure OpenAI service enabled
- [x] Sufficient quotas for AKS, Cosmos DB, and other services

**Development Environment**:
- [x] Python 3.11+ installed
- [x] Node.js 18+ for dashboard
- [x] Docker Desktop running
- [x] Azure CLI installed and authenticated
- [x] kubectl configured
- [x] Terraform 1.5+ installed

### Configuration Verification

**Environment Variables**:
```bash
# Required for deployment
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-app-id"
export AZURE_CLIENT_SECRET="your-app-secret"
export AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/"
export AZURE_OPENAI_KEY="your-openai-key"
export JWT_SECRET_KEY="your-jwt-secret-key"
```

**Resource Naming**:
```bash
# Default resource names (customizable)
export RESOURCE_GROUP="rg-ai-copilot"
export LOCATION="eastus"
export ACR_NAME="aicopilotacr"
export AKS_CLUSTER_NAME="aks-ai-copilot"
export REDIS_NAME="redis-ai-copilot"
```

## Phase 1: Infrastructure Deployment

### Step 1: Deploy Core Infrastructure

```bash
# Navigate to project directory
cd /path/to/azure-ai-it-copilot

# Initialize Terraform
cd infrastructure/terraform
terraform init

# Plan deployment
terraform plan -out=tfplan

# Apply infrastructure
terraform apply tfplan
```

**Expected Resources**:
- Resource Group
- AKS Cluster (3 nodes)
- Azure Container Registry
- Cosmos DB account
- Redis Cache
- Application Insights
- Log Analytics Workspace
- Storage Account
- Key Vault

### Step 2: Verify Infrastructure

```bash
# Check resource group
az group show --name $RESOURCE_GROUP

# Verify AKS cluster
az aks show --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER_NAME

# Check ACR
az acr show --name $ACR_NAME

# Test database connectivity
az cosmosdb show --resource-group $RESOURCE_GROUP --name cosmos-ai-copilot
```

### Step 3: Configure Kubernetes

```bash
# Get AKS credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER_NAME

# Verify cluster access
kubectl cluster-info

# Create namespace
kubectl create namespace ai-copilot

# Verify namespace
kubectl get namespaces
```

## Phase 2: Application Deployment

### Step 1: Build and Push Container Images

```bash
# Return to project root
cd /path/to/azure-ai-it-copilot

# Login to ACR
az acr login --name $ACR_NAME

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)

# Build API image
docker build -t $ACR_LOGIN_SERVER/ai-copilot-api:latest -f Dockerfile.api .

# Push API image
docker push $ACR_LOGIN_SERVER/ai-copilot-api:latest

# Build Dashboard image
docker build -t $ACR_LOGIN_SERVER/ai-copilot-dashboard:latest -f dashboard/Dockerfile ./dashboard

# Push Dashboard image
docker push $ACR_LOGIN_SERVER/ai-copilot-dashboard:latest
```

### Step 2: Deploy Application to Kubernetes

```bash
# Create secrets
kubectl create secret generic ai-copilot-secrets \
  --from-literal=AZURE_SUBSCRIPTION_ID="$AZURE_SUBSCRIPTION_ID" \
  --from-literal=AZURE_TENANT_ID="$AZURE_TENANT_ID" \
  --from-literal=AZURE_CLIENT_ID="$AZURE_CLIENT_ID" \
  --from-literal=AZURE_CLIENT_SECRET="$AZURE_CLIENT_SECRET" \
  --from-literal=AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
  --from-literal=AZURE_OPENAI_KEY="$AZURE_OPENAI_KEY" \
  --from-literal=JWT_SECRET_KEY="$JWT_SECRET_KEY" \
  --namespace=ai-copilot

# Apply Kubernetes manifests
kubectl apply -f k8s/ --namespace=ai-copilot

# Verify deployments
kubectl get deployments --namespace=ai-copilot
kubectl get services --namespace=ai-copilot
kubectl get pods --namespace=ai-copilot
```

### Step 3: Wait for Services to be Ready

```bash
# Wait for API deployment
kubectl wait --for=condition=available --timeout=300s deployment/ai-copilot-api --namespace=ai-copilot

# Wait for Dashboard deployment
kubectl wait --for=condition=available --timeout=300s deployment/ai-copilot-dashboard --namespace=ai-copilot

# Check pod status
kubectl get pods --namespace=ai-copilot -w
```

## Phase 3: Application Verification

### Step 1: Get Service URLs

```bash
# Get external IPs
API_IP=$(kubectl get svc ai-copilot-api --namespace=ai-copilot -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
DASHBOARD_IP=$(kubectl get svc ai-copilot-dashboard --namespace=ai-copilot -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "API URL: http://$API_IP:8000"
echo "Dashboard URL: http://$DASHBOARD_IP"
```

### Step 2: Health Check Verification

```bash
# Test API health endpoint
curl -f http://$API_IP:8000/health

# Expected response:
# {"status":"healthy","timestamp":"2025-01-16T..."}

# Test detailed health (requires authentication)
# First get auth token:
TOKEN=$(curl -X POST http://$API_IP:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r .access_token)

# Test detailed health
curl -H "Authorization: Bearer $TOKEN" http://$API_IP:8000/health/detailed
```

### Step 3: API Functionality Tests

```bash
# Test command processing endpoint
curl -X POST http://$API_IP:8000/api/v1/command \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "Show me the status of all resources",
    "dry_run": true
  }'

# Test resource query endpoint
curl -X POST http://$API_IP:8000/api/v1/resources/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "resource_type": "virtualMachines",
    "include_metrics": false
  }'
```

### Step 4: WebSocket Connection Test

```javascript
// Test WebSocket connection (use browser console or WebSocket client)
const ws = new WebSocket(`ws://${API_IP}:8000/ws`);

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
    ws.send(JSON.stringify({type: 'ping'}));
};

ws.onmessage = function(event) {
    console.log('Received:', JSON.parse(event.data));
};
```

### Step 5: Dashboard Verification

```bash
# Open dashboard in browser
open http://$DASHBOARD_IP

# Or test dashboard health
curl -f http://$DASHBOARD_IP/
```

## Phase 4: Integration Testing

### Step 1: Test AI Orchestrator

```bash
# Test intent classification
curl -X POST http://$API_IP:8000/api/v1/command \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "Create a new virtual machine in East US with 4GB RAM",
    "dry_run": true
  }'

# Expected: Intent classified as RESOURCE_CREATE
```

### Step 2: Test Database Connectivity

```bash
# Check Cosmos DB connection through API
curl -X GET http://$API_IP:8000/api/v1/history?limit=10 \
  -H "Authorization: Bearer $TOKEN"

# Should return command history (may be empty initially)
```

### Step 3: Test Redis Cache

```bash
# Test caching through command execution
COMMAND_ID=$(curl -X POST http://$API_IP:8000/api/v1/command \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"command": "List all resource groups", "dry_run": true}' | jq -r .request_id)

# Retrieve cached result
curl -X GET http://$API_IP:8000/api/v1/command/$COMMAND_ID \
  -H "Authorization: Bearer $TOKEN"
```

## Phase 5: Performance Verification

### Step 1: Load Testing

```bash
# Install load testing tools
pip install locust

# Run basic load test
locust -f tests/performance/locustfile.py --host=http://$API_IP:8000 --users 10 --spawn-rate 2 -t 60s
```

### Step 2: Resource Monitoring

```bash
# Monitor pod resources
kubectl top pods --namespace=ai-copilot

# Check cluster node status
kubectl top nodes

# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s http://$API_IP:8000/health
```

### Step 3: Scaling Verification

```bash
# Test horizontal pod autoscaling
kubectl autoscale deployment ai-copilot-api --cpu-percent=70 --min=2 --max=10 --namespace=ai-copilot

# Generate load and monitor scaling
kubectl get hpa --namespace=ai-copilot -w
```

## Phase 6: Security Verification

### Step 1: Authentication Testing

```bash
# Test unauthorized access (should fail)
curl -X GET http://$API_IP:8000/health/detailed
# Expected: 401 Unauthorized

# Test with invalid token (should fail)
curl -H "Authorization: Bearer invalid-token" http://$API_IP:8000/health/detailed
# Expected: 401 Unauthorized

# Test with valid token (should succeed)
curl -H "Authorization: Bearer $TOKEN" http://$API_IP:8000/health/detailed
# Expected: 200 OK with detailed health info
```

### Step 2: RBAC Testing

```bash
# Test owner permissions (create token with owner role)
OWNER_TOKEN=$(curl -X POST http://$API_IP:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r .access_token)

# Test resource deletion (owner only)
curl -X DELETE http://$API_IP:8000/api/v1/resources/test-resource \
  -H "Authorization: Bearer $OWNER_TOKEN"
```

### Step 3: Network Security

```bash
# Test TLS configuration (if enabled)
curl -k https://$API_IP:8000/health

# Check pod security contexts
kubectl describe pod -l app=ai-copilot-api --namespace=ai-copilot | grep -A 5 "Security Context"
```

## Phase 7: Monitoring and Observability

### Step 1: Application Insights

```bash
# Verify Application Insights integration
az monitor app-insights component show --app ai-copilot-insights --resource-group $RESOURCE_GROUP

# Check metrics collection
curl http://$API_IP:8000/metrics
```

### Step 2: Log Analytics

```bash
# Query application logs
az monitor log-analytics query --workspace ai-copilot-logs \
  --analytics-query "ContainerLog | where TimeGenerated > ago(1h) | limit 10"
```

### Step 3: Custom Metrics

```bash
# Test Prometheus metrics endpoint
curl http://$API_IP:8000/metrics | grep api_requests_total

# Expected: Prometheus formatted metrics
```

## Phase 8: Disaster Recovery Testing

### Step 1: Pod Failure Simulation

```bash
# Delete a pod and verify automatic recovery
kubectl delete pod -l app=ai-copilot-api --namespace=ai-copilot

# Monitor pod recreation
kubectl get pods --namespace=ai-copilot -w
```

### Step 2: Node Failure Simulation

```bash
# Drain a node and verify pod rescheduling
kubectl drain <node-name> --ignore-daemonsets --delete-local-data

# Monitor pod distribution
kubectl get pods --namespace=ai-copilot -o wide
```

### Step 3: Database Connectivity Issues

```bash
# Test API behavior with database issues
# (This would require temporarily blocking Cosmos DB access)
```

## Troubleshooting Common Issues

### Pod Startup Issues

```bash
# Check pod logs
kubectl logs -l app=ai-copilot-api --namespace=ai-copilot

# Check events
kubectl get events --namespace=ai-copilot --sort-by='.lastTimestamp'

# Describe problematic pods
kubectl describe pod <pod-name> --namespace=ai-copilot
```

### Service Discovery Issues

```bash
# Check service endpoints
kubectl get endpoints --namespace=ai-copilot

# Test internal service communication
kubectl run test-pod --image=busybox --rm -it --restart=Never --namespace=ai-copilot -- /bin/sh
# Inside pod: wget -qO- http://ai-copilot-api:8000/health
```

### Image Pull Issues

```bash
# Check if ACR is attached to AKS
az aks check-acr --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER_NAME --acr $ACR_NAME

# Verify image exists in ACR
az acr repository list --name $ACR_NAME
az acr repository show-tags --name $ACR_NAME --repository ai-copilot-api
```

## Post-Deployment Validation

### Automated Health Checks

Create monitoring script:
```bash
#!/bin/bash
# health-check.sh

API_URL="http://$API_IP:8000"

# Test health endpoint
if curl -f $API_URL/health > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# Test authentication
TOKEN=$(curl -s -X POST $API_URL/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r .access_token)

if [ "$TOKEN" != "null" ] && [ "$TOKEN" != "" ]; then
    echo "✅ Authentication working"
else
    echo "❌ Authentication failed"
    exit 1
fi

# Test API functionality
if curl -s -H "Authorization: Bearer $TOKEN" $API_URL/health/detailed > /dev/null; then
    echo "✅ API functionality verified"
else
    echo "❌ API functionality failed"
    exit 1
fi

echo "🎉 All verification checks passed!"
```

### Continuous Monitoring Setup

```bash
# Set up automated monitoring (example with Azure Monitor)
az monitor metric alert create \
  --name "AI Copilot API High Response Time" \
  --resource-group $RESOURCE_GROUP \
  --scopes "/subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Insights/components/ai-copilot-insights" \
  --condition "avg requests/duration > 1000" \
  --description "Alert when API response time exceeds 1 second"
```

## Success Criteria

The deployment is considered successful when:

✅ **Infrastructure**: All Azure resources created and accessible
✅ **Application**: All pods running and healthy
✅ **API**: All endpoints responding correctly
✅ **Authentication**: JWT authentication working
✅ **Database**: Cosmos DB and Redis connectivity verified
✅ **Monitoring**: Metrics and logs being collected
✅ **Security**: RBAC and network policies enforced
✅ **Performance**: Response times within acceptable limits
✅ **Scalability**: Auto-scaling functioning correctly
✅ **Resilience**: Self-healing capabilities verified

## Next Steps

After successful deployment:

1. **User Onboarding**: Set up user accounts and roles
2. **Training**: Conduct user training sessions
3. **Gradual Rollout**: Start with pilot users before full deployment
4. **Monitoring**: Establish ongoing monitoring and alerting
5. **Maintenance**: Set up regular maintenance windows
6. **Backup**: Implement backup and disaster recovery procedures

---

**Deployment Guide Version**: 1.0
**Last Updated**: 2025-01-16
**Next Review**: After first production deployment