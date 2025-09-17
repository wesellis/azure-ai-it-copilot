#!/bin/bash

# Azure AI IT Copilot - Deployment Script
# This script deploys the application to Azure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Azure AI IT Copilot - Deployment Script${NC}"
echo "========================================"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        echo -e "${RED}Azure CLI is not installed. Please install it first.${NC}"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install it first.${NC}"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}kubectl is not installed. Please install it first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All prerequisites met!${NC}"
}

# Load environment variables
load_env() {
    if [ -f .env ]; then
        echo -e "${YELLOW}Loading environment variables...${NC}"
        export $(cat .env | grep -v '^#' | xargs)
    else
        echo -e "${RED}.env file not found!${NC}"
        exit 1
    fi
}

# Login to Azure
az_login() {
    echo -e "${YELLOW}Logging into Azure...${NC}"
    az login --output table
    az account set --subscription "$AZURE_SUBSCRIPTION_ID"
    echo -e "${GREEN}Successfully logged into Azure!${NC}"
}

# Create resource group
create_resource_group() {
    echo -e "${YELLOW}Creating resource group...${NC}"
    az group create \
        --name "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --output table
    echo -e "${GREEN}Resource group created!${NC}"
}

# Create Azure Container Registry
create_acr() {
    echo -e "${YELLOW}Creating Azure Container Registry...${NC}"
    az acr create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$ACR_NAME" \
        --sku Basic \
        --admin-enabled true \
        --output table
    echo -e "${GREEN}ACR created!${NC}"
}

# Build and push Docker images
build_and_push_images() {
    echo -e "${YELLOW}Building and pushing Docker images...${NC}"
    
    # Login to ACR
    az acr login --name "$ACR_NAME"
    
    # Get ACR login server
    ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer --output tsv)
    
    # Build and push API image
    echo "Building API image..."
    docker build -t "$ACR_LOGIN_SERVER/ai-copilot-api:latest" -f Dockerfile.api .
    docker push "$ACR_LOGIN_SERVER/ai-copilot-api:latest"
    
    # Build and push Dashboard image
    echo "Building Dashboard image..."
    docker build -t "$ACR_LOGIN_SERVER/ai-copilot-dashboard:latest" -f dashboard/Dockerfile ./dashboard
    docker push "$ACR_LOGIN_SERVER/ai-copilot-dashboard:latest"
    
    echo -e "${GREEN}Images built and pushed!${NC}"
}

# Create AKS cluster
create_aks() {
    echo -e "${YELLOW}Creating AKS cluster...${NC}"
    az aks create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$AKS_CLUSTER_NAME" \
        --node-count 3 \
        --node-vm-size Standard_D2s_v3 \
        --enable-managed-identity \
        --generate-ssh-keys \
        --attach-acr "$ACR_NAME" \
        --output table
    echo -e "${GREEN}AKS cluster created!${NC}"
}

# Deploy to AKS
deploy_to_aks() {
    echo -e "${YELLOW}Deploying to AKS...${NC}"
    
    # Get AKS credentials
    az aks get-credentials --resource-group "$RESOURCE_GROUP" --name "$AKS_CLUSTER_NAME" --overwrite-existing
    
    # Create namespace
    kubectl create namespace ai-copilot --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic ai-copilot-secrets \
        --from-literal=AZURE_SUBSCRIPTION_ID="$AZURE_SUBSCRIPTION_ID" \
        --from-literal=AZURE_TENANT_ID="$AZURE_TENANT_ID" \
        --from-literal=AZURE_CLIENT_ID="$AZURE_CLIENT_ID" \
        --from-literal=AZURE_CLIENT_SECRET="$AZURE_CLIENT_SECRET" \
        --from-literal=AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
        --from-literal=AZURE_OPENAI_KEY="$AZURE_OPENAI_KEY" \
        --from-literal=JWT_SECRET_KEY="$JWT_SECRET_KEY" \
        --namespace=ai-copilot \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/ --namespace=ai-copilot
    
    echo -e "${GREEN}Application deployed to AKS!${NC}"
}

# Create Azure Redis Cache
create_redis() {
    echo -e "${YELLOW}Creating Azure Redis Cache...${NC}"
    az redis create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$REDIS_NAME" \
        --location "$LOCATION" \
        --sku Basic \
        --vm-size c0 \
        --output table
    echo -e "${GREEN}Redis Cache created!${NC}"
}

# Create Application Insights
create_app_insights() {
    echo -e "${YELLOW}Creating Application Insights...${NC}"
    az monitor app-insights component create \
        --app "$APP_INSIGHTS_NAME" \
        --location "$LOCATION" \
        --resource-group "$RESOURCE_GROUP" \
        --output table
    echo -e "${GREEN}Application Insights created!${NC}"
}

# Setup monitoring
setup_monitoring() {
    echo -e "${YELLOW}Setting up monitoring...${NC}"
    
    # Create Log Analytics Workspace
    az monitor log-analytics workspace create \
        --resource-group "$RESOURCE_GROUP" \
        --workspace-name "$LOG_ANALYTICS_WORKSPACE" \
        --location "$LOCATION" \
        --output table
    
    # Enable monitoring for AKS
    az aks enable-addons \
        --resource-group "$RESOURCE_GROUP" \
        --name "$AKS_CLUSTER_NAME" \
        --addons monitoring \
        --output table
    
    echo -e "${GREEN}Monitoring setup complete!${NC}"
}

# Get application URLs
get_urls() {
    echo -e "${YELLOW}Getting application URLs...${NC}"
    
    # Get external IPs
    API_IP=$(kubectl get svc ai-copilot-api --namespace=ai-copilot -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    DASHBOARD_IP=$(kubectl get svc ai-copilot-dashboard --namespace=ai-copilot -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    echo -e "${GREEN}Application deployed successfully!${NC}"
    echo "========================================"
    echo -e "API URL: ${GREEN}http://$API_IP:8000${NC}"
    echo -e "Dashboard URL: ${GREEN}http://$DASHBOARD_IP${NC}"
    echo "========================================"
}

# Main deployment flow
main() {
    echo "Starting deployment..."
    
    # Set default values
    RESOURCE_GROUP=${RESOURCE_GROUP:-"rg-ai-copilot"}
    LOCATION=${LOCATION:-"eastus"}
    ACR_NAME=${ACR_NAME:-"aicopilotacr"}
    AKS_CLUSTER_NAME=${AKS_CLUSTER_NAME:-"aks-ai-copilot"}
    REDIS_NAME=${REDIS_NAME:-"redis-ai-copilot"}
    APP_INSIGHTS_NAME=${APP_INSIGHTS_NAME:-"ai-copilot-insights"}
    LOG_ANALYTICS_WORKSPACE=${LOG_ANALYTICS_WORKSPACE:-"ai-copilot-logs"}
    
    check_prerequisites
    load_env
    az_login
    
    # Check if resource group exists
    if az group exists --name "$RESOURCE_GROUP" | grep -q true; then
        echo -e "${YELLOW}Resource group already exists. Using existing group.${NC}"
    else
        create_resource_group
    fi
    
    # Deploy resources
    create_acr
    build_and_push_images
    create_aks
    create_redis
    create_app_insights
    setup_monitoring
    deploy_to_aks
    
    # Wait for services to be ready
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    kubectl wait --for=condition=available --timeout=300s deployment/ai-copilot-api --namespace=ai-copilot
    kubectl wait --for=condition=available --timeout=300s deployment/ai-copilot-dashboard --namespace=ai-copilot
    
    get_urls
    
    echo -e "${GREEN}Deployment completed successfully!${NC}"
}

# Handle command line arguments
case "${1}" in
    --help)
        echo "Usage: ./deploy.sh [options]"
        echo "Options:"
        echo "  --help          Show this help message"
        echo "  --destroy       Destroy all resources"
        echo "  --update        Update existing deployment"
        exit 0
        ;;
    --destroy)
        echo -e "${RED}Destroying all resources...${NC}"
        az group delete --name "$RESOURCE_GROUP" --yes --no-wait
        echo -e "${GREEN}Resources marked for deletion.${NC}"
        exit 0
        ;;
    --update)
        echo -e "${YELLOW}Updating deployment...${NC}"
        load_env
        build_and_push_images
        kubectl rollout restart deployment --namespace=ai-copilot
        echo -e "${GREEN}Update completed!${NC}"
        exit 0
        ;;
    *)
        main
        ;;
esac