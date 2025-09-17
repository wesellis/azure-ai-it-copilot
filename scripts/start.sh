#!/bin/bash

# Azure AI IT Copilot - Quick Start Script
# This script starts the application in development mode

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Azure AI IT Copilot - Starting Services${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${RED}Please update .env with your Azure credentials before continuing!${NC}"
    echo "Edit .env file and run this script again."
    exit 1
fi

# Source environment variables
set -a
source .env
set +a

# Check required environment variables
required_vars=("AZURE_SUBSCRIPTION_ID" "AZURE_OPENAI_ENDPOINT" "AZURE_OPENAI_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}Error: $var is not set in .env file${NC}"
        exit 1
    fi
done

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Start services
echo -e "${YELLOW}Starting Docker containers...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"

# Wait for API
max_attempts=30
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API server is ready${NC}"
        break
    fi
    echo -ne "\rWaiting for API... ($attempt/$max_attempts)"
    sleep 2
    ((attempt++))
done

if [ $attempt -gt $max_attempts ]; then
    echo -e "\n${RED}API server failed to start. Check logs with: docker-compose logs api${NC}"
    exit 1
fi

# Display service URLs
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Services are running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Dashboard:   ${GREEN}http://localhost:3000${NC}"
echo -e "API Docs:    ${GREEN}http://localhost:8000/docs${NC}"
echo -e "Grafana:     ${GREEN}http://localhost:3001${NC} (admin/admin)"
echo -e "Prometheus:  ${GREEN}http://localhost:9090${NC}"
echo ""
echo -e "${YELLOW}Default login: admin / admin123${NC}"
echo ""
echo -e "To view logs:    ${YELLOW}docker-compose logs -f${NC}"
echo -e "To stop:         ${YELLOW}docker-compose down${NC}"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"