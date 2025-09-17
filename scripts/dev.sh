#!/bin/bash

# Development mode startup script
# Runs API and Dashboard without Docker

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting in development mode...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed${NC}"
    exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed${NC}"
    exit 1
fi

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

if ! pip show fastapi > /dev/null 2>&1; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -r requirements.txt
fi

# Install Node dependencies if needed
if [ ! -d "dashboard/node_modules" ]; then
    echo -e "${YELLOW}Installing Node dependencies...${NC}"
    cd dashboard && npm install && cd ..
fi

# Start Redis using Docker
echo -e "${YELLOW}Starting Redis...${NC}"
docker run -d --name redis-dev -p 6379:6379 redis:7-alpine 2>/dev/null || docker start redis-dev

# Source environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Start API server in background
echo -e "${YELLOW}Starting API server...${NC}"
cd api && uvicorn server:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!
cd ..

# Start React dashboard
echo -e "${YELLOW}Starting React dashboard...${NC}"
cd dashboard && npm start &
DASHBOARD_PID=$!
cd ..

echo -e "${GREEN}Development servers started!${NC}"
echo "API PID: $API_PID"
echo "Dashboard PID: $DASHBOARD_PID"
echo ""
echo "API: http://localhost:8000"
echo "Dashboard: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Trap Ctrl+C and kill background processes
trap "kill $API_PID $DASHBOARD_PID; docker stop redis-dev; exit" INT

# Wait
wait