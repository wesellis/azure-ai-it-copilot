#!/bin/bash

# Azure AI IT Copilot - Development Setup Script
# This script sets up the development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Azure AI IT Copilot - Development Setup${NC}"
echo "========================================"

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_status "Checking operating system..."

    case "$(uname -s)" in
        Linux*)     MACHINE=Linux;;
        Darwin*)    MACHINE=Mac;;
        CYGWIN*)    MACHINE=Cygwin;;
        MINGW*)     MACHINE=MinGw;;
        *)          MACHINE="UNKNOWN:$(uname -s)"
    esac

    print_status "Detected OS: $MACHINE"

    if [[ "$MACHINE" == "UNKNOWN"* ]]; then
        print_error "Unsupported operating system"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3.11+ is required but not installed"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_status "Python version: $PYTHON_VERSION"

    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_warning "Node.js not found. Dashboard will not be available."
        SKIP_DASHBOARD=true
    else
        NODE_VERSION=$(node --version)
        print_status "Node.js version: $NODE_VERSION"
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker not found. Container deployment will not be available."
        SKIP_DOCKER=true
    else
        DOCKER_VERSION=$(docker --version)
        print_status "Docker version: $DOCKER_VERSION"
    fi

    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        print_warning "Azure CLI not found. Azure deployment will not be available."
        print_warning "Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    else
        AZ_VERSION=$(az --version | head -n 1)
        print_status "Azure CLI: $AZ_VERSION"
    fi
}

# Setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."

    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        print_status "Created virtual environment"
    else
        print_status "Virtual environment already exists"
    fi

    # Activate virtual environment
    source .venv/bin/activate
    print_status "Activated virtual environment"

    # Upgrade pip
    python -m pip install --upgrade pip

    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt

    # Install development dependencies
    pip install -r requirements-dev.txt 2>/dev/null || {
        print_warning "Development requirements not found, installing basic dev tools"
        pip install pytest pytest-asyncio pytest-mock black ruff mypy
    }

    print_status "Python environment setup complete"
}

# Setup configuration
setup_config() {
    print_status "Setting up configuration..."

    # Copy .env.example to .env if it doesn't exist
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_status "Created .env file from template"
        print_warning "Please edit .env file with your actual configuration values"
    else
        print_status ".env file already exists"
    fi

    # Create logs directory
    mkdir -p logs
    print_status "Created logs directory"

    # Create data directory for local development
    mkdir -p data
    print_status "Created data directory"
}

# Setup dashboard
setup_dashboard() {
    if [ "$SKIP_DASHBOARD" = true ]; then
        print_warning "Skipping dashboard setup (Node.js not available)"
        return
    fi

    print_status "Setting up React dashboard..."

    cd dashboard

    # Install dependencies
    npm install
    print_status "Installed dashboard dependencies"

    # Build dashboard
    npm run build
    print_status "Built dashboard"

    cd ..
}

# Setup Docker environment
setup_docker() {
    if [ "$SKIP_DOCKER" = true ]; then
        print_warning "Skipping Docker setup (Docker not available)"
        return
    fi

    print_status "Setting up Docker environment..."

    # Create Docker network
    docker network create ai-copilot-network 2>/dev/null || print_status "Docker network already exists"

    print_status "Docker environment ready"
}

# Setup Redis (local development)
setup_redis() {
    print_status "Setting up Redis for local development..."

    if command -v redis-server &> /dev/null; then
        print_status "Redis server found locally"
    elif [ "$SKIP_DOCKER" != true ]; then
        print_status "Starting Redis container..."
        docker run -d \
            --name ai-copilot-redis \
            --network ai-copilot-network \
            -p 6379:6379 \
            redis:alpine || print_status "Redis container already running"
    else
        print_warning "Redis not available. Install Redis or Docker for local development."
    fi
}

# Run initial tests
run_tests() {
    print_status "Running initial tests..."

    # Activate virtual environment
    source .venv/bin/activate

    # Run basic tests
    python -m pytest tests/ -x -v --tb=short || {
        print_warning "Some tests failed. This is normal for initial setup without Azure credentials."
    }

    print_status "Test run complete"
}

# Display setup summary
show_summary() {
    echo ""
    echo -e "${BLUE}Setup Summary${NC}"
    echo "============="
    echo -e "${GREEN}✓${NC} Python virtual environment created and activated"
    echo -e "${GREEN}✓${NC} Dependencies installed"
    echo -e "${GREEN}✓${NC} Configuration template created"

    if [ "$SKIP_DASHBOARD" != true ]; then
        echo -e "${GREEN}✓${NC} Dashboard built and ready"
    fi

    if [ "$SKIP_DOCKER" != true ]; then
        echo -e "${GREEN}✓${NC} Docker environment configured"
    fi

    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Edit .env file with your Azure credentials"
    echo "2. Start the development server: make dev"
    echo "3. Access the API at: http://localhost:8000"

    if [ "$SKIP_DASHBOARD" != true ]; then
        echo "4. Access the dashboard at: http://localhost:3000"
    fi

    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "- make dev          : Start development server"
    echo "- make test         : Run tests"
    echo "- make lint         : Run code linting"
    echo "- make docker-up    : Start with Docker"
    echo "- make clean        : Clean up environment"
    echo ""
}

# Main setup function
main() {
    print_status "Starting Azure AI IT Copilot setup..."

    check_os
    check_prerequisites
    setup_python_env
    setup_config
    setup_dashboard
    setup_docker
    setup_redis
    run_tests
    show_summary

    print_status "Setup completed successfully!"
    echo -e "${GREEN}You can now start developing with Azure AI IT Copilot!${NC}"
}

# Run main function
main "$@"