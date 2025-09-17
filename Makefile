.PHONY: help install dev test lint format clean docker-build docker-up docker-down deploy

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := azure-ai-it-copilot
DOCKER_REGISTRY := registry.hub.docker.com
VERSION := $(shell git describe --tags --always --dirty)

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo '$(BLUE)Azure AI IT Copilot - Makefile Commands$(NC)'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PYTHON) setup.py install

dev: ## Install development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(PIP) install -e ".[dev,ml]"
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=ai_orchestrator --cov=api --cov=automation_engine --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	pytest tests/unit -v

test-integration: ## Run integration tests
	pytest tests/integration -v

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	ruff check .
	mypy ai_orchestrator api automation_engine

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black ai_orchestrator api automation_engine tests
	ruff check --fix .

clean: ## Clean build artifacts and cache
	@echo "$(RED)Cleaning up...$(NC)"
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build dist .coverage htmlcov .pytest_cache
	rm -rf dashboard/node_modules dashboard/build

# Docker commands
docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker build -f Dockerfile.api -t $(PROJECT_NAME)-api:$(VERSION) .
	docker build -f dashboard/Dockerfile -t $(PROJECT_NAME)-dashboard:$(VERSION) ./dashboard

docker-up: ## Start all services with Docker Compose
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Dashboard: http://localhost:3000"
	@echo "Redis: localhost:6379"

docker-down: ## Stop all Docker services
	@echo "$(RED)Stopping services...$(NC)"
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

# Database commands
db-init: ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	alembic upgrade head

db-migrate: ## Create new migration
	@echo "$(BLUE)Creating migration...$(NC)"
	alembic revision --autogenerate -m "$(msg)"

db-upgrade: ## Apply database migrations
	alembic upgrade head

db-downgrade: ## Rollback last migration
	alembic downgrade -1

# API commands
api-run: ## Run API server locally
	@echo "$(BLUE)Starting API server...$(NC)"
	uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

api-docs: ## Generate API documentation
	@echo "$(BLUE)Generating API documentation...$(NC)"
	$(PYTHON) scripts/generate_api_docs.py

# Dashboard commands
dashboard-install: ## Install dashboard dependencies
	@echo "$(BLUE)Installing dashboard dependencies...$(NC)"
	cd dashboard && npm install

dashboard-dev: ## Run dashboard in development mode
	@echo "$(BLUE)Starting dashboard in dev mode...$(NC)"
	cd dashboard && npm run dev

dashboard-build: ## Build dashboard for production
	@echo "$(BLUE)Building dashboard...$(NC)"
	cd dashboard && npm run build

# Deployment commands
deploy-terraform: ## Deploy infrastructure with Terraform
	@echo "$(BLUE)Deploying infrastructure...$(NC)"
	cd infrastructure/terraform && terraform init && terraform apply

deploy-k8s: ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	kubectl apply -f infrastructure/kubernetes/

deploy-azure: ## Deploy to Azure
	@echo "$(BLUE)Deploying to Azure...$(NC)"
	./scripts/deploy_azure.sh

# Security commands
security-scan: ## Run security vulnerability scan
	@echo "$(BLUE)Running security scan...$(NC)"
	pip-audit
	bandit -r ai_orchestrator api automation_engine
	safety check

secrets-check: ## Check for exposed secrets
	@echo "$(BLUE)Checking for secrets...$(NC)"
	trufflesec3 --regex --entropy=False .

# ML Model commands
train-models: ## Train ML models
	@echo "$(BLUE)Training ML models...$(NC)"
	$(PYTHON) ml_models/train_all.py

evaluate-models: ## Evaluate ML model performance
	@echo "$(BLUE)Evaluating models...$(NC)"
	$(PYTHON) ml_models/evaluate.py

# Monitoring commands
monitor-start: ## Start monitoring stack (Prometheus + Grafana)
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	docker-compose -f docker-compose.monitoring.yml up -d

monitor-stop: ## Stop monitoring stack
	docker-compose -f docker-compose.monitoring.yml down

# Utility commands
version: ## Show version
	@echo "$(GREEN)Version: $(VERSION)$(NC)"

env-setup: ## Setup environment variables
	@echo "$(BLUE)Setting up environment...$(NC)"
	cp .env.example .env
	@echo "$(GREEN)Please edit .env with your configuration$(NC)"

logs: ## Show application logs
	tail -f logs/*.log

shell: ## Open Python shell with project context
	$(PYTHON) -i -c "from ai_orchestrator import *; from api import *"

# CI/CD commands
ci-test: lint test security-scan ## Run all CI checks

release: ## Create a new release
	@echo "$(BLUE)Creating release...$(NC)"
	$(PYTHON) scripts/create_release.py --version $(version)

# Performance commands
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTHON) tests/benchmarks/run_benchmarks.py

profile: ## Profile application performance
	$(PYTHON) -m cProfile -o profile.stats api/server.py
	$(PYTHON) -m pstats profile.stats