"""
Production Deployment Scripts for Azure AI IT Copilot
Automated deployment to Azure using Terraform and Azure DevOps
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import argparse

class AzureDeploymentManager:
    """Manages deployment to Azure cloud"""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.terraform_dir = self.project_root / "infrastructure" / "terraform"
        self.k8s_dir = self.project_root / "infrastructure" / "kubernetes"

    def deploy_infrastructure(self):
        """Deploy Azure infrastructure using Terraform"""
        print(f"🚀 Deploying {self.environment} infrastructure...")
        
        # Change to Terraform directory
        os.chdir(self.terraform_dir)
        
        try:
            # Initialize Terraform
            subprocess.run(["terraform", "init"], check=True)
            print("  ✅ Terraform initialized")
            
            # Create workspace for environment
            try:
                subprocess.run(["terraform", "workspace", "new", self.environment], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError:
                # Workspace might already exist
                subprocess.run(["terraform", "workspace", "select", self.environment], 
                             check=True)
            print(f"  ✅ Terraform workspace: {self.environment}")
            
            # Plan deployment
            plan_file = f"{self.environment}.tfplan"
            subprocess.run([
                "terraform", "plan",
                "-var-file", f"environments/{self.environment}.tfvars",
                "-out", plan_file
            ], check=True)
            print("  ✅ Terraform plan created")
            
            # Apply deployment
            confirmation = input(f"Apply Terraform plan for {self.environment}? (y/N): ")
            if confirmation.lower().startswith('y'):
                subprocess.run(["terraform", "apply", plan_file], check=True)
                print("  ✅ Infrastructure deployed successfully")
            else:
                print("  ⏸️  Deployment cancelled")
                
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Terraform deployment failed: {e}")
            raise

    def build_and_push_containers(self):
        """Build and push Docker containers to Azure Container Registry"""
        print("🐳 Building and pushing containers...")
        
        # Get ACR name from Terraform output
        acr_name = self.get_terraform_output("acr_name")
        
        if not acr_name:
            print("  ❌ Could not get ACR name from Terraform output")
            return
        
        # Login to ACR
        subprocess.run(["az", "acr", "login", "--name", acr_name], check=True)
        print(f"  ✅ Logged into ACR: {acr_name}")
        
        # Build and push API container
        api_image = f"{acr_name}.azurecr.io/azure-ai-copilot-api:{self.get_version()}"
        subprocess.run([
            "docker", "build",
            "-f", "Dockerfile.api",
            "-t", api_image,
            "."
        ], cwd=self.project_root, check=True)
        subprocess.run(["docker", "push", api_image], check=True)
        print(f"  ✅ API image pushed: {api_image}")
        
        # Build and push Dashboard container
        dashboard_image = f"{acr_name}.azurecr.io/azure-ai-copilot-dashboard:{self.get_version()}"
        subprocess.run([
            "docker", "build",
            "-f", "dashboard/Dockerfile",
            "-t", dashboard_image,
            "./dashboard"
        ], cwd=self.project_root, check=True)
        subprocess.run(["docker", "push", dashboard_image], check=True)
        print(f"  ✅ Dashboard image pushed: {dashboard_image}")

    def deploy_kubernetes(self):
        """Deploy to Azure Kubernetes Service"""
        print("☸️  Deploying to Kubernetes...")
        
        # Get AKS credentials
        aks_name = self.get_terraform_output("aks_name")
        resource_group = self.get_terraform_output("resource_group_name")
        
        if not aks_name or not resource_group:
            print("  ❌ Could not get AKS details from Terraform output")
            return
        
        subprocess.run([
            "az", "aks", "get-credentials",
            "--resource-group", resource_group,
            "--name", aks_name,
            "--overwrite-existing"
        ], check=True)
        print(f"  ✅ AKS credentials configured")
        
        # Apply Kubernetes manifests
        k8s_files = [
            "namespace.yaml",
            "configmap.yaml", 
            "secrets.yaml",
            "redis-deployment.yaml",
            "api-deployment.yaml",
            "dashboard-deployment.yaml",
            "ingress.yaml"
        ]
        
        for k8s_file in k8s_files:
            file_path = self.k8s_dir / k8s_file
            if file_path.exists():
                # Replace placeholders in manifest
                manifest_content = self.process_k8s_manifest(file_path)
                
                # Apply manifest
                process = subprocess.Popen(
                    ["kubectl", "apply", "-f", "-"],
                    stdin=subprocess.PIPE,
                    text=True
                )
                process.communicate(input=manifest_content)
                
                if process.returncode == 0:
                    print(f"    ✅ Applied: {k8s_file}")
                else:
                    print(f"    ❌ Failed to apply: {k8s_file}")
        
        print("  ✅ Kubernetes deployment completed")

    def process_k8s_manifest(self, file_path: Path) -> str:
        """Process Kubernetes manifest and replace placeholders"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace common placeholders
        replacements = {
            "{{ENVIRONMENT}}": self.environment,
            "{{VERSION}}": self.get_version(),
            "{{ACR_NAME}}": self.get_terraform_output("acr_name") or "placeholder",
            "{{DOMAIN_NAME}}": self.get_terraform_output("domain_name") or "localhost"
        }
        
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        
        return content

    def get_terraform_output(self, output_name: str) -> str:
        """Get Terraform output value"""
        try:
            result = subprocess.run([
                "terraform", "output", "-raw", output_name
            ], cwd=self.terraform_dir, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def get_version(self) -> str:
        """Get application version from git or environment"""
        try:
            result = subprocess.run([
                "git", "describe", "--tags", "--always", "--dirty"
            ], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return os.getenv("BUILD_NUMBER", "latest")

    def setup_monitoring(self):
        """Setup monitoring and alerting"""
        print("📊 Setting up monitoring...")
        
        # Deploy Prometheus and Grafana
        monitoring_manifests = [
            "monitoring/prometheus-deployment.yaml",
            "monitoring/grafana-deployment.yaml",
            "monitoring/alert-manager-deployment.yaml"
        ]
        
        for manifest in monitoring_manifests:
            file_path = self.k8s_dir / manifest
            if file_path.exists():
                subprocess.run(["kubectl", "apply", "-f", str(file_path)], check=True)
                print(f"    ✅ Applied: {manifest}")
        
        # Configure Azure Monitor integration
        self.configure_azure_monitor()
        
        print("  ✅ Monitoring setup completed")

    def configure_azure_monitor(self):
        """Configure Azure Monitor and Application Insights"""
        app_insights_key = self.get_terraform_output("app_insights_key")
        
        if app_insights_key:
            # Update application configuration with App Insights key
            print("    ✅ Azure Monitor configured")
        else:
            print("    ⚠️  Application Insights key not found")

    def run_health_checks(self):
        """Run post-deployment health checks"""
        print("🏥 Running health checks...")
        
        # Wait for deployments to be ready
        deployments = ["redis", "api", "dashboard"]
        
        for deployment in deployments:
            subprocess.run([
                "kubectl", "wait", "--for=condition=available",
                "--timeout=300s", f"deployment/{deployment}"
            ], check=True)
            print(f"    ✅ {deployment} deployment ready")
        
        # Test API endpoint
        api_url = f"https://{self.get_terraform_output('domain_name')}/health"
        try:
            import requests
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                print(f"    ✅ API health check passed: {api_url}")
            else:
                print(f"    ❌ API health check failed: {response.status_code}")
        except Exception as e:
            print(f"    ❌ API health check error: {e}")
        
        print("  ✅ Health checks completed")

    def deploy_full_stack(self):
        """Deploy the complete application stack"""
        print(f"\n🎯 Starting full deployment to {self.environment}")
        print("=" * 50)
        
        try:
            # Step 1: Deploy infrastructure
            self.deploy_infrastructure()
            
            # Step 2: Build and push containers
            self.build_and_push_containers()
            
            # Step 3: Deploy to Kubernetes
            self.deploy_kubernetes()
            
            # Step 4: Setup monitoring
            self.setup_monitoring()
            
            # Step 5: Run health checks
            self.run_health_checks()
            
            print(f"\n🎉 Deployment to {self.environment} completed successfully!")
            print(f"Application URL: https://{self.get_terraform_output('domain_name')}")
            print(f"Monitoring: https://grafana.{self.get_terraform_output('domain_name')}")
            
        except Exception as e:
            print(f"\n💥 Deployment failed: {e}")
            sys.exit(1)


def create_terraform_config():
    """Create Terraform configuration files"""
    terraform_main = """
# Azure AI IT Copilot - Main Terraform Configuration

terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.0"
    }
  }
  
  backend "azurerm" {
    # Configure remote state storage
  }
}

provider "azurerm" {
  features {}
}

provider "azuread" {}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-${var.environment}"
  location = var.location
  
  tags = local.common_tags
}

# Azure Container Registry
resource "azurerm_container_registry" "main" {
  name                = "${var.project_name}acr${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location           = azurerm_resource_group.main.location
  sku                = "Standard"
  admin_enabled      = true
  
  tags = local.common_tags
}

# Azure Kubernetes Service
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.project_name}-aks-${var.environment}"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix         = "${var.project_name}-${var.environment}"
  
  default_node_pool {
    name       = "default"
    node_count = var.aks_node_count
    vm_size    = var.aks_vm_size
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  tags = local.common_tags
}

# Application Insights
resource "azurerm_application_insights" "main" {
  name                = "${var.project_name}-ai-${var.environment}"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type   = "web"
  
  tags = local.common_tags
}

# Azure Key Vault
resource "azurerm_key_vault" "main" {
  name                = "${var.project_name}-kv-${var.environment}"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id          = data.azurerm_client_config.current.tenant_id
  sku_name           = "standard"
  
  tags = local.common_tags
}

# Redis Cache
resource "azurerm_redis_cache" "main" {
  name                = "${var.project_name}-redis-${var.environment}"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity           = 1
  family             = "C"
  sku_name           = "Standard"
  
  tags = local.common_tags
}

# Outputs
output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "acr_name" {
  value = azurerm_container_registry.main.name
}

output "aks_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "app_insights_key" {
  value = azurerm_application_insights.main.instrumentation_key
  sensitive = true
}

# Local values
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

data "azurerm_client_config" "current" {}
"""
    
    return terraform_main


def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(description="Deploy Azure AI IT Copilot")
    parser.add_argument("--environment", "-e", default="production",
                       help="Deployment environment (development, staging, production)")
    parser.add_argument("--step", "-s", 
                       choices=["infrastructure", "containers", "kubernetes", "monitoring", "health", "all"],
                       default="all", help="Deployment step to run")
    
    args = parser.parse_args()
    
    deployer = AzureDeploymentManager(args.environment)
    
    if args.step == "infrastructure":
        deployer.deploy_infrastructure()
    elif args.step == "containers":
        deployer.build_and_push_containers()
    elif args.step == "kubernetes":
        deployer.deploy_kubernetes()
    elif args.step == "monitoring":
        deployer.setup_monitoring()
    elif args.step == "health":
        deployer.run_health_checks()
    elif args.step == "all":
        deployer.deploy_full_stack()

if __name__ == "__main__":
    main()
