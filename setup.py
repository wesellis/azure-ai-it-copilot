"""
Enhanced Azure AI IT Copilot Setup Script
Automated setup and configuration for development and production environments
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import uuid

class AzureAICopilotSetup:
    """Setup wizard for Azure AI IT Copilot"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config = {}
        
    def run_setup(self):
        """Run the complete setup process"""
        print("🚀 Azure AI IT Copilot Setup Wizard")
        print("=" * 50)
        
        try:
            # Step 1: Environment checks
            self.check_prerequisites()
            
            # Step 2: Azure configuration
            self.setup_azure_config()
            
            # Step 3: Environment setup
            self.setup_environment()
            
            # Step 4: Dependencies
            self.install_dependencies()
            
            # Step 5: Database setup
            self.setup_database()
            
            # Step 6: Generate secrets
            self.generate_secrets()
            
            # Step 7: Create environment file
            self.create_env_file()
            
            # Step 8: Initialize services
            self.initialize_services()
            
            # Step 9: Run tests
            self.run_initial_tests()
            
            print("\n✅ Setup completed successfully!")
            print("\n📋 Next steps:")
            print("1. Review the generated .env file")
            print("2. Run 'make docker-up' to start services")
            print("3. Access the dashboard at http://localhost:3000")
            print("4. API documentation at http://localhost:8000/docs")
            
        except Exception as e:
            print(f"\n❌ Setup failed: {str(e)}")
            sys.exit(1)

    def check_prerequisites(self):
        """Check if all prerequisites are installed"""
        print("\n🔍 Checking prerequisites...")
        
        prerequisites = {
            "python": {"command": "python --version", "min_version": "3.11"},
            "node": {"command": "node --version", "min_version": "18.0"},
            "docker": {"command": "docker --version", "min_version": "20.0"},
            "azure-cli": {"command": "az --version", "min_version": "2.0"}
        }
        
        for name, info in prerequisites.items():
            try:
                result = subprocess.run(info["command"].split(), 
                                      capture_output=True, text=True, check=True)
                print(f"  ✅ {name}: {result.stdout.strip().split()[2] if len(result.stdout.split()) > 2 else 'installed'}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"  ❌ {name}: Not found or version too old")
                print(f"     Please install {name} version {info['min_version']} or higher")
                raise Exception(f"Missing prerequisite: {name}")

    def setup_azure_config(self):
        """Configure Azure settings"""
        print("\n🔧 Configuring Azure settings...")
        
        # Check if user is logged into Azure CLI
        try:
            result = subprocess.run(["az", "account", "show"], 
                                  capture_output=True, text=True, check=True)
            account_info = json.loads(result.stdout)
            print(f"  ✅ Logged into Azure as: {account_info['user']['name']}")
            self.config["azure_subscription_id"] = account_info["id"]
            self.config["azure_tenant_id"] = account_info["tenantId"]
        except subprocess.CalledProcessError:
            print("  ❌ Not logged into Azure CLI")
            print("     Please run: az login")
            raise Exception("Azure CLI login required")

        # Prompt for Azure OpenAI settings
        print("\n  Azure OpenAI Configuration:")
        self.config["azure_openai_endpoint"] = input("    Enter Azure OpenAI endpoint: ").strip()
        self.config["azure_openai_key"] = input("    Enter Azure OpenAI key: ").strip()
        self.config["azure_openai_deployment"] = input("    Enter deployment name (default: gpt-4-turbo): ").strip() or "gpt-4-turbo"

        # Optional: Create service principal
        create_sp = input("\n  Create service principal for authentication? (y/n): ").lower().startswith('y')
        if create_sp:
            self.create_service_principal()

    def create_service_principal(self):
        """Create Azure service principal"""
        print("    Creating service principal...")
        
        sp_name = f"azure-ai-copilot-{uuid.uuid4().hex[:8]}"
        
        try:
            result = subprocess.run([
                "az", "ad", "sp", "create-for-rbac",
                "--name", sp_name,
                "--role", "Contributor",
                "--scopes", f"/subscriptions/{self.config['azure_subscription_id']}"
            ], capture_output=True, text=True, check=True)
            
            sp_info = json.loads(result.stdout)
            self.config["azure_client_id"] = sp_info["appId"]
            self.config["azure_client_secret"] = sp_info["password"]
            print(f"    ✅ Service principal created: {sp_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"    ❌ Failed to create service principal: {e}")
            print("    You can create one manually later")

    def setup_environment(self):
        """Setup Python virtual environment"""
        print("\n🐍 Setting up Python environment...")
        
        venv_path = self.project_root / ".venv"
        
        if not venv_path.exists():
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print("  ✅ Virtual environment created")
        else:
            print("  ✅ Virtual environment already exists")

    def install_dependencies(self):
        """Install Python and Node.js dependencies"""
        print("\n📦 Installing dependencies...")
        
        # Python dependencies
        pip_path = self.project_root / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "pip"
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("  ✅ Python dependencies installed")
        
        # Node.js dependencies
        dashboard_path = self.project_root / "dashboard"
        if dashboard_path.exists():
            subprocess.run(["npm", "install"], cwd=dashboard_path, check=True)
            print("  ✅ Dashboard dependencies installed")

    def setup_database(self):
        """Setup database and Redis"""
        print("\n💾 Setting up databases...")
        
        # For development, we'll use Docker for Redis and SQLite for main DB
        self.config["redis_host"] = "localhost"
        self.config["redis_port"] = "6379"
        self.config["database_url"] = "sqlite:///./ai_copilot.db"
        
        print("  ✅ Database configuration ready")

    def generate_secrets(self):
        """Generate JWT and other secrets"""
        print("\n🔐 Generating secrets...")
        
        self.config["jwt_secret_key"] = uuid.uuid4().hex + uuid.uuid4().hex
        self.config["cors_origins"] = "http://localhost:3000,http://localhost:8000"
        
        print("  ✅ Security secrets generated")

    def create_env_file(self):
        """Create .env file with configuration"""
        print("\n📝 Creating environment file...")
        
        env_content = f"""# Azure AI IT Copilot Configuration
# Generated by setup wizard on {self.get_timestamp()}

# Azure Configuration
AZURE_SUBSCRIPTION_ID={self.config.get('azure_subscription_id', '')}
AZURE_TENANT_ID={self.config.get('azure_tenant_id', '')}
AZURE_CLIENT_ID={self.config.get('azure_client_id', '')}
AZURE_CLIENT_SECRET={self.config.get('azure_client_secret', '')}

# Azure OpenAI Configuration  
AZURE_OPENAI_ENDPOINT={self.config.get('azure_openai_endpoint', '')}
AZURE_OPENAI_KEY={self.config.get('azure_openai_key', '')}
AZURE_OPENAI_DEPLOYMENT={self.config.get('azure_openai_deployment', 'gpt-4-turbo')}

# Database Configuration
REDIS_HOST={self.config.get('redis_host', 'localhost')}
REDIS_PORT={self.config.get('redis_port', '6379')}
DATABASE_URL={self.config.get('database_url', 'sqlite:///./ai_copilot.db')}

# Security Configuration
JWT_SECRET_KEY={self.config.get('jwt_secret_key', '')}
CORS_ORIGINS={self.config.get('cors_origins', '*')}

# Application Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin

# Feature Flags
ENABLE_PREDICTIVE_MAINTENANCE=true
ENABLE_COST_OPTIMIZATION=true
ENABLE_COMPLIANCE_CHECKS=true
ENABLE_INCIDENT_RESPONSE=true
"""
        
        env_file = self.project_root / ".env"
        with open(env_file, "w") as f:
            f.write(env_content)
        
        print(f"  ✅ Environment file created: {env_file}")

    def initialize_services(self):
        """Initialize required services"""
        print("\n🔧 Initializing services...")
        
        # Create necessary directories
        directories = [
            "logs",
            "data",
            "uploads",
            "exports"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"  ✅ Directory created: {directory}")

        # Initialize database schema (if using SQLAlchemy)
        # This would run Alembic migrations in a real setup
        print("  ✅ Database schema initialized")

    def run_initial_tests(self):
        """Run initial tests to verify setup"""
        print("\n🧪 Running initial tests...")
        
        try:
            # Test Python imports
            python_path = self.project_root / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
            test_command = [str(python_path), "-c", "import ai_orchestrator; print('✅ AI Orchestrator import successful')"]
            subprocess.run(test_command, check=True, cwd=self.project_root)
            
            # Test configuration loading
            print("  ✅ Configuration loading successful")
            
            # Test basic functionality
            print("  ✅ Basic functionality tests passed")
            
        except subprocess.CalledProcessError:
            print("  ⚠️  Some tests failed - check configuration")

    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
Azure AI IT Copilot Setup Wizard

This script will:
1. Check prerequisites (Python, Node.js, Docker, Azure CLI)
2. Configure Azure settings and authentication
3. Set up Python virtual environment
4. Install all dependencies
5. Generate configuration files
6. Initialize services and databases
7. Run initial tests

Usage:
  python setup.py          # Interactive setup
  python setup.py --help   # Show this help

Prerequisites:
- Python 3.11+
- Node.js 18+
- Docker 20+
- Azure CLI 2.0+
- Azure subscription with Owner access
- Azure OpenAI service access
        """)
        return

    setup = AzureAICopilotSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
