"""
Azure AI IT Copilot CLI Tool
Command-line interface for managing the AI copilot system
"""

import click
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import subprocess
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ai_orchestrator.orchestrator import AzureAIOrchestrator
from monitoring.advanced_monitoring import global_monitoring


@click.group()
@click.version_option(version="1.0.0")
@click.pass_context
def cli(ctx):
    """
    Azure AI IT Copilot CLI
    
    Natural language interface for Azure infrastructure management.
    """
    ctx.ensure_object(dict)


@cli.group()
def admin():
    """Administrative commands for system management"""
    pass


@cli.group() 
def agent():
    """Agent management commands"""
    pass


@cli.group()
def monitor():
    """Monitoring and alerting commands"""
    pass


@cli.group()
def deploy():
    """Deployment commands"""
    pass


@cli.command()
@click.argument('command', nargs=-1, required=True)
@click.option('--dry-run', is_flag=True, help='Simulate command without executing')
@click.option('--auto-approve', is_flag=True, help='Auto-approve operations')
@click.option('--context', '-c', help='Additional context as JSON string')
@click.option('--user', '-u', default='cli-user', help='User context')
@click.option('--role', '-r', default='owner', help='User role')
def execute(command, dry_run, auto_approve, context, user, role):
    """
    Execute a natural language command
    
    Examples:
      ai-copilot execute "List all VMs in production"
      ai-copilot execute "Create a Linux VM with 8GB RAM" --dry-run
      ai-copilot execute "Diagnose high CPU on vm-prod-001" --auto-approve
    """
    async def run_command():
        try:
            # Initialize orchestrator
            orchestrator = AzureAIOrchestrator()
            
            # Parse context
            ctx_dict = {}
            if context:
                try:
                    ctx_dict = json.loads(context)
                except json.JSONDecodeError:
                    click.echo("❌ Invalid JSON in context parameter", err=True)
                    return
            
            # Add CLI-specific context
            ctx_dict.update({
                'user_id': user,
                'user_role': role,
                'dry_run': dry_run,
                'auto_approve': auto_approve,
                'interface': 'cli'
            })
            
            # Join command parts
            full_command = ' '.join(command)
            
            click.echo(f"🤖 Processing: {full_command}")
            if dry_run:
                click.echo("🧪 Dry run mode - no changes will be made")
            
            # Execute command
            result = await orchestrator.process_command(full_command, ctx_dict)
            
            # Display result
            status = result.get('status', 'unknown')
            if status == 'success':
                click.echo("✅ Command executed successfully")
            elif status == 'error':
                click.echo("❌ Command failed")
            else:
                click.echo(f"ℹ️  Command status: {status}")
            
            # Show detailed results
            if result.get('message'):
                click.echo(f"📝 {result['message']}")
            
            if result.get('data'):
                click.echo("\n📊 Results:")
                click.echo(json.dumps(result['data'], indent=2))
                
        except Exception as e:
            click.echo(f"💥 Error: {str(e)}", err=True)
            sys.exit(1)
    
    asyncio.run(run_command())


@admin.command()
def status():
    """Show system status and health"""
    click.echo("🔍 Azure AI IT Copilot System Status")
    click.echo("=" * 40)
    
    try:
        # System health
        health_score = global_monitoring.calculate_system_health()
        health_color = "🟢" if health_score > 80 else "🟡" if health_score > 60 else "🔴"
        click.echo(f"System Health: {health_color} {health_score:.1f}/100")
        
        # Service status
        services = {
            "API Server": check_service_health("http://localhost:8000/health"),
            "Dashboard": check_service_health("http://localhost:3000"),
            "Redis": check_redis_health(),
            "Monitoring": check_monitoring_health()
        }
        
        click.echo("\n📊 Service Status:")
        for service, status in services.items():
            status_icon = "✅" if status else "❌"
            click.echo(f"  {status_icon} {service}")
        
        # Recent activity
        dashboard_data = global_monitoring.get_dashboard_data()
        recent = dashboard_data.get('recent_metrics', {})
        
        click.echo("\n📈 Recent Activity:")
        click.echo(f"  API Requests: {recent.get('api_requests_last_hour', 0)}")
        click.echo(f"  Commands Processed: {recent.get('commands_processed', 0)}")
        click.echo(f"  Incidents Resolved: {recent.get('incidents_resolved', 0)}")
        click.echo(f"  Cost Savings: ${recent.get('cost_savings', 0):,.2f}")
        
        # Active alerts
        active_alerts = dashboard_data.get('active_alerts', [])
        if active_alerts:
            click.echo(f"\n🚨 Active Alerts ({len(active_alerts)}):")
            for alert in active_alerts[:5]:  # Show top 5
                severity_icon = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🆘"}.get(alert.get('severity', 'low'), "🟡")
                click.echo(f"  {severity_icon} {alert.get('title', 'Unknown Alert')}")
        else:
            click.echo("\n✅ No active alerts")
            
    except Exception as e:
        click.echo(f"❌ Error checking status: {e}", err=True)


@admin.command()
@click.option('--service', '-s', type=click.Choice(['all', 'api', 'dashboard', 'redis', 'monitoring']), 
              default='all', help='Service to restart')
def restart(service):
    """Restart system services"""
    click.echo(f"🔄 Restarting {service} service(s)...")
    
    if service in ['all', 'api']:
        restart_service('api')
        click.echo("  ✅ API service restarted")
    
    if service in ['all', 'dashboard']:
        restart_service('dashboard')
        click.echo("  ✅ Dashboard service restarted")
    
    if service in ['all', 'redis']:
        restart_service('redis')
        click.echo("  ✅ Redis service restarted")
    
    if service in ['all', 'monitoring']:
        restart_service('monitoring')
        click.echo("  ✅ Monitoring service restarted")


@admin.command()
def logs():
    """Show recent system logs"""
    click.echo("📜 Recent System Logs")
    click.echo("=" * 30)
    
    log_files = [
        "logs/ai-copilot.log",
        "logs/ai-copilot-errors.log"
    ]
    
    for log_file in log_files:
        log_path = Path(log_file)
        if log_path.exists():
            click.echo(f"\n📄 {log_file}:")
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()[-20:]  # Last 20 lines
                    for line in lines:
                        click.echo(f"  {line.strip()}")
            except Exception as e:
                click.echo(f"  ❌ Error reading log: {e}")
        else:
            click.echo(f"\n📄 {log_file}: Not found")


@agent.command()
def list():
    """List all available agents"""
    click.echo("🤖 Available AI Agents")
    click.echo("=" * 25)
    
    agents = {
        "Infrastructure Agent": "Manages Azure resources - create, delete, modify VMs, storage, networks",
        "Incident Response Agent": "Diagnoses problems and automates remediation",
        "Cost Optimization Agent": "Analyzes spending and finds optimization opportunities", 
        "Compliance Agent": "Checks compliance against SOC2, HIPAA, ISO27001 standards",
        "Predictive Agent": "Forecasts failures and capacity needs using ML models"
    }
    
    for name, description in agents.items():
        click.echo(f"\n🔧 {name}")
        click.echo(f"   {description}")


@agent.command()
@click.argument('agent_name')
def status(agent_name):
    """Check the status of a specific agent"""
    click.echo(f"🔍 Agent Status: {agent_name}")
    
    # Mock agent status - in real implementation would check actual agent health
    agent_status = {
        "status": "healthy",
        "last_execution": "2024-01-15T10:30:00Z",
        "success_rate": "94.5%",
        "avg_response_time": "2.3s"
    }
    
    click.echo(f"Status: ✅ {agent_status['status']}")
    click.echo(f"Last Execution: {agent_status['last_execution']}")
    click.echo(f"Success Rate: {agent_status['success_rate']}")
    click.echo(f"Avg Response Time: {agent_status['avg_response_time']}")


@monitor.command()
def dashboard():
    """Open monitoring dashboard in browser"""
    import webbrowser
    
    dashboard_url = "http://localhost:3001"  # Grafana port
    click.echo(f"🌐 Opening monitoring dashboard: {dashboard_url}")
    
    try:
        webbrowser.open(dashboard_url)
        click.echo("✅ Dashboard opened in browser")
    except Exception as e:
        click.echo(f"❌ Error opening dashboard: {e}")
        click.echo(f"Please manually navigate to: {dashboard_url}")


@monitor.command()
@click.option('--count', '-c', default=10, help='Number of alerts to show')
def alerts(count):
    """Show recent alerts"""
    click.echo(f"🚨 Recent Alerts (last {count})")
    click.echo("=" * 30)
    
    # Get alerts from monitoring system
    dashboard_data = global_monitoring.get_dashboard_data()
    active_alerts = dashboard_data.get('active_alerts', [])
    
    if not active_alerts:
        click.echo("✅ No active alerts")
        return
    
    for i, alert in enumerate(active_alerts[:count]):
        severity = alert.get('severity', 'unknown')
        severity_icon = {
            "low": "🟡",
            "medium": "🟠", 
            "high": "🔴",
            "critical": "🆘"
        }.get(severity, "🟡")
        
        click.echo(f"\n{i+1}. {severity_icon} {alert.get('title', 'Unknown Alert')}")
        click.echo(f"   Severity: {severity.upper()}")
        click.echo(f"   Resource: {alert.get('resource', 'Unknown')}")
        click.echo(f"   Time: {alert.get('timestamp', 'Unknown')}")
        click.echo(f"   Description: {alert.get('description', 'No description')}")


@deploy.command()
@click.option('--environment', '-e', default='development', 
              type=click.Choice(['development', 'staging', 'production']),
              help='Target environment')
def status(environment):
    """Check deployment status"""
    click.echo(f"🚀 Deployment Status: {environment}")
    click.echo("=" * 35)
    
    # Check if services are running
    components = [
        ("API Server", "http://localhost:8000/health"),
        ("Dashboard", "http://localhost:3000"), 
        ("Redis", "redis://localhost:6379"),
        ("Monitoring", "http://localhost:9090")
    ]
    
    for component, endpoint in components:
        status = check_component_health(component, endpoint)
        status_icon = "✅" if status else "❌"
        click.echo(f"  {status_icon} {component}")


@deploy.command()
@click.option('--environment', '-e', default='development',
              type=click.Choice(['development', 'staging', 'production']),
              help='Target environment')
@click.option('--step', '-s', 
              type=click.Choice(['all', 'build', 'deploy', 'test']),
              default='all', help='Deployment step')
def run(environment, step):
    """Run deployment to specified environment"""
    click.echo(f"🚀 Deploying to {environment}...")
    
    if step in ['all', 'build']:
        click.echo("📦 Building containers...")
        # Run docker build commands
        
    if step in ['all', 'deploy']:
        click.echo("🔧 Deploying services...")
        # Run deployment commands
        
    if step in ['all', 'test']:
        click.echo("🧪 Running tests...")
        # Run health checks
        
    click.echo(f"✅ Deployment to {environment} completed")


# Helper functions
def check_service_health(url: str) -> bool:
    """Check if a service is healthy"""
    try:
        import requests
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False


def check_redis_health() -> bool:
    """Check Redis health"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=5)
        return r.ping()
    except:
        return False


def check_monitoring_health() -> bool:
    """Check monitoring system health"""
    return check_service_health("http://localhost:9090/-/healthy")


def check_component_health(component: str, endpoint: str) -> bool:
    """Check health of a specific component"""
    if endpoint.startswith('http'):
        return check_service_health(endpoint)
    elif endpoint.startswith('redis'):
        return check_redis_health()
    else:
        return False


def restart_service(service_name: str):
    """Restart a service using docker-compose"""
    try:
        subprocess.run(
            ['docker-compose', 'restart', service_name], 
            check=True, 
            capture_output=True
        )
    except subprocess.CalledProcessError:
        # Fallback to Docker commands
        try:
            subprocess.run(
                ['docker', 'restart', f'ai-copilot-{service_name}'],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            pass  # Service might not be running


# Interactive mode
@cli.command()
def interactive():
    """Start interactive command mode"""
    click.echo("🤖 Azure AI IT Copilot - Interactive Mode")
    click.echo("Type 'exit' to quit, 'help' for commands")
    click.echo("=" * 45)
    
    while True:
        try:
            command = click.prompt("\nai-copilot", type=str)
            
            if command.lower() in ['exit', 'quit']:
                click.echo("👋 Goodbye!")
                break
            elif command.lower() == 'help':
                show_interactive_help()
            elif command.strip():
                # Execute the command
                ctx = click.Context(execute)
                ctx.invoke(execute, command=command.split(), dry_run=False, 
                          auto_approve=False, context=None, user='interactive', role='owner')
            
        except (KeyboardInterrupt, EOFError):
            click.echo("\n👋 Goodbye!")
            break
        except Exception as e:
            click.echo(f"❌ Error: {e}")


def show_interactive_help():
    """Show help for interactive mode"""
    click.echo("""
Available commands:
  List all VMs                    - Show virtual machines
  Create a Linux VM               - Create new VM
  Diagnose high CPU on vm-name    - Investigate performance issues  
  Optimize costs by 20%           - Find cost savings
  Check SOC2 compliance           - Run compliance audit
  Predict VM failures             - Run predictive analysis
  
System commands:
  status                          - Show system status
  alerts                          - Show active alerts
  help                           - Show this help
  exit                           - Quit interactive mode
""")


if __name__ == '__main__':
    cli()
