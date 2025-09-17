"""
Azure AI IT Copilot - Core AI Orchestrator
Processes natural language commands and coordinates agent execution
"""

import os
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import Tool

from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.monitor.query import LogsQueryClient

import redis
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of intents the system can handle"""
    RESOURCE_CREATE = "resource_create"
    RESOURCE_DELETE = "resource_delete"
    RESOURCE_QUERY = "resource_query"
    INCIDENT_DIAGNOSIS = "incident_diagnosis"
    COST_OPTIMIZATION = "cost_optimization"
    COMPLIANCE_CHECK = "compliance_check"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    UNKNOWN = "unknown"


class AzureAIOrchestrator:
    """Main orchestrator for processing natural language commands"""

    def __init__(self):
        """Initialize the orchestrator with Azure services"""
        self.setup_azure_clients()
        self.setup_ai_model()
        self.setup_memory()
        self.setup_redis()
        self.agents = self.load_agents()

    def setup_azure_clients(self):
        """Initialize Azure SDK clients"""
        credential = DefaultAzureCredential()
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")

        self.resource_client = ResourceManagementClient(credential, self.subscription_id)
        self.compute_client = ComputeManagementClient(credential, self.subscription_id)
        self.network_client = NetworkManagementClient(credential, self.subscription_id)
        self.storage_client = StorageManagementClient(credential, self.subscription_id)

        # Azure Monitor clients for incident response
        self.logs_client = LogsQueryClient(credential)
        self.metrics_client = MetricsQueryClient(credential)

    def setup_ai_model(self):
        """Initialize Azure OpenAI model"""
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4-turbo"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01",
            temperature=0.3,
            max_tokens=2000
        )

    def setup_memory(self):
        """Initialize conversation memory"""
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=2000,
            return_messages=True
        )

    def setup_redis(self):
        """Initialize Redis connection for caching"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
        except Exception as e:
            logger.error(f"Unexpected error setting up Redis: {e}")
            self.redis_client = None

    def load_agents(self) -> Dict[str, Any]:
        """Load specialized agents for different tasks"""
        from ai_orchestrator.agents.infrastructure_agent import InfrastructureAgent
        from ai_orchestrator.agents.incident_agent import IncidentAgent
        from ai_orchestrator.agents.cost_agent import CostOptimizationAgent
        from ai_orchestrator.agents.compliance_agent import ComplianceAgent
        from ai_orchestrator.agents.predictive_agent import PredictiveAgent

        return {
            IntentType.RESOURCE_CREATE: InfrastructureAgent(self),
            IntentType.RESOURCE_DELETE: InfrastructureAgent(self),
            IntentType.RESOURCE_QUERY: InfrastructureAgent(self),
            IntentType.INCIDENT_DIAGNOSIS: IncidentAgent(self),
            IntentType.COST_OPTIMIZATION: CostOptimizationAgent(self),
            IntentType.COMPLIANCE_CHECK: ComplianceAgent(self),
            IntentType.PREDICTIVE_ANALYSIS: PredictiveAgent(self)
        }

    async def process_command(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language command

        Args:
            command: Natural language command from user
            context: Optional context information

        Returns:
            Execution result with status and details
        """
        if not command or not command.strip():
            return {
                "status": "error",
                "message": "Command cannot be empty"
            }

        try:
            # Log command
            logger.info(f"Processing command: {command}")

            # Parse intent with timeout
            try:
                intent = await asyncio.wait_for(
                    self.classify_intent(command),
                    timeout=30.0
                )
                logger.info(f"Classified intent: {intent}")
            except asyncio.TimeoutError:
                logger.error("Intent classification timed out")
                return {
                    "status": "error",
                    "message": "Command processing timed out"
                }

            # Validate permissions
            if not await self.validate_permissions(intent, context):
                return {
                    "status": "error",
                    "message": "Insufficient permissions for this operation",
                    "required_role": self._get_required_role(intent)
                }

            # Select appropriate agent
            agent = self.agents.get(intent)
            if not agent:
                logger.error(f"No agent found for intent: {intent}")
                return {
                    "status": "error",
                    "message": f"No handler available for operation type: {intent.value}"
                }

            # Generate execution plan with timeout
            try:
                plan = await asyncio.wait_for(
                    agent.create_plan(command, context),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                logger.error("Plan generation timed out")
                return {
                    "status": "error",
                    "message": "Plan generation timed out"
                }
            except Exception as e:
                logger.error(f"Plan generation failed: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to create execution plan: {str(e)}"
                }

            # Request approval if needed
            if plan.get("requires_approval", False):
                try:
                    approval = await self.request_approval(plan)
                    if not approval:
                        return {
                            "status": "cancelled",
                            "message": "Operation cancelled - approval denied"
                        }
                except Exception as e:
                    logger.error(f"Approval process failed: {str(e)}")
                    return {
                        "status": "error",
                        "message": "Approval process failed"
                    }

            # Execute plan with timeout
            try:
                result = await asyncio.wait_for(
                    agent.execute(plan),
                    timeout=300.0  # 5 minutes
                )
            except asyncio.TimeoutError:
                logger.error("Plan execution timed out")
                return {
                    "status": "error",
                    "message": "Operation timed out"
                }
            except Exception as e:
                logger.error(f"Plan execution failed: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Execution failed: {str(e)}"
                }

            # Store in memory safely
            try:
                self.memory.save_context(
                    {"input": command},
                    {"output": json.dumps(result, default=str)}
                )
            except Exception as e:
                logger.warning(f"Failed to save to memory: {e}")

            # Cache result if Redis is available
            if self.redis_client:
                try:
                    cache_data = {
                        "command": command,
                        "intent": intent.value,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.redis_client.setex(
                        f"command:{datetime.now().isoformat()}",
                        3600,
                        json.dumps(cache_data, default=str)
                    )
                except redis.ConnectionError:
                    logger.warning("Failed to cache result: Redis connection lost")
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")

            return result

        except Exception as e:
            logger.error(f"Unexpected error processing command: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": "An unexpected error occurred while processing your command",
                "error_type": type(e).__name__
            }

    async def classify_intent(self, command: str) -> IntentType:
        """
        Classify the intent of a natural language command

        Args:
            command: Natural language command

        Returns:
            IntentType enum value
        """
        prompt = f"""
        Classify the following command into one of these categories:
        - resource_create: Creating new Azure resources
        - resource_delete: Deleting existing resources
        - resource_query: Querying or listing resources
        - incident_diagnosis: Diagnosing problems or incidents
        - cost_optimization: Optimizing costs or finding savings
        - compliance_check: Checking compliance or security
        - predictive_analysis: Predictive maintenance or forecasting

        Command: {command}

        Respond with only the category name.
        """

        response = await self.llm.ainvoke(prompt)
        intent_str = response.content.strip().lower()

        # Map to enum
        for intent_type in IntentType:
            if intent_type.value == intent_str:
                return intent_type

        return IntentType.UNKNOWN

    async def validate_permissions(
        self,
        intent: IntentType,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Validate user permissions for the intended operation

        Args:
            intent: Type of operation
            context: User context including roles

        Returns:
            True if permitted, False otherwise
        """
        if not context:
            return True  # Default allow for demo

        user_role = context.get("user_role", "reader")

        # Define permission matrix
        permissions = {
            "reader": [IntentType.RESOURCE_QUERY, IntentType.PREDICTIVE_ANALYSIS],
            "contributor": [
                IntentType.RESOURCE_QUERY,
                IntentType.INCIDENT_DIAGNOSIS,
                IntentType.COST_OPTIMIZATION,
                IntentType.COMPLIANCE_CHECK,
                IntentType.PREDICTIVE_ANALYSIS
            ],
            "owner": list(IntentType)  # All permissions
        }

        allowed = permissions.get(user_role, [])
        return intent in allowed

    def _get_required_role(self, intent: IntentType) -> str:
        """Get the minimum required role for an intent"""
        role_requirements = {
            IntentType.RESOURCE_CREATE: "contributor",
            IntentType.RESOURCE_DELETE: "owner",
            IntentType.RESOURCE_MODIFY: "contributor",
            IntentType.RESOURCE_QUERY: "reader",
            IntentType.INCIDENT_DIAGNOSIS: "contributor",
            IntentType.COST_OPTIMIZATION: "contributor",
            IntentType.COMPLIANCE_CHECK: "contributor",
            IntentType.PREDICTIVE_ANALYSIS: "reader"
        }
        return role_requirements.get(intent, "contributor")

    async def request_approval(self, plan: Dict[str, Any]) -> bool:
        """
        Request user approval for execution plan

        Args:
            plan: Execution plan requiring approval

        Returns:
            True if approved, False otherwise
        """
        # In production, this would send notification and wait for approval
        # For demo, auto-approve non-destructive operations

        if plan.get("operation_type") == "delete":
            logger.warning("Delete operation requires manual approval")
            return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        redis_connected = False
        if self.redis_client:
            try:
                redis_connected = self.redis_client.ping()
            except (redis.ConnectionError, redis.TimeoutError):
                redis_connected = False

        return {
            "status": "healthy",
            "agents_loaded": len(self.agents),
            "memory_size": len(self.memory.buffer),
            "redis_connected": redis_connected
        }


class BaseAgent:
    """Base class for specialized agents"""

    def __init__(self, orchestrator: AzureAIOrchestrator):
        self.orchestrator = orchestrator
        self.llm = orchestrator.llm

    async def create_plan(
        self,
        command: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create execution plan from command"""
        raise NotImplementedError

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan"""
        raise NotImplementedError


# Example usage
async def main():
    """Example usage of the orchestrator"""
    orchestrator = AzureAIOrchestrator()

    # Example commands
    commands = [
        "Create a Linux VM with 8GB RAM in East US",
        "Show me all VMs in the production resource group",
        "Diagnose high CPU usage on vm-prod-001",
        "Optimize our Azure costs without affecting production",
        "Check if all storage accounts are encrypted"
    ]

    for command in commands:
        print(f"\n📝 Command: {command}")
        result = await orchestrator.process_command(command)
        print(f"✅ Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())