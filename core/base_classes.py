"""
Base classes for Azure AI IT Copilot
Provides common functionality and enforces contracts
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .interfaces import IAgent, IConfigurationProvider, IAzureClientFactory
from .exceptions import AgentExecutionError


class BaseAgent(ABC, IAgent):
    """Base class for all AI agents"""

    def __init__(self, config_provider: IConfigurationProvider,
                 azure_client_factory: IAzureClientFactory):
        self.config_provider = config_provider
        self.azure_client_factory = azure_client_factory
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize agent-specific configuration"""
        pass

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Type identifier for the agent"""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """List of capabilities this agent provides"""
        pass

    @abstractmethod
    async def create_plan(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create execution plan from command"""
        return {
            "operation": "unknown",
            "command": command,
            "context": context or {},
            "steps": [],
            "requires_approval": False,
            "estimated_time": "unknown",
            "agent_type": self.agent_type,
            "created_at": datetime.utcnow().isoformat()
        }

    @abstractmethod
    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan"""
        try:
            self.logger.info(f"Executing plan: {plan.get('operation', 'unknown')}")

            # Default implementation - should be overridden
            return {
                "success": True,
                "message": "Plan executed successfully",
                "agent_type": self.agent_type,
                "executed_at": datetime.utcnow().isoformat(),
                "plan_id": plan.get("plan_id")
            }
        except Exception as e:
            self.logger.error(f"Plan execution failed: {str(e)}")
            raise AgentExecutionError(f"Agent {self.agent_type} execution failed: {str(e)}")

    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if agent can handle the command"""
        # Default implementation - can be overridden
        command_lower = command.lower()
        return any(capability.lower() in command_lower for capability in self.capabilities)

    def _validate_plan(self, plan: Dict[str, Any]) -> bool:
        """Validate plan structure"""
        required_fields = ["operation", "command", "steps", "agent_type"]
        return all(field in plan for field in required_fields)

    def _estimate_execution_time(self, plan: Dict[str, Any]) -> str:
        """Estimate execution time for plan"""
        step_count = len(plan.get("steps", []))
        if step_count <= 2:
            return "1-2 minutes"
        elif step_count <= 5:
            return "2-5 minutes"
        else:
            return "5-10 minutes"


class BaseService(ABC):
    """Base class for all services"""

    def __init__(self, config_provider: IConfigurationProvider):
        self.config_provider = config_provider
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "service": self.__class__.__name__,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }


class BaseRepository(ABC):
    """Base class for data repositories"""

    def __init__(self, config_provider: IConfigurationProvider):
        self.config_provider = config_provider
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the repository"""
        pass

    @abstractmethod
    async def create(self, entity: Dict[str, Any]) -> str:
        """Create a new entity"""
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID"""
        pass

    @abstractmethod
    async def update(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update entity"""
        pass

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete entity"""
        pass

    async def find(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Find entities by filters - default implementation"""
        return []


class BaseClient(ABC):
    """Base class for external service clients"""

    def __init__(self, config_provider: IConfigurationProvider):
        self.config_provider = config_provider
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to external service"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to external service"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check health of external service"""
        pass


class BaseEventHandler(ABC):
    """Base class for event handlers"""

    def __init__(self, config_provider: IConfigurationProvider):
        self.config_provider = config_provider
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def supported_events(self) -> List[str]:
        """List of event types this handler supports"""
        pass

    @abstractmethod
    async def handle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle an event"""
        pass

    def can_handle(self, event_type: str) -> bool:
        """Check if handler can process event type"""
        return event_type in self.supported_events