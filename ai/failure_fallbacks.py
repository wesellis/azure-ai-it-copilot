"""
AI Failure Fallback Strategies for Azure AI IT Copilot
Comprehensive failover mechanisms for AI service failures and degraded performance
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of AI failures"""
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    AUTHENTICATION_ERROR = "authentication_error"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_ERROR = "model_error"
    NETWORK_ERROR = "network_error"
    PARSING_ERROR = "parsing_error"
    CONTENT_FILTER = "content_filter"


class FallbackStrategy(Enum):
    """Available fallback strategies"""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    SWITCH_MODEL = "switch_model"
    CACHED_RESPONSE = "cached_response"
    RULE_BASED_RESPONSE = "rule_based_response"
    SIMPLIFIED_PROMPT = "simplified_prompt"
    HUMAN_ESCALATION = "human_escalation"
    DEFAULT_RESPONSE = "default_response"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class FailoverPriority(Enum):
    """Priority levels for failover"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FailureEvent:
    """Represents an AI failure event"""
    failure_id: str
    failure_type: FailureType
    service_name: str
    model_name: str
    prompt: str
    error_message: str
    timestamp: str
    metadata: Dict[str, Any]
    recovery_strategy: Optional[str] = None
    recovery_success: bool = False
    recovery_time_ms: Optional[float] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class FallbackRule:
    """Configuration for a fallback strategy"""
    rule_id: str
    failure_types: List[FailureType]
    strategy: FallbackStrategy
    priority: FailoverPriority
    conditions: Dict[str, Any]
    configuration: Dict[str, Any]
    enabled: bool = True
    max_attempts: int = 3
    timeout_seconds: int = 30


@dataclass
class AIServiceConfig:
    """Configuration for an AI service"""
    service_id: str
    service_name: str
    model_name: str
    endpoint: str
    api_key: str
    priority: int = 1
    max_requests_per_minute: int = 60
    timeout_seconds: int = 30
    enabled: bool = True
    cost_per_token: float = 0.0001


class AIFailureHandler:
    """Comprehensive AI failure handling and fallback system"""

    def __init__(self):
        self.failure_history: List[FailureEvent] = []
        self.fallback_rules: List[FallbackRule] = []
        self.ai_services: Dict[str, AIServiceConfig] = {}
        self.response_cache: Dict[str, Any] = {}
        self.rule_based_responses: Dict[str, str] = {}

        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.service_metrics: Dict[str, Dict[str, float]] = {}

        # Initialize default configurations
        self._setup_default_rules()
        self._setup_default_services()
        self._setup_rule_based_responses()

    def _setup_default_rules(self):
        """Setup default fallback rules"""
        self.fallback_rules = [
            # High priority: Retry with exponential backoff for transient errors
            FallbackRule(
                rule_id="retry_transient",
                failure_types=[
                    FailureType.TIMEOUT,
                    FailureType.NETWORK_ERROR,
                    FailureType.SERVICE_UNAVAILABLE
                ],
                strategy=FallbackStrategy.RETRY_WITH_BACKOFF,
                priority=FailoverPriority.HIGH,
                conditions={"max_retries": 3, "backoff_multiplier": 2},
                configuration={"initial_delay": 1, "max_delay": 60},
                max_attempts=3
            ),

            # High priority: Switch to backup model for rate limits
            FallbackRule(
                rule_id="switch_on_rate_limit",
                failure_types=[FailureType.RATE_LIMIT_EXCEEDED, FailureType.QUOTA_EXCEEDED],
                strategy=FallbackStrategy.SWITCH_MODEL,
                priority=FailoverPriority.HIGH,
                conditions={"backup_service_available": True},
                configuration={"prefer_lower_cost": True}
            ),

            # Medium priority: Use cached response for repeated queries
            FallbackRule(
                rule_id="use_cache",
                failure_types=[
                    FailureType.SERVICE_UNAVAILABLE,
                    FailureType.RATE_LIMIT_EXCEEDED,
                    FailureType.TIMEOUT
                ],
                strategy=FallbackStrategy.CACHED_RESPONSE,
                priority=FailoverPriority.MEDIUM,
                conditions={"cache_max_age_minutes": 60},
                configuration={"similarity_threshold": 0.8}
            ),

            # Medium priority: Simplify prompt for complex queries
            FallbackRule(
                rule_id="simplify_prompt",
                failure_types=[FailureType.MODEL_ERROR, FailureType.CONTENT_FILTER],
                strategy=FallbackStrategy.SIMPLIFIED_PROMPT,
                priority=FailoverPriority.MEDIUM,
                conditions={"prompt_length_threshold": 1000},
                configuration={"simplification_strategies": ["remove_examples", "reduce_context"]}
            ),

            # Low priority: Rule-based response for common queries
            FallbackRule(
                rule_id="rule_based_fallback",
                failure_types=[
                    FailureType.SERVICE_UNAVAILABLE,
                    FailureType.MODEL_ERROR,
                    FailureType.INVALID_RESPONSE
                ],
                strategy=FallbackStrategy.RULE_BASED_RESPONSE,
                priority=FailoverPriority.LOW,
                conditions={"query_type": "azure_operation"},
                configuration={"confidence_threshold": 0.7}
            ),

            # Low priority: Default response when all else fails
            FallbackRule(
                rule_id="default_response",
                failure_types=list(FailureType),  # All failure types
                strategy=FallbackStrategy.DEFAULT_RESPONSE,
                priority=FailoverPriority.LOW,
                conditions={},
                configuration={"message": "I'm experiencing technical difficulties. Please try again later."}
            )
        ]

    def _setup_default_services(self):
        """Setup default AI service configurations"""
        self.ai_services = {
            "azure_openai_gpt4": AIServiceConfig(
                service_id="azure_openai_gpt4",
                service_name="Azure OpenAI GPT-4",
                model_name="gpt-4",
                endpoint="https://your-openai.openai.azure.com",
                api_key="your-api-key",
                priority=1,
                max_requests_per_minute=60,
                cost_per_token=0.00003
            ),
            "azure_openai_gpt35": AIServiceConfig(
                service_id="azure_openai_gpt35",
                service_name="Azure OpenAI GPT-3.5",
                model_name="gpt-3.5-turbo",
                endpoint="https://your-openai.openai.azure.com",
                api_key="your-api-key",
                priority=2,
                max_requests_per_minute=120,
                cost_per_token=0.000002
            ),
            "openai_gpt4": AIServiceConfig(
                service_id="openai_gpt4",
                service_name="OpenAI GPT-4",
                model_name="gpt-4",
                endpoint="https://api.openai.com/v1",
                api_key="your-openai-key",
                priority=3,
                max_requests_per_minute=40,
                cost_per_token=0.00003
            )
        }

    def _setup_rule_based_responses(self):
        """Setup rule-based response templates"""
        self.rule_based_responses = {
            "list_vms": "I can help you list virtual machines. Please specify the resource group or subscription.",
            "create_vm": "To create a virtual machine, I need the following information: name, size, location, and resource group.",
            "list_resources": "I can help you list Azure resources. Please specify the resource group or type of resource.",
            "check_status": "I can check the status of Azure resources. Please provide the resource name or ID.",
            "get_costs": "I can help with cost analysis. Please specify the time period and scope.",
            "security_check": "I can perform security assessments. Please specify the resources to analyze.",
            "backup_status": "I can check backup status for your resources. Please specify which resources to check.",
            "scale_resources": "I can help with scaling resources. Please specify which resources and target configuration.",
            "monitor_alerts": "I can help with monitoring and alerts. Please specify what you'd like to monitor.",
            "compliance_check": "I can perform compliance checks. Please specify the compliance framework."
        }

    async def handle_ai_failure(self,
                              failure_type: FailureType,
                              service_name: str,
                              model_name: str,
                              prompt: str,
                              error_message: str,
                              metadata: Dict[str, Any] = None) -> Tuple[bool, Any, str]:
        """
        Handle AI failure and attempt recovery using fallback strategies

        Returns:
            (success: bool, response: Any, strategy_used: str)
        """

        failure_event = FailureEvent(
            failure_id=f"{service_name}_{int(time.time() * 1000)}",
            failure_type=failure_type,
            service_name=service_name,
            model_name=model_name,
            prompt=prompt,
            error_message=error_message,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        self.failure_history.append(failure_event)

        logger.warning(f"AI failure detected: {failure_type.value} in {service_name}")

        # Update circuit breaker
        self._update_circuit_breaker(service_name, failure_type)

        # Find applicable fallback rules
        applicable_rules = self._find_applicable_rules(failure_type, prompt, metadata or {})

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority.value, reverse=True)

        # Try each fallback strategy
        for rule in applicable_rules:
            if not rule.enabled:
                continue

            logger.info(f"Attempting fallback strategy: {rule.strategy.value}")

            try:
                start_time = time.time()
                success, response = await self._execute_fallback_strategy(rule, prompt, metadata or {})
                recovery_time = (time.time() - start_time) * 1000

                if success:
                    failure_event.recovery_strategy = rule.strategy.value
                    failure_event.recovery_success = True
                    failure_event.recovery_time_ms = recovery_time

                    logger.info(f"Fallback successful using {rule.strategy.value} in {recovery_time:.2f}ms")
                    return True, response, rule.strategy.value

            except Exception as e:
                logger.error(f"Fallback strategy {rule.strategy.value} failed: {e}")
                continue

        # All fallback strategies failed
        failure_event.recovery_success = False
        logger.error("All fallback strategies failed")

        return False, None, "no_strategy_succeeded"

    def _find_applicable_rules(self,
                              failure_type: FailureType,
                              prompt: str,
                              metadata: Dict[str, Any]) -> List[FallbackRule]:
        """Find fallback rules applicable to the current failure"""

        applicable_rules = []

        for rule in self.fallback_rules:
            # Check if failure type matches
            if failure_type not in rule.failure_types:
                continue

            # Check conditions
            if not self._check_rule_conditions(rule, prompt, metadata):
                continue

            applicable_rules.append(rule)

        return applicable_rules

    def _check_rule_conditions(self,
                              rule: FallbackRule,
                              prompt: str,
                              metadata: Dict[str, Any]) -> bool:
        """Check if rule conditions are met"""

        conditions = rule.conditions

        # Check prompt length condition
        if "prompt_length_threshold" in conditions:
            if len(prompt) < conditions["prompt_length_threshold"]:
                return False

        # Check query type condition
        if "query_type" in conditions:
            query_type = metadata.get("query_type", "")
            if query_type != conditions["query_type"]:
                return False

        # Check backup service availability
        if "backup_service_available" in conditions:
            if not self._has_available_backup_service():
                return False

        # Check cache age condition
        if "cache_max_age_minutes" in conditions:
            cache_entry = self._find_cached_response(prompt)
            if cache_entry:
                age_minutes = (time.time() - cache_entry["timestamp"]) / 60
                if age_minutes > conditions["cache_max_age_minutes"]:
                    return False

        return True

    async def _execute_fallback_strategy(self,
                                       rule: FallbackRule,
                                       prompt: str,
                                       metadata: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute a specific fallback strategy"""

        strategy = rule.strategy

        if strategy == FallbackStrategy.RETRY_WITH_BACKOFF:
            return await self._retry_with_backoff(rule, prompt, metadata)

        elif strategy == FallbackStrategy.SWITCH_MODEL:
            return await self._switch_model(rule, prompt, metadata)

        elif strategy == FallbackStrategy.CACHED_RESPONSE:
            return await self._use_cached_response(rule, prompt, metadata)

        elif strategy == FallbackStrategy.RULE_BASED_RESPONSE:
            return await self._rule_based_response(rule, prompt, metadata)

        elif strategy == FallbackStrategy.SIMPLIFIED_PROMPT:
            return await self._simplified_prompt(rule, prompt, metadata)

        elif strategy == FallbackStrategy.DEFAULT_RESPONSE:
            return await self._default_response(rule, prompt, metadata)

        elif strategy == FallbackStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation(rule, prompt, metadata)

        elif strategy == FallbackStrategy.HUMAN_ESCALATION:
            return await self._human_escalation(rule, prompt, metadata)

        else:
            logger.error(f"Unknown fallback strategy: {strategy}")
            return False, None

    async def _retry_with_backoff(self,
                                rule: FallbackRule,
                                prompt: str,
                                metadata: Dict[str, Any]) -> Tuple[bool, Any]:
        """Retry the original request with exponential backoff"""

        config = rule.configuration
        initial_delay = config.get("initial_delay", 1)
        max_delay = config.get("max_delay", 60)
        max_retries = rule.max_attempts

        for attempt in range(max_retries):
            if attempt > 0:
                delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
                await asyncio.sleep(delay)

            try:
                # Attempt to call the original AI service
                # This would be replaced with actual service call
                response = await self._call_ai_service_with_retry(prompt, metadata)
                return True, response

            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return False, None

        return False, None

    async def _switch_model(self,
                          rule: FallbackRule,
                          prompt: str,
                          metadata: Dict[str, Any]) -> Tuple[bool, Any]:
        """Switch to a backup AI model/service"""

        config = rule.configuration
        prefer_lower_cost = config.get("prefer_lower_cost", False)

        # Find available backup services
        available_services = [
            service for service in self.ai_services.values()
            if service.enabled and not self._is_circuit_breaker_open(service.service_id)
        ]

        if not available_services:
            return False, None

        # Sort by priority or cost
        if prefer_lower_cost:
            available_services.sort(key=lambda s: s.cost_per_token)
        else:
            available_services.sort(key=lambda s: s.priority)

        # Try each available service
        for service in available_services:
            try:
                response = await self._call_specific_ai_service(service, prompt, metadata)
                return True, response
            except Exception as e:
                logger.warning(f"Backup service {service.service_name} failed: {e}")
                continue

        return False, None

    async def _use_cached_response(self,
                                 rule: FallbackRule,
                                 prompt: str,
                                 metadata: Dict[str, Any]) -> Tuple[bool, Any]:
        """Use a cached response for similar queries"""

        config = rule.configuration
        similarity_threshold = config.get("similarity_threshold", 0.8)

        cached_response = self._find_similar_cached_response(prompt, similarity_threshold)

        if cached_response:
            logger.info("Using cached response for similar query")
            return True, cached_response["response"]

        return False, None

    async def _rule_based_response(self,
                                 rule: FallbackRule,
                                 prompt: str,
                                 metadata: Dict[str, Any]) -> Tuple[bool, Any]:
        """Generate rule-based response for common queries"""

        query_intent = self._classify_query_intent(prompt)

        if query_intent in self.rule_based_responses:
            response = self.rule_based_responses[query_intent]
            logger.info(f"Using rule-based response for intent: {query_intent}")
            return True, {"response": response, "source": "rule_based", "intent": query_intent}

        return False, None

    async def _simplified_prompt(self,
                               rule: FallbackRule,
                               prompt: str,
                               metadata: Dict[str, Any]) -> Tuple[bool, Any]:
        """Simplify the prompt and retry"""

        config = rule.configuration
        strategies = config.get("simplification_strategies", ["remove_examples"])

        simplified_prompt = self._simplify_prompt(prompt, strategies)

        if simplified_prompt != prompt:
            try:
                response = await self._call_ai_service_with_retry(simplified_prompt, metadata)
                return True, response
            except Exception as e:
                logger.warning(f"Simplified prompt failed: {e}")

        return False, None

    async def _default_response(self,
                              rule: FallbackRule,
                              prompt: str,
                              metadata: Dict[str, Any]) -> Tuple[bool, Any]:
        """Return a default response"""

        config = rule.configuration
        message = config.get("message", "I'm unable to process your request at the moment.")

        response = {
            "response": message,
            "source": "default_fallback",
            "timestamp": datetime.now().isoformat()
        }

        return True, response

    async def _graceful_degradation(self,
                                  rule: FallbackRule,
                                  prompt: str,
                                  metadata: Dict[str, Any]) -> Tuple[bool, Any]:
        """Provide reduced functionality gracefully"""

        # Analyze what the user is trying to do
        intent = self._classify_query_intent(prompt)

        degraded_response = {
            "response": f"I can help with {intent}, but with limited capabilities right now.",
            "available_actions": self._get_available_actions(intent),
            "source": "graceful_degradation",
            "full_service_eta": "5-10 minutes"
        }

        return True, degraded_response

    async def _human_escalation(self,
                              rule: FallbackRule,
                              prompt: str,
                              metadata: Dict[str, Any]) -> Tuple[bool, Any]:
        """Escalate to human operator"""

        escalation_ticket = {
            "ticket_id": f"ESC_{int(time.time())}",
            "prompt": prompt,
            "failure_context": metadata,
            "escalated_at": datetime.now().isoformat(),
            "priority": "high" if "urgent" in prompt.lower() else "normal"
        }

        # In a real implementation, this would integrate with a ticketing system
        logger.info(f"Escalating to human operator: {escalation_ticket['ticket_id']}")

        response = {
            "response": "Your request has been escalated to a human operator who will assist you shortly.",
            "ticket_id": escalation_ticket["ticket_id"],
            "estimated_response_time": "15-30 minutes",
            "source": "human_escalation"
        }

        return True, response

    def _update_circuit_breaker(self, service_name: str, failure_type: FailureType):
        """Update circuit breaker state for a service"""

        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = {
                "failure_count": 0,
                "last_failure": time.time(),
                "state": "closed",  # closed, open, half_open
                "open_until": 0
            }

        breaker = self.circuit_breakers[service_name]
        breaker["failure_count"] += 1
        breaker["last_failure"] = time.time()

        # Open circuit breaker if too many failures
        if breaker["failure_count"] >= 5 and breaker["state"] == "closed":
            breaker["state"] = "open"
            breaker["open_until"] = time.time() + 300  # 5 minutes
            logger.warning(f"Circuit breaker opened for {service_name}")

    def _is_circuit_breaker_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service"""

        if service_name not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[service_name]

        if breaker["state"] == "open":
            if time.time() > breaker["open_until"]:
                breaker["state"] = "half_open"
                logger.info(f"Circuit breaker half-open for {service_name}")
                return False
            return True

        return False

    def _has_available_backup_service(self) -> bool:
        """Check if any backup services are available"""
        return any(
            service.enabled and not self._is_circuit_breaker_open(service.service_id)
            for service in self.ai_services.values()
        )

    def _find_cached_response(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Find exact cached response"""
        prompt_hash = str(hash(prompt))
        return self.response_cache.get(prompt_hash)

    def _find_similar_cached_response(self, prompt: str, threshold: float) -> Optional[Dict[str, Any]]:
        """Find similar cached response using simple similarity"""
        # Simplified similarity check - in practice, use embeddings
        prompt_words = set(prompt.lower().split())

        for cached_prompt, cache_entry in self.response_cache.items():
            cached_words = set(cache_entry["prompt"].lower().split())

            if len(prompt_words) == 0 or len(cached_words) == 0:
                continue

            similarity = len(prompt_words & cached_words) / len(prompt_words | cached_words)

            if similarity >= threshold:
                return cache_entry

        return None

    def _classify_query_intent(self, prompt: str) -> str:
        """Classify the intent of a query using simple keyword matching"""

        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["list", "show", "get", "vm", "virtual machine"]):
            return "list_vms"
        elif any(word in prompt_lower for word in ["create", "deploy", "new", "vm"]):
            return "create_vm"
        elif any(word in prompt_lower for word in ["resources", "list resources"]):
            return "list_resources"
        elif any(word in prompt_lower for word in ["status", "health", "check"]):
            return "check_status"
        elif any(word in prompt_lower for word in ["cost", "billing", "spend"]):
            return "get_costs"
        elif any(word in prompt_lower for word in ["security", "vulnerabilities"]):
            return "security_check"
        elif any(word in prompt_lower for word in ["backup", "restore"]):
            return "backup_status"
        elif any(word in prompt_lower for word in ["scale", "resize", "size"]):
            return "scale_resources"
        elif any(word in prompt_lower for word in ["monitor", "alert", "notification"]):
            return "monitor_alerts"
        elif any(word in prompt_lower for word in ["compliance", "policy", "governance"]):
            return "compliance_check"
        else:
            return "general_query"

    def _simplify_prompt(self, prompt: str, strategies: List[str]) -> str:
        """Simplify a prompt using various strategies"""

        simplified = prompt

        if "remove_examples" in strategies:
            # Remove example sections
            lines = simplified.split('\n')
            filtered_lines = []
            skip_example = False

            for line in lines:
                if any(word in line.lower() for word in ["example:", "for example", "e.g."]):
                    skip_example = True
                elif line.strip() == "" and skip_example:
                    skip_example = False
                elif not skip_example:
                    filtered_lines.append(line)

            simplified = '\n'.join(filtered_lines)

        if "reduce_context" in strategies:
            # Keep only the last 500 characters
            if len(simplified) > 500:
                simplified = "..." + simplified[-500:]

        if "remove_formatting" in strategies:
            # Remove markdown formatting
            simplified = simplified.replace("**", "").replace("*", "").replace("#", "")

        return simplified.strip()

    def _get_available_actions(self, intent: str) -> List[str]:
        """Get available actions for degraded mode"""

        action_map = {
            "list_vms": ["List VMs in resource group", "Check VM status"],
            "create_vm": ["Create basic VM", "Use VM template"],
            "list_resources": ["List by resource group", "List by type"],
            "check_status": ["Basic health check", "Simple status query"],
            "get_costs": ["Current month costs", "Resource group costs"],
            "security_check": ["Basic security scan", "Check recommendations"],
            "backup_status": ["Check backup jobs", "List backup policies"],
            "scale_resources": ["Manual scaling", "Predefined scaling options"],
            "monitor_alerts": ["View active alerts", "Check alert rules"],
            "compliance_check": ["Basic compliance scan", "Policy evaluation"]
        }

        return action_map.get(intent, ["Basic assistance available"])

    async def _call_ai_service_with_retry(self, prompt: str, metadata: Dict[str, Any]) -> Any:
        """Call AI service with retry logic (mock implementation)"""
        # This would be replaced with actual AI service call
        await asyncio.sleep(0.1)  # Simulate API call

        # Simulate random failures for testing
        if random.random() < 0.3:  # 30% failure rate
            raise Exception("Simulated AI service failure")

        return {"response": f"AI response for: {prompt[:50]}...", "model": "mock_ai"}

    async def _call_specific_ai_service(self,
                                      service: AIServiceConfig,
                                      prompt: str,
                                      metadata: Dict[str, Any]) -> Any:
        """Call a specific AI service (mock implementation)"""
        await asyncio.sleep(0.1)  # Simulate API call

        return {
            "response": f"Response from {service.service_name}: {prompt[:50]}...",
            "model": service.model_name,
            "service": service.service_name
        }

    def cache_response(self, prompt: str, response: Any):
        """Cache a successful response"""
        prompt_hash = str(hash(prompt))
        self.response_cache[prompt_hash] = {
            "prompt": prompt,
            "response": response,
            "timestamp": time.time()
        }

        # Limit cache size
        if len(self.response_cache) > 1000:
            # Remove oldest entries
            oldest_key = min(self.response_cache.keys(),
                           key=lambda k: self.response_cache[k]["timestamp"])
            del self.response_cache[oldest_key]

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure statistics and metrics"""

        if not self.failure_history:
            return {"total_failures": 0}

        total_failures = len(self.failure_history)
        successful_recoveries = len([f for f in self.failure_history if f.recovery_success])

        # Group by failure type
        failure_types = {}
        for failure in self.failure_history:
            failure_type = failure.failure_type.value
            if failure_type not in failure_types:
                failure_types[failure_type] = 0
            failure_types[failure_type] += 1

        # Group by service
        service_failures = {}
        for failure in self.failure_history:
            service = failure.service_name
            if service not in service_failures:
                service_failures[service] = 0
            service_failures[service] += 1

        # Calculate average recovery time
        recovery_times = [f.recovery_time_ms for f in self.failure_history
                         if f.recovery_time_ms is not None]
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0

        return {
            "total_failures": total_failures,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": (successful_recoveries / total_failures) * 100 if total_failures > 0 else 0,
            "avg_recovery_time_ms": avg_recovery_time,
            "failure_types": failure_types,
            "service_failures": service_failures,
            "circuit_breaker_status": {
                service: breaker["state"]
                for service, breaker in self.circuit_breakers.items()
            }
        }

    def reset_circuit_breaker(self, service_name: str):
        """Manually reset a circuit breaker"""
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name] = {
                "failure_count": 0,
                "last_failure": 0,
                "state": "closed",
                "open_until": 0
            }
            logger.info(f"Circuit breaker reset for {service_name}")


# Global failure handler instance
failure_handler = AIFailureHandler()


# Decorator for automatic failure handling
def handle_ai_failures(service_name: str, model_name: str = "default"):
    """Decorator to automatically handle AI failures"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)

                # Cache successful response if applicable
                if "prompt" in kwargs:
                    failure_handler.cache_response(kwargs["prompt"], result)

                return result

            except Exception as e:
                # Determine failure type based on exception
                failure_type = _classify_exception(e)

                # Extract prompt from arguments
                prompt = kwargs.get("prompt", str(args[0]) if args else "")

                # Handle the failure
                success, response, strategy = await failure_handler.handle_ai_failure(
                    failure_type=failure_type,
                    service_name=service_name,
                    model_name=model_name,
                    prompt=prompt,
                    error_message=str(e),
                    metadata={"function": func.__name__, "args": len(args), "kwargs": list(kwargs.keys())}
                )

                if success:
                    logger.info(f"Recovered from AI failure using {strategy}")
                    return response
                else:
                    logger.error(f"Failed to recover from AI failure: {e}")
                    raise

        return wrapper
    return decorator


def _classify_exception(exception: Exception) -> FailureType:
    """Classify an exception into a failure type"""
    error_message = str(exception).lower()

    if "timeout" in error_message or "timed out" in error_message:
        return FailureType.TIMEOUT
    elif "rate limit" in error_message or "quota" in error_message:
        return FailureType.RATE_LIMIT_EXCEEDED
    elif "authentication" in error_message or "unauthorized" in error_message:
        return FailureType.AUTHENTICATION_ERROR
    elif "service unavailable" in error_message or "502" in error_message or "503" in error_message:
        return FailureType.SERVICE_UNAVAILABLE
    elif "network" in error_message or "connection" in error_message:
        return FailureType.NETWORK_ERROR
    elif "content filter" in error_message or "inappropriate" in error_message:
        return FailureType.CONTENT_FILTER
    elif "parsing" in error_message or "json" in error_message:
        return FailureType.PARSING_ERROR
    else:
        return FailureType.MODEL_ERROR


# Example usage
if __name__ == "__main__":
    async def main():
        # Example AI function with failure handling
        @handle_ai_failures("azure_openai", "gpt-4")
        async def ask_ai(prompt: str) -> str:
            # Simulate AI call that might fail
            if random.random() < 0.5:
                raise Exception("Rate limit exceeded")
            return f"AI response: {prompt}"

        # Test the failure handling
        for i in range(10):
            try:
                response = await ask_ai(f"Test prompt {i}")
                print(f"Success: {response}")
            except Exception as e:
                print(f"Final failure: {e}")

        # Get statistics
        stats = failure_handler.get_failure_statistics()
        print(f"Failure statistics: {json.dumps(stats, indent=2)}")

    asyncio.run(main())