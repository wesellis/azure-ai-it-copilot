"""
Cost Optimization Agent - Analyzes and optimizes Azure spending
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Optional Azure SDK imports - may not be available in all environments
try:
    from azure.mgmt.costmanagement import CostManagementClient
except ImportError:
    CostManagementClient = None

try:
    from azure.mgmt.consumption import ConsumptionManagementClient
except ImportError:
    ConsumptionManagementClient = None

try:
    from azure.mgmt.advisor import AdvisorManagementClient
except ImportError:
    AdvisorManagementClient = None
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError

from langchain.tools import Tool
import logging

# Import BaseAgent from parent module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator import BaseAgent

logger = logging.getLogger(__name__)


class CostOptimizationAgent(BaseAgent):
    """Agent for analyzing and optimizing Azure costs"""

    def __init__(self, orchestrator):
        """Initialize the cost optimization agent"""
        super().__init__(orchestrator)
        self.credential = DefaultAzureCredential()
        self.subscription_id = orchestrator.subscription_id

        # Use centralized Azure clients from orchestrator with enhanced error handling
        self.cost_client = getattr(orchestrator, 'cost_client', None)
        self.consumption_client = getattr(orchestrator, 'consumption_client', None)
        self.advisor_client = getattr(orchestrator, 'advisor_client', None)

        # Performance optimizations
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="CostAgent")

        # Rate limiting
        self._last_api_call = 0
        self._min_api_interval = 1.0  # Minimum seconds between API calls

        # Enhanced cost optimization thresholds with machine learning parameters
        self.thresholds = {
            'idle_vm_cpu': 5,  # CPU % threshold for idle VMs
            'idle_vm_days': 7,  # Days of idleness before recommendation
            'storage_unused_days': 30,  # Days before marking storage unused
            'reserved_instance_usage': 70,  # Minimum % usage for RI recommendation
            'cost_anomaly_threshold': 20,  # % increase to flag as anomaly
            'ml_confidence_threshold': 0.8,  # Minimum confidence for ML recommendations
            'savings_potential_threshold': 100,  # Minimum dollar savings to recommend
        }

        # Analytics tracking
        self._analysis_count = 0
        self._total_savings_identified = 0.0

        logger.info("💰 Optimized Cost Agent initialized")

    async def create_plan(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create cost optimization plan

        Args:
            command: Natural language command
            context: Additional context

        Returns:
            Cost optimization plan
        """
        # Parse the cost optimization request
        prompt = f"""
        Analyze this cost optimization request:

        Request: {command}
        Context: {json.dumps(context) if context else 'None'}

        Extract:
        1. Optimization goals (reduce cost by X%, find waste, etc.)
        2. Scope (all resources, specific resource groups, services)
        3. Risk tolerance (aggressive, moderate, conservative)
        4. Time frame for implementation
        5. Budget constraints

        Respond in JSON format with:
        - goal: specific optimization goal
        - target_savings: percentage or dollar amount
        - scope: resources to analyze
        - risk_tolerance: level of acceptable risk
        - implementation_timeframe: when to implement
        - constraints: any limitations
        """

        response = await self.llm.ainvoke(prompt)
        plan = json.loads(response.content)

        # Enhance with automated cost analysis
        plan['current_spend'] = await self._get_current_spend()
        plan['optimization_opportunities'] = await self._find_opportunities()
        plan['recommendations'] = self._generate_recommendations(plan)
        plan['implementation_plan'] = self._create_implementation_plan(plan)

        logger.info(f"Created cost optimization plan: {plan}")
        return plan

    def _generate_cache_key(self, method: str, *args) -> str:
        """Generate cache key for method and arguments"""
        content = f"{method}:" + ":".join(str(arg) for arg in args)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        if cache_key in self._cache:
            timestamp, data = self._cache[cache_key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl:
                return data
            else:
                del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, data: Any):
        """Set item in cache with timestamp"""
        self._cache[cache_key] = (datetime.utcnow(), data)

    async def _rate_limit(self):
        """Apply rate limiting to API calls"""
        now = time.time()
        time_since_last = now - self._last_api_call
        if time_since_last < self._min_api_interval:
            await asyncio.sleep(self._min_api_interval - time_since_last)
        self._last_api_call = time.time()

    async def _get_current_spend(self) -> Dict:
        """Get optimized current spending analysis with caching"""
        cache_key = self._generate_cache_key("current_spend")
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result

        await self._rate_limit()

        try:
            # Try to get real data from Azure Cost Management API
            if self.cost_client:
                # Real implementation would query Azure APIs here
                # For now, return enhanced simulated data
                pass
        except Exception as e:
            logger.warning(f"Failed to fetch real cost data, using simulated data: {e}")

        # Enhanced simulated cost data with more realistic patterns
        result = {
            'monthly_total': 85420,
            'daily_average': 2847,
            'by_service': {
                'Virtual Machines': 35000,
                'Storage': 12000,
                'SQL Database': 15000,
                'App Services': 8000,
                'Networking': 5000,
                'Other': 10420
            },
            'by_resource_group': {
                'rg-production': 45000,
                'rg-development': 20000,
                'rg-test': 15420,
                'rg-shared': 5000
            },
            'trend': 'increasing',
            'month_over_month_change': 8.5,
            'cost_per_hour': 2847 / 24,
            'peak_hours': [9, 10, 11, 14, 15, 16],  # Business hours
            'cost_efficiency_score': 72.5,  # 0-100 scale
            'waste_percentage': 15.2,
            'last_updated': datetime.utcnow().isoformat()
        }

        self._set_cache(cache_key, result)
        return result

    async def _find_opportunities(self) -> List[Dict]:
        """Find cost optimization opportunities"""
        opportunities = []

        # 1. Identify idle resources
        idle_resources = await self._find_idle_resources()
        opportunities.extend(idle_resources)

        # 2. Rightsizing opportunities
        rightsizing = await self._find_rightsizing_opportunities()
        opportunities.extend(rightsizing)

        # 3. Reserved Instance opportunities
        ri_opportunities = await self._find_ri_opportunities()
        opportunities.extend(ri_opportunities)

        # 4. Storage optimization
        storage_opt = await self._find_storage_optimizations()
        opportunities.extend(storage_opt)

        # 5. Orphaned resources
        orphaned = await self._find_orphaned_resources()
        opportunities.extend(orphaned)

        return sorted(opportunities, key=lambda x: x['savings'], reverse=True)

    async def _find_idle_resources(self) -> List[Dict]:
        """Find idle and underutilized resources"""
        idle_resources = []

        # Simulated idle resource detection
        idle_vms = [
            {
                'type': 'idle_vm',
                'resource': 'vm-dev-test-01',
                'reason': 'CPU < 5% for 30 days',
                'current_cost': 350,
                'savings': 350,
                'action': 'delete',
                'risk': 'low'
            },
            {
                'type': 'idle_vm',
                'resource': 'vm-staging-02',
                'reason': 'Stopped for 15 days',
                'current_cost': 280,
                'savings': 280,
                'action': 'delete',
                'risk': 'low'
            }
        ]

        idle_resources.extend(idle_vms)

        # Add other idle resource types
        idle_databases = [
            {
                'type': 'idle_database',
                'resource': 'sql-dev-analytics',
                'reason': 'No queries in 20 days',
                'current_cost': 500,
                'savings': 500,
                'action': 'pause_or_delete',
                'risk': 'medium'
            }
        ]

        idle_resources.extend(idle_databases)

        return idle_resources

    async def _find_rightsizing_opportunities(self) -> List[Dict]:
        """Find resources that can be rightsized"""
        rightsizing = []

        # Simulated rightsizing analysis
        oversized_vms = [
            {
                'type': 'rightsize',
                'resource': 'vm-prod-api-01',
                'current_size': 'D8s_v3',
                'recommended_size': 'D4s_v3',
                'reason': 'Average CPU 15%, Memory 20%',
                'current_cost': 600,
                'new_cost': 300,
                'savings': 300,
                'action': 'resize',
                'risk': 'medium'
            },
            {
                'type': 'rightsize',
                'resource': 'vm-prod-web-01',
                'current_size': 'D16s_v3',
                'recommended_size': 'D8s_v3',
                'reason': 'Average CPU 25%, Memory 30%',
                'current_cost': 1200,
                'new_cost': 600,
                'savings': 600,
                'action': 'resize',
                'risk': 'medium'
            }
        ]

        rightsizing.extend(oversized_vms)

        return rightsizing

    async def _find_ri_opportunities(self) -> List[Dict]:
        """Find Reserved Instance purchase opportunities"""
        ri_opportunities = []

        # Simulated RI analysis
        steady_state_vms = [
            {
                'type': 'reserved_instance',
                'vm_size': 'D4s_v3',
                'quantity': 5,
                'current_cost': 1500,
                'ri_cost': 900,
                'savings': 600,
                'term': '3-year',
                'payment': 'upfront',
                'action': 'purchase_ri',
                'risk': 'low'
            },
            {
                'type': 'reserved_instance',
                'vm_size': 'B2ms',
                'quantity': 10,
                'current_cost': 800,
                'ri_cost': 480,
                'savings': 320,
                'term': '1-year',
                'payment': 'monthly',
                'action': 'purchase_ri',
                'risk': 'low'
            }
        ]

        ri_opportunities.extend(steady_state_vms)

        return ri_opportunities

    async def _find_storage_optimizations(self) -> List[Dict]:
        """Find storage optimization opportunities"""
        storage_opt = []

        # Simulated storage analysis
        storage_issues = [
            {
                'type': 'storage_tier',
                'resource': 'storageaccount01',
                'current_tier': 'Hot',
                'recommended_tier': 'Cool',
                'reason': 'Accessed < 1 time per month',
                'current_cost': 200,
                'new_cost': 50,
                'savings': 150,
                'action': 'change_tier',
                'risk': 'low'
            },
            {
                'type': 'unused_disk',
                'resource': 'disk-old-backup-01',
                'size': '1TB',
                'reason': 'Unattached for 60 days',
                'current_cost': 100,
                'savings': 100,
                'action': 'delete',
                'risk': 'low'
            }
        ]

        storage_opt.extend(storage_issues)

        return storage_opt

    async def _find_orphaned_resources(self) -> List[Dict]:
        """Find orphaned resources"""
        orphaned = []

        # Simulated orphaned resource detection
        orphaned_resources = [
            {
                'type': 'orphaned_nic',
                'resource': 'nic-deleted-vm-01',
                'reason': 'Not attached to any VM',
                'current_cost': 10,
                'savings': 10,
                'action': 'delete',
                'risk': 'low'
            },
            {
                'type': 'orphaned_ip',
                'resource': 'pip-old-lb-01',
                'reason': 'Not associated with any resource',
                'current_cost': 15,
                'savings': 15,
                'action': 'delete',
                'risk': 'low'
            }
        ]

        orphaned.extend(orphaned_resources)

        return orphaned

    def _generate_recommendations(self, plan: Dict) -> List[Dict]:
        """Generate prioritized recommendations"""
        recommendations = []
        opportunities = plan.get('optimization_opportunities', [])

        # Group by risk level
        low_risk = [o for o in opportunities if o.get('risk') == 'low']
        medium_risk = [o for o in opportunities if o.get('risk') == 'medium']
        high_risk = [o for o in opportunities if o.get('risk') == 'high']

        # Create recommendations based on risk tolerance
        risk_tolerance = plan.get('risk_tolerance', 'moderate')

        if risk_tolerance == 'aggressive':
            recommendations = low_risk + medium_risk + high_risk
        elif risk_tolerance == 'moderate':
            recommendations = low_risk + medium_risk[:len(medium_risk)//2]
        else:  # conservative
            recommendations = low_risk[:len(low_risk)//2]

        # Add implementation priority
        for i, rec in enumerate(recommendations):
            rec['priority'] = i + 1
            rec['implementation_effort'] = self._assess_effort(rec)

        return recommendations

    def _assess_effort(self, recommendation: Dict) -> str:
        """Assess implementation effort"""
        action = recommendation.get('action', '')

        if action in ['delete', 'change_tier']:
            return 'low'
        elif action in ['resize', 'pause_or_delete']:
            return 'medium'
        elif action in ['purchase_ri', 'migrate']:
            return 'high'
        else:
            return 'medium'

    def _create_implementation_plan(self, plan: Dict) -> Dict:
        """Create detailed implementation plan"""
        recommendations = plan.get('recommendations', [])
        total_savings = sum(r.get('savings', 0) for r in recommendations)

        implementation = {
            'phases': [],
            'total_savings': total_savings,
            'implementation_time': '2-4 weeks',
            'resources_required': ['Cloud Administrator', 'Finance Approval']
        }

        # Phase 1: Quick wins (low risk, low effort)
        phase1 = [r for r in recommendations
                  if r.get('risk') == 'low' and r.get('implementation_effort') == 'low']
        if phase1:
            implementation['phases'].append({
                'phase': 1,
                'name': 'Quick Wins',
                'duration': '1-2 days',
                'actions': phase1,
                'savings': sum(r.get('savings', 0) for r in phase1)
            })

        # Phase 2: Medium effort optimizations
        phase2 = [r for r in recommendations
                  if r.get('implementation_effort') == 'medium']
        if phase2:
            implementation['phases'].append({
                'phase': 2,
                'name': 'Standard Optimizations',
                'duration': '1 week',
                'actions': phase2,
                'savings': sum(r.get('savings', 0) for r in phase2)
            })

        # Phase 3: Strategic changes
        phase3 = [r for r in recommendations
                  if r.get('implementation_effort') == 'high']
        if phase3:
            implementation['phases'].append({
                'phase': 3,
                'name': 'Strategic Changes',
                'duration': '2-3 weeks',
                'actions': phase3,
                'savings': sum(r.get('savings', 0) for r in phase3)
            })

        return implementation

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute cost optimization plan

        Args:
            plan: Cost optimization plan

        Returns:
            Execution result
        """
        result = {
            'status': 'executing',
            'start_time': datetime.now().isoformat(),
            'optimizations_applied': [],
            'savings_achieved': 0,
            'errors': []
        }

        try:
            # Execute recommendations by phase
            for phase in plan['implementation_plan']['phases']:
                if phase['phase'] == 1:  # Only auto-execute phase 1 (quick wins)
                    for action in phase['actions']:
                        if action.get('risk') == 'low':
                            execution_result = await self._execute_optimization(action)
                            result['optimizations_applied'].append(execution_result)
                            if execution_result['status'] == 'success':
                                result['savings_achieved'] += action.get('savings', 0)

            result['status'] = 'completed'
            result['end_time'] = datetime.now().isoformat()
            result['summary'] = self._generate_summary(result)

        except Exception as e:
            logger.error(f"Cost optimization execution failed: {str(e)}")
            result['status'] = 'failed'
            result['error'] = str(e)

        return result

    async def _execute_optimization(self, optimization: Dict) -> Dict:
        """Execute a specific optimization action"""
        action = optimization.get('action')
        resource = optimization.get('resource')

        logger.info(f"Executing optimization: {action} on {resource}")

        # Simulate execution based on action type
        if action == 'delete':
            return await self._delete_resource(resource)
        elif action == 'resize':
            return await self._resize_resource(
                resource,
                optimization.get('recommended_size')
            )
        elif action == 'change_tier':
            return await self._change_storage_tier(
                resource,
                optimization.get('recommended_tier')
            )
        else:
            return {
                'action': action,
                'resource': resource,
                'status': 'manual_action_required',
                'message': f"Action {action} requires manual intervention"
            }

    async def _delete_resource(self, resource: str) -> Dict:
        """Delete a resource"""
        # Simulated deletion
        await asyncio.sleep(2)

        return {
            'action': 'delete',
            'resource': resource,
            'status': 'success',
            'message': f"Successfully deleted {resource}"
        }

    async def _resize_resource(self, resource: str, new_size: str) -> Dict:
        """Resize a resource"""
        # Simulated resizing
        await asyncio.sleep(3)

        return {
            'action': 'resize',
            'resource': resource,
            'new_size': new_size,
            'status': 'success',
            'message': f"Successfully resized {resource} to {new_size}"
        }

    async def _change_storage_tier(self, resource: str, new_tier: str) -> Dict:
        """Change storage tier"""
        # Simulated tier change
        await asyncio.sleep(1)

        return {
            'action': 'change_tier',
            'resource': resource,
            'new_tier': new_tier,
            'status': 'success',
            'message': f"Successfully changed {resource} to {new_tier} tier"
        }

    def _generate_summary(self, result: Dict) -> Dict:
        """Generate execution summary"""
        return {
            'total_optimizations': len(result['optimizations_applied']),
            'successful': len([o for o in result['optimizations_applied']
                             if o['status'] == 'success']),
            'failed': len([o for o in result['optimizations_applied']
                         if o['status'] == 'failed']),
            'savings_achieved': result['savings_achieved'],
            'monthly_savings': result['savings_achieved'],
            'annual_savings': result['savings_achieved'] * 12
        }

    async def generate_cost_report(
        self,
        timeframe: str = '30d'
    ) -> Dict:
        """Generate detailed cost report"""
        report = {
            'timeframe': timeframe,
            'total_spend': 85420,
            'forecast_next_month': 92000,
            'top_resources': [],
            'cost_trends': [],
            'recommendations': []
        }

        # Add detailed analysis
        # This would query actual Azure Cost Management data

        return report