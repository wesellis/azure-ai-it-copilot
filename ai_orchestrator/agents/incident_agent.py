"""
Incident Response Agent - Diagnoses and resolves IT incidents automatically
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

from azure.monitor.query import LogsQueryClient, MetricsQueryClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
import pandas as pd
import numpy as np

from langchain.tools import Tool
from langchain.schema import HumanMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)


class IncidentAgent:
    """Agent for diagnosing and resolving IT incidents"""

    def __init__(self, orchestrator):
        """Initialize the incident response agent"""
        self.orchestrator = orchestrator
        self.llm = orchestrator.llm

        # Use the orchestrator's Azure Monitor clients
        self.logs_client = orchestrator.logs_client
        self.metrics_client = orchestrator.metrics_client

        # Knowledge base of known issues and solutions
        self.knowledge_base = self._load_knowledge_base()

        # Diagnostic patterns
        self.diagnostic_patterns = self._load_diagnostic_patterns()

    def _load_knowledge_base(self) -> Dict:
        """Load knowledge base of known issues"""
        return {
            'high_cpu': {
                'symptoms': ['cpu_percent > 90', 'response_time > 1000ms'],
                'causes': [
                    'Memory leak in application',
                    'Infinite loop in code',
                    'Resource-intensive query',
                    'Insufficient compute resources'
                ],
                'solutions': [
                    'Restart application service',
                    'Scale up VM size',
                    'Optimize database queries',
                    'Implement caching'
                ]
            },
            'high_memory': {
                'symptoms': ['memory_percent > 90', 'page_faults increasing'],
                'causes': [
                    'Memory leak',
                    'Large data processing',
                    'Too many concurrent connections',
                    'Cache overflow'
                ],
                'solutions': [
                    'Restart application',
                    'Increase memory allocation',
                    'Implement memory limits',
                    'Clear cache'
                ]
            },
            'slow_response': {
                'symptoms': ['response_time > 3000ms', 'timeout_errors increasing'],
                'causes': [
                    'Database performance issues',
                    'Network latency',
                    'API rate limiting',
                    'Downstream service failure'
                ],
                'solutions': [
                    'Optimize database indexes',
                    'Implement caching',
                    'Increase connection pool',
                    'Circuit breaker implementation'
                ]
            },
            'service_down': {
                'symptoms': ['health_check_failed', 'connection_refused'],
                'causes': [
                    'Service crashed',
                    'Port blocked',
                    'Certificate expired',
                    'Deployment failure'
                ],
                'solutions': [
                    'Restart service',
                    'Check firewall rules',
                    'Renew certificates',
                    'Rollback deployment'
                ]
            }
        }

    def _load_diagnostic_patterns(self) -> List[Dict]:
        """Load diagnostic patterns for root cause analysis"""
        return [
            {
                'pattern': 'gradual_degradation',
                'indicators': ['metrics increasing over time', 'no sudden changes'],
                'likely_cause': 'resource leak or growing dataset'
            },
            {
                'pattern': 'sudden_spike',
                'indicators': ['immediate jump in metrics', 'correlated with event'],
                'likely_cause': 'deployment, config change, or traffic surge'
            },
            {
                'pattern': 'periodic_issue',
                'indicators': ['regular pattern', 'time-based correlation'],
                'likely_cause': 'scheduled job or batch process'
            },
            {
                'pattern': 'cascading_failure',
                'indicators': ['multiple services affected', 'sequential failures'],
                'likely_cause': 'dependency failure or resource exhaustion'
            }
        ]

    async def create_plan(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create diagnostic and remediation plan

        Args:
            command: Natural language description of the incident
            context: Additional context about the incident

        Returns:
            Diagnostic and remediation plan
        """
        # Parse the incident description
        prompt = f"""
        Analyze this incident report and create a diagnostic plan:

        Incident: {command}
        Context: {json.dumps(context) if context else 'None'}

        Extract:
        1. Affected resource(s)
        2. Symptoms described
        3. Time frame
        4. Severity (critical, high, medium, low)
        5. Initial hypothesis

        Respond in JSON format with:
        - affected_resources: list of resource names/IDs
        - symptoms: list of observed symptoms
        - timeframe: when the issue started
        - severity: severity level
        - hypotheses: list of potential causes
        - diagnostic_steps: ordered list of diagnostic actions
        """

        response = await self.llm.ainvoke(prompt)
        plan = json.loads(response.content)

        # Enhance with automated diagnostics
        plan['automated_diagnostics'] = await self._generate_diagnostics(plan)
        plan['remediation_options'] = self._generate_remediations(plan)
        plan['estimated_resolution_time'] = self._estimate_resolution_time(plan)

        logger.info(f"Created incident response plan: {plan}")
        return plan

    async def _generate_diagnostics(self, plan: Dict) -> List[Dict]:
        """Generate automated diagnostic steps"""
        diagnostics = []

        # Add standard diagnostic steps based on symptoms
        for symptom in plan.get('symptoms', []):
            if 'cpu' in symptom.lower():
                diagnostics.append({
                    'type': 'metrics',
                    'query': 'cpu_percent',
                    'timeframe': '1h',
                    'threshold': 80
                })
            if 'memory' in symptom.lower():
                diagnostics.append({
                    'type': 'metrics',
                    'query': 'memory_percent',
                    'timeframe': '1h',
                    'threshold': 85
                })
            if 'error' in symptom.lower():
                diagnostics.append({
                    'type': 'logs',
                    'query': 'severity == "Error"',
                    'timeframe': '1h',
                    'analyze': 'error_patterns'
                })

        # Add correlation analysis
        diagnostics.append({
            'type': 'correlation',
            'description': 'Correlate metrics with recent changes',
            'lookback': '24h'
        })

        return diagnostics

    def _generate_remediations(self, plan: Dict) -> List[Dict]:
        """Generate remediation options based on diagnosis"""
        remediations = []

        # Match symptoms to knowledge base
        for hypothesis in plan.get('hypotheses', []):
            for issue_type, issue_data in self.knowledge_base.items():
                if any(cause in hypothesis.lower() for cause in issue_data['causes']):
                    for solution in issue_data['solutions']:
                        remediations.append({
                            'action': solution,
                            'risk_level': self._assess_risk(solution),
                            'automated': self._can_automate(solution),
                            'estimated_time': self._estimate_action_time(solution)
                        })

        # Remove duplicates and sort by risk
        seen = set()
        unique_remediations = []
        for r in remediations:
            key = r['action']
            if key not in seen:
                seen.add(key)
                unique_remediations.append(r)

        return sorted(unique_remediations, key=lambda x: x['risk_level'])

    def _assess_risk(self, action: str) -> str:
        """Assess risk level of a remediation action"""
        high_risk_keywords = ['delete', 'remove', 'terminate', 'shutdown']
        medium_risk_keywords = ['restart', 'reboot', 'scale', 'modify']

        action_lower = action.lower()

        if any(keyword in action_lower for keyword in high_risk_keywords):
            return 'high'
        elif any(keyword in action_lower for keyword in medium_risk_keywords):
            return 'medium'
        else:
            return 'low'

    def _can_automate(self, action: str) -> bool:
        """Determine if an action can be automated"""
        automatable_actions = [
            'restart', 'scale', 'clear cache', 'optimize',
            'increase', 'implement', 'check', 'analyze'
        ]
        return any(keyword in action.lower() for keyword in automatable_actions)

    def _estimate_action_time(self, action: str) -> str:
        """Estimate time to complete an action"""
        if 'restart' in action.lower():
            return '2-5 minutes'
        elif 'scale' in action.lower():
            return '5-10 minutes'
        elif 'optimize' in action.lower():
            return '10-30 minutes'
        else:
            return '5-15 minutes'

    def _estimate_resolution_time(self, plan: Dict) -> str:
        """Estimate total resolution time"""
        severity = plan.get('severity', 'medium')

        if severity == 'critical':
            return '15-30 minutes'
        elif severity == 'high':
            return '30-60 minutes'
        elif severity == 'medium':
            return '1-2 hours'
        else:
            return '2-4 hours'

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the incident response plan

        Args:
            plan: Incident response plan

        Returns:
            Execution result with diagnosis and remediation
        """
        if not plan:
            return {
                'status': 'failed',
                'message': 'No incident response plan provided'
            }

        result = {
            'status': 'diagnosing',
            'start_time': datetime.now().isoformat(),
            'diagnostics_completed': [],
            'root_cause': None,
            'remediation_applied': [],
            'resolution_status': 'pending',
            'incident_id': plan.get('incident_id', f"incident-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        }

        try:
            # Phase 1: Run diagnostics with timeout
            logger.info("Starting incident diagnosis")
            try:
                diagnosis = await asyncio.wait_for(
                    self._run_diagnostics(plan),
                    timeout=120.0  # 2 minutes for diagnostics
                )
                result['diagnostics_completed'] = diagnosis['completed']
                result['root_cause'] = diagnosis['root_cause']
                result['findings'] = diagnosis.get('findings', [])
            except asyncio.TimeoutError:
                logger.error("Incident diagnosis timed out")
                result['status'] = 'failed'
                result['error'] = 'Diagnosis phase timed out'
                return result
            except Exception as diag_error:
                logger.error(f"Diagnosis failed: {str(diag_error)}")
                result['status'] = 'failed'
                result['error'] = f'Diagnosis failed: {str(diag_error)}'
                return result

            # Phase 2: Apply remediation if enabled
            logger.info(f"Root cause identified: {diagnosis['root_cause']}")
            result['status'] = 'remediating'

            if plan.get('auto_remediate', True) and plan.get('remediation_options'):
                try:
                    remediation = await asyncio.wait_for(
                        self._apply_remediation(
                            plan['remediation_options'],
                            diagnosis['root_cause']
                        ),
                        timeout=300.0  # 5 minutes for remediation
                    )
                    result['remediation_applied'] = remediation['actions']
                    result['resolution_status'] = remediation['status']
                except asyncio.TimeoutError:
                    logger.error("Remediation timed out")
                    result['resolution_status'] = 'timeout'
                    result['remediation_error'] = 'Remediation timed out'
                except Exception as remediation_error:
                    logger.error(f"Remediation failed: {str(remediation_error)}")
                    result['resolution_status'] = 'failed'
                    result['remediation_error'] = str(remediation_error)
            else:
                result['resolution_status'] = 'manual_intervention_required'
                logger.info("Auto-remediation disabled or no remediation options available")

            # Phase 3: Verify resolution if remediation was successful
            if result['resolution_status'] == 'success':
                try:
                    verification = await asyncio.wait_for(
                        self._verify_resolution(plan),
                        timeout=60.0  # 1 minute for verification
                    )
                    result['verification'] = verification

                    # Update final status based on verification
                    if not verification.get('fully_resolved', False):
                        result['resolution_status'] = 'partially_resolved'
                        logger.warning("Incident not fully resolved according to verification")
                except asyncio.TimeoutError:
                    logger.warning("Resolution verification timed out")
                    result['verification_error'] = 'Verification timed out'
                except Exception as verify_error:
                    logger.warning(f"Verification failed: {str(verify_error)}")
                    result['verification_error'] = str(verify_error)

            result['status'] = 'completed'
            result['end_time'] = datetime.now().isoformat()

            # Calculate total execution time
            start_time = datetime.fromisoformat(result['start_time'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(result['end_time'].replace('Z', '+00:00'))
            result['total_execution_time'] = str(end_time - start_time)

        except Exception as e:
            logger.error(f"Incident response failed: {str(e)}", exc_info=True)
            result['status'] = 'failed'
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            result['end_time'] = datetime.now().isoformat()

        return result

    async def _run_diagnostics(self, plan: Dict) -> Dict:
        """Run diagnostic procedures"""
        completed = []
        findings = []

        for diagnostic in plan.get('automated_diagnostics', []):
            if diagnostic['type'] == 'metrics':
                metric_result = await self._query_metrics(diagnostic)
                completed.append({
                    'type': 'metrics',
                    'query': diagnostic['query'],
                    'result': metric_result
                })
                findings.extend(metric_result.get('anomalies', []))

            elif diagnostic['type'] == 'logs':
                log_result = await self._query_logs(diagnostic)
                completed.append({
                    'type': 'logs',
                    'query': diagnostic['query'],
                    'result': log_result
                })
                findings.extend(log_result.get('errors', []))

            elif diagnostic['type'] == 'correlation':
                correlation_result = await self._run_correlation_analysis(plan)
                completed.append({
                    'type': 'correlation',
                    'result': correlation_result
                })
                findings.append(correlation_result)

        # Analyze findings to determine root cause
        root_cause = await self._analyze_findings(findings)

        return {
            'completed': completed,
            'findings': findings,
            'root_cause': root_cause
        }

    async def _query_metrics(self, diagnostic: Dict) -> Dict:
        """Query Azure Monitor metrics"""
        # Simulated metric query result
        return {
            'metric': diagnostic['query'],
            'current_value': 85.5,
            'average': 45.2,
            'max': 95.3,
            'anomalies': ['Sustained high usage for 30 minutes']
        }

    async def _query_logs(self, diagnostic: Dict) -> Dict:
        """Query Azure Monitor logs"""
        # Simulated log query result
        return {
            'query': diagnostic['query'],
            'count': 127,
            'errors': [
                'OutOfMemoryException in app.exe',
                'Connection timeout to database'
            ],
            'patterns': ['Error spike at 14:30 UTC']
        }

    async def _run_correlation_analysis(self, plan: Dict) -> Dict:
        """Run correlation analysis between metrics and events"""
        # Simulated correlation analysis
        return {
            'correlated_events': [
                {
                    'time': '14:25 UTC',
                    'event': 'New deployment completed',
                    'correlation': 0.92
                }
            ],
            'pattern_detected': 'sudden_spike',
            'confidence': 0.85
        }

    async def _analyze_findings(self, findings: List) -> str:
        """Analyze diagnostic findings to determine root cause"""
        # Use AI to analyze findings
        prompt = f"""
        Analyze these diagnostic findings and determine the root cause:

        Findings:
        {json.dumps(findings, indent=2)}

        Provide a concise root cause statement.
        """

        response = await self.llm.ainvoke(prompt)
        return response.content

    async def _apply_remediation(
        self,
        remediation_options: List[Dict],
        root_cause: str
    ) -> Dict:
        """Apply remediation actions"""
        applied_actions = []

        # Select best remediation based on root cause
        for option in remediation_options:
            if option['risk_level'] == 'low' and option['automated']:
                logger.info(f"Applying remediation: {option['action']}")

                # Execute remediation
                action_result = await self._execute_remediation_action(option)
                applied_actions.append({
                    'action': option['action'],
                    'status': action_result['status'],
                    'timestamp': datetime.now().isoformat()
                })

                if action_result['status'] == 'success':
                    break

        return {
            'actions': applied_actions,
            'status': 'success' if applied_actions else 'no_action_taken'
        }

    async def _execute_remediation_action(self, action: Dict) -> Dict:
        """Execute a specific remediation action"""
        # Simulated remediation execution
        await asyncio.sleep(2)  # Simulate action execution

        return {
            'action': action['action'],
            'status': 'success',
            'details': f"Successfully executed: {action['action']}"
        }

    async def _verify_resolution(self, plan: Dict) -> Dict:
        """Verify that the incident has been resolved"""
        # Re-run key diagnostics to verify
        verification_checks = []

        for symptom in plan.get('symptoms', []):
            check_result = await self._verify_symptom_resolved(symptom)
            verification_checks.append({
                'symptom': symptom,
                'resolved': check_result
            })

        all_resolved = all(check['resolved'] for check in verification_checks)

        return {
            'checks': verification_checks,
            'fully_resolved': all_resolved,
            'timestamp': datetime.now().isoformat()
        }

    async def _verify_symptom_resolved(self, symptom: str) -> bool:
        """Verify if a specific symptom has been resolved"""
        # Simulated verification
        await asyncio.sleep(1)
        return True  # In production, actually re-check the symptom

    async def analyze_incident_pattern(
        self,
        incidents: List[Dict]
    ) -> Dict:
        """Analyze patterns across multiple incidents"""
        patterns = {
            'recurring_issues': [],
            'common_root_causes': [],
            'peak_times': [],
            'affected_services': []
        }

        # Analyze incident data for patterns
        # This would use ML in production

        return patterns

    async def predict_incidents(
        self,
        metrics_history: Dict,
        timeframe: str = '7d'
    ) -> List[Dict]:
        """Predict potential incidents based on current trends"""
        predictions = []

        # Analyze trends and predict issues
        # This would use ML models in production

        return predictions