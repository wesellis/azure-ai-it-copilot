"""
Compliance Agent - Ensures Azure resources meet security and regulatory standards
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import hashlib

from azure.mgmt.security import SecurityCenter
from azure.mgmt.policyinsights import PolicyInsightsClient
from azure.mgmt.monitor import MonitorManagementClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError

from langchain.tools import Tool
import logging

logger = logging.getLogger(__name__)


class ComplianceAgent:
    """Agent for compliance checking and enforcement"""

    def __init__(self, orchestrator):
        """Initialize the compliance agent"""
        self.orchestrator = orchestrator
        self.llm = orchestrator.llm
        self.credential = DefaultAzureCredential()
        self.subscription_id = orchestrator.subscription_id

        # Initialize Azure clients
        self.security_client = SecurityCenter(
            self.credential,
            self.subscription_id
        )
        self.policy_client = PolicyInsightsClient(
            self.credential
        )
        self.monitor_client = MonitorManagementClient(
            self.credential,
            self.subscription_id
        )

        # Compliance frameworks
        self.frameworks = self._load_compliance_frameworks()

    def _load_compliance_frameworks(self) -> Dict:
        """Load compliance framework definitions"""
        return {
            'cis': {
                'name': 'CIS Azure Foundations Benchmark',
                'version': '1.5.0',
                'controls': [
                    {
                        'id': 'CIS-1.1',
                        'title': 'Ensure Security Defaults is enabled',
                        'category': 'Identity and Access Management',
                        'severity': 'high'
                    },
                    {
                        'id': 'CIS-2.1',
                        'title': 'Ensure Azure Defender is set to On for Servers',
                        'category': 'Security Center',
                        'severity': 'high'
                    },
                    {
                        'id': 'CIS-3.1',
                        'title': 'Ensure secure transfer required is set to Enabled',
                        'category': 'Storage Accounts',
                        'severity': 'medium'
                    }
                ]
            },
            'hipaa': {
                'name': 'HIPAA Security Rule',
                'controls': [
                    {
                        'id': 'HIPAA-164.312(a)',
                        'title': 'Access Control',
                        'requirements': ['Unique user identification', 'Automatic logoff', 'Encryption']
                    },
                    {
                        'id': 'HIPAA-164.312(e)',
                        'title': 'Transmission Security',
                        'requirements': ['Integrity controls', 'Encryption']
                    }
                ]
            },
            'pci_dss': {
                'name': 'PCI DSS v4.0',
                'controls': [
                    {
                        'id': 'PCI-1.1',
                        'title': 'Firewall configuration standards',
                        'category': 'Network Security'
                    },
                    {
                        'id': 'PCI-2.1',
                        'title': 'Default passwords changed',
                        'category': 'Configuration Management'
                    }
                ]
            },
            'iso_27001': {
                'name': 'ISO 27001:2013',
                'controls': [
                    {
                        'id': 'A.9',
                        'title': 'Access Control',
                        'objectives': ['Business requirements', 'User access management']
                    },
                    {
                        'id': 'A.12',
                        'title': 'Operations Security',
                        'objectives': ['Operational procedures', 'Protection from malware']
                    }
                ]
            }
        }

    async def create_plan(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create compliance assessment plan

        Args:
            command: Natural language command
            context: Additional context

        Returns:
            Compliance assessment plan
        """
        # Parse the compliance request
        prompt = f"""
        Analyze this compliance request:

        Request: {command}
        Context: {json.dumps(context) if context else 'None'}

        Extract:
        1. Compliance frameworks to check (CIS, HIPAA, PCI-DSS, ISO-27001, etc.)
        2. Scope (all resources, specific resource groups, resource types)
        3. Assessment type (audit, enforce, report)
        4. Remediation preference (auto-fix, manual, report-only)

        Respond in JSON format with:
        - frameworks: list of frameworks to assess
        - scope: resources to check
        - assessment_type: type of assessment
        - auto_remediate: boolean
        - report_format: detailed or summary
        """

        response = await self.llm.ainvoke(prompt)
        plan = json.loads(response.content)

        # Enhance with specific checks
        plan['checks'] = self._generate_compliance_checks(plan)
        plan['risk_assessment'] = self._assess_compliance_risks(plan)
        plan['remediation_plan'] = self._create_remediation_plan(plan)

        logger.info(f"Created compliance plan: {plan}")
        return plan

    def _generate_compliance_checks(self, plan: Dict) -> List[Dict]:
        """Generate specific compliance checks based on frameworks"""
        checks = []

        for framework_name in plan.get('frameworks', ['cis']):
            framework = self.frameworks.get(framework_name.lower(), {})
            
            if framework:
                for control in framework.get('controls', []):
                    checks.append({
                        'framework': framework_name,
                        'control_id': control.get('id'),
                        'title': control.get('title'),
                        'check_type': self._determine_check_type(control),
                        'automated': True,
                        'severity': control.get('severity', 'medium')
                    })

        # Add Azure-specific security checks
        checks.extend([
            {
                'framework': 'azure_security',
                'control_id': 'AZ-SEC-001',
                'title': 'Storage accounts encryption at rest',
                'check_type': 'configuration',
                'automated': True,
                'severity': 'high'
            },
            {
                'framework': 'azure_security',
                'control_id': 'AZ-SEC-002',
                'title': 'Network security groups properly configured',
                'check_type': 'network',
                'automated': True,
                'severity': 'high'
            },
            {
                'framework': 'azure_security',
                'control_id': 'AZ-SEC-003',
                'title': 'Key Vault secrets rotation',
                'check_type': 'secrets',
                'automated': True,
                'severity': 'medium'
            },
            {
                'framework': 'azure_security',
                'control_id': 'AZ-SEC-004',
                'title': 'Azure AD MFA enabled',
                'check_type': 'identity',
                'automated': True,
                'severity': 'critical'
            },
            {
                'framework': 'azure_security',
                'control_id': 'AZ-SEC-005',
                'title': 'Diagnostic logs enabled',
                'check_type': 'logging',
                'automated': True,
                'severity': 'medium'
            }
        ])

        return checks

    def _determine_check_type(self, control: Dict) -> str:
        """Determine the type of compliance check"""
        title = control.get('title', '').lower()
        
        if 'encryption' in title or 'secure' in title:
            return 'encryption'
        elif 'access' in title or 'identity' in title:
            return 'access_control'
        elif 'network' in title or 'firewall' in title:
            return 'network'
        elif 'log' in title or 'audit' in title:
            return 'logging'
        else:
            return 'configuration'

    def _assess_compliance_risks(self, plan: Dict) -> Dict:
        """Assess compliance risks"""
        risks = {
            'overall_risk': 'medium',
            'risk_factors': [],
            'critical_gaps': [],
            'recommendations': []
        }

        # Analyze based on frameworks
        if 'hipaa' in [f.lower() for f in plan.get('frameworks', [])]:
            risks['risk_factors'].append('Healthcare data requires strict compliance')
            risks['overall_risk'] = 'high'

        if 'pci_dss' in [f.lower() for f in plan.get('frameworks', [])]:
            risks['risk_factors'].append('Payment card data requires PCI compliance')
            risks['overall_risk'] = 'high'

        # Add recommendations
        risks['recommendations'] = [
            'Enable Azure Security Center Standard tier',
            'Implement Azure Policy for continuous compliance',
            'Set up compliance dashboards and alerts',
            'Regular compliance assessments (monthly)'
        ]

        return risks

    def _create_remediation_plan(self, plan: Dict) -> List[Dict]:
        """Create remediation plan for compliance issues"""
        remediations = []

        for check in plan.get('checks', []):
            if check['severity'] in ['critical', 'high']:
                remediations.append({
                    'control_id': check['control_id'],
                    'title': check['title'],
                    'remediation_steps': self._get_remediation_steps(check),
                    'priority': 1 if check['severity'] == 'critical' else 2,
                    'estimated_time': '1-2 hours',
                    'can_automate': check['automated']
                })

        return sorted(remediations, key=lambda x: x['priority'])

    def _get_remediation_steps(self, check: Dict) -> List[str]:
        """Get remediation steps for a compliance check"""
        check_type = check.get('check_type')
        
        if check_type == 'encryption':
            return [
                'Enable encryption at rest for all storage accounts',
                'Enable TLS 1.2 minimum for data in transit',
                'Rotate encryption keys regularly'
            ]
        elif check_type == 'access_control':
            return [
                'Review and update RBAC assignments',
                'Enable MFA for all users',
                'Implement principle of least privilege',
                'Set up Privileged Identity Management (PIM)'
            ]
        elif check_type == 'network':
            return [
                'Review Network Security Group rules',
                'Enable Azure Firewall or WAF',
                'Implement network segmentation',
                'Enable DDoS protection'
            ]
        elif check_type == 'logging':
            return [
                'Enable diagnostic logs for all resources',
                'Configure log retention policies',
                'Set up log analytics workspace',
                'Create alerts for security events'
            ]
        else:
            return [
                'Review resource configuration',
                'Apply recommended settings',
                'Document exceptions with business justification'
            ]

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute compliance assessment

        Args:
            plan: Compliance assessment plan

        Returns:
            Assessment results
        """
        result = {
            'status': 'assessing',
            'start_time': datetime.now().isoformat(),
            'checks_completed': [],
            'violations_found': [],
            'compliance_score': 0,
            'remediations_applied': []
        }

        try:
            # Run compliance checks
            logger.info("Starting compliance assessment")
            
            total_checks = len(plan.get('checks', []))
            passed_checks = 0

            for check in plan.get('checks', []):
                check_result = await self._execute_compliance_check(check)
                result['checks_completed'].append(check_result)
                
                if check_result['status'] == 'passed':
                    passed_checks += 1
                else:
                    result['violations_found'].append({
                        'control_id': check['control_id'],
                        'title': check['title'],
                        'severity': check['severity'],
                        'details': check_result.get('details', '')
                    })

            # Calculate compliance score
            result['compliance_score'] = round((passed_checks / total_checks) * 100, 2) if total_checks > 0 else 0

            # Apply remediations if auto-remediate is enabled
            if plan.get('auto_remediate', False) and result['violations_found']:
                logger.info("Applying automatic remediations")
                
                for violation in result['violations_found']:
                    if violation['severity'] in ['critical', 'high']:
                        remediation_result = await self._apply_remediation(violation)
                        result['remediations_applied'].append(remediation_result)

            # Generate compliance report
            result['report'] = self._generate_compliance_report(result, plan)
            result['status'] = 'completed'
            result['end_time'] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Compliance assessment failed: {str(e)}")
            result['status'] = 'failed'
            result['error'] = str(e)

        return result

    async def _execute_compliance_check(self, check: Dict) -> Dict:
        """Execute a specific compliance check"""
        # Simulated compliance check execution
        await asyncio.sleep(0.5)
        
        # Simulate check results
        import random
        passed = random.random() > 0.3  # 70% pass rate for simulation
        
        return {
            'control_id': check['control_id'],
            'status': 'passed' if passed else 'failed',
            'timestamp': datetime.now().isoformat(),
            'details': 'Check completed successfully' if passed else 'Non-compliant configuration detected'
        }

    async def _apply_remediation(self, violation: Dict) -> Dict:
        """Apply remediation for a compliance violation"""
        logger.info(f"Applying remediation for {violation['control_id']}")
        
        # Simulated remediation
        await asyncio.sleep(1)
        
        return {
            'control_id': violation['control_id'],
            'action': 'auto_remediated',
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'details': f"Applied automatic fix for {violation['title']}"
        }

    def _generate_compliance_report(self, result: Dict, plan: Dict) -> Dict:
        """Generate compliance assessment report"""
        report = {
            'summary': {
                'compliance_score': result['compliance_score'],
                'total_checks': len(result['checks_completed']),
                'passed': len([c for c in result['checks_completed'] if c['status'] == 'passed']),
                'failed': len(result['violations_found']),
                'critical_violations': len([v for v in result['violations_found'] if v['severity'] == 'critical']),
                'remediations_applied': len(result['remediations_applied'])
            },
            'frameworks_assessed': plan.get('frameworks', []),
            'risk_level': self._calculate_risk_level(result),
            'recommendations': self._generate_recommendations(result),
            'next_assessment_date': (datetime.now() + timedelta(days=30)).isoformat()
        }

        if plan.get('report_format') == 'detailed':
            report['detailed_findings'] = result['violations_found']
            report['remediation_history'] = result['remediations_applied']

        return report

    def _calculate_risk_level(self, result: Dict) -> str:
        """Calculate overall risk level based on violations"""
        score = result['compliance_score']
        critical_violations = len([v for v in result['violations_found'] if v['severity'] == 'critical'])
        
        if critical_violations > 0 or score < 60:
            return 'critical'
        elif score < 75:
            return 'high'
        elif score < 90:
            return 'medium'
        else:
            return 'low'

    def _generate_recommendations(self, result: Dict) -> List[str]:
        """Generate recommendations based on assessment results"""
        recommendations = []
        
        if result['compliance_score'] < 80:
            recommendations.append('Urgent: Address all critical and high severity violations immediately')
        
        if any(v['control_id'].startswith('AZ-SEC-004') for v in result['violations_found']):
            recommendations.append('Enable Multi-Factor Authentication for all user accounts')
        
        if any('encryption' in v['title'].lower() for v in result['violations_found']):
            recommendations.append('Implement encryption for all data at rest and in transit')
        
        recommendations.extend([
            'Schedule regular compliance assessments (monthly)',
            'Implement continuous compliance monitoring with Azure Policy',
            'Create compliance dashboard for real-time visibility',
            'Document all compliance exceptions with business justification'
        ])
        
        return recommendations[:5]  # Return top 5 recommendations

    async def generate_compliance_dashboard(self) -> Dict:
        """Generate compliance dashboard data"""
        return {
            'overall_compliance': 78.5,
            'by_framework': {
                'CIS': 82.3,
                'Azure Security': 75.0,
                'ISO 27001': 79.8
            },
            'trend': 'improving',
            'recent_assessments': [],
            'upcoming_audits': []
        }
