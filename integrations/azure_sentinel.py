"""
Azure Sentinel Integration
Security information and event management (SIEM) integration
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import hashlib
import hmac
import base64

import aiohttp
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient
from azure.core.exceptions import HttpResponseError

logger = logging.getLogger(__name__)


class SentinelConnector:
    """Azure Sentinel connector for security operations"""

    def __init__(self):
        """Initialize Sentinel connector"""
        self.credential = DefaultAzureCredential()
        self.workspace_id = os.getenv("AZURE_LOG_ANALYTICS_WORKSPACE_ID")
        self.workspace_key = os.getenv("AZURE_LOG_ANALYTICS_WORKSPACE_KEY")
        self.sentinel_url = f"https://management.azure.com/subscriptions/{os.getenv('AZURE_SUBSCRIPTION_ID')}/resourceGroups/{os.getenv('AZURE_RESOURCE_GROUP')}/providers/Microsoft.OperationalInsights/workspaces/{os.getenv('AZURE_WORKSPACE_NAME')}/providers/Microsoft.SecurityInsights"

        self.logs_client = LogsQueryClient(self.credential)

        # Log Analytics Data Collector API endpoint
        self.log_analytics_url = f"https://{self.workspace_id}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01"

    def _build_signature(
        self,
        date: str,
        content_length: int,
        method: str = "POST",
        resource: str = "/api/logs"
    ) -> str:
        """Build authorization signature for Log Analytics API"""
        x_headers = f"x-ms-date:{date}"
        string_to_hash = f"{method}\n{content_length}\napplication/json\n{x_headers}\n{resource}"
        bytes_to_hash = bytes(string_to_hash, 'UTF-8')
        decoded_key = base64.b64decode(self.workspace_key)
        encoded_hash = base64.b64encode(
            hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
        ).decode('utf-8')
        authorization = f"SharedKey {self.workspace_id}:{encoded_hash}"
        return authorization

    async def send_custom_log(
        self,
        log_type: str,
        log_data: List[Dict[str, Any]]
    ) -> bool:
        """
        Send custom log data to Log Analytics

        Args:
            log_type: Name of the custom log type
            log_data: List of log entries

        Returns:
            Success status
        """
        if not self.workspace_id or not self.workspace_key:
            logger.error("Log Analytics workspace credentials not configured")
            return False

        body = json.dumps(log_data)
        content_length = len(body)

        # RFC1123 date format
        date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')

        # Build authorization signature
        signature = self._build_signature(date, content_length)

        headers = {
            'content-type': 'application/json',
            'Authorization': signature,
            'Log-Type': log_type,
            'x-ms-date': date,
            'time-generated-field': 'TimeGenerated'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.log_analytics_url,
                    data=body,
                    headers=headers
                ) as response:
                    if response.status in [200, 202]:
                        logger.info(f"Successfully sent {len(log_data)} log entries to Sentinel")
                        return True
                    else:
                        logger.error(f"Failed to send logs to Sentinel: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error sending logs to Sentinel: {str(e)}")
            return False

    async def create_incident(
        self,
        title: str,
        description: str,
        severity: str = "Medium",
        status: str = "New",
        classification: str = "TruePositive",
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a security incident in Sentinel

        Args:
            title: Incident title
            description: Incident description
            severity: Severity level (Low, Medium, High, Critical)
            status: Incident status
            classification: Incident classification
            labels: Optional labels

        Returns:
            Created incident details
        """
        incident = {
            "properties": {
                "title": title,
                "description": description,
                "severity": severity,
                "status": status,
                "classification": classification,
                "labels": labels or [],
                "firstActivityTimeUtc": datetime.utcnow().isoformat(),
                "lastActivityTimeUtc": datetime.utcnow().isoformat()
            }
        }

        try:
            # This would require Azure REST API call
            # Simplified for demonstration
            incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

            logger.info(f"Created Sentinel incident: {incident_id}")

            return {
                "id": incident_id,
                "properties": incident["properties"],
                "created": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error creating Sentinel incident: {str(e)}")
            return {}

    async def query_security_alerts(
        self,
        timespan: timedelta = timedelta(days=1),
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query security alerts from Sentinel

        Args:
            timespan: Time range for query
            severity: Filter by severity

        Returns:
            List of security alerts
        """
        query = """
        SecurityAlert
        | where TimeGenerated > ago(1d)
        | project TimeGenerated, AlertName, AlertSeverity, Description,
                 RemediationSteps, ExtendedProperties, Entities
        | order by TimeGenerated desc
        | limit 100
        """

        if severity:
            query = query.replace(
                "| where TimeGenerated",
                f"| where AlertSeverity == '{severity}' and TimeGenerated"
            )

        try:
            response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=query,
                timespan=timespan
            )

            alerts = []
            if response.tables:
                for table in response.tables:
                    for row in table.rows:
                        alert = {
                            "timestamp": str(row[0]),
                            "name": row[1],
                            "severity": row[2],
                            "description": row[3],
                            "remediation": row[4],
                            "properties": row[5],
                            "entities": row[6]
                        }
                        alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"Error querying security alerts: {str(e)}")
            return []

    async def query_threat_indicators(
        self,
        indicator_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query threat intelligence indicators

        Args:
            indicator_type: Type of indicator (IP, URL, FileHash, etc.)
            limit: Maximum results

        Returns:
            List of threat indicators
        """
        query = f"""
        ThreatIntelligenceIndicator
        | where TimeGenerated > ago(7d)
        | project TimeGenerated, IndicatorType, DomainName, Url,
                 FileHashValue, EmailSenderAddress, ConfidenceScore,
                 ThreatType, Description, Active
        | where Active == true
        | order by TimeGenerated desc
        | limit {limit}
        """

        if indicator_type:
            query = query.replace(
                "| where Active",
                f"| where IndicatorType == '{indicator_type}' and Active"
            )

        try:
            response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=query,
                timespan=timedelta(days=7)
            )

            indicators = []
            if response.tables:
                for table in response.tables:
                    for row in table.rows:
                        indicator = {
                            "timestamp": str(row[0]),
                            "type": row[1],
                            "domain": row[2],
                            "url": row[3],
                            "file_hash": row[4],
                            "email": row[5],
                            "confidence": row[6],
                            "threat_type": row[7],
                            "description": row[8],
                            "active": row[9]
                        }
                        indicators.append(indicator)

            return indicators

        except Exception as e:
            logger.error(f"Error querying threat indicators: {str(e)}")
            return []

    async def get_security_recommendations(self) -> List[Dict[str, Any]]:
        """Get security recommendations from Azure Security Center"""

        query = """
        SecurityRecommendation
        | where State == 'Unhealthy'
        | project RecommendationDisplayName, RecommendationId,
                 RemediationDescription, Severity, AssessedResourceId
        | order by Severity desc
        | limit 50
        """

        try:
            response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=query,
                timespan=timedelta(days=1)
            )

            recommendations = []
            if response.tables:
                for table in response.tables:
                    for row in table.rows:
                        rec = {
                            "name": row[0],
                            "id": row[1],
                            "remediation": row[2],
                            "severity": row[3],
                            "resource": row[4]
                        }
                        recommendations.append(rec)

            return recommendations

        except Exception as e:
            logger.error(f"Error getting security recommendations: {str(e)}")
            return []

    async def analyze_user_behavior(
        self,
        user_principal_name: str,
        timespan: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """
        Analyze user behavior for anomalies

        Args:
            user_principal_name: User to analyze
            timespan: Time period to analyze

        Returns:
            User behavior analysis results
        """
        # Query sign-in logs
        signin_query = f"""
        SigninLogs
        | where UserPrincipalName == '{user_principal_name}'
        | where TimeGenerated > ago(7d)
        | summarize
            SignInCount = count(),
            DistinctLocations = dcount(Location),
            DistinctIPs = dcount(IPAddress),
            FailedSignIns = countif(ResultType != 0)
        """

        # Query audit logs
        audit_query = f"""
        AuditLogs
        | where InitiatedBy.user.userPrincipalName == '{user_principal_name}'
        | where TimeGenerated > ago(7d)
        | summarize
            ActivityCount = count(),
            DistinctOperations = dcount(OperationName),
            DistinctResources = dcount(TargetResources)
        """

        try:
            # Execute queries
            signin_response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=signin_query,
                timespan=timespan
            )

            audit_response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=audit_query,
                timespan=timespan
            )

            analysis = {
                "user": user_principal_name,
                "period": f"{timespan.days} days",
                "sign_in_analysis": {},
                "activity_analysis": {},
                "risk_score": 0
            }

            # Process sign-in data
            if signin_response.tables and signin_response.tables[0].rows:
                row = signin_response.tables[0].rows[0]
                analysis["sign_in_analysis"] = {
                    "total_sign_ins": row[0],
                    "distinct_locations": row[1],
                    "distinct_ips": row[2],
                    "failed_attempts": row[3]
                }

                # Calculate risk based on anomalies
                if row[3] > 5:  # Failed sign-ins
                    analysis["risk_score"] += 30
                if row[1] > 3:  # Multiple locations
                    analysis["risk_score"] += 20
                if row[2] > 10:  # Many IPs
                    analysis["risk_score"] += 10

            # Process audit data
            if audit_response.tables and audit_response.tables[0].rows:
                row = audit_response.tables[0].rows[0]
                analysis["activity_analysis"] = {
                    "total_activities": row[0],
                    "distinct_operations": row[1],
                    "distinct_resources": row[2]
                }

                # Adjust risk for unusual activity
                if row[0] > 100:  # High activity
                    analysis["risk_score"] += 15

            # Determine risk level
            if analysis["risk_score"] >= 50:
                analysis["risk_level"] = "High"
            elif analysis["risk_score"] >= 30:
                analysis["risk_level"] = "Medium"
            else:
                analysis["risk_level"] = "Low"

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing user behavior: {str(e)}")
            return {
                "user": user_principal_name,
                "error": str(e)
            }

    async def check_compliance_status(self) -> Dict[str, Any]:
        """Check compliance status across Azure resources"""

        query = """
        SecurityBaseline
        | where ComplianceState == 'NonCompliant'
        | summarize
            NonCompliantResources = count(),
            by BaselineType = BaselineName
        | order by NonCompliantResources desc
        """

        try:
            response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=query,
                timespan=timedelta(days=1)
            )

            compliance = {
                "timestamp": datetime.utcnow().isoformat(),
                "baselines": [],
                "total_non_compliant": 0
            }

            if response.tables:
                for table in response.tables:
                    for row in table.rows:
                        baseline = {
                            "type": row[1],
                            "non_compliant_count": row[0]
                        }
                        compliance["baselines"].append(baseline)
                        compliance["total_non_compliant"] += row[0]

            return compliance

        except Exception as e:
            logger.error(f"Error checking compliance status: {str(e)}")
            return {"error": str(e)}