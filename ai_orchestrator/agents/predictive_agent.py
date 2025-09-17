"""
Predictive Agent - AI-powered predictive maintenance and forecasting
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# Optional ML imports - may not be available in all environments
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestRegressor = None
    IsolationForest = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

from azure.monitor.query import LogsQueryClient
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


class PredictiveAgent(BaseAgent):
    """Agent for predictive analytics and maintenance"""

    def __init__(self, orchestrator):
        """Initialize the predictive agent"""
        super().__init__(orchestrator)
        self.credential = DefaultAzureCredential()

        # Use orchestrator's Azure Monitor clients
        self.logs_client = orchestrator.logs_client
        # Note: metrics_client not available in current Azure SDK version
        self.metrics_client = None
        
        # Initialize ML models
        self.models = self._initialize_models()
        
        # Prediction thresholds
        self.thresholds = {
            'cpu_warning': 75,
            'cpu_critical': 90,
            'memory_warning': 80,
            'memory_critical': 95,
            'disk_warning': 85,
            'disk_critical': 95,
            'anomaly_score': 0.7
        }

    def _initialize_models(self) -> Dict:
        """Initialize or load ML models"""
        models = {}
        
        # Initialize models (in production, load pre-trained models)
        models['resource_usage'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        models['anomaly_detection'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        models['capacity_planning'] = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        models['scaler'] = StandardScaler()
        
        return models

    async def create_plan(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create predictive analysis plan

        Args:
            command: Natural language command
            context: Additional context

        Returns:
            Predictive analysis plan
        """
        # Parse the predictive analysis request
        prompt = f"""
        Analyze this predictive analytics request:

        Request: {command}
        Context: {json.dumps(context) if context else 'None'}

        Extract:
        1. Analysis type (failure prediction, capacity planning, anomaly detection, trend analysis)
        2. Target resources (VMs, databases, storage, network, all)
        3. Time horizon (24h, 7d, 30d, 90d)
        4. Alert preferences (immediate, daily digest, weekly report)
        5. Action preferences (auto-scale, alert-only, preventive maintenance)

        Respond in JSON format with:
        - analysis_type: type of predictive analysis
        - resources: list of resource types or IDs
        - time_horizon: prediction timeframe
        - confidence_threshold: minimum confidence for predictions (0-1)
        - enable_alerts: boolean
        - auto_remediate: boolean
        """

        response = await self.llm.ainvoke(prompt)
        plan = json.loads(response.content)

        # Enhance with specific predictions
        plan['predictions'] = await self._generate_predictions(plan)
        plan['risk_assessment'] = self._assess_risks(plan)
        plan['recommendations'] = self._generate_recommendations(plan)
        plan['monitoring_plan'] = self._create_monitoring_plan(plan)

        logger.info(f"Created predictive analysis plan: {plan}")
        return plan

    async def _generate_predictions(self, plan: Dict) -> List[Dict]:
        """Generate predictions based on analysis type"""
        predictions = []
        analysis_type = plan.get('analysis_type', 'general')
        
        if analysis_type in ['failure_prediction', 'general']:
            predictions.extend(await self._predict_failures(plan))
        
        if analysis_type in ['capacity_planning', 'general']:
            predictions.extend(await self._predict_capacity_needs(plan))
        
        if analysis_type in ['anomaly_detection', 'general']:
            predictions.extend(await self._detect_anomalies(plan))
        
        if analysis_type in ['trend_analysis', 'general']:
            predictions.extend(await self._analyze_trends(plan))
        
        return sorted(predictions, key=lambda x: x.get('probability', 0), reverse=True)

    async def _predict_failures(self, plan: Dict) -> List[Dict]:
        """Predict potential failures"""
        # Simulated failure predictions
        failures = [
            {
                'type': 'failure_prediction',
                'resource': 'vm-prod-db-01',
                'component': 'disk',
                'probability': 0.78,
                'timeframe': '48-72 hours',
                'indicators': ['Increasing I/O errors', 'SMART warnings', 'Performance degradation'],
                'impact': 'high',
                'recommended_action': 'Schedule maintenance window for disk replacement'
            },
            {
                'type': 'failure_prediction',
                'resource': 'vm-web-02',
                'component': 'memory',
                'probability': 0.65,
                'timeframe': '5-7 days',
                'indicators': ['Memory leak detected', 'Increasing page faults'],
                'impact': 'medium',
                'recommended_action': 'Restart application service, investigate memory leak'
            },
            {
                'type': 'failure_prediction',
                'resource': 'sql-analytics-01',
                'component': 'performance',
                'probability': 0.82,
                'timeframe': '24-48 hours',
                'indicators': ['Query performance degradation', 'Lock contention increasing'],
                'impact': 'high',
                'recommended_action': 'Optimize queries, rebuild indexes, consider scaling'
            }
        ]
        
        return failures

    async def _predict_capacity_needs(self, plan: Dict) -> List[Dict]:
        """Predict capacity requirements"""
        # Simulated capacity predictions
        capacity_needs = [
            {
                'type': 'capacity_prediction',
                'resource_type': 'compute',
                'current_usage': '72%',
                'predicted_peak': '95%',
                'timeframe': '2 weeks',
                'growth_rate': '3.5% per week',
                'recommended_action': 'Add 2 additional VMs to cluster',
                'probability': 0.89,
                'cost_impact': '+$1,200/month'
            },
            {
                'type': 'capacity_prediction',
                'resource_type': 'storage',
                'current_usage': '68%',
                'predicted_full': '30 days',
                'growth_rate': '50GB/day',
                'recommended_action': 'Expand storage by 2TB',
                'probability': 0.92,
                'cost_impact': '+$200/month'
            },
            {
                'type': 'capacity_prediction',
                'resource_type': 'database_connections',
                'current_usage': '450/500',
                'predicted_exhaustion': '7 days',
                'peak_time': '2PM-4PM daily',
                'recommended_action': 'Increase connection pool size or implement connection pooling',
                'probability': 0.76
            }
        ]
        
        return capacity_needs

    async def _detect_anomalies(self, plan: Dict) -> List[Dict]:
        """Detect anomalies in resource behavior"""
        # Simulated anomaly detection
        anomalies = [
            {
                'type': 'anomaly',
                'resource': 'vm-api-gateway',
                'anomaly_type': 'traffic_spike',
                'severity': 'medium',
                'description': 'Unusual traffic pattern detected - 300% above normal',
                'timestamp': datetime.now().isoformat(),
                'probability': 0.85,
                'possible_causes': ['DDoS attack', 'Legitimate traffic surge', 'Bot activity'],
                'recommended_action': 'Enable rate limiting, investigate traffic sources'
            },
            {
                'type': 'anomaly',
                'resource': 'storage-account-01',
                'anomaly_type': 'access_pattern',
                'severity': 'low',
                'description': 'Unusual access pattern from new IP range',
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'probability': 0.72,
                'possible_causes': ['New application deployment', 'Security breach attempt'],
                'recommended_action': 'Verify IP addresses, review access logs'
            }
        ]
        
        return anomalies

    async def _analyze_trends(self, plan: Dict) -> List[Dict]:
        """Analyze resource usage trends"""
        # Simulated trend analysis
        trends = [
            {
                'type': 'trend',
                'metric': 'overall_cpu_usage',
                'current_trend': 'increasing',
                'rate': '+2.3% per day',
                'projection': 'Will reach 90% in 14 days',
                'confidence': 0.88,
                'seasonal_pattern': 'Higher on weekdays, peaks at month-end',
                'recommended_action': 'Plan for scaling before month-end'
            },
            {
                'type': 'trend',
                'metric': 'api_response_time',
                'current_trend': 'degrading',
                'rate': '+50ms per week',
                'projection': 'Will exceed SLA in 10 days',
                'confidence': 0.79,
                'correlation': 'Correlated with database growth',
                'recommended_action': 'Optimize database queries, implement caching'
            },
            {
                'type': 'trend',
                'metric': 'error_rate',
                'current_trend': 'stable',
                'rate': '0.02%',
                'projection': 'Remains within acceptable limits',
                'confidence': 0.91,
                'pattern': 'Slight increase during deployments',
                'recommended_action': 'Continue monitoring, implement blue-green deployment'
            }
        ]
        
        return trends

    def _assess_risks(self, plan: Dict) -> Dict:
        """Assess risks based on predictions"""
        predictions = plan.get('predictions', [])
        
        high_risk_items = [p for p in predictions if p.get('probability', 0) > 0.75]
        medium_risk_items = [p for p in predictions if 0.5 <= p.get('probability', 0) <= 0.75]
        
        risk_assessment = {
            'overall_risk': 'high' if len(high_risk_items) > 2 else 'medium' if high_risk_items else 'low',
            'high_risk_count': len(high_risk_items),
            'medium_risk_count': len(medium_risk_items),
            'top_risks': high_risk_items[:3],
            'mitigation_priority': self._prioritize_mitigations(high_risk_items)
        }
        
        return risk_assessment

    def _prioritize_mitigations(self, risks: List[Dict]) -> List[Dict]:
        """Prioritize risk mitigations"""
        mitigations = []
        
        for risk in risks:
            mitigation = {
                'resource': risk.get('resource', 'Unknown'),
                'action': risk.get('recommended_action', 'Investigate'),
                'urgency': 'immediate' if risk.get('probability', 0) > 0.85 else 'planned',
                'impact': risk.get('impact', 'medium'),
                'estimated_effort': self._estimate_effort(risk)
            }
            mitigations.append(mitigation)
        
        return sorted(mitigations, key=lambda x: (x['urgency'] == 'immediate', x['impact']), reverse=True)

    def _estimate_effort(self, risk: Dict) -> str:
        """Estimate effort for mitigation"""
        action = risk.get('recommended_action', '').lower()
        
        if 'restart' in action or 'reboot' in action:
            return '15-30 minutes'
        elif 'scale' in action or 'add' in action:
            return '1-2 hours'
        elif 'optimize' in action or 'investigate' in action:
            return '2-4 hours'
        else:
            return '1-3 hours'

    def _generate_recommendations(self, plan: Dict) -> List[str]:
        """Generate recommendations based on predictions"""
        recommendations = []
        predictions = plan.get('predictions', [])
        
        # Check for failure predictions
        failure_predictions = [p for p in predictions if p.get('type') == 'failure_prediction']
        if failure_predictions:
            recommendations.append('Schedule preventive maintenance for high-risk resources')
            recommendations.append('Implement automated backup before predicted failures')
        
        # Check for capacity issues
        capacity_predictions = [p for p in predictions if p.get('type') == 'capacity_prediction']
        if capacity_predictions:
            recommendations.append('Enable auto-scaling for resources approaching capacity')
            recommendations.append('Review and optimize resource allocation')
        
        # Check for anomalies
        anomalies = [p for p in predictions if p.get('type') == 'anomaly']
        if anomalies:
            recommendations.append('Review security alerts and access patterns')
            recommendations.append('Update monitoring thresholds based on detected anomalies')
        
        # General recommendations
        recommendations.extend([
            'Set up automated alerts for predicted issues',
            'Create runbooks for common predicted failures',
            'Schedule regular model retraining with latest data'
        ])
        
        return recommendations[:5]  # Return top 5 recommendations

    def _create_monitoring_plan(self, plan: Dict) -> Dict:
        """Create monitoring plan based on predictions"""
        return {
            'metrics_to_monitor': [
                {'metric': 'cpu_usage', 'threshold': 80, 'window': '5m'},
                {'metric': 'memory_usage', 'threshold': 85, 'window': '5m'},
                {'metric': 'disk_io', 'threshold': 1000, 'window': '10m'},
                {'metric': 'error_rate', 'threshold': 0.05, 'window': '15m'}
            ],
            'alert_rules': [
                {
                    'name': 'predicted_failure_alert',
                    'condition': 'failure_probability > 0.7',
                    'action': 'email,slack',
                    'severity': 'high'
                },
                {
                    'name': 'capacity_warning',
                    'condition': 'usage > 85%',
                    'action': 'email',
                    'severity': 'medium'
                }
            ],
            'review_frequency': 'daily',
            'model_retraining': 'weekly'
        }

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute predictive analysis

        Args:
            plan: Predictive analysis plan

        Returns:
            Analysis results
        """
        result = {
            'status': 'analyzing',
            'start_time': datetime.now().isoformat(),
            'predictions_generated': [],
            'alerts_created': [],
            'actions_taken': [],
            'model_performance': {}
        }

        try:
            # Generate predictions
            logger.info("Generating predictions")
            
            for prediction in plan.get('predictions', []):
                if prediction.get('probability', 0) >= plan.get('confidence_threshold', 0.7):
                    result['predictions_generated'].append({
                        'id': f"pred_{datetime.now().timestamp()}",
                        'type': prediction['type'],
                        'resource': prediction.get('resource', 'general'),
                        'prediction': prediction,
                        'confidence': prediction.get('probability', 0)
                    })
                    
                    # Create alert if enabled
                    if plan.get('enable_alerts', True) and prediction.get('probability', 0) > 0.8:
                        alert = await self._create_alert(prediction)
                        result['alerts_created'].append(alert)
                    
                    # Take action if auto-remediate is enabled
                    if plan.get('auto_remediate', False) and prediction.get('probability', 0) > 0.85:
                        action = await self._take_preventive_action(prediction)
                        result['actions_taken'].append(action)
            
            # Evaluate model performance
            result['model_performance'] = await self._evaluate_models()
            
            # Generate insights
            result['insights'] = self._generate_insights(result)
            
            # Create monitoring dashboard data
            result['dashboard'] = self._create_dashboard_data(result)
            
            result['status'] = 'completed'
            result['end_time'] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Predictive analysis failed: {str(e)}")
            result['status'] = 'failed'
            result['error'] = str(e)

        return result

    async def _create_alert(self, prediction: Dict) -> Dict:
        """Create alert for prediction"""
        alert = {
            'id': f"alert_{datetime.now().timestamp()}",
            'type': prediction.get('type'),
            'severity': 'high' if prediction.get('probability', 0) > 0.85 else 'medium',
            'resource': prediction.get('resource', 'unknown'),
            'message': f"{prediction.get('type', 'Issue')} predicted with {prediction.get('probability', 0)*100:.1f}% confidence",
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"Created alert: {alert['id']}")
        return alert

    async def _take_preventive_action(self, prediction: Dict) -> Dict:
        """Take preventive action based on prediction"""
        action_type = self._determine_action(prediction)
        
        # Simulated action execution
        await asyncio.sleep(2)
        
        action = {
            'id': f"action_{datetime.now().timestamp()}",
            'type': action_type,
            'resource': prediction.get('resource'),
            'reason': f"Preventive action for {prediction.get('type')}",
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Executed preventive action: {action['id']}")
        return action

    def _determine_action(self, prediction: Dict) -> str:
        """Determine appropriate action based on prediction"""
        pred_type = prediction.get('type', '')
        
        if pred_type == 'failure_prediction':
            return 'backup_and_prepare_failover'
        elif pred_type == 'capacity_prediction':
            return 'auto_scale'
        elif pred_type == 'anomaly':
            return 'increase_monitoring'
        else:
            return 'alert_only'

    async def _evaluate_models(self) -> Dict:
        """Evaluate model performance"""
        # Simulated model evaluation
        return {
            'accuracy': 0.87,
            'precision': 0.89,
            'recall': 0.85,
            'f1_score': 0.87,
            'last_training': (datetime.now() - timedelta(days=7)).isoformat(),
            'data_points': 150000,
            'model_version': '1.2.3'
        }

    def _generate_insights(self, result: Dict) -> List[str]:
        """Generate insights from predictions"""
        insights = []
        
        predictions = result.get('predictions_generated', [])
        
        if len(predictions) > 5:
            insights.append(f"Multiple issues predicted ({len(predictions)}), consider comprehensive maintenance window")
        
        failure_preds = [p for p in predictions if 'failure' in p.get('type', '')]
        if failure_preds:
            insights.append(f"{len(failure_preds)} potential failures detected, preventive action recommended")
        
        high_confidence = [p for p in predictions if p.get('confidence', 0) > 0.9]
        if high_confidence:
            insights.append(f"{len(high_confidence)} high-confidence predictions require immediate attention")
        
        return insights

    def _create_dashboard_data(self, result: Dict) -> Dict:
        """Create dashboard visualization data"""
        return {
            'prediction_summary': {
                'total': len(result.get('predictions_generated', [])),
                'by_type': self._group_by_type(result.get('predictions_generated', [])),
                'by_severity': self._group_by_severity(result.get('predictions_generated', []))
            },
            'risk_matrix': {
                'high': len([p for p in result.get('predictions_generated', []) if p.get('confidence', 0) > 0.8]),
                'medium': len([p for p in result.get('predictions_generated', []) if 0.5 <= p.get('confidence', 0) <= 0.8]),
                'low': len([p for p in result.get('predictions_generated', []) if p.get('confidence', 0) < 0.5])
            },
            'timeline': self._create_timeline(result.get('predictions_generated', [])),
            'actions_summary': {
                'alerts': len(result.get('alerts_created', [])),
                'preventive_actions': len(result.get('actions_taken', [])),
                'pending_reviews': 3  # Simulated
            }
        }

    def _group_by_type(self, predictions: List[Dict]) -> Dict:
        """Group predictions by type"""
        groups = {}
        for pred in predictions:
            pred_type = pred.get('type', 'unknown')
            groups[pred_type] = groups.get(pred_type, 0) + 1
        return groups

    def _group_by_severity(self, predictions: List[Dict]) -> Dict:
        """Group predictions by severity"""
        return {
            'critical': len([p for p in predictions if p.get('confidence', 0) > 0.9]),
            'high': len([p for p in predictions if 0.75 <= p.get('confidence', 0) <= 0.9]),
            'medium': len([p for p in predictions if 0.5 <= p.get('confidence', 0) < 0.75]),
            'low': len([p for p in predictions if p.get('confidence', 0) < 0.5])
        }

    def _create_timeline(self, predictions: List[Dict]) -> List[Dict]:
        """Create timeline of predicted events"""
        timeline = []
        
        for pred in predictions[:5]:  # Top 5 predictions
            prediction_data = pred.get('prediction', {})
            timeline.append({
                'time': prediction_data.get('timeframe', 'unknown'),
                'event': prediction_data.get('type', 'event'),
                'resource': prediction_data.get('resource', 'unknown'),
                'probability': prediction_data.get('probability', 0)
            })
        
        return sorted(timeline, key=lambda x: x.get('probability', 0), reverse=True)
