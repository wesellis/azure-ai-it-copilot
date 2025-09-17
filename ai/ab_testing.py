"""
A/B Testing Framework for AI Decisions
Comprehensive framework for testing AI model performance and decision quality
"""

import asyncio
import json
import hashlib
import random
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import numpy as np
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """A/B test experiment status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class TestType(Enum):
    """Types of A/B tests"""
    MODEL_COMPARISON = "model_comparison"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    PARAMETER_TUNING = "parameter_tuning"
    FEATURE_TOGGLE = "feature_toggle"
    UI_OPTIMIZATION = "ui_optimization"
    ALGORITHM_SELECTION = "algorithm_selection"


class MetricType(Enum):
    """Types of metrics to track"""
    CONVERSION_RATE = "conversion_rate"
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    USER_SATISFACTION = "user_satisfaction"
    COST_PER_REQUEST = "cost_per_request"
    ERROR_RATE = "error_rate"
    ENGAGEMENT = "engagement"
    RETENTION = "retention"


class SignificanceLevel(Enum):
    """Statistical significance levels"""
    ALPHA_01 = 0.01
    ALPHA_05 = 0.05
    ALPHA_10 = 0.10


@dataclass
class ExperimentVariant:
    """A/B test variant configuration"""
    variant_id: str
    name: str
    description: str
    traffic_percentage: float
    configuration: Dict[str, Any]
    is_control: bool = False
    enabled: bool = True

    def __post_init__(self):
        if self.traffic_percentage < 0 or self.traffic_percentage > 100:
            raise ValueError("Traffic percentage must be between 0 and 100")


@dataclass
class MetricDefinition:
    """Metric definition for A/B testing"""
    metric_id: str
    name: str
    metric_type: MetricType
    goal: str  # "maximize" or "minimize"
    primary: bool = False
    aggregation: str = "mean"  # "mean", "sum", "count", "rate"
    threshold: Optional[float] = None


@dataclass
class ExperimentConfig:
    """A/B test experiment configuration"""
    experiment_id: str
    name: str
    description: str
    test_type: TestType
    variants: List[ExperimentVariant]
    metrics: List[MetricDefinition]

    # Duration and traffic
    duration_days: int
    min_sample_size: int
    max_sample_size: Optional[int] = None

    # Statistical parameters
    significance_level: SignificanceLevel = SignificanceLevel.ALPHA_05
    power: float = 0.8
    minimum_detectable_effect: float = 0.05

    # Targeting
    user_filters: Dict[str, Any] = field(default_factory=dict)
    feature_flags: List[str] = field(default_factory=list)

    # Status and metadata
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_by: str = ""
    created_at: str = ""
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

        # Validate traffic allocation
        total_traffic = sum(v.traffic_percentage for v in self.variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Total traffic allocation must equal 100%, got {total_traffic}%")


@dataclass
class ExperimentEvent:
    """Individual event/measurement in an A/B test"""
    event_id: str
    experiment_id: str
    variant_id: str
    user_id: str
    session_id: Optional[str]
    timestamp: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class VariantResults:
    """Results for a single variant"""
    variant_id: str
    variant_name: str
    sample_size: int
    metrics: Dict[str, Dict[str, float]]  # metric_id -> {mean, std, count, etc.}
    confidence_intervals: Dict[str, Tuple[float, float]]
    is_control: bool = False


@dataclass
class ExperimentResults:
    """Complete A/B test results"""
    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    duration_days: int
    total_participants: int

    # Results by variant
    variant_results: List[VariantResults]

    # Statistical analysis
    statistical_significance: Dict[str, bool]  # metric_id -> significant
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    confidence_level: float

    # Recommendations
    winning_variant: Optional[str] = None
    recommendation: str = ""
    recommendation_confidence: float = 0.0

    # Metadata
    analyzed_at: str = ""

    def __post_init__(self):
        if not self.analyzed_at:
            self.analyzed_at = datetime.now().isoformat()


class ABTestFramework:
    """Comprehensive A/B testing framework for AI systems"""

    def __init__(self, storage_path: str = "ab_tests"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        self.experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_events: Dict[str, List[ExperimentEvent]] = {}
        self.results_cache: Dict[str, ExperimentResults] = {}

        # Load existing experiments
        self._load_experiments()

    def create_experiment(self,
                         experiment_id: str,
                         name: str,
                         description: str,
                         test_type: TestType,
                         variants: List[ExperimentVariant],
                         metrics: List[MetricDefinition],
                         duration_days: int = 14,
                         min_sample_size: int = 100,
                         created_by: str = "system") -> ExperimentConfig:
        """Create a new A/B test experiment"""

        # Validate experiment ID uniqueness
        if experiment_id in self.experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")

        # Ensure one control variant
        control_variants = [v for v in variants if v.is_control]
        if len(control_variants) != 1:
            raise ValueError("Exactly one variant must be marked as control")

        # Ensure at least one primary metric
        primary_metrics = [m for m in metrics if m.primary]
        if not primary_metrics:
            raise ValueError("At least one metric must be marked as primary")

        experiment = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            test_type=test_type,
            variants=variants,
            metrics=metrics,
            duration_days=duration_days,
            min_sample_size=min_sample_size,
            created_by=created_by
        )

        self.experiments[experiment_id] = experiment
        self.experiment_events[experiment_id] = []

        self._save_experiment(experiment)

        logger.info(f"Created experiment: {experiment_id}")
        return experiment

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Can only start experiments in DRAFT status, current: {experiment.status}")

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now().isoformat()

        self._save_experiment(experiment)

        logger.info(f"Started experiment: {experiment_id}")
        return True

    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause a running experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Can only pause running experiments")

        experiment.status = ExperimentStatus.PAUSED
        self._save_experiment(experiment)

        logger.info(f"Paused experiment: {experiment_id}")
        return True

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment and mark as completed"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        if experiment.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            raise ValueError(f"Can only stop running or paused experiments")

        experiment.status = ExperimentStatus.COMPLETED
        experiment.ended_at = datetime.now().isoformat()

        self._save_experiment(experiment)

        logger.info(f"Stopped experiment: {experiment_id}")
        return True

    def assign_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Assign a user to a variant using consistent hashing"""
        if experiment_id not in self.experiments:
            return None

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            return None

        # Use consistent hashing for variant assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        percentage = (hash_value % 10000) / 100.0  # 0-99.99

        # Find variant based on traffic allocation
        cumulative_percentage = 0
        for variant in experiment.variants:
            if not variant.enabled:
                continue

            cumulative_percentage += variant.traffic_percentage
            if percentage < cumulative_percentage:
                return variant.variant_id

        # Fallback to control if no variant found
        control_variant = next((v for v in experiment.variants if v.is_control), None)
        return control_variant.variant_id if control_variant else None

    async def track_event(self,
                         experiment_id: str,
                         user_id: str,
                         variant_id: str,
                         metrics: Dict[str, float],
                         session_id: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> bool:
        """Track an event/measurement for the experiment"""

        if experiment_id not in self.experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return False

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            logger.warning(f"Experiment {experiment_id} is not running")
            return False

        # Validate variant
        if not any(v.variant_id == variant_id for v in experiment.variants):
            logger.warning(f"Variant {variant_id} not found in experiment {experiment_id}")
            return False

        # Validate metrics
        experiment_metric_ids = {m.metric_id for m in experiment.metrics}
        if not set(metrics.keys()).issubset(experiment_metric_ids):
            logger.warning(f"Invalid metrics for experiment {experiment_id}")
            return False

        event = ExperimentEvent(
            event_id=f"{experiment_id}:{user_id}:{int(time.time() * 1000)}",
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            metadata=metadata or {}
        )

        self.experiment_events[experiment_id].append(event)

        # Periodically save events to disk
        if len(self.experiment_events[experiment_id]) % 100 == 0:
            await self._save_events(experiment_id)

        return True

    async def analyze_experiment(self, experiment_id: str) -> ExperimentResults:
        """Perform statistical analysis of experiment results"""

        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        events = self.experiment_events.get(experiment_id, [])

        if not events:
            raise ValueError(f"No events found for experiment {experiment_id}")

        # Group events by variant
        variant_events = {}
        for event in events:
            if event.variant_id not in variant_events:
                variant_events[event.variant_id] = []
            variant_events[event.variant_id].append(event)

        # Calculate results for each variant
        variant_results = []
        for variant in experiment.variants:
            variant_id = variant.variant_id
            variant_event_list = variant_events.get(variant_id, [])

            result = self._calculate_variant_results(variant, variant_event_list, experiment.metrics)
            variant_results.append(result)

        # Perform statistical analysis
        control_variant = next((v for v in variant_results if v.is_control), None)
        if not control_variant:
            raise ValueError("No control variant found")

        statistical_significance = {}
        p_values = {}
        effect_sizes = {}

        for metric in experiment.metrics:
            metric_id = metric.metric_id

            # Get control and treatment data
            control_data = self._get_metric_data(control_variant, metric_id)

            for variant_result in variant_results:
                if variant_result.is_control:
                    continue

                treatment_data = self._get_metric_data(variant_result, metric_id)

                # Perform statistical test
                is_significant, p_value, effect_size = self._perform_statistical_test(
                    control_data, treatment_data, experiment.significance_level
                )

                key = f"{metric_id}_{variant_result.variant_id}"
                statistical_significance[key] = is_significant
                p_values[key] = p_value
                effect_sizes[key] = effect_size

        # Determine winning variant and recommendation
        winning_variant, recommendation, confidence = self._determine_winner(
            experiment, variant_results, statistical_significance, effect_sizes
        )

        results = ExperimentResults(
            experiment_id=experiment_id,
            experiment_name=experiment.name,
            status=experiment.status,
            duration_days=experiment.duration_days,
            total_participants=len(set(e.user_id for e in events)),
            variant_results=variant_results,
            statistical_significance=statistical_significance,
            p_values=p_values,
            effect_sizes=effect_sizes,
            confidence_level=1.0 - experiment.significance_level.value,
            winning_variant=winning_variant,
            recommendation=recommendation,
            recommendation_confidence=confidence
        )

        # Cache results
        self.results_cache[experiment_id] = results

        # Save results
        await self._save_results(results)

        logger.info(f"Analyzed experiment {experiment_id}: {recommendation}")
        return results

    def _calculate_variant_results(self,
                                  variant: ExperimentVariant,
                                  events: List[ExperimentEvent],
                                  metric_definitions: List[MetricDefinition]) -> VariantResults:
        """Calculate results for a single variant"""

        if not events:
            return VariantResults(
                variant_id=variant.variant_id,
                variant_name=variant.name,
                sample_size=0,
                metrics={},
                confidence_intervals={},
                is_control=variant.is_control
            )

        metrics = {}
        confidence_intervals = {}

        for metric_def in metric_definitions:
            metric_id = metric_def.metric_id

            # Extract metric values from events
            values = []
            for event in events:
                if metric_id in event.metrics:
                    values.append(event.metrics[metric_id])

            if not values:
                metrics[metric_id] = {"mean": 0, "std": 0, "count": 0}
                confidence_intervals[metric_id] = (0, 0)
                continue

            # Calculate statistics
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            count_val = len(values)

            metrics[metric_id] = {
                "mean": mean_val,
                "std": std_val,
                "count": count_val,
                "min": min(values),
                "max": max(values)
            }

            # Calculate confidence interval (95%)
            if len(values) > 1:
                se = std_val / (len(values) ** 0.5)
                margin_error = 1.96 * se  # 95% CI
                ci_lower = mean_val - margin_error
                ci_upper = mean_val + margin_error
                confidence_intervals[metric_id] = (ci_lower, ci_upper)
            else:
                confidence_intervals[metric_id] = (mean_val, mean_val)

        return VariantResults(
            variant_id=variant.variant_id,
            variant_name=variant.name,
            sample_size=len(set(e.user_id for e in events)),
            metrics=metrics,
            confidence_intervals=confidence_intervals,
            is_control=variant.is_control
        )

    def _get_metric_data(self, variant_result: VariantResults, metric_id: str) -> List[float]:
        """Get metric data for statistical testing"""
        if metric_id not in variant_result.metrics:
            return []

        # For simplicity, simulate data points based on summary statistics
        # In practice, you'd store raw data points
        metric_data = variant_result.metrics[metric_id]
        mean_val = metric_data["mean"]
        std_val = metric_data["std"]
        count_val = metric_data["count"]

        if count_val == 0:
            return []

        # Generate synthetic data points with same mean/std
        np.random.seed(42)  # For reproducibility
        return np.random.normal(mean_val, std_val, count_val).tolist()

    def _perform_statistical_test(self,
                                control_data: List[float],
                                treatment_data: List[float],
                                significance_level: SignificanceLevel) -> Tuple[bool, float, float]:
        """Perform statistical test (t-test) between control and treatment"""

        if not control_data or not treatment_data:
            return False, 1.0, 0.0

        try:
            from scipy import stats

            # Perform independent t-test
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data)

            # Calculate effect size (Cohen's d)
            pooled_std = ((np.std(control_data) ** 2 + np.std(treatment_data) ** 2) / 2) ** 0.5
            effect_size = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std if pooled_std > 0 else 0

            # Check significance
            is_significant = p_value < significance_level.value

            return is_significant, p_value, effect_size

        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return False, 1.0, 0.0

    def _determine_winner(self,
                         experiment: ExperimentConfig,
                         variant_results: List[VariantResults],
                         statistical_significance: Dict[str, bool],
                         effect_sizes: Dict[str, float]) -> Tuple[Optional[str], str, float]:
        """Determine winning variant and generate recommendation"""

        # Get primary metrics
        primary_metrics = [m for m in experiment.metrics if m.primary]

        if not primary_metrics:
            return None, "No primary metrics defined", 0.0

        control_variant = next((v for v in variant_results if v.is_control), None)
        if not control_variant:
            return None, "No control variant found", 0.0

        # Analyze each non-control variant
        best_variant = None
        best_score = float('-inf')
        confidence_sum = 0
        significant_improvements = 0

        for variant_result in variant_results:
            if variant_result.is_control:
                continue

            variant_score = 0
            variant_confidence = 0

            for metric in primary_metrics:
                metric_id = metric.metric_id
                key = f"{metric_id}_{variant_result.variant_id}"

                if key in statistical_significance and statistical_significance[key]:
                    effect_size = effect_sizes.get(key, 0)

                    # Adjust effect size based on goal
                    if metric.goal == "minimize":
                        effect_size = -effect_size

                    if effect_size > 0:  # Improvement
                        variant_score += effect_size
                        variant_confidence += 0.5
                        significant_improvements += 1

            if variant_score > best_score:
                best_score = variant_score
                best_variant = variant_result.variant_id
                confidence_sum = variant_confidence

        # Generate recommendation
        if best_variant and significant_improvements > 0:
            confidence = min(confidence_sum / len(primary_metrics), 1.0)
            recommendation = f"Variant {best_variant} shows significant improvement in primary metrics with {confidence:.1%} confidence"
            return best_variant, recommendation, confidence
        else:
            recommendation = "No variant shows statistically significant improvement over control"
            return None, recommendation, 0.0

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        events = self.experiment_events.get(experiment_id, [])

        # Calculate basic statistics
        total_events = len(events)
        unique_users = len(set(e.user_id for e in events))

        # Events by variant
        variant_stats = {}
        for variant in experiment.variants:
            variant_events = [e for e in events if e.variant_id == variant.variant_id]
            variant_stats[variant.variant_id] = {
                "name": variant.name,
                "events": len(variant_events),
                "users": len(set(e.user_id for e in variant_events)),
                "traffic_percentage": variant.traffic_percentage
            }

        # Check if experiment should be stopped
        should_stop = False
        stop_reason = ""

        if experiment.status == ExperimentStatus.RUNNING:
            # Check duration
            if experiment.started_at:
                start_time = datetime.fromisoformat(experiment.started_at)
                duration = datetime.now() - start_time
                if duration.days >= experiment.duration_days:
                    should_stop = True
                    stop_reason = "Duration limit reached"

            # Check sample size
            if unique_users >= (experiment.max_sample_size or float('inf')):
                should_stop = True
                stop_reason = "Sample size limit reached"

        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "total_events": total_events,
            "unique_users": unique_users,
            "min_sample_size": experiment.min_sample_size,
            "variant_stats": variant_stats,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
            "created_at": experiment.created_at,
            "started_at": experiment.started_at,
            "ended_at": experiment.ended_at
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with basic info"""
        experiments = []

        for exp_id, experiment in self.experiments.items():
            events = self.experiment_events.get(exp_id, [])

            experiments.append({
                "experiment_id": exp_id,
                "name": experiment.name,
                "status": experiment.status.value,
                "test_type": experiment.test_type.value,
                "variants": len(experiment.variants),
                "events": len(events),
                "created_at": experiment.created_at,
                "started_at": experiment.started_at
            })

        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)

    async def _save_experiment(self, experiment: ExperimentConfig):
        """Save experiment configuration to disk"""
        file_path = self.storage_path / f"experiment_{experiment.experiment_id}.json"

        with open(file_path, 'w') as f:
            json.dump(asdict(experiment), f, indent=2, default=str)

    async def _save_events(self, experiment_id: str):
        """Save experiment events to disk"""
        events = self.experiment_events.get(experiment_id, [])
        if not events:
            return

        file_path = self.storage_path / f"events_{experiment_id}.jsonl"

        with open(file_path, 'w') as f:
            for event in events:
                f.write(json.dumps(asdict(event), default=str) + '\n')

    async def _save_results(self, results: ExperimentResults):
        """Save experiment results to disk"""
        file_path = self.storage_path / f"results_{results.experiment_id}.json"

        with open(file_path, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)

    def _load_experiments(self):
        """Load experiments from disk"""
        if not self.storage_path.exists():
            return

        for file_path in self.storage_path.glob("experiment_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Reconstruct objects
                variants = [ExperimentVariant(**v) for v in data['variants']]
                metrics = [MetricDefinition(**m) for m in data['metrics']]

                data['variants'] = variants
                data['metrics'] = metrics
                data['test_type'] = TestType(data['test_type'])
                data['status'] = ExperimentStatus(data['status'])
                data['significance_level'] = SignificanceLevel(data['significance_level'])

                experiment = ExperimentConfig(**data)
                self.experiments[experiment.experiment_id] = experiment

                # Load events
                self._load_events(experiment.experiment_id)

            except Exception as e:
                logger.error(f"Error loading experiment from {file_path}: {e}")

    def _load_events(self, experiment_id: str):
        """Load experiment events from disk"""
        file_path = self.storage_path / f"events_{experiment_id}.jsonl"

        if not file_path.exists():
            self.experiment_events[experiment_id] = []
            return

        events = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    event = ExperimentEvent(**data)
                    events.append(event)

            self.experiment_events[experiment_id] = events

        except Exception as e:
            logger.error(f"Error loading events for {experiment_id}: {e}")
            self.experiment_events[experiment_id] = []


# Global A/B testing framework instance
ab_test_framework = ABTestFramework()


# Decorator for automatic A/B testing
def ab_test(experiment_id: str,
           metric_id: str,
           user_id_func: Callable[[], str] = None):
    """Decorator for automatic A/B testing of functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get user ID
            if user_id_func:
                user_id = user_id_func()
            else:
                user_id = kwargs.get('user_id', 'anonymous')

            # Get variant assignment
            variant_id = ab_test_framework.assign_variant(experiment_id, user_id)

            if not variant_id:
                # No active experiment, use original function
                return await func(*args, **kwargs)

            # Get variant configuration
            experiment = ab_test_framework.experiments.get(experiment_id)
            if not experiment:
                return await func(*args, **kwargs)

            variant = next((v for v in experiment.variants if v.variant_id == variant_id), None)
            if not variant:
                return await func(*args, **kwargs)

            # Apply variant configuration
            kwargs.update(variant.configuration)

            # Execute function with timing
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
                raise

            finally:
                # Track metrics
                duration = time.time() - start_time
                metrics = {
                    metric_id: 1.0 if success else 0.0,
                    "response_time": duration * 1000,  # milliseconds
                    "error_rate": 0.0 if success else 1.0
                }

                await ab_test_framework.track_event(
                    experiment_id=experiment_id,
                    user_id=user_id,
                    variant_id=variant_id,
                    metrics=metrics,
                    metadata={"function": func.__name__, "error": error}
                )

            return result

        return wrapper
    return decorator


# Utility functions
async def create_model_comparison_test(experiment_id: str,
                                     model_a_config: Dict[str, Any],
                                     model_b_config: Dict[str, Any],
                                     traffic_split: float = 50.0) -> ExperimentConfig:
    """Create a model comparison A/B test"""

    variants = [
        ExperimentVariant(
            variant_id="control",
            name="Model A (Control)",
            description="Current model configuration",
            traffic_percentage=traffic_split,
            configuration=model_a_config,
            is_control=True
        ),
        ExperimentVariant(
            variant_id="treatment",
            name="Model B (Treatment)",
            description="New model configuration",
            traffic_percentage=100 - traffic_split,
            configuration=model_b_config
        )
    ]

    metrics = [
        MetricDefinition(
            metric_id="accuracy",
            name="Model Accuracy",
            metric_type=MetricType.ACCURACY,
            goal="maximize",
            primary=True
        ),
        MetricDefinition(
            metric_id="response_time",
            name="Response Time",
            metric_type=MetricType.RESPONSE_TIME,
            goal="minimize"
        ),
        MetricDefinition(
            metric_id="cost_per_request",
            name="Cost per Request",
            metric_type=MetricType.COST_PER_REQUEST,
            goal="minimize"
        )
    ]

    return ab_test_framework.create_experiment(
        experiment_id=experiment_id,
        name=f"Model Comparison: {experiment_id}",
        description="A/B test comparing two AI model configurations",
        test_type=TestType.MODEL_COMPARISON,
        variants=variants,
        metrics=metrics
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create a model comparison test
        experiment = await create_model_comparison_test(
            experiment_id="gpt4_vs_gpt35",
            model_a_config={"model": "gpt-3.5-turbo", "temperature": 0.1},
            model_b_config={"model": "gpt-4", "temperature": 0.1}
        )

        print(f"Created experiment: {experiment.experiment_id}")

        # Start the experiment
        ab_test_framework.start_experiment(experiment.experiment_id)

        # Simulate some usage
        for i in range(100):
            user_id = f"user_{i}"
            variant_id = ab_test_framework.assign_variant(experiment.experiment_id, user_id)

            if variant_id:
                # Simulate metrics
                accuracy = random.uniform(0.8, 0.95)
                response_time = random.uniform(100, 500)
                cost = random.uniform(0.001, 0.01)

                await ab_test_framework.track_event(
                    experiment_id=experiment.experiment_id,
                    user_id=user_id,
                    variant_id=variant_id,
                    metrics={
                        "accuracy": accuracy,
                        "response_time": response_time,
                        "cost_per_request": cost
                    }
                )

        # Analyze results
        results = await ab_test_framework.analyze_experiment(experiment.experiment_id)
        print(f"Results: {results.recommendation}")

    asyncio.run(main())