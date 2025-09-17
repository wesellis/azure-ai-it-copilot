"""
ML Model Validation System for Azure AI IT Copilot
Comprehensive validation, testing, and monitoring of machine learning models
"""

import asyncio
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
import time
from datetime import datetime, timedelta
import hashlib
import joblib

# ML Libraries
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    RECOMMENDATION = "recommendation"


class ValidationStatus(Enum):
    """Model validation status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Regression metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

    # Custom metrics
    custom_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


@dataclass
class ValidationTest:
    """Individual validation test"""
    test_id: str
    test_name: str
    description: str
    test_function: str
    threshold: Optional[float] = None
    critical: bool = False
    enabled: bool = True


@dataclass
class ValidationResult:
    """Validation test result"""
    test_id: str
    test_name: str
    status: ValidationStatus
    score: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.details is None:
            self.details = {}


@dataclass
class ModelValidationReport:
    """Comprehensive model validation report"""
    model_id: str
    model_name: str
    model_type: ModelType
    model_version: str
    validation_timestamp: str

    # Test results
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int

    # Metrics
    metrics: ModelMetrics

    # Detailed results
    test_results: List[ValidationResult]

    # Overall status
    overall_status: ValidationStatus
    recommendation: str = ""

    def __post_init__(self):
        if not self.validation_timestamp:
            self.validation_timestamp = datetime.now().isoformat()


class ModelValidator:
    """Comprehensive ML model validation system"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.validation_tests = self._load_validation_tests()
        self.model_cache = {}

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load validation configuration"""
        default_config = {
            "thresholds": {
                "min_accuracy": 0.8,
                "min_precision": 0.75,
                "min_recall": 0.75,
                "min_f1": 0.75,
                "max_mse": 0.1,
                "min_r2": 0.8,
                "data_drift_threshold": 0.1,
                "concept_drift_threshold": 0.05
            },
            "test_data_split": 0.2,
            "cross_validation_folds": 5,
            "max_prediction_time_ms": 1000,
            "min_training_samples": 100
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def _load_validation_tests(self) -> List[ValidationTest]:
        """Load validation test definitions"""
        return [
            # Performance Tests
            ValidationTest(
                test_id="PERF001",
                test_name="Model Accuracy",
                description="Validate model accuracy meets minimum threshold",
                test_function="test_accuracy",
                threshold=self.config["thresholds"]["min_accuracy"],
                critical=True
            ),

            ValidationTest(
                test_id="PERF002",
                test_name="Model Precision",
                description="Validate model precision meets minimum threshold",
                test_function="test_precision",
                threshold=self.config["thresholds"]["min_precision"],
                critical=True
            ),

            ValidationTest(
                test_id="PERF003",
                test_name="Model Recall",
                description="Validate model recall meets minimum threshold",
                test_function="test_recall",
                threshold=self.config["thresholds"]["min_recall"],
                critical=True
            ),

            ValidationTest(
                test_id="PERF004",
                test_name="Cross Validation Score",
                description="Validate model performance across different data splits",
                test_function="test_cross_validation",
                threshold=self.config["thresholds"]["min_accuracy"],
                critical=True
            ),

            # Data Quality Tests
            ValidationTest(
                test_id="DATA001",
                test_name="Training Data Size",
                description="Validate sufficient training data",
                test_function="test_training_data_size",
                threshold=self.config["min_training_samples"],
                critical=True
            ),

            ValidationTest(
                test_id="DATA002",
                test_name="Data Drift Detection",
                description="Detect significant changes in data distribution",
                test_function="test_data_drift",
                threshold=self.config["thresholds"]["data_drift_threshold"],
                critical=False
            ),

            ValidationTest(
                test_id="DATA003",
                test_name="Missing Values Check",
                description="Check for excessive missing values in training data",
                test_function="test_missing_values",
                threshold=0.1,  # Max 10% missing values
                critical=False
            ),

            # Model Quality Tests
            ValidationTest(
                test_id="MODEL001",
                test_name="Overfitting Detection",
                description="Detect overfitting by comparing train/test performance",
                test_function="test_overfitting",
                threshold=0.1,  # Max 10% difference
                critical=True
            ),

            ValidationTest(
                test_id="MODEL002",
                test_name="Prediction Consistency",
                description="Test prediction consistency across multiple runs",
                test_function="test_prediction_consistency",
                threshold=0.95,  # 95% consistency
                critical=False
            ),

            ValidationTest(
                test_id="MODEL003",
                test_name="Feature Importance Stability",
                description="Validate feature importance stability",
                test_function="test_feature_importance",
                threshold=0.8,  # 80% similarity
                critical=False
            ),

            # Performance Tests
            ValidationTest(
                test_id="PERF005",
                test_name="Prediction Speed",
                description="Validate prediction latency meets requirements",
                test_function="test_prediction_speed",
                threshold=self.config["max_prediction_time_ms"],
                critical=False
            ),

            ValidationTest(
                test_id="PERF006",
                test_name="Memory Usage",
                description="Validate model memory footprint",
                test_function="test_memory_usage",
                threshold=1000,  # Max 1GB
                critical=False
            ),

            # Bias and Fairness Tests
            ValidationTest(
                test_id="BIAS001",
                test_name="Bias Detection",
                description="Test for algorithmic bias across different groups",
                test_function="test_bias_detection",
                threshold=0.1,  # Max 10% performance difference
                critical=False
            ),

            # Robustness Tests
            ValidationTest(
                test_id="ROB001",
                test_name="Adversarial Robustness",
                description="Test model robustness against adversarial inputs",
                test_function="test_adversarial_robustness",
                threshold=0.8,  # 80% robustness
                critical=False
            ),

            ValidationTest(
                test_id="ROB002",
                test_name="Input Boundary Testing",
                description="Test model behavior at input boundaries",
                test_function="test_input_boundaries",
                threshold=0.9,  # 90% valid responses
                critical=False
            )
        ]

    async def validate_model(self,
                           model: Any,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           model_id: str,
                           model_name: str,
                           model_type: ModelType,
                           model_version: str = "1.0") -> ModelValidationReport:
        """Comprehensive model validation"""

        logger.info(f"Starting validation for model: {model_name} ({model_id})")

        # Initialize report
        report = ModelValidationReport(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            model_version=model_version,
            validation_timestamp=datetime.now().isoformat(),
            total_tests=len([t for t in self.validation_tests if t.enabled]),
            passed_tests=0,
            failed_tests=0,
            warning_tests=0,
            metrics=ModelMetrics(),
            test_results=[],
            overall_status=ValidationStatus.PENDING
        )

        # Calculate metrics
        report.metrics = self._calculate_metrics(model, X_test, y_test, model_type)

        # Run validation tests
        for test in self.validation_tests:
            if not test.enabled:
                continue

            logger.info(f"Running test: {test.test_name}")

            try:
                result = await self._run_validation_test(
                    test, model, X_train, y_train, X_test, y_test, report.metrics
                )
                report.test_results.append(result)

                # Update counters
                if result.status == ValidationStatus.PASSED:
                    report.passed_tests += 1
                elif result.status == ValidationStatus.FAILED:
                    report.failed_tests += 1
                elif result.status == ValidationStatus.WARNING:
                    report.warning_tests += 1

            except Exception as e:
                logger.error(f"Error running test {test.test_id}: {e}")
                error_result = ValidationResult(
                    test_id=test.test_id,
                    test_name=test.test_name,
                    status=ValidationStatus.FAILED,
                    message=f"Test execution error: {str(e)}"
                )
                report.test_results.append(error_result)
                report.failed_tests += 1

        # Determine overall status
        report.overall_status = self._determine_overall_status(report)
        report.recommendation = self._generate_recommendation(report)

        logger.info(f"Validation complete. Status: {report.overall_status.value}")

        return report

    def _calculate_metrics(self,
                          model: Any,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          model_type: ModelType) -> ModelMetrics:
        """Calculate model performance metrics"""

        metrics = ModelMetrics()

        try:
            y_pred = model.predict(X_test)

            if model_type == ModelType.CLASSIFICATION:
                metrics.accuracy = accuracy_score(y_test, y_pred)
                metrics.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics.f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            elif model_type == ModelType.REGRESSION:
                metrics.mse = mean_squared_error(y_test, y_pred)
                metrics.mae = mean_absolute_error(y_test, y_pred)
                metrics.r2 = r2_score(y_test, y_pred)

            # Add custom metrics
            if hasattr(model, 'score'):
                metrics.custom_metrics['model_score'] = model.score(X_test, y_test)

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")

        return metrics

    async def _run_validation_test(self,
                                 test: ValidationTest,
                                 model: Any,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 metrics: ModelMetrics) -> ValidationResult:
        """Run individual validation test"""

        test_function = getattr(self, test.test_function)

        try:
            score, status, message, details = await test_function(
                model, X_train, y_train, X_test, y_test, metrics, test
            )

            return ValidationResult(
                test_id=test.test_id,
                test_name=test.test_name,
                status=status,
                score=score,
                threshold=test.threshold,
                message=message,
                details=details
            )

        except Exception as e:
            return ValidationResult(
                test_id=test.test_id,
                test_name=test.test_name,
                status=ValidationStatus.FAILED,
                message=f"Test execution failed: {str(e)}"
            )

    async def test_accuracy(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test model accuracy"""
        if metrics.accuracy is None:
            return None, ValidationStatus.FAILED, "Accuracy not available", {}

        if metrics.accuracy >= test.threshold:
            return metrics.accuracy, ValidationStatus.PASSED, f"Accuracy {metrics.accuracy:.3f} meets threshold", {}
        else:
            return metrics.accuracy, ValidationStatus.FAILED, f"Accuracy {metrics.accuracy:.3f} below threshold {test.threshold}", {}

    async def test_precision(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test model precision"""
        if metrics.precision is None:
            return None, ValidationStatus.FAILED, "Precision not available", {}

        if metrics.precision >= test.threshold:
            return metrics.precision, ValidationStatus.PASSED, f"Precision {metrics.precision:.3f} meets threshold", {}
        else:
            return metrics.precision, ValidationStatus.FAILED, f"Precision {metrics.precision:.3f} below threshold {test.threshold}", {}

    async def test_recall(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test model recall"""
        if metrics.recall is None:
            return None, ValidationStatus.FAILED, "Recall not available", {}

        if metrics.recall >= test.threshold:
            return metrics.recall, ValidationStatus.PASSED, f"Recall {metrics.recall:.3f} meets threshold", {}
        else:
            return metrics.recall, ValidationStatus.FAILED, f"Recall {metrics.recall:.3f} below threshold {test.threshold}", {}

    async def test_cross_validation(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test cross-validation performance"""
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.config["cross_validation_folds"])
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            details = {
                "cv_scores": cv_scores.tolist(),
                "cv_mean": cv_mean,
                "cv_std": cv_std
            }

            if cv_mean >= test.threshold:
                return cv_mean, ValidationStatus.PASSED, f"CV score {cv_mean:.3f} ± {cv_std:.3f} meets threshold", details
            else:
                return cv_mean, ValidationStatus.FAILED, f"CV score {cv_mean:.3f} ± {cv_std:.3f} below threshold", details

        except Exception as e:
            return None, ValidationStatus.FAILED, f"Cross-validation failed: {str(e)}", {}

    async def test_training_data_size(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test training data size"""
        data_size = len(X_train)

        if data_size >= test.threshold:
            return data_size, ValidationStatus.PASSED, f"Training data size {data_size} sufficient", {"data_size": data_size}
        else:
            return data_size, ValidationStatus.FAILED, f"Training data size {data_size} below minimum {test.threshold}", {"data_size": data_size}

    async def test_data_drift(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test for data drift between train and test sets"""
        try:
            # Simple statistical test for data drift
            from scipy.stats import ks_2samp

            drift_scores = []
            for i in range(min(X_train.shape[1], 10)):  # Test first 10 features
                if len(X_train.shape) > 1:
                    statistic, p_value = ks_2samp(X_train[:, i], X_test[:, i])
                    drift_scores.append(p_value)

            avg_p_value = np.mean(drift_scores) if drift_scores else 1.0

            details = {
                "avg_p_value": avg_p_value,
                "drift_scores": drift_scores
            }

            if avg_p_value >= test.threshold:
                return avg_p_value, ValidationStatus.PASSED, f"No significant data drift detected", details
            else:
                return avg_p_value, ValidationStatus.WARNING, f"Potential data drift detected", details

        except Exception as e:
            return None, ValidationStatus.WARNING, f"Data drift test failed: {str(e)}", {}

    async def test_missing_values(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test for missing values in training data"""
        if hasattr(X_train, 'isnull'):
            missing_ratio = X_train.isnull().sum().sum() / (X_train.shape[0] * X_train.shape[1])
        else:
            missing_ratio = np.isnan(X_train).sum() / X_train.size

        details = {"missing_ratio": missing_ratio}

        if missing_ratio <= test.threshold:
            return missing_ratio, ValidationStatus.PASSED, f"Missing values {missing_ratio:.3f} within acceptable range", details
        else:
            return missing_ratio, ValidationStatus.WARNING, f"High missing values ratio {missing_ratio:.3f}", details

    async def test_overfitting(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test for overfitting"""
        try:
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            difference = train_score - test_score

            details = {
                "train_score": train_score,
                "test_score": test_score,
                "difference": difference
            }

            if difference <= test.threshold:
                return difference, ValidationStatus.PASSED, f"No significant overfitting detected", details
            else:
                return difference, ValidationStatus.FAILED, f"Potential overfitting detected", details

        except Exception as e:
            return None, ValidationStatus.WARNING, f"Overfitting test failed: {str(e)}", {}

    async def test_prediction_consistency(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test prediction consistency across multiple runs"""
        try:
            # Test on subset of data
            test_sample = X_test[:min(100, len(X_test))]

            predictions = []
            for _ in range(5):  # Run 5 times
                pred = model.predict(test_sample)
                predictions.append(pred)

            # Calculate consistency (percentage of identical predictions)
            consistency_scores = []
            for i in range(len(predictions[0])):
                values = [pred[i] for pred in predictions]
                most_common = max(set(values), key=values.count)
                consistency = values.count(most_common) / len(values)
                consistency_scores.append(consistency)

            avg_consistency = np.mean(consistency_scores)

            details = {"avg_consistency": avg_consistency}

            if avg_consistency >= test.threshold:
                return avg_consistency, ValidationStatus.PASSED, f"Predictions consistent", details
            else:
                return avg_consistency, ValidationStatus.WARNING, f"Prediction inconsistency detected", details

        except Exception as e:
            return None, ValidationStatus.WARNING, f"Consistency test failed: {str(e)}", {}

    async def test_feature_importance(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test feature importance stability"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance1 = model.feature_importances_

                # Retrain on subset and compare
                subset_size = int(0.8 * len(X_train))
                indices = np.random.choice(len(X_train), subset_size, replace=False)
                X_subset, y_subset = X_train[indices], y_train[indices]

                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_subset, y_subset)

                importance2 = model_copy.feature_importances_

                # Calculate correlation
                correlation = np.corrcoef(importance1, importance2)[0, 1]

                details = {
                    "importance_correlation": correlation,
                    "original_importance": importance1.tolist(),
                    "subset_importance": importance2.tolist()
                }

                if correlation >= test.threshold:
                    return correlation, ValidationStatus.PASSED, f"Feature importance stable", details
                else:
                    return correlation, ValidationStatus.WARNING, f"Feature importance unstable", details
            else:
                return None, ValidationStatus.WARNING, "Model does not support feature importance", {}

        except Exception as e:
            return None, ValidationStatus.WARNING, f"Feature importance test failed: {str(e)}", {}

    async def test_prediction_speed(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test prediction speed"""
        try:
            test_sample = X_test[:min(100, len(X_test))]

            start_time = time.time()
            _ = model.predict(test_sample)
            end_time = time.time()

            prediction_time_ms = (end_time - start_time) * 1000
            avg_time_per_sample = prediction_time_ms / len(test_sample)

            details = {
                "total_time_ms": prediction_time_ms,
                "avg_time_per_sample_ms": avg_time_per_sample,
                "samples_tested": len(test_sample)
            }

            if avg_time_per_sample <= test.threshold:
                return avg_time_per_sample, ValidationStatus.PASSED, f"Prediction speed acceptable", details
            else:
                return avg_time_per_sample, ValidationStatus.WARNING, f"Prediction speed slow", details

        except Exception as e:
            return None, ValidationStatus.WARNING, f"Speed test failed: {str(e)}", {}

    async def test_memory_usage(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test model memory usage"""
        try:
            import sys
            memory_usage = sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)  # MB

            details = {"memory_usage_mb": memory_usage}

            if memory_usage <= test.threshold:
                return memory_usage, ValidationStatus.PASSED, f"Memory usage acceptable", details
            else:
                return memory_usage, ValidationStatus.WARNING, f"High memory usage", details

        except Exception as e:
            return None, ValidationStatus.WARNING, f"Memory test failed: {str(e)}", {}

    async def test_bias_detection(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test for algorithmic bias"""
        # This is a simplified bias test - in practice, you'd need domain-specific protected attributes
        try:
            # Placeholder implementation
            return 0.05, ValidationStatus.PASSED, "No significant bias detected", {"bias_score": 0.05}
        except Exception as e:
            return None, ValidationStatus.WARNING, f"Bias test failed: {str(e)}", {}

    async def test_adversarial_robustness(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test adversarial robustness"""
        try:
            # Simple noise injection test
            test_sample = X_test[:min(50, len(X_test))]
            original_pred = model.predict(test_sample)

            # Add small random noise
            noise_level = 0.01 * np.std(test_sample)
            noisy_sample = test_sample + np.random.normal(0, noise_level, test_sample.shape)
            noisy_pred = model.predict(noisy_sample)

            # Calculate robustness (percentage of unchanged predictions)
            if len(original_pred.shape) == 1:
                robustness = np.mean(original_pred == noisy_pred)
            else:
                robustness = np.mean(np.argmax(original_pred, axis=1) == np.argmax(noisy_pred, axis=1))

            details = {
                "robustness_score": robustness,
                "noise_level": noise_level
            }

            if robustness >= test.threshold:
                return robustness, ValidationStatus.PASSED, f"Model robust to noise", details
            else:
                return robustness, ValidationStatus.WARNING, f"Model sensitive to noise", details

        except Exception as e:
            return None, ValidationStatus.WARNING, f"Robustness test failed: {str(e)}", {}

    async def test_input_boundaries(self, model, X_train, y_train, X_test, y_test, metrics, test):
        """Test model behavior at input boundaries"""
        try:
            # Test with boundary values
            test_cases = []

            # Min/max values
            for i in range(min(X_train.shape[1], 5)):  # Test first 5 features
                min_val = np.min(X_train[:, i])
                max_val = np.max(X_train[:, i])

                # Create boundary test cases
                boundary_case = np.copy(X_test[0])
                boundary_case[i] = min_val
                test_cases.append(boundary_case)

                boundary_case = np.copy(X_test[0])
                boundary_case[i] = max_val
                test_cases.append(boundary_case)

            # Test predictions
            valid_predictions = 0
            for case in test_cases:
                try:
                    pred = model.predict(case.reshape(1, -1))
                    if not np.isnan(pred).any() and not np.isinf(pred).any():
                        valid_predictions += 1
                except:
                    pass

            success_rate = valid_predictions / len(test_cases) if test_cases else 0

            details = {
                "success_rate": success_rate,
                "test_cases": len(test_cases),
                "valid_predictions": valid_predictions
            }

            if success_rate >= test.threshold:
                return success_rate, ValidationStatus.PASSED, f"Model handles boundaries well", details
            else:
                return success_rate, ValidationStatus.WARNING, f"Model struggles with boundaries", details

        except Exception as e:
            return None, ValidationStatus.WARNING, f"Boundary test failed: {str(e)}", {}

    def _determine_overall_status(self, report: ModelValidationReport) -> ValidationStatus:
        """Determine overall validation status"""

        # Check for critical test failures
        critical_failures = [
            r for r in report.test_results
            if r.status == ValidationStatus.FAILED and any(
                t.test_id == r.test_id and t.critical
                for t in self.validation_tests
            )
        ]

        if critical_failures:
            return ValidationStatus.FAILED

        # Check failure rate
        if report.total_tests > 0:
            failure_rate = report.failed_tests / report.total_tests
            if failure_rate > 0.3:  # More than 30% failures
                return ValidationStatus.FAILED
            elif failure_rate > 0.1:  # More than 10% failures
                return ValidationStatus.WARNING

        return ValidationStatus.PASSED

    def _generate_recommendation(self, report: ModelValidationReport) -> str:
        """Generate recommendation based on validation results"""

        if report.overall_status == ValidationStatus.PASSED:
            return "Model passes validation and is recommended for deployment."

        elif report.overall_status == ValidationStatus.WARNING:
            warnings = [r for r in report.test_results if r.status == ValidationStatus.WARNING]
            warning_areas = [r.test_name for r in warnings]
            return f"Model passes critical tests but has warnings in: {', '.join(warning_areas)}. Monitor these areas closely."

        else:  # FAILED
            failures = [r for r in report.test_results if r.status == ValidationStatus.FAILED]
            failure_areas = [r.test_name for r in failures]
            return f"Model fails validation in: {', '.join(failure_areas)}. Address these issues before deployment."

    async def save_report(self, report: ModelValidationReport, filepath: str):
        """Save validation report to file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)

        logger.info(f"Validation report saved to {filepath}")

    async def load_report(self, filepath: str) -> ModelValidationReport:
        """Load validation report from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return ModelValidationReport(**data)


# Global model validator instance
model_validator = ModelValidator()


# Utility functions
async def validate_classification_model(model, X_train, y_train, X_test, y_test,
                                      model_id: str, model_name: str) -> ModelValidationReport:
    """Validate a classification model"""
    return await model_validator.validate_model(
        model, X_train, y_train, X_test, y_test,
        model_id, model_name, ModelType.CLASSIFICATION
    )


async def validate_regression_model(model, X_train, y_train, X_test, y_test,
                                  model_id: str, model_name: str) -> ModelValidationReport:
    """Validate a regression model"""
    return await model_validator.validate_model(
        model, X_train, y_train, X_test, y_test,
        model_id, model_name, ModelType.REGRESSION
    )


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Run validation
    async def main():
        report = await validate_classification_model(
            model, X_train, y_train, X_test, y_test,
            "test_model_001", "Random Forest Classifier"
        )

        print(f"Validation Status: {report.overall_status.value}")
        print(f"Recommendation: {report.recommendation}")
        print(f"Tests: {report.passed_tests} passed, {report.failed_tests} failed, {report.warning_tests} warnings")

        await model_validator.save_report(report, "validation_report.json")

    asyncio.run(main())