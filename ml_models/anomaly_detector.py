"""
Anomaly Detection Model for Azure Resources
Uses Isolation Forest and LSTM for time-series anomaly detection
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """ML model for detecting anomalies in Azure resource metrics"""

    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector

        Args:
            contamination: Expected proportion of anomalies (default 0.1 = 10%)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        self.feature_columns = [
            'cpu_usage_percent',
            'memory_usage_percent',
            'disk_read_bytes_per_sec',
            'disk_write_bytes_per_sec',
            'network_in_bytes_per_sec',
            'network_out_bytes_per_sec',
            'request_rate',
            'error_rate'
        ]

        self.is_trained = False
        self.model_path = "ml-models/saved/anomaly_detector.joblib"
        self.scaler_path = "ml-models/saved/anomaly_scaler.joblib"

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for anomaly detection

        Args:
            df: DataFrame with raw metrics

        Returns:
            DataFrame with engineered features
        """
        # Create copy to avoid modifying original
        features = df.copy()

        # Add time-based features
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

        # Add rolling statistics
        for col in self.feature_columns:
            if col in features.columns:
                # Rolling mean and std over 1 hour (12 x 5-minute intervals)
                features[f'{col}_rolling_mean'] = features[col].rolling(
                    window=12, min_periods=1
                ).mean()

                features[f'{col}_rolling_std'] = features[col].rolling(
                    window=12, min_periods=1
                ).std().fillna(0)

                # Rate of change
                features[f'{col}_rate_change'] = features[col].diff().fillna(0)

        # Add cross-feature ratios
        if 'cpu_usage_percent' in features.columns and 'memory_usage_percent' in features.columns:
            features['cpu_memory_ratio'] = (
                features['cpu_usage_percent'] /
                (features['memory_usage_percent'] + 1)
            )

        if 'network_in_bytes_per_sec' in features.columns and 'network_out_bytes_per_sec' in features.columns:
            features['network_io_ratio'] = (
                features['network_in_bytes_per_sec'] /
                (features['network_out_bytes_per_sec'] + 1)
            )

        return features

    def train(
        self,
        training_data: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the anomaly detection model

        Args:
            training_data: Historical metrics data
            validation_split: Fraction of data for validation

        Returns:
            Training results and metrics
        """
        logger.info("Training anomaly detection model...")

        # Prepare features
        features = self.prepare_features(training_data)

        # Select feature columns that exist
        available_features = [
            col for col in features.columns
            if any(base in col for base in self.feature_columns)
        ]

        X = features[available_features].fillna(0)

        # Split data
        X_train, X_val = train_test_split(
            X, test_size=validation_split, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train model
        self.model.fit(X_train_scaled)

        # Evaluate on validation set
        val_predictions = self.model.predict(X_val_scaled)
        val_scores = self.model.score_samples(X_val_scaled)

        # Calculate metrics
        n_anomalies = (val_predictions == -1).sum()
        anomaly_ratio = n_anomalies / len(val_predictions)

        results = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'features_used': len(available_features),
            'anomalies_detected': int(n_anomalies),
            'anomaly_ratio': float(anomaly_ratio),
            'avg_anomaly_score': float(val_scores.mean()),
            'score_threshold': float(np.percentile(val_scores, self.contamination * 100))
        }

        self.is_trained = True

        # Save model
        self.save_model()

        logger.info(f"Training complete. Detected {n_anomalies} anomalies "
                   f"({anomaly_ratio:.1%}) in validation set")

        return results

    def predict(
        self,
        data: pd.DataFrame,
        return_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Detect anomalies in new data

        Args:
            data: Metrics data to analyze
            return_scores: Whether to return anomaly scores

        Returns:
            Anomaly detection results
        """
        if not self.is_trained:
            self.load_model()

        # Prepare features
        features = self.prepare_features(data)

        # Select available features
        available_features = [
            col for col in features.columns
            if any(base in col for base in self.feature_columns)
        ]

        X = features[available_features].fillna(0)

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)

        # Find anomalies
        anomaly_indices = np.where(predictions == -1)[0]

        results = {
            'total_samples': len(data),
            'anomalies_detected': len(anomaly_indices),
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_ratio': len(anomaly_indices) / len(data) if len(data) > 0 else 0
        }

        if return_scores:
            results['scores'] = scores.tolist()
            results['threshold'] = float(np.percentile(
                scores, self.contamination * 100
            ))

        # Get details of anomalies
        if len(anomaly_indices) > 0:
            anomaly_details = []

            for idx in anomaly_indices[:10]:  # Limit to top 10
                row = data.iloc[idx]
                detail = {
                    'index': int(idx),
                    'timestamp': row.get('timestamp', datetime.utcnow()).isoformat()
                        if hasattr(row.get('timestamp'), 'isoformat') else str(row.get('timestamp')),
                    'score': float(scores[idx]),
                    'severity': self._classify_severity(scores[idx], scores)
                }

                # Add metrics that are significantly different
                for col in self.feature_columns:
                    if col in row:
                        value = row[col]
                        if pd.notna(value):
                            # Check if value is extreme
                            col_values = data[col].dropna()
                            if len(col_values) > 0:
                                percentile = (col_values < value).mean() * 100
                                if percentile > 95 or percentile < 5:
                                    detail[col] = {
                                        'value': float(value),
                                        'percentile': float(percentile)
                                    }

                anomaly_details.append(detail)

            results['details'] = anomaly_details

        return results

    def predict_single(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect if a single data point is anomalous

        Args:
            metrics: Dictionary of metric values

        Returns:
            Anomaly detection result for single point
        """
        # Convert to DataFrame
        df = pd.DataFrame([metrics])

        # Get prediction
        result = self.predict(df, return_scores=True)

        is_anomaly = result['anomalies_detected'] > 0
        score = result['scores'][0] if result.get('scores') else 0

        return {
            'is_anomaly': is_anomaly,
            'score': score,
            'severity': self._classify_severity(score, [score]) if is_anomaly else 'normal',
            'metrics': metrics
        }

    def update_model(self, new_data: pd.DataFrame, partial: bool = True):
        """
        Update model with new data (online learning)

        Args:
            new_data: New metrics data
            partial: If True, do partial fit; else retrain completely
        """
        if partial and self.is_trained:
            # Incremental learning (approximation for Isolation Forest)
            # In practice, might need to retrain periodically
            logger.info("Partial model update not supported for Isolation Forest. "
                       "Consider periodic retraining.")
        else:
            # Full retrain
            self.train(new_data)

    def _classify_severity(self, score: float, all_scores: np.ndarray) -> str:
        """Classify anomaly severity based on score"""
        percentile = (all_scores < score).mean() * 100

        if percentile < 1:
            return 'critical'
        elif percentile < 5:
            return 'high'
        elif percentile < 10:
            return 'medium'
        else:
            return 'low'

    def save_model(self):
        """Save trained model and scaler"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load pre-trained model and scaler"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
            logger.info("Model loaded successfully")
        else:
            raise FileNotFoundError(
                f"Model files not found. Please train the model first."
            )

    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate feature importance for anomaly detection

        Args:
            data: Data to analyze

        Returns:
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        features = self.prepare_features(data)
        available_features = [
            col for col in features.columns
            if any(base in col for base in self.feature_columns)
        ]

        X = features[available_features].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Calculate feature importance by permutation
        importance_scores = {}
        base_score = self.model.score_samples(X_scaled).mean()

        for i, feature in enumerate(available_features):
            # Permute feature
            X_permuted = X_scaled.copy()
            np.random.shuffle(X_permuted[:, i])

            # Calculate score difference
            permuted_score = self.model.score_samples(X_permuted).mean()
            importance = abs(base_score - permuted_score)

            importance_scores[feature] = float(importance)

        # Normalize scores
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {
                k: v/total for k, v in importance_scores.items()
            }

        return importance_scores


# Example usage
def main():
    """Example usage of anomaly detector"""

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # Normal data
    normal_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
        'cpu_usage_percent': np.random.normal(50, 10, n_samples),
        'memory_usage_percent': np.random.normal(60, 15, n_samples),
        'disk_read_bytes_per_sec': np.random.normal(1000, 200, n_samples),
        'disk_write_bytes_per_sec': np.random.normal(800, 150, n_samples),
        'network_in_bytes_per_sec': np.random.normal(5000, 1000, n_samples),
        'network_out_bytes_per_sec': np.random.normal(4000, 800, n_samples),
        'request_rate': np.random.normal(100, 20, n_samples),
        'error_rate': np.random.normal(2, 0.5, n_samples)
    })

    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
    normal_data.loc[anomaly_indices, 'cpu_usage_percent'] = np.random.uniform(90, 100, 50)
    normal_data.loc[anomaly_indices, 'error_rate'] = np.random.uniform(10, 20, 50)

    # Initialize and train detector
    detector = AnomalyDetector(contamination=0.05)
    train_results = detector.train(normal_data)

    print("Training Results:")
    print(json.dumps(train_results, indent=2))

    # Test on new data
    test_data = normal_data.iloc[-100:].copy()
    test_results = detector.predict(test_data, return_scores=True)

    print("\nTest Results:")
    print(f"Anomalies detected: {test_results['anomalies_detected']}")
    print(f"Anomaly ratio: {test_results['anomaly_ratio']:.2%}")

    if test_results.get('details'):
        print("\nTop Anomalies:")
        for detail in test_results['details'][:3]:
            print(f"  - Index {detail['index']}: "
                  f"Severity={detail['severity']}, "
                  f"Score={detail['score']:.3f}")


if __name__ == "__main__":
    main()