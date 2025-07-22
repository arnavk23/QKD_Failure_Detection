"""
Utility Functions and Helper Classes for QKD Failure Detection

Common utilities, data processing functions, and helper classes used across
the QKD failure detection system.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import logging
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing utilities for QKD analysis"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def clean_qkd_data(self, data: List[Dict]) -> pd.DataFrame:
        """Clean and validate QKD session data"""
        df = pd.DataFrame(data)

        # Remove invalid sessions
        initial_count = len(df)

        # Remove sessions with invalid QBER
        df = df[df["qber"] >= 0]
        df = df[df["qber"] <= 1.0]

        # Remove sessions with invalid sift ratios
        df = df[df["sift_ratio"] >= 0]
        df = df[df["sift_ratio"] <= 1.0]

        # Remove sessions with negative key lengths
        df = df[df["final_key_length"] >= 0]
        df = df[df["initial_length"] > 0]

        # Handle missing values - using newer pandas API
        df = df.ffill().bfill()

        final_count = len(df)
        if final_count != initial_count:
            logger.info(
                f"Cleaned data: removed {initial_count - final_count} invalid sessions"
            )

        return df

    def normalize_features(
        self, data: pd.DataFrame, method: str = "standard"
    ) -> pd.DataFrame:
        """Normalize features using specified method"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        normalized_data = data.copy()
        normalized_data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        self.scalers[method] = scaler
        return normalized_data

    def create_time_windows(
        self, data: pd.DataFrame, window_size: int = 10
    ) -> List[pd.DataFrame]:
        """Create sliding time windows for temporal analysis"""
        windows = []

        for i in range(len(data) - window_size + 1):
            window = data.iloc[i : i + window_size].copy()
            windows.append(window)

        return windows

    def detect_outliers(self, data: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Detect outliers in the data"""
        outlier_mask = pd.Series([False] * len(data), index=data.index)

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if method == "iqr":
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (data[column] < lower_bound) | (data[column] > upper_bound)

            elif method == "zscore":
                z_scores = np.abs(stats.zscore(data[column]))
                outliers = z_scores > 3

            elif method == "modified_zscore":
                median = data[column].median()
                mad = np.median(np.abs(data[column] - median))
                modified_z_scores = 0.6745 * (data[column] - median) / mad
                outliers = np.abs(modified_z_scores) > 3.5

            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

            outlier_mask |= outliers

        return data[~outlier_mask]


class StatisticalAnalyzer:
    """Statistical analysis utilities"""

    @staticmethod
    def calculate_autocorrelation(
        data: np.ndarray, max_lag: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate autocorrelation function"""
        lags = np.arange(0, min(max_lag, len(data)))
        autocorr = [
            (
                np.corrcoef(data[: -lag if lag > 0 else None], data[lag:])[0, 1]
                if lag < len(data)
                else 0
            )
            for lag in lags
        ]
        return lags, np.array(autocorr)

    @staticmethod
    def estimate_autocorr_time(data: np.ndarray) -> float:
        """Estimate autocorrelation time"""
        lags, autocorr = StatisticalAnalyzer.calculate_autocorrelation(data)

        # Find where autocorrelation drops to 1/e
        threshold = 1 / np.e
        crossing_points = np.where(autocorr < threshold)[0]

        if len(crossing_points) > 0:
            return float(lags[crossing_points[0]])
        else:
            return float(len(data))

    @staticmethod
    def jackknife_resample(data: np.ndarray, statistic_func) -> Tuple[float, float]:
        """Jackknife resampling for error estimation"""
        n = len(data)
        jackknife_estimates = []

        for i in range(n):
            # Remove one data point
            jackknife_sample = np.delete(data, i)
            estimate = statistic_func(jackknife_sample)
            jackknife_estimates.append(estimate)

        jackknife_mean = np.mean(jackknife_estimates)
        jackknife_std = np.sqrt(
            (n - 1) / n * np.sum((jackknife_estimates - jackknife_mean) ** 2)
        )

        return jackknife_mean, jackknife_std

    @staticmethod
    def bootstrap_resample(
        data: np.ndarray, statistic_func, n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Bootstrap resampling for error estimation"""
        bootstrap_estimates = []
        n = len(data)

        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            estimate = statistic_func(bootstrap_sample)
            bootstrap_estimates.append(estimate)

        bootstrap_mean = np.mean(bootstrap_estimates)
        bootstrap_std = np.std(bootstrap_estimates)

        return bootstrap_mean, bootstrap_std

    @staticmethod
    def calculate_effective_sample_size(data: np.ndarray) -> float:
        """Calculate effective sample size accounting for autocorrelation"""
        autocorr_time = StatisticalAnalyzer.estimate_autocorr_time(data)
        effective_size = len(data) / (2 * autocorr_time + 1)
        return max(1.0, effective_size)


class PerformanceEvaluator:
    """Performance evaluation utilities for detection algorithms"""

    def __init__(self):
        self.evaluation_history = []

    def evaluate_binary_classification(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None
    ) -> Dict[str, float]:
        """Evaluate binary classification performance"""
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average="binary")
        metrics["recall"] = recall_score(y_true, y_pred, average="binary")
        metrics["f1_score"] = f1_score(y_true, y_pred, average="binary")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positive_rate"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics["true_negative_rate"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # ROC AUC if scores provided
        if y_scores is not None:
            try:
                from sklearn.metrics import roc_auc_score

                metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
            except:
                metrics["roc_auc"] = 0.5

        return metrics

    def evaluate_anomaly_detection(
        self,
        normal_data: np.ndarray,
        anomaly_data: np.ndarray,
        detection_function,
        threshold: float = None,
    ) -> Dict[str, float]:
        """Evaluate anomaly detection performance"""
        # Generate predictions
        normal_scores = detection_function(normal_data)
        anomaly_scores = detection_function(anomaly_data)

        # Determine threshold if not provided
        if threshold is None:
            # Use 95th percentile of normal scores
            threshold = np.percentile(normal_scores, 95)

        # Create labels and predictions
        y_true = np.concatenate(
            [np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))]
        )
        y_scores = np.concatenate([normal_scores, anomaly_scores])
        y_pred = (y_scores > threshold).astype(int)

        # Evaluate
        metrics = self.evaluate_binary_classification(y_true, y_pred, y_scores)
        metrics["threshold"] = threshold

        return metrics

    def calculate_detection_latency(
        self,
        timestamps: List[datetime],
        true_anomaly_times: List[datetime],
        detected_anomaly_times: List[datetime],
    ) -> Dict[str, float]:
        """Calculate detection latency metrics"""
        latencies = []

        for true_time in true_anomaly_times:
            # Find nearest detection time after true anomaly
            future_detections = [dt for dt in detected_anomaly_times if dt >= true_time]

            if future_detections:
                detection_time = min(future_detections)
                latency = (detection_time - true_time).total_seconds()
                latencies.append(latency)

        if latencies:
            return {
                "mean_latency": np.mean(latencies),
                "median_latency": np.median(latencies),
                "max_latency": np.max(latencies),
                "detection_rate": len(latencies) / len(true_anomaly_times),
            }
        else:
            return {
                "mean_latency": float("inf"),
                "median_latency": float("inf"),
                "max_latency": float("inf"),
                "detection_rate": 0.0,
            }


class VisualizationUtils:
    """Visualization utilities for QKD analysis"""

    @staticmethod
    def plot_time_series(
        data: pd.DataFrame,
        columns: List[str],
        title: str = "Time Series Plot",
        save_path: str = None,
    ):
        """Plot time series data"""
        fig, axes = plt.subplots(
            len(columns), 1, figsize=(12, 3 * len(columns)), sharex=True
        )

        if len(columns) == 1:
            axes = [axes]

        for i, column in enumerate(columns):
            if column in data.columns:
                axes[i].plot(data.index, data[column], linewidth=1.5)
                axes[i].set_ylabel(column)
                axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time")
        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_correlation_matrix(
        data: pd.DataFrame, title: str = "Correlation Matrix", save_path: str = None
    ):
        """Plot correlation matrix heatmap"""
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_distribution_comparison(
        data1: np.ndarray,
        data2: np.ndarray,
        labels: List[str],
        title: str = "Distribution Comparison",
        save_path: str = None,
    ):
        """Plot distribution comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histograms
        axes[0].hist(data1, bins=30, alpha=0.7, label=labels[0], density=True)
        axes[0].hist(data2, bins=30, alpha=0.7, label=labels[1], density=True)
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Distribution Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plots
        axes[1].boxplot([data1, data2], labels=labels)
        axes[1].set_ylabel("Value")
        axes[1].set_title("Box Plot Comparison")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
        save_path: str = None,
    ):
        """Plot ROC curve"""
        try:
            from sklearn.metrics import roc_curve, auc

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        except ImportError:
            logger.warning("sklearn not available for ROC curve plotting")


class ConfigManager:
    """Configuration management for QKD system"""

    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = self._load_default_config()

        if config_file:
            self.load_config(config_file)

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "qkd_system": {
                "protocols": ["BB84", "SARG04", "E91"],
                "default_protocol": "BB84",
                "key_length": 1000,
                "security_threshold": 0.11,
            },
            "detection": {
                "anomaly_threshold": 0.3,
                "ml_contamination": 0.1,
                "statistical_sigma": 3.0,
                "window_size": 50,
            },
            "monitoring": {
                "real_time_enabled": True,
                "alert_threshold": 0.7,
                "log_level": "INFO",
            },
            "analysis": {
                "autocorr_max_lag": 50,
                "bootstrap_samples": 1000,
                "confidence_interval": 0.95,
            },
        }

    def load_config(self, config_file: str):
        """Load configuration from file"""
        try:
            with open(config_file, "r") as f:
                file_config = json.load(f)

            # Deep merge with default config
            self.config = self._deep_merge(self.config, file_config)
            logger.info(f"Configuration loaded from {config_file}")

        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file {config_file}")

    def save_config(self, config_file: str):
        """Save current configuration to file"""
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {config_file}")

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split(".")
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()

        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


class DataLogger:
    """Data logging utilities"""

    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.session_data = []

    def log_session(self, session_data: Dict):
        """Log QKD session data"""
        session_data["timestamp"] = datetime.now().isoformat()
        self.session_data.append(session_data)

        if self.log_file:
            self._write_to_file(session_data)

    def _write_to_file(self, data: Dict):
        """Write data to log file"""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def export_sessions(self, filename: str, format: str = "json"):
        """Export session data to file"""
        if format == "json":
            with open(filename, "w") as f:
                json.dump(self.session_data, f, indent=2)
        elif format == "csv":
            df = pd.DataFrame(self.session_data)
            df.to_csv(filename, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(self.session_data)} sessions to {filename}")


def calculate_quantum_fisher_information(qber: float) -> float:
    """Calculate quantum Fisher information for parameter estimation"""
    if qber <= 0 or qber >= 1:
        return 0.0

    # Simplified model for QFI in QKD
    fisher_info = 1 / (qber * (1 - qber))
    return fisher_info


def estimate_key_rate(qber: float, protocol: str = "BB84") -> float:
    """Estimate secret key rate for given QBER and protocol"""
    if qber >= 0.11:  # Above security threshold
        return 0.0

    def binary_entropy(p):
        if p <= 0 or p >= 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    if protocol == "BB84":
        # Simplified GLLP formula
        key_rate = 1 - 2 * binary_entropy(qber)
    elif protocol == "SARG04":
        # SARG04 specific formula
        key_rate = 0.5 * (1 - 2 * binary_entropy(qber))
    else:
        # Default to BB84
        key_rate = 1 - 2 * binary_entropy(qber)

    return max(0, key_rate)


def calculate_channel_capacity(snr_db: float) -> float:
    """Calculate channel capacity using Shannon's formula"""
    snr_linear = 10 ** (snr_db / 10)
    capacity = np.log2(1 + snr_linear)  # bits per channel use
    return capacity


if __name__ == "__main__":
    # Example usage of utilities

    # Test data preprocessing
    preprocessor = DataPreprocessor()

    # Sample QKD data
    sample_data = [
        {
            "qber": 0.02,
            "sift_ratio": 0.5,
            "final_key_length": 800,
            "initial_length": 1000,
        },
        {
            "qber": 0.03,
            "sift_ratio": 0.48,
            "final_key_length": 750,
            "initial_length": 1000,
        },
        {
            "qber": 0.15,
            "sift_ratio": 0.45,
            "final_key_length": 0,
            "initial_length": 1000,
        },  # Invalid
    ]

    cleaned_data = preprocessor.clean_qkd_data(sample_data)
    print(f"Cleaned data shape: {cleaned_data.shape}")

    # Test statistical analysis
    data = np.random.randn(1000)
    autocorr_time = StatisticalAnalyzer.estimate_autocorr_time(data)
    print(f"Autocorrelation time: {autocorr_time:.2f}")

    # Test performance evaluation
    evaluator = PerformanceEvaluator()
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    metrics = evaluator.evaluate_binary_classification(y_true, y_pred)
    print(f"Classification metrics: {metrics}")

    # Test configuration management
    config = ConfigManager()
    print(f"Default security threshold: {config.get('qkd_system.security_threshold')}")

    # Test key rate calculation
    key_rate = estimate_key_rate(0.05)
    print(f"Estimated key rate for QBER=0.05: {key_rate:.3f}")

# Create aliases for backward compatibility with test modules
DataProcessor = DataPreprocessor
Logger = DataLogger


# Additional utility classes for test compatibility
class TimeUtils:
    """Time utility functions"""

    @staticmethod
    def current_timestamp():
        """Get current timestamp"""
        import time

        return time.time()

    @staticmethod
    def format_timestamp(timestamp):
        """Format timestamp for display"""
        from datetime import datetime

        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class ValidationUtils:
    """Validation utility functions"""

    @staticmethod
    def validate_qber(qber):
        """Validate QBER value"""
        return 0 <= qber <= 0.5

    @staticmethod
    def validate_key_rate(rate):
        """Validate key rate"""
        return rate >= 0


class FileHandler:
    """File handling utilities"""

    @staticmethod
    def save_data(data, filename):
        """Save data to file"""
        import json

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_data(filename):
        """Load data from file"""
        import json

        with open(filename, "r") as f:
            return json.load(f)


class MathUtils:
    """Mathematical utility functions"""

    @staticmethod
    def calculate_entropy(data: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        values, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    @staticmethod
    def calculate_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between two variables"""
        from sklearn.metrics import mutual_info_score

        return mutual_info_score(x, y)

    @staticmethod
    def moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average"""
        return np.convolve(data, np.ones(window) / window, mode="valid")

    @staticmethod
    def exponential_smoothing(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Apply exponential smoothing"""
        result = [data[0]]
        for i in range(1, len(data)):
            result.append(alpha * data[i] + (1 - alpha) * result[-1])
        return np.array(result)


class MetricsCalculator:
    """Calculate various performance metrics"""

    @staticmethod
    def calculate_qkd_metrics(sessions: List[Dict]) -> Dict:
        """Calculate QKD performance metrics"""
        df = pd.DataFrame(sessions)
        return {
            "avg_qber": df["qber"].mean(),
            "avg_sift_ratio": df["sift_ratio"].mean(),
            "avg_key_rate": df.get("key_rate", pd.Series([0])).mean(),
            "total_sessions": len(df),
        }

    @staticmethod
    def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate detection performance metrics"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }


# Create aliases for backward compatibility with tests
DataProcessor = DataPreprocessor
Logger = DataLogger
