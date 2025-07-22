"""
Anomaly Detection Module for QKD Systems

Statistical anomaly detection algorithms for identifying failures and security breaches
in quantum key distribution systems.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings
import logging
from dataclasses import dataclass

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result class for anomaly detection"""

    is_anomaly: bool
    confidence: float
    score: float
    method: str
    timestamp: Optional[float] = None
    details: Optional[Dict] = None
    anomaly_type: Optional[str] = None


class StatisticalAnomalyDetector:
    """Statistical Process Control based anomaly detection"""

    def __init__(self, window_size: int = 50, sigma_threshold: float = 3.0):
        self.window_size = window_size
        self.sigma_threshold = sigma_threshold
        self.baseline_stats = {}
        self.control_limits = {}

    def establish_baseline(self, data: np.ndarray, metric_name: str):
        """Establish baseline statistics for a metric"""
        self.baseline_stats[metric_name] = {
            "mean": np.mean(data),
            "std": np.std(data),
            "median": np.median(data),
            "q25": np.percentile(data, 25),
            "q75": np.percentile(data, 75),
        }

        # Calculate control limits
        mean = self.baseline_stats[metric_name]["mean"]
        std = self.baseline_stats[metric_name]["std"]

        self.control_limits[metric_name] = {
            "ucl": mean + self.sigma_threshold * std,  # Upper Control Limit
            "lcl": max(0, mean - self.sigma_threshold * std),  # Lower Control Limit
            "uwl": mean + 2 * std,  # Upper Warning Limit
            "lwl": max(0, mean - 2 * std),  # Lower Warning Limit
        }

        logger.info(
            f"Baseline established for {metric_name}: mean={mean:.4f}, std={std:.4f}"
        )

    def detect_outliers_zscore(self, data: np.ndarray, metric_name: str) -> np.ndarray:
        """Detect outliers using Z-score method"""
        if metric_name not in self.baseline_stats:
            raise ValueError(f"Baseline not established for {metric_name}")

        mean = self.baseline_stats[metric_name]["mean"]
        std = self.baseline_stats[metric_name]["std"]

        z_scores = np.abs((data - mean) / std)
        return z_scores > self.sigma_threshold

    def detect_outliers_iqr(self, data: np.ndarray, metric_name: str) -> np.ndarray:
        """Detect outliers using Interquartile Range method"""
        if metric_name not in self.baseline_stats:
            raise ValueError(f"Baseline not established for {metric_name}")

        q25 = self.baseline_stats[metric_name]["q25"]
        q75 = self.baseline_stats[metric_name]["q75"]
        iqr = q75 - q25

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        return (data < lower_bound) | (data > upper_bound)

    def control_chart_analysis(
        self, data: np.ndarray, metric_name: str
    ) -> Dict[str, np.ndarray]:
        """Perform control chart analysis"""
        if metric_name not in self.control_limits:
            raise ValueError(f"Control limits not established for {metric_name}")

        limits = self.control_limits[metric_name]

        violations = {
            "out_of_control": (data > limits["ucl"]) | (data < limits["lcl"]),
            "warning_zone": ((data > limits["uwl"]) & (data <= limits["ucl"]))
            | ((data < limits["lwl"]) & (data >= limits["lcl"])),
            "trend_violation": self._detect_trend_violations(data),
            "run_violation": self._detect_run_violations(data, metric_name),
        }

        return violations

    def _detect_trend_violations(
        self, data: np.ndarray, trend_length: int = 7
    ) -> np.ndarray:
        """Detect trend violations (7 consecutive increasing or decreasing points)"""
        violations = np.zeros(len(data), dtype=bool)

        for i in range(trend_length - 1, len(data)):
            window = data[i - trend_length + 1 : i + 1]
            if np.all(np.diff(window) > 0) or np.all(np.diff(window) < 0):
                violations[i] = True

        return violations

    def _detect_run_violations(
        self, data: np.ndarray, metric_name: str, run_length: int = 8
    ) -> np.ndarray:
        """Detect run violations (8 consecutive points on same side of centerline)"""
        violations = np.zeros(len(data), dtype=bool)
        mean = self.baseline_stats[metric_name]["mean"]

        for i in range(run_length - 1, len(data)):
            window = data[i - run_length + 1 : i + 1]
            if np.all(window > mean) or np.all(window < mean):
                violations[i] = True

        return violations


class MLAnomalyDetector:
    """Machine Learning based anomaly detection"""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, features: np.ndarray):
        """Fit the anomaly detection models"""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Fit Isolation Forest
        self.isolation_forest.fit(features_scaled)

        # Fit DBSCAN for clustering-based anomaly detection
        self.dbscan.fit(features_scaled)

        self.is_fitted = True
        logger.info("ML anomaly detection models fitted successfully")

    def predict_anomalies(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict anomalies using multiple ML algorithms"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")

        features_scaled = self.scaler.transform(features)

        # Isolation Forest predictions
        if_predictions = self.isolation_forest.predict(features_scaled)
        if_anomalies = if_predictions == -1

        # DBSCAN predictions (outliers have label -1)
        dbscan_labels = self.dbscan.fit_predict(features_scaled)
        dbscan_anomalies = dbscan_labels == -1

        # Anomaly scores
        anomaly_scores = self.isolation_forest.score_samples(features_scaled)

        return {
            "isolation_forest": if_anomalies,
            "dbscan": dbscan_anomalies,
            "anomaly_scores": -anomaly_scores,  # Higher scores indicate more anomalous
            "combined": if_anomalies | dbscan_anomalies,
        }


class QKDAnomalyDetector:
    """Comprehensive QKD system anomaly detector"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.statistical_detector = StatisticalAnomalyDetector(
            window_size=self.config["window_size"],
            sigma_threshold=self.config["sigma_threshold"],
        )
        self.ml_detector = MLAnomalyDetector(contamination=self.config["contamination"])
        self.detection_history = []

    def _default_config(self) -> Dict:
        """Default configuration for anomaly detection"""
        return {
            "window_size": 50,
            "sigma_threshold": 3.0,
            "contamination": 0.1,
            "qber_threshold": 0.11,
            "sift_ratio_threshold": 0.3,
            "key_length_threshold": 100,
        }

    def extract_features(self, qkd_data: List[Dict]) -> pd.DataFrame:
        """Extract features from QKD session data"""
        features = []

        for session in qkd_data:
            feature_vector = {
                "qber": session.get("qber", 0),
                "sift_ratio": session.get("sift_ratio", 0),
                "final_key_length": session.get("final_key_length", 0),
                "channel_loss": session.get("channel_loss", 0),
                "error_rate": session.get("error_rate", 0),
                "initial_length": session.get("initial_length", 0),
                "sifted_length": session.get("sifted_length", 0),
                "session_id": session.get("session_id", 0),
            }

            # Derived features
            feature_vector["key_efficiency"] = (
                feature_vector["final_key_length"] / feature_vector["initial_length"]
                if feature_vector["initial_length"] > 0
                else 0
            )

            feature_vector["error_amplification"] = (
                feature_vector["qber"] / feature_vector["error_rate"]
                if feature_vector["error_rate"] > 0
                else 1
            )

            features.append(feature_vector)

        return pd.DataFrame(features)

    def establish_baseline(self, training_data: List[Dict]):
        """Establish baseline from normal operation data"""
        features_df = self.extract_features(training_data)

        # Establish statistical baselines for key metrics
        for metric in ["qber", "sift_ratio", "final_key_length", "key_efficiency"]:
            if metric in features_df.columns:
                self.statistical_detector.establish_baseline(
                    features_df[metric].values, metric
                )

        # Fit ML models
        ml_features = features_df[
            [
                "qber",
                "sift_ratio",
                "key_efficiency",
                "channel_loss",
                "error_rate",
                "error_amplification",
            ]
        ].values
        self.ml_detector.fit(ml_features)

        logger.info("Baseline established for QKD anomaly detection")

    def detect(self, data: List[Dict]) -> AnomalyResult:
        """Test-compatible single detection method"""
        result = self.detect_anomalies(data)

        # Calculate overall anomaly status
        is_anomaly = False
        total_score = 0.0
        method_count = 0

        # Check ML anomalies
        if "ml" in result and "isolation_forest" in result["ml"]:
            if_anomalies = result["ml"]["isolation_forest"]
            is_anomaly = is_anomaly or np.any(if_anomalies == -1)
            total_score += np.mean(if_anomalies == -1)
            method_count += 1

        # Check statistical anomalies
        if "statistical" in result:
            for key, anomalies in result["statistical"].items():
                if isinstance(anomalies, np.ndarray) and len(anomalies) > 0:
                    is_anomaly = is_anomaly or np.any(anomalies)
                    total_score += np.mean(anomalies.astype(float))
                    method_count += 1

        # Calculate confidence
        confidence = total_score / max(method_count, 1)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=float(confidence),
            score=float(total_score),
            method="qkd_anomaly_detector",
            details=result,
        )

    def set_threshold(self, threshold: float):
        """Set anomaly detection threshold - test-compatible interface"""
        self.config["sigma_threshold"] = threshold
        self.statistical_detector.sigma_threshold = threshold

    def train(self, training_data: List[Dict]):
        """Train the detector - test-compatible interface (alias for establish_baseline)"""
        self.establish_baseline(training_data)

    def detect_qber_anomaly(self, qber_value: float) -> AnomalyResult:
        """Detect QBER-specific anomaly - test-compatible interface"""
        threshold = self.config.get("qber_threshold", 0.11)
        is_anomaly = qber_value > threshold
        confidence = min(abs(qber_value - threshold) / threshold, 1.0)
        anomaly_type = "high_qber" if is_anomaly else None

        return AnomalyResult(
            is_anomaly=is_anomaly,
            confidence=float(confidence),
            score=float(qber_value),
            method="qber_threshold",
            anomaly_type=anomaly_type,
        )

    def shewhart_control_chart(self, data: np.ndarray) -> Dict:
        """Shewhart control chart analysis - test-compatible interface"""
        mean = np.mean(data)
        std = np.std(data)

        ucl = mean + 3 * std
        lcl = max(0.0, mean - 3 * std)

        violations = (data > ucl) | (data < lcl)

        return {
            "violations": int(np.sum(violations)),
            "violation_indices": violations,
            "upper_limit": float(ucl),
            "lower_limit": float(lcl),
            "center_line": float(mean),
            "ucl": float(ucl),
            "lcl": float(lcl),
            "mean": float(mean),
            "std": float(std),
            "violation_count": int(np.sum(violations)),
        }

    def cusum_analysis(
        self, data: np.ndarray, target: float = None, h: float = 5.0, k: float = 0.5
    ) -> Dict:
        """CUSUM analysis - test-compatible interface"""
        if target is None:
            target = np.mean(data)

        n = len(data)
        pos_cusum = np.zeros(n)
        neg_cusum = np.zeros(n)
        alarm_points = []

        for i in range(1, n):
            pos_cusum[i] = max(0, pos_cusum[i - 1] + data[i] - target - k)
            neg_cusum[i] = max(0, neg_cusum[i - 1] - data[i] + target - k)

            if pos_cusum[i] > h or neg_cusum[i] > h:
                alarm_points.append(i)

        return {
            "positive_cusum": pos_cusum,
            "negative_cusum": neg_cusum,
            "alarm_points": alarm_points,
            "target": target,
            "h_value": h,
            "k_value": k,
        }

    def detect_outliers_zscore(
        self, data: np.ndarray, threshold: float = 3.0
    ) -> List[int]:
        """Z-score outlier detection - test-compatible interface"""
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        outlier_mask = z_scores > threshold
        return [int(i) for i in np.where(outlier_mask)[0]]

    def detect_outliers_iqr(self, data: np.ndarray) -> List[int]:
        """IQR outlier detection - test-compatible interface"""
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outlier_mask = (data < lower) | (data > upper)
        return [int(i) for i in np.where(outlier_mask)[0]]

    def detect_outliers_modified_zscore(
        self, data: np.ndarray, threshold: float = 3.5
    ) -> List[int]:
        """Modified Z-score outlier detection - test-compatible interface"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            mad = 1e-6  # Avoid division by zero
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        return [int(i) for i in np.where(outlier_mask)[0]]

    def detect_outliers(self, data: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Generic outlier detection - test-compatible interface"""
        if method == "zscore":
            z_scores = np.abs(stats.zscore(data))
            return z_scores > 3.0
        elif method == "iqr":
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            return (data < lower) | (data > upper)
        else:
            raise ValueError(f"Unknown method: {method}")

    def time_series_analysis(self, data: List[float], window_size: int = 10) -> Dict:
        """Time series analysis - test-compatible interface"""
        data_array = np.array(data)

        # Moving average
        if len(data_array) >= window_size:
            moving_avg = np.convolve(
                data_array, np.ones(window_size) / window_size, mode="valid"
            )
        else:
            moving_avg = np.array([np.mean(data_array)])

        # Trend detection
        if len(data_array) > 1:
            slope, _ = np.polyfit(range(len(data_array)), data_array, 1)
        else:
            slope = 0.0

        # Volatility
        volatility = np.std(data_array)

        return {
            "moving_average": moving_avg,
            "trend_slope": slope,
            "volatility": volatility,
            "is_trending_up": slope > 0.01,
            "is_trending_down": slope < -0.01,
            "is_volatile": volatility > np.mean(data_array) * 0.2,
        }

    def multivariate_detection(self, data: np.ndarray) -> Dict:
        """Multivariate anomaly detection - test-compatible interface"""
        if not self.ml_detector.is_fitted:
            # Fit on the provided data if not already fitted
            self.ml_detector.fit(data)

        results = self.ml_detector.predict_anomalies(data)

        return {
            "anomalies": results["isolation_forest"],
            "scores": results["anomaly_scores"],
            "method": "isolation_forest",
        }

    def adaptive_threshold(self, data: np.ndarray, sensitivity: float = 0.95) -> float:
        """Adaptive threshold calculation - test-compatible interface"""
        return np.percentile(data, sensitivity * 100)

    def classify_anomaly(self, anomaly_data: Dict) -> str:
        """Classify type of anomaly - test-compatible interface"""
        if "qber" in anomaly_data:
            qber = anomaly_data["qber"]
            if qber > 0.11:
                return "security_breach"
            elif qber > 0.05:
                return "quality_degradation"

        if "sift_ratio" in anomaly_data:
            sift_ratio = anomaly_data["sift_ratio"]
            if sift_ratio < 0.3:
                return "channel_degradation"

        return "unknown_anomaly"

    def real_time_detect(self, new_data: Dict) -> AnomalyResult:
        """Real-time detection for single data point - test-compatible interface"""
        return self.detect([new_data])

    def ensemble_detect(self, data: List[Dict], methods: List[str] = None) -> Dict:
        """Ensemble detection using multiple methods - test-compatible interface"""
        if methods is None:
            methods = ["statistical", "ml", "domain"]

        results = self.detect_anomalies(data)
        ensemble_score = 0.0
        method_count = 0

        for method in methods:
            if method in results:
                if method == "statistical":
                    # Average statistical anomalies
                    stat_scores = [
                        np.mean(anomalies.astype(float))
                        for anomalies in results["statistical"].values()
                    ]
                    ensemble_score += np.mean(stat_scores) if stat_scores else 0
                    method_count += 1
                elif method == "ml":
                    # Use ML combined score
                    ensemble_score += np.mean(results["ml"]["combined"].astype(float))
                    method_count += 1
                elif method == "domain":
                    # Average domain anomalies
                    domain_scores = [
                        np.mean(anomalies.astype(float))
                        for anomalies in results["domain_specific"].values()
                    ]
                    ensemble_score += np.mean(domain_scores) if domain_scores else 0
                    method_count += 1

        ensemble_score = ensemble_score / max(method_count, 1)

        return {
            "ensemble_score": ensemble_score,
            "is_anomaly": ensemble_score > 0.3,
            "individual_results": results,
            "methods_used": methods,
        }

    def detect_anomalies(self, test_data: List[Dict]) -> Dict:
        """Detect anomalies in QKD session data"""
        features_df = self.extract_features(test_data)

        # Statistical anomaly detection
        statistical_anomalies = {}
        for metric in ["qber", "sift_ratio", "final_key_length", "key_efficiency"]:
            if metric in features_df.columns:
                data = features_df[metric].values

                # Multiple statistical tests
                statistical_anomalies[f"{metric}_zscore"] = (
                    self.statistical_detector.detect_outliers_zscore(data, metric)
                )
                statistical_anomalies[f"{metric}_iqr"] = (
                    self.statistical_detector.detect_outliers_iqr(data, metric)
                )

                # Control chart analysis
                control_violations = self.statistical_detector.control_chart_analysis(
                    data, metric
                )
                for violation_type, violations in control_violations.items():
                    statistical_anomalies[f"{metric}_{violation_type}"] = violations

        # ML-based anomaly detection
        ml_features = features_df[
            [
                "qber",
                "sift_ratio",
                "key_efficiency",
                "channel_loss",
                "error_rate",
                "error_amplification",
            ]
        ].values
        ml_anomalies = self.ml_detector.predict_anomalies(ml_features)

        # Domain-specific anomaly detection
        domain_anomalies = self._detect_domain_specific_anomalies(features_df)

        # Combine all anomaly detections
        combined_results = {
            "statistical": statistical_anomalies,
            "ml": ml_anomalies,
            "domain_specific": domain_anomalies,
            "features": features_df,
        }

        # Overall anomaly score
        combined_results["overall_anomaly"] = self._calculate_overall_anomaly_score(
            statistical_anomalies, ml_anomalies, domain_anomalies
        )

        self.detection_history.append(combined_results)
        return combined_results

    def _detect_domain_specific_anomalies(
        self, features_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Detect domain-specific QKD anomalies"""
        anomalies = {}

        # QBER security threshold violation
        anomalies["qber_security_violation"] = (
            features_df["qber"] > self.config["qber_threshold"]
        )

        # Key generation efficiency too low
        anomalies["low_key_efficiency"] = features_df["key_efficiency"] < 0.1

        # Sift ratio too low (indicates high channel loss or timing issues)
        anomalies["low_sift_ratio"] = (
            features_df["sift_ratio"] < self.config["sift_ratio_threshold"]
        )

        # Final key length too short
        anomalies["short_key_length"] = (
            features_df["final_key_length"] < self.config["key_length_threshold"]
        )

        # Error amplification (indicates potential eavesdropping)
        anomalies["error_amplification"] = features_df["error_amplification"] > 2.0

        # Sudden drops in performance
        if len(features_df) > 1:
            qber_diff = np.diff(features_df["qber"])
            sift_diff = np.diff(features_df["sift_ratio"])

            anomalies["sudden_qber_increase"] = np.concatenate(
                [[False], qber_diff > 0.05]
            )
            anomalies["sudden_sift_decrease"] = np.concatenate(
                [[False], sift_diff < -0.2]
            )
        else:
            anomalies["sudden_qber_increase"] = np.array([False] * len(features_df))
            anomalies["sudden_sift_decrease"] = np.array([False] * len(features_df))

        return anomalies

    def _calculate_overall_anomaly_score(
        self, statistical: Dict, ml: Dict, domain: Dict
    ) -> np.ndarray:
        """Calculate overall anomaly score combining all methods"""
        # Count anomalies from each method
        stat_count = sum(anomalies.astype(int) for anomalies in statistical.values())
        ml_count = ml["combined"].astype(int) * 3  # Weight ML detection higher
        domain_count = (
            sum(anomalies.astype(int) for anomalies in domain.values()) * 2
        )  # Weight domain knowledge

        # Combine scores
        total_score = stat_count + ml_count + domain_count

        # Normalize to 0-1 range
        max_possible_score = len(statistical) + 3 + len(domain) * 2
        normalized_score = total_score / max_possible_score

        return normalized_score

    def plot_anomaly_analysis(self, results: Dict, save_path: str = None):
        """Plot comprehensive anomaly analysis"""
        features_df = results["features"]
        overall_anomaly = results["overall_anomaly"]

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # QBER with anomaly overlay
        axes[0, 0].plot(features_df["session_id"], features_df["qber"], "b-", alpha=0.7)
        axes[0, 0].scatter(
            features_df["session_id"][overall_anomaly > 0.3],
            features_df["qber"][overall_anomaly > 0.3],
            color="red",
            s=50,
            label="Anomalies",
        )
        axes[0, 0].axhline(
            y=self.config["qber_threshold"],
            color="r",
            linestyle="--",
            label=f'Security Threshold ({self.config["qber_threshold"]})',
        )
        axes[0, 0].set_title("QBER with Detected Anomalies")
        axes[0, 0].set_xlabel("Session ID")
        axes[0, 0].set_ylabel("QBER")
        axes[0, 0].legend()

        # Sift ratio with anomaly overlay
        axes[0, 1].plot(
            features_df["session_id"], features_df["sift_ratio"], "g-", alpha=0.7
        )
        axes[0, 1].scatter(
            features_df["session_id"][overall_anomaly > 0.3],
            features_df["sift_ratio"][overall_anomaly > 0.3],
            color="red",
            s=50,
            label="Anomalies",
        )
        axes[0, 1].set_title("Sift Ratio with Detected Anomalies")
        axes[0, 1].set_xlabel("Session ID")
        axes[0, 1].set_ylabel("Sift Ratio")
        axes[0, 1].legend()

        # Overall anomaly score
        axes[1, 0].plot(features_df["session_id"], overall_anomaly, "r-", linewidth=2)
        axes[1, 0].axhline(
            y=0.3, color="orange", linestyle="--", label="Anomaly Threshold"
        )
        axes[1, 0].fill_between(features_df["session_id"], overall_anomaly, alpha=0.3)
        axes[1, 0].set_title("Overall Anomaly Score")
        axes[1, 0].set_xlabel("Session ID")
        axes[1, 0].set_ylabel("Anomaly Score")
        axes[1, 0].legend()

        # Key efficiency
        axes[1, 1].plot(
            features_df["session_id"], features_df["key_efficiency"], "m-", alpha=0.7
        )
        axes[1, 1].scatter(
            features_df["session_id"][overall_anomaly > 0.3],
            features_df["key_efficiency"][overall_anomaly > 0.3],
            color="red",
            s=50,
            label="Anomalies",
        )
        axes[1, 1].set_title("Key Generation Efficiency")
        axes[1, 1].set_xlabel("Session ID")
        axes[1, 1].set_ylabel("Key Efficiency")
        axes[1, 1].legend()

        # Anomaly type distribution
        anomaly_types = []
        for method, anomalies_dict in [
            ("Statistical", results["statistical"]),
            ("Domain", results["domain_specific"]),
        ]:
            for anomaly_type, anomalies in anomalies_dict.items():
                anomaly_types.extend([f"{method}: {anomaly_type}"] * np.sum(anomalies))

        if anomaly_types:
            from collections import Counter

            anomaly_counts = Counter(anomaly_types)
            axes[2, 0].barh(list(anomaly_counts.keys()), list(anomaly_counts.values()))
            axes[2, 0].set_title("Anomaly Type Distribution")
            axes[2, 0].set_xlabel("Count")

        # ML anomaly scores distribution
        ml_scores = results["ml"]["anomaly_scores"]
        axes[2, 1].hist(ml_scores, bins=20, alpha=0.7, color="purple")
        axes[2, 1].axvline(
            x=np.percentile(ml_scores, 90),
            color="red",
            linestyle="--",
            label="90th Percentile",
        )
        axes[2, 1].set_title("ML Anomaly Scores Distribution")
        axes[2, 1].set_xlabel("Anomaly Score")
        axes[2, 1].set_ylabel("Frequency")
        axes[2, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def generate_anomaly_report(self, results: Dict) -> str:
        """Generate detailed anomaly detection report"""
        features_df = results["features"]
        overall_anomaly = results["overall_anomaly"]

        report = []
        report.append("QKD SYSTEM ANOMALY DETECTION REPORT")
        report.append("=" * 50)
        report.append(
            f"Analysis Period: Sessions {features_df['session_id'].min()} to {features_df['session_id'].max()}"
        )
        report.append(f"Total Sessions Analyzed: {len(features_df)}")
        report.append(f"Anomalous Sessions Detected: {np.sum(overall_anomaly > 0.3)}")
        report.append(
            f"Anomaly Rate: {np.sum(overall_anomaly > 0.3) / len(features_df) * 100:.2f}%"
        )
        report.append("")

        # Statistical summary
        report.append("STATISTICAL SUMMARY:")
        report.append("-" * 20)
        report.append(
            f"Mean QBER: {features_df['qber'].mean():.4f} ± {features_df['qber'].std():.4f}"
        )
        report.append(
            f"Mean Sift Ratio: {features_df['sift_ratio'].mean():.4f} ± {features_df['sift_ratio'].std():.4f}"
        )
        report.append(
            f"Mean Key Efficiency: {features_df['key_efficiency'].mean():.4f} ± {features_df['key_efficiency'].std():.4f}"
        )
        report.append("")

        # Security violations
        qber_violations = np.sum(features_df["qber"] > self.config["qber_threshold"])
        if qber_violations > 0:
            report.append("SECURITY ALERTS:")
            report.append("-" * 15)
            report.append(f"QBER Security Threshold Violations: {qber_violations}")
            report.append(
                "⚠️  IMMEDIATE ATTENTION REQUIRED - Potential security breach detected!"
            )
            report.append("")

        # Top anomalous sessions
        if np.any(overall_anomaly > 0.3):
            # Get indices of sessions with high anomaly scores
            high_anomaly_mask = overall_anomaly > 0.3
            if np.sum(high_anomaly_mask) > 0:
                high_anomaly_indices = np.where(high_anomaly_mask)[0]
                high_anomaly_scores = overall_anomaly[high_anomaly_indices]
                # Sort by anomaly score and take top 5
                top_indices = high_anomaly_indices[np.argsort(high_anomaly_scores)][-5:]

                report.append("TOP ANOMALOUS SESSIONS:")
                report.append("-" * 25)
                for idx in reversed(top_indices):
                    session_id = features_df.iloc[idx]["session_id"]
                    qber = features_df.iloc[idx]["qber"]
                    score = overall_anomaly[idx]
                    report.append(
                        f"Session {session_id}: Anomaly Score = {score:.3f}, QBER = {qber:.4f}"
                    )

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage with sample data
    from qkd_simulator import QKDSystemSimulator, QKDParameters

    # Generate sample QKD data
    params = QKDParameters(key_length=1000, error_rate=0.02)
    simulator = QKDSystemSimulator(params)

    # Generate baseline data (normal operation)
    baseline_data = simulator.simulate_multiple_sessions(100)

    # Generate test data with some failures
    simulator.inject_failure("eavesdropping", 0.03)
    test_data = simulator.simulate_multiple_sessions(50)

    # Initialize anomaly detector
    detector = QKDAnomalyDetector()

    # Establish baseline
    detector.establish_baseline(baseline_data)

    # Detect anomalies
    results = detector.detect_anomalies(test_data)

    # Generate report
    report = detector.generate_anomaly_report(results)
    print(report)

    # Plot analysis
    detector.plot_anomaly_analysis(results)

# Create aliases for backward compatibility with test modules
AnomalyDetector = QKDAnomalyDetector
