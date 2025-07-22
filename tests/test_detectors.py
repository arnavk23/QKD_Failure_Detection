"""
Comprehensive Test Suite for QKD Failure Detection System

Unit tests and integration tests for all components of the QKD failure detection system.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qkd_simulator import QKDSystemSimulator, QKDParameters, BB84Protocol
from anomaly_detector import QKDAnomalyDetector, StatisticalAnomalyDetector
from ml_detector import MLDetectionSystem, FeatureEngineer
from signal_analyzer import QKDSignalAnalyzer, FrequencyAnalyzer
from security_monitor import QKDSecuritySystem, EavesdroppingDetector
from utils import DataPreprocessor, StatisticalAnalyzer, PerformanceEvaluator


class TestQKDSimulator(unittest.TestCase):
    """Test QKD system simulation components"""

    def setUp(self):
        self.params = QKDParameters(
            key_length=100, error_rate=0.02, detection_efficiency=0.8, channel_loss=0.05
        )
        self.simulator = QKDSystemSimulator(self.params)

    def test_bb84_protocol_initialization(self):
        """Test BB84 protocol initialization"""
        protocol = BB84Protocol(self.params)
        self.assertIsInstance(protocol, BB84Protocol)
        self.assertEqual(protocol.params.key_length, 100)

    def test_random_bit_generation(self):
        """Test random bit generation"""
        protocol = BB84Protocol(self.params)
        bits = protocol.generate_random_bits(10)

        self.assertEqual(len(bits), 10)
        self.assertTrue(all(bit in [0, 1] for bit in bits))

    def test_random_basis_generation(self):
        """Test random basis generation"""
        protocol = BB84Protocol(self.params)
        bases = protocol.generate_random_bases(10)

        self.assertEqual(len(bases), 10)
        self.assertTrue(all(basis in ["+", "x"] for basis in bases))

    def test_qkd_session_simulation(self):
        """Test complete QKD session simulation"""
        result = self.simulator.simulate_session()

        self.assertIn("qber", result)
        self.assertIn("final_key_length", result)
        self.assertIn("secure", result)
        self.assertGreaterEqual(result["qber"], 0)
        self.assertLessEqual(result["qber"], 1)

    def test_multiple_sessions(self):
        """Test multiple session simulation"""
        results = self.simulator.simulate_multiple_sessions(5)

        self.assertEqual(len(results), 5)
        self.assertTrue(all("qber" in result for result in results))

    def test_failure_injection(self):
        """Test failure injection"""
        original_loss = self.simulator.params.channel_loss
        self.simulator.inject_failure("channel_loss", 0.1)

        self.assertGreater(self.simulator.params.channel_loss, original_loss)

    def test_system_statistics(self):
        """Test system statistics calculation"""
        self.simulator.simulate_multiple_sessions(10)
        stats = self.simulator.get_system_statistics()

        self.assertIn("total_sessions", stats)
        self.assertIn("mean_qber", stats)
        self.assertEqual(stats["total_sessions"], 10)


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detection components"""

    def setUp(self):
        self.detector = StatisticalAnomalyDetector()
        self.qkd_detector = QKDAnomalyDetector()

        # Generate sample data
        self.normal_data = np.random.normal(0.02, 0.005, 100)  # Normal QBER
        self.anomaly_data = np.random.normal(0.15, 0.02, 20)  # High QBER

    def test_baseline_establishment(self):
        """Test baseline establishment"""
        self.detector.establish_baseline(self.normal_data, "test_metric")

        self.assertIn("test_metric", self.detector.baseline_stats)
        self.assertIn("mean", self.detector.baseline_stats["test_metric"])

    def test_zscore_outlier_detection(self):
        """Test Z-score based outlier detection"""
        self.detector.establish_baseline(self.normal_data, "test_metric")

        # Test with anomalous data
        combined_data = np.concatenate([self.normal_data, self.anomaly_data])
        outliers = self.detector.detect_outliers_zscore(combined_data, "test_metric")

        self.assertIsInstance(outliers, np.ndarray)
        self.assertEqual(len(outliers), len(combined_data))
        # Should detect some outliers in the anomalous part
        self.assertTrue(np.any(outliers[-20:]))

    def test_iqr_outlier_detection(self):
        """Test IQR based outlier detection"""
        self.detector.establish_baseline(self.normal_data, "test_metric")

        combined_data = np.concatenate([self.normal_data, self.anomaly_data])
        outliers = self.detector.detect_outliers_iqr(combined_data, "test_metric")

        self.assertIsInstance(outliers, np.ndarray)
        self.assertEqual(len(outliers), len(combined_data))

    def test_control_chart_analysis(self):
        """Test control chart analysis"""
        self.detector.establish_baseline(self.normal_data, "test_metric")

        violations = self.detector.control_chart_analysis(
            self.anomaly_data, "test_metric"
        )

        self.assertIn("out_of_control", violations)
        self.assertIn("warning_zone", violations)
        self.assertIsInstance(violations["out_of_control"], np.ndarray)

    def test_qkd_feature_extraction(self):
        """Test QKD feature extraction"""
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
        ]

        features = self.qkd_detector.extract_features(sample_data)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertIn("qber", features.columns)
        self.assertIn("key_efficiency", features.columns)


class TestMLDetector(unittest.TestCase):
    """Test machine learning detection components"""

    def setUp(self):
        self.ml_system = MLDetectionSystem()
        self.feature_engineer = FeatureEngineer()

        # Generate sample QKD data
        self.sample_data = []
        for i in range(50):
            session = {
                "session_id": i,
                "qber": np.random.normal(0.02, 0.005),
                "sift_ratio": np.random.normal(0.5, 0.05),
                "final_key_length": np.random.randint(700, 900),
                "initial_length": 1000,
                "channel_loss": np.random.normal(0.05, 0.01),
                "error_rate": np.random.normal(0.02, 0.005),
            }
            self.sample_data.append(session)

    def test_feature_engineering(self):
        """Test feature engineering"""
        df = pd.DataFrame(self.sample_data)

        # Add basic derived features
        df["key_efficiency"] = df["final_key_length"] / df["initial_length"]

        features = self.feature_engineer.extract_temporal_features(df)

        self.assertIsInstance(features, pd.DataFrame)
        # Should have rolling statistics
        rolling_cols = [
            col for col in features.columns if "mean_" in col or "std_" in col
        ]
        self.assertGreater(len(rolling_cols), 0)

    def test_ml_system_initialization(self):
        """Test ML system initialization"""
        self.assertIsInstance(self.ml_system, MLDetectionSystem)
        self.assertIsNotNone(self.ml_system.config)

    def test_data_preparation(self):
        """Test data preparation for ML"""
        features = self.ml_system.prepare_data(self.sample_data)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(
            len(features.columns), len(self.sample_data[0])
        )  # Should have engineered features

    @patch("joblib.dump")
    def test_model_saving(self, mock_dump):
        """Test model saving functionality"""
        self.ml_system.save_models("test_path.joblib")
        mock_dump.assert_called_once()


class TestSignalAnalyzer(unittest.TestCase):
    """Test signal processing and analysis components"""

    def setUp(self):
        self.analyzer = QKDSignalAnalyzer()
        self.freq_analyzer = FrequencyAnalyzer()

        # Generate sample QKD session
        self.sample_session = {
            "session_id": 1,
            "qber": 0.02,
            "sift_ratio": 0.5,
            "initial_length": 1000,
            "channel_loss": 0.05,
        }

    def test_signal_simulation(self):
        """Test quantum signal simulation"""
        signal = self.analyzer.signal_processor.simulate_quantum_signal(
            self.sample_session
        )

        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(len(signal), self.sample_session["initial_length"])

    def test_signal_feature_extraction(self):
        """Test signal feature extraction"""
        signal = np.random.randn(1000)
        features = self.analyzer.signal_processor.extract_signal_features(signal)

        self.assertIsInstance(features, dict)
        self.assertIn("mean", features)
        self.assertIn("std", features)
        self.assertIn("rms", features)

    def test_fft_analysis(self):
        """Test FFT analysis"""
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))  # 10 Hz sine wave
        freqs, magnitudes = self.freq_analyzer.fft_analysis(signal)

        self.assertIsInstance(freqs, np.ndarray)
        self.assertIsInstance(magnitudes, np.ndarray)
        self.assertEqual(len(freqs), len(magnitudes))

        # Should detect peak around 10 Hz
        peak_idx = np.argmax(magnitudes)
        peak_freq = freqs[peak_idx]
        self.assertAlmostEqual(peak_freq, 10.0, delta=2.0)

    def test_session_analysis(self):
        """Test complete session analysis"""
        result = self.analyzer.analyze_qkd_session(self.sample_session)

        self.assertIsInstance(result, dict)
        self.assertIn("time_features", result)
        self.assertIn("frequency_features", result)
        self.assertIn("quality_metrics", result)


class TestSecurityMonitor(unittest.TestCase):
    """Test security monitoring components"""

    def setUp(self):
        self.security_system = QKDSecuritySystem()
        self.eavesdrop_detector = EavesdroppingDetector()

        # Sample secure session
        self.secure_session = {
            "qber": 0.02,
            "sift_ratio": 0.5,
            "final_key_length": 800,
            "channel_loss": 0.05,
            "session_id": 1,
        }

        # Sample insecure session
        self.insecure_session = {
            "qber": 0.15,  # Above security threshold
            "sift_ratio": 0.3,
            "final_key_length": 0,
            "channel_loss": 0.2,
            "session_id": 2,
        }

    def test_eavesdropping_detection_initialization(self):
        """Test eavesdropping detector initialization"""
        self.assertIsInstance(self.eavesdrop_detector, EavesdroppingDetector)
        self.assertEqual(self.eavesdrop_detector.security_threshold, 0.11)

    def test_intercept_resend_detection(self):
        """Test intercept-resend attack detection"""
        # Test with secure session
        result_secure = self.eavesdrop_detector.detect_intercept_resend_attack(
            self.secure_session
        )
        self.assertFalse(result_secure["attack_detected"])

        # Test with insecure session
        result_insecure = self.eavesdrop_detector.detect_intercept_resend_attack(
            self.insecure_session
        )
        self.assertTrue(result_insecure["attack_detected"])
        self.assertGreater(result_insecure["confidence"], 0)

    def test_beam_splitting_detection(self):
        """Test beam splitting attack detection"""
        result = self.eavesdrop_detector.detect_beam_splitting_attack(
            self.insecure_session
        )

        self.assertIsInstance(result, dict)
        self.assertIn("attack_detected", result)
        self.assertIn("confidence", result)
        self.assertIn("indicators", result)

    def test_security_system_initialization(self):
        """Test security system initialization"""
        baseline_sessions = [self.secure_session] * 10
        self.security_system.initialize_system(baseline_sessions)

        # Should have established baseline
        self.assertIsNotNone(self.security_system.eavesdropper.baseline_stats)

    def test_session_monitoring(self):
        """Test individual session monitoring"""
        # Initialize with baseline
        baseline_sessions = [self.secure_session] * 10
        self.security_system.initialize_system(baseline_sessions)

        # Monitor insecure session
        result = self.security_system.monitor_session(self.insecure_session)

        self.assertIsInstance(result, dict)
        self.assertIn("attack_detected", result)
        self.assertIn("overall_security_score", result)


class TestUtils(unittest.TestCase):
    """Test utility functions and helper classes"""

    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.evaluator = PerformanceEvaluator()

        # Sample data
        self.sample_qkd_data = [
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
                "qber": -0.01,
                "sift_ratio": 1.5,
                "final_key_length": -100,
                "initial_length": 0,
            },  # Invalid
        ]

    def test_data_cleaning(self):
        """Test data cleaning functionality"""
        cleaned_data = self.preprocessor.clean_qkd_data(self.sample_qkd_data)

        self.assertIsInstance(cleaned_data, pd.DataFrame)
        # Should remove invalid data
        self.assertLess(len(cleaned_data), len(self.sample_qkd_data))
        # All remaining data should be valid
        self.assertTrue(all(cleaned_data["qber"] >= 0))
        self.assertTrue(all(cleaned_data["sift_ratio"] <= 1.0))
        self.assertTrue(all(cleaned_data["final_key_length"] >= 0))

    def test_normalization(self):
        """Test feature normalization"""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]}
        )

        normalized = self.preprocessor.normalize_features(df, method="standard")

        self.assertIsInstance(normalized, pd.DataFrame)
        # Should have zero mean and unit variance (approximately)
        self.assertAlmostEqual(normalized["feature1"].mean(), 0, places=5)
        self.assertAlmostEqual(
            normalized["feature1"].std(ddof=0), 1, places=5
        )  # Use population std

    def test_autocorrelation_calculation(self):
        """Test autocorrelation calculation"""
        # Generate correlated data
        data = np.cumsum(np.random.randn(100))  # Random walk (highly correlated)

        lags, autocorr = StatisticalAnalyzer.calculate_autocorrelation(data, max_lag=20)

        self.assertEqual(len(lags), len(autocorr))
        self.assertAlmostEqual(
            autocorr[0], 1.0, places=5
        )  # Perfect correlation at lag 0 (with tolerance)
        self.assertLess(autocorr[-1], autocorr[0])  # Decreasing correlation

    def test_performance_evaluation(self):
        """Test performance evaluation"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        metrics = self.evaluator.evaluate_binary_classification(y_true, y_pred)

        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)

        # Values should be between 0 and 1
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete system"""

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # 1. Generate QKD data
        params = QKDParameters(key_length=100, error_rate=0.02)
        simulator = QKDSystemSimulator(params)
        sessions = simulator.simulate_multiple_sessions(20)

        # 2. Anomaly detection
        detector = QKDAnomalyDetector()
        detector.establish_baseline(sessions[:10])
        anomaly_results = detector.detect_anomalies(sessions[10:])

        # 3. Signal analysis
        analyzer = QKDSignalAnalyzer()
        analyzer.establish_signal_baseline(sessions[:10])
        signal_results = analyzer.batch_analyze_sessions(sessions[10:])

        # 4. Security monitoring
        security_system = QKDSecuritySystem()
        security_system.initialize_system(sessions[:10])
        security_results = security_system.batch_security_analysis(sessions[10:])

        # Verify all components produced results
        self.assertIsNotNone(anomaly_results)
        self.assertEqual(len(signal_results), 10)
        self.assertEqual(len(security_results), 10)

        # Check result structure
        self.assertIn("overall_anomaly", anomaly_results)
        self.assertIn("time_features", signal_results[0])
        self.assertIn("attack_detected", security_results[0])


def run_performance_benchmarks():
    """Run performance benchmarks for key components"""
    import time

    print("\nRunning performance benchmarks...")

    # Benchmark QKD simulation
    params = QKDParameters(key_length=1000)
    simulator = QKDSystemSimulator(params)

    start_time = time.time()
    simulator.simulate_multiple_sessions(100)
    simulation_time = time.time() - start_time
    print(f"QKD Simulation (100 sessions): {simulation_time:.2f} seconds")

    # Benchmark anomaly detection
    detector = QKDAnomalyDetector()
    normal_data = [
        {
            "qber": 0.02,
            "sift_ratio": 0.5,
            "final_key_length": 800,
            "initial_length": 1000,
        }
    ] * 100

    start_time = time.time()
    detector.establish_baseline(normal_data)
    results = detector.detect_anomalies(normal_data)
    detection_time = time.time() - start_time
    print(f"Anomaly Detection (100 sessions): {detection_time:.2f} seconds")

    print("Performance benchmarks completed.")


if __name__ == "__main__":
    # Run unit tests
    print("QKD Failure Detection System - Test Suite")
    print("=" * 50)
    print("Under the guidance of Vijayalaxmi Mogiligidda")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestQKDSimulator,
        TestAnomalyDetector,
        TestMLDetector,
        TestSignalAnalyzer,
        TestSecurityMonitor,
        TestUtils,
        TestIntegration,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%"
    )

    if result.failures:
        print("\nFAILURES:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")

    if result.errors:
        print("\nERRORS:")
        for test, error in result.errors:
            print(f"- {test}: {error}")

    # Run performance benchmarks if tests passed
    if not result.failures and not result.errors:
        run_performance_benchmarks()

    print("\nTest suite completed.")
