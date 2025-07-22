"""
Test suite for utility functions and helper modules.

This module contains comprehensive tests for utility functions,
data processing helpers, and common functionality.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    DataProcessor,
    ConfigManager,
    Logger,
    MathUtils,
    FileHandler,
    TimeUtils,
    ValidationUtils,
    MetricsCalculator,
)


class TestDataProcessor:
    """Test suite for data processing utilities."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.processor = DataProcessor()

        # Create test data
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            {
                "qber": np.random.normal(0.05, 0.01, 1000),
                "key_rate": np.random.normal(1000, 100, 1000),
                "timestamp": pd.date_range("2025-01-01", periods=1000, freq="1min"),
                "detector_efficiency": np.random.uniform(0.7, 0.9, 1000),
                "channel_loss": np.random.uniform(0.05, 0.15, 1000),
            }
        )

        # Add some missing values for testing
        self.test_data.loc[100:110, "qber"] = np.nan
        self.test_data.loc[500:505, "key_rate"] = np.nan

    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        # Test missing value handling
        cleaned_data = self.processor.handle_missing_values(
            self.test_data, method="interpolate"
        )

        assert cleaned_data.isnull().sum().sum() == 0  # No missing values
        assert len(cleaned_data) == len(self.test_data)  # Same length

        # Test outlier removal
        outlier_removed = self.processor.remove_outliers(
            self.test_data, columns=["qber", "key_rate"], method="iqr"
        )

        assert len(outlier_removed) <= len(self.test_data)  # Outliers removed

        # Test data validation
        validation_result = self.processor.validate_data(self.test_data)

        assert "missing_values" in validation_result
        assert "outliers" in validation_result
        assert "data_types" in validation_result
        assert "range_violations" in validation_result

    def test_data_transformation(self):
        """Test data transformation methods."""
        # Test normalization
        normalized = self.processor.normalize_data(
            self.test_data[["qber", "key_rate"]], method="standard"
        )

        # Check that means are close to 0 and stds close to 1
        means = normalized.mean()
        stds = normalized.std()

        assert all(abs(mean) < 0.1 for mean in means)
        assert all(abs(std - 1.0) < 0.1 for std in stds)

        # Test feature scaling
        scaled = self.processor.scale_features(
            self.test_data[["qber", "key_rate"]], method="minmax"
        )

        assert scaled.min().min() >= 0.0
        assert scaled.max().max() <= 1.0

        # Test log transformation
        log_transformed = self.processor.log_transform(self.test_data["key_rate"])

        assert len(log_transformed) == len(self.test_data)
        assert all(log_transformed > 0)  # Log values should be positive

    def test_feature_engineering(self):
        """Test feature engineering capabilities."""
        # Test rolling statistics
        rolling_features = self.processor.create_rolling_features(
            self.test_data, columns=["qber", "key_rate"], windows=[5, 10, 20]
        )

        expected_cols = []
        for col in ["qber", "key_rate"]:
            for window in [5, 10, 20]:
                expected_cols.extend(
                    [
                        f"{col}_rolling_mean_{window}",
                        f"{col}_rolling_std_{window}",
                        f"{col}_rolling_min_{window}",
                        f"{col}_rolling_max_{window}",
                    ]
                )

        for col in expected_cols:
            assert col in rolling_features.columns

        # Test lag features
        lag_features = self.processor.create_lag_features(
            self.test_data, columns=["qber", "key_rate"], lags=[1, 2, 3, 5]
        )

        for col in ["qber", "key_rate"]:
            for lag in [1, 2, 3, 5]:
                assert f"{col}_lag_{lag}" in lag_features.columns

        # Test time-based features
        time_features = self.processor.create_time_features(
            self.test_data, timestamp_col="timestamp"
        )

        expected_time_cols = [
            "hour",
            "day_of_week",
            "month",
            "quarter",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        ]

        for col in expected_time_cols:
            assert col in time_features.columns

    def test_data_aggregation(self):
        """Test data aggregation methods."""
        # Test time-based aggregation
        hourly_agg = self.processor.aggregate_by_time(
            self.test_data,
            timestamp_col="timestamp",
            freq="H",
            agg_funcs=["mean", "std", "min", "max", "count"],
        )

        assert len(hourly_agg) < len(
            self.test_data
        )  # Aggregated data should be shorter

        # Check aggregation columns
        for col in ["qber", "key_rate"]:
            for func in ["mean", "std", "min", "max"]:
                assert f"{col}_{func}" in hourly_agg.columns

        # Test custom aggregation
        custom_agg = self.processor.custom_aggregate(
            self.test_data,
            group_by=["detector_efficiency"],
            agg_functions={
                "qber": ["mean", "median", "std"],
                "key_rate": ["mean", "min", "max"],
            },
        )

        assert "qber_mean" in custom_agg.columns
        assert "key_rate_mean" in custom_agg.columns

    def test_data_filtering(self):
        """Test data filtering methods."""
        # Test range filtering
        filtered_qber = self.processor.filter_by_range(
            self.test_data, "qber", min_val=0.02, max_val=0.08
        )

        assert all(filtered_qber["qber"] >= 0.02)
        assert all(filtered_qber["qber"] <= 0.08)

        # Test conditional filtering
        high_efficiency = self.processor.filter_by_condition(
            self.test_data, "detector_efficiency > 0.8"
        )

        assert all(high_efficiency["detector_efficiency"] > 0.8)

        # Test time-based filtering
        time_filtered = self.processor.filter_by_time(
            self.test_data,
            timestamp_col="timestamp",
            start_time="2025-01-01 06:00:00",
            end_time="2025-01-01 18:00:00",
        )

        assert len(time_filtered) < len(self.test_data)

    def test_statistical_processing(self):
        """Test statistical processing methods."""
        # Test distribution fitting
        dist_fit = self.processor.fit_distribution(
            self.test_data["qber"], distributions=["normal", "exponential"]
        )

        assert "best_distribution" in dist_fit
        assert "parameters" in dist_fit
        assert "goodness_of_fit" in dist_fit

        # Test correlation analysis
        correlation_matrix = self.processor.calculate_correlations(
            self.test_data[["qber", "key_rate", "detector_efficiency"]]
        )

        assert correlation_matrix.shape == (3, 3)
        assert all(correlation_matrix.diagonal() == 1.0)  # Self-correlation = 1

        # Test statistical tests
        test_results = self.processor.statistical_tests(
            self.test_data["qber"],
            tests=["normality", "stationarity", "autocorrelation"],
        )

        assert "normality_test" in test_results
        assert "stationarity_test" in test_results
        assert "autocorrelation_test" in test_results


class TestConfigManager:
    """Test suite for configuration management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()

        # Create test configuration
        self.test_config = {
            "system": {
                "qber_threshold": 0.11,
                "key_rate_threshold": 500,
                "detection_sensitivity": 0.8,
            },
            "monitoring": {
                "update_interval": 60,
                "alert_window": 300,
                "log_level": "INFO",
            },
            "algorithms": {
                "anomaly_detection": "statistical",
                "ml_model": "random_forest",
                "feature_selection": "auto",
            },
        }

    def test_config_loading_saving(self):
        """Test configuration loading and saving."""
        # Save configuration to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.test_config, f)
            config_file = f.name

        try:
            # Load configuration
            loaded_config = self.config_manager.load_config(config_file)

            assert loaded_config == self.test_config

            # Modify and save
            loaded_config["system"]["qber_threshold"] = 0.12
            self.config_manager.save_config(loaded_config, config_file)

            # Reload and verify change
            reloaded_config = self.config_manager.load_config(config_file)
            assert reloaded_config["system"]["qber_threshold"] == 0.12

        finally:
            os.unlink(config_file)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        is_valid, errors = self.config_manager.validate_config(self.test_config)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid configuration
        invalid_config = self.test_config.copy()
        invalid_config["system"]["qber_threshold"] = 1.5  # Invalid: >1

        is_valid, errors = self.config_manager.validate_config(invalid_config)
        assert is_valid is False
        assert len(errors) > 0
        assert any("qber_threshold" in error for error in errors)

    def test_config_merging(self):
        """Test configuration merging."""
        # Override configuration
        override_config = {
            "system": {
                "qber_threshold": 0.09,  # Different value
                "new_parameter": "test",  # New parameter
            },
            "new_section": {"parameter": "value"},
        }

        merged_config = self.config_manager.merge_configs(
            self.test_config, override_config
        )

        # Check that override values are used
        assert merged_config["system"]["qber_threshold"] == 0.09

        # Check that new parameters are added
        assert merged_config["system"]["new_parameter"] == "test"
        assert "new_section" in merged_config

        # Check that non-overridden values are preserved
        assert merged_config["system"]["key_rate_threshold"] == 500

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in config."""
        # Set test environment variable
        os.environ["TEST_QBER_THRESHOLD"] = "0.10"

        config_with_env = {
            "system": {
                "qber_threshold": "${TEST_QBER_THRESHOLD}",
                "key_rate_threshold": 500,
            }
        }

        try:
            substituted_config = self.config_manager.substitute_env_vars(
                config_with_env
            )

            assert substituted_config["system"]["qber_threshold"] == "0.10"
            assert substituted_config["system"]["key_rate_threshold"] == 500

        finally:
            del os.environ["TEST_QBER_THRESHOLD"]


class TestLogger:
    """Test suite for logging utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Logger("test_logger")

    def test_logging_levels(self):
        """Test different logging levels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            self.logger.configure(
                log_file=log_file,
                level="DEBUG",
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

            # Test different log levels
            self.logger.debug("Debug message")
            self.logger.info("Info message")
            self.logger.warning("Warning message")
            self.logger.error("Error message")
            self.logger.critical("Critical message")

            # Check that log file was created and contains messages
            assert os.path.exists(log_file)

            with open(log_file, "r") as f:
                log_content = f.read()

            assert "Debug message" in log_content
            assert "Info message" in log_content
            assert "Warning message" in log_content
            assert "Error message" in log_content
            assert "Critical message" in log_content

    def test_structured_logging(self):
        """Test structured logging with context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "structured.log")

            self.logger.configure_structured_logging(log_file=log_file, format="json")

            # Log with structured data
            self.logger.log_event(
                "qkd_measurement",
                qber=0.05,
                key_rate=1000,
                detector_efficiency=0.85,
                timestamp=datetime.now().isoformat(),
            )

            # Verify structured log
            assert os.path.exists(log_file)

            with open(log_file, "r") as f:
                log_lines = f.readlines()

            assert len(log_lines) > 0

            # Parse JSON log entry
            log_entry = json.loads(log_lines[0])
            assert log_entry["event_type"] == "qkd_measurement"
            assert log_entry["qber"] == 0.05
            assert log_entry["key_rate"] == 1000


class TestMathUtils:
    """Test suite for mathematical utility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.math_utils = MathUtils()

    def test_statistical_functions(self):
        """Test statistical calculation functions."""
        data = np.random.normal(10, 2, 1000)

        # Test basic statistics
        stats = self.math_utils.calculate_statistics(data)

        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "variance" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "min" in stats
        assert "max" in stats
        assert "quartiles" in stats

        # Verify reasonable values
        assert 9.5 < stats["mean"] < 10.5  # Should be close to 10
        assert 1.8 < stats["std"] < 2.2  # Should be close to 2

    def test_entropy_calculations(self):
        """Test entropy and information theory calculations."""
        # Test Shannon entropy
        probabilities = [0.25, 0.25, 0.25, 0.25]  # Uniform distribution
        shannon_entropy = self.math_utils.shannon_entropy(probabilities)

        assert abs(shannon_entropy - 2.0) < 0.01  # log2(4) = 2

        # Test conditional entropy
        joint_probs = np.array(
            [[0.125, 0.125], [0.125, 0.125], [0.125, 0.125], [0.125, 0.125]]
        )
        marginal_probs = [0.25, 0.25, 0.25, 0.25]

        cond_entropy = self.math_utils.conditional_entropy(joint_probs, marginal_probs)
        assert cond_entropy >= 0  # Conditional entropy should be non-negative

        # Test mutual information
        mi = self.math_utils.mutual_information(joint_probs)
        assert mi >= 0  # Mutual information should be non-negative

    def test_signal_processing_functions(self):
        """Test signal processing mathematical functions."""
        # Test autocorrelation
        signal = np.sin(2 * np.pi * 0.1 * np.arange(100))
        autocorr = self.math_utils.autocorrelation(signal)

        assert len(autocorr) == len(signal)
        assert autocorr[0] == 1.0  # Autocorrelation at lag 0 should be 1

        # Test cross-correlation
        signal2 = np.sin(2 * np.pi * 0.1 * np.arange(100) + np.pi / 4)  # Phase shift
        cross_corr = self.math_utils.cross_correlation(signal, signal2)

        assert len(cross_corr) == 2 * len(signal) - 1

        # Test Fourier transform utilities
        fft_result = self.math_utils.compute_fft(signal)

        assert "frequencies" in fft_result
        assert "magnitude" in fft_result
        assert "phase" in fft_result
        assert len(fft_result["frequencies"]) == len(signal) // 2 + 1

    def test_optimization_functions(self):
        """Test optimization utility functions."""

        # Test golden section search
        def quadratic(x):
            return (x - 2) ** 2 + 1

        minimum = self.math_utils.golden_section_search(
            quadratic, a=-5, b=5, tolerance=1e-6
        )

        assert abs(minimum - 2.0) < 1e-5  # Minimum should be at x=2

        # Test gradient descent
        def gradient_func(x):
            return 2 * (x - 2)

        result = self.math_utils.gradient_descent(
            gradient_func, initial_x=0, learning_rate=0.1, max_iterations=100
        )

        assert abs(result - 2.0) < 0.1  # Should converge to minimum

    def test_matrix_operations(self):
        """Test matrix operation utilities."""
        # Test matrix decompositions
        A = np.random.randn(5, 5)
        A = A @ A.T  # Make positive semi-definite

        # Eigenvalue decomposition
        eigenvals, eigenvecs = self.math_utils.eigendecomposition(A)

        assert len(eigenvals) == 5
        assert eigenvecs.shape == (5, 5)
        assert all(
            eigenvals >= -1e-10
        )  # Should be non-negative (within numerical precision)

        # Singular value decomposition
        B = np.random.randn(4, 6)
        U, s, Vt = self.math_utils.svd(B)

        assert U.shape == (4, 4)
        assert len(s) == 4
        assert Vt.shape == (6, 6)

        # Verify SVD reconstruction
        B_reconstructed = U @ np.diag(s) @ Vt[:4, :]
        assert np.allclose(B, B_reconstructed, atol=1e-10)


class TestFileHandler:
    """Test suite for file handling utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.file_handler = FileHandler()

    def test_file_operations(self):
        """Test basic file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            test_data = "Test file content\nLine 2\nLine 3"

            # Write file
            self.file_handler.write_text_file(test_file, test_data)
            assert os.path.exists(test_file)

            # Read file
            read_data = self.file_handler.read_text_file(test_file)
            assert read_data == test_data

            # Append to file
            append_data = "\nAppended line"
            self.file_handler.append_text_file(test_file, append_data)

            updated_data = self.file_handler.read_text_file(test_file)
            assert updated_data == test_data + append_data

    def test_json_operations(self):
        """Test JSON file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = os.path.join(temp_dir, "test.json")
            test_data = {
                "qber": 0.05,
                "key_rate": 1000,
                "measurements": [1, 2, 3, 4, 5],
                "metadata": {"detector": "SPAD", "wavelength": 850},
            }

            # Write JSON
            self.file_handler.write_json_file(json_file, test_data)
            assert os.path.exists(json_file)

            # Read JSON
            read_data = self.file_handler.read_json_file(json_file)
            assert read_data == test_data

    def test_csv_operations(self):
        """Test CSV file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, "test.csv")

            # Create test DataFrame
            test_df = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2025-01-01", periods=10, freq="1min"),
                    "qber": np.random.normal(0.05, 0.01, 10),
                    "key_rate": np.random.normal(1000, 100, 10),
                }
            )

            # Write CSV
            self.file_handler.write_csv_file(csv_file, test_df)
            assert os.path.exists(csv_file)

            # Read CSV
            read_df = self.file_handler.read_csv_file(csv_file)

            # Compare DataFrames (excluding exact timestamp comparison)
            assert len(read_df) == len(test_df)
            assert list(read_df.columns) == list(test_df.columns)

    def test_binary_operations(self):
        """Test binary file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            binary_file = os.path.join(temp_dir, "test.bin")

            # Test numpy array serialization
            test_array = np.random.randn(100, 50)

            self.file_handler.write_numpy_array(binary_file, test_array)
            assert os.path.exists(binary_file)

            read_array = self.file_handler.read_numpy_array(binary_file)
            assert np.array_equal(test_array, read_array)

    def test_file_compression(self):
        """Test file compression utilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            large_text = "Test data " * 10000
            text_file = os.path.join(temp_dir, "large_file.txt")
            compressed_file = os.path.join(temp_dir, "compressed.gz")

            # Write and compress
            self.file_handler.write_text_file(text_file, large_text)
            self.file_handler.compress_file(text_file, compressed_file)

            assert os.path.exists(compressed_file)

            # Verify compression reduces size
            original_size = os.path.getsize(text_file)
            compressed_size = os.path.getsize(compressed_file)
            assert compressed_size < original_size

            # Decompress and verify
            decompressed_file = os.path.join(temp_dir, "decompressed.txt")
            self.file_handler.decompress_file(compressed_file, decompressed_file)

            decompressed_data = self.file_handler.read_text_file(decompressed_file)
            assert decompressed_data == large_text


class TestTimeUtils:
    """Test suite for time utility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.time_utils = TimeUtils()

    def test_time_parsing(self):
        """Test time parsing and formatting."""
        # Test string to datetime conversion
        time_string = "2025-01-15 14:30:45"
        parsed_time = self.time_utils.parse_time_string(time_string)

        assert isinstance(parsed_time, datetime)
        assert parsed_time.year == 2025
        assert parsed_time.month == 1
        assert parsed_time.day == 15
        assert parsed_time.hour == 14
        assert parsed_time.minute == 30
        assert parsed_time.second == 45

        # Test datetime to string conversion
        formatted_string = self.time_utils.format_datetime(parsed_time)
        assert formatted_string == time_string

    def test_time_arithmetic(self):
        """Test time arithmetic operations."""
        base_time = datetime(2025, 1, 15, 12, 0, 0)

        # Add duration
        future_time = self.time_utils.add_duration(base_time, hours=2, minutes=30)
        expected_time = datetime(2025, 1, 15, 14, 30, 0)
        assert future_time == expected_time

        # Calculate duration between times
        duration = self.time_utils.time_difference(future_time, base_time)
        assert duration.total_seconds() == 9000  # 2.5 hours = 9000 seconds

    def test_time_zone_handling(self):
        """Test time zone operations."""
        # Test UTC conversion
        local_time = datetime(2025, 1, 15, 12, 0, 0)
        utc_time = self.time_utils.to_utc(local_time, timezone="US/Eastern")

        # Eastern time in January is UTC-5
        expected_utc = datetime(2025, 1, 15, 17, 0, 0)
        assert utc_time.replace(tzinfo=None) == expected_utc

        # Test time zone conversion
        pst_time = self.time_utils.convert_timezone(utc_time, "US/Pacific")

        # Pacific time is UTC-8
        expected_pst = datetime(2025, 1, 15, 9, 0, 0)
        assert pst_time.replace(tzinfo=None) == expected_pst

    def test_time_windows(self):
        """Test time window operations."""
        base_time = datetime(2025, 1, 15, 12, 0, 0)

        # Test window creation
        window_start, window_end = self.time_utils.create_time_window(
            base_time, window_size=timedelta(hours=1)
        )

        assert window_start == base_time
        assert window_end == base_time + timedelta(hours=1)

        # Test overlapping windows
        windows = self.time_utils.create_sliding_windows(
            start_time=base_time,
            end_time=base_time + timedelta(hours=3),
            window_size=timedelta(hours=1),
            step_size=timedelta(minutes=30),
        )

        assert len(windows) == 5  # 3 hours with 30-minute steps


class TestValidationUtils:
    """Test suite for validation utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ValidationUtils()

    def test_data_type_validation(self):
        """Test data type validation."""
        # Test numeric validation
        assert self.validator.is_numeric(42) is True
        assert self.validator.is_numeric(3.14) is True
        assert self.validator.is_numeric("42") is False
        assert self.validator.is_numeric("not_a_number") is False

        # Test range validation
        assert self.validator.is_in_range(0.05, 0.0, 1.0) is True
        assert self.validator.is_in_range(-0.1, 0.0, 1.0) is False
        assert self.validator.is_in_range(1.5, 0.0, 1.0) is False

    def test_qkd_specific_validation(self):
        """Test QKD-specific validation rules."""
        # Test QBER validation
        assert self.validator.validate_qber(0.05) is True
        assert self.validator.validate_qber(0.11) is True
        assert self.validator.validate_qber(-0.01) is False
        assert self.validator.validate_qber(0.6) is False

        # Test key rate validation
        assert self.validator.validate_key_rate(1000) is True
        assert self.validator.validate_key_rate(0) is True
        assert self.validator.validate_key_rate(-100) is False

        # Test detector efficiency validation
        assert self.validator.validate_detector_efficiency(0.85) is True
        assert self.validator.validate_detector_efficiency(0.0) is True
        assert self.validator.validate_detector_efficiency(1.0) is True
        assert self.validator.validate_detector_efficiency(-0.1) is False
        assert self.validator.validate_detector_efficiency(1.1) is False

    def test_data_consistency_validation(self):
        """Test data consistency validation."""
        # Test measurement consistency
        measurements = {
            "qber": 0.05,
            "key_rate": 1000,
            "sift_ratio": 0.5,
            "detector_efficiency": 0.85,
        }

        consistency_result = self.validator.validate_measurement_consistency(
            measurements
        )
        assert consistency_result["is_consistent"] is True

        # Test inconsistent measurements
        inconsistent_measurements = {
            "qber": 0.5,  # Very high QBER
            "key_rate": 1000,  # But high key rate (inconsistent)
            "sift_ratio": 0.5,
            "detector_efficiency": 0.85,
        }

        inconsistent_result = self.validator.validate_measurement_consistency(
            inconsistent_measurements
        )
        assert inconsistent_result["is_consistent"] is False
        assert len(inconsistent_result["violations"]) > 0


class TestMetricsCalculator:
    """Test suite for metrics calculation utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()

    def test_qkd_metrics(self):
        """Test QKD-specific metric calculations."""
        # Test secret key rate calculation
        secret_key_rate = self.calculator.calculate_secret_key_rate(
            raw_key_rate=1000,
            qber=0.05,
            error_correction_efficiency=1.16,
            privacy_amplification_factor=1.4,
        )

        assert secret_key_rate > 0
        assert secret_key_rate < 1000  # Should be less than raw key rate

        # Test security parameter calculation
        security_param = self.calculator.calculate_security_parameter(
            qber=0.05, key_length=1000, error_correction_info=100
        )

        assert security_param > 0

        # Test channel capacity
        channel_capacity = self.calculator.calculate_channel_capacity(
            qber=0.05, detector_efficiency=0.85, channel_transmittance=0.9
        )

        assert 0 < channel_capacity <= 1

    def test_performance_metrics(self):
        """Test performance metric calculations."""
        # Create test predictions and ground truth
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])

        # Calculate classification metrics
        metrics = self.calculator.calculate_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics

        # Verify metric ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

        # Test regression metrics
        y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_reg = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        reg_metrics = self.calculator.calculate_regression_metrics(
            y_true_reg, y_pred_reg
        )

        assert "mse" in reg_metrics
        assert "rmse" in reg_metrics
        assert "mae" in reg_metrics
        assert "r2_score" in reg_metrics

        assert reg_metrics["mse"] >= 0
        assert reg_metrics["rmse"] >= 0
        assert reg_metrics["mae"] >= 0

    @pytest.mark.benchmark
    def test_calculation_performance(self, benchmark):
        """Benchmark metric calculation performance."""
        y_true = np.random.randint(0, 2, 10000)
        y_pred = np.random.randint(0, 2, 10000)

        result = benchmark(
            self.calculator.calculate_classification_metrics, y_true, y_pred
        )

        # Metric calculation should be fast
        assert benchmark.stats.mean < 0.1  # <100ms


if __name__ == "__main__":
    pytest.main([__file__])
