"""
Test suite for Anomaly Detector module.

This module contains comprehensive tests for statistical anomaly detection
functionality, including control charts, QBER monitoring, and threshold detection.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from anomaly_detector import AnomalyDetector, AnomalyResult


class TestAnomalyDetector:
    """Test suite for Anomaly Detector functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.detector = AnomalyDetector()
        
        # Create sample QKD data
        np.random.seed(42)
        self.normal_data = {
            'qber': np.random.normal(0.05, 0.01, 100),
            'sift_ratio': np.random.normal(0.5, 0.05, 100),
            'key_rate': np.random.normal(1000, 100, 100),
            'timestamps': pd.date_range('2025-01-01', periods=100, freq='1min')
        }
        
        # Create anomalous data
        self.anomalous_data = self.normal_data.copy()
        self.anomalous_data['qber'][-10:] = 0.2  # Inject anomalies
    
    def test_detector_initialization(self):
        """Test anomaly detector initialization."""
        assert self.detector is not None
        assert hasattr(self.detector, 'detect')
        assert hasattr(self.detector, 'set_threshold')
        assert hasattr(self.detector, 'train')
    
    def test_qber_threshold_detection(self):
        """Test QBER threshold-based anomaly detection."""
        # Normal QBER values
        normal_qber = 0.05
        result = self.detector.detect_qber_anomaly(normal_qber)
        assert result.is_anomaly is False
        assert result.confidence > 0
        
        # Anomalous QBER values
        high_qber = 0.25
        result = self.detector.detect_qber_anomaly(high_qber)
        assert result.is_anomaly is True
        assert result.anomaly_type == 'high_qber'
        assert result.confidence > 0.8
    
    def test_statistical_control_charts(self):
        """Test statistical control chart analysis."""
        data = self.normal_data['qber']
        
        # Shewhart control chart
        results = self.detector.shewhart_control_chart(data)
        assert 'upper_limit' in results
        assert 'lower_limit' in results
        assert 'center_line' in results
        assert results['violations'] >= 0
        
        # CUSUM chart
        cusum_results = self.detector.cusum_analysis(data)
        assert 'positive_cusum' in cusum_results
        assert 'negative_cusum' in cusum_results
        assert 'alarm_points' in cusum_results
    
    def test_outlier_detection(self):
        """Test outlier detection methods."""
        data = self.normal_data['qber']
        
        # Z-score outlier detection
        outliers_zscore = self.detector.detect_outliers_zscore(data, threshold=3.0)
        assert isinstance(outliers_zscore, list)
        
        # IQR outlier detection
        outliers_iqr = self.detector.detect_outliers_iqr(data)
        assert isinstance(outliers_iqr, list)
        
        # Modified Z-score
        outliers_modified = self.detector.detect_outliers_modified_zscore(data)
        assert isinstance(outliers_modified, list)
    
    def test_time_series_analysis(self):
        """Test time series anomaly detection."""
        df = pd.DataFrame(self.normal_data)
        
        # Trend analysis
        trend_result = self.detector.detect_trend_anomalies(df['qber'])
        assert 'trend_direction' in trend_result
        assert 'trend_strength' in trend_result
        assert 'change_points' in trend_result
        
        # Seasonal decomposition
        if len(df) >= 24:  # Need sufficient data for seasonality
            seasonal_result = self.detector.seasonal_decomposition(
                df['qber'], 
                freq=24  # Hourly data with daily seasonality
            )
            assert 'trend' in seasonal_result
            assert 'seasonal' in seasonal_result
            assert 'residual' in seasonal_result
    
    def test_multivariate_anomaly_detection(self):
        """Test multivariate anomaly detection."""
        df = pd.DataFrame(self.normal_data)
        
        # Isolation Forest
        results = self.detector.isolation_forest_detection(
            df[['qber', 'sift_ratio', 'key_rate']]
        )
        assert 'anomaly_scores' in results
        assert 'anomalies' in results
        assert len(results['anomaly_scores']) == len(df)
        
        # Mahalanobis distance
        mahal_results = self.detector.mahalanobis_distance_detection(
            df[['qber', 'sift_ratio', 'key_rate']]
        )
        assert 'distances' in mahal_results
        assert 'threshold' in mahal_results
        assert 'anomalies' in mahal_results
    
    def test_adaptive_thresholding(self):
        """Test adaptive threshold adjustment."""
        data = self.normal_data['qber']
        
        # Initial threshold
        initial_threshold = self.detector.calculate_adaptive_threshold(data[:50])
        
        # Updated threshold with more data
        updated_threshold = self.detector.calculate_adaptive_threshold(data)
        
        assert initial_threshold > 0
        assert updated_threshold > 0
        # Threshold should adapt to data characteristics
    
    def test_anomaly_classification(self):
        """Test anomaly type classification."""
        # Different types of anomalies
        test_cases = [
            {'qber': 0.25, 'expected_type': 'high_qber'},
            {'sift_ratio': 0.1, 'expected_type': 'low_sift_ratio'},
            {'key_rate': 100, 'expected_type': 'low_key_rate'},
            {'qber': 0.25, 'sift_ratio': 0.1, 'expected_type': 'multiple_anomalies'}
        ]
        
        for case in test_cases:
            result = self.detector.classify_anomaly(case)
            if 'expected_type' in case:
                assert case['expected_type'] in result.anomaly_type
    
    def test_detection_performance(self):
        """Test detection performance metrics."""
        # Create labeled test data
        normal_samples = np.random.normal(0.05, 0.01, 100)
        anomalous_samples = np.random.normal(0.2, 0.02, 20)
        
        data = np.concatenate([normal_samples, anomalous_samples])
        labels = np.concatenate([np.zeros(100), np.ones(20)])
        
        predictions = []
        for sample in data:
            result = self.detector.detect_qber_anomaly(sample)
            predictions.append(1 if result.is_anomaly else 0)
        
        # Calculate performance metrics
        performance = self.detector.calculate_performance_metrics(labels, predictions)
        
        assert 'accuracy' in performance
        assert 'precision' in performance
        assert 'recall' in performance
        assert 'f1_score' in performance
        assert performance['accuracy'] > 0.8  # Should have good accuracy
    
    def test_real_time_detection(self):
        """Test real-time anomaly detection."""
        # Simulate real-time data stream
        for i, qber_value in enumerate(self.normal_data['qber']):
            result = self.detector.detect_realtime(
                qber=qber_value,
                timestamp=self.normal_data['timestamps'][i]
            )
            
            assert isinstance(result, AnomalyResult)
            assert hasattr(result, 'is_anomaly')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'processing_time')
            
            # Real-time detection should be fast
            assert result.processing_time < 0.05  # <50ms
    
    def test_ensemble_detection(self):
        """Test ensemble anomaly detection methods."""
        data = self.anomalous_data['qber']
        
        ensemble_result = self.detector.ensemble_detection(data)
        
        assert 'individual_results' in ensemble_result
        assert 'ensemble_score' in ensemble_result
        assert 'final_decision' in ensemble_result
        assert 'confidence' in ensemble_result
        
        # Ensemble should combine multiple methods
        assert len(ensemble_result['individual_results']) >= 2
    
    @pytest.mark.benchmark
    def test_detection_benchmark(self, benchmark):
        """Benchmark anomaly detection performance."""
        data = self.normal_data['qber']
        
        result = benchmark(self.detector.detect, data)
        assert result is not None
        
        # Should process 100 samples quickly
        assert benchmark.stats.mean < 0.1  # <100ms average
    
    def test_false_positive_control(self):
        """Test false positive rate control."""
        # Generate purely normal data
        normal_data = np.random.normal(0.05, 0.01, 1000)
        
        false_positives = 0
        for sample in normal_data:
            result = self.detector.detect_qber_anomaly(sample)
            if result.is_anomaly:
                false_positives += 1
        
        false_positive_rate = false_positives / len(normal_data)
        
        # False positive rate should be controlled
        assert false_positive_rate < 0.05  # <5% false positive rate
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to detection parameters."""
        data = self.normal_data['qber']
        
        # Test different threshold values
        thresholds = [0.08, 0.10, 0.12, 0.15]
        detection_rates = []
        
        for threshold in thresholds:
            self.detector.set_threshold(threshold)
            anomalies = 0
            for sample in data:
                result = self.detector.detect_qber_anomaly(sample)
                if result.is_anomaly:
                    anomalies += 1
            detection_rates.append(anomalies / len(data))
        
        # Higher thresholds should result in fewer detections
        assert detection_rates[0] >= detection_rates[-1]


class TestAnomalyResult:
    """Test suite for AnomalyResult data structure."""
    
    def test_result_creation(self):
        """Test anomaly result object creation."""
        result = AnomalyResult(
            is_anomaly=True,
            anomaly_type='high_qber',
            confidence=0.95,
            anomaly_score=2.5,
            threshold=0.11
        )
        
        assert result.is_anomaly is True
        assert result.anomaly_type == 'high_qber'
        assert result.confidence == 0.95
        assert result.anomaly_score == 2.5
        assert result.threshold == 0.11
    
    def test_result_validation(self):
        """Test result data validation."""
        # Test invalid confidence
        with pytest.raises(ValueError):
            AnomalyResult(confidence=1.5)
        
        # Test invalid anomaly score
        with pytest.raises(ValueError):
            AnomalyResult(anomaly_score=-1)
    
    def test_result_serialization(self):
        """Test result serialization."""
        result = AnomalyResult(
            is_anomaly=True,
            anomaly_type='high_qber',
            confidence=0.95
        )
        
        json_data = result.to_json()
        assert 'is_anomaly' in json_data
        assert 'anomaly_type' in json_data
        assert 'confidence' in json_data


if __name__ == "__main__":
    pytest.main([__file__])
