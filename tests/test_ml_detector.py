"""
Test suite for ML Detector module.

This module contains comprehensive tests for machine learning-based
anomaly detection and classification functionality.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_detector import MLDetector, MLResult, FeatureEngineer


class TestMLDetector:
    """Test suite for ML Detector functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.detector = MLDetector()
        self.feature_engineer = FeatureEngineer()
        
        # Create synthetic training data
        np.random.seed(42)
        self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic QKD data for testing."""
        n_normal = 800
        n_anomalous = 200
        
        # Normal samples
        normal_qber = np.random.normal(0.05, 0.01, n_normal)
        normal_sift = np.random.normal(0.5, 0.05, n_normal)
        normal_key_rate = np.random.normal(1000, 100, n_normal)
        normal_labels = np.zeros(n_normal)
        
        # Anomalous samples (attacks)
        anomalous_qber = np.random.normal(0.2, 0.03, n_anomalous)
        anomalous_sift = np.random.normal(0.3, 0.08, n_anomalous)
        anomalous_key_rate = np.random.normal(600, 150, n_anomalous)
        anomalous_labels = np.ones(n_anomalous)
        
        # Combine data
        self.X = pd.DataFrame({
            'qber': np.concatenate([normal_qber, anomalous_qber]),
            'sift_ratio': np.concatenate([normal_sift, anomalous_sift]),
            'key_rate': np.concatenate([normal_key_rate, anomalous_key_rate]),
            'mutual_information': np.random.uniform(0.8, 1.0, n_normal + n_anomalous),
            'channel_loss': np.random.uniform(0.05, 0.15, n_normal + n_anomalous),
            'detector_efficiency': np.random.uniform(0.7, 0.9, n_normal + n_anomalous)
        })
        
        self.y = np.concatenate([normal_labels, anomalous_labels])
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
    
    def test_detector_initialization(self):
        """Test ML detector initialization."""
        assert self.detector is not None
        assert hasattr(self.detector, 'train')
        assert hasattr(self.detector, 'predict')
        assert hasattr(self.detector, 'predict_proba')
        assert hasattr(self.detector, 'evaluate')
    
    def test_random_forest_training(self):
        """Test Random Forest model training."""
        # Train model
        self.detector.train(
            self.X_train, 
            self.y_train, 
            model_type='random_forest'
        )
        
        assert self.detector.is_trained is True
        assert self.detector.model_type == 'random_forest'
        assert hasattr(self.detector.model, 'predict')
        assert hasattr(self.detector.model, 'predict_proba')
    
    def test_neural_network_training(self):
        """Test Neural Network model training."""
        self.detector.train(
            self.X_train, 
            self.y_train, 
            model_type='neural_network',
            hidden_layers=[100, 50],
            epochs=50
        )
        
        assert self.detector.is_trained is True
        assert self.detector.model_type == 'neural_network'
    
    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        # Basic features
        basic_features = self.feature_engineer.extract_basic_features(self.X_train)
        assert 'qber' in basic_features.columns
        assert 'sift_ratio' in basic_features.columns
        
        # Temporal features
        temporal_data = self.X_train.copy()
        temporal_data['timestamp'] = pd.date_range('2025-01-01', periods=len(temporal_data), freq='1min')
        
        temporal_features = self.feature_engineer.extract_temporal_features(temporal_data)
        assert any('rolling_mean' in col for col in temporal_features.columns)
        assert any('lag_' in col for col in temporal_features.columns)
        
        # Domain-specific features
        domain_features = self.feature_engineer.extract_domain_features(self.X_train)
        assert 'security_parameter' in domain_features.columns
        assert 'efficiency_ratio' in domain_features.columns
    
    def test_model_prediction(self):
        """Test model prediction functionality."""
        # Train model first
        self.detector.train(self.X_train, self.y_train)
        
        # Test single prediction
        sample = self.X_test.iloc[0:1]
        result = self.detector.predict(sample)
        
        assert isinstance(result, MLResult)
        assert hasattr(result, 'prediction')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'probabilities')
        assert result.prediction in [0, 1]
        assert 0 <= result.confidence <= 1
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        self.detector.train(self.X_train, self.y_train)
        
        # Batch prediction
        predictions = self.detector.predict_batch(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        assert all(isinstance(pred, MLResult) for pred in predictions)
        assert all(pred.prediction in [0, 1] for pred in predictions)
    
    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        self.detector.train(self.X_train, self.y_train)
        
        evaluation = self.detector.evaluate(self.X_test, self.y_test)
        
        assert 'accuracy' in evaluation
        assert 'precision' in evaluation
        assert 'recall' in evaluation
        assert 'f1_score' in evaluation
        assert 'auc_roc' in evaluation
        assert 'confusion_matrix' in evaluation
        
        # Check metric ranges
        assert 0 <= evaluation['accuracy'] <= 1
        assert 0 <= evaluation['precision'] <= 1
        assert 0 <= evaluation['recall'] <= 1
        assert 0 <= evaluation['f1_score'] <= 1
        assert 0 <= evaluation['auc_roc'] <= 1
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        cv_results = self.detector.cross_validate(
            self.X, self.y, cv=5, model_type='random_forest'
        )
        
        assert 'mean_accuracy' in cv_results
        assert 'std_accuracy' in cv_results
        assert 'individual_scores' in cv_results
        assert len(cv_results['individual_scores']) == 5
        
        # Mean accuracy should be reasonable
        assert cv_results['mean_accuracy'] > 0.7
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        best_params = self.detector.optimize_hyperparameters(
            self.X_train, self.y_train,
            model_type='random_forest',
            param_grid={
                'n_estimators': [50, 100],
                'max_depth': [5, 10]
            },
            cv=3
        )
        
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert best_params['n_estimators'] in [50, 100]
        assert best_params['max_depth'] in [5, 10]
    
    def test_feature_importance(self):
        """Test feature importance analysis."""
        self.detector.train(
            self.X_train, self.y_train, 
            model_type='random_forest'
        )
        
        importance = self.detector.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(self.X_train.columns)
        assert all(0 <= score <= 1 for score in importance.values())
        
        # Sum of importance scores should be close to 1
        assert abs(sum(importance.values()) - 1.0) < 0.01
    
    def test_model_interpretability(self):
        """Test model interpretability features."""
        self.detector.train(self.X_train, self.y_train)
        
        # SHAP values (if available)
        sample = self.X_test.iloc[0:1]
        try:
            explanation = self.detector.explain_prediction(sample)
            assert 'shap_values' in explanation or 'feature_contributions' in explanation
        except ImportError:
            # SHAP not available, skip this test
            pytest.skip("SHAP not available for model explanation")
    
    def test_ensemble_methods(self):
        """Test ensemble model functionality."""
        ensemble_detector = MLDetector(ensemble=True)
        
        ensemble_detector.train(
            self.X_train, self.y_train,
            models=['random_forest', 'neural_network', 'svm']
        )
        
        assert ensemble_detector.is_ensemble is True
        assert len(ensemble_detector.models) >= 2
        
        # Test ensemble prediction
        sample = self.X_test.iloc[0:1]
        result = ensemble_detector.predict(sample)
        
        assert isinstance(result, MLResult)
        assert hasattr(result, 'ensemble_predictions')
        assert hasattr(result, 'final_prediction')
    
    def test_online_learning(self):
        """Test online/incremental learning capability."""
        # Train initial model
        self.detector.train(self.X_train, self.y_train)
        initial_accuracy = self.detector.evaluate(self.X_test, self.y_test)['accuracy']
        
        # Simulate new data arrival
        new_X = self.X_test[:50]
        new_y = self.y_test[:50]
        
        # Update model with new data
        self.detector.update_model(new_X, new_y)
        
        # Model should still be trained
        assert self.detector.is_trained is True
        
        # Evaluate updated model
        updated_accuracy = self.detector.evaluate(self.X_test[50:], self.y_test[50:])['accuracy']
        
        # Updated model should maintain reasonable performance
        assert updated_accuracy > 0.6
    
    def test_anomaly_type_classification(self):
        """Test multi-class anomaly type classification."""
        # Create multi-class data
        multi_class_data = self.create_multiclass_data()
        X_multi, y_multi = multi_class_data['X'], multi_class_data['y']
        
        # Train multi-class classifier
        self.detector.train(X_multi, y_multi, task='multiclass')
        
        # Test prediction
        sample = X_multi.iloc[0:1]
        result = self.detector.predict(sample)
        
        assert result.prediction in [0, 1, 2, 3]  # Normal + 3 attack types
        assert hasattr(result, 'class_probabilities')
        assert len(result.class_probabilities) == 4
    
    def create_multiclass_data(self):
        """Create multi-class synthetic data for testing."""
        n_samples_per_class = 200
        
        # Class 0: Normal
        normal_data = pd.DataFrame({
            'qber': np.random.normal(0.05, 0.01, n_samples_per_class),
            'sift_ratio': np.random.normal(0.5, 0.05, n_samples_per_class),
            'key_rate': np.random.normal(1000, 100, n_samples_per_class)
        })
        
        # Class 1: Intercept-resend attack
        intercept_data = pd.DataFrame({
            'qber': np.random.normal(0.25, 0.03, n_samples_per_class),
            'sift_ratio': np.random.normal(0.5, 0.05, n_samples_per_class),
            'key_rate': np.random.normal(750, 100, n_samples_per_class)
        })
        
        # Class 2: Beam-splitting attack
        beam_split_data = pd.DataFrame({
            'qber': np.random.normal(0.15, 0.02, n_samples_per_class),
            'sift_ratio': np.random.normal(0.4, 0.06, n_samples_per_class),
            'key_rate': np.random.normal(800, 120, n_samples_per_class)
        })
        
        # Class 3: PNS attack
        pns_data = pd.DataFrame({
            'qber': np.random.normal(0.08, 0.015, n_samples_per_class),
            'sift_ratio': np.random.normal(0.45, 0.07, n_samples_per_class),
            'key_rate': np.random.normal(900, 150, n_samples_per_class)
        })
        
        # Combine all classes
        X = pd.concat([normal_data, intercept_data, beam_split_data, pns_data], ignore_index=True)
        y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class + 
                     [2] * n_samples_per_class + [3] * n_samples_per_class)
        
        return {'X': X, 'y': y}
    
    @pytest.mark.benchmark
    def test_training_benchmark(self, benchmark):
        """Benchmark model training performance."""
        result = benchmark(
            self.detector.train,
            self.X_train, self.y_train,
            model_type='random_forest'
        )
        
        # Training should complete in reasonable time
        assert benchmark.stats.mean < 10.0  # <10 seconds
    
    @pytest.mark.benchmark
    def test_prediction_benchmark(self, benchmark):
        """Benchmark prediction performance."""
        self.detector.train(self.X_train, self.y_train)
        
        sample = self.X_test.iloc[0:1]
        result = benchmark(self.detector.predict, sample)
        
        # Prediction should be fast
        assert benchmark.stats.mean < 0.01  # <10ms
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train model
        self.detector.train(self.X_train, self.y_train)
        original_accuracy = self.detector.evaluate(self.X_test, self.y_test)['accuracy']
        
        # Save model
        model_path = '/tmp/test_qkd_model.pkl'
        self.detector.save_model(model_path)
        
        # Create new detector and load model
        new_detector = MLDetector()
        new_detector.load_model(model_path)
        
        # Test loaded model
        loaded_accuracy = new_detector.evaluate(self.X_test, self.y_test)['accuracy']
        
        # Loaded model should have same performance
        assert abs(original_accuracy - loaded_accuracy) < 0.01
        
        # Clean up
        os.remove(model_path)


class TestMLResult:
    """Test suite for MLResult data structure."""
    
    def test_result_creation(self):
        """Test ML result object creation."""
        result = MLResult(
            prediction=1,
            confidence=0.95,
            probabilities=[0.05, 0.95],
            processing_time=0.01
        )
        
        assert result.prediction == 1
        assert result.confidence == 0.95
        assert result.probabilities == [0.05, 0.95]
        assert result.processing_time == 0.01
    
    def test_result_validation(self):
        """Test result validation."""
        # Test invalid confidence
        with pytest.raises(ValueError):
            MLResult(confidence=1.5)
        
        # Test invalid probabilities
        with pytest.raises(ValueError):
            MLResult(probabilities=[0.3, 0.8])  # Don't sum to 1


class TestFeatureEngineer:
    """Test suite for Feature Engineering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()
        self.sample_data = pd.DataFrame({
            'qber': np.random.normal(0.05, 0.01, 100),
            'sift_ratio': np.random.normal(0.5, 0.05, 100),
            'key_rate': np.random.normal(1000, 100, 100),
            'timestamp': pd.date_range('2025-01-01', periods=100, freq='1min')
        })
    
    def test_feature_scaling(self):
        """Test feature scaling methods."""
        # Standard scaling
        scaled_data = self.engineer.standard_scale(self.sample_data[['qber', 'sift_ratio']])
        assert abs(scaled_data.mean().mean()) < 0.01  # Mean should be close to 0
        assert abs(scaled_data.std().mean() - 1.0) < 0.01  # Std should be close to 1
        
        # Min-max scaling
        minmax_scaled = self.engineer.minmax_scale(self.sample_data[['qber', 'sift_ratio']])
        assert minmax_scaled.min().min() >= 0.0
        assert minmax_scaled.max().max() <= 1.0
    
    def test_feature_selection(self):
        """Test feature selection methods."""
        X = self.sample_data[['qber', 'sift_ratio', 'key_rate']]
        y = np.random.randint(0, 2, len(X))
        
        # Select k best features
        selected_features = self.engineer.select_k_best(X, y, k=2)
        assert selected_features.shape[1] == 2
        
        # Recursive feature elimination
        rfe_features = self.engineer.recursive_feature_elimination(X, y, n_features=2)
        assert rfe_features.shape[1] == 2


if __name__ == "__main__":
    pytest.main([__file__])
