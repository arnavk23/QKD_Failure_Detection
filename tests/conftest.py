"""
pytest configuration and shared fixtures for the QKD failure detection test suite.

This module provides common test fixtures, test configuration,
and utility functions used across multiple test modules.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
import sys

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test configuration - handle missing plugins gracefully
try:
    import pytest_benchmark
    pytest_plugins = ["pytest_benchmark"]
except ImportError:
    pytest_plugins = []


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration parameters."""
    return {
        'qber_threshold': 0.11,
        'key_rate_threshold': 500,
        'detection_sensitivity': 0.8,
        'test_data_size': 1000,
        'random_seed': 42
    }


@pytest.fixture(scope="session")
def temp_directory():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    yield 42


@pytest.fixture
def sample_qkd_data(random_seed, test_config):
    """Generate sample QKD measurement data for testing."""
    size = test_config['test_data_size']
    
    # Normal operation data
    normal_qber = np.random.normal(0.05, 0.01, size // 2)
    normal_key_rate = np.random.normal(1000, 100, size // 2)
    normal_sift_ratio = np.random.normal(0.5, 0.05, size // 2)
    normal_detector_eff = np.random.uniform(0.8, 0.9, size // 2)
    normal_channel_loss = np.random.uniform(0.05, 0.1, size // 2)
    
    # Anomalous data
    anomalous_qber = np.random.normal(0.15, 0.03, size // 2)
    anomalous_key_rate = np.random.normal(700, 150, size // 2)
    anomalous_sift_ratio = np.random.normal(0.4, 0.08, size // 2)
    anomalous_detector_eff = np.random.uniform(0.6, 0.8, size // 2)
    anomalous_channel_loss = np.random.uniform(0.1, 0.2, size // 2)
    
    # Combine data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=size, freq='1min'),
        'qber': np.concatenate([normal_qber, anomalous_qber]),
        'key_rate': np.concatenate([normal_key_rate, anomalous_key_rate]),
        'sift_ratio': np.concatenate([normal_sift_ratio, anomalous_sift_ratio]),
        'detector_efficiency': np.concatenate([normal_detector_eff, anomalous_detector_eff]),
        'channel_loss': np.concatenate([normal_channel_loss, anomalous_channel_loss]),
        'mutual_information': np.random.uniform(0.7, 1.0, size),
        'is_anomaly': np.concatenate([np.zeros(size // 2), np.ones(size // 2)])
    })
    
    return data


@pytest.fixture
def sample_photon_stream(random_seed):
    """Generate sample photon detection stream for testing."""
    duration = 10.0  # seconds
    detection_rate = 100  # Hz
    
    # Generate random detection times
    inter_arrival_times = np.random.exponential(1/detection_rate, int(duration * detection_rate * 1.5))
    detection_times = np.cumsum(inter_arrival_times)
    detection_times = detection_times[detection_times < duration]
    
    # Create time series
    time_points = np.linspace(0, duration, int(duration * 1000))  # 1ms resolution
    photon_stream = np.zeros_like(time_points)
    
    for det_time in detection_times:
        idx = np.argmin(np.abs(time_points - det_time))
        photon_stream[idx] = 1
    
    return {
        'time': time_points,
        'detections': photon_stream,
        'detection_times': detection_times,
        'detection_rate': detection_rate
    }


@pytest.fixture
def sample_signal_data(random_seed):
    """Generate sample signal data for testing."""
    fs = 1000  # Sampling frequency
    duration = 2.0  # Duration in seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Clean signal: 50 Hz sine wave
    clean_signal = np.sin(2 * np.pi * 50 * t)
    
    # Noisy signal
    noise_power = 0.1
    noisy_signal = clean_signal + np.random.normal(0, noise_power, len(t))
    
    # Multi-frequency signal
    multi_freq_signal = (np.sin(2 * np.pi * 50 * t) + 
                        0.5 * np.sin(2 * np.pi * 120 * t) + 
                        0.3 * np.sin(2 * np.pi * 300 * t))
    
    return {
        'time': t,
        'sampling_frequency': fs,
        'clean_signal': clean_signal,
        'noisy_signal': noisy_signal,
        'multi_freq_signal': multi_freq_signal
    }


@pytest.fixture
def sample_attack_scenarios():
    """Generate sample attack scenario data for testing."""
    base_time = datetime.now()
    
    scenarios = {
        'intercept_resend': {
            'qber_range': (0.2, 0.3),
            'key_rate_reduction': 0.25,
            'duration': timedelta(minutes=5),
            'characteristics': {
                'high_qber': True,
                'reduced_key_rate': True,
                'low_mutual_info': True
            }
        },
        'beam_splitting': {
            'qber_range': (0.12, 0.18),
            'key_rate_reduction': 0.15,
            'duration': timedelta(minutes=10),
            'characteristics': {
                'moderate_qber_increase': True,
                'reduced_efficiency': True,
                'pattern_detection': True
            }
        },
        'photon_number_splitting': {
            'qber_range': (0.06, 0.1),
            'key_rate_reduction': 0.1,
            'duration': timedelta(minutes=15),
            'characteristics': {
                'subtle_qber_increase': True,
                'multi_photon_exploitation': True,
                'statistical_deviation': True
            }
        }
    }
    
    return scenarios


@pytest.fixture
def mock_qkd_system():
    """Create a mock QKD system for testing."""
    class MockQKDSystem:
        def __init__(self):
            self.is_running = False
            self.measurements = []
            self.alerts = []
            self.current_qber = 0.05
            self.current_key_rate = 1000
            
        def start(self):
            self.is_running = True
            
        def stop(self):
            self.is_running = False
            
        def get_measurement(self):
            if not self.is_running:
                return None
                
            # Simulate measurement with some randomness
            measurement = {
                'timestamp': datetime.now(),
                'qber': self.current_qber + np.random.normal(0, 0.01),
                'key_rate': self.current_key_rate + np.random.normal(0, 50),
                'detector_efficiency': np.random.uniform(0.8, 0.9),
                'channel_loss': np.random.uniform(0.05, 0.1)
            }
            
            self.measurements.append(measurement)
            return measurement
            
        def inject_attack(self, attack_type, duration=60):
            """Simulate an attack by modifying system parameters."""
            if attack_type == 'intercept_resend':
                self.current_qber = 0.25
                self.current_key_rate = 750
            elif attack_type == 'beam_splitting':
                self.current_qber = 0.15
                self.current_key_rate = 850
            elif attack_type == 'pns':
                self.current_qber = 0.08
                self.current_key_rate = 900
                
        def reset_to_normal(self):
            """Reset system to normal operation."""
            self.current_qber = 0.05
            self.current_key_rate = 1000
            
        def add_alert(self, alert):
            self.alerts.append(alert)
            
    return MockQKDSystem()


@pytest.fixture
def ml_test_data(sample_qkd_data):
    """Prepare ML-ready test data."""
    data = sample_qkd_data.copy()
    
    # Features for ML
    features = ['qber', 'key_rate', 'sift_ratio', 'detector_efficiency', 'channel_loss', 'mutual_information']
    X = data[features]
    y = data['is_anomaly'].astype(int)
    
    return {'X': X, 'y': y, 'feature_names': features}


@pytest.fixture
def security_test_events():
    """Generate security test events."""
    events = []
    base_time = datetime.now()
    
    # Normal events
    for i in range(50):
        events.append({
            'timestamp': base_time + timedelta(seconds=i * 60),
            'event_type': 'measurement',
            'qber': 0.05 + np.random.normal(0, 0.01),
            'key_rate': 1000 + np.random.normal(0, 50),
            'severity': 'info',
            'source': 'alice'
        })
    
    # Attack events
    attack_start = base_time + timedelta(minutes=60)
    for i in range(20):
        events.append({
            'timestamp': attack_start + timedelta(seconds=i * 30),
            'event_type': 'anomaly',
            'qber': 0.2 + np.random.normal(0, 0.02),
            'key_rate': 700 + np.random.normal(0, 100),
            'severity': 'high',
            'source': 'alice',
            'attack_type': 'intercept_resend'
        })
    
    return events


@pytest.fixture
def test_cryptographic_data():
    """Generate test cryptographic data."""
    return {
        'message': b"Test quantum key distribution message",
        'key_material': np.random.randint(0, 2, 1000).astype(np.uint8),
        'shared_secret': b"shared_secret_key_32_bytes_long!",
        'authentication_key': np.random.randint(0, 2, 256).astype(np.uint8)
    }


# Test utilities
def assert_qkd_measurement_valid(measurement):
    """Assert that a QKD measurement is valid."""
    assert 'qber' in measurement
    assert 'key_rate' in measurement
    assert 'timestamp' in measurement
    
    assert 0 <= measurement['qber'] <= 1
    assert measurement['key_rate'] >= 0
    assert isinstance(measurement['timestamp'], (datetime, str))


def assert_signal_properties(signal, expected_properties):
    """Assert signal has expected properties."""
    if 'length' in expected_properties:
        assert len(signal) == expected_properties['length']
    
    if 'mean_close_to' in expected_properties:
        assert abs(np.mean(signal) - expected_properties['mean_close_to']) < 0.1
    
    if 'frequency_content' in expected_properties:
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/expected_properties.get('sampling_rate', 1000))
        power_spectrum = np.abs(fft) ** 2
        
        for freq in expected_properties['frequency_content']:
            freq_idx = np.argmin(np.abs(freqs - freq))
            # Check that there's significant power at this frequency
            assert power_spectrum[freq_idx] > 0.1 * np.max(power_spectrum)


def assert_classification_metrics_valid(metrics):
    """Assert that classification metrics are valid."""
    required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for metric in required_metrics:
        assert metric in metrics
        assert 0 <= metrics[metric] <= 1
    
    assert 'confusion_matrix' in metrics
    cm = metrics['confusion_matrix']
    assert cm.shape == (2, 2)  # Binary classification
    assert np.all(cm >= 0)  # Non-negative values


def assert_security_event_valid(event):
    """Assert that a security event is valid."""
    required_fields = ['timestamp', 'event_type', 'severity']
    
    for field in required_fields:
        assert field in event
    
    assert event['severity'] in ['info', 'low', 'medium', 'high', 'critical']
    assert event['event_type'] in ['measurement', 'anomaly', 'attack', 'system', 'user']


def create_test_config_file(config_data, temp_dir):
    """Create a temporary configuration file for testing."""
    import json
    
    config_file = os.path.join(temp_dir, 'test_config.json')
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    return config_file


def create_test_data_file(data, file_path, file_format='csv'):
    """Create a test data file in the specified format."""
    if file_format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        else:
            pd.DataFrame(data).to_csv(file_path, index=False)
    elif file_format == 'json':
        import json
        with open(file_path, 'w') as f:
            if isinstance(data, pd.DataFrame):
                json.dump(data.to_dict('records'), f, indent=2, default=str)
            else:
                json.dump(data, f, indent=2, default=str)
    elif file_format == 'numpy':
        np.save(file_path, data)


# Benchmark configuration
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        'min_rounds': 5,
        'max_time': 1.0,  # Maximum time per benchmark in seconds
        'warmup': False
    }


# Parametrized test fixtures
@pytest.fixture(params=[0.05, 0.1, 0.15, 0.2])
def qber_values(request):
    """Parametrized QBER values for testing."""
    return request.param


@pytest.fixture(params=[500, 750, 1000, 1250])
def key_rate_values(request):
    """Parametrized key rate values for testing."""
    return request.param


@pytest.fixture(params=['random_forest', 'neural_network', 'svm'])
def ml_models(request):
    """Parametrized ML model types for testing."""
    return request.param


@pytest.fixture(params=['statistical', 'ml_based', 'ensemble'])
def anomaly_detection_methods(request):
    """Parametrized anomaly detection methods for testing."""
    return request.param


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security-related"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests as machine learning related"
    )


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Automatically cleanup test environment after each test."""
    yield
    
    # Reset numpy random state
    np.random.seed(None)
    
    # Clear any temporary files or states if needed
    # This runs after each test automatically


# Mock fixtures for external dependencies
@pytest.fixture
def mock_hardware_interface():
    """Mock hardware interface for testing without actual hardware."""
    class MockHardware:
        def __init__(self):
            self.is_connected = True
            self.detector_efficiency = 0.85
            self.laser_power = 1.0
            
        def get_photon_count(self):
            return np.random.poisson(100)
            
        def measure_qber(self):
            return 0.05 + np.random.normal(0, 0.01)
            
        def get_system_status(self):
            return {
                'temperature': 20.0 + np.random.normal(0, 1),
                'humidity': 45.0 + np.random.normal(0, 5),
                'vibration_level': np.random.uniform(0, 0.1)
            }
    
    return MockHardware()


@pytest.fixture
def mock_network_interface():
    """Mock network interface for testing communication protocols."""
    class MockNetwork:
        def __init__(self):
            self.is_connected = True
            self.latency = 0.001  # 1ms
            self.packet_loss = 0.0
            
        def send_message(self, message):
            # Simulate network delay
            import time
            time.sleep(self.latency)
            return {'status': 'sent', 'message_id': np.random.randint(1000, 9999)}
            
        def receive_message(self):
            # Simulate receiving a message
            if np.random.random() > self.packet_loss:
                return {
                    'message': 'test_message',
                    'timestamp': datetime.now(),
                    'sender': 'bob'
                }
            return None
    
    return MockNetwork()


if __name__ == "__main__":
    # This allows running conftest.py directly for testing fixtures
    print("QKD Test Configuration and Fixtures")
    print("Available fixtures:")
    print("- test_config: Test configuration parameters")
    print("- sample_qkd_data: Sample QKD measurement data")
    print("- sample_photon_stream: Sample photon detection stream")
    print("- sample_signal_data: Sample signal data")
    print("- sample_attack_scenarios: Attack scenario data")
    print("- mock_qkd_system: Mock QKD system")
    print("- And many more...")
