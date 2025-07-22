"""
Test suite for QKD Simulator module.

This module contains comprehensive tests for the QKD simulator functionality,
including BB84 protocol implementation, noise modeling, and performance validation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qkd_simulator import QKDSimulator, QKDSession


class TestQKDSimulator:
    """Test suite for QKD Simulator functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.simulator = QKDSimulator()
        self.default_params = {
            'key_length': 1000,
            'noise_level': 0.05,
            'channel_loss': 0.1,
            'detector_efficiency': 0.85
        }
    
    def test_simulator_initialization(self):
        """Test QKD simulator initialization."""
        assert self.simulator is not None
        assert hasattr(self.simulator, 'run_bb84')
        assert hasattr(self.simulator, 'add_noise')
        assert hasattr(self.simulator, 'calculate_qber')
    
    def test_bb84_protocol_basic(self):
        """Test basic BB84 protocol execution."""
        session = self.simulator.run_bb84(**self.default_params)
        
        assert isinstance(session, QKDSession)
        assert session.key_length == self.default_params['key_length']
        assert session.qber >= 0.0
        assert session.qber <= 1.0
        assert session.sift_ratio > 0.0
        assert session.sift_ratio <= 1.0
        assert len(session.final_key) >= 0
    
    def test_bb84_with_noise(self):
        """Test BB84 protocol with different noise levels."""
        noise_levels = [0.0, 0.05, 0.1, 0.2]
        
        for noise in noise_levels:
            params = self.default_params.copy()
            params['noise_level'] = noise
            session = self.simulator.run_bb84(**params)
            
            # Higher noise should generally lead to higher QBER
            assert session.qber >= noise / 2  # Approximate lower bound
            assert session.success is True
    
    def test_channel_loss_effects(self):
        """Test the effects of channel loss on QKD performance."""
        loss_levels = [0.0, 0.1, 0.3, 0.5]
        
        for loss in loss_levels:
            params = self.default_params.copy()
            params['channel_loss'] = loss
            session = self.simulator.run_bb84(**params)
            
            # Higher loss should reduce final key length
            expected_efficiency = (1 - loss) * 0.5  # Approximate efficiency
            assert session.sift_ratio <= 1.0
            assert session.final_key_length >= 0
    
    def test_detector_efficiency(self):
        """Test detector efficiency parameter effects."""
        efficiencies = [0.5, 0.7, 0.9, 1.0]
        
        for eff in efficiencies:
            params = self.default_params.copy()
            params['detector_efficiency'] = eff
            session = self.simulator.run_bb84(**params)
            
            assert session.success is True
            assert session.detector_efficiency == eff
    
    def test_attack_injection(self):
        """Test attack scenario injection."""
        attack_types = ['intercept_resend', 'beam_splitting', 'pns_attack']
        
        for attack in attack_types:
            session = self.simulator.run_bb84_with_attack(
                attack_type=attack,
                attack_strength=0.3,
                **self.default_params
            )
            
            # Attacks should increase QBER
            normal_session = self.simulator.run_bb84(**self.default_params)
            assert session.qber >= normal_session.qber
            assert session.attack_detected is not None
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        session = self.simulator.run_bb84(**self.default_params)
        
        # Check all required metrics are present
        assert hasattr(session, 'qber')
        assert hasattr(session, 'sift_ratio')
        assert hasattr(session, 'mutual_information')
        assert hasattr(session, 'final_key_length')
        assert hasattr(session, 'processing_time')
        
        # Validate metric ranges
        assert 0 <= session.qber <= 1
        assert 0 <= session.sift_ratio <= 1
        assert 0 <= session.mutual_information <= 1
        assert session.processing_time > 0
    
    def test_error_conditions(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError):
            # Invalid key length
            self.simulator.run_bb84(key_length=-1)
        
        with pytest.raises(ValueError):
            # Invalid noise level
            self.simulator.run_bb84(noise_level=1.5)
        
        with pytest.raises(ValueError):
            # Invalid channel loss
            self.simulator.run_bb84(channel_loss=2.0)
    
    def test_statistical_properties(self):
        """Test statistical properties of generated data."""
        # Run multiple sessions for statistical analysis
        sessions = []
        for _ in range(50):
            session = self.simulator.run_bb84(**self.default_params)
            sessions.append(session)
        
        qber_values = [s.qber for s in sessions]
        sift_ratios = [s.sift_ratio for s in sessions]
        
        # Check statistical properties
        mean_qber = np.mean(qber_values)
        std_qber = np.std(qber_values)
        
        # QBER should be close to noise level with some variance
        assert abs(mean_qber - self.default_params['noise_level']) < 0.02
        assert std_qber > 0  # Should have some variance
        
        # Sift ratios should be around 0.5 for random bases
        mean_sift = np.mean(sift_ratios)
        assert abs(mean_sift - 0.5) < 0.1
    
    @pytest.mark.benchmark
    def test_performance_benchmark(self, benchmark):
        """Benchmark QKD simulation performance."""
        result = benchmark(self.simulator.run_bb84, **self.default_params)
        assert result.success is True
        assert result.processing_time < 0.1  # Should complete in <100ms
    
    def test_reproducibility(self):
        """Test simulation reproducibility with fixed seed."""
        # Set fixed seed for reproducibility
        np.random.seed(42)
        session1 = self.simulator.run_bb84(**self.default_params)
        
        np.random.seed(42)
        session2 = self.simulator.run_bb84(**self.default_params)
        
        # Results should be identical with same seed
        assert session1.qber == session2.qber
        assert session1.sift_ratio == session2.sift_ratio
        assert len(session1.final_key) == len(session2.final_key)
    
    def test_protocol_variants(self):
        """Test different QKD protocol variants."""
        protocols = ['BB84', 'SARG04', 'E91']
        
        for protocol in protocols:
            session = self.simulator.run_protocol(
                protocol=protocol,
                **self.default_params
            )
            
            assert session.protocol == protocol
            assert session.success is True
            assert session.qber >= 0
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Reset any global state if needed
        pass


class TestQKDSession:
    """Test suite for QKD Session data structure."""
    
    def test_session_creation(self):
        """Test QKD session object creation."""
        session = QKDSession(
            key_length=1000,
            qber=0.08,
            sift_ratio=0.5,
            final_key=np.random.randint(0, 2, 400),
            mutual_information=0.92
        )
        
        assert session.key_length == 1000
        assert session.qber == 0.08
        assert session.sift_ratio == 0.5
        assert len(session.final_key) == 400
        assert session.mutual_information == 0.92
    
    def test_session_validation(self):
        """Test session data validation."""
        # Test invalid QBER
        with pytest.raises(ValueError):
            QKDSession(qber=1.5)
        
        # Test invalid sift ratio
        with pytest.raises(ValueError):
            QKDSession(sift_ratio=-0.1)
    
    def test_session_serialization(self):
        """Test session serialization to JSON."""
        session = QKDSession(
            key_length=1000,
            qber=0.08,
            sift_ratio=0.5,
            final_key=np.array([1, 0, 1, 0]),
            mutual_information=0.92
        )
        
        json_data = session.to_json()
        assert 'qber' in json_data
        assert 'sift_ratio' in json_data
        assert 'final_key' in json_data
        
        # Test deserialization
        restored_session = QKDSession.from_json(json_data)
        assert restored_session.qber == session.qber
        assert restored_session.sift_ratio == session.sift_ratio


if __name__ == "__main__":
    pytest.main([__file__])
