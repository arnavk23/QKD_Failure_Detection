"""
Test suite for Security Monitor module.

This module contains comprehensive tests for security monitoring, 
threat detection, and cryptographic verification functionality.
"""

import pytest
import numpy as np
import pandas as pd
import hashlib
import hmac
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from security_monitor import (
    SecurityMonitor,
    ThreatDetector,
    CryptographicVerifier,
    SecurityEvent,
)


class TestSecurityMonitor:
    """Test suite for Security Monitor functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.monitor = SecurityMonitor()
        self.threat_detector = ThreatDetector()
        self.crypto_verifier = CryptographicVerifier()

        # Create test security events
        self.create_test_events()

        # Set up monitoring parameters
        self.monitor.configure(
            qber_threshold=0.11,
            key_rate_threshold=500,
            detection_sensitivity=0.8,
            alert_window=300,  # 5 minutes
        )

    def create_test_events(self):
        """Create test security events for analysis."""
        base_time = datetime.now()

        self.normal_events = [
            SecurityEvent(
                timestamp=base_time + timedelta(seconds=i),
                event_type="measurement",
                qber=0.05 + 0.01 * np.random.randn(),
                key_rate=1000 + 50 * np.random.randn(),
                source="alice",
                severity="info",
            )
            for i in range(100)
        ]

        self.attack_events = [
            SecurityEvent(
                timestamp=base_time + timedelta(seconds=200 + i),
                event_type="anomaly",
                qber=0.2 + 0.03 * np.random.randn(),
                key_rate=600 + 100 * np.random.randn(),
                source="alice",
                severity="high",
                attack_type="intercept_resend",
            )
            for i in range(20)
        ]

        self.all_events = self.normal_events + self.attack_events

    def test_monitor_initialization(self):
        """Test security monitor initialization."""
        assert self.monitor is not None
        assert hasattr(self.monitor, "monitor_system")
        assert hasattr(self.monitor, "detect_threats")
        assert hasattr(self.monitor, "verify_integrity")
        assert hasattr(self.monitor, "generate_alerts")

    def test_real_time_monitoring(self):
        """Test real-time security monitoring."""
        # Start monitoring
        self.monitor.start_monitoring()

        # Feed events to monitor
        for event in self.normal_events[:10]:
            self.monitor.process_event(event)

        # Check monitoring status
        status = self.monitor.get_status()

        assert status["is_monitoring"] is True
        assert status["events_processed"] == 10
        assert status["alerts_generated"] >= 0
        assert "last_update" in status

        # Stop monitoring
        self.monitor.stop_monitoring()
        assert self.monitor.get_status()["is_monitoring"] is False

    def test_threat_detection(self):
        """Test threat detection algorithms."""
        # Analyze normal events
        normal_analysis = self.threat_detector.analyze_events(self.normal_events)

        assert "threat_level" in normal_analysis
        assert "detected_attacks" in normal_analysis
        assert "confidence_scores" in normal_analysis

        # Normal events should have low threat level
        assert normal_analysis["threat_level"] < 0.3

        # Analyze attack events
        attack_analysis = self.threat_detector.analyze_events(self.attack_events)

        # Attack events should have high threat level
        assert attack_analysis["threat_level"] > 0.7
        assert len(attack_analysis["detected_attacks"]) > 0

    def test_anomaly_detection_algorithms(self):
        """Test various anomaly detection methods."""
        # Statistical anomaly detection
        statistical_results = self.threat_detector.statistical_anomaly_detection(
            self.all_events
        )

        assert "anomalies" in statistical_results
        assert "z_scores" in statistical_results
        assert "threshold" in statistical_results

        # Machine learning-based detection
        ml_results = self.threat_detector.ml_anomaly_detection(self.all_events)

        assert "anomaly_scores" in ml_results
        assert "predictions" in ml_results
        assert "model_confidence" in ml_results

        # Time series anomaly detection
        ts_results = self.threat_detector.time_series_anomaly_detection(self.all_events)

        assert "change_points" in ts_results
        assert "seasonal_anomalies" in ts_results
        assert "trend_anomalies" in ts_results

    def test_attack_type_classification(self):
        """Test attack type classification."""
        # Create specific attack signatures
        intercept_resend = self.create_intercept_resend_attack()
        beam_splitting = self.create_beam_splitting_attack()
        pns_attack = self.create_pns_attack()

        # Classify attacks
        ir_classification = self.threat_detector.classify_attack(intercept_resend)
        bs_classification = self.threat_detector.classify_attack(beam_splitting)
        pns_classification = self.threat_detector.classify_attack(pns_attack)

        # Verify correct classification
        assert ir_classification["predicted_attack"] == "intercept_resend"
        assert ir_classification["confidence"] > 0.8

        assert bs_classification["predicted_attack"] == "beam_splitting"
        assert bs_classification["confidence"] > 0.8

        assert pns_classification["predicted_attack"] == "photon_number_splitting"
        assert pns_classification["confidence"] > 0.8

    def create_intercept_resend_attack(self):
        """Create synthetic intercept-resend attack data."""
        return [
            SecurityEvent(
                timestamp=datetime.now() + timedelta(seconds=i),
                event_type="attack",
                qber=0.25 + 0.02 * np.random.randn(),  # High QBER
                key_rate=750 + 50 * np.random.randn(),  # Reduced key rate
                mutual_information=0.5 + 0.1 * np.random.randn(),  # Low MI
                source="alice",
                severity="critical",
            )
            for i in range(50)
        ]

    def create_beam_splitting_attack(self):
        """Create synthetic beam-splitting attack data."""
        return [
            SecurityEvent(
                timestamp=datetime.now() + timedelta(seconds=i),
                event_type="attack",
                qber=0.15 + 0.015 * np.random.randn(),  # Moderate QBER increase
                key_rate=800 + 75 * np.random.randn(),  # Reduced key rate
                detection_efficiency=0.6
                + 0.05 * np.random.randn(),  # Reduced efficiency
                source="alice",
                severity="high",
            )
            for i in range(50)
        ]

    def create_pns_attack(self):
        """Create synthetic photon-number-splitting attack data."""
        return [
            SecurityEvent(
                timestamp=datetime.now() + timedelta(seconds=i),
                event_type="attack",
                qber=0.08 + 0.01 * np.random.randn(),  # Slightly elevated QBER
                key_rate=900 + 60 * np.random.randn(),  # Reduced key rate
                multi_photon_probability=0.15
                + 0.02 * np.random.randn(),  # High multi-photon
                source="alice",
                severity="medium",
            )
            for i in range(50)
        ]

    def test_cryptographic_verification(self):
        """Test cryptographic integrity verification."""
        # Test message authentication
        message = b"QKD key material test data"
        key = b"shared_secret_key_32_bytes_long!"

        # Generate MAC
        mac = self.crypto_verifier.generate_mac(message, key)

        # Verify MAC
        is_valid = self.crypto_verifier.verify_mac(message, key, mac)
        assert is_valid is True

        # Test with tampered message
        tampered_message = b"QKD key material test DATA"  # Changed 'data' to 'DATA'
        is_valid_tampered = self.crypto_verifier.verify_mac(tampered_message, key, mac)
        assert is_valid_tampered is False

    def test_key_integrity_verification(self):
        """Test quantum key integrity verification."""
        # Generate test key material
        key_material = np.random.randint(0, 2, 1000).astype(np.uint8)

        # Test key randomness
        randomness_test = self.crypto_verifier.test_key_randomness(key_material)

        assert "entropy" in randomness_test
        assert "chi_square_p_value" in randomness_test
        assert "runs_test_p_value" in randomness_test
        assert "autocorrelation" in randomness_test

        # Entropy should be close to 1 for random data
        assert randomness_test["entropy"] > 0.9

        # Test key correlation with previous keys
        previous_key = np.random.randint(0, 2, 1000).astype(np.uint8)
        correlation = self.crypto_verifier.test_key_correlation(
            key_material, previous_key
        )

        assert "correlation_coefficient" in correlation
        assert "p_value" in correlation

        # Keys should be uncorrelated
        assert abs(correlation["correlation_coefficient"]) < 0.1

    def test_privacy_amplification_verification(self):
        """Test privacy amplification process verification."""
        # Raw key with some correlation
        raw_key = np.random.randint(0, 2, 2000).astype(np.uint8)

        # Simulate privacy amplification
        amplified_key = self.crypto_verifier.verify_privacy_amplification(
            raw_key, compression_ratio=0.5, hash_function="sha256"
        )

        assert len(amplified_key) == len(raw_key) // 2

        # Test that amplified key has better randomness properties
        raw_entropy = self.crypto_verifier.calculate_entropy(raw_key)
        amplified_entropy = self.crypto_verifier.calculate_entropy(amplified_key)

        # Amplified key should have high entropy
        assert amplified_entropy > 0.95

    def test_authentication_protocols(self):
        """Test quantum authentication protocols."""
        # Test unconditionally secure authentication
        message = b"Quantum key distribution test message"
        auth_key = np.random.randint(0, 2, 256).astype(np.uint8)

        # Generate authentication tag
        auth_tag = self.crypto_verifier.quantum_authenticate(message, auth_key)

        # Verify authentication
        is_authentic = self.crypto_verifier.verify_quantum_authentication(
            message, auth_key, auth_tag
        )

        assert is_authentic is True

        # Test with modified message
        modified_message = message + b"x"
        is_authentic_modified = self.crypto_verifier.verify_quantum_authentication(
            modified_message, auth_key, auth_tag
        )

        assert is_authentic_modified is False

    def test_security_parameter_calculation(self):
        """Test security parameter calculations."""
        # Test security parameter for different scenarios
        qber_values = [0.05, 0.1, 0.11, 0.15, 0.2]

        for qber in qber_values:
            security_params = self.monitor.calculate_security_parameters(
                qber=qber,
                key_length=1000,
                error_correction_efficiency=1.16,
                privacy_amplification_factor=1.4,
            )

            assert "mutual_information" in security_params
            assert "secret_key_rate" in security_params
            assert "security_level" in security_params

            # Higher QBER should result in lower security
            if qber <= 0.11:  # Below threshold
                assert security_params["security_level"] > 0
            else:  # Above threshold
                assert security_params["security_level"] <= 0

    def test_alert_generation(self):
        """Test security alert generation."""
        # Configure alert thresholds
        self.monitor.set_alert_thresholds(
            {
                "qber_high": 0.1,
                "key_rate_low": 700,
                "detection_efficiency_low": 0.7,
                "consecutive_anomalies": 3,
            }
        )

        # Process events that should trigger alerts
        for event in self.attack_events[:5]:
            self.monitor.process_event(event)

        # Check generated alerts
        alerts = self.monitor.get_alerts()

        assert len(alerts) > 0

        for alert in alerts:
            assert "timestamp" in alert
            assert "alert_type" in alert
            assert "severity" in alert
            assert "description" in alert
            assert "recommended_action" in alert

    def test_incident_response(self):
        """Test automated incident response."""
        # Configure response actions
        self.monitor.configure_incident_response(
            {
                "qber_high": ["log_incident", "notify_admin", "reduce_key_rate"],
                "attack_detected": ["log_incident", "notify_admin", "shutdown_link"],
                "key_exhaustion": ["log_incident", "notify_admin", "initiate_rekey"],
            }
        )

        # Trigger high QBER incident
        high_qber_event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="anomaly",
            qber=0.15,
            key_rate=500,
            source="alice",
            severity="critical",
        )

        # Process incident
        response = self.monitor.handle_incident(high_qber_event)

        assert "actions_taken" in response
        assert "incident_id" in response
        assert "timestamp" in response

        # Verify correct actions were taken
        actions = response["actions_taken"]
        assert "log_incident" in actions
        assert "notify_admin" in actions

    def test_forensic_analysis(self):
        """Test forensic analysis capabilities."""
        # Conduct forensic analysis on attack events
        forensic_report = self.monitor.conduct_forensic_analysis(
            self.attack_events, analysis_window=timedelta(minutes=10)
        )

        assert "attack_timeline" in forensic_report
        assert "attack_vectors" in forensic_report
        assert "impact_assessment" in forensic_report
        assert "evidence_collected" in forensic_report
        assert "recommendations" in forensic_report

        # Timeline should show progression of attack
        timeline = forensic_report["attack_timeline"]
        assert len(timeline) > 0
        assert all("timestamp" in entry for entry in timeline)
        assert all("event_type" in entry for entry in timeline)

    def test_compliance_monitoring(self):
        """Test compliance with security standards."""
        # Check compliance with quantum cryptography standards
        compliance_report = self.monitor.check_compliance(
            standards=["ETSI_QKD", "NIST_PQC", "ISO_IEC_23837"]
        )

        assert "overall_compliance" in compliance_report
        assert "standard_results" in compliance_report
        assert "non_compliance_issues" in compliance_report
        assert "recommendations" in compliance_report

        # Check specific compliance requirements
        for standard, result in compliance_report["standard_results"].items():
            assert "compliance_score" in result
            assert "required_measures" in result
            assert "implemented_measures" in result

    def test_security_metrics_collection(self):
        """Test security metrics collection and analysis."""
        # Collect security metrics over time
        metrics = self.monitor.collect_security_metrics(
            events=self.all_events, time_window=timedelta(hours=1)
        )

        assert "avg_qber" in metrics
        assert "avg_key_rate" in metrics
        assert "security_violations" in metrics
        assert "uptime_percentage" in metrics
        assert "threat_detection_rate" in metrics
        assert "false_positive_rate" in metrics

        # Verify metric ranges
        assert 0 <= metrics["uptime_percentage"] <= 100
        assert 0 <= metrics["threat_detection_rate"] <= 1
        assert 0 <= metrics["false_positive_rate"] <= 1

    @pytest.mark.benchmark
    def test_real_time_processing_benchmark(self, benchmark):
        """Benchmark real-time security event processing."""
        event = self.normal_events[0]

        result = benchmark(self.monitor.process_event, event)

        # Real-time processing should be fast
        assert benchmark.stats.mean < 0.01  # <10ms

    @pytest.mark.benchmark
    def test_threat_detection_benchmark(self, benchmark):
        """Benchmark threat detection performance."""
        result = benchmark(self.threat_detector.analyze_events, self.all_events)

        # Threat detection should complete in reasonable time
        assert benchmark.stats.mean < 1.0  # <1 second

    def test_secure_communication(self):
        """Test secure communication protocols."""
        # Test secure channel establishment
        channel = self.monitor.establish_secure_channel(
            endpoint="bob", authentication_method="quantum_signature"
        )

        assert channel["is_secure"] is True
        assert "session_key" in channel
        assert "authentication_verified" in channel

        # Test message transmission
        message = b"Secure quantum communication test"
        encrypted_message = self.monitor.secure_transmit(message, channel)

        # Test message reception and decryption
        decrypted_message = self.monitor.secure_receive(encrypted_message, channel)

        assert decrypted_message == message

    def test_key_escrow_monitoring(self):
        """Test key escrow and recovery monitoring."""
        # Monitor key escrow operations
        escrow_events = [
            {
                "operation": "key_deposit",
                "key_id": f"key_{i}",
                "timestamp": datetime.now() + timedelta(seconds=i),
                "authorized_parties": ["alice", "bob", "escrow_agent"],
            }
            for i in range(10)
        ]

        escrow_analysis = self.monitor.monitor_key_escrow(escrow_events)

        assert "escrow_integrity" in escrow_analysis
        assert "unauthorized_access_attempts" in escrow_analysis
        assert "key_recovery_requests" in escrow_analysis

        # Escrow integrity should be maintained
        assert escrow_analysis["escrow_integrity"] is True


class TestSecurityEvent:
    """Test suite for SecurityEvent data structure."""

    def test_event_creation(self):
        """Test security event creation."""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="measurement",
            qber=0.05,
            key_rate=1000,
            source="alice",
            severity="info",
        )

        assert event.timestamp is not None
        assert event.event_type == "measurement"
        assert event.qber == 0.05
        assert event.key_rate == 1000
        assert event.source == "alice"
        assert event.severity == "info"

    def test_event_validation(self):
        """Test event validation."""
        # Test invalid QBER
        with pytest.raises(ValueError):
            SecurityEvent(qber=-0.1)

        with pytest.raises(ValueError):
            SecurityEvent(qber=1.5)

        # Test invalid severity
        with pytest.raises(ValueError):
            SecurityEvent(severity="invalid")

    def test_event_serialization(self):
        """Test event serialization/deserialization."""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="anomaly",
            qber=0.1,
            key_rate=800,
            source="bob",
            severity="medium",
        )

        # Serialize to dict
        event_dict = event.to_dict()
        assert isinstance(event_dict, dict)
        assert "timestamp" in event_dict
        assert "qber" in event_dict

        # Deserialize from dict
        reconstructed = SecurityEvent.from_dict(event_dict)
        assert reconstructed.qber == event.qber
        assert reconstructed.key_rate == event.key_rate
        assert reconstructed.source == event.source


class TestThreatDetector:
    """Test suite for ThreatDetector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ThreatDetector()
        self.detector.configure(
            sensitivity=0.8, false_positive_rate=0.05, detection_window=60
        )

    def test_signature_based_detection(self):
        """Test signature-based threat detection."""
        # Load threat signatures
        signatures = {
            "intercept_resend": {
                "qber_min": 0.2,
                "qber_max": 0.3,
                "key_rate_reduction": 0.25,
            },
            "beam_splitting": {
                "qber_min": 0.12,
                "qber_max": 0.18,
                "efficiency_reduction": 0.3,
            },
        }

        self.detector.load_signatures(signatures)

        # Test detection
        test_event = SecurityEvent(
            timestamp=datetime.now(), qber=0.25, key_rate=750, source="alice"
        )

        detection_result = self.detector.signature_detect(test_event)

        assert "matches" in detection_result
        assert "confidence" in detection_result

        # Should match intercept-resend signature
        matches = detection_result["matches"]
        assert "intercept_resend" in matches


class TestCryptographicVerifier:
    """Test suite for CryptographicVerifier functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = CryptographicVerifier()

    def test_hash_chain_verification(self):
        """Test hash chain integrity verification."""
        # Create hash chain
        initial_value = b"genesis_block"
        chain_length = 10

        hash_chain = self.verifier.create_hash_chain(initial_value, chain_length)

        assert len(hash_chain) == chain_length + 1  # Include initial value

        # Verify chain integrity
        is_valid = self.verifier.verify_hash_chain(hash_chain)
        assert is_valid is True

        # Test with broken chain
        broken_chain = hash_chain.copy()
        broken_chain[5] = b"tampered_value"

        is_valid_broken = self.verifier.verify_hash_chain(broken_chain)
        assert is_valid_broken is False

    def test_merkle_tree_verification(self):
        """Test Merkle tree construction and verification."""
        # Create test data
        data_blocks = [f"block_{i}".encode() for i in range(8)]

        # Build Merkle tree
        merkle_tree = self.verifier.build_merkle_tree(data_blocks)

        assert "root" in merkle_tree
        assert "leaves" in merkle_tree
        assert "tree" in merkle_tree

        # Verify individual blocks
        for i, block in enumerate(data_blocks):
            proof = self.verifier.get_merkle_proof(merkle_tree, i)
            is_valid = self.verifier.verify_merkle_proof(
                block, proof, merkle_tree["root"]
            )
            assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__])
