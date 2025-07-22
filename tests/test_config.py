"""
Comprehensive test configuration and data for QKD failure detection system

This module provides test fixtures, mock data, and configuration for testing
all components of the QKD failure detection system.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class TestConfiguration:
    """Configuration for test scenarios"""

    # QKD System Parameters
    test_key_length: int = 1000
    test_error_rate: float = 0.02
    test_detection_efficiency: float = 0.8
    test_channel_loss: float = 0.05

    # Test Data Parameters
    num_baseline_sessions: int = 50
    num_test_sessions: int = 20
    anomaly_ratio: float = 0.1

    # Detection Thresholds
    qber_threshold: float = 0.11
    security_threshold: float = 0.11
    anomaly_threshold: float = 2.0

    # Signal Processing Parameters
    sampling_rate: float = 1000.0
    signal_duration: float = 1.0
    noise_level: float = 0.1

    # ML Parameters
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5


class TestDataGenerator:
    """Generate realistic test data for QKD systems"""

    def __init__(self, config: TestConfiguration):
        self.config = config
        np.random.seed(config.random_state)

    def generate_normal_qkd_session(self, session_id: int) -> Dict[str, Any]:
        """Generate a normal QKD session"""
        return {
            "session_id": session_id,
            "timestamp": 1600000000 + session_id * 60,  # Sequential timestamps
            "qber": np.random.normal(self.config.test_error_rate, 0.005),
            "sift_ratio": np.random.normal(0.5, 0.05),
            "initial_length": self.config.test_key_length,
            "final_key_length": int(np.random.normal(800, 50)),
            "channel_loss": np.random.normal(self.config.test_channel_loss, 0.01),
            "detection_efficiency": np.random.normal(
                self.config.test_detection_efficiency, 0.02
            ),
            "secure": True,
            "protocol": "BB84",
            "basis_mismatch_rate": np.random.normal(0.5, 0.05),
            "privacy_amplification_factor": np.random.normal(0.8, 0.1),
            "error_correction_overhead": np.random.normal(0.15, 0.02),
        }

    def generate_anomalous_qkd_session(
        self, session_id: int, anomaly_type: str
    ) -> Dict[str, Any]:
        """Generate an anomalous QKD session"""
        session = self.generate_normal_qkd_session(session_id)

        if anomaly_type == "high_qber":
            session["qber"] = np.random.normal(0.15, 0.02)
            session["secure"] = False

        elif anomaly_type == "low_sift_ratio":
            session["sift_ratio"] = np.random.normal(0.2, 0.05)
            session["final_key_length"] = int(session["final_key_length"] * 0.5)

        elif anomaly_type == "channel_attack":
            session["channel_loss"] = np.random.normal(0.2, 0.03)
            session["qber"] = np.random.normal(0.12, 0.02)
            session["secure"] = False

        elif anomaly_type == "detector_malfunction":
            session["detection_efficiency"] = np.random.normal(0.3, 0.1)
            session["final_key_length"] = int(session["final_key_length"] * 0.3)

        elif anomaly_type == "eavesdropping":
            session["qber"] = np.random.normal(0.25, 0.03)
            session["basis_mismatch_rate"] = np.random.normal(0.7, 0.05)
            session["secure"] = False

        return session

    def generate_baseline_sessions(self) -> List[Dict[str, Any]]:
        """Generate baseline normal sessions"""
        sessions = []
        for i in range(self.config.num_baseline_sessions):
            session = self.generate_normal_qkd_session(i)
            sessions.append(session)
        return sessions

    def generate_test_sessions(self) -> List[Dict[str, Any]]:
        """Generate test sessions with some anomalies"""
        sessions = []
        anomaly_types = [
            "high_qber",
            "low_sift_ratio",
            "channel_attack",
            "detector_malfunction",
            "eavesdropping",
        ]

        num_anomalies = int(self.config.num_test_sessions * self.config.anomaly_ratio)
        anomaly_indices = np.random.choice(
            self.config.num_test_sessions, num_anomalies, replace=False
        )

        for i in range(self.config.num_test_sessions):
            if i in anomaly_indices:
                anomaly_type = np.random.choice(anomaly_types)
                session = self.generate_anomalous_qkd_session(
                    self.config.num_baseline_sessions + i, anomaly_type
                )
                session["anomaly_type"] = anomaly_type
                session["is_anomaly"] = True
            else:
                session = self.generate_normal_qkd_session(
                    self.config.num_baseline_sessions + i
                )
                session["anomaly_type"] = "normal"
                session["is_anomaly"] = False

            sessions.append(session)

        return sessions

    def generate_time_series_data(
        self, num_points: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Generate time series data for signal analysis"""
        t = np.linspace(0, self.config.signal_duration, num_points)

        # Base signal components
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz carrier
        signal += 0.5 * np.sin(2 * np.pi * 50 * t)  # 50 Hz component
        signal += self.config.noise_level * np.random.randn(len(t))  # Noise

        return {
            "time": t,
            "signal": signal,
            "sampling_rate": self.config.sampling_rate,
            "duration": self.config.signal_duration,
        }

    def generate_quantum_channel_data(self, session: Dict[str, Any]) -> np.ndarray:
        """Generate quantum channel data for a session"""
        length = session["initial_length"]

        # Base quantum state
        states = np.random.choice([0, 1], size=length)

        # Apply channel effects
        noise = np.random.normal(0, session["channel_loss"], length)

        # Detection events
        detected = np.random.random(length) < session["detection_efficiency"]

        # Combine effects
        channel_data = states.astype(float) + noise
        channel_data = channel_data * detected

        return channel_data


class MockQKDHardware:
    """Mock QKD hardware for testing"""

    def __init__(self, config: TestConfiguration):
        self.config = config
        self.is_connected = False
        self.calibration_data = {}
        self.error_simulation = False

    def connect(self) -> bool:
        """Simulate hardware connection"""
        self.is_connected = True
        return True

    def disconnect(self) -> bool:
        """Simulate hardware disconnection"""
        self.is_connected = False
        return True

    def calibrate(self) -> Dict[str, float]:
        """Simulate hardware calibration"""
        if not self.is_connected:
            raise ConnectionError("Hardware not connected")

        self.calibration_data = {
            "detector_dark_count": np.random.normal(100, 10),
            "source_intensity": np.random.normal(1e6, 1e4),
            "optical_loss": np.random.normal(0.05, 0.01),
            "polarization_drift": np.random.normal(0, 0.01),
        }

        return self.calibration_data

    def measure_session(self, session_id: int) -> Dict[str, Any]:
        """Simulate measuring a QKD session"""
        if not self.is_connected:
            raise ConnectionError("Hardware not connected")

        generator = TestDataGenerator(self.config)

        if self.error_simulation:
            return generator.generate_anomalous_qkd_session(
                session_id, "detector_malfunction"
            )
        else:
            return generator.generate_normal_qkd_session(session_id)

    def enable_error_simulation(self):
        """Enable error simulation for testing"""
        self.error_simulation = True

    def disable_error_simulation(self):
        """Disable error simulation"""
        self.error_simulation = False


class TestScenarios:
    """Predefined test scenarios for comprehensive testing"""

    @staticmethod
    def scenario_normal_operation() -> Dict[str, Any]:
        """Normal QKD operation scenario"""
        config = TestConfiguration(
            test_error_rate=0.02, anomaly_ratio=0.0, num_test_sessions=50
        )
        generator = TestDataGenerator(config)

        return {
            "name": "Normal Operation",
            "description": "QKD system operating under normal conditions",
            "config": config,
            "baseline_sessions": generator.generate_baseline_sessions(),
            "test_sessions": generator.generate_test_sessions(),
            "expected_anomalies": 0,
        }

    @staticmethod
    def scenario_eavesdropping_attack() -> Dict[str, Any]:
        """Eavesdropping attack scenario"""
        config = TestConfiguration(
            test_error_rate=0.02, anomaly_ratio=0.3, num_test_sessions=30
        )
        generator = TestDataGenerator(config)

        # Force all anomalies to be eavesdropping
        test_sessions = []
        for i in range(config.num_test_sessions):
            if i < int(config.num_test_sessions * config.anomaly_ratio):
                session = generator.generate_anomalous_qkd_session(
                    config.num_baseline_sessions + i, "eavesdropping"
                )
                session["anomaly_type"] = "eavesdropping"
                session["is_anomaly"] = True
            else:
                session = generator.generate_normal_qkd_session(
                    config.num_baseline_sessions + i
                )
                session["anomaly_type"] = "normal"
                session["is_anomaly"] = False
            test_sessions.append(session)

        return {
            "name": "Eavesdropping Attack",
            "description": "Simulated eavesdropping attacks on QKD channel",
            "config": config,
            "baseline_sessions": generator.generate_baseline_sessions(),
            "test_sessions": test_sessions,
            "expected_anomalies": int(config.num_test_sessions * config.anomaly_ratio),
        }

    @staticmethod
    def scenario_hardware_degradation() -> Dict[str, Any]:
        """Hardware degradation scenario"""
        config = TestConfiguration(
            test_error_rate=0.02, anomaly_ratio=0.2, num_test_sessions=40
        )
        generator = TestDataGenerator(config)

        # Simulate gradual degradation
        test_sessions = []
        degradation_factor = 1.0

        for i in range(config.num_test_sessions):
            session = generator.generate_normal_qkd_session(
                config.num_baseline_sessions + i
            )

            # Apply degradation
            if i > config.num_test_sessions // 2:
                degradation_factor *= 0.98  # 2% degradation per session
                session["detection_efficiency"] *= degradation_factor
                session["channel_loss"] *= 2 - degradation_factor
                session["final_key_length"] = int(
                    session["final_key_length"] * degradation_factor
                )

                if degradation_factor < 0.8:
                    session["is_anomaly"] = True
                    session["anomaly_type"] = "hardware_degradation"
                else:
                    session["is_anomaly"] = False
                    session["anomaly_type"] = "normal"
            else:
                session["is_anomaly"] = False
                session["anomaly_type"] = "normal"

            test_sessions.append(session)

        return {
            "name": "Hardware Degradation",
            "description": "Gradual hardware performance degradation",
            "config": config,
            "baseline_sessions": generator.generate_baseline_sessions(),
            "test_sessions": test_sessions,
            "expected_anomalies": len([s for s in test_sessions if s["is_anomaly"]]),
        }

    @staticmethod
    def scenario_environmental_interference() -> Dict[str, Any]:
        """Environmental interference scenario"""
        config = TestConfiguration(
            test_error_rate=0.02,
            anomaly_ratio=0.15,
            num_test_sessions=35,
            noise_level=0.2,
        )
        generator = TestDataGenerator(config)

        # Simulate environmental effects
        test_sessions = []
        for i in range(config.num_test_sessions):
            session = generator.generate_normal_qkd_session(
                config.num_baseline_sessions + i
            )

            # Random environmental events
            if np.random.random() < config.anomaly_ratio:
                # Simulate temperature fluctuation, vibrations, etc.
                session["channel_loss"] += np.random.normal(0.05, 0.02)
                session["qber"] += np.random.normal(0.03, 0.01)
                session["basis_mismatch_rate"] += np.random.normal(0.1, 0.05)
                session["final_key_length"] = int(session["final_key_length"] * 0.7)
                session["is_anomaly"] = True
                session["anomaly_type"] = "environmental_interference"
            else:
                session["is_anomaly"] = False
                session["anomaly_type"] = "normal"

            test_sessions.append(session)

        return {
            "name": "Environmental Interference",
            "description": "Environmental factors affecting QKD performance",
            "config": config,
            "baseline_sessions": generator.generate_baseline_sessions(),
            "test_sessions": test_sessions,
            "expected_anomalies": len([s for s in test_sessions if s["is_anomaly"]]),
        }


class TestDataExporter:
    """Export test data for external analysis"""

    @staticmethod
    def export_to_csv(sessions: List[Dict[str, Any]], filename: str):
        """Export sessions to CSV"""
        df = pd.DataFrame(sessions)
        df.to_csv(filename, index=False)

    @staticmethod
    def export_to_json(data: Dict[str, Any], filename: str):
        """Export data to JSON"""
        with open(filename, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                elif isinstance(value, dict):
                    json_data[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    json_data[key] = value
            json.dump(json_data, f, indent=2)

    @staticmethod
    def create_test_report(
        scenario: Dict[str, Any], results: Dict[str, Any], filename: str
    ):
        """Create detailed test report"""
        report = {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "configuration": {
                "num_baseline_sessions": scenario["config"].num_baseline_sessions,
                "num_test_sessions": scenario["config"].num_test_sessions,
                "anomaly_ratio": scenario["config"].anomaly_ratio,
                "expected_anomalies": scenario["expected_anomalies"],
            },
            "results": results,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        with open(filename, "w") as f:
            json.dump(report, f, indent=2, default=str)


def setup_test_environment() -> Dict[str, Any]:
    """Setup complete test environment"""
    config = TestConfiguration()
    generator = TestDataGenerator(config)
    hardware = MockQKDHardware(config)

    # Create test data directory
    test_data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(test_data_dir, exist_ok=True)

    return {
        "config": config,
        "generator": generator,
        "hardware": hardware,
        "test_data_dir": test_data_dir,
        "scenarios": {
            "normal": TestScenarios.scenario_normal_operation(),
            "eavesdropping": TestScenarios.scenario_eavesdropping_attack(),
            "degradation": TestScenarios.scenario_hardware_degradation(),
            "interference": TestScenarios.scenario_environmental_interference(),
        },
    }


if __name__ == "__main__":
    # Demonstrate test data generation
    print("QKD Failure Detection - Test Data Configuration")
    print("=" * 50)
    print("Under the guidance of Vijayalaxmi Mogiligidda")
    print("=" * 50)

    # Setup test environment
    env = setup_test_environment()

    print(f"\nTest Configuration:")
    print(f"- Baseline sessions: {env['config'].num_baseline_sessions}")
    print(f"- Test sessions: {env['config'].num_test_sessions}")
    print(f"- Anomaly ratio: {env['config'].anomaly_ratio}")
    print(f"- QBER threshold: {env['config'].qber_threshold}")

    print(f"\nAvailable test scenarios:")
    for name, scenario in env["scenarios"].items():
        print(f"- {scenario['name']}: {scenario['description']}")
        print(f"  Expected anomalies: {scenario['expected_anomalies']}")

    # Generate sample data for each scenario
    print(f"\nGenerating sample test data...")
    for name, scenario in env["scenarios"].items():
        print(f"- Generating {name} scenario data...")

        # Export baseline sessions
        baseline_file = os.path.join(env["test_data_dir"], f"{name}_baseline.csv")
        TestDataExporter.export_to_csv(scenario["baseline_sessions"], baseline_file)

        # Export test sessions
        test_file = os.path.join(env["test_data_dir"], f"{name}_test.csv")
        TestDataExporter.export_to_csv(scenario["test_sessions"], test_file)

        print(f"  Saved to: {baseline_file}, {test_file}")

    print(f"\nTest environment setup complete!")
    print(f"Test data saved to: {env['test_data_dir']}")
