"""
QKD System Simulator

This module implements a comprehensive quantum key distribution system simulator
supporting multiple protocols and failure injection capabilities.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import numpy as np
import random  # Only for non-crypto uses
import secrets  # For cryptographically secure randomness
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QKDParameters:
    """QKD system parameters"""

    key_length: int = 1000
    error_rate: float = 0.02
    detection_efficiency: float = 0.8
    dark_count_rate: float = 1e-6
    channel_loss: float = 0.1
    protocol: str = "BB84"


@dataclass
class QuantumState:
    """Quantum state representation"""

    basis: str  # '+' for rectilinear, 'x' for diagonal
    bit: int  # 0 or 1
    measured: bool = False
    error: bool = False


class BB84Protocol:
    """BB84 Quantum Key Distribution Protocol Implementation"""

    def __init__(self, params: QKDParameters):
        self.params = params
        self.alice_bits = []
        self.alice_bases = []
        self.bob_bases = []
        self.bob_measurements = []
        self.sifted_key = []
        self.final_key = []

    def generate_random_bits(self, length: int) -> List[int]:
        """Generate cryptographically secure random bit sequence"""
        return [secrets.randbelow(2) for _ in range(length)]

    def generate_random_bases(self, length: int) -> List[str]:
        """Generate cryptographically secure random basis sequence"""
        return [secrets.choice(["+", "x"]) for _ in range(length)]

    def prepare_quantum_states(
        self, bits: List[int], bases: List[str]
    ) -> List[QuantumState]:
        """Prepare quantum states based on bits and bases"""
        states = []
        for bit, basis in zip(bits, bases):
            state = QuantumState(basis=basis, bit=bit)
            states.append(state)
        return states

    def simulate_channel_transmission(
        self, states: List[QuantumState]
    ) -> List[QuantumState]:
        """Simulate quantum channel transmission with noise"""
        transmitted_states = []
        for state in states:
            new_state = QuantumState(basis=state.basis, bit=state.bit)

            # Apply channel loss
            if secrets.randbelow(10**6) < int(self.params.channel_loss * 10**6):
                new_state.bit = -1  # Lost photon

            # Apply bit flip error
            elif secrets.randbelow(10**6) < int(self.params.error_rate * 10**6):
                new_state.bit = 1 - new_state.bit
                new_state.error = True

            transmitted_states.append(new_state)
        return transmitted_states

    def measure_states(self, states: List[QuantumState], bases: List[str]) -> List[int]:
        """Bob's measurement of quantum states"""
        measurements = []
        for state, basis in zip(states, bases):
            if state.bit == -1:  # Lost photon
                measurements.append(-1)
                continue

            # Same basis measurement
            if state.basis == basis:
                # Perfect measurement with detection efficiency
                if secrets.randbelow(10**6) < int(self.params.detection_efficiency * 10**6):
                    measurements.append(state.bit)
                else:
                    measurements.append(-1)  # No detection
            else:
                # Different basis - random result
                if secrets.randbelow(10**6) < int(self.params.detection_efficiency * 10**6):
                    measurements.append(secrets.randbelow(2))
                else:
                    measurements.append(-1)  # No detection
        return measurements

    def sift_key(
        self,
        alice_bits: List[int],
        alice_bases: List[str],
        bob_bases: List[str],
        bob_measurements: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Sift the key keeping only matching bases and successful measurements"""
        alice_sifted = []
        bob_sifted = []

        for a_bit, a_basis, b_basis, b_measurement in zip(
            alice_bits, alice_bases, bob_bases, bob_measurements
        ):
            if a_basis == b_basis and b_measurement != -1:
                alice_sifted.append(a_bit)
                bob_sifted.append(b_measurement)

        return alice_sifted, bob_sifted

    def estimate_error_rate(
        self, alice_key: List[int], bob_key: List[int], sample_size: Optional[int] = None
    ) -> float:
        """Estimate quantum bit error rate (QBER)"""
        if sample_size is None:
            sample_size = min(len(alice_key), len(bob_key)) // 10

        if len(alice_key) < sample_size or len(bob_key) < sample_size:
            return 0.0

        errors = 0
        for i in range(sample_size):
            if alice_key[i] != bob_key[i]:
                errors += 1

        return errors / sample_size

    def privacy_amplification(
        self, key: List[int], compression_ratio: float = 0.8
    ) -> List[int]:
        """Apply privacy amplification to reduce key length"""
        final_length = int(len(key) * compression_ratio)
        return key[:final_length]

    def run_protocol(self) -> Dict:
        """Run complete BB84 protocol"""
        logger.info("Starting BB84 protocol simulation...")

        # Step 1: Alice generates random bits and bases
        self.alice_bits = self.generate_random_bits(self.params.key_length)
        self.alice_bases = self.generate_random_bases(self.params.key_length)

        # Step 2: Alice prepares quantum states
        quantum_states = self.prepare_quantum_states(self.alice_bits, self.alice_bases)

        # Step 3: Quantum channel transmission
        transmitted_states = self.simulate_channel_transmission(quantum_states)

        # Step 4: Bob chooses random measurement bases
        self.bob_bases = self.generate_random_bases(self.params.key_length)

        # Step 5: Bob measures the quantum states
        self.bob_measurements = self.measure_states(transmitted_states, self.bob_bases)

        # Step 6: Basis reconciliation and key sifting
        alice_sifted, bob_sifted = self.sift_key(
            self.alice_bits, self.alice_bases, self.bob_bases, self.bob_measurements
        )

        # Step 7: Error rate estimation
        qber = self.estimate_error_rate(alice_sifted, bob_sifted)

        # Step 8: Privacy amplification
        if qber < 0.11:  # Security threshold
            self.final_key = self.privacy_amplification(alice_sifted)
            secure = True
        else:
            self.final_key = []
            secure = False

        results = {
            "initial_length": self.params.key_length,
            "sifted_length": len(alice_sifted),
            "final_key_length": len(self.final_key),
            "qber": qber,
            "secure": secure,
            "sift_ratio": (
                len(alice_sifted) / self.params.key_length
                if self.params.key_length > 0
                else 0
            ),
            "alice_key": alice_sifted[:100],  # First 100 bits for analysis
            "bob_key": bob_sifted[:100],
            "channel_loss": self.params.channel_loss,
            "error_rate": self.params.error_rate,
        }

        logger.info(
            f"Protocol completed. QBER: {qber:.4f}, Final key length: {len(self.final_key)}"
        )
        return results


class QKDSystemSimulator:
    """Main QKD System Simulator"""

    def __init__(self, params: Optional[QKDParameters] = None):
        self.params = params or QKDParameters()
        self.simulation_history = []

    def inject_failure(self, failure_type: str, intensity: float = 0.1):
        """Inject various types of failures into the system"""
        if failure_type == "channel_loss":
            self.params.channel_loss += intensity
        elif failure_type == "detector_noise":
            self.params.dark_count_rate += intensity
        elif failure_type == "timing_drift":
            self.params.detection_efficiency -= intensity
        elif failure_type == "eavesdropping":
            self.params.error_rate += intensity
        elif failure_type == "source_instability":
            self.params.error_rate += intensity * 0.5

    def simulate_session(self, session_id: int = 0) -> Dict:
        """Simulate a single QKD session"""
        if self.params.protocol == "BB84":
            protocol = BB84Protocol(self.params)
            results = protocol.run_protocol()
            results["session_id"] = session_id
            results["timestamp"] = np.datetime64("now")

            self.simulation_history.append(results)
            return results
        else:
            raise NotImplementedError(
                f"Protocol {self.params.protocol} not implemented"
            )

    def simulate_multiple_sessions(self, num_sessions: int = 100) -> List[Dict]:
        """Simulate multiple QKD sessions"""
        logger.info(f"Simulating {num_sessions} QKD sessions...")
        results = []

        for i in range(num_sessions):
            session_result = self.simulate_session(session_id=i)
            results.append(session_result)

            # Occasionally inject random failures
            if secrets.randbelow(10) == 0:  # 10% chance of failure
                failure_type = secrets.choice(
                    [
                        "channel_loss",
                        "detector_noise",
                        "timing_drift",
                        "eavesdropping",
                        "source_instability",
                    ]
                )
                # secrets does not have uniform, so use a secure float in [0.01, 0.05)
                intensity = 0.01 + (secrets.randbelow(4000) / 100000.0)
                self.inject_failure(failure_type, intensity)

        return results

    def get_system_statistics(self) -> Dict:
        """Calculate system performance statistics"""
        if not self.simulation_history:
            return {}

        qbers = [session["qber"] for session in self.simulation_history]
        sift_ratios = [session["sift_ratio"] for session in self.simulation_history]
        final_lengths = [
            session["final_key_length"] for session in self.simulation_history
        ]

        stats = {
            "total_sessions": len(self.simulation_history),
            "mean_qber": np.mean(qbers),
            "std_qber": np.std(qbers),
            "mean_sift_ratio": np.mean(sift_ratios),
            "mean_final_key_length": np.mean(final_lengths),
            "success_rate": sum(
                1 for session in self.simulation_history if session["secure"]
            )
            / len(self.simulation_history),
        }

        return stats

    def plot_performance_metrics(self, save_path: Optional[str] = None):
        """Plot system performance metrics"""
        if not self.simulation_history:
            logger.warning("No simulation history available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # QBER over time
        qbers = [session["qber"] for session in self.simulation_history]
        axes[0, 0].plot(qbers)
        axes[0, 0].set_title("Quantum Bit Error Rate (QBER)")
        axes[0, 0].set_xlabel("Session")
        axes[0, 0].set_ylabel("QBER")
        axes[0, 0].axhline(
            y=0.11, color="r", linestyle="--", label="Security Threshold"
        )
        axes[0, 0].legend()

        # Sift ratio over time
        sift_ratios = [session["sift_ratio"] for session in self.simulation_history]
        axes[0, 1].plot(sift_ratios)
        axes[0, 1].set_title("Key Sifting Efficiency")
        axes[0, 1].set_xlabel("Session")
        axes[0, 1].set_ylabel("Sift Ratio")

        # Final key length distribution
        final_lengths = [
            session["final_key_length"] for session in self.simulation_history
        ]
        axes[1, 0].hist(final_lengths, bins=20, alpha=0.7)
        axes[1, 0].set_title("Final Key Length Distribution")
        axes[1, 0].set_xlabel("Key Length")
        axes[1, 0].set_ylabel("Frequency")

        # QBER distribution
        axes[1, 1].hist(qbers, bins=20, alpha=0.7)
        axes[1, 1].set_title("QBER Distribution")
        axes[1, 1].set_xlabel("QBER")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].axvline(
            x=0.11, color="r", linestyle="--", label="Security Threshold"
        )
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def export_data(self, filename: str):
        """Export simulation data to JSON file"""
        with open(filename, "w") as f:
            # Convert numpy datetime to string for JSON serialization
            export_data = []
            for session in self.simulation_history:
                session_copy = session.copy()
                session_copy["timestamp"] = str(session_copy["timestamp"])
                export_data.append(session_copy)
            json.dump(export_data, f, indent=2)


if __name__ == "__main__":
    # Example usage
    params = QKDParameters(
        key_length=2000, error_rate=0.02, detection_efficiency=0.8, channel_loss=0.1
    )

    simulator = QKDSystemSimulator(params)

    # Run single session
    result = simulator.simulate_session()
    print(f"Single session result: QBER = {result['qber']:.4f}")

    # Run multiple sessions
    results = simulator.simulate_multiple_sessions(50)

    # Print statistics
    stats = simulator.get_system_statistics()
    print("\nSystem Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

    # Plot results
    simulator.plot_performance_metrics()

    # Export data
    simulator.export_data("qkd_simulation_data.json")

# Create aliases for backward compatibility with test modules
QKDSimulator = QKDSystemSimulator


# QKDSession class for test compatibility
@dataclass
class QKDSession:
    """QKD session data class"""

    session_id: str
    qber: float
    key_rate: float
    sift_ratio: float
    success: bool
    timestamp: Optional[float] = None
    protocol: str = "BB84"
    raw_data: Optional[Dict] = None
