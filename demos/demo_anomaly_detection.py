"""
Demo: Anomaly Detection for QKD Systems

This demonstration shows the statistical and ML-based anomaly detection
capabilities for quantum key distribution systems.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from qkd_simulator import QKDSystemSimulator, QKDParameters
from anomaly_detector import QKDAnomalyDetector
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function"""
    print("=" * 60)
    print("QKD SYSTEM ANOMALY DETECTION DEMONSTRATION")
    print("=" * 60)
    print()

    # Step 1: Generate baseline data (normal operation)
    print("1. Generating baseline QKD data (normal operation)...")
    params = QKDParameters(
        key_length=1000,
        error_rate=0.02,
        detection_efficiency=0.8,
        channel_loss=0.05,
        protocol="BB84",
    )

    simulator = QKDSystemSimulator(params)
    baseline_sessions = simulator.simulate_multiple_sessions(100)

    baseline_stats = simulator.get_system_statistics()
    print(f"   Baseline sessions: {baseline_stats['total_sessions']}")
    print(f"   Mean QBER: {baseline_stats['mean_qber']:.4f}")
    print(f"   Success rate: {baseline_stats['success_rate']:.2%}")
    print()

    # Step 2: Initialize anomaly detector
    print("2. Initializing anomaly detection system...")
    detector = QKDAnomalyDetector(
        {
            "window_size": 50,
            "sigma_threshold": 3.0,
            "contamination": 0.1,
            "qber_threshold": 0.11,
            "sift_ratio_threshold": 0.3,
            "key_length_threshold": 100,
        }
    )

    # Establish baseline
    detector.establish_baseline(baseline_sessions)
    print("   Baseline established for statistical anomaly detection")
    print()

    # Step 3: Generate test data with injected failures
    print("3. Generating test data with simulated failures...")

    # Inject various types of failures
    test_sessions = []

    # Normal operation (first 30 sessions)
    normal_test = simulator.simulate_multiple_sessions(30)
    test_sessions.extend(normal_test)

    # Eavesdropping attack (sessions 31-40)
    simulator.inject_failure("eavesdropping", 0.05)
    eavesdrop_sessions = simulator.simulate_multiple_sessions(10)
    test_sessions.extend(eavesdrop_sessions)

    # Channel degradation (sessions 41-50)
    simulator.inject_failure("channel_loss", 0.08)
    channel_sessions = simulator.simulate_multiple_sessions(10)
    test_sessions.extend(channel_sessions)

    # Detector noise (sessions 51-60)
    simulator.inject_failure("detector_noise", 0.03)
    noise_sessions = simulator.simulate_multiple_sessions(10)
    test_sessions.extend(noise_sessions)

    print(f"   Generated {len(test_sessions)} test sessions with various failures")
    print()

    # Step 4: Perform anomaly detection
    print("4. Performing comprehensive anomaly detection...")
    results = detector.detect_anomalies(test_sessions)

    # Analysis summary
    overall_anomaly = results["overall_anomaly"]
    anomaly_count = np.sum(overall_anomaly > 0.3)

    print(f"   Total anomalies detected: {anomaly_count}/{len(test_sessions)}")
    print(f"   Anomaly detection rate: {anomaly_count/len(test_sessions):.1%}")
    print()

    # Step 5: Generate detailed report
    print("5. Generating anomaly detection report...")
    report = detector.generate_anomaly_report(results)
    print(report)
    print()

    # Step 6: Visualize results
    print("6. Creating visualization plots...")
    plt.rcParams["figure.figsize"] = (15, 10)
    detector.plot_anomaly_analysis(
        results, save_path="../plots/anomaly_detection/demo_analysis.png"
    )

    # Step 7: Detailed analysis of detection methods
    print("7. Analyzing individual detection methods...")

    # Statistical anomalies
    stat_anomalies = results["statistical"]
    print("   Statistical Anomaly Detection:")
    for method, anomalies in stat_anomalies.items():
        if isinstance(anomalies, np.ndarray):
            count = np.sum(anomalies)
            print(f"     {method}: {count} anomalies")

    # ML anomalies
    ml_anomalies = results["ml"]
    print("   Machine Learning Anomaly Detection:")
    for method, anomalies in ml_anomalies.items():
        if isinstance(anomalies, np.ndarray):
            count = np.sum(anomalies)
            print(f"     {method}: {count} anomalies")

    # Domain-specific anomalies
    domain_anomalies = results["domain_specific"]
    print("   Domain-Specific Anomaly Detection:")
    for method, anomalies in domain_anomalies.items():
        if isinstance(anomalies, np.ndarray):
            count = np.sum(anomalies)
            print(f"     {method}: {count} anomalies")

    print()

    # Step 8: Performance evaluation
    print("8. Evaluating detection performance...")

    # Create ground truth (we know sessions 31-60 have failures)
    ground_truth = np.array([0] * 30 + [1] * 30)  # 0 = normal, 1 = anomaly
    predicted = (overall_anomaly > 0.3).astype(int)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(ground_truth, predicted)
    precision = precision_score(ground_truth, predicted)
    recall = recall_score(ground_truth, predicted)
    f1 = f1_score(ground_truth, predicted)

    print(f"   Detection Performance:")
    print(f"     Accuracy: {accuracy:.3f}")
    print(f"     Precision: {precision:.3f}")
    print(f"     Recall: {recall:.3f}")
    print(f"     F1-Score: {f1:.3f}")
    print()

    # Step 9: Real-time simulation
    print("9. Simulating real-time anomaly detection...")

    # Simulate streaming sessions
    print("   Processing streaming QKD sessions...")

    streaming_sessions = simulator.simulate_multiple_sessions(20)

    for i, session in enumerate(streaming_sessions):
        # Process each session individually
        single_result = detector.detect_anomalies([session])
        anomaly_score = single_result["overall_anomaly"][0]

        status = "ANOMALY" if anomaly_score > 0.3 else "NORMAL"
        print(
            f"   Session {session['session_id']}: {status} (Score: {anomaly_score:.3f})"
        )

    print()
    print("=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Key Achievements:")
    print("• Statistical anomaly detection with control charts")
    print("• Machine learning based pattern recognition")
    print("• Domain-specific QKD failure detection")
    print("• Real-time processing capabilities")
    print("• Comprehensive performance evaluation")
    print()
    print("Check the generated plots in: ../plots/anomaly_detection/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
