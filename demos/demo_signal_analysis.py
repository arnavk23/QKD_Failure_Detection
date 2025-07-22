"""
Demo: Signal Analysis for QKD Systems

This demonstration shows advanced signal processing and analysis techniques
for quantum key distribution systems.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from qkd_simulator import QKDSystemSimulator, QKDParameters
from signal_analyzer import QKDSignalAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function"""
    print("=" * 60)
    print("QKD SYSTEM SIGNAL ANALYSIS DEMONSTRATION")
    print("=" * 60)
    print()

    # Step 1: Initialize signal analyzer
    print("1. Initializing QKD signal analyzer...")
    sampling_rate = 1000.0  # Hz
    analyzer = QKDSignalAnalyzer(sampling_rate)
    print(f"   Sampling rate: {sampling_rate} Hz")
    print()

    # Step 2: Generate normal operation data for baseline
    print("2. Generating baseline signal data...")
    params = QKDParameters(
        key_length=1000,
        error_rate=0.015,
        detection_efficiency=0.85,
        channel_loss=0.03,
        protocol="BB84",
    )

    simulator = QKDSystemSimulator(params)
    normal_sessions = simulator.simulate_multiple_sessions(20)

    # Establish signal baseline
    analyzer.establish_signal_baseline(normal_sessions)
    print(f"   Baseline established from {len(normal_sessions)} normal sessions")
    print()

    # Step 3: Generate test sessions with various signal conditions
    print("3. Generating test sessions with signal variations...")

    test_sessions = []
    session_labels = []

    # Normal operation
    normal_test = simulator.simulate_multiple_sessions(15)
    test_sessions.extend(normal_test)
    session_labels.extend(["normal"] * 15)

    # High noise environment
    simulator.inject_failure("detector_noise", 0.05)
    noisy_sessions = simulator.simulate_multiple_sessions(10)
    test_sessions.extend(noisy_sessions)
    session_labels.extend(["high_noise"] * 10)

    # Channel degradation
    simulator.inject_failure("channel_loss", 0.08)
    degraded_sessions = simulator.simulate_multiple_sessions(10)
    test_sessions.extend(degraded_sessions)
    session_labels.extend(["channel_degraded"] * 10)

    # Timing instability
    simulator.inject_failure("timing_drift", 0.04)
    timing_sessions = simulator.simulate_multiple_sessions(8)
    test_sessions.extend(timing_sessions)
    session_labels.extend(["timing_unstable"] * 8)

    print(f"   Generated {len(test_sessions)} test sessions")
    print("   Test distribution:")
    for label in set(session_labels):
        count = session_labels.count(label)
        print(f"     {label}: {count} sessions")
    print()

    # Step 4: Perform comprehensive signal analysis
    print("4. Performing comprehensive signal analysis...")
    analysis_results = analyzer.batch_analyze_sessions(test_sessions)

    print(f"   Analyzed {len(analysis_results)} sessions")
    print("   Signal analysis completed")
    print()

    # Step 5: Extract and display key signal metrics
    print("5. Analyzing key signal metrics...")

    # Calculate statistics for different session types
    metrics_by_type = {}

    for i, (result, label) in enumerate(zip(analysis_results, session_labels)):
        if label not in metrics_by_type:
            metrics_by_type[label] = {
                "snr_values": [],
                "spectral_centroids": [],
                "signal_anomalies": [],
            }

        # Extract metrics
        snr = result["quality_metrics"]["snr_db"]
        spectral_centroid = result["frequency_features"]["spectral_centroid"]
        anomaly_count = (
            sum(result["signal_anomalies"].values())
            if result["signal_anomalies"]
            else 0
        )

        metrics_by_type[label]["snr_values"].append(snr)
        metrics_by_type[label]["spectral_centroids"].append(spectral_centroid)
        metrics_by_type[label]["signal_anomalies"].append(anomaly_count)

    # Display statistics
    print("   Signal Quality Statistics by Session Type:")
    print("   " + "-" * 60)

    for label, metrics in metrics_by_type.items():
        snr_mean = np.mean(metrics["snr_values"])
        snr_std = np.std(metrics["snr_values"])
        centroid_mean = np.mean(metrics["spectral_centroids"])
        anomaly_rate = np.mean(metrics["signal_anomalies"])

        print(f"   {label.upper():<15}")
        print(f"     SNR: {snr_mean:6.2f} ± {snr_std:5.2f} dB")
        print(f"     Spectral Centroid: {centroid_mean:6.2f} Hz")
        print(f"     Anomaly Rate: {anomaly_rate:6.2f} per session")
        print()

    # Step 6: Time domain analysis
    print("6. Performing time domain analysis...")

    # Analyze first few sessions in detail
    sample_sessions = analysis_results[:3]

    print("   Time Domain Features (First 3 Sessions):")
    for i, result in enumerate(sample_sessions):
        time_features = result["time_features"]
        session_id = result["session_id"]

        print(f"   Session {session_id}:")
        print(f"     RMS: {time_features['rms']:.4f}")
        print(f"     Peak-to-Peak: {time_features['peak_to_peak']:.4f}")
        print(f"     Crest Factor: {time_features['crest_factor']:.4f}")
        print(f"     Zero Crossing Rate: {time_features['zero_crossing_rate']:.4f}")
        print()

    # Step 7: Frequency domain analysis
    print("7. Performing frequency domain analysis...")

    print("   Frequency Domain Features (First 3 Sessions):")
    for i, result in enumerate(sample_sessions):
        freq_features = result["frequency_features"]
        session_id = result["session_id"]

        print(f"   Session {session_id}:")
        print(f"     Peak Frequency: {freq_features['peak_frequency']:.2f} Hz")
        print(f"     Spectral Spread: {freq_features['spectral_spread']:.2f} Hz")
        print(f"     Bandwidth: {freq_features['bandwidth']:.2f} Hz")
        print(f"     Spectral Flatness: {freq_features['spectral_flatness']:.4f}")
        print()

    # Step 8: Signal quality assessment
    print("8. Performing signal quality assessment...")

    quality_summary = {
        "excellent": 0,  # SNR > 20 dB
        "good": 0,  # 15 < SNR <= 20 dB
        "fair": 0,  # 10 < SNR <= 15 dB
        "poor": 0,  # SNR <= 10 dB
    }

    for result in analysis_results:
        snr = result["quality_metrics"]["snr_db"]

        if snr > 20:
            quality_summary["excellent"] += 1
        elif snr > 15:
            quality_summary["good"] += 1
        elif snr > 10:
            quality_summary["fair"] += 1
        else:
            quality_summary["poor"] += 1

    print("   Signal Quality Distribution:")
    for quality, count in quality_summary.items():
        percentage = count / len(analysis_results) * 100
        print(
            f"     {quality.capitalize():<10}: {count:3d} sessions ({percentage:5.1f}%)"
        )
    print()

    # Step 9: Anomaly pattern detection
    print("9. Analyzing signal anomaly patterns...")

    # Count different types of signal anomalies
    anomaly_types = {}

    for result in analysis_results:
        if result["signal_anomalies"]:
            for anomaly_type, detected in result["signal_anomalies"].items():
                if detected:
                    if anomaly_type not in anomaly_types:
                        anomaly_types[anomaly_type] = 0
                    anomaly_types[anomaly_type] += 1

    if anomaly_types:
        print("   Signal Anomaly Types Detected:")
        for anomaly_type, count in sorted(
            anomaly_types.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"     {anomaly_type:<25}: {count:3d} occurrences")
    else:
        print("   No signal anomalies detected")
    print()

    # Step 10: Pattern analysis across sessions
    print("10. Analyzing patterns across sessions...")

    # Check for pattern anomalies if available
    pattern_count = 0
    for result in analysis_results:
        if "pattern_anomalies" in result:
            pattern_anomalies = result["pattern_anomalies"]
            for pattern_type, detected in pattern_anomalies.items():
                if detected:
                    pattern_count += 1

    if pattern_count > 0:
        print(f"    Pattern anomalies detected: {pattern_count}")
        print("    This indicates temporal correlations in signal degradation")
    else:
        print("    No significant pattern anomalies detected")
        print("    Signal behavior appears independent across sessions")
    print()

    # Step 11: Visualize signal analysis
    print("11. Creating signal analysis visualizations...")
    plt.rcParams["figure.figsize"] = (18, 12)
    analyzer.plot_signal_analysis(
        analysis_results, save_path="../plots/signal_analysis/demo_signal_analysis.png"
    )

    # Step 12: Correlation analysis
    print("12. Performing correlation analysis...")

    # Extract key metrics for correlation analysis
    qber_values = [result["qkd_params"]["qber"] for result in analysis_results]
    snr_values = [result["quality_metrics"]["snr_db"] for result in analysis_results]
    spectral_centroids = [
        result["frequency_features"]["spectral_centroid"] for result in analysis_results
    ]

    # Calculate correlations
    qber_snr_corr = np.corrcoef(qber_values, snr_values)[0, 1]
    qber_centroid_corr = np.corrcoef(qber_values, spectral_centroids)[0, 1]

    print("   Signal-QKD Parameter Correlations:")
    print(f"     QBER vs SNR: {qber_snr_corr:6.3f}")
    print(f"     QBER vs Spectral Centroid: {qber_centroid_corr:6.3f}")
    print()

    # Step 13: Signal degradation analysis
    print("13. Analyzing signal degradation patterns...")

    # Track signal quality over sessions
    degradation_detected = False

    if len(snr_values) > 10:
        # Split into early and late sessions
        early_snr = np.mean(snr_values[: len(snr_values) // 2])
        late_snr = np.mean(snr_values[len(snr_values) // 2 :])

        degradation = early_snr - late_snr

        if degradation > 2.0:  # Significant degradation
            degradation_detected = True
            print(f"    ⚠️  Signal degradation detected: {degradation:.2f} dB")
            print("    Recommendation: Check system calibration")
        else:
            print(f"    ✅ Signal quality stable: {degradation:.2f} dB variation")

    print()

    # Step 14: Real-time signal monitoring simulation
    print("14. Simulating real-time signal monitoring...")

    # Process new sessions in real-time
    monitoring_sessions = simulator.simulate_multiple_sessions(5)

    print("   Real-time Signal Monitoring Results:")
    for i, session in enumerate(monitoring_sessions):
        result = analyzer.analyze_qkd_session(session)

        snr = result["quality_metrics"]["snr_db"]
        anomaly_count = (
            sum(result["signal_anomalies"].values())
            if result["signal_anomalies"]
            else 0
        )

        status = "⚠️  ALERT" if anomaly_count > 0 or snr < 10 else "✅ OK"
        print(
            f"   Session {i+1}: SNR = {snr:5.1f} dB, Anomalies = {anomaly_count}, Status = {status}"
        )

    print()

    print("=" * 60)
    print("SIGNAL ANALYSIS DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Key Achievements:")
    print("• Time domain signal characterization")
    print("• Frequency domain spectral analysis")
    print("• Signal quality assessment and monitoring")
    print("• Anomaly detection in signal patterns")
    print("• Real-time signal processing capabilities")
    print("• Correlation analysis with QKD parameters")
    print("• Signal degradation detection")
    print()
    print("Generated artifacts:")
    print("• Signal analysis plots: ../plots/signal_analysis/")
    print("• Signal quality reports and metrics")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Signal analysis demo failed: {e}")
        raise
