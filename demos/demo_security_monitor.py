"""
Demo: Security Monitoring for QKD Systems

This demonstration shows comprehensive security monitoring and eavesdropping
detection capabilities for quantum key distribution systems.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from qkd_simulator import QKDSystemSimulator, QKDParameters
from security_monitor import QKDSecuritySystem
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function"""
    print("=" * 60)
    print("QKD SYSTEM SECURITY MONITORING DEMONSTRATION")
    print("=" * 60)
    print()

    # Step 1: Initialize security monitoring system
    print("1. Initializing QKD security monitoring system...")

    security_config = {
        "security_threshold": 0.11,
        "alert_threshold": 0.7,
        "monitoring_enabled": True,
    }

    security_system = QKDSecuritySystem(security_config)
    print("   Security system initialized")
    print(f"   Security threshold: {security_config['security_threshold']}")
    print(f"   Alert threshold: {security_config['alert_threshold']}")
    print()

    # Step 2: Generate secure baseline data
    print("2. Generating secure baseline operations...")

    secure_params = QKDParameters(
        key_length=1200,
        error_rate=0.015,
        detection_efficiency=0.9,
        channel_loss=0.02,
        protocol="BB84",
    )

    simulator = QKDSystemSimulator(secure_params)
    baseline_sessions = simulator.simulate_multiple_sessions(60)

    # Initialize security system with baseline
    security_system.initialize_system(baseline_sessions)

    baseline_stats = simulator.get_system_statistics()
    print(
        f"   Baseline established from {baseline_stats['total_sessions']} secure sessions"
    )
    print(
        f"   Baseline QBER: {baseline_stats['mean_qber']:.4f} ± {baseline_stats['std_qber']:.4f}"
    )
    print(f"   Success rate: {baseline_stats['success_rate']:.1%}")
    print()

    # Step 3: Generate test scenarios with security threats
    print("3. Generating test scenarios with security threats...")

    test_sessions = []
    threat_labels = []

    # Normal secure operation (30 sessions)
    normal_sessions = simulator.simulate_multiple_sessions(30)
    test_sessions.extend(normal_sessions)
    threat_labels.extend(["secure"] * 30)

    # Intercept-resend attack simulation (15 sessions)
    print("   Simulating intercept-resend attack...")
    simulator.inject_failure("eavesdropping", 0.08)  # High QBER increase
    intercept_sessions = simulator.simulate_multiple_sessions(15)
    test_sessions.extend(intercept_sessions)
    threat_labels.extend(["intercept_resend"] * 15)

    # Beam splitting attack simulation (10 sessions)
    print("   Simulating beam splitting attack...")
    simulator.inject_failure("channel_loss", 0.12)  # High channel loss
    beam_split_sessions = simulator.simulate_multiple_sessions(10)
    test_sessions.extend(beam_split_sessions)
    threat_labels.extend(["beam_splitting"] * 10)

    # Photon number splitting vulnerability (8 sessions)
    print("   Simulating PNS vulnerability...")
    # Simulate higher multi-photon rate
    for session in beam_split_sessions[-8:]:
        session["mean_photon_number"] = 0.15  # Higher than secure level
    pns_sessions = beam_split_sessions[-8:]
    test_sessions.extend(pns_sessions)
    threat_labels.extend(["pns_vulnerability"] * 8)

    # System degradation (12 sessions)
    print("   Simulating system degradation...")
    simulator.inject_failure("detector_noise", 0.06)
    degradation_sessions = simulator.simulate_multiple_sessions(12)
    test_sessions.extend(degradation_sessions)
    threat_labels.extend(["system_degradation"] * 12)

    print(f"   Generated {len(test_sessions)} test sessions")
    print("   Threat scenario distribution:")
    for label in set(threat_labels):
        count = threat_labels.count(label)
        print(f"     {label}: {count} sessions")
    print()

    # Step 4: Perform comprehensive security analysis
    print("4. Performing comprehensive security analysis...")
    security_results = security_system.batch_security_analysis(test_sessions)

    print(f"   Analyzed {len(security_results)} sessions")

    # Count security events
    attacks_detected = sum(
        1 for result in security_results if result["attack_detected"]
    )
    high_confidence = sum(
        1 for result in security_results if result["max_confidence"] > 0.8
    )

    print(f"   Attacks detected: {attacks_detected}")
    print(f"   High confidence detections: {high_confidence}")
    print()

    # Step 5: Analyze detection performance by threat type
    print("5. Analyzing detection performance by threat type...")

    detection_by_threat = {}

    for i, (result, label) in enumerate(zip(security_results, threat_labels)):
        if label not in detection_by_threat:
            detection_by_threat[label] = {
                "total": 0,
                "detected": 0,
                "high_confidence": 0,
                "avg_security_score": [],
            }

        detection_by_threat[label]["total"] += 1

        if result["attack_detected"]:
            detection_by_threat[label]["detected"] += 1

        if result["max_confidence"] > 0.8:
            detection_by_threat[label]["high_confidence"] += 1

        detection_by_threat[label]["avg_security_score"].append(
            result["overall_security_score"]
        )

    print("   Detection Performance by Threat Type:")
    print("   " + "-" * 70)

    for threat, stats in detection_by_threat.items():
        detection_rate = stats["detected"] / stats["total"] * 100
        avg_score = np.mean(stats["avg_security_score"])

        print(f"   {threat.upper():<20}")
        print(
            f"     Detection Rate: {detection_rate:5.1f}% ({stats['detected']}/{stats['total']})"
        )
        print(f"     Avg Security Score: {avg_score:5.3f}")
        print(f"     High Confidence: {stats['high_confidence']}")
        print()

    # Step 6: Security metrics analysis
    print("6. Analyzing security metrics...")

    # Extract security metrics
    info_metrics = []
    channel_metrics = []

    for result in security_results:
        info_metrics.append(result["security_metrics"]["information"])
        channel_metrics.append(result["security_metrics"]["channel"])

    # Calculate average metrics
    avg_secret_key_rate = np.mean([m["secret_key_rate"] for m in info_metrics])
    avg_security_margin = np.mean([m["security_margin"] for m in channel_metrics])
    avg_channel_fidelity = np.mean([m["channel_fidelity"] for m in channel_metrics])

    print("   Average Security Metrics:")
    print(f"     Secret Key Rate: {avg_secret_key_rate:.4f}")
    print(f"     Security Margin: {avg_security_margin:.4f}")
    print(f"     Channel Fidelity: {avg_channel_fidelity:.4f}")
    print()

    # Step 7: Alert analysis
    print("7. Analyzing security alerts...")

    # Check for alerts in monitoring system
    if hasattr(security_system.monitor, "alert_history"):
        alerts = security_system.monitor.alert_history

        if alerts:
            print(f"   Total alerts generated: {len(alerts)}")

            # Analyze alert severity
            high_severity = sum(1 for alert in alerts if alert["severity"] == "HIGH")
            medium_severity = sum(
                1 for alert in alerts if alert["severity"] == "MEDIUM"
            )

            print(f"     High severity: {high_severity}")
            print(f"     Medium severity: {medium_severity}")

            # Show recent alerts
            print("   Recent Security Alerts:")
            for alert in alerts[-5:]:  # Last 5 alerts
                print(f"     {alert['severity']}: {alert['message']}")
        else:
            print("   No security alerts generated")

    print()

    # Step 8: Attack signature analysis
    print("8. Analyzing attack signatures...")

    # Analyze attack detection patterns
    attack_signatures = {}

    for result in security_results:
        for attack_type, attack_result in result["attack_types"].items():
            if attack_type not in attack_signatures:
                attack_signatures[attack_type] = {
                    "detections": 0,
                    "total_confidence": 0,
                    "indicators": {},
                }

            if attack_result["attack_detected"]:
                attack_signatures[attack_type]["detections"] += 1
                attack_signatures[attack_type]["total_confidence"] += attack_result[
                    "confidence"
                ]

                # Count indicators
                for indicator in attack_result["indicators"]:
                    if indicator not in attack_signatures[attack_type]["indicators"]:
                        attack_signatures[attack_type]["indicators"][indicator] = 0
                    attack_signatures[attack_type]["indicators"][indicator] += 1

    print("   Attack Signature Analysis:")
    for attack_type, sig in attack_signatures.items():
        if sig["detections"] > 0:
            avg_confidence = sig["total_confidence"] / sig["detections"]
            print(f"   {attack_type.replace('_', ' ').title()}:")
            print(f"     Detections: {sig['detections']}")
            print(f"     Avg Confidence: {avg_confidence:.3f}")
            print(f"     Top Indicators:")

            # Show top 3 indicators
            sorted_indicators = sorted(
                sig["indicators"].items(), key=lambda x: x[1], reverse=True
            )
            for indicator, count in sorted_indicators[:3]:
                print(f"       {indicator}: {count}")
            print()

    # Step 9: Real-time monitoring simulation
    print("9. Simulating real-time security monitoring...")

    # Process streaming sessions
    streaming_sessions = simulator.simulate_multiple_sessions(8)

    print("   Real-time Security Monitoring:")
    for i, session in enumerate(streaming_sessions):
        result = security_system.monitor_session(session)

        security_score = result["overall_security_score"]
        attack_detected = result["attack_detected"]
        max_confidence = result["max_confidence"]

        if attack_detected:
            status = f"🚨 ATTACK (Conf: {max_confidence:.2f})"
        elif security_score < 0.5:
            status = "⚠️  LOW SECURITY"
        else:
            status = "✅ SECURE"

        print(f"   Session {i+1}: Score={security_score:.3f}, {status}")

    print()

    # Step 10: Generate comprehensive security report
    print("10. Generating comprehensive security report...")

    security_report = security_system.generate_security_report()
    print("   SECURITY REPORT:")
    print("   " + "=" * 50)
    print(security_report)
    print()

    # Step 11: Visualize security analysis
    print("11. Creating security analysis visualizations...")
    plt.rcParams["figure.figsize"] = (18, 10)
    security_system.plot_security_analysis(
        security_results,
        save_path="../plots/security_monitoring/demo_security_analysis.png",
    )

    # Step 12: Advanced threat correlation analysis
    print("12. Performing advanced threat correlation analysis...")

    # Analyze correlations between threats and system parameters
    qber_values = [session["qber"] for session in test_sessions]
    security_scores = [result["overall_security_score"] for result in security_results]
    confidence_values = [result["max_confidence"] for result in security_results]

    # Calculate correlations
    qber_security_corr = np.corrcoef(qber_values, security_scores)[0, 1]
    qber_confidence_corr = np.corrcoef(qber_values, confidence_values)[0, 1]

    print("   Threat Correlation Analysis:")
    print(f"     QBER vs Security Score: {qber_security_corr:6.3f}")
    print(f"     QBER vs Detection Confidence: {qber_confidence_corr:6.3f}")
    print()

    # Step 13: Security system performance evaluation
    print("13. Evaluating security system performance...")

    # Create ground truth based on threat labels
    ground_truth = np.array([1 if label != "secure" else 0 for label in threat_labels])
    predictions = np.array(
        [1 if result["attack_detected"] else 0 for result in security_results]
    )

    # Calculate performance metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)

    print("   Security Detection Performance:")
    print(f"     Accuracy: {accuracy:.3f}")
    print(f"     Precision: {precision:.3f}")
    print(f"     Recall: {recall:.3f}")
    print(f"     F1-Score: {f1:.3f}")
    print()

    # Step 14: Security recommendations
    print("14. Generating security recommendations...")

    # Analyze system vulnerabilities
    vulnerabilities = []

    if avg_secret_key_rate < 0.5:
        vulnerabilities.append("Low secret key generation rate")

    if avg_security_margin < 0.05:
        vulnerabilities.append("Insufficient security margin")

    if (
        detection_by_threat.get("intercept_resend", {}).get("detected", 0)
        < detection_by_threat.get("intercept_resend", {}).get("total", 1) * 0.8
    ):
        vulnerabilities.append("Intercept-resend detection needs improvement")

    if avg_channel_fidelity < 0.9:
        vulnerabilities.append("Channel fidelity degradation detected")

    print("   Security Assessment:")
    if vulnerabilities:
        print("   ⚠️  VULNERABILITIES IDENTIFIED:")
        for i, vuln in enumerate(vulnerabilities, 1):
            print(f"     {i}. {vuln}")
        print()
        print("   RECOMMENDATIONS:")
        print("     • Increase monitoring frequency")
        print("     • Implement additional security protocols")
        print("     • Regular system calibration")
        print("     • Enhanced threat detection algorithms")
    else:
        print("   ✅ SECURITY SYSTEM OPERATING OPTIMALLY")
        print("     • Continue current monitoring protocols")
        print("     • Maintain regular security assessments")

    print()

    print("=" * 60)
    print("SECURITY MONITORING DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Key Achievements:")
    print("• Multi-attack detection (intercept-resend, beam-splitting, PNS)")
    print("• Real-time security monitoring and alerting")
    print("• Comprehensive security metrics calculation")
    print("• Information-theoretic security analysis")
    print("• Attack signature pattern recognition")
    print("• Security system performance evaluation")
    print("• Automated threat correlation analysis")
    print()
    print("Generated artifacts:")
    print("• Security analysis plots: ../plots/security_monitoring/")
    print("• Security reports and recommendations")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Security monitoring demo failed: {e}")
        raise
