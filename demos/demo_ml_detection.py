"""
Demo: Machine Learning Detection for QKD Systems

This demonstration shows advanced machine learning algorithms for failure
detection and classification in quantum key distribution systems.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from qkd_simulator import QKDSystemSimulator, QKDParameters
from ml_detector import MLDetectionSystem
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function"""
    print("=" * 60)
    print("QKD SYSTEM ML DETECTION DEMONSTRATION")
    print("=" * 60)
    print()

    # Step 1: Generate comprehensive training dataset
    print("1. Generating comprehensive training dataset...")
    params = QKDParameters(
        key_length=1500,
        error_rate=0.015,
        detection_efficiency=0.85,
        channel_loss=0.04,
        protocol="BB84",
    )

    simulator = QKDSystemSimulator(params)

    # Generate diverse training scenarios
    training_data = []

    # Normal operation (200 sessions)
    normal_sessions = simulator.simulate_multiple_sessions(200)
    training_data.extend(normal_sessions)

    # Various failure types for training
    failure_types = [
        ("channel_loss", 0.03, 20),
        ("detector_noise", 0.02, 20),
        ("timing_drift", 0.025, 20),
        ("source_instability", 0.015, 20),
    ]

    for failure_type, intensity, count in failure_types:
        # Reset parameters
        simulator.params = QKDParameters(
            key_length=1500,
            error_rate=0.015,
            detection_efficiency=0.85,
            channel_loss=0.04,
            protocol="BB84",
        )
        simulator.inject_failure(failure_type, intensity)
        failure_sessions = simulator.simulate_multiple_sessions(count)
        training_data.extend(failure_sessions)

    print(f"   Generated {len(training_data)} training sessions")
    print()

    # Step 2: Initialize and train ML detection system
    print("2. Initializing and training ML detection system...")
    ml_system = MLDetectionSystem({"test_size": 0.2, "cv_folds": 5, "random_state": 42})

    # Train models
    ml_system.train_models(training_data)
    print("   ML models trained successfully")
    print()

    # Step 3: Generate test dataset with known failures
    print("3. Generating test dataset with known failure patterns...")

    # Reset for test data generation
    test_simulator = QKDSystemSimulator(
        QKDParameters(
            key_length=1000,
            error_rate=0.02,
            detection_efficiency=0.8,
            channel_loss=0.05,
            protocol="BB84",
        )
    )

    test_sessions = []
    test_labels = []

    # Normal operation
    normal_test = test_simulator.simulate_multiple_sessions(30)
    test_sessions.extend(normal_test)
    test_labels.extend(["normal"] * 30)

    # Security breach (eavesdropping)
    test_simulator.inject_failure("eavesdropping", 0.06)
    security_sessions = test_simulator.simulate_multiple_sessions(15)
    test_sessions.extend(security_sessions)
    test_labels.extend(["security_breach"] * 15)

    # Channel failure
    test_simulator.inject_failure("channel_loss", 0.15)
    channel_sessions = test_simulator.simulate_multiple_sessions(10)
    test_sessions.extend(channel_sessions)
    test_labels.extend(["channel_failure"] * 10)

    # Low efficiency
    test_simulator.inject_failure("detector_noise", 0.08)
    efficiency_sessions = test_simulator.simulate_multiple_sessions(10)
    test_sessions.extend(efficiency_sessions)
    test_labels.extend(["low_efficiency"] * 10)

    print(f"   Generated {len(test_sessions)} test sessions")
    print("   Test distribution:")
    for label in set(test_labels):
        count = test_labels.count(label)
        print(f"     {label}: {count} sessions")
    print()

    # Step 4: Perform ML-based failure detection
    print("4. Performing ML-based failure detection...")
    results = ml_system.detect_failures(test_sessions)

    # Extract predictions
    if "random_forest" in results["classifications"]:
        rf_predictions = results["classifications"]["random_forest"]
        print("   Random Forest Classifications:")
        for label in set(rf_predictions):
            count = list(rf_predictions).count(label)
            print(f"     {label}: {count} detections")

    # Anomaly detection results
    anomaly_results = results["anomalies"]
    combined_anomalies = anomaly_results["combined"]
    print(
        f"   Combined anomaly detection: {np.sum(combined_anomalies)} anomalies detected"
    )
    print()

    # Step 5: Evaluate performance
    print("5. Evaluating ML detection performance...")

    # Convert test labels to numpy array
    true_labels = np.array(test_labels)

    # Evaluate classification performance
    if "random_forest" in results["classifications"]:
        from sklearn.metrics import classification_report, accuracy_score

        rf_pred = results["classifications"]["random_forest"]
        accuracy = accuracy_score(true_labels, rf_pred)

        print(f"   Random Forest Accuracy: {accuracy:.3f}")
        print("   Detailed Classification Report:")
        print(classification_report(true_labels, rf_pred, target_names=None))

    # Evaluate anomaly detection (binary: normal vs any failure)
    binary_true = (true_labels != "normal").astype(int)
    binary_pred = combined_anomalies.astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(binary_true, binary_pred)
    recall = recall_score(binary_true, binary_pred)
    f1 = f1_score(binary_true, binary_pred)

    print(f"   Anomaly Detection Performance:")
    print(f"     Precision: {precision:.3f}")
    print(f"     Recall: {recall:.3f}")
    print(f"     F1-Score: {f1:.3f}")
    print()

    # Step 6: Feature importance analysis
    print("6. Analyzing feature importance...")

    if hasattr(ml_system.failure_classifier, "feature_importance"):
        importance = ml_system.failure_classifier.feature_importance

        if "random_forest" in importance:
            rf_importance = importance["random_forest"]

            # Top 10 most important features
            top_features = sorted(
                rf_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]

            print("   Top 10 Most Important Features (Random Forest):")
            for i, (feature, imp) in enumerate(top_features, 1):
                print(f"     {i:2d}. {feature:<30} {imp:.4f}")
    print()

    # Step 7: Visualize ML analysis results
    print("7. Creating ML analysis visualizations...")
    plt.rcParams["figure.figsize"] = (15, 12)
    ml_system.plot_ml_analysis(
        results, save_path="../plots/ml_performance/demo_ml_analysis.png"
    )

    # Step 8: Advanced pattern analysis
    print("8. Performing advanced pattern analysis...")

    # Analyze feature distributions for different failure types
    features_df = results["features"]

    print("   Feature Statistics by Session Type:")
    print("   " + "-" * 50)

    # Calculate statistics for key features
    key_features = ["qber", "sift_ratio", "key_efficiency"]

    for i, (session, label) in enumerate(zip(test_sessions, test_labels)):
        if i < len(features_df):
            for feature in key_features:
                if feature in features_df.columns:
                    value = features_df.iloc[i][feature]
                    if i < 5:  # Print first few examples
                        print(f"     Session {i}: {label:<15} {feature}: {value:.4f}")

    print()

    # Step 9: Real-time classification simulation
    print("9. Simulating real-time ML classification...")

    # Process streaming sessions
    streaming_sessions = test_simulator.simulate_multiple_sessions(10)

    print("   Real-time Classification Results:")
    for i, session in enumerate(streaming_sessions):
        # Process single session
        single_result = ml_system.detect_failures([session])

        if "random_forest" in single_result["classifications"]:
            classification = single_result["classifications"]["random_forest"][0]
            anomaly_detected = single_result["anomalies"]["combined"][0]

            status = "⚠️  ALERT" if anomaly_detected else "✅ OK"
            print(f"   Session {i+1}: {classification:<15} {status}")

    print()

    # Step 10: Model persistence demonstration
    print("10. Demonstrating model persistence...")

    # Save trained models
    model_path = "../resources/qkd_ml_models.joblib"
    ml_system.save_models(model_path)
    print(f"    Models saved to: {model_path}")

    # Create new instance and load models
    new_ml_system = MLDetectionSystem()
    new_ml_system.load_models(model_path)
    print("    Models loaded successfully in new instance")

    # Verify loaded models work
    verify_result = new_ml_system.detect_failures(test_sessions[:5])
    print(
        f"    Verification: Processed {len(verify_result['features'])} sessions with loaded models"
    )
    print()

    print("=" * 60)
    print("ML DETECTION DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Key Achievements:")
    print("• Multi-class failure classification with Random Forest")
    print("• Neural network based pattern recognition")
    print("• Unsupervised anomaly detection with multiple algorithms")
    print("• Advanced feature engineering for QKD parameters")
    print("• Real-time classification capabilities")
    print("• Model persistence and deployment")
    print("• Comprehensive performance evaluation")
    print()
    print("Generated artifacts:")
    print("• ML analysis plots: ../plots/ml_performance/")
    print("• Trained models: ../resources/qkd_ml_models.joblib")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"ML demo failed: {e}")
        raise
