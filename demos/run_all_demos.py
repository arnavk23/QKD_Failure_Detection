"""
Master Demo Runner for QKD Failure Detection System

This script runs all demonstration modules in sequence to showcase the complete
QKD failure detection capabilities.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import sys
import os
import subprocess
import time
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories for plots and outputs"""
    directories = [
        "../plots/anomaly_detection",
        "../plots/ml_performance",
        "../plots/signal_analysis",
        "../plots/security_monitoring",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def run_demo(demo_script: str, demo_name: str) -> Dict[str, any]:
    """Run a single demonstration script"""
    print(f"\n{'='*80}")
    print(f"RUNNING DEMO: {demo_name}")
    print(f"Script: {demo_script}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Run the demo script
        result = subprocess.run(
            [sys.executable, demo_script],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print(f"\n✅ {demo_name} completed successfully in {duration:.1f} seconds")
            status = "SUCCESS"
        else:
            print(f"\n❌ {demo_name} failed with return code {result.returncode}")
            print("STDERR:", result.stderr)
            status = "FAILED"

        return {
            "name": demo_name,
            "script": demo_script,
            "status": status,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        print(f"\n⏰ {demo_name} timed out after 5 minutes")
        return {
            "name": demo_name,
            "script": demo_script,
            "status": "TIMEOUT",
            "duration": 300,
            "stdout": "",
            "stderr": "Demo timed out",
        }

    except Exception as e:
        print(f"\n💥 {demo_name} encountered an error: {e}")
        return {
            "name": demo_name,
            "script": demo_script,
            "status": "ERROR",
            "duration": 0,
            "stdout": "",
            "stderr": str(e),
        }


def main():
    """Main demonstration runner"""
    print("QKD SYSTEM FAILURE AUTO DETECTION - COMPLETE DEMONSTRATION SUITE")
    print("=" * 80)
    print("Under the guidance of Vijayalaxmi Mogiligidda")
    print("=" * 80)
    print()

    # Create necessary directories
    print("Setting up demonstration environment...")
    create_directories()
    print()

    # Define demonstration sequence
    demos = [
        {
            "script": "demo_anomaly_detection.py",
            "name": "Statistical & ML Anomaly Detection",
            "description": "Demonstrates statistical process control and machine learning based anomaly detection",
        },
        {
            "script": "demo_ml_detection.py",
            "name": "Advanced Machine Learning Detection",
            "description": "Shows multi-class failure classification and unsupervised anomaly detection",
        },
        {
            "script": "demo_signal_analysis.py",
            "name": "Signal Processing & Analysis",
            "description": "Demonstrates time-frequency analysis and signal quality assessment",
        },
        {
            "script": "demo_security_monitor.py",
            "name": "Security Monitoring & Eavesdropping Detection",
            "description": "Shows comprehensive security monitoring and attack detection",
        },
    ]

    # Display demonstration plan
    print("DEMONSTRATION PLAN:")
    print("-" * 40)
    for i, demo in enumerate(demos, 1):
        print(f"{i}. {demo['name']}")
        print(f"   {demo['description']}")
        print()

    # Run demonstrations
    results = []
    total_start_time = time.time()

    for i, demo in enumerate(demos, 1):
        print(f"\nStarting demonstration {i}/{len(demos)}...")
        result = run_demo(demo["script"], demo["name"])
        result["description"] = demo["description"]
        results.append(result)

        # Brief pause between demos
        if i < len(demos):
            print("\nPausing for 3 seconds before next demonstration...")
            time.sleep(3)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Generate summary report
    print(f"\n{'='*80}")
    print("DEMONSTRATION SUITE SUMMARY")
    print(f"{'='*80}")
    print(
        f"Total execution time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)"
    )
    print()

    # Status summary
    successful = sum(1 for r in results if r["status"] == "SUCCESS")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    timeouts = sum(1 for r in results if r["status"] == "TIMEOUT")

    print("EXECUTION SUMMARY:")
    print(f"  ✅ Successful: {successful}/{len(results)}")
    print(f"  ❌ Failed: {failed}/{len(results)}")
    print(f"  💥 Errors: {errors}/{len(results)}")
    print(f"  ⏰ Timeouts: {timeouts}/{len(results)}")
    print()

    # Detailed results
    print("DETAILED RESULTS:")
    print("-" * 60)
    for result in results:
        status_emoji = {
            "SUCCESS": "✅",
            "FAILED": "❌",
            "ERROR": "💥",
            "TIMEOUT": "⏰",
        }.get(result["status"], "❓")

        print(f"{status_emoji} {result['name']}")
        print(f"   Duration: {result['duration']:.1f}s")
        print(f"   Status: {result['status']}")

        if result["status"] != "SUCCESS" and result["stderr"]:
            print(f"   Error: {result['stderr'][:100]}...")
        print()

    # Generated artifacts summary
    print("GENERATED ARTIFACTS:")
    print("-" * 30)

    artifact_dirs = [
        "../plots/anomaly_detection",
        "../plots/ml_performance",
        "../plots/signal_analysis",
        "../plots/security_monitoring",
        "../resources",
    ]

    total_files = 0
    for directory in artifact_dirs:
        if os.path.exists(directory):
            files = [
                f
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
            ]
            if files:
                print(f"{directory}: {len(files)} files")
                total_files += len(files)

    print(f"Total artifacts generated: {total_files}")
    print()

    # System capabilities demonstrated
    print("CAPABILITIES DEMONSTRATED:")
    print("-" * 40)
    capabilities = [
        "BB84 quantum key distribution simulation",
        "Statistical process control for anomaly detection",
        "Machine learning failure classification",
        "Real-time anomaly detection with multiple algorithms",
        "Time-frequency signal analysis",
        "Signal quality assessment and monitoring",
        "Eavesdropping attack detection (intercept-resend, beam-splitting, PNS)",
        "Information-theoretic security analysis",
        "Comprehensive performance evaluation",
        "Real-time security monitoring and alerting",
    ]

    for i, capability in enumerate(capabilities, 1):
        print(f"{i:2d}. {capability}")

    print()

    # Final assessment
    if successful == len(results):
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("   The QKD Failure Detection System is fully operational.")
    elif successful > len(results) / 2:
        print("⚠️  MOST DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print(
            f"   {successful}/{len(results)} demos passed. Check failed demos for issues."
        )
    else:
        print("🚨 MULTIPLE DEMONSTRATION FAILURES")
        print("   Please check system dependencies and configuration.")

    print()
    print("=" * 80)
    print("DEMONSTRATION SUITE COMPLETED")
    print("=" * 80)
    print()
    print("Next steps:")
    print("• Review generated plots and analysis results")
    print("• Check individual demo outputs for detailed insights")
    print("• Adapt algorithms for specific QKD system requirements")
    print("• Integrate with production QKD systems")
    print()
    print("For questions or support, contact the development team.")
    print("Project developed under the guidance of Vijayalaxmi Mogiligidda")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Demonstration suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Demonstration suite failed: {e}")
        print(f"\n💥 Fatal error in demonstration suite: {e}")
        sys.exit(1)
