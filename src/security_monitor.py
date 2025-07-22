"""
Security Monitoring Module for QKD Systems

Real-time security monitoring and eavesdropping detection for quantum key distribution systems.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings
import logging
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def h_binary(p: float) -> float:
    """Binary entropy function H(p) = -p*log2(p) - (1-p)*log2(1-p)"""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


class EavesdroppingDetector:
    """Eavesdropping detection using statistical analysis"""
    
    def __init__(self, security_threshold: float = 0.11):
        self.security_threshold = security_threshold  # Standard QKD security threshold
        self.baseline_stats = {}
        self.attack_signatures = {}
        
    def establish_security_baseline(self, normal_sessions: List[Dict]):
        """Establish baseline security parameters"""
        qber_values = [session['qber'] for session in normal_sessions]
        sift_ratios = [session['sift_ratio'] for session in normal_sessions]
        key_lengths = [session['final_key_length'] for session in normal_sessions]
        
        self.baseline_stats = {
            'qber': {
                'mean': np.mean(qber_values),
                'std': np.std(qber_values),
                'percentile_95': np.percentile(qber_values, 95),
                'max_normal': np.max(qber_values)
            },
            'sift_ratio': {
                'mean': np.mean(sift_ratios),
                'std': np.std(sift_ratios),
                'min_normal': np.min(sift_ratios)
            },
            'key_length': {
                'mean': np.mean(key_lengths),
                'std': np.std(key_lengths),
                'min_normal': np.min(key_lengths)
            }
        }
        
        logger.info("Security baseline established")
    
    def detect_intercept_resend_attack(self, session: Dict) -> Dict[str, Any]:
        """Detect intercept-resend attack patterns"""
        qber = session['qber']
        sift_ratio = session['sift_ratio']
        
        detection_result = {
            'attack_detected': False,
            'confidence': 0.0,
            'indicators': []
        }
        
        # Check QBER threshold
        if qber > self.security_threshold:
            detection_result['attack_detected'] = True
            detection_result['indicators'].append('QBER_threshold_exceeded')
            detection_result['confidence'] += 0.4
        
        # Check for sudden QBER increase
        if hasattr(self, 'baseline_stats') and 'qber' in self.baseline_stats:
            baseline_qber = self.baseline_stats['qber']['mean']
            if qber > baseline_qber + 3 * self.baseline_stats['qber']['std']:
                detection_result['attack_detected'] = True
                detection_result['indicators'].append('sudden_QBER_increase')
                detection_result['confidence'] += 0.3
        
        # Check sift ratio drop (indicator of measurement in wrong basis)
        if hasattr(self, 'baseline_stats') and 'sift_ratio' in self.baseline_stats:
            baseline_sift = self.baseline_stats['sift_ratio']['mean']
            if sift_ratio < baseline_sift - 2 * self.baseline_stats['sift_ratio']['std']:
                detection_result['indicators'].append('low_sift_ratio')
                detection_result['confidence'] += 0.2
        
        # Pattern-based detection
        error_pattern = self._analyze_error_pattern(session)
        if error_pattern['suspicious']:
            detection_result['indicators'].append('suspicious_error_pattern')
            detection_result['confidence'] += 0.3
        
        detection_result['confidence'] = min(detection_result['confidence'], 1.0)
        
        return detection_result
    
    def detect_beam_splitting_attack(self, session: Dict) -> Dict[str, Any]:
        """Detect beam splitting attack patterns"""
        detection_result = {
            'attack_detected': False,
            'confidence': 0.0,
            'indicators': []
        }
        
        channel_loss = session.get('channel_loss', 0)
        final_key_length = session.get('final_key_length', 0)
        initial_length = session.get('initial_length', 1)
        
        # Calculate effective transmission efficiency
        transmission_efficiency = final_key_length / initial_length
        
        # Check for unusual channel loss patterns
        if channel_loss > 0.3:  # Suspicious high loss
            detection_result['indicators'].append('high_channel_loss')
            detection_result['confidence'] += 0.2
        
        # Check for low transmission efficiency
        if transmission_efficiency < 0.1:
            detection_result['indicators'].append('low_transmission_efficiency')
            detection_result['confidence'] += 0.3
        
        # Check for correlation between loss and QBER
        qber = session['qber']
        if channel_loss > 0.2 and qber > 0.05:
            detection_result['attack_detected'] = True
            detection_result['indicators'].append('correlated_loss_and_errors')
            detection_result['confidence'] += 0.4
        
        detection_result['confidence'] = min(detection_result['confidence'], 1.0)
        
        return detection_result
    
    def detect_photon_number_splitting_attack(self, session: Dict) -> Dict[str, Any]:
        """Detect photon number splitting attack"""
        detection_result = {
            'attack_detected': False,
            'confidence': 0.0,
            'indicators': []
        }
        
        # Analyze pulse intensity statistics (simulated)
        pulse_intensity_var = session.get('pulse_intensity_variance', 0.1)
        detection_efficiency = session.get('detection_efficiency', 0.8)
        
        # PNS attacks often show specific pulse statistics
        if pulse_intensity_var > 0.2:  # High variance in pulse intensity
            detection_result['indicators'].append('high_pulse_variance')
            detection_result['confidence'] += 0.2
        
        # Check detection efficiency patterns
        if detection_efficiency < 0.6:
            detection_result['indicators'].append('low_detection_efficiency')
            detection_result['confidence'] += 0.2
        
        # Multi-photon vulnerability analysis
        multi_photon_rate = self._estimate_multi_photon_rate(session)
        if multi_photon_rate > 0.1:
            detection_result['attack_detected'] = True
            detection_result['indicators'].append('high_multi_photon_rate')
            detection_result['confidence'] += 0.5
        
        return detection_result
    
    def _analyze_error_pattern(self, session: Dict) -> Dict[str, Any]:
        """Analyze error patterns for suspicious activity"""
        qber = session['qber']
        error_rate = session.get('error_rate', qber)
        
        # Check for unnatural error correlations
        error_correlation = abs(qber - error_rate) / (error_rate + 1e-8)
        
        pattern_analysis = {
            'suspicious': False,
            'correlation_factor': error_correlation
        }
        
        # Suspicious if error patterns are too regular or irregular
        if error_correlation > 2.0 or error_correlation < 0.1:
            pattern_analysis['suspicious'] = True
        
        return pattern_analysis
    
    def _estimate_multi_photon_rate(self, session: Dict) -> float:
        """Estimate multi-photon emission rate"""
        # Simplified model based on Poisson statistics
        mean_photon_number = session.get('mean_photon_number', 0.1)
        
        # Probability of multi-photon states
        multi_photon_rate = 1 - np.exp(-mean_photon_number) * (1 + mean_photon_number)
        
        return multi_photon_rate


class SecurityMetricsCalculator:
    """Calculate comprehensive security metrics"""
    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_information_security(self, session: Dict) -> Dict[str, float]:
        """Calculate information-theoretic security metrics"""
        qber = session['qber']
        sift_ratio = session['sift_ratio']
        final_key_length = session['final_key_length']
        
        metrics = {}
        
        # Define h_binary function for binary entropy
        h_binary = lambda p: -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10)
        
        # Mutual information between Alice and Eve (upper bound)
        if qber > 0:
            metrics['mutual_info_eve'] = h_binary(qber)
        else:
            metrics['mutual_info_eve'] = 0
        
        # Mutual information between Alice and Bob
        metrics['mutual_info_bob'] = 1 - h_binary(qber)  # Simplified
        
        # Secret key rate (simplified GLLP formula)
        if qber < 0.11:
            metrics['secret_key_rate'] = max(0, 1 - 2 * h_binary(qber))
        else:
            metrics['secret_key_rate'] = 0
        
        # Security parameter
        if final_key_length > 0:
            metrics['security_parameter'] = -np.log2(2**(-final_key_length) + 1e-10)
        else:
            metrics['security_parameter'] = 0
        
        # Privacy amplification efficiency
        initial_sifted = session.get('sifted_length', final_key_length)
        if initial_sifted > 0:
            metrics['privacy_amplification_ratio'] = final_key_length / initial_sifted
        else:
            metrics['privacy_amplification_ratio'] = 0
        
        return metrics
    
    def calculate_channel_security(self, session: Dict) -> Dict[str, float]:
        """Calculate channel-specific security metrics"""
        channel_loss = session.get('channel_loss', 0)
        detection_efficiency = session.get('detection_efficiency', 0.8)
        dark_count_rate = session.get('dark_count_rate', 1e-6)
        
        metrics = {}
        
        # Channel transmission security
        metrics['effective_transmission'] = (1 - channel_loss) * detection_efficiency
        
        # Background noise ratio
        signal_rate = session.get('signal_rate', 1e6)  # Simplified
        metrics['background_noise_ratio'] = dark_count_rate / (signal_rate + 1e-10)
        
        # Channel fidelity (simplified)
        qber = session['qber']
        metrics['channel_fidelity'] = 1 - 2 * qber
        
        # Security margin
        security_threshold = 0.11
        metrics['security_margin'] = max(0, security_threshold - qber)
        
        return metrics
    
    def calculate_protocol_security(self, session: Dict) -> Dict[str, float]:
        """Calculate protocol-specific security metrics"""
        protocol = session.get('protocol', 'BB84')
        
        metrics = {}
        
        if protocol == 'BB84':
            # BB84 specific metrics
            sift_ratio = session['sift_ratio']
            
            # Theoretical maximum sift ratio for BB84 is 0.5
            metrics['sift_efficiency'] = sift_ratio / 0.5
            
            # Basis reconciliation efficiency
            metrics['basis_reconciliation_efficiency'] = min(1.0, sift_ratio * 2)
            
            # Protocol overhead
            initial_length = session.get('initial_length', 1)
            final_length = session.get('final_key_length', 0)
            metrics['protocol_overhead'] = 1 - (final_length / initial_length)
        
        # Error correction efficiency
        qber = session['qber']
        shannon_limit = 1 + qber * np.log2(qber) + (1-qber) * np.log2(1-qber)
        metrics['error_correction_efficiency'] = shannon_limit if qber > 0 else 1.0
        
        return metrics


class RealTimeSecurityMonitor:
    """Real-time security monitoring system"""
    
    def __init__(self, alert_threshold: float = 0.7):
        self.alert_threshold = alert_threshold
        self.monitoring_active = False
        self.alert_history = []
        self.security_log = []
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring_active = True
        logger.info("Real-time security monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        logger.info("Real-time security monitoring stopped")
    
    def process_session(self, session: Dict, eavesdropper: EavesdroppingDetector, 
                       metrics_calc: SecurityMetricsCalculator) -> Dict[str, Any]:
        """Process a single session for security analysis"""
        if not self.monitoring_active:
            return {}
        
        timestamp = datetime.now()
        
        # Detect attacks
        intercept_result = eavesdropper.detect_intercept_resend_attack(session)
        beam_split_result = eavesdropper.detect_beam_splitting_attack(session)
        pns_result = eavesdropper.detect_photon_number_splitting_attack(session)
        
        # Calculate security metrics
        info_security = metrics_calc.calculate_information_security(session)
        channel_security = metrics_calc.calculate_channel_security(session)
        protocol_security = metrics_calc.calculate_protocol_security(session)
        
        # Combine attack detection results
        max_confidence = max(
            intercept_result['confidence'],
            beam_split_result['confidence'],
            pns_result['confidence']
        )
        
        attack_detected = (
            intercept_result['attack_detected'] or
            beam_split_result['attack_detected'] or
            pns_result['attack_detected']
        )
        
        # Generate security assessment
        security_assessment = {
            'timestamp': timestamp,
            'session_id': session.get('session_id', 0),
            'attack_detected': attack_detected,
            'max_confidence': max_confidence,
            'attack_types': {
                'intercept_resend': intercept_result,
                'beam_splitting': beam_split_result,
                'photon_number_splitting': pns_result
            },
            'security_metrics': {
                'information': info_security,
                'channel': channel_security,
                'protocol': protocol_security
            },
            'overall_security_score': self._calculate_overall_security_score(
                info_security, channel_security, protocol_security
            )
        }
        
        # Check for alerts
        if max_confidence > self.alert_threshold or attack_detected:
            alert = self._generate_security_alert(security_assessment)
            self.alert_history.append(alert)
            logger.warning(f"SECURITY ALERT: {alert['message']}")
        
        # Log security event
        self.security_log.append(security_assessment)
        
        return security_assessment
    
    def _calculate_overall_security_score(self, info_metrics: Dict, 
                                        channel_metrics: Dict, 
                                        protocol_metrics: Dict) -> float:
        """Calculate overall security score (0-1, higher is better)"""
        # Weight different security aspects
        weights = {
            'secret_key_rate': 0.3,
            'security_margin': 0.2,
            'channel_fidelity': 0.2,
            'effective_transmission': 0.15,
            'sift_efficiency': 0.15
        }
        
        score = 0.0
        
        # Information security contribution
        score += weights['secret_key_rate'] * info_metrics.get('secret_key_rate', 0)
        score += weights['security_margin'] * channel_metrics.get('security_margin', 0) * 10  # Scale margin
        
        # Channel security contribution
        score += weights['channel_fidelity'] * max(0, channel_metrics.get('channel_fidelity', 0))
        score += weights['effective_transmission'] * channel_metrics.get('effective_transmission', 0)
        
        # Protocol security contribution
        score += weights['sift_efficiency'] * protocol_metrics.get('sift_efficiency', 0)
        
        return min(1.0, score)
    
    def _generate_security_alert(self, assessment: Dict) -> Dict[str, Any]:
        """Generate security alert"""
        alert = {
            'timestamp': assessment['timestamp'],
            'session_id': assessment['session_id'],
            'severity': 'HIGH' if assessment['max_confidence'] > 0.8 else 'MEDIUM',
            'confidence': assessment['max_confidence'],
            'message': self._format_alert_message(assessment),
            'recommended_actions': self._get_recommended_actions(assessment)
        }
        
        return alert
    
    def _format_alert_message(self, assessment: Dict) -> str:
        """Format security alert message"""
        messages = []
        
        for attack_type, result in assessment['attack_types'].items():
            if result['attack_detected']:
                messages.append(f"{attack_type.replace('_', ' ').title()} attack detected")
        
        if not messages:
            messages.append("Security threshold exceeded")
        
        confidence = assessment['max_confidence']
        return f"{'; '.join(messages)} (Confidence: {confidence:.2f})"
    
    def _get_recommended_actions(self, assessment: Dict) -> List[str]:
        """Get recommended security actions"""
        actions = []
        
        # Check specific attack types
        for attack_type, result in assessment['attack_types'].items():
            if result['attack_detected']:
                if attack_type == 'intercept_resend':
                    actions.extend([
                        "Immediately halt key generation",
                        "Verify channel integrity",
                        "Check for unauthorized access points"
                    ])
                elif attack_type == 'beam_splitting':
                    actions.extend([
                        "Inspect optical connections",
                        "Verify detector dark count rates",
                        "Check for optical taps or splitters"
                    ])
                elif attack_type == 'photon_number_splitting':
                    actions.extend([
                        "Reduce mean photon number",
                        "Implement decoy state protocols",
                        "Verify source stability"
                    ])
        
        # General security recommendations
        if assessment['overall_security_score'] < 0.5:
            actions.extend([
                "Review system calibration",
                "Increase error correction redundancy",
                "Consider alternative protocols"
            ])
        
        return list(set(actions))  # Remove duplicates
    
    def get_security_report(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        if time_window:
            cutoff_time = datetime.now() - time_window
            recent_logs = [log for log in self.security_log if log['timestamp'] > cutoff_time]
            recent_alerts = [alert for alert in self.alert_history if alert['timestamp'] > cutoff_time]
        else:
            recent_logs = self.security_log
            recent_alerts = self.alert_history
        
        if not recent_logs:
            return {'message': 'No security data available'}
        
        # Calculate statistics
        security_scores = [log['overall_security_score'] for log in recent_logs]
        attack_detections = [log['attack_detected'] for log in recent_logs]
        
        report = {
            'reporting_period': {
                'start': recent_logs[0]['timestamp'] if recent_logs else None,
                'end': recent_logs[-1]['timestamp'] if recent_logs else None,
                'total_sessions': len(recent_logs)
            },
            'security_summary': {
                'mean_security_score': np.mean(security_scores),
                'min_security_score': np.min(security_scores),
                'security_score_std': np.std(security_scores),
                'attack_detection_rate': np.mean(attack_detections),
                'total_attacks_detected': np.sum(attack_detections)
            },
            'alert_summary': {
                'total_alerts': len(recent_alerts),
                'high_severity_alerts': len([a for a in recent_alerts if a['severity'] == 'HIGH']),
                'medium_severity_alerts': len([a for a in recent_alerts if a['severity'] == 'MEDIUM'])
            },
            'recommendations': self._generate_system_recommendations(recent_logs)
        }
        
        return report
    
    def _generate_system_recommendations(self, logs: List[Dict]) -> List[str]:
        """Generate system-wide security recommendations"""
        recommendations = []
        
        if not logs:
            return recommendations
        
        # Analyze trends
        security_scores = [log['overall_security_score'] for log in logs]
        attack_rates = [log['attack_detected'] for log in logs]
        
        # Check for declining security
        if len(security_scores) > 10:
            recent_avg = np.mean(security_scores[-10:])
            overall_avg = np.mean(security_scores)
            
            if recent_avg < overall_avg * 0.9:
                recommendations.append("Security performance is declining - investigate system degradation")
        
        # Check attack frequency
        if np.mean(attack_rates) > 0.1:
            recommendations.append("High attack detection rate - enhance security monitoring")
        
        # Check specific metrics across sessions
        channel_fidelities = [log['security_metrics']['channel']['channel_fidelity'] 
                            for log in logs if 'channel_fidelity' in log['security_metrics']['channel']]
        
        if channel_fidelities and np.mean(channel_fidelities) < 0.8:
            recommendations.append("Low channel fidelity detected - check optical components")
        
        if not recommendations:
            recommendations.append("System security appears nominal - continue regular monitoring")
        
        return recommendations


class QKDSecuritySystem:
    """Complete QKD security monitoring system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.eavesdropper = EavesdroppingDetector(self.config['security_threshold'])
        self.metrics_calculator = SecurityMetricsCalculator()
        self.monitor = RealTimeSecurityMonitor(self.config['alert_threshold'])
        
    def _default_config(self) -> Dict:
        """Default security configuration"""
        return {
            'security_threshold': 0.11,
            'alert_threshold': 0.7,
            'monitoring_enabled': True
        }
    
    def initialize_system(self, baseline_sessions: List[Dict]):
        """Initialize security system with baseline data"""
        self.eavesdropper.establish_security_baseline(baseline_sessions)
        
        if self.config['monitoring_enabled']:
            self.monitor.start_monitoring()
        
        logger.info("QKD security system initialized")
    
    def monitor_session(self, session: Dict) -> Dict[str, Any]:
        """Monitor a single QKD session"""
        return self.monitor.process_session(session, self.eavesdropper, self.metrics_calculator)
    
    def batch_security_analysis(self, sessions: List[Dict]) -> List[Dict[str, Any]]:
        """Perform security analysis on multiple sessions"""
        results = []
        
        for session in sessions:
            result = self.monitor_session(session)
            results.append(result)
        
        return results
    
    def plot_security_analysis(self, analysis_results: List[Dict], save_path: str = None):
        """Plot comprehensive security analysis"""
        if not analysis_results:
            logger.warning("No analysis results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Security scores over time
        session_ids = [result['session_id'] for result in analysis_results]
        security_scores = [result['overall_security_score'] for result in analysis_results]
        
        axes[0, 0].plot(session_ids, security_scores, 'b-', linewidth=2)
        axes[0, 0].set_title('Overall Security Score')
        axes[0, 0].set_xlabel('Session ID')
        axes[0, 0].set_ylabel('Security Score')
        axes[0, 0].grid(True)
        
        # Attack detection confidence
        confidences = [result['max_confidence'] for result in analysis_results]
        attacks_detected = [result['attack_detected'] for result in analysis_results]
        
        colors = ['red' if attack else 'blue' for attack in attacks_detected]
        axes[0, 1].scatter(session_ids, confidences, c=colors, alpha=0.6)
        axes[0, 1].axhline(y=self.config['alert_threshold'], color='orange', linestyle='--', 
                          label=f'Alert Threshold ({self.config["alert_threshold"]})')
        axes[0, 1].set_title('Attack Detection Confidence')
        axes[0, 1].set_xlabel('Session ID')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].legend()
        
        # Security metrics distribution
        secret_key_rates = [result['security_metrics']['information']['secret_key_rate'] 
                          for result in analysis_results]
        axes[0, 2].hist(secret_key_rates, bins=20, alpha=0.7, color='green')
        axes[0, 2].set_title('Secret Key Rate Distribution')
        axes[0, 2].set_xlabel('Secret Key Rate')
        axes[0, 2].set_ylabel('Frequency')
        
        # Attack types detected
        attack_types = {'intercept_resend': 0, 'beam_splitting': 0, 'photon_number_splitting': 0}
        for result in analysis_results:
            for attack_type, attack_result in result['attack_types'].items():
                if attack_result['attack_detected']:
                    attack_types[attack_type] += 1
        
        if any(attack_types.values()):
            axes[1, 0].bar(attack_types.keys(), attack_types.values())
            axes[1, 0].set_title('Attack Types Detected')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Channel fidelity vs security score
        channel_fidelities = [result['security_metrics']['channel']['channel_fidelity'] 
                            for result in analysis_results]
        axes[1, 1].scatter(channel_fidelities, security_scores, alpha=0.6)
        axes[1, 1].set_title('Channel Fidelity vs Security Score')
        axes[1, 1].set_xlabel('Channel Fidelity')
        axes[1, 1].set_ylabel('Security Score')
        
        # Security margin over time
        security_margins = [result['security_metrics']['channel']['security_margin'] 
                          for result in analysis_results]
        axes[1, 2].plot(session_ids, security_margins, 'g-', linewidth=2)
        axes[1, 2].fill_between(session_ids, security_margins, alpha=0.3)
        axes[1, 2].set_title('Security Margin Over Time')
        axes[1, 2].set_xlabel('Session ID')
        axes[1, 2].set_ylabel('Security Margin')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security report"""
        report = self.monitor.get_security_report()
        
        if 'message' in report:
            return report['message']
        
        report_text = []
        report_text.append("QKD SYSTEM SECURITY REPORT")
        report_text.append("=" * 40)
        report_text.append("")
        
        # Reporting period
        period = report['reporting_period']
        report_text.append(f"Reporting Period: {period['start']} to {period['end']}")
        report_text.append(f"Total Sessions Analyzed: {period['total_sessions']}")
        report_text.append("")
        
        # Security summary
        security = report['security_summary']
        report_text.append("SECURITY SUMMARY:")
        report_text.append("-" * 20)
        report_text.append(f"Mean Security Score: {security['mean_security_score']:.3f}")
        report_text.append(f"Minimum Security Score: {security['min_security_score']:.3f}")
        report_text.append(f"Attack Detection Rate: {security['attack_detection_rate']:.1%}")
        report_text.append(f"Total Attacks Detected: {security['total_attacks_detected']}")
        report_text.append("")
        
        # Alert summary
        alerts = report['alert_summary']
        if alerts['total_alerts'] > 0:
            report_text.append("SECURITY ALERTS:")
            report_text.append("-" * 15)
            report_text.append(f"Total Alerts: {alerts['total_alerts']}")
            report_text.append(f"High Severity: {alerts['high_severity_alerts']}")
            report_text.append(f"Medium Severity: {alerts['medium_severity_alerts']}")
            report_text.append("")
        
        # Recommendations
        if report['recommendations']:
            report_text.append("RECOMMENDATIONS:")
            report_text.append("-" * 15)
            for i, rec in enumerate(report['recommendations'], 1):
                report_text.append(f"{i}. {rec}")
        
        return "\n".join(report_text)


if __name__ == "__main__":
    # Example usage
    from qkd_simulator import QKDSystemSimulator, QKDParameters
    
    # Generate sample data
    params = QKDParameters(key_length=1000, error_rate=0.02)
    simulator = QKDSystemSimulator(params)
    
    # Generate baseline (secure) sessions
    baseline_sessions = simulator.simulate_multiple_sessions(50)
    
    # Generate test sessions with potential attacks
    simulator.inject_failure("eavesdropping", 0.05)
    test_sessions = simulator.simulate_multiple_sessions(30)
    
    # Initialize security system
    security_system = QKDSecuritySystem()
    security_system.initialize_system(baseline_sessions)
    
    # Monitor sessions
    security_results = security_system.batch_security_analysis(test_sessions)
    
    # Generate and print report
    report = security_system.generate_security_report()
    print(report)
    
    # Plot analysis
    security_system.plot_security_analysis(security_results)

# Create aliases for backward compatibility with test modules
SecurityMonitor = QKDSecuritySystem
ThreatDetector = RealTimeSecurityMonitor  
CryptographicVerifier = SecurityMetricsCalculator

# SecurityEvent class for test compatibility
from dataclasses import dataclass

@dataclass
class SecurityEvent:
    """Security event data class"""
    event_type: str
    severity: str
    confidence: float
    timestamp: float
    details: Dict[str, Any]
    session_id: Optional[str] = None
