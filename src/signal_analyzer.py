"""
Signal Processing and Analysis Module for QKD Systems

Advanced signal processing techniques for analyzing quantum signals and detecting
failure patterns in QKD systems.

Author: Under guidance of Vijayalaxmi Mogiligidda
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import signal, fft
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QKDSignalProcessor:
    """Signal processing for QKD quantum signals"""
    
    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate
        self.signal_history = []
        
    def simulate_quantum_signal(self, qkd_session: Dict, noise_level: float = 0.1) -> np.ndarray:
        """Simulate quantum signal based on QKD session parameters"""
        # Generate base signal based on QKD parameters
        length = qkd_session.get('initial_length', 1000)
        qber = qkd_session.get('qber', 0.02)
        
        # Generate ideal signal
        t = np.linspace(0, length / self.sampling_rate, length)
        base_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz carrier
        
        # Add quantum noise and errors
        quantum_noise = np.random.normal(0, noise_level, length)
        error_spikes = np.random.random(length) < qber
        
        # Simulate detector response
        signal_with_noise = base_signal + quantum_noise
        signal_with_noise[error_spikes] += np.random.normal(0, 0.5, np.sum(error_spikes))
        
        # Simulate channel effects
        channel_loss = qkd_session.get('channel_loss', 0.1)
        signal_with_noise *= (1 - channel_loss)
        
        # Add dark counts
        dark_count_rate = 1e-3
        dark_counts = np.random.poisson(dark_count_rate, length)
        signal_with_noise += dark_counts * 0.1
        
        return signal_with_noise
    
    def extract_signal_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive signal features"""
        features = {}
        
        # Time domain features
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['variance'] = np.var(signal)
        features['skewness'] = self._calculate_skewness(signal)
        features['kurtosis'] = self._calculate_kurtosis(signal)
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['peak_to_peak'] = np.max(signal) - np.min(signal)
        features['crest_factor'] = np.max(np.abs(signal)) / features['rms']
        
        # Statistical moments
        features['moment_2'] = np.mean(signal**2)
        features['moment_3'] = np.mean(signal**3)
        features['moment_4'] = np.mean(signal**4)
        
        # Zero crossing rate
        features['zero_crossing_rate'] = self._zero_crossing_rate(signal)
        
        # Energy and power
        features['energy'] = np.sum(signal**2)
        features['power'] = features['energy'] / len(signal)
        
        # Entropy measures
        features['spectral_entropy'] = self._spectral_entropy(signal)
        features['shannon_entropy'] = self._shannon_entropy(signal)
        
        return features
    
    def _calculate_skewness(self, signal: np.ndarray) -> float:
        """Calculate skewness of signal"""
        mean = np.mean(signal)
        std = np.std(signal)
        return np.mean(((signal - mean) / std)**3) if std > 0 else 0
    
    def _calculate_kurtosis(self, signal: np.ndarray) -> float:
        """Calculate kurtosis of signal"""
        mean = np.mean(signal)
        std = np.std(signal)
        return np.mean(((signal - mean) / std)**4) - 3 if std > 0 else 0
    
    def _zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Calculate zero crossing rate"""
        return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
    
    def _spectral_entropy(self, signal: np.ndarray) -> float:
        """Calculate spectral entropy"""
        # Compute power spectral density
        freqs, psd = scipy.signal.welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
        
        # Normalize PSD
        psd_normalized = psd / np.sum(psd)
        
        # Calculate entropy
        return entropy(psd_normalized)
    
    def _shannon_entropy(self, signal: np.ndarray) -> float:
        """Calculate Shannon entropy of signal"""
        # Discretize signal into bins
        hist, _ = np.histogram(signal, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        
        return entropy(hist)


class FrequencyAnalyzer:
    """Frequency domain analysis for QKD signals"""
    
    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate
        self.signal_history = []  # Add missing attribute
        
    def fft_analysis(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform FFT analysis"""
        # Compute FFT
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)
        
        # Take only positive frequencies
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]
        
        return freqs, np.abs(fft_vals)
    
    def power_spectral_density(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectral density using Welch's method"""
        nperseg = min(256, len(signal)//4)
        freqs, psd = scipy.signal.welch(
            signal, 
            fs=self.sampling_rate, 
            nperseg=nperseg,
            noverlap=int(nperseg * 0.5)
        )
        return freqs, psd
    
    def spectrogram_analysis(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram for time-frequency analysis"""
        freqs, times, Sxx = scipy.signal.spectrogram(
            signal,
            fs=self.sampling_rate,
            nperseg=min(256, len(signal)//8)
        )
        return freqs, times, Sxx
    
    def extract_frequency_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features"""
        freqs, fft_magnitude = self.fft_analysis(signal)
        psd_freqs, psd = self.power_spectral_density(signal)
        
        features = {}
        
        # Spectral centroid (center of mass of spectrum)
        features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
        
        # Spectral spread (standard deviation around centroid)
        features['spectral_spread'] = np.sqrt(
            np.sum(((freqs - features['spectral_centroid'])**2) * fft_magnitude) / np.sum(fft_magnitude)
        )
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(fft_magnitude)
        features['spectral_rolloff'] = freqs[np.where(cumsum >= 0.85 * cumsum[-1])[0][0]]
        
        # Spectral flux (measure of how quickly the power spectrum changes)
        if len(self.signal_history) > 0:
            prev_spectrum = np.abs(np.fft.fft(self.signal_history[-1]))[:len(fft_magnitude)]
            if len(prev_spectrum) == len(fft_magnitude):
                features['spectral_flux'] = np.sum((fft_magnitude - prev_spectrum)**2)
            else:
                features['spectral_flux'] = 0
        else:
            features['spectral_flux'] = 0
        
        # Peak frequency
        peak_idx = np.argmax(fft_magnitude)
        features['peak_frequency'] = freqs[peak_idx]
        features['peak_magnitude'] = fft_magnitude[peak_idx]
        
        # Bandwidth (frequency range containing 90% of energy)
        energy_threshold = 0.9 * np.sum(fft_magnitude)
        cumulative_energy = np.cumsum(fft_magnitude)
        bandwidth_indices = np.where(cumulative_energy <= energy_threshold)[0]
        if len(bandwidth_indices) > 0:
            features['bandwidth'] = freqs[bandwidth_indices[-1]] - freqs[bandwidth_indices[0]]
        else:
            features['bandwidth'] = 0
        
        # Spectral flatness (measure of how noise-like vs. tone-like)
        geometric_mean = np.exp(np.mean(np.log(fft_magnitude + 1e-10)))
        arithmetic_mean = np.mean(fft_magnitude)
        features['spectral_flatness'] = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        
        # Store signal for flux calculation
        self.signal_history.append(signal)
        if len(self.signal_history) > 10:  # Keep only recent history
            self.signal_history.pop(0)
        
        return features


class AnomalyPatternDetector:
    """Detect anomaly patterns in signal characteristics"""
    
    def __init__(self):
        self.baseline_features = {}
        self.thresholds = {}
        
    def establish_baseline(self, normal_signals: List[np.ndarray], processor: QKDSignalProcessor):
        """Establish baseline signal characteristics"""
        all_features = []
        
        for signal in normal_signals:
            features = processor.extract_signal_features(signal)
            all_features.append(features)
        
        # Calculate baseline statistics
        feature_df = pd.DataFrame(all_features)
        
        for feature_name in feature_df.columns:
            self.baseline_features[feature_name] = {
                'mean': feature_df[feature_name].mean(),
                'std': feature_df[feature_name].std(),
                'min': feature_df[feature_name].min(),
                'max': feature_df[feature_name].max(),
                'median': feature_df[feature_name].median()
            }
            
            # Set thresholds (3-sigma rule)
            mean = self.baseline_features[feature_name]['mean']
            std = self.baseline_features[feature_name]['std']
            self.thresholds[feature_name] = {
                'lower': mean - 3 * std,
                'upper': mean + 3 * std
            }
        
        logger.info("Signal baseline established for anomaly detection")
    
    def detect_signal_anomalies(self, signal: np.ndarray, processor: QKDSignalProcessor) -> Dict[str, bool]:
        """Detect anomalies in signal characteristics"""
        features = processor.extract_signal_features(signal)
        anomalies = {}
        
        for feature_name, value in features.items():
            if feature_name in self.thresholds:
                threshold = self.thresholds[feature_name]
                anomalies[f'{feature_name}_anomaly'] = (
                    value < threshold['lower'] or value > threshold['upper']
                )
        
        return anomalies
    
    def detect_pattern_anomalies(self, signals: List[np.ndarray], processor: QKDSignalProcessor) -> Dict:
        """Detect pattern-based anomalies across multiple signals"""
        # Extract features for all signals
        all_features = []
        for signal in signals:
            features = processor.extract_signal_features(signal)
            all_features.append(features)
        
        feature_df = pd.DataFrame(all_features)
        
        # Detect trends and patterns
        pattern_anomalies = {}
        
        # Trend detection
        for feature_name in feature_df.columns:
            values = feature_df[feature_name].values
            
            # Linear trend
            if len(values) > 5:
                trend_coeff = np.polyfit(range(len(values)), values, 1)[0]
                baseline_std = self.baseline_features.get(feature_name, {}).get('std', 1)
                
                pattern_anomalies[f'{feature_name}_strong_trend'] = abs(trend_coeff) > 2 * baseline_std
            
            # Sudden jumps
            if len(values) > 1:
                diffs = np.diff(values)
                baseline_std = self.baseline_features.get(feature_name, {}).get('std', 1)
                
                pattern_anomalies[f'{feature_name}_sudden_jump'] = np.any(
                    np.abs(diffs) > 3 * baseline_std
                )
        
        return pattern_anomalies


class SignalQualityAnalyzer:
    """Analyze signal quality metrics for QKD systems"""
    
    def __init__(self):
        self.quality_metrics = {}
        
    def calculate_snr(self, signal: np.ndarray, noise_estimate: np.ndarray = None) -> float:
        """Calculate signal-to-noise ratio"""
        if noise_estimate is None:
            # Estimate noise as high-frequency components
            filtered_signal = self._lowpass_filter(signal, cutoff=50)
            noise_estimate = signal - filtered_signal
        
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise_estimate**2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
        
        return snr_db
    
    def _lowpass_filter(self, signal: np.ndarray, cutoff: float, order: int = 5) -> np.ndarray:
        """Apply lowpass filter"""
        nyquist = 0.5 * 1000  # Assuming 1000 Hz sampling rate
        normal_cutoff = cutoff / nyquist
        b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        return filtered_signal
    
    def calculate_thd(self, signal: np.ndarray, fundamental_freq: float) -> float:
        """Calculate Total Harmonic Distortion"""
        # Perform FFT
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/1000)
        
        # Find fundamental and harmonics
        fundamental_idx = np.argmin(np.abs(freqs - fundamental_freq))
        fundamental_magnitude = np.abs(fft_vals[fundamental_idx])
        
        # Find harmonics (2f, 3f, 4f, 5f)
        harmonic_power = 0
        for harmonic in range(2, 6):
            harmonic_freq = harmonic * fundamental_freq
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonic_power += np.abs(fft_vals[harmonic_idx])**2
        
        # Calculate THD
        if fundamental_magnitude > 0:
            thd = np.sqrt(harmonic_power) / fundamental_magnitude
        else:
            thd = 0
        
        return thd
    
    def analyze_signal_quality(self, signal: np.ndarray, qkd_params: Dict) -> Dict[str, float]:
        """Comprehensive signal quality analysis"""
        quality_metrics = {}
        
        # SNR analysis
        quality_metrics['snr_db'] = self.calculate_snr(signal)
        
        # Dynamic range
        quality_metrics['dynamic_range'] = 20 * np.log10(np.max(signal) / np.min(np.abs(signal[signal != 0])))
        
        # Peak-to-average power ratio
        peak_power = np.max(signal**2)
        avg_power = np.mean(signal**2)
        quality_metrics['papr_db'] = 10 * np.log10(peak_power / avg_power) if avg_power > 0 else 0
        
        # Effective number of bits (ENOB)
        noise_floor = np.std(signal) / np.sqrt(2)  # RMS noise
        full_scale = np.max(signal) - np.min(signal)
        quality_metrics['enob'] = np.log2(full_scale / (2 * noise_floor)) if noise_floor > 0 else 16
        
        # Spurious-free dynamic range
        fft_magnitude = np.abs(np.fft.fft(signal))
        signal_peak = np.max(fft_magnitude)
        spurious_peak = np.max(fft_magnitude[fft_magnitude < signal_peak])
        quality_metrics['sfdr_db'] = 20 * np.log10(signal_peak / spurious_peak) if spurious_peak > 0 else 100
        
        # Total harmonic distortion
        quality_metrics['thd'] = self.calculate_thd(signal, 10.0)  # Assuming 10 Hz fundamental
        
        return quality_metrics


class QKDSignalAnalyzer:
    """Complete signal analysis system for QKD"""
    
    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate
        self.signal_processor = QKDSignalProcessor(sampling_rate)
        self.frequency_analyzer = FrequencyAnalyzer(sampling_rate)
        self.anomaly_detector = AnomalyPatternDetector()
        self.quality_analyzer = SignalQualityAnalyzer()
        self.analysis_history = []
        
    def analyze_qkd_session(self, qkd_session: Dict) -> Dict:
        """Comprehensive analysis of QKD session signal"""
        # Simulate quantum signal
        signal = self.signal_processor.simulate_quantum_signal(qkd_session)
        
        # Extract all features
        time_features = self.signal_processor.extract_signal_features(signal)
        freq_features = self.frequency_analyzer.extract_frequency_features(signal)
        quality_metrics = self.quality_analyzer.analyze_signal_quality(signal, qkd_session)
        
        # Detect anomalies if baseline exists
        signal_anomalies = {}
        if self.anomaly_detector.baseline_features:
            signal_anomalies = self.anomaly_detector.detect_signal_anomalies(
                signal, self.signal_processor
            )
        
        # Combine all analysis results
        analysis_result = {
            'session_id': qkd_session.get('session_id', 0),
            'signal': signal,
            'time_features': time_features,
            'frequency_features': freq_features,
            'quality_metrics': quality_metrics,
            'signal_anomalies': signal_anomalies,
            'qkd_params': qkd_session
        }
        
        self.analysis_history.append(analysis_result)
        return analysis_result
    
    def establish_signal_baseline(self, normal_sessions: List[Dict]):
        """Establish baseline from normal QKD sessions"""
        normal_signals = []
        
        for session in normal_sessions:
            signal = self.signal_processor.simulate_quantum_signal(session)
            normal_signals.append(signal)
        
        self.anomaly_detector.establish_baseline(normal_signals, self.signal_processor)
        logger.info("Signal analysis baseline established")
    
    def batch_analyze_sessions(self, qkd_sessions: List[Dict]) -> List[Dict]:
        """Analyze multiple QKD sessions"""
        results = []
        
        for session in qkd_sessions:
            result = self.analyze_qkd_session(session)
            results.append(result)
        
        # Detect pattern anomalies across sessions
        if len(results) > 5:
            signals = [result['signal'] for result in results]
            pattern_anomalies = self.anomaly_detector.detect_pattern_anomalies(
                signals, self.signal_processor
            )
            
            # Add pattern anomalies to results
            for i, result in enumerate(results):
                result['pattern_anomalies'] = {
                    key: value[i] if isinstance(value, np.ndarray) and len(value) > i 
                    else value for key, value in pattern_anomalies.items()
                }
        
        return results
    
    def plot_signal_analysis(self, analysis_results: List[Dict], save_path: str = None):
        """Plot comprehensive signal analysis"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        # Time domain signals
        for i, result in enumerate(analysis_results[:5]):  # Plot first 5 signals
            axes[0, 0].plot(result['signal'][:500], alpha=0.7, label=f"Session {result['session_id']}")
        axes[0, 0].set_title('Time Domain Signals')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        
        # Frequency domain analysis
        if analysis_results:
            freqs, fft_mag = self.frequency_analyzer.fft_analysis(analysis_results[0]['signal'])
            axes[0, 1].plot(freqs[:len(freqs)//2], fft_mag[:len(fft_mag)//2])
            axes[0, 1].set_title('Frequency Spectrum (First Session)')
            axes[0, 1].set_xlabel('Frequency (Hz)')
            axes[0, 1].set_ylabel('Magnitude')
        
        # SNR over sessions
        snr_values = [result['quality_metrics']['snr_db'] for result in analysis_results]
        session_ids = [result['session_id'] for result in analysis_results]
        axes[0, 2].plot(session_ids, snr_values, 'o-')
        axes[0, 2].set_title('SNR Over Sessions')
        axes[0, 2].set_xlabel('Session ID')
        axes[0, 2].set_ylabel('SNR (dB)')
        
        # Feature correlation matrix
        time_features_df = pd.DataFrame([result['time_features'] for result in analysis_results])
        if not time_features_df.empty:
            correlation_matrix = time_features_df.corr()
            sns.heatmap(correlation_matrix, ax=axes[1, 0], cmap='coolwarm', center=0)
            axes[1, 0].set_title('Time Features Correlation')
        
        # Quality metrics distribution
        quality_df = pd.DataFrame([result['quality_metrics'] for result in analysis_results])
        if not quality_df.empty and 'snr_db' in quality_df.columns:
            axes[1, 1].hist(quality_df['snr_db'], bins=20, alpha=0.7)
            axes[1, 1].set_title('SNR Distribution')
            axes[1, 1].set_xlabel('SNR (dB)')
            axes[1, 1].set_ylabel('Frequency')
        
        # Spectral centroid vs QBER
        spectral_centroids = [result['frequency_features']['spectral_centroid'] for result in analysis_results]
        qber_values = [result['qkd_params']['qber'] for result in analysis_results]
        axes[1, 2].scatter(qber_values, spectral_centroids, alpha=0.6)
        axes[1, 2].set_title('Spectral Centroid vs QBER')
        axes[1, 2].set_xlabel('QBER')
        axes[1, 2].set_ylabel('Spectral Centroid (Hz)')
        
        # Anomaly detection results
        if analysis_results[0]['signal_anomalies']:
            anomaly_counts = {}
            for result in analysis_results:
                for anomaly_type, is_anomaly in result['signal_anomalies'].items():
                    if anomaly_type not in anomaly_counts:
                        anomaly_counts[anomaly_type] = 0
                    if is_anomaly:
                        anomaly_counts[anomaly_type] += 1
            
            if anomaly_counts:
                axes[2, 0].bar(range(len(anomaly_counts)), list(anomaly_counts.values()))
                axes[2, 0].set_xticks(range(len(anomaly_counts)))
                axes[2, 0].set_xticklabels(list(anomaly_counts.keys()), rotation=45, ha='right')
                axes[2, 0].set_title('Signal Anomaly Counts')
                axes[2, 0].set_ylabel('Count')
        
        # Power spectral density comparison
        if len(analysis_results) >= 2:
            freqs1, psd1 = self.frequency_analyzer.power_spectral_density(analysis_results[0]['signal'])
            freqs2, psd2 = self.frequency_analyzer.power_spectral_density(analysis_results[-1]['signal'])
            
            axes[2, 1].semilogy(freqs1, psd1, label='First Session')
            axes[2, 1].semilogy(freqs2, psd2, label='Last Session')
            axes[2, 1].set_title('Power Spectral Density Comparison')
            axes[2, 1].set_xlabel('Frequency (Hz)')
            axes[2, 1].set_ylabel('PSD')
            axes[2, 1].legend()
        
        # Overall signal quality trends
        quality_score = []
        for result in analysis_results:
            # Combine multiple quality metrics into single score
            snr = result['quality_metrics']['snr_db']
            enob = result['quality_metrics']['enob']
            thd = result['quality_metrics']['thd']
            
            score = snr + enob * 5 - thd * 50  # Weighted combination
            quality_score.append(score)
        
        axes[2, 2].plot(session_ids, quality_score, 'g-', linewidth=2)
        axes[2, 2].set_title('Overall Signal Quality Trend')
        axes[2, 2].set_xlabel('Session ID')
        axes[2, 2].set_ylabel('Quality Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    from qkd_simulator import QKDSystemSimulator, QKDParameters
    
    # Generate sample QKD data
    params = QKDParameters(key_length=1000, error_rate=0.02)
    simulator = QKDSystemSimulator(params)
    
    # Generate sessions
    normal_sessions = simulator.simulate_multiple_sessions(20)
    
    # Inject failures and generate test sessions
    simulator.inject_failure("detector_noise", 0.05)
    test_sessions = simulator.simulate_multiple_sessions(30)
    
    # Initialize signal analyzer
    analyzer = QKDSignalAnalyzer()
    
    # Establish baseline
    analyzer.establish_signal_baseline(normal_sessions[:10])
    
    # Analyze sessions
    analysis_results = analyzer.batch_analyze_sessions(test_sessions)
    
    # Plot results
    analyzer.plot_signal_analysis(analysis_results)
    
    # Print some analysis results
    for result in analysis_results[:3]:
        print(f"\nSession {result['session_id']}:")
        print(f"  SNR: {result['quality_metrics']['snr_db']:.2f} dB")
        print(f"  Spectral Centroid: {result['frequency_features']['spectral_centroid']:.2f} Hz")
        print(f"  Signal Anomalies: {sum(result['signal_anomalies'].values())}")

# Create aliases for backward compatibility with test modules
SignalAnalyzer = QKDSignalAnalyzer
SpectralAnalyzer = SignalQualityAnalyzer

# SignalFeatures class for test compatibility
@dataclass
class SignalFeatures:
    """Signal features data class"""
    time_features: Dict[str, float]
    frequency_features: Dict[str, float]
    quality_metrics: Dict[str, float]
    session_id: str
    timestamp: Optional[float] = None
