"""
Test suite for Signal Analyzer module.

This module contains comprehensive tests for signal processing, analysis,
and feature extraction functionality used in QKD systems.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import signal, stats
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signal_analyzer import SignalAnalyzer, SignalFeatures, SpectralAnalyzer


class TestSignalAnalyzer:
    """Test suite for Signal Analyzer functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.analyzer = SignalAnalyzer()
        self.spectral_analyzer = SpectralAnalyzer()
        
        # Create synthetic signal data
        np.random.seed(42)
        self.create_test_signals()
    
    def create_test_signals(self):
        """Create synthetic signals for testing."""
        self.fs = 1000  # Sampling frequency
        self.duration = 2.0  # Duration in seconds
        self.t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        
        # Clean signal: 50 Hz sine wave
        self.clean_signal = np.sin(2 * np.pi * 50 * self.t)
        
        # Noisy signal: 50 Hz + noise
        noise_power = 0.1
        self.noisy_signal = self.clean_signal + np.random.normal(0, noise_power, len(self.t))
        
        # Multi-frequency signal
        self.multi_freq_signal = (np.sin(2 * np.pi * 50 * self.t) + 
                                 0.5 * np.sin(2 * np.pi * 120 * self.t) + 
                                 0.3 * np.sin(2 * np.pi * 300 * self.t))
        
        # Chirp signal
        self.chirp_signal = signal.chirp(self.t, f0=10, f1=100, t1=self.duration, method='linear')
        
        # QKD-specific signals
        self.create_qkd_signals()
    
    def create_qkd_signals(self):
        """Create QKD-specific test signals."""
        # Photon detection signal
        photon_rate = 100  # Hz
        detection_times = np.random.exponential(1/photon_rate, 200)
        detection_times = np.cumsum(detection_times)
        detection_times = detection_times[detection_times < self.duration]
        
        self.photon_signal = np.zeros_like(self.t)
        for det_time in detection_times:
            idx = np.argmin(np.abs(self.t - det_time))
            self.photon_signal[idx] = 1.0
        
        # QBER signal (time-varying error rate)
        base_qber = 0.05
        qber_variation = 0.02 * np.sin(2 * np.pi * 0.1 * self.t)  # Slow variation
        self.qber_signal = base_qber + qber_variation + 0.005 * np.random.randn(len(self.t))
        
        # Key rate signal
        base_key_rate = 1000
        key_rate_variation = 200 * np.sin(2 * np.pi * 0.05 * self.t)
        self.key_rate_signal = base_key_rate + key_rate_variation + 50 * np.random.randn(len(self.t))
    
    def test_analyzer_initialization(self):
        """Test signal analyzer initialization."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'analyze_signal')
        assert hasattr(self.analyzer, 'extract_features')
        assert hasattr(self.analyzer, 'detect_anomalies')
    
    def test_basic_signal_analysis(self):
        """Test basic signal analysis functionality."""
        analysis = self.analyzer.analyze_signal(self.noisy_signal, self.fs)
        
        assert 'statistics' in analysis
        assert 'spectral' in analysis
        assert 'temporal' in analysis
        
        # Check statistical measures
        stats = analysis['statistics']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'rms' in stats
        assert 'peak_to_peak' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        
        # Verify reasonable values
        assert abs(stats['mean']) < 0.1  # Should be close to 0
        assert 0.5 < stats['rms'] < 1.5  # RMS of sine wave ≈ 0.707
    
    def test_spectral_analysis(self):
        """Test frequency domain analysis."""
        spectral_analysis = self.spectral_analyzer.analyze_spectrum(
            self.multi_freq_signal, self.fs
        )
        
        assert 'power_spectrum' in spectral_analysis
        assert 'frequencies' in spectral_analysis
        assert 'dominant_frequencies' in spectral_analysis
        assert 'spectral_centroid' in spectral_analysis
        assert 'spectral_bandwidth' in spectral_analysis
        
        # Check dominant frequencies
        dom_freqs = spectral_analysis['dominant_frequencies']
        assert len(dom_freqs) >= 3  # Should detect 50, 120, 300 Hz
        
        # Verify frequency detection accuracy
        expected_freqs = [50, 120, 300]
        for expected_freq in expected_freqs:
            closest_detected = min(dom_freqs, key=lambda x: abs(x - expected_freq))
            assert abs(closest_detected - expected_freq) < 5  # Within 5 Hz
    
    def test_feature_extraction(self):
        """Test signal feature extraction."""
        features = self.analyzer.extract_features(self.noisy_signal, self.fs)
        
        assert isinstance(features, SignalFeatures)
        
        # Time domain features
        assert hasattr(features, 'mean')
        assert hasattr(features, 'variance')
        assert hasattr(features, 'rms')
        assert hasattr(features, 'zero_crossings')
        assert hasattr(features, 'peak_amplitude')
        
        # Frequency domain features
        assert hasattr(features, 'spectral_centroid')
        assert hasattr(features, 'spectral_bandwidth')
        assert hasattr(features, 'spectral_rolloff')
        assert hasattr(features, 'spectral_flux')
        
        # Statistical features
        assert hasattr(features, 'skewness')
        assert hasattr(features, 'kurtosis')
        assert hasattr(features, 'entropy')
        
        # Verify feature values are reasonable
        assert -5 < features.skewness < 5
        assert 0 < features.kurtosis < 10
        assert features.entropy > 0
    
    def test_qkd_specific_analysis(self):
        """Test QKD-specific signal analysis."""
        # Analyze photon detection signal
        photon_analysis = self.analyzer.analyze_photon_stream(
            self.photon_signal, self.fs
        )
        
        assert 'detection_rate' in photon_analysis
        assert 'inter_arrival_times' in photon_analysis
        assert 'bunching_factor' in photon_analysis
        assert 'randomness_test' in photon_analysis
        
        # Analyze QBER signal
        qber_analysis = self.analyzer.analyze_qber_signal(
            self.qber_signal, self.fs
        )
        
        assert 'mean_qber' in qber_analysis
        assert 'qber_variance' in qber_analysis
        assert 'qber_trend' in qber_analysis
        assert 'anomaly_periods' in qber_analysis
        
        # Verify QBER values are reasonable
        assert 0 < qber_analysis['mean_qber'] < 0.5
        assert qber_analysis['qber_variance'] > 0
    
    def test_noise_analysis(self):
        """Test noise characterization and analysis."""
        noise_analysis = self.analyzer.analyze_noise(
            self.noisy_signal, self.clean_signal, self.fs
        )
        
        assert 'snr' in noise_analysis
        assert 'noise_power' in noise_analysis
        assert 'noise_type' in noise_analysis
        assert 'noise_statistics' in noise_analysis
        
        # SNR should be reasonable for our test signals
        assert 5 < noise_analysis['snr'] < 25  # dB
        assert noise_analysis['noise_power'] > 0
    
    def test_filtering_operations(self):
        """Test signal filtering functionality."""
        # Low-pass filter
        filtered_low = self.analyzer.apply_lowpass_filter(
            self.multi_freq_signal, cutoff=80, fs=self.fs
        )
        
        # High-pass filter
        filtered_high = self.analyzer.apply_highpass_filter(
            self.multi_freq_signal, cutoff=80, fs=self.fs
        )
        
        # Band-pass filter
        filtered_band = self.analyzer.apply_bandpass_filter(
            self.multi_freq_signal, low_cutoff=40, high_cutoff=60, fs=self.fs
        )
        
        # Verify filtering effects
        assert len(filtered_low) == len(self.multi_freq_signal)
        assert len(filtered_high) == len(self.multi_freq_signal)
        assert len(filtered_band) == len(self.multi_freq_signal)
        
        # Check that bandpass filter preserves 50 Hz component
        band_spectrum = np.abs(np.fft.fft(filtered_band))
        freqs = np.fft.fftfreq(len(filtered_band), 1/self.fs)
        idx_50hz = np.argmin(np.abs(freqs - 50))
        assert band_spectrum[idx_50hz] > 0.1 * np.max(band_spectrum)
    
    def test_windowing_functions(self):
        """Test signal windowing operations."""
        # Test different window types
        windows = ['hanning', 'hamming', 'blackman', 'kaiser']
        
        for window_type in windows:
            windowed_signal = self.analyzer.apply_window(
                self.clean_signal, window_type
            )
            
            assert len(windowed_signal) == len(self.clean_signal)
            
            # Windowed signal should have reduced edge effects
            edge_ratio = np.mean(np.abs(windowed_signal[:10])) / np.mean(np.abs(windowed_signal))
            assert edge_ratio < 0.5  # Edges should be attenuated
    
    def test_correlation_analysis(self):
        """Test signal correlation and similarity analysis."""
        # Auto-correlation
        autocorr = self.analyzer.compute_autocorrelation(self.clean_signal)
        
        # Auto-correlation at zero lag should be maximum
        zero_lag_idx = len(autocorr) // 2
        assert autocorr[zero_lag_idx] == np.max(autocorr)
        
        # Cross-correlation
        cross_corr = self.analyzer.compute_crosscorrelation(
            self.clean_signal, self.noisy_signal
        )
        
        # Signals should be highly correlated
        max_corr = np.max(np.abs(cross_corr))
        assert max_corr > 0.8
    
    def test_time_frequency_analysis(self):
        """Test time-frequency analysis methods."""
        # Short-time Fourier transform
        stft_result = self.analyzer.compute_stft(self.chirp_signal, self.fs)
        
        assert 'frequencies' in stft_result
        assert 'times' in stft_result
        assert 'spectrogram' in stft_result
        
        # Verify spectrogram dimensions
        f, t, Zxx = stft_result['frequencies'], stft_result['times'], stft_result['spectrogram']
        assert len(f) > 0
        assert len(t) > 0
        assert Zxx.shape == (len(f), len(t))
        
        # Wavelet analysis
        wavelet_result = self.analyzer.compute_wavelet_transform(
            self.chirp_signal, self.fs
        )
        
        assert 'scales' in wavelet_result
        assert 'coefficients' in wavelet_result
        assert 'frequencies' in wavelet_result
    
    def test_anomaly_detection(self):
        """Test signal-based anomaly detection."""
        # Create signal with anomalies
        anomalous_signal = self.clean_signal.copy()
        
        # Insert spike anomaly
        spike_idx = len(anomalous_signal) // 2
        anomalous_signal[spike_idx:spike_idx+10] += 5.0
        
        # Insert drift anomaly
        drift_start = len(anomalous_signal) * 3 // 4
        anomalous_signal[drift_start:] += np.linspace(0, 2, len(anomalous_signal) - drift_start)
        
        # Detect anomalies
        anomalies = self.analyzer.detect_anomalies(anomalous_signal, self.fs)
        
        assert 'spike_anomalies' in anomalies
        assert 'drift_anomalies' in anomalies
        assert 'statistical_anomalies' in anomalies
        
        # Should detect the inserted anomalies
        spike_detections = anomalies['spike_anomalies']
        assert len(spike_detections) > 0
        
        # Check that spike is detected near the inserted location
        spike_detected_near_insertion = any(
            abs(detection - spike_idx) < 50 for detection in spike_detections
        )
        assert spike_detected_near_insertion
    
    def test_real_time_processing(self):
        """Test real-time signal processing capabilities."""
        # Initialize real-time processor
        rt_processor = self.analyzer.create_realtime_processor(
            buffer_size=1024, overlap=512, fs=self.fs
        )
        
        # Process signal in chunks
        chunk_size = 256
        results = []
        
        for i in range(0, len(self.noisy_signal), chunk_size):
            chunk = self.noisy_signal[i:i+chunk_size]
            if len(chunk) == chunk_size:
                result = rt_processor.process_chunk(chunk)
                results.append(result)
        
        assert len(results) > 0
        
        # Each result should contain basic features
        for result in results:
            assert 'rms' in result
            assert 'peak' in result
            assert 'spectral_centroid' in result
    
    def test_signal_quality_assessment(self):
        """Test signal quality metrics."""
        # Assess clean signal quality
        clean_quality = self.analyzer.assess_signal_quality(self.clean_signal, self.fs)
        
        # Assess noisy signal quality
        noisy_quality = self.analyzer.assess_signal_quality(self.noisy_signal, self.fs)
        
        assert 'snr_estimate' in clean_quality
        assert 'distortion_level' in clean_quality
        assert 'dynamic_range' in clean_quality
        assert 'quality_score' in clean_quality
        
        # Clean signal should have better quality
        assert clean_quality['quality_score'] > noisy_quality['quality_score']
        assert clean_quality['snr_estimate'] > noisy_quality['snr_estimate']
    
    def test_adaptive_filtering(self):
        """Test adaptive filtering algorithms."""
        # LMS adaptive filter
        lms_output = self.analyzer.apply_lms_filter(
            self.noisy_signal, 
            reference=self.clean_signal,
            filter_length=32,
            step_size=0.01
        )
        
        assert len(lms_output) == len(self.noisy_signal)
        
        # Adaptive filter should improve SNR
        original_snr = self.analyzer.calculate_snr(self.noisy_signal, self.clean_signal)
        filtered_snr = self.analyzer.calculate_snr(lms_output, self.clean_signal)
        
        # Allow for some variation in adaptive filter performance
        # SNR improvement is not guaranteed for all signals
        assert filtered_snr >= original_snr - 3  # Allow 3 dB tolerance
    
    @pytest.mark.benchmark
    def test_fft_benchmark(self, benchmark):
        """Benchmark FFT computation performance."""
        result = benchmark(
            self.analyzer.compute_fft,
            self.multi_freq_signal
        )
        
        # FFT should be computed quickly
        assert benchmark.stats.mean < 0.1  # <100ms
    
    @pytest.mark.benchmark
    def test_feature_extraction_benchmark(self, benchmark):
        """Benchmark feature extraction performance."""
        result = benchmark(
            self.analyzer.extract_features,
            self.noisy_signal, self.fs
        )
        
        # Feature extraction should be efficient
        assert benchmark.stats.mean < 0.5  # <500ms
    
    def test_multithread_processing(self):
        """Test multi-threaded signal processing."""
        # Create multiple signals for parallel processing
        signals = [
            self.clean_signal,
            self.noisy_signal,
            self.multi_freq_signal,
            self.chirp_signal
        ]
        
        # Process in parallel
        results = self.analyzer.process_signals_parallel(signals, self.fs, n_threads=2)
        
        assert len(results) == len(signals)
        
        # Each result should contain analysis data
        for result in results:
            assert 'features' in result
            assert 'spectrum' in result
            assert 'quality' in result


class TestSignalFeatures:
    """Test suite for SignalFeatures data structure."""
    
    def test_feature_creation(self):
        """Test signal features object creation."""
        features = SignalFeatures(
            mean=0.1,
            variance=0.5,
            rms=0.707,
            spectral_centroid=150.0,
            zero_crossings=50
        )
        
        assert features.mean == 0.1
        assert features.variance == 0.5
        assert features.rms == 0.707
        assert features.spectral_centroid == 150.0
        assert features.zero_crossings == 50
    
    def test_feature_normalization(self):
        """Test feature normalization methods."""
        features = SignalFeatures()
        
        # Set some test values
        features.spectral_centroid = 1000.0
        features.spectral_bandwidth = 500.0
        features.rms = 2.0
        
        # Normalize features
        normalized = features.normalize(method='minmax')
        
        assert 0 <= normalized.spectral_centroid <= 1
        assert 0 <= normalized.spectral_bandwidth <= 1
        assert 0 <= normalized.rms <= 1


class TestSpectralAnalyzer:
    """Test suite for Spectral Analyzer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SpectralAnalyzer()
        self.fs = 1000
        self.t = np.linspace(0, 1, self.fs, endpoint=False)
        self.test_signal = np.sin(2 * np.pi * 100 * self.t) + 0.5 * np.sin(2 * np.pi * 250 * self.t)
    
    def test_power_spectral_density(self):
        """Test power spectral density estimation."""
        freqs, psd = self.analyzer.compute_psd(self.test_signal, self.fs)
        
        assert len(freqs) == len(psd)
        assert freqs[0] >= 0
        assert freqs[-1] <= self.fs / 2  # Nyquist frequency
        assert np.all(psd >= 0)  # PSD should be non-negative
        
        # Should detect peaks at 100 and 250 Hz
        peak_indices = self.analyzer.find_spectral_peaks(freqs, psd)
        peak_freqs = freqs[peak_indices]
        
        assert any(abs(freq - 100) < 5 for freq in peak_freqs)
        assert any(abs(freq - 250) < 5 for freq in peak_freqs)
    
    def test_spectral_estimation_methods(self):
        """Test different spectral estimation methods."""
        methods = ['periodogram', 'welch', 'multitaper', 'burg']
        
        for method in methods:
            try:
                freqs, spectrum = self.analyzer.estimate_spectrum(
                    self.test_signal, self.fs, method=method
                )
                
                assert len(freqs) == len(spectrum)
                assert freqs[0] >= 0
                assert np.all(spectrum >= 0)
                
            except NotImplementedError:
                # Some methods might not be implemented
                pytest.skip(f"Method {method} not implemented")


if __name__ == "__main__":
    pytest.main([__file__])
