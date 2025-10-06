"""Preprocessing pipelines for biomarker data"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
import logging
from scipy import signal
from scipy.stats import skew, kurtosis
import librosa
import cv2
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """Base class for all preprocessors"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process the input data"""
        pass
    
    def __call__(self, data: Any) -> Any:
        return self.process(data)


class SignalQualityChecker:
    """Check signal quality for various biomarker types"""
    
    def __init__(self, 
                 modality: str,
                 quality_threshold: float = 0.7):
        self.modality = modality
        self.quality_threshold = quality_threshold
        self.checks = self._get_quality_checks(modality)
    
    def _get_quality_checks(self, modality: str) -> List[Callable]:
        """Get quality checks for specific modality"""
        checks = {
            'voice': [self._check_voice_quality],
            'movement': [self._check_movement_quality],
            'ecg': [self._check_ecg_quality],
            'eeg': [self._check_eeg_quality],
            'vision': [self._check_vision_quality]
        }
        return checks.get(modality, [self._check_generic_quality])
    
    def check_quality(self, data: np.ndarray) -> Dict[str, Any]:
        """Check data quality and return metrics"""
        quality_scores = []
        issues = []
        
        for check in self.checks:
            score, issue = check(data)
            quality_scores.append(score)
            if issue:
                issues.append(issue)
        
        overall_score = np.mean(quality_scores) if quality_scores else 0.0
        
        return {
            'quality_score': overall_score,
            'passes': overall_score >= self.quality_threshold,
            'issues': issues,
            'detailed_scores': quality_scores
        }
    
    def _check_voice_quality(self, audio: np.ndarray) -> Tuple[float, Optional[str]]:
        """Check voice signal quality"""
        # Check for clipping
        max_val = np.max(np.abs(audio))
        if max_val >= 0.99:
            return 0.3, "Signal clipping detected"
        
        # Check SNR
        signal_power = np.mean(audio ** 2)
        if signal_power < 1e-6:
            return 0.0, "Signal too weak"

        frame_length = 2048
        hop_length = 512
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # NEW: detect stable tone-like signals
        if np.std(energy) / (np.mean(energy) + 1e-10) < 0.1:
            # Very stable energy = likely a clean tone (no noise variation)
            return 1.0, None

        threshold = np.percentile(energy, 10)
        noise_frames = energy < threshold

        if np.any(noise_frames):
            noise_level = np.mean(energy[noise_frames])
            signal_level = np.mean(energy[~noise_frames]) if np.any(~noise_frames) else 0

            if signal_level > 0:
                snr = 20 * np.log10(signal_level / (noise_level + 1e-10))
                quality_score = min(1.0, snr / 40)
                issue = None if snr > 10 else "Low SNR"
                return quality_score, issue

        return 0.8, None

    
    def _check_movement_quality(self, sensor_data: np.ndarray) -> Tuple[float, Optional[str]]:
        """Check movement sensor data quality"""
        issues = []
        
        # Check for missing values
        if np.any(np.isnan(sensor_data)):
            return 0.0, "Missing values detected"
        
        # Check for constant values (sensor stuck)
        for i in range(sensor_data.shape[1] if len(sensor_data.shape) > 1 else 1):
            channel = sensor_data[:, i] if len(sensor_data.shape) > 1 else sensor_data
            if np.std(channel) < 1e-6:
                issues.append(f"Channel {i} appears stuck")
        
        if issues:
            return 0.3, "; ".join(issues)
        
        # Check for outliers
        z_scores = np.abs((sensor_data - np.mean(sensor_data, axis=0)) / (np.std(sensor_data, axis=0) + 1e-10))
        outlier_ratio = np.mean(z_scores > 5)
        
        if outlier_ratio > 0.1:
            return 0.5, "High outlier ratio"
        
        quality_score = 1.0 - outlier_ratio * 2  # Penalize outliers
        return max(0.3, quality_score), None
    
    def _check_ecg_quality(self, ecg_signal: np.ndarray) -> Tuple[float, Optional[str]]:
        """Check ECG signal quality"""
        # Check baseline wander
        from scipy import signal as scipy_signal
        
        # High-pass filter to remove baseline
        b, a = scipy_signal.butter(4, 0.5, 'highpass', fs=250)
        filtered = scipy_signal.filtfilt(b, a, ecg_signal.flatten())
        
        baseline_power = np.mean((ecg_signal.flatten() - filtered) ** 2)
        signal_power = np.mean(filtered ** 2)
        
        if baseline_power > signal_power * 0.5:
            return 0.4, "Excessive baseline wander"
        
        # Check for power line interference (50/60 Hz)
        freqs = np.fft.fftfreq(len(ecg_signal.flatten()), 1/250)
        fft = np.abs(np.fft.fft(ecg_signal.flatten()))
        
        # Check 50Hz and 60Hz
        for freq in [50, 60]:
            freq_idx = np.argmin(np.abs(freqs - freq))
            if fft[freq_idx] > np.mean(fft) * 10:
                return 0.5, f"Power line interference at {freq}Hz"
        
        return 0.9, None
    
    def _check_eeg_quality(self, eeg_signal: np.ndarray) -> Tuple[float, Optional[str]]:
        """Check EEG signal quality"""
        # Check for flat channels
        flat_channels = []
        for ch in range(eeg_signal.shape[1] if len(eeg_signal.shape) > 1 else 1):
            channel = eeg_signal[:, ch] if len(eeg_signal.shape) > 1 else eeg_signal
            if np.std(channel) < 1e-6:
                flat_channels.append(ch)
        
        if flat_channels:
            return 0.3, f"Flat channels detected: {flat_channels}"
        
        # Check for excessive artifacts (simplified)
        max_amplitude = np.max(np.abs(eeg_signal))
        if max_amplitude > 200e-6:  # 200 µV threshold
            return 0.5, "Excessive amplitude - possible artifacts"
        
        return 0.85, None
    
    def _check_vision_quality(self, frames: np.ndarray) -> Tuple[float, Optional[str]]:
        """Check video quality"""
        issues = []
        
        # Check brightness
        mean_brightness = np.mean(frames)
        if mean_brightness < 30:
            issues.append("Too dark")
        elif mean_brightness > 225:
            issues.append("Too bright")
        
        # Check blur (using Laplacian variance)
        for frame_idx in range(min(5, len(frames))):
            frame = frames[frame_idx]
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
            if laplacian_var < 100:
                issues.append(f"Frame {frame_idx} is blurry")
        
        if issues:
            quality_score = max(0.3, 1.0 - len(issues) * 0.2)
            return quality_score, "; ".join(issues)
        
        return 0.9, None
    
    def _check_generic_quality(self, data: np.ndarray) -> Tuple[float, Optional[str]]:
        """Generic quality check"""
        if np.any(np.isnan(data)):
            return 0.0, "NaN values detected"
        if np.any(np.isinf(data)):
            return 0.0, "Inf values detected"
        
        return 0.7, None


class VoicePreprocessor(BasePreprocessor):
    """Preprocessing for voice/audio data"""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 trim_silence: bool = True,
                 normalize: bool = True,
                 extract_features: List[str] = None):
        super().__init__()
        self.target_sr = target_sr
        self.trim_silence = trim_silence
        self.normalize = normalize
        self.extract_features = extract_features or ['mel_spectrogram']
    
    def process(self, audio_path: Union[str, Path, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process audio data"""
        # Load audio if path provided
        if isinstance(audio_path, (str, Path)):
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
        else:
            audio = audio_path
            sr = self.target_sr
        
        # Trim silence
        if self.trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        if self.normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # Extract features
        features = {'waveform': audio}
        
        if 'mel_spectrogram' in self.extract_features:
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, n_fft=512, hop_length=160
            )
            features['mel_spectrogram'] = librosa.power_to_db(mel_spec, ref=np.max)
        
        if 'mfcc' in self.extract_features:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc'] = mfcc
        
        if 'spectral_features' in self.extract_features:
            features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
        
        if 'prosodic_features' in self.extract_features:
            # F0 extraction
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            features['f0'] = f0
            features['voiced_flag'] = voiced_flag
            
            # Energy
            features['energy'] = librosa.feature.rms(y=audio)
        
        return features


class MovementPreprocessor(BasePreprocessor):
    """Preprocessing for movement/sensor data"""
    
    def __init__(self,
                 sampling_rate: int = 100,
                 apply_filters: List[str] = None,
                 normalize: bool = True,
                 remove_gravity: bool = True):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.apply_filters = apply_filters or ['butterworth']
        self.normalize = normalize
        self.remove_gravity = remove_gravity
    
    def process(self, sensor_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Process sensor data"""
        processed = sensor_data.copy()
        
        # Remove gravity component (for accelerometer)
        if self.remove_gravity and processed.shape[1] >= 3:
            # High-pass filter to remove DC component
            b, a = signal.butter(4, 0.5, 'highpass', fs=self.sampling_rate)
            for i in range(3):  # Apply to x, y, z
                processed[:, i] = signal.filtfilt(b, a, processed[:, i])
        
        # Apply filters
        for filter_name in self.apply_filters:
            if filter_name == 'butterworth':
                b, a = signal.butter(4, [0.5, 40], 'bandpass', fs=self.sampling_rate)
                for i in range(processed.shape[1]):
                    processed[:, i] = signal.filtfilt(b, a, processed[:, i])
            
            elif filter_name == 'median':
                from scipy.ndimage import median_filter
                for i in range(processed.shape[1]):
                    processed[:, i] = median_filter(processed[:, i], size=5)
        
        # Normalize
        if self.normalize:
            processed = (processed - np.mean(processed, axis=0)) / (np.std(processed, axis=0) + 1e-10)
        
        # Extract additional features
        features = {
            'processed_signal': processed,
            'magnitude': np.sqrt(np.sum(processed[:, :3]**2, axis=1)) if processed.shape[1] >= 3 else processed
        }
        
        # Statistical features
        features['mean'] = np.mean(processed, axis=0)
        features['std'] = np.std(processed, axis=0)
        features['skewness'] = skew(processed, axis=0)
        features['kurtosis'] = kurtosis(processed, axis=0)
        
        return features


class VisionPreprocessor(BasePreprocessor):
    """Preprocessing for vision/video data"""
    
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 extract_landmarks: bool = False,
                 extract_optical_flow: bool = False):
        super().__init__()
        self.target_size = target_size
        self.normalize = normalize
        self.extract_landmarks = extract_landmarks
        self.extract_optical_flow = extract_optical_flow
    
    def process(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """Process video frames"""
        processed_frames = []
        
        for frame in frames:
            # Resize
            if frame.shape[:2] != self.target_size:
                frame = cv2.resize(frame, self.target_size)
            
            # Normalize
            if self.normalize:
                frame = frame.astype(np.float32) / 255.0
            
            processed_frames.append(frame)
        
        processed_frames = np.array(processed_frames)
        
        features = {'frames': processed_frames}
        
        # Extract optical flow
        if self.extract_optical_flow and len(frames) > 1:
            flows = []
            for i in range(len(frames) - 1):
                gray1 = cv2.cvtColor(frames[i].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frames[i+1].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                flows.append(flow)
            
            features['optical_flow'] = np.array(flows)
        
        # Extract landmarks (placeholder - use mediapipe in practice)
        if self.extract_landmarks:
            # This would use MediaPipe or similar
            features['landmarks'] = np.zeros((len(frames), 68, 2))  # Placeholder
        
        return features


class TextPreprocessor(BasePreprocessor):
    """Preprocessing for text data"""
    
    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 tokenizer: Optional[Any] = None,
                 max_length: int = 512):
        super().__init__()
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process text data"""
        import string
        import re
        
        # Basic cleaning
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        features = {'cleaned_text': text}
        
        # Tokenization
        if self.tokenizer:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            features['input_ids'] = tokens['input_ids'].squeeze()
            features['attention_mask'] = tokens['attention_mask'].squeeze()
        else:
            # Simple word tokenization
            words = text.split()[:self.max_length]
            features['tokens'] = words
        
        # Extract linguistic features
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0
        
        return features


class DataCleaner:
    """Clean and validate biomarker data"""
    
    def __init__(self,
                 remove_outliers: bool = True,
                 interpolate_missing: bool = True,
                 outlier_threshold: float = 3.0):
        self.remove_outliers = remove_outliers
        self.interpolate_missing = interpolate_missing
        self.outlier_threshold = outlier_threshold
    
    def clean(self, data: np.ndarray) -> np.ndarray:
        """Clean data array"""
        cleaned = data.copy()
        
        # Handle missing values
        if np.any(np.isnan(cleaned)):
            if self.interpolate_missing:
                # Linear interpolation for missing values
                for i in range(cleaned.shape[1] if len(cleaned.shape) > 1 else 1):
                    channel = cleaned[:, i] if len(cleaned.shape) > 1 else cleaned
                    mask = ~np.isnan(channel)
                    if np.any(mask):
                        indices = np.arange(len(channel))
                        channel[~mask] = np.interp(indices[~mask], indices[mask], channel[mask])
            else:
                # Remove rows with missing values
                cleaned = cleaned[~np.isnan(cleaned).any(axis=1 if len(cleaned.shape) > 1 else 0)]
        
        # Remove outliers
        if self.remove_outliers:
            if len(cleaned.shape) == 1:
                # Use median absolute deviation for robust outlier detection
                # First, get a clean set of data without NaNs
                valid_data = cleaned[~np.isnan(cleaned)]
                if len(valid_data) > 0:
                    median = np.median(valid_data)
                    diff = np.abs(cleaned - median)
                    mad = np.median(diff[~np.isnan(diff)]) # MAD of non-NaN differences
                    if mad > 1e-10:
                        # Z-score based on MAD
                        z_scores = 0.6745 * diff / mad
                        outliers = z_scores > self.outlier_threshold

                        # Identify all points to be replaced (outliers and original NaNs)
                        points_to_replace = outliers | np.isnan(cleaned)

                        if np.any(points_to_replace):
                            # Calculate the median of only the good points
                            good_points_median = np.median(cleaned[~points_to_replace])
                            # Replace all bad points with this robust median
                            cleaned[points_to_replace] = good_points_median
            else: 
                # Apply the same robust logic for 2D data
                for i in range(cleaned.shape[1]):
                    channel = cleaned[:, i]
                    valid_data = channel[~np.isnan(channel)]
                    if len(valid_data) > 0:
                        median = np.median(valid_data)
                        diff = np.abs(channel - median)
                        mad = np.median(diff[~np.isnan(diff)])
                        if mad > 1e-10:
                            z_scores = 0.6745 * diff / mad
                            outliers = z_scores > self.outlier_threshold
                            points_to_replace = outliers | np.isnan(channel)
                            if np.any(points_to_replace):
                                good_points_median = np.median(channel[~points_to_replace])
                                channel[points_to_replace] = good_points_median
        return cleaned


class FeatureExtractor:
    """Extract features from preprocessed biomarker data"""
    
    def __init__(self,
                 modality: str,
                 feature_set: str = 'comprehensive'):
        self.modality = modality
        self.feature_set = feature_set
        self.extractors = self._get_extractors(modality, feature_set)
    
    def _get_extractors(self, modality: str, feature_set: str) -> Dict[str, Callable]:
        """Get feature extractors for modality"""
        extractors = {}
        
        if modality == 'voice':
            extractors['temporal'] = self._extract_temporal_features
            extractors['spectral'] = self._extract_spectral_features
            if feature_set == 'comprehensive':
                extractors['prosodic'] = self._extract_prosodic_features
        
        elif modality == 'movement':
            extractors['statistical'] = self._extract_statistical_features
            extractors['frequency'] = self._extract_frequency_features
            if feature_set == 'comprehensive':
                extractors['gait'] = self._extract_gait_features
        
        return extractors
    
    def extract(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all features"""
        features = {}
        
        for name, extractor in self.extractors.items():
            try:
                features[name] = extractor(data)
            except Exception as e:
                logger.warning(f"Failed to extract {name} features: {e}")
                features[name] = np.array([])
        
        return features
    
    def _extract_temporal_features(self, data: np.ndarray) -> np.ndarray:
        """Extract temporal domain features"""
        features = []
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.max(data))
        features.append(np.min(data))
        features.append(skew(data.flatten()))
        features.append(kurtosis(data.flatten()))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(data.flatten())) != 0)
        features.append(zero_crossings / len(data.flatten()))
        
        return np.array(features)
    
    def _extract_spectral_features(self, data: np.ndarray) -> np.ndarray:
        """Extract frequency domain features"""
        # Compute FFT
        fft = np.fft.rfft(data.flatten())
        magnitude = np.abs(fft)
        
        features = []
        
        # Spectral centroid
        freqs = np.fft.rfftfreq(len(data.flatten()))
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        features.append(centroid)
        
        # Spectral bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude))
        features.append(bandwidth)
        
        # Spectral rolloff
        cumsum = np.cumsum(magnitude)
        rolloff = freqs[np.where(cumsum >= 0.85 * cumsum[-1])[0][0]]
        features.append(rolloff)
        
        # Spectral entropy
        magnitude_norm = magnitude / np.sum(magnitude)
        entropy = -np.sum(magnitude_norm * np.log(magnitude_norm + 1e-10))
        features.append(entropy)
        
        return np.array(features)
    
    def _extract_prosodic_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract prosodic features from audio"""
        features = []
        
        # Simplified prosodic feature extraction
        # In practice, use more sophisticated methods
        
        # Energy
        energy = np.sqrt(np.mean(audio ** 2))
        features.append(energy)
        
        # Duration-based features would go here
        
        return np.array(features)
    
    def _extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """Extract statistical features"""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(data),
            np.median(data),
            np.std(data),
            np.var(data),
            np.max(data),
            np.min(data),
            np.ptp(data),  # Peak-to-peak
            skew(data.flatten()),
            kurtosis(data.flatten())
        ])
        
        # Percentiles
        features.extend(np.percentile(data.flatten(), [25, 50, 75]))
        
        return np.array(features)
    
    def _extract_frequency_features(self, data: np.ndarray) -> np.ndarray:
        """Extract frequency domain features"""
        features = []
        
        # Power spectral density
        freqs, psd = signal.welch(data.flatten(), nperseg=min(256, len(data.flatten())))
        
        # Peak frequency
        peak_freq = freqs[np.argmax(psd)]
        features.append(peak_freq)
        
        # Total power
        total_power = np.trapz(psd, freqs)
        features.append(total_power)
        
        # Band powers (example bands)
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            features.append(band_power)
        
        return np.array(features)
    
    def _extract_gait_features(self, movement_data: np.ndarray) -> np.ndarray:
        """Extract gait-specific features"""
        features = []
        
        # Simplified gait feature extraction
        # Detect steps (peaks in acceleration magnitude)
        if movement_data.shape[1] >= 3:
            magnitude = np.sqrt(np.sum(movement_data[:, :3]**2, axis=1))
            
            # Find peaks (steps)
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(magnitude, distance=20, height=np.std(magnitude))
            
            # Step count
            features.append(len(peaks))
            
            # Step regularity (coefficient of variation of step intervals)
            if len(peaks) > 1:
                step_intervals = np.diff(peaks)
                cv = np.std(step_intervals) / np.mean(step_intervals) if np.mean(step_intervals) > 0 else 0
                features.append(cv)
            else:
                features.append(0)
            
            # Average step height
            if len(peaks) > 0:
                features.append(np.mean(properties['peak_heights']))
            else:
                features.append(0)
        
        return np.array(features)# Complete Data Directory Implementation
