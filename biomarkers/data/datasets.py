"""Specialized dataset implementations for different biomarker modalities"""

import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
import librosa
import cv2
from scipy import signal as scipy_signal
import logging
from abc import ABC, abstractmethod
import h5py

logger = logging.getLogger(__name__)


class BiomarkerDatasetBase(Dataset, ABC):
    """Base class for all biomarker datasets"""
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 metadata_file: str = 'metadata.csv'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Load metadata
        self.metadata_path = self.data_dir / metadata_file
        if self.metadata_path.exists():
            self.metadata = pd.read_csv(self.metadata_path)
        else:
            self.metadata = self._scan_directory()
        
        self.samples = self._prepare_samples()
    
    @abstractmethod
    def _prepare_samples(self) -> List[Dict]:
        """Prepare sample list"""
        pass
    
    @abstractmethod
    def _load_sample(self, sample_info: Dict) -> Tuple[Any, Any]:
        """Load a single sample"""
        pass
    
    def _scan_directory(self) -> pd.DataFrame:
        """Scan directory to create metadata if not provided"""
        # Override in subclasses
        return pd.DataFrame()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        sample_info = self.samples[idx]
        data, target = self._load_sample(sample_info)
        
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        
        return data, target
    
    def get_sample_metadata(self, idx: int) -> Dict:
        """Get metadata for a specific sample"""
        return self.samples[idx]


class VoiceBiomarkerDataset(BiomarkerDatasetBase):
    """Dataset specifically for voice biomarkers"""
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 n_fft: int = 512,
                 hop_length: int = 160,
                 max_duration: float = 30.0,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 extract_features: List[str] = None):
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.extract_features = extract_features or ['mel_spectrogram']
        
        super().__init__(data_dir, transform, target_transform)
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare voice samples"""
        samples = []
        
        for idx, row in self.metadata.iterrows():
            sample = {
                'path': self.data_dir / 'audio' / row['filename'],
                'patient_id': row['patient_id'],
                'disease': row.get('disease', 'unknown'),
                'duration': row.get('duration', 0),
                'quality': row.get('quality', 1.0)
            }
            
            # Add specific voice metadata
            if 'tremor_score' in row:
                sample['tremor_score'] = row['tremor_score']
            if 'jitter' in row:
                sample['jitter'] = row['jitter']
            if 'shimmer' in row:
                sample['shimmer'] = row['shimmer']
            
            samples.append(sample)
        
        return samples
    
    def _load_sample(self, sample_info: Dict) -> Tuple[torch.Tensor, Dict]:
        """Load voice sample and extract features"""
        # Load audio
        audio, sr = librosa.load(
            sample_info['path'],
            sr=self.sample_rate,
            duration=self.max_duration
        )
        
        features = {}
        
        # Extract requested features
        if 'mel_spectrogram' in self.extract_features:
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=self.n_mels,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            features['mel_spectrogram'] = librosa.power_to_db(mel_spec, ref=np.max)
        
        if 'mfcc' in self.extract_features:
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=13,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            features['mfcc'] = mfcc
        
        if 'prosody' in self.extract_features:
            # Extract prosodic features
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            features['f0'] = f0
            features['voiced_flag'] = voiced_flag
        
        if 'formants' in self.extract_features:
            # Extract formants (simplified)
            formants = self._extract_formants(audio, sr)
            features['formants'] = formants
        
        # Convert to tensors
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                features[key] = torch.FloatTensor(value)
        
        # Create target
        target = {
            'disease': sample_info['disease'],
            'metadata': {
                'patient_id': sample_info['patient_id'],
                'quality': sample_info['quality']
            }
        }
        
        return features, target
    
    def _extract_formants(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract formant frequencies"""
        # Simplified formant extraction using LPC
        try:
            from scipy.signal import lfilter
            # Pre-emphasis
            pre_emphasis = 0.97
            emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # LPC analysis
            order = 2 + sr // 1000
            a = librosa.lpc(emphasized, order=order)
            
            # Find roots and convert to frequencies
            roots = np.roots(a)
            roots = [r for r in roots if np.imag(r) >= 0]
            angles = np.arctan2(np.imag(roots), np.real(roots))
            freqs = sorted(angles * (sr / (2 * np.pi)))
            
            # Return first 4 formants
            formants = freqs[:4] if len(freqs) >= 4 else freqs + [0] * (4 - len(freqs))
            return np.array(formants)
        except Exception as e:
            logger.warning(f"Formant extraction failed: {e}")
            return np.zeros(4)


class MovementBiomarkerDataset(BiomarkerDatasetBase):
    """Dataset for movement/accelerometer biomarkers"""
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 sampling_rate: int = 100,
                 window_size: int = 1000,
                 overlap: float = 0.5,
                 sensors: List[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.sensors = sensors or ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        super().__init__(data_dir, transform, target_transform)
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare movement samples with windowing"""
        samples = []
        
        for idx, row in self.metadata.iterrows():
            # Load data to determine number of windows
            data_path = self.data_dir / 'sensors' / row['filename']
            data = pd.read_csv(data_path)
            
            # Calculate windows
            total_samples = len(data)
            stride = int(self.window_size * (1 - self.overlap))
            num_windows = (total_samples - self.window_size) // stride + 1
            
            for window_idx in range(num_windows):
                start_idx = window_idx * stride
                end_idx = start_idx + self.window_size
                
                sample = {
                    'path': data_path,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'window_idx': window_idx,
                    'patient_id': row['patient_id'],
                    'disease': row.get('disease', 'unknown'),
                    'activity': row.get('activity', 'unknown')
                }
                samples.append(sample)
        
        return samples
    
    def _load_sample(self, sample_info: Dict) -> Tuple[torch.Tensor, Dict]:
        """Load movement window"""
        # Load data
        data = pd.read_csv(sample_info['path'])
        
        # Extract window
        window_data = data.iloc[sample_info['start_idx']:sample_info['end_idx']]
        sensor_data = window_data[self.sensors].values
        
        # Normalize
        sensor_data = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-8)
        
        # Extract features
        features = self._extract_movement_features(sensor_data)
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(sensor_data.T)  # [channels, time]
        
        target = {
            'disease': sample_info['disease'],
            'activity': sample_info['activity'],
            'metadata': {
                'patient_id': sample_info['patient_id'],
                'window_idx': sample_info['window_idx']
            },
            'features': features
        }
        
        return data_tensor, target
    
    def _extract_movement_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract movement-specific features"""
        features = {}
        
        # Time domain features
        features['mean'] = np.mean(data, axis=0).tolist()
        features['std'] = np.std(data, axis=0).tolist()
        features['max'] = np.max(data, axis=0).tolist()
        features['min'] = np.min(data, axis=0).tolist()
        
        # Frequency domain features
        for i, sensor in enumerate(self.sensors):
            fft = np.fft.rfft(data[:, i])
            freqs = np.fft.rfftfreq(len(data[:, i]), 1/self.sampling_rate)
            
            # Dominant frequency
            features[f'{sensor}_dominant_freq'] = freqs[np.argmax(np.abs(fft))]
            
            # Spectral entropy
            psd = np.abs(fft) ** 2
            psd_norm = psd / np.sum(psd)
            features[f'{sensor}_spectral_entropy'] = -np.sum(
                psd_norm * np.log(psd_norm + 1e-10)
            )
        
        return features


class VisionBiomarkerDataset(BiomarkerDatasetBase):
    """Dataset for vision/video biomarkers"""
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 frame_rate: int = 30,
                 num_frames: int = 16,
                 image_size: Tuple[int, int] = (224, 224),
                 extract_landmarks: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        
        self.frame_rate = frame_rate
        self.num_frames = num_frames
        self.image_size = image_size
        self.extract_landmarks = extract_landmarks
        
        super().__init__(data_dir, transform, target_transform)
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare vision samples"""
        samples = []
        
        for idx, row in self.metadata.iterrows():
            sample = {
                'video_path': self.data_dir / 'videos' / row['filename'],
                'patient_id': row['patient_id'],
                'disease': row.get('disease', 'unknown'),
                'task': row.get('task', 'unknown')  # e.g., 'facial_expression', 'gait'
            }
            samples.append(sample)
        
        return samples
    
    def _load_sample(self, sample_info: Dict) -> Tuple[torch.Tensor, Dict]:
        """Load video sample"""
        frames = self._load_video_frames(sample_info['video_path'])
        
        # Extract landmarks if requested
        landmarks = None
        if self.extract_landmarks:
            landmarks = self._extract_landmarks(frames)
        
        # Convert to tensor
        frames_tensor = torch.FloatTensor(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        
        target = {
            'disease': sample_info['disease'],
            'task': sample_info['task'],
            'metadata': {
                'patient_id': sample_info['patient_id']
            }
        }
        
        if landmarks is not None:
            target['landmarks'] = torch.FloatTensor(landmarks)
        
        return frames_tensor, target
    
    def _load_video_frames(self, video_path: Path) -> np.ndarray:
        """Load frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.image_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame / 255.0)  # Normalize to [0, 1]
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < self.num_frames:
            frames.append(np.zeros((*self.image_size, 3)))
        
        return np.array(frames[:self.num_frames])
    
    def _extract_landmarks(self, frames: np.ndarray) -> np.ndarray:
        """Extract facial landmarks (placeholder - use mediapipe in practice)"""
        # Simplified landmark extraction
        num_landmarks = 68  # Standard facial landmarks
        landmarks = np.zeros((len(frames), num_landmarks, 2))
        
        # In practice, use MediaPipe or similar
        # import mediapipe as mp
        # face_mesh = mp.solutions.face_mesh.FaceMesh()
        # for i, frame in enumerate(frames):
        #     results = face_mesh.process(frame)
        #     if results.multi_face_landmarks:
        #         landmarks[i] = ...
        
        return landmarks


class TextBiomarkerDataset(BiomarkerDatasetBase):
    """Dataset for text/linguistic biomarkers"""
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 tokenizer: Optional[Any] = None,
                 max_length: int = 512,
                 extract_linguistic_features: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extract_linguistic_features = extract_linguistic_features
        
        super().__init__(data_dir, transform, target_transform)
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare text samples"""
        samples = []
        
        for idx, row in self.metadata.iterrows():
            sample = {
                'text_path': self.data_dir / 'texts' / row['filename'],
                'patient_id': row['patient_id'],
                'disease': row.get('disease', 'unknown'),
                'text_type': row.get('text_type', 'unknown')  # e.g., 'speech', 'writing'
            }
            samples.append(sample)
        
        return samples
    
    def _load_sample(self, sample_info: Dict) -> Tuple[Dict, Dict]:
        """Load text sample"""
        with open(sample_info['text_path'], 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize if tokenizer provided
        if self.tokenizer:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            data = {
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze()
            }
        else:
            # Simple tokenization
            words = text.split()[:self.max_length]
            word_ids = [hash(w) % 10000 for w in words]
            word_ids += [0] * (self.max_length - len(word_ids))
            data = {'input_ids': torch.LongTensor(word_ids)}
        
        # Extract linguistic features
        if self.extract_linguistic_features:
            features = self._extract_linguistic_features(text)
            data['linguistic_features'] = torch.FloatTensor(features)
        
        target = {
            'disease': sample_info['disease'],
            'text_type': sample_info['text_type'],
            'metadata': {
                'patient_id': sample_info['patient_id']
            }
        }
        
        return data, target
    
    def _extract_linguistic_features(self, text: str) -> np.ndarray:
        """Extract linguistic features from text"""
        features = []
        
        # Basic features
        words = text.split()
        sentences = text.split('.')
        
        features.append(len(words))  # Word count
        features.append(len(sentences))  # Sentence count
        features.append(len(text))  # Character count
        features.append(np.mean([len(w) for w in words]) if words else 0)  # Avg word length
        
        # Vocabulary diversity
        unique_words = set(words)
        features.append(len(unique_words) / len(words) if words else 0)
        
        # Sentiment (placeholder - use proper sentiment analysis)
        positive_words = ['good', 'happy', 'great', 'wonderful', 'excellent']
        negative_words = ['bad', 'sad', 'terrible', 'awful', 'horrible']
        
        pos_count = sum(1 for w in words if w.lower() in positive_words)
        neg_count = sum(1 for w in words if w.lower() in negative_words)
        
        features.append(pos_count / len(words) if words else 0)
        features.append(neg_count / len(words) if words else 0)
        
        return np.array(features)


class TimeSeriesBiomarkerDataset(BiomarkerDatasetBase):
    """Dataset for time-series biomarker data (ECG, EEG, etc.)"""
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 signal_type: str,  # 'ecg', 'eeg', 'emg'
                 sampling_rate: int,
                 duration: float,
                 channels: List[str],
                 normalize: bool = True,
                 extract_features: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        
        self.signal_type = signal_type
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.channels = channels
        self.normalize = normalize
        self.extract_features = extract_features
        self.window_size = int(sampling_rate * duration)
        
        super().__init__(data_dir, transform, target_transform)
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare time-series samples"""
        samples = []
        
        for idx, row in self.metadata.iterrows():
            sample = {
                'signal_path': self.data_dir / self.signal_type / row['filename'],
                'patient_id': row['patient_id'],
                'disease': row.get('disease', 'unknown'),
                'recording_type': row.get('recording_type', 'resting')
            }
            samples.append(sample)
        
        return samples
    
    def _load_sample(self, sample_info: Dict) -> Tuple[torch.Tensor, Dict]:
        """Load time-series signal"""
        # Load signal based on format
        if sample_info['signal_path'].suffix == '.npy':
            signal = np.load(sample_info['signal_path'])
        elif sample_info['signal_path'].suffix == '.h5':
            with h5py.File(sample_info['signal_path'], 'r') as f:
                signal = f['signal'][:]
        else:
            # CSV or other text format
            signal = pd.read_csv(sample_info['signal_path'], header=None).values
        
        # Select channels
        if len(self.channels) < signal.shape[1]:
            signal = signal[:, :len(self.channels)]
        
        # Crop or pad to fixed duration
        if signal.shape[0] > self.window_size:
            signal = signal[:self.window_size]
        elif signal.shape[0] < self.window_size:
            padding = self.window_size - signal.shape[0]
            signal = np.pad(signal, ((0, padding), (0, 0)), mode='constant')
        
        # Normalize
        if self.normalize:
            signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)
        
        # Extract features if requested
        features = {}
        if self.extract_features:
            if self.signal_type == 'ecg':
                features = self._extract_ecg_features(signal)
            elif self.signal_type == 'eeg':
                features = self._extract_eeg_features(signal)
        
        # Convert to tensor
        signal_tensor = torch.FloatTensor(signal.T)  # [channels, time]
        
        target = {
            'disease': sample_info['disease'],
            'recording_type': sample_info['recording_type'],
            'metadata': {
                'patient_id': sample_info['patient_id']
            }
        }
        
        if features:
            target['features'] = features
        
        return signal_tensor, target
    
    def _extract_ecg_features(self, signal: np.ndarray) -> Dict[str, Any]:
        """Extract ECG-specific features"""
        features = {}
        
        # Heart rate variability metrics
        try:
            from scipy.signal import find_peaks
            
            # Find R peaks (simplified)
            peaks, _ = find_peaks(signal[:, 0], distance=self.sampling_rate*0.6)
            
            if len(peaks) > 1:
                # RR intervals
                rr_intervals = np.diff(peaks) / self.sampling_rate * 1000  # ms
                
                features['mean_hr'] = 60000 / np.mean(rr_intervals)  # bpm
                features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))
                features['sdnn'] = np.std(rr_intervals)
            else:
                features['mean_hr'] = 0
                features['rmssd'] = 0
                features['sdnn'] = 0
                
        except Exception as e:
            logger.warning(f"ECG feature extraction failed: {e}")
            features = {'mean_hr': 0, 'rmssd': 0, 'sdnn': 0}
        
        return features
    
    def _extract_eeg_features(self, signal: np.ndarray) -> Dict[str, Any]:
        """Extract EEG-specific features"""
        features = {}
        
        # Frequency band powers
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        for channel_idx in range(signal.shape[1]):
            channel_signal = signal[:, channel_idx]
            
            # Compute power spectral density
            freqs, psd = scipy_signal.welch(
                channel_signal,
                fs=self.sampling_rate,
                nperseg=min(256, len(channel_signal))
            )
            
            for band_name, (low, high) in bands.items():
                # Find frequency range
                idx = np.logical_and(freqs >= low, freqs <= high)
                # Compute band power
                band_power = np.trapz(psd[idx], freqs[idx])
                features[f'ch{channel_idx}_{band_name}'] = band_power
        
        return features


class StreamingBiomarkerDataset(IterableDataset):
    """Dataset for streaming/real-time biomarker data"""
    
    def __init__(self,
                 data_source: Any,  # Can be a file, socket, or generator
                 buffer_size: int = 1000,
                 window_size: int = 100,
                 stride: int = 50,
                 transform: Optional[Callable] = None):
        
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.buffer = []
    
    def __iter__(self):
        """Iterate over streaming data"""
        for data_point in self.data_source:
            self.buffer.append(data_point)
            
            # Maintain buffer size
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
            
            # Yield windows
            if len(self.buffer) >= self.window_size:
                window = self.buffer[-self.window_size:]
                
                # Convert to tensor
                window_tensor = torch.FloatTensor(window)
                
                if self.transform:
                    window_tensor = self.transform(window_tensor)
                
                yield window_tensor