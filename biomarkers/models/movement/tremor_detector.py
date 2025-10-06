"""Tremor detection and analysis modules"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

class TremorDetector(nn.Module):
    """Complete tremor detection and characterization module"""
    def __init__(self,
                 input_channels: int = 6,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        # Sub-components
        self.freq_analyzer = FrequencyAnalyzer(input_channels, hidden_dim // 2)
        self.amp_analyzer = AmplitudeAnalyzer(input_channels, hidden_dim // 2)
        self.tremor_classifier = TremorClassifier(hidden_dim)
        self.tremor_characterizer = TremorCharacterizer(hidden_dim)
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        # Clinical tremor assessment
        self.clinical_assessment = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # UPDRS tremor items
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:
        x: Sensor data [batch, channels, time]
        """
        # Analyze frequency components
        freq_features = self.freq_analyzer(x)
        amp_features = self.amp_analyzer(x)
        # Combine features
        combined = torch.cat([
            freq_features['features'],
            amp_features['features']
        ], dim=-1)
        # Fuse features
        fused = self.feature_fusion(combined)
        # Classify tremor type
        tremor_class = self.tremor_classifier(combined)
        # Characterize tremor
        characteristics = self.tremor_characterizer(combined)
        # Clinical assessment
        clinical_scores = self.clinical_assessment(fused)
        return {
            'tremor_features': fused,
            'tremor_type': tremor_class['type_probs'],
            'tremor_severity': tremor_class['severity'],
            'frequency': freq_features['dominant_freq'],
            'frequency_band': freq_features['frequency_band'],
            'amplitude': amp_features['amplitude'],
            'amplitude_variability': amp_features['variability'],
            'characteristics': characteristics,
            'clinical_scores': clinical_scores,
            'tremor_present': tremor_class['tremor_presence']
        }

class FrequencyAnalyzer(nn.Module):
    """Analyze tremor frequency characteristics"""
    def __init__(self, input_channels: int = 6, output_dim: int = 128):
        super().__init__()
        # Spectral analysis network
        self.spectral_cnn = nn.Sequential(
            # First level: capture high-frequency components
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Second level: capture mid-frequency
            nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Third level: capture low-frequency
            nn.Conv1d(128, 256, kernel_size=21, stride=1, padding=10),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        # Frequency band analyzer
        self.band_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool1d(50),
            nn.Flatten(),
            nn.Linear(256 * 50, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        # Frequency estimators
        self.freq_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Dominant frequency in Hz
        )
        # Frequency band classifier
        self.band_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Rest (4-6Hz), Postural (4-8Hz), Kinetic (6-12Hz), High (>12Hz)
        )
        # Harmonics detector
        self.harmonics_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # fundamental, first_harmonic, second_harmonic
        )
        self.output_proj = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze frequency components"""
        # Spectral analysis
        spectral_features = self.spectral_cnn(x)
        # Analyze frequency bands
        band_features = self.band_analyzer(spectral_features)
        # Estimate dominant frequency
        dominant_freq = self.freq_estimator(band_features).squeeze(-1)
        dominant_freq = torch.abs(dominant_freq) * 20  # Scale to 0-20 Hz range
        # Classify frequency band
        band_logits = self.band_classifier(band_features)
        band_probs = F.softmax(band_logits, dim=-1)
        # Detect harmonics
        harmonics = self.harmonics_detector(band_features)
        features = self.output_proj(band_features)
        return {
            'features': features,
            'dominant_freq': dominant_freq,
            'frequency_band': band_probs,
            'harmonics': harmonics,
            'spectral_features': band_features
        }

class AmplitudeAnalyzer(nn.Module):
    """Analyze tremor amplitude and patterns"""
    def __init__(self, input_channels: int = 6, output_dim: int = 128):
        super().__init__()
        # Amplitude extraction network
        self.amplitude_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        # Temporal patterns
        self.temporal_analyzer = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        # Amplitude metrics
        self.amplitude_metrics = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # mean, peak, RMS, variability, asymmetry, intermittency
        )
        # Pattern analyzer
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # constant, intermittent, crescendo, decrescendo, irregular
        )
        self.output_proj = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze amplitude characteristics"""
        # Extract amplitude features
        amp_features = self.amplitude_extractor(x)
        amp_features = amp_features.transpose(1, 2)
        # Temporal analysis
        temporal_out, _ = self.temporal_analyzer(amp_features)
        pooled = torch.mean(temporal_out, dim=1)
        # Calculate metrics
        metrics = self.amplitude_metrics(pooled)
        patterns = self.pattern_analyzer(pooled)
        patterns = F.softmax(patterns, dim=-1)
        features = self.output_proj(pooled)
        return {
            'features': features,
            'amplitude': torch.abs(metrics[:, 0]),  # Mean amplitude
            'peak_amplitude': torch.abs(metrics[:, 1]),
            'rms_amplitude': torch.abs(metrics[:, 2]),
            'variability': torch.sigmoid(metrics[:, 3]),
            'asymmetry': torch.tanh(metrics[:, 4]),
            'intermittency': torch.sigmoid(metrics[:, 5]),
            'patterns': patterns
        }

class TremorClassifier(nn.Module):
    """Classify tremor type and severity"""
    def __init__(self, input_dim: int = 256):
        super().__init__()
        # Tremor presence detector
        self.presence_detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # Tremor type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # Rest, Action, Postural, Kinetic, Intention, Mixed
        )
        # Severity estimator (0-4 scale like UPDRS)
        self.severity_estimator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Laterality detector
        self.laterality_detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Bilateral, Left-dominant, Right-dominant
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Classify tremor type and severity"""
        # Detect presence
        presence = self.presence_detector(x).squeeze(-1)
        # Classify type
        type_logits = self.type_classifier(x)
        type_probs = F.softmax(type_logits, dim=-1)
        # Estimate severity
        severity = self.severity_estimator(x).squeeze(-1)
        severity = torch.sigmoid(severity) * 4  # Scale to 0-4
        # Detect laterality
        laterality_logits = self.laterality_detector(x)
        laterality_probs = F.softmax(laterality_logits, dim=-1)
        return {
            'tremor_presence': presence,
            'type_logits': type_logits,
            'type_probs': type_probs,
            'severity': severity,
            'laterality': laterality_probs
        }

class TremorCharacterizer(nn.Module):
    """Detailed tremor characterization"""
    def __init__(self, input_dim: int = 256):
        super().__init__()
        # Regularity analyzer
        self.regularity_analyzer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # regularity, rhythmicity, consistency
        )
        # Context analyzer
        self.context_analyzer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # rest, posture, action, stress, fatigue, medication_state
        )
        # Progression analyzer
        self.progression_analyzer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # stable, improving, worsening, fluctuating
        )
        # Re-emergent tremor detector
        self.reemergent_detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # latency, presence
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Characterize tremor in detail"""
        # Analyze regularity
        regularity = self.regularity_analyzer(x)
        regularity = torch.sigmoid(regularity)
        # Analyze context
        context = self.context_analyzer(x)
        context = torch.sigmoid(context)
        # Analyze progression
        progression_logits = self.progression_analyzer(x)
        progression = F.softmax(progression_logits, dim=-1)
        # Detect re-emergent tremor
        reemergent = self.reemergent_detector(x)
        reemergent_latency = torch.abs(reemergent[:, 0]) * 10  # Scale to seconds
        reemergent_presence = torch.sigmoid(reemergent[:, 1])
        return {
            'regularity': regularity[:, 0],
            'rhythmicity': regularity[:, 1],
            'consistency': regularity[:, 2],
            'context_factors': context,
            'progression': progression,
            'reemergent_latency': reemergent_latency,
            'reemergent_presence': reemergent_presence
        }

class EssentialTremorAnalyzer(nn.Module):
    """Specialized analyzer for essential tremor"""
    def __init__(self, input_dim: int = 256):
        super().__init__()
        # ET-specific features
        self.et_features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # kinetic_dominance, intention_component, spiral_severity, handwriting_impact, voice_tremor
        )
        # Task-specific analysis
        self.task_analyzer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # pouring, drinking, eating, writing, spiral_drawing, finger_nose
        )
        # Alcohol responsiveness predictor
        self.alcohol_response = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze essential tremor characteristics"""
        et_features = self.et_features(x)
        et_features = torch.sigmoid(et_features)
        task_severity = self.task_analyzer(x)
        task_severity = torch.sigmoid(task_severity) * 4  # Scale to 0-4
        alcohol_resp = self.alcohol_response(x).squeeze(-1)
        return {
            'et_features': et_features,
            'task_severity': task_severity,
            'alcohol_responsiveness': alcohol_resp
        }

class ParkinsonianTremorAnalyzer(nn.Module):
    """Specialized analyzer for Parkinsonian tremor"""
    def __init__(self, input_dim: int = 256):
        super().__init__()
        # PD tremor features
        self.pd_features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # rest_dominant, pill_rolling, asymmetry, reemergent, suppression_with_action, amplitude_decrement
        )
        # UPDRS tremor items
        self.updrs_items = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # face, RUE, LUE, RLE, LLE
        )
        # Medication state predictor
        self.medication_state = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # ON, OFF, Dyskinetic
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze Parkinsonian tremor"""
        pd_features = self.pd_features(x)
        pd_features = torch.sigmoid(pd_features)
        updrs_scores = self.updrs_items(x)
        updrs_scores = torch.sigmoid(updrs_scores) * 4  # Scale to 0-4
        med_state_logits = self.medication_state(x)
        med_state = F.softmax(med_state_logits, dim=-1)
        return {
            'pd_features': pd_features,
            'updrs_scores': updrs_scores,
            'medication_state': med_state
        }