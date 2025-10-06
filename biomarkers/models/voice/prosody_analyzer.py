"""Prosodic analysis modules for voice biomarkers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import math

class ProsodyAnalyzer(nn.Module):
    """Complete prosody analysis module"""
    
    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 num_prosodic_features: int = 20,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_prosodic_features = num_prosodic_features
        
        # Sub-analyzers
        self.f0_extractor = F0Extractor(input_dim, hidden_dim // 2)
        self.rhythm_analyzer = RhythmAnalyzer(input_dim, hidden_dim // 2)
        self.intensity_analyzer = IntensityAnalyzer(input_dim, hidden_dim // 2)
        self.spectral_tilt = SpectralTiltAnalyzer(input_dim, hidden_dim // 2)
        
        # Feature fusion
        fusion_input_dim = hidden_dim * 2  # 4 analyzers * hidden_dim/2
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_prosodic_features)
        )
        
        # Prosodic pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(num_prosodic_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 prosodic patterns
        )
    
    def forward(self, features: torch.Tensor, 
                raw_audio: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Encoded features from acoustic encoder [batch, input_dim]
            raw_audio: Optional raw audio for direct prosodic extraction
        """
        # Extract individual prosodic components
        f0_features = self.f0_extractor(features)
        rhythm_features = self.rhythm_analyzer(features)
        intensity_features = self.intensity_analyzer(features)
        spectral_features = self.spectral_tilt(features)
        
        # Concatenate all features
        combined = torch.cat([
            f0_features,
            rhythm_features,
            intensity_features,
            spectral_features
        ], dim=-1)
        
        # Fuse features
        prosodic_features = self.feature_fusion(combined)
        
        # Classify prosodic patterns
        pattern_logits = self.pattern_classifier(prosodic_features)
        
        return {
            'prosodic_features': prosodic_features,
            'f0_features': f0_features,
            'rhythm_features': rhythm_features,
            'intensity_features': intensity_features,
            'spectral_tilt': spectral_features,
            'pattern_logits': pattern_logits,
            'pattern_probs': F.softmax(pattern_logits, dim=-1)
        }


class F0Extractor(nn.Module):
    """Fundamental frequency (F0) extraction and analysis"""
    
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        
        self.f0_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # F0 statistics predictor
        self.f0_stats = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # mean, std, range, jitter, shimmer, HNR
        )
        
        # Tremor detector specific to F0
        self.tremor_detector = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # tremor_presence, frequency, amplitude
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Extract F0-related features"""
        f0_features = self.f0_net(features)
        
        # Compute F0 statistics
        f0_statistics = self.f0_stats(f0_features)
        
        # Detect tremor
        tremor_params = self.tremor_detector(f0_features)
        tremor_params = torch.sigmoid(tremor_params)  # Normalize to [0, 1]
        
        # Combine all F0-related features
        combined = torch.cat([
            f0_features,
            f0_statistics,
            tremor_params
        ], dim=-1)
        
        return combined[:, :features.shape[-1]//4]  # Return quarter of input dim


class RhythmAnalyzer(nn.Module):
    """Analyze speech rhythm and timing patterns"""
    
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        
        self.rhythm_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Rhythm metrics predictor
        self.rhythm_metrics = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # speech_rate, pause_ratio, syllable_duration, etc.
        )
        
        # Dysrhythmia detector
        self.dysrhythmia = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),