"""Prosodic analysis modules for voice biomarkers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

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
        self.cepstral_analyzer = CepstralAnalyzer(input_dim, hidden_dim // 2)
        self.articulation_analyzer = ArticulationAnalyzer(input_dim, hidden_dim // 2)
        
        # Feature fusion
        fusion_input_dim = hidden_dim * 3  # 6 analyzers * hidden_dim/2
        
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
        cepstral_features = self.cepstral_analyzer(features)
        articulation_features = self.articulation_analyzer(features)
        
        # Concatenate all features
        combined = torch.cat([
            f0_features,
            rhythm_features,
            intensity_features,
            spectral_features,
            cepstral_features,
            articulation_features
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
            'cepstral_features': cepstral_features,
            'articulation_features': articulation_features,
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
            nn.ReLU(),
            nn.Linear(32, 4)  # dysrhythmia_score, irregularity, hesitation, repetition
        )
        
        self.output_proj = nn.Linear(128 + 8 + 4, output_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Analyze rhythm patterns"""
        rhythm_features = self.rhythm_net(features)
        
        # Compute rhythm metrics
        metrics = self.rhythm_metrics(rhythm_features)
        
        # Detect dysrhythmia
        dysrhythmia_scores = self.dysrhythmia(rhythm_features)
        dysrhythmia_scores = torch.sigmoid(dysrhythmia_scores)
        
        # Combine features
        combined = torch.cat([
            rhythm_features,
            metrics,
            dysrhythmia_scores
        ], dim=-1)
        
        return self.output_proj(combined)[:, :features.shape[-1]//4]


class IntensityAnalyzer(nn.Module):
    """Analyze speech intensity and energy patterns"""
    
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        
        self.intensity_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Intensity statistics predictor
        self.intensity_stats = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # mean, std, range, dynamic_range, peak_intensity, variability, modulation_index, stability
        )
        
        # Voice quality indicators based on intensity
        self.voice_quality = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # breathiness, roughness, strain, weakness, loudness_decay, effort
        )
        
        # Parkinson's-specific intensity features
        self.pd_intensity_features = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # monoloudness, reduced_variation, fading, vocal_effort
        )
        
        self.output_proj = nn.Linear(128 + 8 + 6 + 4, output_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Analyze intensity patterns"""
        intensity_features = self.intensity_net(features)
        
        # Compute intensity statistics
        stats = self.intensity_stats(intensity_features)
        
        # Analyze voice quality indicators
        quality = self.voice_quality(intensity_features)
        quality = torch.sigmoid(quality)  # Normalize to [0, 1]
        
        # Extract PD-specific intensity features
        pd_features = self.pd_intensity_features(intensity_features)
        pd_features = torch.sigmoid(pd_features)
        
        # Combine all intensity-related features
        combined = torch.cat([
            intensity_features,
            stats,
            quality,
            pd_features
        ], dim=-1)
        
        return self.output_proj(combined)[:, :features.shape[-1]//4]


class SpectralTiltAnalyzer(nn.Module):
    """Analyze spectral tilt and spectral balance characteristics"""
    
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        
        self.spectral_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Spectral tilt metrics
        self.tilt_metrics = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # overall_tilt, low_freq_energy, mid_freq_energy, high_freq_energy, h1_h2, spectral_slope, cepstral_peak
        )
        
        # Spectral balance indicators
        self.spectral_balance = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # balance_ratio, spectral_centroid, spectral_spread, spectral_flatness, spectral_rolloff
        )
        
        # Voice pathology indicators from spectral features
        self.pathology_indicators = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # hoarseness, aspiration, hyponasality, hypernasality, tension, weakness_indicator
        )
        
        # Neurological disorder-specific spectral features
        self.neuro_spectral_features = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # reduced_harmonics, spectral_noise, formant_clarity, harmonic_richness, spectral_decay
        )
        
        self.output_proj = nn.Linear(128 + 7 + 5 + 6 + 5, output_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Analyze spectral tilt and balance"""
        spectral_features = self.spectral_net(features)
        
        # Compute spectral tilt metrics
        tilt = self.tilt_metrics(spectral_features)
        
        # Analyze spectral balance
        balance = self.spectral_balance(spectral_features)
        
        # Detect pathology indicators
        pathology = self.pathology_indicators(spectral_features)
        pathology = torch.sigmoid(pathology)
        
        # Extract neurological disorder-specific features
        neuro_features = self.neuro_spectral_features(spectral_features)
        neuro_features = torch.sigmoid(neuro_features)
        
        # Combine all spectral features
        combined = torch.cat([
            spectral_features,
            tilt,
            balance,
            pathology,
            neuro_features
        ], dim=-1)
        
        return self.output_proj(combined)[:, :features.shape[-1]//4]


class CepstralAnalyzer(nn.Module):
    """Analyze cepstral features for voice quality assessment"""
    
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        
        self.cepstral_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Cepstral peak prominence (CPP) estimator
        self.cpp_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # cpp_value, cpp_stability, harmonic_strength
        )
        
        # Cepstral-based voice quality
        self.voice_quality_cepstral = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # periodicity, aperiodicity, noise_to_harmonics, voice_break_probability
        )
        
        self.output_proj = nn.Linear(128 + 3 + 4, output_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Analyze cepstral characteristics"""
        cepstral_features = self.cepstral_net(features)
        
        # Estimate CPP
        cpp = self.cpp_estimator(cepstral_features)
        
        # Assess voice quality
        quality = self.voice_quality_cepstral(cepstral_features)
        quality = torch.sigmoid(quality)
        
        # Combine features
        combined = torch.cat([
            cepstral_features,
            cpp,
            quality
        ], dim=-1)
        
        return self.output_proj(combined)[:, :features.shape[-1]//4]


class ArticulationAnalyzer(nn.Module):
    """Analyze articulatory precision and characteristics"""
    
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        
        self.articulation_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Articulation precision metrics
        self.precision_metrics = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # consonant_precision, vowel_precision, coarticulation, speech_clarity, mumbling_score, slurring
        )
        
        # Dysarthria indicators
        self.dysarthria_indicators = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)  # spastic, flaccid, ataxic, hypokinetic, hyperkinetic, mixed, unilateral_upper_motor
        )
        
        # Phoneme-specific analysis
        self.phoneme_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # stop_consonants, fricatives, nasals, liquids, glides
        )
        
        self.output_proj = nn.Linear(128 + 6 + 7 + 5, output_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Analyze articulation patterns"""
        articulation_features = self.articulation_net(features)
        
        # Compute precision metrics
        precision = self.precision_metrics(articulation_features)
        precision = torch.sigmoid(precision)
        
        # Detect dysarthria type
        dysarthria = self.dysarthria_indicators(articulation_features)
        dysarthria = F.softmax(dysarthria, dim=-1)
        
        # Analyze phoneme-specific features
        phoneme_features = self.phoneme_analyzer(articulation_features)
        phoneme_features = torch.sigmoid(phoneme_features)
        
        # Combine features
        combined = torch.cat([
            articulation_features,
            precision,
            dysarthria,
            phoneme_features
        ], dim=-1)
        
        return self.output_proj(combined)[:, :features.shape[-1]//4]