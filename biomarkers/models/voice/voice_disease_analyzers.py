"""Disease-specific speech analyzers for voice biomarkers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ParkinsonSpeechAnalyzer(nn.Module):
    """
    Specialized analyzer for Parkinsonian speech characteristics.
    
    Detects hypokinetic dysarthria features:
    - Hypophonia (reduced loudness)
    - Monopitch (reduced pitch variability)
    - Monoloudness (reduced loudness variability)
    - Speech rate reduction
    - Imprecise articulation
    - Voice quality deterioration
    
    References:
    - Rusz et al. (2011) "Quantitative acoustic measurements for characterization of speech..."
    - Skodda et al. (2011) "Progression of dysprosody in Parkinson's disease..."
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Hypophonia detector (reduced loudness)
        self.hypophonia_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Monopitch detector (reduced pitch variability)
        self.monopitch_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Monoloudness detector (reduced loudness variability)
        self.monoloudness_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Speech rate reduction analyzer
        self.speech_rate_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Imprecise articulation analyzer
        self.articulation_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Voice quality analyzer (roughness, breathiness, strain)
        self.voice_quality_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 3)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Acoustic features [batch, input_dim]
            
        Returns:
            Dictionary with PD speech features
        """
        features = self.feature_extractor(x)
        
        hypophonia = self.hypophonia_detector(features).squeeze(-1)
        monopitch = self.monopitch_detector(features).squeeze(-1)
        monoloudness = self.monoloudness_detector(features).squeeze(-1)
        speech_rate_reduction = self.speech_rate_analyzer(features).squeeze(-1)
        imprecise_articulation = self.articulation_analyzer(features).squeeze(-1)
        voice_quality = torch.sigmoid(self.voice_quality_analyzer(features))
        
        return {
            'features': features,
            'hypophonia': hypophonia,
            'monopitch': monopitch,
            'monoloudness': monoloudness,
            'speech_rate_reduction': speech_rate_reduction,
            'imprecise_articulation': imprecise_articulation,
            'voice_quality': voice_quality,
            'voice_quality_score': voice_quality.mean(dim=-1)
        }


class DepressionSpeechAnalyzer(nn.Module):
    """
    Specialized analyzer for depression-related speech characteristics.
    
    Detects:
    - Flat affect (reduced emotional expression)
    - Reduced pitch variability
    - Slowed speech tempo
    - Extended pause duration
    - Low vocal energy
    
    References:
    - Cummins et al. (2015) "A review of depression and suicide risk assessment..."
    - Low et al. (2020) "Detection of clinical depression in adolescents' speech..."
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Flat affect detector
        self.flat_affect_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Reduced pitch variability analyzer
        self.pitch_variability_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Speech tempo analyzer
        self.speech_tempo_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Pause duration analyzer
        self.pause_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Energy analyzer
        self.energy_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Acoustic features [batch, input_dim]
            
        Returns:
            Dictionary with depression speech features
        """
        features = self.feature_extractor(x)
        
        return {
            'features': features,
            'flat_affect': self.flat_affect_detector(features).squeeze(-1),
            'reduced_pitch_variability': self.pitch_variability_analyzer(features).squeeze(-1),
            'slow_speech': self.speech_tempo_analyzer(features).squeeze(-1),
            'long_pauses': self.pause_analyzer(features).squeeze(-1),
            'low_energy': self.energy_analyzer(features).squeeze(-1)
        }


class CognitiveDeclineSpeechAnalyzer(nn.Module):
    """
    Specialized analyzer for cognitive decline speech markers.
    
    Detects:
    - Hesitations and filled pauses
    - Word-finding difficulties
    - Semantic pauses
    - Reduced syntactic complexity
    - Repetitions and false starts
    
    References:
    - Fraser et al. (2016) "Linguistic features identify Alzheimer's disease..."
    - Konig et al. (2015) "Automatic speech analysis for the assessment of patients..."
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Hesitation detector
        self.hesitation_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Word-finding difficulty analyzer
        self.word_finding_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Semantic pause analyzer
        self.semantic_pause_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Syntactic complexity analyzer
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Repetition detector
        self.repetition_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Acoustic features [batch, input_dim]
            
        Returns:
            Dictionary with cognitive decline speech features
        """
        features = self.feature_extractor(x)
        
        return {
            'features': features,
            'hesitations': self.hesitation_detector(features).squeeze(-1),
            'word_finding': self.word_finding_analyzer(features).squeeze(-1),
            'semantic_pauses': self.semantic_pause_analyzer(features).squeeze(-1),
            'reduced_complexity': self.complexity_analyzer(features).squeeze(-1),
            'repetitions': self.repetition_detector(features).squeeze(-1)
        }


class DysarthriaAnalyzer(nn.Module):
    """
    Comprehensive dysarthria analysis.
    
    Classifies dysarthria type and assesses severity:
    - Spastic (upper motor neuron)
    - Flaccid (lower motor neuron)
    - Ataxic (cerebellar)
    - Hypokinetic (Parkinson's)
    - Hyperkinetic (Huntington's, dystonia)
    - Mixed
    - Unilateral upper motor neuron
    
    References:
    - Duffy (2019) "Motor Speech Disorders"
    - Yorkston et al. (2010) "Management of Motor Speech Disorders..."
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Dysarthria type classifier (7 types)
        self.type_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 7)
        )
        
        # Severity estimator (mild to severe)
        self.severity_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Intelligibility predictor
        self.intelligibility_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Articulation precision
        self.articulation_precision = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Acoustic features [batch, input_dim]
            
        Returns:
            Dictionary with dysarthria analysis
        """
        features = self.feature_extractor(x)
        
        type_logits = self.type_classifier(features)
        
        return {
            'features': features,
            'dysarthria_type': F.softmax(type_logits, dim=-1),
            'severity': self.severity_estimator(features).squeeze(-1),
            'intelligibility': self.intelligibility_predictor(features).squeeze(-1),
            'articulation_precision': self.articulation_precision(features).squeeze(-1),
            'type_logits': type_logits
        }