"""Biomarker models module"""

from biomarkers.core.base import BiomarkerModel

# Voice models
from biomarkers.models.voice.speech_biomarker import VoiceBiomarkerModel
from biomarkers.models.voice.acoustic_encoder import (
    AcousticEncoder,
    SpectralEncoder,
    WaveformEncoder,
    MelSpectrogramEncoder,
    MFCCEncoder,
    SincConv1d,
    ConformerEncoder
)
from biomarkers.models.voice.prosody_analyzer import (
    ProsodyAnalyzer,
    F0Extractor,
    RhythmAnalyzer,
    IntensityAnalyzer,
    SpectralTiltAnalyzer,
    CepstralAnalyzer,
    ArticulationAnalyzer
)

# Movement models
from biomarkers.models.movement.movement_biomarker import MovementBiomarkerModel
from biomarkers.models.movement.gait_analyzer import (
    GaitAnalyzer,
    StepDetector,
    GaitCycleAnalyzer,
    BalanceAnalyzer,
    FreezingOfGaitDetector,
    DualTaskGaitAnalyzer
)
from biomarkers.models.movement.tremor_detector import (
    TremorDetector,
    TremorClassifier,
    FrequencyAnalyzer,
    AmplitudeAnalyzer,
    TremorCharacterizer,
    EssentialTremorAnalyzer,
    ParkinsonianTremorAnalyzer
)

# Fusion models
from biomarkers.models.fusion.multimodal_fusion import (
    MultiModalBiomarkerFusion,
    CrossModalAttention
)

# Discovery models
from biomarkers.models.discovery.feature_discovery import (
    AutomatedFeatureDiscovery,
    NeuralArchitectureSearch
)

__all__ = [
    # Core
    'BiomarkerModel',
    
    # Voice models
    'VoiceBiomarkerModel',
    'AcousticEncoder',
    'SpectralEncoder',
    'WaveformEncoder',
    'MelSpectrogramEncoder',
    'MFCCEncoder',
    'SincConv1d',
    'ConformerEncoder',
    'ProsodyAnalyzer',
    'F0Extractor',
    'RhythmAnalyzer',
    'IntensityAnalyzer',
    'SpectralTiltAnalyzer',
    'CepstralAnalyzer',
    'ArticulationAnalyzer',
    
    # Movement models
    'MovementBiomarkerModel',
    'GaitAnalyzer',
    'StepDetector',
    'GaitCycleAnalyzer',
    'BalanceAnalyzer',
    'FreezingOfGaitDetector',
    'DualTaskGaitAnalyzer',
    'TremorDetector',
    'TremorClassifier',
    'FrequencyAnalyzer',
    'AmplitudeAnalyzer',
    'TremorCharacterizer',
    'EssentialTremorAnalyzer',
    'ParkinsonianTremorAnalyzer',
    
    # Fusion models
    'MultiModalBiomarkerFusion',
    'CrossModalAttention',
    
    # Discovery models
    'AutomatedFeatureDiscovery',
    'NeuralArchitectureSearch'
]