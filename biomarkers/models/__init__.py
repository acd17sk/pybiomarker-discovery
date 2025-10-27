"""Biomarker models module"""

from biomarkers.core.base import BiomarkerModel

# Voice models
from biomarkers.models.voice.speech_biomarker import VoiceBiomarkerModel
from biomarkers.models.voice.voice_disease_analyzers import (
    ParkinsonSpeechAnalyzer,
    DepressionSpeechAnalyzer,
    CognitiveDeclineSpeechAnalyzer,
    DysarthriaAnalyzer
)
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
    RawAudioProsodyExtractor,
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

# Vision models
from biomarkers.models.vision.visual_biomarker import VisualBiomarkerModel
from biomarkers.models.vision.face_analyzer import (
    FaceAnalyzer,
    FacialExpressionAnalyzer,
    FacialAsymmetryDetector,
    MicroExpressionDetector,
    BlinkAnalyzer,
    FacialMaskDetector
)
from biomarkers.models.vision.eye_tracker import (
    EyeTracker,
    GazeEstimator,
    PupilAnalyzer,
    SaccadeDetector,
    SmoothPursuitAnalyzer,
    FixationAnalyzer,
    VergenceAnalyzer
)
from biomarkers.models.vision.skin_analyzer import (
    SkinColorAnalyzer,
    SkinLesionDetector
)

from biomarkers.models.text.text_biomarker import TextBiomarkerModel
from biomarkers.models.text.linguistic_analyzer import (
    LinguisticAnalyzer,
    LexicalDiversityAnalyzer,
    SyntacticComplexityAnalyzer,
    SemanticCoherenceAnalyzer,
    DiscourseStructureAnalyzer,
    CognitiveLoadAnalyzer,
    LinguisticDeclineAnalyzer,
    TemporalAnalyzer
)

# Fusion models
from biomarkers.models.fusion.multimodal_fusion import MultiModalBiomarkerFusion
from biomarkers.models.fusion.fusion_components import (
    CrossModalAttention,
    ModalityEncoder,
    TemporalFusion,
    HierarchicalFusion
)
from biomarkers.models.fusion.uncertainty_quantification import (
    EvidentialUncertainty,
    EnsembleUncertainty,
    DropoutUncertainty
)
from biomarkers.models.fusion.attention_fusion import (
    AttentionFusion,
    MultiHeadCrossModalAttention,
    ModalitySpecificAttention,
    TemporalCrossAttention,
    GatedAttentionFusion,
    PerceiverFusion,
    TransformerFusion
)
from biomarkers.models.fusion.graph_fusion import (
    GraphFusion,
    BiomarkerGraphNetwork,
    ModalityGraphAttention,
    DynamicGraphFusion,
    HeterogeneousGraphFusion,
    TemporalGraphFusion,
    AdaptiveGraphStructure
)
    



# Discovery models
from biomarkers.models.discovery.feature_discovery import (
    AutomatedFeatureDiscovery,
    AttentionBasedDiscovery,
    BiomarkerCombinationFinder,
    FeatureInteractionNetwork,
    AdaptiveFeatureSelector,
    CrossModalFeatureDiscovery
)
from biomarkers.models.discovery.neural_architecture_search import (
    NeuralArchitectureSearch,
    DARTSSearchSpace,
    ENASController,
    BiomarkerNASController,
    ArchitectureEvaluator,
    SuperNet,
    DifferentiableArchitecture
)
from biomarkers.models.discovery.contrastive_learner import (
    ContrastiveBiomarkerLearner,
    SimCLRBiomarker,
    MoCoBiomarker,
    SupConBiomarker,
    PrototypicalBiomarker,
    TripletBiomarker,
    ContrastiveAugmentation,
    HealthyRiskContrastive
)

__all__ = [
    # Core
    'BiomarkerModel',
    
    # Voice models
    'VoiceBiomarkerModel',
    'ParkinsonSpeechAnalyzer',
    'DepressionSpeechAnalyzer',
    'CognitiveDeclineSpeechAnalyzer',
    'DysarthriaAnalyzer',
    'AcousticEncoder',
    'SpectralEncoder',
    'WaveformEncoder',
    'MelSpectrogramEncoder',
    'MFCCEncoder',
    'SincConv1d',
    'ConformerEncoder',
    'ProsodyAnalyzer',
    'RawAudioProsodyExtractor',
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
    
    # Vision models
    'VisualBiomarkerModel',
    'FaceAnalyzer',
    'FacialExpressionAnalyzer',
    'FacialAsymmetryDetector',
    'MicroExpressionDetector',
    'BlinkAnalyzer',
    'FacialMaskDetector',
    'EyeTracker',
    'GazeEstimator',
    'PupilAnalyzer',
    'SaccadeDetector',
    'SmoothPursuitAnalyzer',
    'FixationAnalyzer',
    'VergenceAnalyzer',
    'SkinColorAnalyzer',
    'SkinLesionDetector',

    # Text models
    'TextBiomarkerModel',
    'LinguisticAnalyzer',
    'LexicalDiversityAnalyzer',
    'SyntacticComplexityAnalyzer',
    'SemanticCoherenceAnalyzer',
    'DiscourseStructureAnalyzer',
    'CognitiveLoadAnalyzer',
    'LinguisticDeclineAnalyzer',
    'TemporalAnalyzer',
    
    # Fusion models
    'MultiModalBiomarkerFusion',
    'CrossModalAttention',
    'ModalityEncoder',
    'TemporalFusion',
    'HierarchicalFusion',
    'EvidentialUncertainty',
    'EnsembleUncertainty',
    'DropoutUncertainty',
    'AttentionFusion',
    'MultiHeadCrossModalAttention',
    'ModalitySpecificAttention',
    'TemporalCrossAttention',
    'GatedAttentionFusion',
    'PerceiverFusion',
    'TransformerFusion',
    'GraphFusion',
    'BiomarkerGraphNetwork',
    'ModalityGraphAttention',
    'DynamicGraphFusion',
    'HeterogeneousGraphFusion',
    'TemporalGraphFusion',
    'AdaptiveGraphStructure',
    
    # Discovery models
    'AutomatedFeatureDiscovery',
    'AttentionBasedDiscovery',
    'BiomarkerCombinationFinder',
    'FeatureInteractionNetwork',
    'AdaptiveFeatureSelector',
    'CrossModalFeatureDiscovery',
    'NeuralArchitectureSearch',
    'DARTSSearchSpace',
    'ENASController',
    'BiomarkerNASController',
    'ArchitectureEvaluator',
    'SuperNet',
    'DifferentiableArchitecture',
    'ContrastiveBiomarkerLearner',
    'SimCLRBiomarker',
    'MoCoBiomarker',
    'SupConBiomarker',
    'PrototypicalBiomarker',
    'TripletBiomarker',
    'ContrastiveAugmentation',
    'HealthyRiskContrastive'
]