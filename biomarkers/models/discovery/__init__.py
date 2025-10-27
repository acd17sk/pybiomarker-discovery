"""Discovery models for automated biomarker feature discovery"""

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
    # Feature Discovery
    'AutomatedFeatureDiscovery',
    'AttentionBasedDiscovery',
    'BiomarkerCombinationFinder',
    'FeatureInteractionNetwork',
    'AdaptiveFeatureSelector',
    'CrossModalFeatureDiscovery',
    
    # Neural Architecture Search
    'NeuralArchitectureSearch',
    'DARTSSearchSpace',
    'ENASController',
    'BiomarkerNASController',
    'ArchitectureEvaluator',
    'SuperNet',
    'DifferentiableArchitecture',
    
    # Contrastive Learning
    'ContrastiveBiomarkerLearner',
    'SimCLRBiomarker',
    'MoCoBiomarker',
    'SupConBiomarker',
    'PrototypicalBiomarker',
    'TripletBiomarker',
    'ContrastiveAugmentation',
    'HealthyRiskContrastive'
]