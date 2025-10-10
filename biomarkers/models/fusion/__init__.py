"""Multi-modal fusion models for biomarker discovery"""

from biomarkers.models.fusion.multimodal_fusion import (
    MultiModalBiomarkerFusion
)
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

__all__ = [
    # Main fusion model
    'MultiModalBiomarkerFusion',

    # Fusion Components
    'CrossModalAttention',
    'ModalityEncoder',
    'TemporalFusion',
    'HierarchicalFusion',

    # Uncertainty quantification
    'EvidentialUncertainty',
    'EnsembleUncertainty',
    'DropoutUncertainty',
    
    # Attention-based fusion
    'AttentionFusion',
    'MultiHeadCrossModalAttention',
    'ModalitySpecificAttention',
    'TemporalCrossAttention',
    'GatedAttentionFusion',
    'PerceiverFusion',
    'TransformerFusion',
    
    # Graph-based fusion
    'GraphFusion',
    'BiomarkerGraphNetwork',
    'ModalityGraphAttention',
    'DynamicGraphFusion',
    'HeterogeneousGraphFusion',
    'TemporalGraphFusion',
    'AdaptiveGraphStructure'
]