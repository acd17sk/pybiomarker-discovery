"""Complete multi-modal fusion with advanced uncertainty quantification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from biomarkers.core.base import BiomarkerModel


class MultiModalBiomarkerFusion(BiomarkerModel):
    """
    Advanced multi-modal fusion model combining attention-based and graph-based fusion
    with comprehensive uncertainty quantification for biomarker discovery
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Store modality configuration
        self.modality_dims = config.get('modality_dims', {
            'voice': 512,
            'movement': 256,
            'vision': 512,
            'text': 768
        })
        self.fusion_dim = config.get('fusion_dim', 512)
        self.num_diseases = config.get('num_diseases', 10)
        self.dropout = config.get('dropout', 0.3)
        self.fusion_strategy = config.get('fusion_strategy', 'hybrid')  # 'attention', 'graph', 'hybrid'
        self.use_uncertainty = config.get('use_uncertainty', True)
        self.uncertainty_method = config.get('uncertainty_method', 'evidential')  # 'evidential', 'ensemble', 'dropout'
        
        super().__init__(config)
        
    def _build_model(self):
        """Build complete multi-modal fusion architecture"""
        from biomarkers.models.fusion.attention_fusion import AttentionFusion
        from biomarkers.models.fusion.graph_fusion import GraphFusion
        from biomarkers.models.fusion.fusion_components import (
            CrossModalAttention, TemporalFusion, HierarchicalFusion
        )
        from biomarkers.models.fusion.uncertainty_quantification import (
            EvidentialUncertainty, EnsembleUncertainty, DropoutUncertainty
        )

        
        # Modality projectors to common dimension
        self.projectors = nn.ModuleDict({
            modality: nn.Linear(dim, self.fusion_dim)
            for modality, dim in self.modality_dims.items()
        })
        
        # Attention-based fusion
        if self.fusion_strategy in ['attention', 'hybrid']:
            self.attention_fusion = AttentionFusion(
                modality_dims=self.modality_dims,
                fusion_dim=self.fusion_dim,
                num_heads=8,
                num_layers=3,
                dropout=self.dropout,
                use_gating=True,
                use_perceiver=True
            )
        
        # Graph-based fusion
        if self.fusion_strategy in ['graph', 'hybrid']:
            self.graph_fusion = GraphFusion(
                modality_dims=self.modality_dims,
                fusion_dim=self.fusion_dim,
                num_graph_layers=3,
                num_heads=8,
                dropout=self.dropout,
                use_heterogeneous=True,
                use_temporal=True,
                use_adaptive=True
            )
        
        # Hierarchical fusion (combine attention and graph)
        if self.fusion_strategy == 'hybrid':
            self.hierarchical_fusion = HierarchicalFusion(
                fusion_dim=self.fusion_dim,
                dropout=self.dropout
            )
        
        # Temporal fusion for sequential data
        self.temporal_fusion = TemporalFusion(
            fusion_dim=self.fusion_dim,
            num_layers=2,
            dropout=self.dropout
        )
        
        # Cross-modal attention blocks
        self.cross_attention_blocks = nn.ModuleList([
            CrossModalAttention(self.fusion_dim, num_heads=8)
            for _ in range(2)
        ])
        
        # Disease-specific attention
        self.disease_attention = nn.Parameter(
            torch.randn(self.num_diseases, self.fusion_dim * 2)
        )
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_diseases)
        )
        
        # Uncertainty quantification
        if self.use_uncertainty:
            if self.uncertainty_method == 'evidential':
                self.uncertainty_head = EvidentialUncertainty(
                    input_dim=self.fusion_dim * 2,
                    num_classes=self.num_diseases,
                    dropout=self.dropout
                )
            elif self.uncertainty_method == 'ensemble':
                self.uncertainty_head = EnsembleUncertainty(
                    input_dim=self.fusion_dim * 2,
                    num_classes=self.num_diseases,
                    num_estimators=5,
                    dropout=self.dropout
                )
            else:  # dropout-based
                self.uncertainty_head = DropoutUncertainty(
                    input_dim=self.fusion_dim * 2,
                    num_classes=self.num_diseases,
                    dropout=self.dropout
                )
        
        # Modality importance scoring
        self.modality_scorer = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def extract_features(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract fused features from multiple modalities"""
        # Project modalities
        projected = {}
        for modality, features in modality_features.items():
            if modality in self.projectors:
                projected[modality] = self.projectors[modality](features)
        
        # Apply fusion strategy
        if self.fusion_strategy == 'attention':
            fusion_output = self.attention_fusion(projected)
            fused = fusion_output['fused_features']
            
        elif self.fusion_strategy == 'graph':
            fusion_output = self.graph_fusion(projected)
            fused = fusion_output['fused_features']
            
        else:  # hybrid
            # First apply attention fusion
            attention_output = self.attention_fusion(projected)
            attention_features = attention_output['fused_features']
            
            # Then apply graph fusion
            graph_output = self.graph_fusion(projected)
            graph_features = graph_output['fused_features']
            
            # Hierarchical fusion
            fused = self.hierarchical_fusion(attention_features, graph_features)
        
        return fused
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                return_uncertainty: bool = True,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-modal fusion
        
        Args:
            modality_features: Dictionary of modality features
            return_uncertainty: Whether to compute uncertainty estimates
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions, features, and optional uncertainty
        """
        # Extract fused features
        fused_features = self.extract_features(modality_features)
        
        # Apply cross-modal attention
        attended = fused_features
        attention_weights = []
        for cross_attn in self.cross_attention_blocks:
            attended_new, weights = cross_attn(attended, attended, attended)
            attended = attended_new
            attention_weights.append(weights)
        
        # Temporal fusion if sequential data
        if len(attended.shape) == 3:  # [B, T, D]
            temporal_output = self.temporal_fusion(attended)
            attended = temporal_output['fused']
        
        # Duplicate features for disease-specific processing
        if len(attended.shape) == 2:
            attended = torch.cat([attended, attended], dim=-1)
        
        # Disease-specific attention
        disease_attn_scores = F.softmax(
            torch.matmul(self.disease_attention, attended.T).T,
            dim=1
        )
        
        # Main classification
        disease_logits = self.classifier(attended)
        
        output = {
            'logits': disease_logits,
            'probabilities': F.softmax(disease_logits, dim=1),
            'features': fused_features,
            'disease_attention': disease_attn_scores
        }
        
        # Uncertainty quantification
        if return_uncertainty and self.use_uncertainty:
            uncertainty_output = self.uncertainty_head(attended, disease_logits)
            output.update(uncertainty_output)
        
        # Modality importance
        modality_importance = {}
        for modality, features in modality_features.items():
            if modality in self.projectors:
                projected = self.projectors[modality](features)
                if len(projected.shape) == 3:
                    projected = projected.mean(dim=1)
                importance = self.modality_scorer(projected)
                modality_importance[modality] = importance.mean()
        
        output['modality_importance'] = modality_importance
        
        if return_attention:
            output['attention_weights'] = attention_weights
        
        return output

