import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from biomarkers.core.base import BiomarkerModel

class CrossModalAttention(nn.Module):
    """Cross-modal attention for feature fusion"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Cross attention
        attn_output, _ = self.attention(query, key, value)
        query = self.norm1(query + attn_output)
        
        # FFN
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output

class MultiModalBiomarkerFusion(BiomarkerModel):
    """Fusion model for multiple biomarker modalities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.modality_dims = config.get('modality_dims', {
            'voice': 512,
            'movement': 256,
            'vision': 512,
            'text': 768
        })
        self.fusion_dim = config.get('fusion_dim', 512)
        self.num_diseases = config.get('num_diseases', 10)
        self.dropout = config.get('dropout', 0.3)
        super().__init__(config)
        
    def _build_model(self):
        # Modality projectors
        self.projectors = nn.ModuleDict({
            modality: nn.Linear(dim, self.fusion_dim)
            for modality, dim in self.modality_dims.items()
        })
        
        # Cross-modal attention blocks
        self.cross_attention_blocks = nn.ModuleList([
            CrossModalAttention(self.fusion_dim)
            for _ in range(3)
        ])
        
        # Graph neural network for biomarker interaction
        self.gnn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.fusion_dim,
                nhead=8,
                dim_feedforward=self.fusion_dim * 4,
                dropout=self.dropout
            )
            for _ in range(2)
        ])
        
        # Temporal dynamics modeling
        self.temporal_lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=self.fusion_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Disease-specific attention
        self.disease_attention = nn.Parameter(
            torch.randn(self.num_diseases, self.fusion_dim * 2)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_diseases)
        )
        
        # Uncertainty quantification
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_diseases)
        )
        
    def extract_features(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract fused features from multiple modalities"""
        # Project each modality to common dimension
        projected = {}
        for modality, features in modality_features.items():
            if modality in self.projectors:
                projected[modality] = self.projectors[modality](features)
        
        # Stack for batch processing
        stacked = torch.stack(list(projected.values()), dim=1)  # [B, M, D]
        
        # Cross-modal attention
        for attention_block in self.cross_attention_blocks:
            stacked = attention_block(stacked, stacked, stacked)
        
        # Graph neural network processing
        for gnn_layer in self.gnn_layers:
            stacked = gnn_layer(stacked)
        
        # Temporal modeling if sequence data
        if len(stacked.shape) == 4:  # [B, T, M, D]
            B, T, M, D = stacked.shape
            stacked = stacked.reshape(B, T, -1)
            temporal_features, _ = self.temporal_lstm(stacked)
            fused = temporal_features[:, -1, :]  # Take last timestep
        else:
            # Global pooling
            fused = torch.mean(stacked, dim=1)
            # Add bidirectional dimension for consistency
            fused = torch.cat([fused, fused], dim=-1)
        
        return fused
    
    def forward(self, 
                modality_features: Dict[str, torch.Tensor],
                return_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-modal fusion"""
        
        # Extract fused features
        fused_features = self.extract_features(modality_features)
        
        # Disease prediction
        disease_logits = self.classifier(fused_features)
        
        # Disease-specific attention scores
        attention_scores = F.softmax(
            torch.matmul(self.disease_attention, fused_features.T).T,
            dim=1
        )
        
        output = {
            'logits': disease_logits,
            'probabilities': F.softmax(disease_logits, dim=1),
            'features': fused_features,
            'attention_scores': attention_scores
        }
        
        if return_uncertainty:
            # Uncertainty quantification
            log_variance = self.uncertainty_head(fused_features)
            uncertainty = torch.exp(log_variance)
            output['uncertainty'] = uncertainty
            output['confidence'] = 1.0 / (1.0 + uncertainty)
        
        return output