"""Advanced attention-based fusion mechanisms for multi-modal biomarker integration"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class AttentionFusion(nn.Module):
    """
    Main attention-based fusion module combining multiple attention mechanisms
    for optimal multi-modal biomarker integration
    """
    
    def __init__(self,
                 modality_dims: Dict[str, int],
                 fusion_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 use_gating: bool = True,
                 use_perceiver: bool = False):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.modality_names = list(modality_dims.keys())
        
        # Modality projections to common dimension
        self.modality_projectors = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for modality, dim in modality_dims.items()
        })
        
        # Multi-head cross-modal attention
        self.cross_modal_attention = nn.ModuleList([
            MultiHeadCrossModalAttention(
                dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Modality-specific attention (self-attention within each modality)
        self.modality_specific_attention = nn.ModuleDict({
            modality: ModalitySpecificAttention(
                dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for modality in self.modality_names
        })
        
        # Temporal cross-attention for sequential data
        self.temporal_cross_attention = TemporalCrossAttention(
            dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Gated attention fusion
        if use_gating:
            self.gated_fusion = GatedAttentionFusion(
                num_modalities=len(modality_dims),
                dim=fusion_dim,
                dropout=dropout
            )
        else:
            self.gated_fusion = None
        
        # Perceiver-based fusion for handling large inputs
        if use_perceiver:
            self.perceiver_fusion = PerceiverFusion(
                dim=fusion_dim,
                num_latents=32,
                num_heads=num_heads,
                depth=2,
                dropout=dropout
            )
        else:
            self.perceiver_fusion = None
        
        # Transformer fusion layers
        self.transformer_fusion = TransformerFusion(
            dim=fusion_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout
        )
        
        # Modality importance learning
        self.modality_importance = nn.Parameter(
            torch.ones(len(modality_dims))
        )
        
        # Final fusion projection
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * len(modality_dims), fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                modality_masks: Optional[Dict[str, torch.Tensor]] = None,
                return_attention_weights: bool = False) -> Dict[str, torch.Tensor]:
        """
        Fuse multi-modal features using advanced attention mechanisms
        
        Args:
            modality_features: Dictionary of modality features
            modality_masks: Optional masks for each modality
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dictionary containing fused features and attention weights
        """
        batch_size = next(iter(modality_features.values())).shape[0]
        
        # Project all modalities to common dimension
        projected_features = {}
        for modality, features in modality_features.items():
            if modality in self.modality_projectors:
                # Handle both 2D [B, D] and 3D [B, T, D] inputs
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)  # Add time dimension
                
                projected = self.modality_projectors[modality](features)
                projected_features[modality] = projected
        
        # Step 1: Modality-specific self-attention
        modality_attended = {}
        for modality, features in projected_features.items():
            attended = self.modality_specific_attention[modality](features)
            modality_attended[modality] = attended
        
        # Step 2: Cross-modal attention across all modality pairs
        cross_modal_outputs = {}
        attention_weights = {}
        
        for layer_idx, cross_attention in enumerate(self.cross_modal_attention):
            layer_outputs = {}
            layer_weights = {}
            
            # Perform pairwise cross-attention
            for i, mod1 in enumerate(self.modality_names):
                if mod1 not in modality_attended:
                    continue
                    
                attended_features = []
                weights_list = []
                
                for j, mod2 in enumerate(self.modality_names):
                    if mod2 not in modality_attended or i == j:
                        continue
                    
                    # Cross-attention: mod1 attends to mod2
                    output, weights = cross_attention(
                        modality_attended[mod1],
                        modality_attended[mod2],
                        modality_attended[mod2]
                    )
                    attended_features.append(output)
                    weights_list.append(weights)
                
                if attended_features:
                    # Aggregate cross-modal information
                    layer_outputs[mod1] = torch.stack(attended_features).mean(dim=0)
                    layer_weights[mod1] = weights_list
            
            # Update modality features with cross-modal information
            for modality in layer_outputs:
                modality_attended[modality] = modality_attended[modality] + layer_outputs[modality]
            
            cross_modal_outputs[f'layer_{layer_idx}'] = layer_outputs
            attention_weights[f'layer_{layer_idx}'] = layer_weights
        
        # Step 3: Temporal cross-attention for temporal dependencies
        temporal_features = {}
        for modality, features in modality_attended.items():
            temporal_output = self.temporal_cross_attention(features)
            temporal_features[modality] = temporal_output
        
        # Step 4: Gated fusion
        if self.gated_fusion is not None:
            gated_features = self.gated_fusion(temporal_features)
        else:
            gated_features = temporal_features
        
        # Step 5: Perceiver fusion (optional)
        if self.perceiver_fusion is not None:
            # Stack all modalities for perceiver
            stacked_features = torch.cat([
                gated_features[mod].mean(dim=1, keepdim=True) 
                for mod in self.modality_names 
                if mod in gated_features
            ], dim=1)
            
            perceiver_output = self.perceiver_fusion(stacked_features)
            
            # Distribute perceiver output back to modalities
            for idx, modality in enumerate([m for m in self.modality_names if m in gated_features]):
                gated_features[modality] = gated_features[modality] + perceiver_output[:, idx:idx+1]
        
        # Step 6: Transformer fusion
        # Pool temporal dimension and stack modalities
        pooled_features = []
        modality_order = []
        
        for modality in self.modality_names:
            if modality in gated_features:
                pooled = gated_features[modality].mean(dim=1)  # [B, D]
                pooled_features.append(pooled)
                modality_order.append(modality)
        
        if pooled_features:
            stacked = torch.stack(pooled_features, dim=1)  # [B, M, D]
            
            # Apply transformer fusion
            transformer_output = self.transformer_fusion(stacked)  # [B, M, D]
            
            # Apply learned modality importance
            importance_weights = F.softmax(self.modality_importance[:len(modality_order)], dim=0)
            weighted_output = transformer_output * importance_weights.view(1, -1, 1)
            
            # Final fusion
            flattened = weighted_output.reshape(batch_size, -1)
            fused_features = self.final_fusion(flattened)
        else:
            fused_features = torch.zeros(batch_size, self.fusion_dim).to(
                next(iter(modality_features.values())).device
            )
        
        output = {
            'fused_features': fused_features,
            'modality_features': gated_features,
            'transformer_output': transformer_output if pooled_features else None,
            'modality_importance': importance_weights if pooled_features else None
        }
        
        if return_attention_weights:
            output['attention_weights'] = attention_weights
            output['cross_modal_outputs'] = cross_modal_outputs
        
        return output


class MultiHeadCrossModalAttention(nn.Module):
    """Multi-head cross-modal attention mechanism"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, T_q, D]
            key: [B, T_k, D]
            value: [B, T_v, D]
            mask: Optional attention mask
        """
        B, T_q, D = query.shape
        T_k = key.shape[1]
        
        # Project and reshape
        q = self.q_proj(query).reshape(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T_q, T_k]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T_q, D)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + output)
        
        return output, attn_weights


class ModalitySpecificAttention(nn.Module):
    """Self-attention within a specific modality"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        return x


class TemporalCrossAttention(nn.Module):
    """Cross-attention across temporal dimensions"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        """
        # Temporal self-attention
        attn_output, _ = self.temporal_attention(x, x, x)
        attn_output = self.dropout(attn_output)
        
        # Residual and norm
        output = self.layer_norm(x + attn_output)
        
        return output


class GatedAttentionFusion(nn.Module):
    """Gated mechanism for adaptive fusion of modalities"""
    
    def __init__(self, num_modalities: int, dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.num_modalities = num_modalities
        self.dim = dim
        
        # Gating networks for each modality
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.Tanh(),
                nn.Linear(dim, dim),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
        # Cross-gating (modality interaction gates)
        self.cross_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.Sigmoid()
            ) for _ in range(num_modalities * (num_modalities - 1) // 2)
        ])
        
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply gated fusion to modality features
        
        Args:
            modality_features: Dictionary of modality features [B, T, D]
        """
        modality_names = list(modality_features.keys())
        
        # Pool temporal dimension
        pooled_features = {
            name: features.mean(dim=1) 
            for name, features in modality_features.items()
        }
        
        # Apply individual gates
        gated_features = {}
        for idx, (name, features) in enumerate(pooled_features.items()):
            if idx < len(self.gates):
                gate = self.gates[idx](features)
                gated_features[name] = features * gate
        
        # Apply cross-gates
        cross_gate_idx = 0
        enhanced_features = {name: feat.clone() for name, feat in gated_features.items()}
        
        for i, name1 in enumerate(modality_names):
            if name1 not in gated_features:
                continue
                
            for j, name2 in enumerate(modality_names):
                if j <= i or name2 not in gated_features:
                    continue
                
                if cross_gate_idx < len(self.cross_gates):
                    # Concatenate features for cross-gating
                    combined = torch.cat([
                        gated_features[name1],
                        gated_features[name2]
                    ], dim=-1)
                    
                    # Compute cross-gate
                    cross_gate = self.cross_gates[cross_gate_idx](combined)
                    
                    # Apply to both modalities
                    enhanced_features[name1] = enhanced_features[name1] + gated_features[name2] * cross_gate
                    enhanced_features[name2] = enhanced_features[name2] + gated_features[name1] * cross_gate
                    
                    cross_gate_idx += 1
        
        # Expand back to temporal dimension
        output_features = {}
        for name, features in enhanced_features.items():
            temporal_len = modality_features[name].shape[1]
            expanded = features.unsqueeze(1).expand(-1, temporal_len, -1)
            output_features[name] = modality_features[name] + expanded
        
        return output_features


class PerceiverFusion(nn.Module):
    """Perceiver-based fusion for handling variable-length inputs"""
    
    def __init__(self,
                 dim: int,
                 num_latents: int = 32,
                 num_heads: int = 8,
                 depth: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_latents = num_latents
        self.dim = dim
        
        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        
        # Cross-attention from latents to inputs
        self.cross_attend_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Self-attention within latents
        self.self_attend_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Layer norms
        self.cross_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.self_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            ) for _ in range(depth)
        ])
        
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, N, D]
        Returns:
            Latent representations [B, num_latents, D]
        """
        B = x.shape[0]
        
        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        
        for cross_attn, self_attn, cross_norm, self_norm, ffn, ffn_norm in zip(
            self.cross_attend_blocks,
            self.self_attend_blocks,
            self.cross_norms,
            self.self_norms,
            self.ffns,
            self.ffn_norms
        ):
            # Cross-attention: latents attend to input
            cross_out, _ = cross_attn(latents, x, x)
            latents = cross_norm(latents + cross_out)
            
            # Self-attention within latents
            self_out, _ = self_attn(latents, latents, latents)
            latents = self_norm(latents + self_out)
            
            # Feed-forward
            ffn_out = ffn(latents)
            latents = ffn_norm(latents + ffn_out)
        
        return latents


class TransformerFusion(nn.Module):
    """Transformer-based fusion layer"""
    
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, M, D] where M is number of modalities
        """
        return self.transformer_encoder(x)