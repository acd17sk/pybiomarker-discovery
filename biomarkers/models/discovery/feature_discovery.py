import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

class AutomatedFeatureDiscovery(nn.Module):
    """Automated discovery of novel biomarker patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.input_dim = config.get('input_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_prototypes = config.get('num_prototypes', 50)
        self.temperature = config.get('temperature', 0.07)
        
        self._build_model()
        
    def _build_model(self):
        # Feature encoder with self-attention
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Learnable biomarker prototypes
        self.prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, self.hidden_dim)
        )
        
        # Attention mechanism for prototype importance
        self.prototype_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Meta-learner for quick adaptation
        self.meta_learner = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(3)
        ])
        
        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 128)
        )
        
    def discover_patterns(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Discover novel biomarker patterns"""
        batch_size = features.shape[0]
        
        # Encode features
        encoded = self.feature_encoder(features)
        
        # Compare with learned prototypes
        prototype_similarity = F.cosine_similarity(
            encoded.unsqueeze(1),
            self.prototypes.unsqueeze(0),
            dim=2
        )
        
        # Soft assignment to prototypes
        prototype_assignment = F.softmax(prototype_similarity / self.temperature, dim=1)
        
        # Attention-weighted prototype aggregation
        attended_prototypes, attention_weights = self.prototype_attention(
            encoded.unsqueeze(1),
            self.prototypes.unsqueeze(0).repeat(batch_size, 1, 1),
            self.prototypes.unsqueeze(0).repeat(batch_size, 1, 1)
        )
        
        # Meta-learning adaptation
        adapted_features = encoded
        for meta_layer in self.meta_learner:
            adapted_features = adapted_features + meta_layer(adapted_features)
            adapted_features = F.relu(adapted_features)
        
        return {
            'encoded_features': encoded,
            'prototype_assignment': prototype_assignment,
            'attention_weights': attention_weights.squeeze(1),
            'adapted_features': adapted_features,
            'discovered_patterns': attended_prototypes.squeeze(1)
        }
    
    def contrastive_loss(self, 
                         anchor: torch.Tensor,
                         positive: torch.Tensor,
                         negative: torch.Tensor) -> torch.Tensor:
        """Contrastive loss for pattern discovery"""
        anchor_proj = self.projection_head(anchor)
        positive_proj = self.projection_head(positive)
        negative_proj = self.projection_head(negative)
        
        # Positive similarity
        pos_sim = F.cosine_similarity(anchor_proj, positive_proj, dim=1)
        
        # Negative similarities
        neg_sim = F.cosine_similarity(
            anchor_proj.unsqueeze(1),
            negative_proj.unsqueeze(0),
            dim=2
        )
        
        # Contrastive loss
        loss = -torch.log(
            torch.exp(pos_sim / self.temperature) /
            (torch.exp(pos_sim / self.temperature) + 
             torch.sum(torch.exp(neg_sim / self.temperature), dim=1))
        )
        
        return loss.mean()
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.discover_patterns(features)

class NeuralArchitectureSearch(nn.Module):
    """Neural Architecture Search for optimal biomarker combinations"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.search_space = config.get('search_space', {
            'fusion_ops': ['add', 'concat', 'attention', 'gating'],
            'activation_fns': ['relu', 'gelu', 'swish', 'mish'],
            'norm_types': ['batch', 'layer', 'instance', 'group']
        })
        self.num_layers = config.get('num_layers', 5)
        self.hidden_dim = config.get('hidden_dim', 256)
        
        self._build_search_modules()
        
    def _build_search_modules(self):
        # Architecture parameters (learnable)
        self.arch_params = nn.ParameterDict({
            'fusion_weights': nn.Parameter(torch.randn(len(self.search_space['fusion_ops']))),
            'activation_weights': nn.Parameter(torch.randn(len(self.search_space['activation_fns']))),
            'norm_weights': nn.Parameter(torch.randn(len(self.search_space['norm_types'])))
        })
        
        # Fusion operations
        self.fusion_ops = nn.ModuleDict({
            'add': nn.Identity(),
            'concat': nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            'attention': nn.MultiheadAttention(self.hidden_dim, 4),
            'gating': nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid()
            )
        })
        
        # Activation functions
        self.activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': nn.Mish()
        }
        
        # Normalization layers
        self.norms = nn.ModuleDict({
            'batch': nn.BatchNorm1d(self.hidden_dim),
            'layer': nn.LayerNorm(self.hidden_dim),
            'instance': nn.InstanceNorm1d(self.hidden_dim),
            'group': nn.GroupNorm(8, self.hidden_dim)
        })
        
    def search_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Execute one step of architecture search"""
        # Softmax over architecture parameters
        fusion_probs = F.softmax(self.arch_params['fusion_weights'], dim=0)
        activation_probs = F.softmax(self.arch_params['activation_weights'], dim=0)
        norm_probs = F.softmax(self.arch_params['norm_weights'], dim=0)
        
        # Apply weighted operations
        fused = 0
        for i, (op_name, op) in enumerate(self.fusion_ops.items()):
            if op_name == 'add':
                fused += fusion_probs[i] * (x + y)
            elif op_name == 'concat':
                cat = torch.cat([x, y], dim=-1)
                fused += fusion_probs[i] * op(cat)
            elif op_name == 'attention':
                att_out, _ = op(x.unsqueeze(1), y.unsqueeze(1), y.unsqueeze(1))
                fused += fusion_probs[i] * att_out.squeeze(1)
            elif op_name == 'gating':
                gate = op(torch.cat([x, y], dim=-1))
                fused += fusion_probs[i] * (gate * x + (1 - gate) * y)
        
        # Apply weighted activation
        activated = 0
        for i, (act_name, act) in enumerate(self.activations.items()):
            activated += activation_probs[i] * act(fused)
        
        # Apply weighted normalization
        normalized = 0
        for i, (norm_name, norm) in enumerate(self.norms.items()):
            if len(activated.shape) == 2:  # [B, D]
                normalized += norm_probs[i] * norm(activated)
            else:  # Reshape if needed
                shape = activated.shape
                activated_reshaped = activated.view(-1, self.hidden_dim)
                normalized += norm_probs[i] * norm(activated_reshaped).view(shape)
        
        return normalized
    
    def get_best_architecture(self) -> Dict[str, str]:
        """Get the best architecture based on learned parameters"""
        best_fusion = self.search_space['fusion_ops'][
            self.arch_params['fusion_weights'].argmax().item()
        ]
        best_activation = self.search_space['activation_fns'][
            self.arch_params['activation_weights'].argmax().item()
        ]
        best_norm = self.search_space['norm_types'][
            self.arch_params['norm_weights'].argmax().item()
        ]
        
        return {
            'fusion': best_fusion,
            'activation': best_activation,
            'normalization': best_norm
        }