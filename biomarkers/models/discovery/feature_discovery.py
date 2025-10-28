"""Automated feature discovery using attention mechanisms and adaptive selection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AutomatedFeatureDiscovery(nn.Module):
    """
    Main automated feature discovery module that combines multiple discovery strategies
    to identify predictive biomarker patterns
    """
    
    def __init__(self,
                 input_dim: int,
                 num_modalities: int = 4,
                 hidden_dim: int = 256,
                 num_diseases: int = 10,
                 dropout: float = 0.3,
                 use_attention: bool = True,
                 use_interaction: bool = True,
                 use_cross_modal: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.num_diseases = num_diseases
        
        # Input projection to ensure consistent dimensions
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Attention-based discovery
        if use_attention:
            self.attention_discovery = AttentionBasedDiscovery(
                input_dim=hidden_dim,  # Use hidden_dim after projection
                hidden_dim=hidden_dim,
                num_heads=8,
                dropout=dropout
            )
        else:
            self.attention_discovery = None
        
        # Feature interaction network
        if use_interaction:
            self.interaction_network = FeatureInteractionNetwork(
                input_dim=hidden_dim,  # Use hidden_dim after projection
                hidden_dim=hidden_dim,
                num_interactions=3,
                dropout=dropout
            )
        else:
            self.interaction_network = None
        
        # Cross-modal discovery
        if use_cross_modal:
            self.cross_modal_discovery = CrossModalFeatureDiscovery(
                input_dim=hidden_dim,  # Use hidden_dim after projection
                num_modalities=num_modalities,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            self.cross_modal_discovery = None
        
        # Adaptive feature selector
        self.adaptive_selector = AdaptiveFeatureSelector(
            input_dim=hidden_dim,  # Use hidden_dim after projection
            hidden_dim=hidden_dim,
            selection_ratio=0.5,
            dropout=dropout
        )
        
        # Combination finder
        self.combination_finder = BiomarkerCombinationFinder(
            input_dim=hidden_dim,  # Use hidden_dim after projection
            hidden_dim=hidden_dim,
            max_combinations=100,
            dropout=dropout
        )
        
        # Calculate total feature dimension
        # Count actual concatenated features
        num_feature_sources = 1  # adaptive_selector always produces hidden_dim
        if use_attention:
            num_feature_sources += 1  # attention produces hidden_dim
        if use_interaction:
            num_feature_sources += 1  # interaction produces hidden_dim
        if use_cross_modal:
            num_feature_sources += num_modalities  # cross_modal produces num_modalities × hidden_dim
        
        total_dim = num_feature_sources * hidden_dim
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Disease prediction head
        self.disease_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_diseases)
        )
        
        # Feature importance estimator
        self.importance_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # Discovered pattern tracker
        self.register_buffer('discovered_patterns', torch.zeros(100, input_dim))
        self.register_buffer('pattern_scores', torch.zeros(100))
        self.register_buffer('pattern_count', torch.tensor(0))
    
    def forward(self,
                x: torch.Tensor,
                modality_masks: Optional[torch.Tensor] = None,
                return_importance: bool = True,
                return_patterns: bool = True) -> Dict[str, torch.Tensor]:
        """
        Discover predictive features and patterns
        
        Args:
            x: Input features [batch, features]
            modality_masks: Binary masks for each modality [batch, num_modalities, features]
            return_importance: Return feature importance scores
            return_patterns: Return discovered patterns
        """
        # Project input to hidden dimension
        x_proj = self.input_projection(x)
        
        discovered_features = []
        
        # Adaptive feature selection
        selected_features, selection_scores = self.adaptive_selector(x_proj)
        discovered_features.append(selected_features)
        
        # Attention-based discovery
        if self.attention_discovery is not None:
            attention_features = self.attention_discovery(x_proj)
            discovered_features.append(attention_features['attended_features'])
        
        # Feature interaction discovery
        if self.interaction_network is not None:
            interaction_features = self.interaction_network(x_proj)
            discovered_features.append(interaction_features['interaction_features'])
        
        # Cross-modal discovery
        if self.cross_modal_discovery is not None and modality_masks is not None:
            # Project modality masks to match hidden_dim
            cross_modal_features = self.cross_modal_discovery(x_proj, modality_masks)
            for modal_features in cross_modal_features['modal_features']:
                discovered_features.append(modal_features)
        
        # Concatenate all discovered features
        all_features = torch.cat(discovered_features, dim=-1)
        
        # Fuse features
        fused_features = self.feature_fusion(all_features)
        
        # Predict diseases
        disease_logits = self.disease_predictor(fused_features)
        disease_probs = F.softmax(disease_logits, dim=-1)
        
        output = {
            'logits': disease_logits,
            'probabilities': disease_probs,
            'features': fused_features,
            'selected_features': selected_features,
            'selection_scores': selection_scores
        }
        
        # Feature importance
        if return_importance:
            importance_scores = self.importance_estimator(fused_features)
            output['importance_scores'] = importance_scores
            output['top_features'] = torch.topk(importance_scores, k=min(20, self.input_dim), dim=-1)
        
        # Discovered patterns
        if return_patterns:
            combination_output = self.combination_finder(x_proj)
            output['combinations'] = combination_output['combinations']
            output['combination_scores'] = combination_output['scores']
            
            # Update pattern tracker
            self._update_patterns(combination_output['combinations'],
                                combination_output['scores'])
        
        # Add component outputs
        if self.attention_discovery is not None:
            output['attention_weights'] = attention_features['attention_weights']
        
        if self.interaction_network is not None:
            output['interaction_weights'] = interaction_features['interaction_weights']
        
        return output
    
    def _update_patterns(self, combinations: torch.Tensor, scores: torch.Tensor):
        """Update discovered pattern tracker"""
        batch_size = combinations.shape[0]
        
        # Get top patterns from batch
        top_scores, top_indices = torch.topk(scores, k=min(10, scores.shape[1]), dim=1)
        
        for b in range(batch_size):
            for idx in top_indices[b]:
                pattern = combinations[b, idx]
                score = scores[b, idx]
                
                # Add to tracker if space available or better than existing
                if self.pattern_count < 100:
                    pos = self.pattern_count.item()
                    self.discovered_patterns[pos] = pattern[:self.input_dim]  # Ensure correct size
                    self.pattern_scores[pos] = score
                    self.pattern_count += 1
                else:
                    # Replace worst pattern if current is better
                    min_score, min_idx = torch.min(self.pattern_scores, dim=0)
                    if score > min_score:
                        self.discovered_patterns[min_idx] = pattern[:self.input_dim]  # Ensure correct size
                        self.pattern_scores[min_idx] = score
    
    def get_top_patterns(self, k: int = 10) -> Dict[str, torch.Tensor]:
        """Get top k discovered patterns"""
        if self.pattern_count == 0:
            return {'patterns': None, 'scores': None}
        
        valid_count = min(self.pattern_count.item(), 100)
        valid_patterns = self.discovered_patterns[:valid_count]
        valid_scores = self.pattern_scores[:valid_count]
        
        top_scores, top_indices = torch.topk(valid_scores, k=min(k, valid_count))
        top_patterns = valid_patterns[top_indices]
        
        return {
            'patterns': top_patterns,
            'scores': top_scores,
            'indices': top_indices
        }


class AttentionBasedDiscovery(nn.Module):
    """
    Use multi-head attention to discover which feature combinations are predictive
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection (now expects input_dim == hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Feature interaction attention
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply attention-based feature discovery
        
        Args:
            x: Input features [batch, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # Project input
        h = self.input_proj(x)
        
        # Create sequence by replicating features to allow self-attention
        seq_len = min(32, self.input_dim)  # Limit sequence length for efficiency
        h = h.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden]
        
        # Store attention weights
        all_attention_weights = []
        
        # Apply attention layers
        for i, (attn, ln, ff) in enumerate(zip(self.attention_layers,
                                               self.layer_norms,
                                               self.feed_forwards)):
            # Self-attention with residual
            h_attn, attn_weights = attn(h, h, h)
            all_attention_weights.append(attn_weights)
            h = ln(h + h_attn)
            
            # Feed-forward with residual
            h_ff = ff(h)
            h = ln(h + h_ff)
        
        # Pool sequence dimension
        h = h.mean(dim=1)  # [batch, hidden]
        
        # Apply interaction attention with proper sequence
        h_seq = h.unsqueeze(1).expand(-1, 2, -1)  # [batch, 2, hidden]
        
        interaction_features, interaction_weights = self.interaction_attention(
            h_seq, h_seq, h_seq
        )
        
        # Pool interaction features
        attended_features = self.output_proj(interaction_features.mean(dim=1))
        
        return {
            'attended_features': attended_features,
            'attention_weights': all_attention_weights[-1] if all_attention_weights else torch.ones(batch_size, seq_len, seq_len, device=x.device) / seq_len,
            'interaction_weights': interaction_weights
        }


class FeatureInteractionNetwork(nn.Module):
    """
    Discover higher-order feature interactions using neural networks
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_interactions: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_interactions = num_interactions
        
        # Simple pairwise interaction (avoid sampling)
        self.pairwise_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Triple interaction network
        self.triple_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Factorization machines for efficient interactions
        self.fm_v = nn.Parameter(torch.randn(input_dim, hidden_dim // 4))
        self.fm_linear = nn.Linear(input_dim, 1)
        
        # Deep interaction network
        self.deep_interaction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Interaction fusion
        self.interaction_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3 + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Interaction importance
        self.interaction_importance = nn.Sequential(
            nn.Linear(hidden_dim, input_dim * input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Discover feature interactions
        
        Args:
            x: Input features [batch, input_dim]
        """
        batch_size = x.shape[0]
        
        # Process through networks (no sampling needed)
        pairwise_pooled = self.pairwise_net(x)
        triple_pooled = self.triple_net(x)
        
        # Factorization machines
        fm_linear_out = self.fm_linear(x)
        
        # Pairwise FM interactions
        x_v = torch.matmul(x, self.fm_v)  # [batch, hidden//4]
        fm_interactions = 0.5 * (
            torch.pow(x_v.sum(dim=1, keepdim=True), 2) -
            torch.pow(x_v, 2).sum(dim=1, keepdim=True)
        )
        fm_out = fm_linear_out + fm_interactions
        
        # Deep interactions
        deep_features = self.deep_interaction(x)
        
        # Combine all interactions
        all_interactions = torch.cat([
            pairwise_pooled,
            triple_pooled,
            deep_features,
            fm_out.expand(-1, self.hidden_dim // 4)
        ], dim=-1)
        
        interaction_features = self.interaction_fusion(all_interactions)
        
        # Compute interaction importance
        interaction_weights = self.interaction_importance(interaction_features)
        interaction_weights = interaction_weights.view(batch_size, self.input_dim, self.input_dim)
        
        return {
            'interaction_features': interaction_features,
            'interaction_weights': interaction_weights,
            'pairwise_features': pairwise_pooled,
            'triple_features': triple_pooled,
            'fm_features': fm_out.squeeze(-1) if fm_out.dim() > 1 else fm_out
        }


class AdaptiveFeatureSelector(nn.Module):
    """
    Adaptively select most informative features using gating mechanisms
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 selection_ratio: float = 0.5,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.selection_ratio = selection_ratio
        self.num_selected = max(1, int(input_dim * selection_ratio))
        
        # Feature importance network
        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        # Context-aware gating
        self.context_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temperature for Gumbel-Softmax
        self.register_buffer('temperature', torch.tensor(1.0))
    
    def forward(self,
                x: torch.Tensor,
                hard_selection: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptively select features
        
        Args:
            x: Input features [batch, input_dim]
            hard_selection: Use hard (discrete) selection
        """
        # Compute importance scores
        importance_logits = self.importance_net(x)
        
        # Context-aware gating
        gates = self.context_gate(x)
        
        # Combine importance and gates
        selection_logits = importance_logits * gates
        
        if self.training and not hard_selection:
            # Gumbel-Softmax for differentiable selection
            selection_probs = F.gumbel_softmax(
                selection_logits,
                tau=self.temperature,
                hard=False,
                dim=-1
            )
        else:
            # Top-k selection
            _, top_indices = torch.topk(selection_logits, k=self.num_selected, dim=-1)
            selection_probs = torch.zeros_like(selection_logits)
            selection_probs.scatter_(1, top_indices, 1.0)
        
        # Apply selection
        selected_features = x * selection_probs
        
        # Transform selected features
        transformed = self.feature_transform(selected_features)
        
        return transformed, selection_probs
    
    def anneal_temperature(self, step: int, total_steps: int):
        """Anneal temperature for Gumbel-Softmax"""
        min_temp = 0.5
        max_temp = 1.0
        new_temp = max_temp - (max_temp - min_temp) * (step / total_steps)
        self.temperature.fill_(new_temp)


class BiomarkerCombinationFinder(nn.Module):
    """
    Find optimal biomarker combinations using learned combination generator
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 max_combinations: int = 100,
                 combination_size: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_combinations = max_combinations
        self.combination_size = combination_size
        
        # Combination generator
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_combinations * input_dim)
        )
        
        # Combination scorer
        self.scorer = nn.Sequential(
            nn.Linear(input_dim * combination_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Synergy detector (detects when combinations are more than sum of parts)
        self.synergy_detector = nn.Sequential(
            nn.Linear(input_dim * combination_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Positive = synergy, negative = redundancy
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Find biomarker combinations
        
        Args:
            x: Input features [batch, input_dim]
        """
        batch_size = x.shape[0]
        
        # Generate combination probabilities
        combination_logits = self.generator(x)
        combination_logits = combination_logits.view(
            batch_size, self.max_combinations, self.input_dim
        )
        
        # Apply Gumbel-Softmax to get differentiable combinations
        combination_probs = F.gumbel_softmax(
            combination_logits,
            tau=0.5,
            hard=self.training,
            dim=-1
        )
        
        # Sample combinations and compute scores
        scores = []
        synergies = []
        actual_combinations = []
        
        for i in range(self.max_combinations):
            # Get combination
            combo_probs = combination_probs[:, i, :]  # [batch, input_dim]
            
            # Select top-k features
            _, top_indices = torch.topk(combo_probs, k=self.combination_size, dim=-1)
            
            # Extract feature values
            combo_features = []
            for b in range(batch_size):
                combo_features.append(x[b, top_indices[b]])
            combo_features = torch.stack(combo_features, dim=0)  # [batch, combination_size]
            
            # Flatten for scoring
            combo_flat = combo_features.view(batch_size, -1)
            
            # Score combination
            score = self.scorer(combo_flat).squeeze(-1)
            scores.append(score)
            
            # Detect synergy
            synergy = self.synergy_detector(combo_flat).squeeze(-1)
            synergies.append(synergy)
            
            # Store combination
            actual_combinations.append(combo_probs)
        
        scores = torch.stack(scores, dim=1)  # [batch, max_combinations]
        synergies = torch.stack(synergies, dim=1)
        combinations = torch.stack(actual_combinations, dim=1)
        
        return {
            'combinations': combinations,
            'scores': scores,
            'synergies': synergies,
            'combination_logits': combination_logits
        }


class CrossModalFeatureDiscovery(nn.Module):
    """
    Discover features across different modalities (voice, movement, vision, text)
    """
    
    def __init__(self,
                 input_dim: int,
                 num_modalities: int = 4,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        
        # Modality-specific encoders
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_modalities)
        ])
        
        # Cross-modal attention
        self.cross_modal_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_modalities)
        ])
        
        # Cross-modal fusion
        self.cross_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Modality importance
        self.modality_importance = nn.Sequential(
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self,
                x: torch.Tensor,
                modality_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Discover cross-modal features
        
        Args:
            x: Input features [batch, input_dim]
            modality_masks: Binary masks [batch, num_modalities, original_features]
        """
        batch_size = x.shape[0]
        
        # Extract modality-specific features
        modal_features = []
        
        for i in range(self.num_modalities):
            # Simply use the projected input (masks were for original space)
            # In practice, you'd apply masks before projection
            modal_feat = self.modality_encoders[i](x)
            modal_features.append(modal_feat)
        
        # Stack for attention
        modal_stack = torch.stack(modal_features, dim=1)  # [batch, num_modalities, hidden]
        
        # Apply cross-modal attention
        attended_features = []
        attention_weights = []
        
        for i in range(self.num_modalities):
            query = modal_features[i].unsqueeze(1)  # [batch, 1, hidden]
            
            attended, attn_w = self.cross_modal_attention[i](
                query, modal_stack, modal_stack
            )
            attended_features.append(attended.squeeze(1))
            attention_weights.append(attn_w)
        
        # Fuse all modalities
        all_modalities = torch.cat(attended_features, dim=-1)
        fused_features = self.cross_fusion(all_modalities)
        
        # Compute modality importance
        importance = self.modality_importance(fused_features)
        
        return {
            'fused_features': fused_features,
            'modal_features': modal_features,
            'attended_features': attended_features,
            'attention_weights': attention_weights,
            'modality_importance': importance
        }