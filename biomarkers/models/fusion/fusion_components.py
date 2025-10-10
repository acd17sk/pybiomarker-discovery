"""Core fusion components and utilities"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class CrossModalAttention(nn.Module):
    """Cross-modal attention for feature fusion"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle 2D inputs
        if len(query.shape) == 2:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Cross attention
        attn_output, attn_weights = self.attention(query, key, value)
        query = self.norm1(query + attn_output)
        
        # FFN
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        if squeeze_output:
            output = output.squeeze(1)
        
        return output, attn_weights


class ModalityEncoder(nn.Module):
    """Encode individual modality features"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TemporalFusion(nn.Module):
    """Temporal fusion for sequential multi-modal data"""
    
    def __init__(self, fusion_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Bi-directional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=fusion_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.Tanh(),
            nn.Linear(fusion_dim // 2, 1)
        )
        
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, D] temporal features
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Temporal attention weights
        attn_weights = self.temporal_attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        attended = (lstm_out * attn_weights).sum(dim=1)
        
        # Normalize
        fused = self.layer_norm(attended)
        
        return {
            'fused': fused,
            'temporal_features': lstm_out,
            'attention_weights': attn_weights
        }


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion combining attention and graph outputs"""
    
    def __init__(self, fusion_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Feature transformation
        self.attention_transform = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.graph_transform = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
    def forward(self, attention_features: torch.Tensor, 
                graph_features: torch.Tensor) -> torch.Tensor:
        """
        Hierarchically fuse attention and graph features
        
        Args:
            attention_features: [B, D] from attention fusion
            graph_features: [B, D] from graph fusion
        """
        # Transform features
        attn_transformed = self.attention_transform(attention_features)
        graph_transformed = self.graph_transform(graph_features)
        
        # Compute gate
        combined = torch.cat([attn_transformed, graph_transformed], dim=-1)
        gate = self.gate(combined)
        
        # Gated fusion
        gated = gate * attn_transformed + (1 - gate) * graph_transformed
        
        # Final fusion
        final_input = torch.cat([gated, combined], dim=-1)
        fused = self.fusion(final_input)
        
        return fused


class EvidentialUncertainty(nn.Module):
    """
    Evidential deep learning for uncertainty quantification
    Based on subjective logic theory
    """
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Evidence network
        self.evidence_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, features: torch.Tensor, 
                logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute evidential uncertainty
        
        Returns:
            Dictionary with uncertainty estimates
        """
        # Get evidence (non-negative)
        evidence = F.softplus(self.evidence_net(features))
        
        # Dirichlet parameters: alpha = evidence + 1
        alpha = evidence + 1
        
        # Belief mass
        S = alpha.sum(dim=1, keepdim=True)
        belief = evidence / S
        
        # Uncertainty
        uncertainty = self.num_classes / S
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric = belief * (1 - belief) / (S + 1)
        aleatoric_uncertainty = aleatoric.sum(dim=1)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = uncertainty.squeeze(1)
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        # Confidence (inverse of uncertainty)
        confidence = 1.0 / (1.0 + total_uncertainty)
        
        return {
            'uncertainty': total_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'confidence': confidence,
            'evidence': evidence,
            'alpha': alpha,
            'belief': belief
        }


class EnsembleUncertainty(nn.Module):
    """Ensemble-based uncertainty quantification"""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 num_estimators: int = 5, dropout: float = 0.1):
        super().__init__()
        
        self.num_estimators = num_estimators
        
        # Multiple prediction heads
        self.estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            ) for _ in range(num_estimators)
        ])
        
    def forward(self, features: torch.Tensor, 
                logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute ensemble uncertainty
        
        Returns:
            Dictionary with uncertainty estimates
        """
        # Get predictions from all estimators
        predictions = []
        for estimator in self.estimators:
            pred = F.softmax(estimator(features), dim=1)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [M, B, C]
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Variance (epistemic uncertainty)
        variance = predictions.var(dim=0).sum(dim=1)
        
        # Entropy (total uncertainty)
        entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
        
        # Mutual information (epistemic uncertainty)
        individual_entropy = -(predictions * torch.log(predictions + 1e-10)).sum(dim=2)
        mean_entropy = individual_entropy.mean(dim=0)
        mutual_info = entropy - mean_entropy
        
        # Confidence
        max_prob = mean_pred.max(dim=1)[0]
        confidence = max_prob
        
        return {
            'uncertainty': entropy,
            'epistemic_uncertainty': mutual_info,
            'aleatoric_uncertainty': mean_entropy,
            'variance': variance,
            'confidence': confidence,
            'ensemble_predictions': predictions
        }


class DropoutUncertainty(nn.Module):
    """Monte Carlo dropout for uncertainty quantification"""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 dropout: float = 0.3, num_samples: int = 20):
        super().__init__()
        
        self.num_samples = num_samples
        
        # Prediction head with dropout
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, features: torch.Tensor, 
                logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute dropout-based uncertainty using MC sampling
        
        Returns:
            Dictionary with uncertainty estimates
        """
        # Enable dropout at inference
        self.train()
        
        # Multiple forward passes
        predictions = []
        for _ in range(self.num_samples):
            pred = F.softmax(self.predictor(features), dim=1)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [T, B, C]
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Predictive variance
        variance = predictions.var(dim=0).sum(dim=1)
        
        # Predictive entropy (total uncertainty)
        entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
        
        # Mutual information (epistemic uncertainty)
        individual_entropy = -(predictions * torch.log(predictions + 1e-10)).sum(dim=2)
        expected_entropy = individual_entropy.mean(dim=0)
        mutual_info = entropy - expected_entropy
        
        # Confidence
        confidence = 1.0 / (1.0 + entropy)
        
        # Restore eval mode
        self.eval()
        
        return {
            'uncertainty': entropy,
            'epistemic_uncertainty': mutual_info,
            'aleatoric_uncertainty': expected_entropy,
            'variance': variance,
            'confidence': confidence
        }