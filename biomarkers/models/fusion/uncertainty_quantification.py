"""Uncertainty quantification methods for multi-modal fusion"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


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