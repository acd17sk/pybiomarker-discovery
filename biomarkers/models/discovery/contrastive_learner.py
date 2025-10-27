"""Contrastive learning for biomarker discovery"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ContrastiveBiomarkerLearner(nn.Module):
    """
    Main contrastive learning module for discovering subtle differences
    between healthy and at-risk populations
    """
    
    def __init__(self,
                 encoder_dim: int,
                 projection_dim: int = 128,
                 temperature: float = 0.07,
                 method: str = 'simclr',
                 num_prototypes: int = 10,
                 dropout: float = 0.3):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.method = method
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, projection_dim)
        )
        
        # Method-specific components
        if method == 'simclr':
            self.contrastive = SimCLRBiomarker(projection_dim, temperature)
        elif method == 'moco':
            self.contrastive = MoCoBiomarker(projection_dim, temperature)
        elif method == 'supcon':
            self.contrastive = SupConBiomarker(projection_dim, temperature)
        elif method == 'prototypical':
            self.contrastive = PrototypicalBiomarker(projection_dim, num_prototypes)
        elif method == 'triplet':
            self.contrastive = TripletBiomarker(projection_dim)
        else:
            raise ValueError(f"Unknown contrastive method: {method}")
        
        # Augmentation module
        self.augmentation = ContrastiveAugmentation()
        
        # Healthy vs risk contrastive module
        self.healthy_risk_contrastive = HealthyRiskContrastive(
            projection_dim, temperature
        )
    
    def forward(self,
                features: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                health_status: Optional[torch.Tensor] = None,
                return_projections: bool = True) -> Dict[str, torch.Tensor]:
        """
        Contrastive learning forward pass
        
        Args:
            features: Encoded features [batch, encoder_dim]
            labels: Class labels for supervised contrastive [batch]
            health_status: Binary health status (0=healthy, 1=at-risk) [batch]
            return_projections: Whether to return projected features
        """
        # Project features
        z = self.projector(features)
        z = F.normalize(z, dim=-1)
        
        output = {}
        
        # Apply contrastive learning
        if self.method in ['simclr', 'moco']:
            # Generate augmented views
            z1 = self.augmentation(z)
            z2 = self.augmentation(z)
            
            contrastive_output = self.contrastive(z1, z2)
            output.update(contrastive_output)
            
        elif self.method == 'supcon':
            if labels is None:
                raise ValueError("SupCon requires labels")
            
            # Generate augmented views
            z1 = self.augmentation(z)
            z2 = self.augmentation(z)
            
            contrastive_output = self.contrastive(z1, z2, labels)
            output.update(contrastive_output)
            
        elif self.method == 'prototypical':
            contrastive_output = self.contrastive(z, labels)
            output.update(contrastive_output)
            
        elif self.method == 'triplet':
            if labels is None:
                raise ValueError("Triplet learning requires labels")
            
            contrastive_output = self.contrastive(z, labels)
            output.update(contrastive_output)
        
        # Healthy vs at-risk contrastive learning
        if health_status is not None:
            hr_output = self.healthy_risk_contrastive(z, health_status)
            output['healthy_risk_loss'] = hr_output['loss']
            output['healthy_risk_accuracy'] = hr_output['accuracy']
        
        if return_projections:
            output['projections'] = z
            output['features'] = features
        
        return output


class SimCLRBiomarker(nn.Module):
    """
    SimCLR: A Simple Framework for Contrastive Learning
    Adapted for biomarker discovery
    """
    
    def __init__(self, projection_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.projection_dim = projection_dim
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute SimCLR loss
        
        Args:
            z1: First augmented view [batch, projection_dim]
            z2: Second augmented view [batch, projection_dim]
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Concatenate views
        z = torch.cat([z1, z2], dim=0)  # [2*batch, projection_dim]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [2*batch, 2*batch]
        
        # Create positive pair mask
        # Each sample is positive with its augmented version
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        mask = mask.roll(shifts=batch_size, dims=0)
        
        # Remove diagonal (self-similarity)
        sim_matrix = sim_matrix.masked_fill(torch.eye(2 * batch_size, dtype=torch.bool, device=device), float('-inf'))
        
        # Compute log probabilities
        log_prob = F.log_softmax(sim_matrix, dim=-1)
        
        # Extract positive pairs
        positive_log_prob = log_prob[mask].view(2 * batch_size, -1)
        
        # SimCLR loss (negative log likelihood)
        loss = -positive_log_prob.mean()
        
        # Compute alignment and uniformity metrics
        alignment = self._alignment(z1, z2)
        uniformity = self._uniformity(z)
        
        return {
            'loss': loss,
            'alignment': alignment,
            'uniformity': uniformity
        }
    
    def _alignment(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Measure alignment between positive pairs"""
        return (z1 - z2).norm(p=2, dim=-1).pow(2).mean()
    
    def _uniformity(self, z: torch.Tensor) -> torch.Tensor:
        """Measure uniformity of feature distribution"""
        sq_distances = torch.cdist(z, z, p=2).pow(2)
        return sq_distances.mul(-2).exp().mean().log()


class MoCoBiomarker(nn.Module):
    """
    MoCo: Momentum Contrast for biomarker learning
    Uses momentum encoder and memory bank
    """
    
    def __init__(self,
                 projection_dim: int = 128,
                 temperature: float = 0.07,
                 queue_size: int = 65536,
                 momentum: float = 0.999):
        super().__init__()
        
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.queue_size = queue_size
        self.momentum = momentum
        
        # Memory bank (queue)
        self.register_buffer('queue', torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    def forward(self, z_q: torch.Tensor, z_k: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute MoCo loss
        
        Args:
            z_q: Query features [batch, projection_dim]
            z_k: Key features [batch, projection_dim]
        """
        batch_size = z_q.shape[0]
        
        # Normalize
        z_q = F.normalize(z_q, dim=-1)
        z_k = F.normalize(z_k, dim=-1)
        
        # Positive pairs: query with corresponding key
        l_pos = torch.einsum('nc,nc->n', [z_q, z_k]).unsqueeze(-1)  # [batch, 1]
        
        # Negative pairs: query with keys from queue
        l_neg = torch.einsum('nc,ck->nk', [z_q, self.queue.clone().detach()])  # [batch, queue_size]
        
        # Logits: [batch, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels: positives are the first element
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_q.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Update queue
        self._dequeue_and_enqueue(z_k)
        
        # Accuracy
        pred = logits.argmax(dim=1)
        accuracy = (pred == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'logits': logits
        }
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update memory bank"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest batch in queue
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr


class SupConBiomarker(nn.Module):
    """
    Supervised Contrastive Learning for biomarkers
    Uses label information to define positive pairs
    """
    
    def __init__(self, projection_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.projection_dim = projection_dim
        self.temperature = temperature
    
    def forward(self,
                z1: torch.Tensor,
                z2: torch.Tensor,
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute supervised contrastive loss
        
        Args:
            z1: First view [batch, projection_dim]
            z2: Second view [batch, projection_dim]
            labels: Class labels [batch]
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Concatenate two views
        z = torch.cat([z1, z2], dim=0)  # [2*batch, projection_dim]
        labels = torch.cat([labels, labels], dim=0)  # [2*batch]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create mask for positive pairs (same class, different augmentation)
        labels_equal = labels.unsqueeze(1) == labels.unsqueeze(0)  # [2*batch, 2*batch]
        mask = labels_equal.float()
        
        # Remove self-similarity
        mask = mask.masked_fill(torch.eye(2 * batch_size, dtype=torch.bool, device=device), 0)
        
        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Compute log probabilities
        exp_logits = torch.exp(logits)
        
        # Mask out self-similarity
        exp_logits = exp_logits * (1 - torch.eye(2 * batch_size, device=device))
        
        # Sum over all negatives
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return {
            'loss': loss,
            'num_positives': mask_sum.mean()
        }


class PrototypicalBiomarker(nn.Module):
    """
    Prototypical Networks for biomarker learning
    Learns class prototypes in embedding space
    """
    
    def __init__(self,
                 projection_dim: int = 128,
                 num_prototypes: int = 10,
                 temperature: float = 1.0):
        super().__init__()
        
        self.projection_dim = projection_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        
        # Learnable prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, projection_dim))
        
        # Prototype momentum updates
        self.register_buffer('prototype_momentum', torch.ones(num_prototypes) * 0.9)
    
    def forward(self,
                z: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute prototypical loss
        
        Args:
            z: Embeddings [batch, projection_dim]
            labels: Class labels [batch] (optional)
        """
        batch_size = z.shape[0]
        
        # Normalize
        z = F.normalize(z, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        
        # Compute distances to prototypes
        distances = torch.cdist(z, prototypes)  # [batch, num_prototypes]
        
        # Convert to similarities (negative distances)
        logits = -distances / self.temperature
        
        if labels is not None:
            # Supervised: assign samples to ground truth prototypes
            loss = F.cross_entropy(logits, labels)
            
            # Update prototypes with batch statistics
            self._update_prototypes(z, labels)
        else:
            # Unsupervised: use soft assignment
            soft_assignment = F.softmax(logits, dim=-1)
            
            # Entropy regularization (encourage confident assignments)
            entropy = -(soft_assignment * torch.log(soft_assignment + 1e-8)).sum(-1).mean()
            
            # Cluster assignment loss
            cluster_loss = -(soft_assignment.max(dim=-1)[0]).mean()
            
            loss = cluster_loss + 0.1 * entropy
        
        # Compute prototype diversity
        prototype_sim = torch.matmul(prototypes, prototypes.T)
        prototype_diversity = -prototype_sim.mean()
        
        return {
            'loss': loss,
            'logits': logits,
            'prototypes': prototypes,
            'prototype_diversity': prototype_diversity
        }
    
    @torch.no_grad()
    def _update_prototypes(self, z: torch.Tensor, labels: torch.Tensor):
        """Update prototypes using exponential moving average"""
        for c in range(self.num_prototypes):
            mask = labels == c
            if mask.sum() > 0:
                # Compute centroid of class c
                centroid = z[mask].mean(dim=0)
                
                # Update prototype with momentum
                momentum = self.prototype_momentum[c]
                self.prototypes.data[c] = (
                    momentum * self.prototypes.data[c] +
                    (1 - momentum) * centroid
                )


class TripletBiomarker(nn.Module):
    """
    Triplet loss for biomarker learning
    Learns by comparing anchor, positive, and negative samples
    """
    
    def __init__(self,
                 projection_dim: int = 128,
                 margin: float = 1.0,
                 mining_strategy: str = 'hard'):
        super().__init__()
        
        self.projection_dim = projection_dim
        self.margin = margin
        self.mining_strategy = mining_strategy
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute triplet loss
        
        Args:
            z: Embeddings [batch, projection_dim]
            labels: Class labels [batch]
        """
        batch_size = z.shape[0]
        
        # Normalize
        z = F.normalize(z, dim=-1)
        
        # Compute pairwise distances
        distances = torch.cdist(z, z, p=2)
        
        # Create masks
        labels_equal = labels.unsqueeze(1) == labels.unsqueeze(0)
        labels_not_equal = ~labels_equal
        
        # Remove self-comparisons
        mask_anchor_out = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
        labels_equal = labels_equal & mask_anchor_out
        
        if self.mining_strategy == 'hard':
            # Hard negative mining
            # For each anchor, find hardest positive and hardest negative
            
            # Mask for valid positives/negatives
            distances_pos = distances.clone()
            distances_pos[~labels_equal] = float('inf')
            
            distances_neg = distances.clone()
            distances_neg[~labels_not_equal] = float('-inf')
            
            # Find hardest positive (furthest positive)
            hardest_positive_dist, _ = distances_pos.min(dim=1)
            
            # Find hardest negative (closest negative)
            hardest_negative_dist, _ = distances_neg.max(dim=1)
            
            # Triplet loss
            losses = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            
        elif self.mining_strategy == 'semi-hard':
            # Semi-hard negative mining
            losses = []
            
            for i in range(batch_size):
                # Positive samples
                pos_mask = labels_equal[i]
                if pos_mask.sum() == 0:
                    continue
                
                # Negative samples
                neg_mask = labels_not_equal[i]
                if neg_mask.sum() == 0:
                    continue
                
                # Get distances
                pos_dists = distances[i][pos_mask]
                neg_dists = distances[i][neg_mask]
                
                # For each positive
                for pos_dist in pos_dists:
                    # Find semi-hard negatives (d_neg > d_pos but d_neg < d_pos + margin)
                    semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + self.margin)
                    
                    if semi_hard_mask.sum() > 0:
                        # Use hardest semi-hard negative
                        neg_dist = neg_dists[semi_hard_mask].min()
                    else:
                        # Use hardest negative
                        neg_dist = neg_dists.min()
                    
                    loss = F.relu(pos_dist - neg_dist + self.margin)
                    losses.append(loss)
            
            if losses:
                losses = torch.stack(losses)
            else:
                losses = torch.tensor(0.0, device=z.device)
        
        else:  # 'all' strategy
            # Use all valid triplets
            losses = []
            
            for i in range(batch_size):
                pos_mask = labels_equal[i]
                neg_mask = labels_not_equal[i]
                
                if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                    continue
                
                pos_dists = distances[i][pos_mask]
                neg_dists = distances[i][neg_mask]
                
                # All combinations
                for pos_dist in pos_dists:
                    for neg_dist in neg_dists:
                        loss = F.relu(pos_dist - neg_dist + self.margin)
                        losses.append(loss)
            
            if losses:
                losses = torch.stack(losses)
            else:
                losses = torch.tensor(0.0, device=z.device)
        
        # Mean loss
        loss = losses.mean()
        
        # Compute statistics
        num_valid_triplets = (losses > 0).sum().float()
        fraction_positive_triplets = num_valid_triplets / max(losses.numel(), 1)
        
        return {
            'loss': loss,
            'num_valid_triplets': num_valid_triplets,
            'fraction_positive': fraction_positive_triplets
        }


class ContrastiveAugmentation(nn.Module):
    """
    Data augmentation for contrastive learning of biomarkers
    Applies noise, masking, and temporal perturbations
    """
    
    def __init__(self,
                 noise_std: float = 0.1,
                 mask_ratio: float = 0.15,
                 temporal_shift: int = 5):
        super().__init__()
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.temporal_shift = temporal_shift
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations
        
        Args:
            x: Input features [batch, features]
        """
        if not self.training:
            return x
        
        batch_size, feat_dim = x.shape
        device = x.device
        
        # Random noise
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Random masking
        if torch.rand(1).item() > 0.5:
            mask = torch.rand(batch_size, feat_dim, device=device) > self.mask_ratio
            x = x * mask
        
        # Feature dropout
        if torch.rand(1).item() > 0.5:
            dropout_mask = torch.rand(batch_size, feat_dim, device=device) > 0.1
            x = x * dropout_mask
        
        # Scaling
        if torch.rand(1).item() > 0.5:
            scale = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.2).to(device)
            x = x * scale
        
        return x


class HealthyRiskContrastive(nn.Module):
    """
    Specialized contrastive learning to distinguish healthy from at-risk
    Focuses on finding subtle biomarker differences
    """
    
    def __init__(self,
                 projection_dim: int = 128,
                 temperature: float = 0.07):
        super().__init__()
        
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Healthy vs risk discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(projection_dim // 2, 2)  # Binary: healthy vs at-risk
        )
    
    def forward(self,
                z: torch.Tensor,
                health_status: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Contrastive learning for healthy vs at-risk
        
        Args:
            z: Embeddings [batch, projection_dim]
            health_status: Binary labels (0=healthy, 1=at-risk) [batch]
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Normalize embeddings
        z_norm = F.normalize(z, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.temperature
        
        # Create contrastive mask
        # Positives: same health status
        # Negatives: different health status
        status_equal = health_status.unsqueeze(1) == health_status.unsqueeze(0)
        
        # Remove diagonal
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        status_equal = status_equal & ~mask_self
        
        # Contrastive loss
        # Pull together same health status, push apart different
        positive_mask = status_equal.float()
        negative_mask = (~status_equal & ~mask_self).float()
        
        # Positive similarity (maximize)
        positive_sim = (sim_matrix * positive_mask).sum() / (positive_mask.sum() + 1e-8)
        
        # Negative similarity (minimize)
        negative_sim = (sim_matrix * negative_mask).sum() / (negative_mask.sum() + 1e-8)
        
        # Contrastive loss: maximize positive, minimize negative
        contrastive_loss = -positive_sim + negative_sim
        
        # Classification loss
        logits = self.discriminator(z)
        classification_loss = F.cross_entropy(logits, health_status)
        
        # Combined loss
        loss = contrastive_loss + classification_loss
        
        # Accuracy
        pred = logits.argmax(dim=1)
        accuracy = (pred == health_status).float().mean()
        
        # Compute separation metric (how well separated are the two groups)
        healthy_mask = health_status == 0
        risk_mask = health_status == 1
        
        if healthy_mask.sum() > 0 and risk_mask.sum() > 0:
            healthy_mean = z_norm[healthy_mask].mean(dim=0)
            risk_mean = z_norm[risk_mask].mean(dim=0)
            separation = F.cosine_similarity(healthy_mean.unsqueeze(0),
                                            risk_mean.unsqueeze(0)).item()
        else:
            separation = 0.0
        
        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'classification_loss': classification_loss,
            'accuracy': accuracy,
            'separation': separation,
            'positive_similarity': positive_sim,
            'negative_similarity': negative_sim
        }