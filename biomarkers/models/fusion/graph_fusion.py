"""Graph-based fusion mechanisms for modeling biomarker interactions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class GraphFusion(nn.Module):
    """
    Main graph-based fusion module for modeling complex biomarker interactions
    Uses graph neural networks to capture relationships between modalities
    """
    
    def __init__(self,
                 modality_dims: Dict[str, int],
                 fusion_dim: int = 512,
                 num_graph_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.3,
                 use_heterogeneous: bool = True,
                 use_temporal: bool = True,
                 use_adaptive: bool = True):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.num_modalities = len(modality_dims)
        self.modality_names = list(modality_dims.keys())
        
        # Modality embeddings
        self.modality_embeddings = nn.Embedding(self.num_modalities, fusion_dim)
        
        # Project modalities to common dimension
        self.modality_projectors = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for modality, dim in modality_dims.items()
        })
        
        # Biomarker graph network
        self.biomarker_graph = BiomarkerGraphNetwork(
            node_dim=fusion_dim,
            edge_dim=fusion_dim // 2,
            num_layers=num_graph_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Heterogeneous graph fusion (different node/edge types)
        if use_heterogeneous:
            self.heterogeneous_fusion = HeterogeneousGraphFusion(
                node_dim=fusion_dim,
                num_node_types=self.num_modalities,
                num_edge_types=self.num_modalities * (self.num_modalities - 1) // 2,
                num_layers=2,
                dropout=dropout
            )
        else:
            self.heterogeneous_fusion = None
        
        # Temporal graph fusion
        if use_temporal:
            self.temporal_graph_fusion = TemporalGraphFusion(
                node_dim=fusion_dim,
                num_layers=2,
                dropout=dropout
            )
        else:
            self.temporal_graph_fusion = None
        
        # Adaptive graph structure learning
        if use_adaptive:
            self.adaptive_structure = AdaptiveGraphStructure(
                num_nodes=self.num_modalities,
                node_dim=fusion_dim,
                dropout=dropout
            )
        else:
            self.adaptive_structure = None
        
        # Modality graph attention
        self.modality_graph_attention = ModalityGraphAttention(
            dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Dynamic graph fusion
        self.dynamic_fusion = DynamicGraphFusion(
            node_dim=fusion_dim,
            num_layers=2,
            dropout=dropout
        )
        
        # Graph readout (aggregate node features)
        self.graph_readout = nn.Sequential(
            nn.Linear(fusion_dim * self.num_modalities, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
        # Edge importance learning
        self.edge_importance = nn.Parameter(
            torch.ones(self.num_modalities, self.num_modalities)
        )
        
    def build_graph(self,
                   modality_features: Dict[str, torch.Tensor],
                   adaptive: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph structure from modality features
        
        Args:
            modality_features: Dictionary of projected modality features
            adaptive: Whether to use adaptive graph structure
            
        Returns:
            node_features: [B, N, D] where N is number of nodes (modalities)
            adjacency: [B, N, N] adjacency matrix
        """
        batch_size = next(iter(modality_features.values())).shape[0]
        
        # Stack node features (modalities as nodes)
        node_features = []
        for modality in self.modality_names:
            if modality in modality_features:
                feat = modality_features[modality]
                # Pool temporal dimension if present
                if len(feat.shape) == 3:
                    feat = feat.mean(dim=1)
                node_features.append(feat)
        
        node_features = torch.stack(node_features, dim=1)  # [B, N, D]
        
        # Build adjacency matrix
        if adaptive and self.adaptive_structure is not None:
            adjacency = self.adaptive_structure(node_features)
        else:
            # Use learned edge importance as base adjacency
            adjacency = F.softmax(self.edge_importance, dim=-1)
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        
        return node_features, adjacency
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                temporal_info: Optional[torch.Tensor] = None,
                return_graph_info: bool = False) -> Dict[str, torch.Tensor]:
        """
        Fuse modalities using graph neural networks
        
        Args:
            modality_features: Dictionary of modality features
            temporal_info: Optional temporal information
            return_graph_info: Whether to return graph structure info
            
        Returns:
            Dictionary with fused features and optional graph info
        """
        batch_size = next(iter(modality_features.values())).shape[0]
        
        # Project all modalities
        projected_features = {}
        for modality, features in modality_features.items():
            if modality in self.modality_projectors:
                # Handle both 2D and 3D inputs
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)
                
                projected = self.modality_projectors[modality](features)
                projected_features[modality] = projected
        
        # Build graph structure
        node_features, adjacency = self.build_graph(projected_features, adaptive=True)
        
        # Apply biomarker graph network
        graph_output = self.biomarker_graph(
            node_features=node_features,
            adjacency=adjacency
        )
        
        # Apply heterogeneous graph fusion
        if self.heterogeneous_fusion is not None:
            # Create node types (one per modality)
            node_types = torch.arange(self.num_modalities).unsqueeze(0).expand(
                batch_size, -1
            ).to(node_features.device)
            
            hetero_output = self.heterogeneous_fusion(
                node_features=graph_output,
                node_types=node_types,
                adjacency=adjacency
            )
            graph_output = hetero_output
        
        # Apply temporal graph fusion if temporal info available
        if self.temporal_graph_fusion is not None and temporal_info is not None:
            temporal_output = self.temporal_graph_fusion(
                node_features=graph_output,
                temporal_info=temporal_info,
                adjacency=adjacency
            )
            graph_output = temporal_output
        
        # Apply modality graph attention
        attended_output = self.modality_graph_attention(
            node_features=graph_output,
            adjacency=adjacency
        )
        
        # Apply dynamic graph fusion
        dynamic_output = self.dynamic_fusion(
            node_features=attended_output,
            adjacency=adjacency
        )
        
        # Graph readout
        flattened = dynamic_output.reshape(batch_size, -1)
        fused_features = self.graph_readout(flattened)
        
        output = {
            'fused_features': fused_features,
            'node_features': dynamic_output,
            'adjacency': adjacency
        }
        
        if return_graph_info:
            output['graph_structure'] = adjacency
            output['edge_importance'] = self.edge_importance
            output['modality_interactions'] = self._compute_modality_interactions(adjacency)
        
        return output
    
    def _compute_modality_interactions(self, adjacency: torch.Tensor) -> Dict[str, float]:
        """Compute interaction strengths between modalities"""
        interactions = {}
        avg_adjacency = adjacency.mean(dim=0)  # Average over batch
        
        for i, mod1 in enumerate(self.modality_names):
            for j, mod2 in enumerate(self.modality_names):
                if i < j:
                    interaction_strength = avg_adjacency[i, j].item()
                    interactions[f"{mod1}_{mod2}"] = interaction_strength
        
        return interactions


class BiomarkerGraphNetwork(nn.Module):
    """Graph neural network for biomarker interaction modeling"""
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Graph convolutional layers
        self.graph_convs = nn.ModuleList([
            GraphConvLayer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Edge feature networks
        self.edge_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim * 2, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_dim, edge_dim)
            ) for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_layers)
        ])
        
    def forward(self,
                node_features: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, D]
            adjacency: [B, N, N]
        """
        x = node_features
        
        for conv, edge_net, norm in zip(self.graph_convs, self.edge_networks, self.layer_norms):
            # Compute edge features
            batch_size, num_nodes, _ = x.shape
            
            # Create pairwise node features
            x_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [B, N, N, D]
            x_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [B, N, N, D]
            edge_input = torch.cat([x_i, x_j], dim=-1)  # [B, N, N, 2D]
            
            # Compute edge features
            edge_features = edge_net(edge_input)  # [B, N, N, edge_dim]
            
            # Apply graph convolution
            x_new = conv(x, adjacency, edge_features)
            
            # Residual connection and normalization
            x = norm(x + x_new)
        
        return x


class GraphConvLayer(nn.Module):
    """Single graph convolutional layer with edge features"""
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        
        # Message passing
        self.message_net = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 2, node_dim)
        )
        
        # Node update
        self.update_net = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 2, node_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                node_features: torch.Tensor,
                adjacency: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, D]
            adjacency: [B, N, N]
            edge_features: [B, N, N, edge_dim]
        """
        batch_size, num_nodes, node_dim = node_features.shape
        
        # Aggregate messages from neighbors
        messages = []
        for i in range(num_nodes):
            # Get neighbors
            neighbor_weights = adjacency[:, i, :]  # [B, N]
            
            # Create message for each neighbor
            node_i = node_features[:, i:i+1, :].expand(-1, num_nodes, -1)  # [B, N, D]
            edge_feat = edge_features[:, i, :, :]  # [B, N, edge_dim]
            
            # Combine node and edge features
            message_input = torch.cat([node_i, edge_feat], dim=-1)
            message = self.message_net(message_input)  # [B, N, D]
            
            # Weight by adjacency
            weighted_message = message * neighbor_weights.unsqueeze(-1)
            
            # Aggregate
            aggregated = weighted_message.sum(dim=1)  # [B, D]
            messages.append(aggregated)
        
        messages = torch.stack(messages, dim=1)  # [B, N, D]
        
        # Update nodes
        update_input = torch.cat([node_features, messages], dim=-1)
        updated = self.update_net(update_input)
        
        return self.dropout(updated)


class ModalityGraphAttention(nn.Module):
    """Graph attention mechanism for modality interactions"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self,
                node_features: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, D]
            adjacency: [B, N, N]
        """
        B, N, D = node_features.shape
        
        # Project
        q = self.q_proj(node_features).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(node_features).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(node_features).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Mask with adjacency
        adjacency_mask = adjacency.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores * adjacency_mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual and norm
        output = self.layer_norm(node_features + output)
        
        return output


class DynamicGraphFusion(nn.Module):
    """Dynamic graph fusion with time-varying structure"""
    
    def __init__(self, node_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Dynamic edge weight predictor
        self.edge_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim * 2, node_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])
        
        # Graph update networks
        self.update_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim * 2, node_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim * 2, node_dim)
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_layers)
        ])
        
    def forward(self,
                node_features: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, D]
            adjacency: [B, N, N]
        """
        x = node_features
        
        for edge_pred, update_net, norm in zip(
            self.edge_predictors, self.update_networks, self.layer_norms
        ):
            batch_size, num_nodes, node_dim = x.shape
            
            # Predict dynamic edge weights
            x_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            x_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            edge_input = torch.cat([x_i, x_j], dim=-1)
            
            dynamic_weights = edge_pred(edge_input).squeeze(-1)  # [B, N, N]
            
            # Combine with base adjacency
            updated_adjacency = adjacency * dynamic_weights
            
            # Message passing
            messages = torch.bmm(updated_adjacency, x)  # [B, N, D]
            
            # Update nodes
            update_input = torch.cat([x, messages], dim=-1)
            x_new = update_net(update_input)
            
            # Residual and norm
            x = norm(x + x_new)
        
        return x


class HeterogeneousGraphFusion(nn.Module):
    """Heterogeneous graph fusion for different node/edge types"""
    
    def __init__(self,
                 node_dim: int,
                 num_node_types: int,
                 num_edge_types: int,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        
        # Type-specific transformations
        self.node_type_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.LayerNorm(node_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_node_types)
        ])
        
        # Edge type embeddings
        self.edge_type_embeddings = nn.Embedding(num_edge_types, node_dim)
        
        # Heterogeneous message passing
        self.message_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim * 3, node_dim * 2),  # node + neighbor + edge_type
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim * 2, node_dim)
            ) for _ in range(num_layers)
        ])
        
        # Type-aware aggregation
        self.aggregators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim * num_node_types, node_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim * 2, node_dim)
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_layers)
        ])
        
    def forward(self,
                node_features: torch.Tensor,
                node_types: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, D]
            node_types: [B, N] node type indices
            adjacency: [B, N, N]
        """
        x = node_features
        batch_size, num_nodes, node_dim = x.shape
        
        # Apply type-specific transformations
        type_specific_features = []
        for node_type_idx in range(self.num_node_types):
            mask = (node_types == node_type_idx).float().unsqueeze(-1)
            transformed = self.node_type_transforms[node_type_idx](x)
            type_specific_features.append(transformed * mask)
        
        x = sum(type_specific_features)
        
        for message_net, aggregator, norm in zip(
            self.message_networks, self.aggregators, self.layer_norms
        ):
            # Compute messages for each node type
            type_messages = []
            
            for target_type in range(self.num_node_types):
                target_mask = (node_types == target_type).float()
                
                # Aggregate from each source type
                source_aggregated = []
                for source_type in range(self.num_node_types):
                    source_mask = (node_types == source_type).float()
                    
                    # Get edge type (simplified: based on source-target pair)
                    edge_type_idx = source_type * self.num_node_types + target_type
                    edge_type_idx = min(edge_type_idx, self.num_edge_types - 1)
                    
                    edge_embedding = self.edge_type_embeddings(
                        torch.tensor([edge_type_idx]).to(x.device)
                    ).squeeze(0)
                    edge_embedding = edge_embedding.unsqueeze(0).unsqueeze(0).expand(
                        batch_size, num_nodes, -1
                    )
                    
                    # Create messages
                    x_expanded = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)
                    x_neighbors = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)
                    edge_expanded = edge_embedding.unsqueeze(2).expand(-1, -1, num_nodes, -1)
                    
                    message_input = torch.cat([
                        x_expanded,
                        x_neighbors,
                        edge_expanded
                    ], dim=-1)
                    
                    messages = message_net(message_input)  # [B, N, N, D]
                    
                    # Weight by adjacency and type masks
                    type_adjacency = adjacency * source_mask.unsqueeze(1) * target_mask.unsqueeze(-1)
                    weighted_messages = messages * type_adjacency.unsqueeze(-1)
                    
                    # Aggregate
                    aggregated = weighted_messages.sum(dim=2)  # [B, N, D]
                    source_aggregated.append(aggregated)
                
                # Combine messages from all source types
                combined = torch.cat(source_aggregated, dim=-1)
                aggregated_messages = aggregator(combined)
                
                # Apply to target type only
                type_messages.append(aggregated_messages * target_mask.unsqueeze(-1))
            
            # Sum messages for all types
            all_messages = sum(type_messages)
            
            # Update
            x = norm(x + all_messages)
        
        return x


class TemporalGraphFusion(nn.Module):
    """Temporal graph fusion for time-evolving graphs"""
    
    def __init__(self, node_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Temporal encoding
        self.temporal_encoder = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal GRU for graph evolution
        self.temporal_gru = nn.GRU(
            input_size=node_dim,
            hidden_size=node_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(node_dim)
        
    def forward(self,
                node_features: torch.Tensor,
                temporal_info: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, D]
            temporal_info: [B, T] temporal context
            adjacency: [B, N, N]
        """
        # Encode temporal information
        temporal_encoded = self.temporal_encoder(
            temporal_info.unsqueeze(-1).expand(-1, -1, node_features.shape[-1])
        )
        
        # Apply temporal GRU
        temporal_output, _ = self.temporal_gru(temporal_encoded)
        
        # Aggregate temporal context to nodes
        temporal_context = temporal_output.mean(dim=1, keepdim=True).expand(
            -1, node_features.shape[1], -1
        )
        
        # Temporal attention
        attn_output, _ = self.temporal_attention(
            node_features,
            temporal_context,
            temporal_context
        )
        
        # Update nodes
        updated = self.layer_norm(node_features + attn_output)
        
        return updated


class AdaptiveGraphStructure(nn.Module):
    """Learn adaptive graph structure from node features"""
    
    def __init__(self, num_nodes: int, node_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.num_nodes = num_nodes
        
        # Learnable structure parameters
        self.structure_net = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, 1)
        )
        
        # Sparsity control
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, D]
        Returns:
            adjacency: [B, N, N]
        """
        batch_size, num_nodes, node_dim = node_features.shape
        
        # Compute pairwise affinities
        x_i = node_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        x_j = node_features.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        pairwise = torch.cat([x_i, x_j], dim=-1)
        affinities = self.structure_net(pairwise).squeeze(-1)  # [B, N, N]
        
        # Apply sparsity
        adjacency = torch.sigmoid(affinities)
        adjacency = adjacency * (adjacency > self.sparsity_threshold).float()
        
        # Symmetrize
        adjacency = (adjacency + adjacency.transpose(-1, -2)) / 2
        
        # Add self-loops
        eye = torch.eye(num_nodes).unsqueeze(0).to(adjacency.device)
        adjacency = adjacency + eye
        
        # Normalize
        degree = adjacency.sum(dim=-1, keepdim=True)
        adjacency = adjacency / (degree + 1e-8)
        
        return adjacency