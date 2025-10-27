"""Neural Architecture Search for optimal biomarker model architectures"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class NeuralArchitectureSearch(nn.Module):
    """
    Main NAS module that searches for optimal biomarker architectures
    using differentiable architecture search (DARTS)
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 search_space: str = 'darts',
                 num_cells: int = 8,
                 num_nodes: int = 4,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Create search space
        if search_space == 'darts':
            self.search_space = DARTSSearchSpace(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_cells=num_cells,
                num_nodes=num_nodes,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown search space: {search_space}")
        
        # Architecture controller
        self.controller = BiomarkerNASController(
            num_cells=num_cells,
            num_nodes=num_nodes,
            num_ops=len(PRIMITIVE_OPS),
            hidden_dim=hidden_dim
        )
        
        # Architecture evaluator
        self.evaluator = ArchitectureEvaluator(
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    
    def forward(self,
                x: torch.Tensor,
                arch_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through searched architecture
        
        Args:
            x: Input tensor [batch, input_dim]
            arch_weights: Optional architecture weights (if None, use controller)
        """
        # Get architecture weights from controller if not provided
        if arch_weights is None:
            arch_weights = self.controller.sample_architecture()
        
        # Forward through search space
        features = self.search_space(x, arch_weights)
        
        # Evaluate architecture
        output = self.evaluator(features)
        
        return {
            'logits': output,
            'features': features,
            'arch_weights': arch_weights
        }
    
    def get_best_architecture(self) -> Dict[str, Any]:
        """Get the best architecture found during search"""
        return self.controller.get_best_architecture()


# Primitive operations for search space
PRIMITIVE_OPS = [
    'none',
    'skip_connect',
    'conv_3x1',
    'conv_5x1',
    'conv_7x1',
    'dilated_conv_3x1',
    'dilated_conv_5x1',
    'sep_conv_3x1',
    'sep_conv_5x1',
    'avg_pool_3x1',
    'max_pool_3x1',
    'mlp',
    'attention',
    'gated_linear',
]


class DARTSSearchSpace(nn.Module):
    """
    Differentiable Architecture Search (DARTS) space for biomarkers
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_cells: int = 8,
                 num_nodes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        
        # Input stem
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Searchable cells
        self.cells = nn.ModuleList()
        for i in range(num_cells):
            reduction = (i in [num_cells // 3, 2 * num_cells // 3])
            cell = SearchCell(
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                reduction=reduction,
                dropout=dropout
            )
            self.cells.append(cell)
        
        # Architecture parameters (alpha)
        num_edges = sum(range(2, num_nodes + 2))
        num_ops = len(PRIMITIVE_OPS)
        self.alphas_normal = nn.Parameter(torch.randn(num_edges, num_ops))
        self.alphas_reduce = nn.Parameter(torch.randn(num_edges, num_ops))
        
        # Output projection
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self,
                x: torch.Tensor,
                arch_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through DARTS space
        
        Args:
            x: Input tensor [batch, input_dim]
            arch_weights: Optional fixed architecture weights
        """
        # Stem
        s0 = s1 = self.stem(x)
        
        # Add dimension for conv operations
        s0 = s0.unsqueeze(-1)
        s1 = s1.unsqueeze(-1)
        
        # Forward through cells
        for i, cell in enumerate(self.cells):
            # Get architecture weights
            if arch_weights is not None:
                weights = arch_weights[i]
            else:
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            
            # Cell forward
            s0, s1 = s1, cell(s0, s1, weights)
        
        # Global pooling
        out = self.global_pooling(s1).squeeze(-1)
        
        return out
    
    def arch_parameters(self):
        """Get architecture parameters for optimization"""
        return [self.alphas_normal, self.alphas_reduce]
    
    def genotype(self):
        """Get discrete genotype from continuous architecture"""
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.num_nodes):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                             key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                              if k != PRIMITIVE_OPS.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVE_OPS.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVE_OPS[k_best], j))
                start = end
                n += 1
            return gene
        
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        
        return {'normal': gene_normal, 'reduce': gene_reduce}


class SearchCell(nn.Module):
    """A searchable cell in DARTS"""
    
    def __init__(self,
                 hidden_dim: int,
                 num_nodes: int = 4,
                 reduction: bool = False,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.reduction = reduction
        self.hidden_dim = hidden_dim
        
        # Build mixed operations for all edges
        self.ops = nn.ModuleList()
        for i in range(num_nodes):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(hidden_dim, stride, dropout)
                self.ops.append(op)
    
    def forward(self,
                s0: torch.Tensor,
                s1: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s0: Previous-previous cell output
            s1: Previous cell output
            weights: Architecture weights for this cell
        """
        states = [s0, s1]
        offset = 0
        
        for i in range(self.num_nodes):
            s = sum(self.ops[offset + j](h, weights[offset + j])
                   for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        
        return torch.cat(states[-self.num_nodes:], dim=1)


class MixedOp(nn.Module):
    """Mixed operation combining all primitive ops"""
    
    def __init__(self, hidden_dim: int, stride: int = 1, dropout: float = 0.3):
        super().__init__()
        self._ops = nn.ModuleList()
        
        for primitive in PRIMITIVE_OPS:
            op = OPS[primitive](hidden_dim, stride, dropout)
            self._ops.append(op)
    
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Weighted sum of operations
        
        Args:
            x: Input tensor
            weights: Operation weights (softmax of architecture parameters)
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


# Operation definitions
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride].mul(0.)


class ReLUConv1D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride,
                     padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(C_out)
        )
    
    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """Separable convolution"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=C_in, bias=False),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out),
        )
    
    def forward(self, x):
        return self.op(x)


class MLPOp(nn.Module):
    """MLP operation"""
    def __init__(self, hidden_dim, stride, dropout):
        super().__init__()
        self.stride = stride
        self.op = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
    
    def forward(self, x):
        # Remove channel dim, apply MLP, restore
        x_flat = x.squeeze(-1)
        out = self.op(x_flat)
        return out.unsqueeze(-1)


class AttentionOp(nn.Module):
    """Self-attention operation"""
    def __init__(self, hidden_dim, stride, dropout):
        super().__init__()
        self.stride = stride
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # Convert from [batch, channels, seq] to [batch, seq, channels]
        x = x.transpose(1, 2)
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(x + attn_out)
        return out.transpose(1, 2)


class GatedLinear(nn.Module):
    """Gated linear unit"""
    def __init__(self, hidden_dim, stride, dropout):
        super().__init__()
        self.stride = stride
        self.linear = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x_flat = x.squeeze(-1)
        out = self.linear(x_flat)
        value, gate = out.chunk(2, dim=-1)
        out = value * torch.sigmoid(gate)
        out = self.dropout(out)
        return out.unsqueeze(-1)


# Operation dictionary
OPS = {
    'none': lambda C, stride, dropout: Zero(stride),
    'skip_connect': lambda C, stride, dropout: Identity() if stride == 1 else
                    nn.AvgPool1d(kernel_size=stride, stride=stride),
    'conv_3x1': lambda C, stride, dropout: ReLUConv1D(C, C, 3, stride, 1),
    'conv_5x1': lambda C, stride, dropout: ReLUConv1D(C, C, 5, stride, 2),
    'conv_7x1': lambda C, stride, dropout: ReLUConv1D(C, C, 7, stride, 3),
    'dilated_conv_3x1': lambda C, stride, dropout: ReLUConv1D(C, C, 3, stride, 2, dilation=2),
    'dilated_conv_5x1': lambda C, stride, dropout: ReLUConv1D(C, C, 5, stride, 4, dilation=2),
    'sep_conv_3x1': lambda C, stride, dropout: SepConv(C, C, 3, stride, 1),
    'sep_conv_5x1': lambda C, stride, dropout: SepConv(C, C, 5, stride, 2),
    'avg_pool_3x1': lambda C, stride, dropout: nn.AvgPool1d(3, stride=stride, padding=1),
    'max_pool_3x1': lambda C, stride, dropout: nn.MaxPool1d(3, stride=stride, padding=1),
    'mlp': lambda C, stride, dropout: MLPOp(C, stride, dropout),
    'attention': lambda C, stride, dropout: AttentionOp(C, stride, dropout),
    'gated_linear': lambda C, stride, dropout: GatedLinear(C, stride, dropout),
}


class ENASController(nn.Module):
    """
    ENAS (Efficient Neural Architecture Search) controller using REINFORCE
    """
    
    def __init__(self,
                 num_layers: int = 12,
                 num_branches: int = 6,
                 hidden_dim: int = 256,
                 temperature: float = 5.0):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_branches = num_branches
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # LSTM controller
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # Embedding for architecture decisions
        self.g_emb = nn.Embedding(1, hidden_dim)
        
        # Decoders for architecture decisions
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            # Branch selection decoder
            self.decoders.append(nn.Linear(hidden_dim, num_branches))
    
    def forward(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample architecture and compute log probabilities
        
        Returns:
            architectures: Sampled architecture decisions
            log_probs: Log probabilities of decisions
            entropies: Entropy of distributions
        """
        # Initialize LSTM state
        h = torch.zeros(batch_size, self.hidden_dim).to(self.g_emb.weight.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(self.g_emb.weight.device)
        
        # Start token
        inputs = self.g_emb.weight.unsqueeze(0).expand(batch_size, -1)
        
        architectures = []
        log_probs = []
        entropies = []
        
        # Sample architecture decisions
        for layer_id in range(self.num_layers):
            # LSTM step
            h, c = self.lstm(inputs, (h, c))
            
            # Decode branch selection
            logits = self.decoders[layer_id](h)
            
            # Sample with temperature
            probs = F.softmax(logits / self.temperature, dim=-1)
            
            # Sample action
            action = torch.multinomial(probs, 1).squeeze(-1)
            
            # Compute log prob
            log_prob = F.log_softmax(logits, dim=-1)
            selected_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
            
            # Compute entropy
            entropy = -(probs * log_prob).sum(-1)
            
            architectures.append(action)
            log_probs.append(selected_log_prob)
            entropies.append(entropy)
            
            # Next input is the sampled action embedding
            inputs = h  # Use hidden state as next input
        
        architectures = torch.stack(architectures, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        
        return architectures, log_probs, entropies


class BiomarkerNASController(nn.Module):
    """
    Biomarker-specific NAS controller that learns to sample optimal architectures
    """
    
    def __init__(self,
                 num_cells: int = 8,
                 num_nodes: int = 4,
                 num_ops: int = 14,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        self.hidden_dim = hidden_dim
        
        # Calculate number of edges per cell
        self.num_edges = sum(range(2, num_nodes + 2))
        
        # Architecture encoding network
        self.arch_encoder = nn.Sequential(
            nn.Linear(num_cells * self.num_edges * num_ops, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Architecture parameters (learnable)
        self.arch_params = nn.Parameter(
            torch.randn(num_cells, self.num_edges, num_ops) * 0.01
        )
        
        # Performance predictor
        self.performance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Architecture memory (store good architectures)
        self.register_buffer('best_arch', torch.zeros(num_cells, self.num_edges, num_ops))
        self.register_buffer('best_performance', torch.tensor(0.0))
    
    def sample_architecture(self,
                          temperature: float = 1.0,
                          hard: bool = False) -> torch.Tensor:
        """
        Sample architecture using Gumbel-Softmax
        
        Args:
            temperature: Temperature for sampling
            hard: Whether to use hard (discrete) sampling
        """
        if self.training:
            # Gumbel-Softmax sampling
            arch_weights = F.gumbel_softmax(
                self.arch_params,
                tau=temperature,
                hard=hard,
                dim=-1
            )
        else:
            # Use best architecture during evaluation
            if self.best_performance > 0:
                arch_weights = self.best_arch
            else:
                arch_weights = F.softmax(self.arch_params, dim=-1)
        
        return arch_weights
    
    def update_best_architecture(self, performance: float):
        """Update best architecture based on performance"""
        if performance > self.best_performance:
            # self.best_performance = 
            self.best_performance.fill_(performance)
            self.best_arch = F.softmax(self.arch_params, dim=-1).detach()
    
    def get_best_architecture(self) -> Dict[str, Any]:
        """Get the best architecture found"""
        arch_weights = self.best_arch.cpu().numpy()
        
        # Decode architecture
        architecture = []
        for cell_id in range(self.num_cells):
            cell_arch = []
            for edge_id in range(self.num_edges):
                op_id = arch_weights[cell_id, edge_id].argmax()
                op_name = PRIMITIVE_OPS[op_id]
                op_prob = arch_weights[cell_id, edge_id, op_id]
                cell_arch.append({
                    'edge': edge_id,
                    'operation': op_name,
                    'probability': float(op_prob)
                })
            architecture.append(cell_arch)
        
        return {
            'architecture': architecture,
            'performance': float(self.best_performance),
            'arch_weights': arch_weights
        }


class ArchitectureEvaluator(nn.Module):
    """
    Evaluate architectures for biomarker discovery
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 output_dim: int = 10):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Evaluation head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Evaluate features from architecture
        
        Args:
            features: Features from searched architecture
        """
        logits = self.classifier(features)
        return logits


class SuperNet(nn.Module):
    """
    SuperNet containing all possible architectures for efficient search
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input stem
        self.stem = nn.Linear(input_dim, hidden_dim)
        
        # Superposition of all operations at each layer
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_ops = nn.ModuleDict()
            for op_name in PRIMITIVE_OPS:
                if op_name != 'none':
                    layer_ops[op_name] = self._create_operation(
                        op_name, hidden_dim, dropout
                    )
            self.layers.append(layer_ops)
        
        # Output head
        self.head = nn.Linear(hidden_dim, output_dim)
    
    def _create_operation(self, op_name: str, hidden_dim: int, dropout: float):
        """Create operation module"""
        if op_name == 'skip_connect':
            return Identity()
        elif 'conv' in op_name:
            # kernel_size = int(op_name.split('_')[1][0])
            import re
            match = re.search(r'_(\d+)x\d+', op_name) # Find the first digit after '_' and before 'x'
            if match:
                kernel_size = int(match.group(1))
            else:
                kernel_size = 3 # Default kernel size if pattern doesn't match
            return nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        elif 'pool' in op_name:
            return nn.AdaptiveAvgPool1d(1)
        elif op_name == 'mlp':
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        elif op_name == 'attention':
            return nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        else:
            return Identity()
    
    def forward(self,
                x: torch.Tensor,
                arch_encoding: List[str]) -> torch.Tensor:
        """
        Forward pass with specific architecture
        
        Args:
            x: Input tensor
            arch_encoding: List of operation names for each layer
        """
        h = self.stem(x)
        
        for i, op_name in enumerate(arch_encoding):
            if i < len(self.layers) and op_name in self.layers[i]:
                h = self.layers[i][op_name](h)
        
        return self.head(h)


class DifferentiableArchitecture(nn.Module):
    """
    Differentiable architecture with continuous relaxation
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_blocks: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Architecture weights (alpha)
        self.arch_weights = nn.ParameterList([
            nn.Parameter(torch.randn(len(PRIMITIVE_OPS)))
            for _ in range(num_blocks)
        ])
        
        # Mixed operations for each block
        self.blocks = nn.ModuleList([
            MixedOp(hidden_dim, stride=1, dropout=0.3)
            for _ in range(num_blocks)
        ])
        
        # Input/output projections
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward with architecture search"""
        h = self.input_proj(x).unsqueeze(-1)
        
        arch_probs = []
        for i, (block, alpha) in enumerate(zip(self.blocks, self.arch_weights)):
            weights = F.softmax(alpha, dim=-1)
            arch_probs.append(weights)
            h = block(h, weights)
        
        h = h.squeeze(-1)
        logits = self.output_proj(h)
        
        return {
            'logits': logits,
            'arch_probs': torch.stack(arch_probs),
            'features': h
        }
    
    def discretize_architecture(self) -> List[str]:
        """Get discrete architecture from continuous weights"""
        discrete_arch = []
        for alpha in self.arch_weights:
            op_id = alpha.argmax().item()
            op_name = PRIMITIVE_OPS[op_id]
            discrete_arch.append(op_name)
        return discrete_arch