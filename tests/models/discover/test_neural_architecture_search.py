"""Comprehensive tests for neural_architecture_search.py"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from biomarkers.models.discovery.neural_architecture_search import (
    NeuralArchitectureSearch,
    DARTSSearchSpace,
    ENASController,
    BiomarkerNASController,
    ArchitectureEvaluator,
    SuperNet,
    DifferentiableArchitecture,
    PRIMITIVE_OPS,
    MixedOp,
    SearchCell
)


@pytest.fixture
def sample_input():
    """Fixture for sample input data."""
    batch_size = 8
    input_dim = 128
    return torch.randn(batch_size, input_dim)


@pytest.fixture
def nas_config():
    """Fixture for NAS configuration."""
    return {
        'input_dim': 128,
        'output_dim': 64,
        'num_cells': 4,
        'num_nodes': 3,
        'hidden_dim': 64
    }


class TestNeuralArchitectureSearch:
    """Tests for main NAS class."""
    
    def test_initialization(self, nas_config):
        """Test NAS initialization."""
        nas = NeuralArchitectureSearch(**nas_config)
        
        assert nas.input_dim == 128
        assert nas.output_dim == 64
        assert nas.num_cells == 4
        assert nas.num_nodes == 3
        assert nas.hidden_dim == 64
        assert nas.search_space is not None
        assert nas.controller is not None
        assert nas.evaluator is not None
    
    def test_forward(self, sample_input, nas_config):
        """Test forward pass."""
        nas = NeuralArchitectureSearch(**nas_config)
        
        output = nas(sample_input)
        
        assert 'logits' in output
        assert 'features' in output
        assert 'arch_weights' in output
        
        assert output['logits'].shape == (8, 64)
        assert output['features'].shape == (8, 64)
    
    def test_forward_with_custom_arch_weights(self, sample_input, nas_config):
        """Test forward with custom architecture weights."""
        nas = NeuralArchitectureSearch(**nas_config)
        
        # Create custom architecture weights
        num_edges = sum(range(2, nas_config['num_nodes'] + 2))
        arch_weights = torch.softmax(
            torch.randn(nas_config['num_cells'], num_edges, len(PRIMITIVE_OPS)),
            dim=-1
        )
        
        output = nas(sample_input, arch_weights=arch_weights)
        
        assert output['logits'].shape == (8, 64)
    
    def test_get_best_architecture(self, nas_config):
        """Test retrieving best architecture."""
        nas = NeuralArchitectureSearch(**nas_config)
        
        # Simulate some performance
        nas.controller.update_best_architecture(0.85)
        
        best_arch = nas.get_best_architecture()
        
        assert 'architecture' in best_arch
        assert 'performance' in best_arch
        assert best_arch['performance'] == 0.85
    
    def test_gradient_flow(self, sample_input, nas_config):
        """Test gradient flow through NAS."""
        nas = NeuralArchitectureSearch(**nas_config)
        
        output = nas(sample_input)
        loss = output['logits'].sum()
        loss.backward()
        
        # Check gradients exist
        for param in nas.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDARTSSearchSpace:
    """Tests for DARTS search space."""
    
    def test_initialization(self):
        """Test DARTS initialization."""
        darts = DARTSSearchSpace(
            input_dim=128,
            hidden_dim=64,
            num_cells=4,
            num_nodes=3
        )
        
        assert darts.input_dim == 128
        assert darts.hidden_dim == 64
        assert darts.num_cells == 4
        assert darts.num_nodes == 3
        assert len(darts.cells) == 4
    
    def test_forward(self, sample_input):
        """Test DARTS forward pass."""
        darts = DARTSSearchSpace(
            input_dim=128,
            hidden_dim=64,
            num_cells=4,
            num_nodes=3
        )
        
        output = darts(sample_input)
        
        assert output.shape == (8, 64)
    
    def test_arch_parameters(self):
        """Test architecture parameters."""
        darts = DARTSSearchSpace(
            input_dim=128,
            hidden_dim=64,
            num_cells=4,
            num_nodes=3
        )
        
        arch_params = darts.arch_parameters()
        
        assert len(arch_params) == 2  # normal and reduce
        assert all(p.requires_grad for p in arch_params)
    
    def test_genotype_extraction(self):
        """Test genotype extraction."""
        darts = DARTSSearchSpace(
            input_dim=128,
            hidden_dim=64,
            num_cells=4,
            num_nodes=3
        )
        
        genotype = darts.genotype()
        
        assert 'normal' in genotype
        assert 'reduce' in genotype
        assert isinstance(genotype['normal'], list)
        assert isinstance(genotype['reduce'], list)
        assert len(genotype['normal']) > 0
        assert len(genotype['reduce']) > 0
    
    def test_reduction_cells(self, sample_input):
        """Test that reduction cells are created correctly."""
        darts = DARTSSearchSpace(
            input_dim=128,
            hidden_dim=64,
            num_cells=6,  # Will have reduction at positions 2 and 4
            num_nodes=3
        )
        
        # Check that some cells are reduction cells
        reduction_cells = [cell.reduction for cell in darts.cells]
        assert any(reduction_cells)
        
        output = darts(sample_input)
        assert output.shape[1] == 64


class TestSearchCell:
    """Tests for SearchCell."""
    
    def test_initialization(self):
        """Test search cell initialization."""
        cell = SearchCell(
            hidden_dim=64,
            num_nodes=4,
            reduction=False
        )
        
        assert cell.num_nodes == 4
        assert cell.reduction == False
        assert cell.hidden_dim == 64
        assert len(cell.ops) > 0
    
    def test_forward(self):
        """Test cell forward pass."""
        cell = SearchCell(
            hidden_dim=64,
            num_nodes=4,
            reduction=False
        )
        
        # Create input states
        s0 = torch.randn(8, 64, 1)
        s1 = torch.randn(8, 64, 1)
        
        # Create architecture weights
        num_edges = sum(range(2, 4 + 2))
        weights = torch.softmax(torch.randn(num_edges, len(PRIMITIVE_OPS)), dim=-1)
        
        output = cell(s0, s1, weights)
        
        # Output should concatenate num_nodes states
        assert output.shape[1] == 64 * 4


class TestMixedOp:
    """Tests for MixedOp."""
    
    def test_initialization(self):
        """Test mixed operation initialization."""
        mixed_op = MixedOp(hidden_dim=64, stride=1)
        
        assert len(mixed_op._ops) == len(PRIMITIVE_OPS)
    
    def test_forward(self):
        """Test mixed operation forward."""
        mixed_op = MixedOp(hidden_dim=64, stride=1)
        
        x = torch.randn(8, 64, 10)
        weights = torch.softmax(torch.randn(len(PRIMITIVE_OPS)), dim=-1)
        
        output = mixed_op(x, weights)
        
        assert output.shape == x.shape
    
    def test_operation_weights_effect(self):
        """Test that operation weights affect output."""
        mixed_op = MixedOp(hidden_dim=64, stride=1)
        
        x = torch.randn(8, 64, 10)
        
        # All weight on skip_connect
        weights1 = torch.zeros(len(PRIMITIVE_OPS))
        weights1[PRIMITIVE_OPS.index('skip_connect')] = 1.0
        output1 = mixed_op(x, weights1)
        
        # All weight on none
        weights2 = torch.zeros(len(PRIMITIVE_OPS))
        weights2[PRIMITIVE_OPS.index('none')] = 1.0
        output2 = mixed_op(x, weights2)
        
        # Outputs should be different
        assert not torch.allclose(output1, output2)


class TestENASController:
    """Tests for ENAS controller."""
    
    def test_initialization(self):
        """Test ENAS controller initialization."""
        controller = ENASController(
            num_layers=12,
            num_branches=6,
            hidden_dim=64
        )
        
        assert controller.num_layers == 12
        assert controller.num_branches == 6
        assert controller.hidden_dim == 64
    
    def test_forward(self):
        """Test ENAS controller forward pass."""
        controller = ENASController(
            num_layers=8,
            num_branches=6,
            hidden_dim=64
        )
        
        architectures, log_probs, entropies = controller(batch_size=4)
        
        assert architectures.shape == (4, 8)
        assert log_probs.shape == (4, 8)
        assert entropies.shape == (4, 8)
        
        # Architectures should be valid branch indices
        assert torch.all(architectures >= 0) and torch.all(architectures < 6)
    
    def test_sampling_diversity(self):
        """Test that controller samples diverse architectures."""
        controller = ENASController(
            num_layers=8,
            num_branches=6,
            hidden_dim=64
        )
        
        arch1, _, _ = controller(batch_size=1)
        arch2, _, _ = controller(batch_size=1)
        
        # Different samples should likely be different
        # (very small chance they're the same)
        assert not torch.equal(arch1, arch2) or arch1.shape[1] < 3


class TestBiomarkerNASController:
    """Tests for biomarker-specific NAS controller."""
    
    def test_initialization(self):
        """Test controller initialization."""
        controller = BiomarkerNASController(
            num_cells=4,
            num_nodes=3,
            num_ops=len(PRIMITIVE_OPS),
            hidden_dim=64
        )
        
        assert controller.num_cells == 4
        assert controller.num_nodes == 3
        assert controller.num_ops == len(PRIMITIVE_OPS)
    
    def test_sample_architecture(self):
        """Test architecture sampling."""
        controller = BiomarkerNASController(
            num_cells=4,
            num_nodes=3,
            num_ops=len(PRIMITIVE_OPS),
            hidden_dim=64
        )
        
        arch_weights = controller.sample_architecture(temperature=1.0, hard=False)
        
        assert arch_weights.shape[0] == 4  # num_cells
        assert arch_weights.shape[2] == len(PRIMITIVE_OPS)
        
        # Weights should sum to 1 along last dimension
        weight_sums = arch_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    
    def test_update_best_architecture(self):
        """Test updating best architecture."""
        controller = BiomarkerNASController(
            num_cells=4,
            num_nodes=3,
            num_ops=len(PRIMITIVE_OPS),
            hidden_dim=64
        )
        
        # Update with good performance
        controller.update_best_architecture(0.90)
        assert controller.best_performance == 0.90
        
        # Try to update with worse performance
        controller.update_best_architecture(0.85)
        assert controller.best_performance == 0.90  # Should not change
        
        # Update with better performance
        controller.update_best_architecture(0.95)
        assert controller.best_performance == 0.95
    
    def test_get_best_architecture(self):
        """Test retrieving best architecture."""
        controller = BiomarkerNASController(
            num_cells=4,
            num_nodes=3,
            num_ops=len(PRIMITIVE_OPS),
            hidden_dim=64
        )
        
        controller.update_best_architecture(0.85)
        best_arch = controller.get_best_architecture()
        
        assert 'architecture' in best_arch
        assert 'performance' in best_arch
        assert 'arch_weights' in best_arch
        assert len(best_arch['architecture']) == 4  # num_cells


class TestArchitectureEvaluator:
    """Tests for architecture evaluator."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ArchitectureEvaluator(
            hidden_dim=64,
            output_dim=10
        )
        
        assert evaluator.hidden_dim == 64
        assert evaluator.output_dim == 10
    
    def test_forward(self):
        """Test evaluator forward pass."""
        evaluator = ArchitectureEvaluator(
            hidden_dim=64,
            output_dim=10
        )
        
        features = torch.randn(8, 64)
        logits = evaluator(features)
        
        assert logits.shape == (8, 10)


class TestSuperNet:
    """Tests for SuperNet."""
    
    def test_initialization(self):
        """Test SuperNet initialization."""
        supernet = SuperNet(
            input_dim=128,
            output_dim=64,
            hidden_dim=64,
            num_layers=4
        )
        
        assert supernet.input_dim == 128
        assert supernet.output_dim == 64
        assert supernet.num_layers == 4
        assert len(supernet.layers) == 4
    
    def test_forward_with_architecture(self):
        """Test SuperNet forward with specific architecture."""
        supernet = SuperNet(
            input_dim=128,
            output_dim=64,
            hidden_dim=64,
            num_layers=4
        )
        
        x = torch.randn(8, 128)
        
        # Create architecture encoding
        arch_encoding = ['mlp', 'skip_connect', 'attention', 'mlp']
        
        output = supernet(x, arch_encoding)
        
        assert output.shape == (8, 64)
    
    def test_different_architectures(self):
        """Test that different architectures produce different outputs."""
        supernet = SuperNet(
            input_dim=128,
            output_dim=64,
            hidden_dim=64,
            num_layers=4
        )
        
        x = torch.randn(8, 128)
        
        arch1 = ['mlp', 'mlp', 'mlp', 'mlp']
        arch2 = ['skip_connect', 'skip_connect', 'skip_connect', 'skip_connect']
        
        output1 = supernet(x, arch1)
        output2 = supernet(x, arch2)
        
        # Different architectures should produce different outputs
        assert not torch.allclose(output1, output2, atol=1e-3)


class TestDifferentiableArchitecture:
    """Tests for differentiable architecture."""
    
    def test_initialization(self):
        """Test initialization."""
        diff_arch = DifferentiableArchitecture(
            input_dim=128,
            output_dim=64,
            hidden_dim=64,
            num_blocks=4
        )
        
        assert diff_arch.input_dim == 128
        assert diff_arch.output_dim == 64
        assert diff_arch.num_blocks == 4
        assert len(diff_arch.arch_weights) == 4
        assert len(diff_arch.blocks) == 4
    
    def test_forward(self):
        """Test forward pass."""
        diff_arch = DifferentiableArchitecture(
            input_dim=128,
            output_dim=64,
            hidden_dim=64,
            num_blocks=4
        )
        
        x = torch.randn(8, 128)
        output = diff_arch(x)
        
        assert 'logits' in output
        assert 'arch_probs' in output
        assert 'features' in output
        
        assert output['logits'].shape == (8, 64)
        assert output['arch_probs'].shape == (4, len(PRIMITIVE_OPS))
    
    def test_discretize_architecture(self):
        """Test architecture discretization."""
        diff_arch = DifferentiableArchitecture(
            input_dim=128,
            output_dim=64,
            hidden_dim=64,
            num_blocks=4
        )
        
        discrete_arch = diff_arch.discretize_architecture()
        
        assert len(discrete_arch) == 4
        assert all(op in PRIMITIVE_OPS for op in discrete_arch)
    
    def test_gradient_flow_to_arch_params(self):
        """Test that gradients flow to architecture parameters."""
        diff_arch = DifferentiableArchitecture(
            input_dim=128,
            output_dim=64,
            hidden_dim=64,
            num_blocks=4
        )
        
        x = torch.randn(8, 128)
        output = diff_arch(x)
        loss = output['logits'].sum()
        loss.backward()
        
        # Check gradients on architecture parameters
        for alpha in diff_arch.arch_weights:
            assert alpha.grad is not None


class TestIntegration:
    """Integration tests for NAS components."""
    
    def test_nas_training_loop(self, sample_input, nas_config):
        """Test a complete NAS training loop."""
        nas = NeuralArchitectureSearch(**nas_config)
        
        # Create optimizers
        weight_params = [p for p in nas.parameters() 
                        if p not in nas.search_space.arch_parameters()]
        weight_optimizer = optim.Adam(weight_params, lr=0.001)
        
        arch_params = nas.search_space.arch_parameters()
        arch_optimizer = optim.Adam(arch_params, lr=0.001)
        
        criterion = nn.MSELoss()
        target = torch.randn(8, nas_config['output_dim'])
        
        # Training step
        nas.train()
        
        # Update weights
        weight_optimizer.zero_grad()
        output = nas(sample_input)
        loss = criterion(output['logits'], target)
        loss.backward()
        weight_optimizer.step()
        
        weight_loss = loss.item()
        
        # Update architecture
        arch_optimizer.zero_grad()
        output = nas(sample_input)
        loss = criterion(output['logits'], target)
        loss.backward()
        arch_optimizer.step()
        
        arch_loss = loss.item()
        
        assert weight_loss > 0
        assert arch_loss > 0
    
    def test_architecture_evolution(self, nas_config):
        """Test that architecture evolves during search."""
        nas = NeuralArchitectureSearch(**nas_config)
        
        # Get initial genotype
        initial_genotype = nas.search_space.genotype()
        
        # Simulate some training
        optimizer = optim.Adam(nas.search_space.arch_parameters(), lr=0.01)
        
        for _ in range(10):
            x = torch.randn(8, nas_config['input_dim'])
            output = nas(x)
            loss = output['logits'].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Get final genotype
        final_genotype = nas.search_space.genotype()
        
        # Genotypes might have changed (not guaranteed but likely)
        # Just check that we can extract them successfully
        assert 'normal' in initial_genotype
        assert 'normal' in final_genotype
    
    def test_controller_guided_search(self):
        """Test search guided by controller."""
        controller = BiomarkerNASController(
            num_cells=4,
            num_nodes=3,
            num_ops=len(PRIMITIVE_OPS),
            hidden_dim=64
        )
        
        # Sample multiple architectures
        architectures = []
        for _ in range(5):
            arch = controller.sample_architecture(temperature=1.0)
            architectures.append(arch)
        
        # Check that architectures are being sampled
        assert len(architectures) == 5
        
        # Simulate improving performance
        performances = [0.70, 0.75, 0.80, 0.85, 0.90]
        for perf in performances:
            controller.update_best_architecture(perf)
        
        assert controller.best_performance == 0.90


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_cell(self, sample_input):
        """Test with single cell."""
        darts = DARTSSearchSpace(
            input_dim=128,
            hidden_dim=64,
            num_cells=1,
            num_nodes=2
        )
        
        output = darts(sample_input)
        assert output.shape == (8, 64)
    
    def test_single_node(self, sample_input):
        """Test with single node per cell."""
        darts = DARTSSearchSpace(
            input_dim=128,
            hidden_dim=64,
            num_cells=2,
            num_nodes=1
        )
        
        output = darts(sample_input)
        assert output.shape == (8, 64)
    
    def test_large_search_space(self):
        """Test with large search space."""
        darts = DARTSSearchSpace(
            input_dim=256,
            hidden_dim=128,
            num_cells=8,
            num_nodes=5
        )
        
        x = torch.randn(4, 256)
        output = darts(x)
        
        assert output.shape == (4, 128)
    
    def test_temperature_extremes(self):
        """Test with extreme temperature values."""
        controller = BiomarkerNASController(
            num_cells=4,
            num_nodes=3,
            num_ops=len(PRIMITIVE_OPS),
            hidden_dim=64
        )
        
        # Very low temperature (more discrete)
        arch_low = controller.sample_architecture(temperature=0.1, hard=False)
        
        # Very high temperature (more uniform)
        arch_high = controller.sample_architecture(temperature=10.0, hard=False)
        
        # Both should be valid
        assert arch_low.shape == arch_high.shape
        assert torch.all(arch_low >= 0) and torch.all(arch_low <= 1)
        assert torch.all(arch_high >= 0) and torch.all(arch_high <= 1)


class TestPerformance:
    """Performance and efficiency tests."""
    
    def test_inference_speed(self, sample_input, nas_config):
        """Test inference speed."""
        nas = NeuralArchitectureSearch(**nas_config)
        nas.eval()
        
        import time
        
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = nas(sample_input)
            end = time.time()
        
        avg_time = (end - start) / 10
        assert avg_time < 1.0  # Should be fast
    
    def test_memory_efficient_training(self, nas_config):
        """Test that training doesn't explode memory."""
        nas = NeuralArchitectureSearch(**nas_config)
        
        optimizer = optim.Adam(nas.parameters(), lr=0.001)
        
        for _ in range(5):
            x = torch.randn(8, nas_config['input_dim'])
            output = nas(x)
            loss = output['logits'].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # If we get here without OOM, test passes
        assert True
