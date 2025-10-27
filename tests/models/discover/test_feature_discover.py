"""Comprehensive tests for feature_discovery.py"""

import pytest
import torch


from biomarkers.models.discovery.feature_discovery import (
    AutomatedFeatureDiscovery,
    AttentionBasedDiscovery,
    FeatureInteractionNetwork,
    AdaptiveFeatureSelector,
    BiomarkerCombinationFinder,
    CrossModalFeatureDiscovery
)


@pytest.fixture
def sample_features():
    """Fixture for sample feature data."""
    batch_size = 16
    input_dim = 128
    return torch.randn(batch_size, input_dim)


@pytest.fixture
def modality_masks():
    """Fixture for modality masks."""
    batch_size = 16
    num_modalities = 4
    input_dim = 128
    
    masks = torch.zeros(batch_size, num_modalities, input_dim)
    features_per_modality = input_dim // num_modalities
    
    for i in range(num_modalities):
        start_idx = i * features_per_modality
        end_idx = (i + 1) * features_per_modality
        masks[:, i, start_idx:end_idx] = 1.0
    
    return masks


class TestAutomatedFeatureDiscovery:
    """Tests for AutomatedFeatureDiscovery class."""
    
    def test_initialization(self):
        """Test model initialization."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5,
            use_attention=True,
            use_interaction=True,
            use_cross_modal=True
        )
        
        assert discovery.input_dim == 128
        assert discovery.num_modalities == 4
        assert discovery.hidden_dim == 256
        assert discovery.num_diseases == 5
        assert discovery.attention_discovery is not None
        assert discovery.interaction_network is not None
        assert discovery.cross_modal_discovery is not None
    
    def test_forward_basic(self, sample_features):
        """Test basic forward pass."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5
        )
        
        output = discovery(sample_features)
        
        assert 'logits' in output
        assert 'probabilities' in output
        assert 'features' in output
        assert 'selected_features' in output
        assert 'selection_scores' in output
        
        assert output['logits'].shape == (16, 5)
        assert output['probabilities'].shape == (16, 5)
        assert output['features'].shape == (16, 256)
        assert torch.allclose(output['probabilities'].sum(dim=1), 
                             torch.ones(16), atol=1e-5)
    
    def test_forward_with_modality_masks(self, sample_features, modality_masks):
        """Test forward pass with modality masks."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5,
            use_cross_modal=True
        )
        
        output = discovery(
            sample_features,
            modality_masks=modality_masks,
            return_importance=True,
            return_patterns=True
        )
        
        assert 'importance_scores' in output
        assert 'top_features' in output
        assert 'combinations' in output
        assert 'combination_scores' in output
        
        assert output['importance_scores'].shape == (16, 128)
        assert output['combinations'].shape[0] == 16
    
    def test_pattern_tracking(self, sample_features):
        """Test pattern tracking functionality."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5
        )
        
        # Run multiple times to accumulate patterns
        for _ in range(3):
            output = discovery(sample_features, return_patterns=True)
        
        # Get top patterns
        top_patterns = discovery.get_top_patterns(k=5)
        
        assert 'patterns' in top_patterns
        assert 'scores' in top_patterns
        
        if top_patterns['patterns'] is not None:
            assert len(top_patterns['patterns']) <= 5
            assert len(top_patterns['scores']) == len(top_patterns['patterns'])
    
    def test_without_optional_components(self, sample_features):
        """Test with optional components disabled."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5,
            use_attention=False,
            use_interaction=False,
            use_cross_modal=False
        )
        
        output = discovery(sample_features)
        
        assert 'logits' in output
        assert 'probabilities' in output
        assert discovery.attention_discovery is None
        assert discovery.interaction_network is None
        assert discovery.cross_modal_discovery is None
    
    def test_gradient_flow(self, sample_features):
        """Test that gradients flow through the model."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5
        )
        
        output = discovery(sample_features)
        loss = output['logits'].sum()
        loss.backward()
        
        # Check that gradients exist
        for param in discovery.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestAttentionBasedDiscovery:
    """Tests for AttentionBasedDiscovery class."""
    
    def test_initialization(self):
        """Test attention discovery initialization."""
        attention = AttentionBasedDiscovery(
            input_dim=128,
            hidden_dim=256,
            num_heads=8,
            num_layers=3
        )
        
        assert attention.input_dim == 128
        assert attention.hidden_dim == 256
        assert attention.num_heads == 8
        assert len(attention.attention_layers) == 3
        assert len(attention.layer_norms) == 3
        assert len(attention.feed_forwards) == 3
    
    def test_forward(self, sample_features):
        """Test attention forward pass."""
        attention = AttentionBasedDiscovery(
            input_dim=128,
            hidden_dim=256,
            num_heads=8
        )
        
        output = attention(sample_features)
        
        assert 'attended_features' in output
        assert 'attention_weights' in output
        assert 'interaction_weights' in output
        
        assert output['attended_features'].shape == (16, 256)
        assert output['attention_weights'].shape[0] == 16
    
    def test_attention_weights_sum(self, sample_features):
        """Test that attention weights sum to 1."""
        attention = AttentionBasedDiscovery(
            input_dim=128,
            hidden_dim=256,
            num_heads=8
        )
        
        output = attention(sample_features)
        attn_weights = output['attention_weights']
        
        # Attention weights should sum to approximately 1 along last dimension
        weight_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4)
    
    def test_different_num_heads(self, sample_features):
        """Test with different numbers of attention heads."""
        for num_heads in [4, 8, 16]:
            attention = AttentionBasedDiscovery(
                input_dim=128,
                hidden_dim=256,
                num_heads=num_heads
            )
            
            output = attention(sample_features)
            assert output['attended_features'].shape == (16, 256)


class TestFeatureInteractionNetwork:
    """Tests for FeatureInteractionNetwork class."""
    
    def test_initialization(self):
        """Test interaction network initialization."""
        interaction = FeatureInteractionNetwork(
            input_dim=128,
            hidden_dim=256,
            num_interactions=3
        )
        
        assert interaction.input_dim == 128
        assert interaction.hidden_dim == 256
        assert interaction.num_interactions == 3
    
    def test_forward(self, sample_features):
        """Test interaction network forward pass."""
        interaction = FeatureInteractionNetwork(
            input_dim=128,
            hidden_dim=256
        )
        
        output = interaction(sample_features)
        
        assert 'interaction_features' in output
        assert 'interaction_weights' in output
        assert 'pairwise_features' in output
        assert 'triple_features' in output
        assert 'fm_features' in output
        
        assert output['interaction_features'].shape == (16, 256)
        assert output['interaction_weights'].shape == (16, 128, 128)
    
    def test_pairwise_interactions(self, sample_features):
        """Test pairwise interaction computation."""
        interaction = FeatureInteractionNetwork(
            input_dim=128,
            hidden_dim=256
        )
        
        output = interaction(sample_features)
        
        # Pairwise features should be computed
        assert output['pairwise_features'].shape == (16, 128)
    
    def test_factorization_machines(self, sample_features):
        """Test factorization machines component."""
        interaction = FeatureInteractionNetwork(
            input_dim=128,
            hidden_dim=256
        )
        
        output = interaction(sample_features)
        
        # FM features should be present
        assert output['fm_features'].shape == (16,)
    
    def test_gradient_flow(self, sample_features):
        """Test gradient flow through interactions."""
        interaction = FeatureInteractionNetwork(
            input_dim=128,
            hidden_dim=256
        )
        
        output = interaction(sample_features)
        loss = output['interaction_features'].sum()
        loss.backward()
        
        # Check gradients
        for param in interaction.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestAdaptiveFeatureSelector:
    """Tests for AdaptiveFeatureSelector class."""
    
    def test_initialization(self):
        """Test selector initialization."""
        selector = AdaptiveFeatureSelector(
            input_dim=128,
            hidden_dim=256,
            selection_ratio=0.5
        )
        
        assert selector.input_dim == 128
        assert selector.hidden_dim == 256
        assert selector.selection_ratio == 0.5
        assert selector.num_selected == 64
    
    def test_forward_training(self, sample_features):
        """Test forward pass in training mode."""
        selector = AdaptiveFeatureSelector(
            input_dim=128,
            hidden_dim=256,
            selection_ratio=0.5
        )
        selector.train()
        
        transformed, selection_probs = selector(sample_features, hard_selection=False)
        
        assert transformed.shape == (16, 256)
        assert selection_probs.shape == (16, 128)
        assert torch.all(selection_probs >= 0) and torch.all(selection_probs <= 1)
    
    def test_forward_evaluation(self, sample_features):
        """Test forward pass in evaluation mode."""
        selector = AdaptiveFeatureSelector(
            input_dim=128,
            hidden_dim=256,
            selection_ratio=0.5
        )
        selector.eval()
        
        transformed, selection_probs = selector(sample_features)
        
        assert transformed.shape == (16, 256)
        assert selection_probs.shape == (16, 128)
        
        # In eval mode with hard selection, should have exactly num_selected non-zero
        non_zero_per_sample = (selection_probs > 0).sum(dim=1)
        assert torch.all(non_zero_per_sample == 64)
    
    def test_temperature_annealing(self):
        """Test temperature annealing."""
        selector = AdaptiveFeatureSelector(
            input_dim=128,
            hidden_dim=256
        )
        
        initial_temp = selector.temperature.item()
        selector.anneal_temperature(step=500, total_steps=1000)
        final_temp = selector.temperature.item()
        
        assert final_temp < initial_temp
        assert final_temp >= 0.5  # min_temp
    
    def test_selection_sparsity(self, sample_features):
        """Test that selection enforces sparsity."""
        selector = AdaptiveFeatureSelector(
            input_dim=128,
            hidden_dim=256,
            selection_ratio=0.3
        )
        selector.eval()
        
        transformed, selection_probs = selector(sample_features)
        
        # Should select approximately 30% of features
        selected_per_sample = (selection_probs > 0).sum(dim=1).float().mean()
        expected = 128 * 0.3
        assert abs(selected_per_sample - expected) < 5


class TestBiomarkerCombinationFinder:
    """Tests for BiomarkerCombinationFinder class."""
    
    def test_initialization(self):
        """Test combination finder initialization."""
        finder = BiomarkerCombinationFinder(
            input_dim=128,
            hidden_dim=256,
            max_combinations=100,
            combination_size=5
        )
        
        assert finder.input_dim == 128
        assert finder.hidden_dim == 256
        assert finder.max_combinations == 100
        assert finder.combination_size == 5
    
    def test_forward(self, sample_features):
        """Test combination finder forward pass."""
        finder = BiomarkerCombinationFinder(
            input_dim=128,
            hidden_dim=256,
            max_combinations=50
        )
        
        output = finder(sample_features)
        
        assert 'combinations' in output
        assert 'scores' in output
        assert 'synergies' in output
        assert 'combination_logits' in output
        
        assert output['combinations'].shape == (16, 50, 128)
        assert output['scores'].shape == (16, 50)
        assert output['synergies'].shape == (16, 50)
    
    def test_combination_scoring(self, sample_features):
        """Test that combinations are scored."""
        finder = BiomarkerCombinationFinder(
            input_dim=128,
            hidden_dim=256,
            max_combinations=50
        )
        
        output = finder(sample_features)
        scores = output['scores']
        
        # Scores should be between 0 and 1 (sigmoid output)
        assert torch.all(scores >= 0) and torch.all(scores <= 1)
    
    def test_synergy_detection(self, sample_features):
        """Test synergy detection."""
        finder = BiomarkerCombinationFinder(
            input_dim=128,
            hidden_dim=256,
            max_combinations=50
        )
        
        output = finder(sample_features)
        synergies = output['synergies']
        
        # Synergies should be between -1 and 1 (tanh output)
        assert torch.all(synergies >= -1) and torch.all(synergies <= 1)
    
    def test_different_combination_sizes(self, sample_features):
        """Test with different combination sizes."""
        for combo_size in [3, 5, 10]:
            finder = BiomarkerCombinationFinder(
                input_dim=128,
                hidden_dim=256,
                max_combinations=20,
                combination_size=combo_size
            )
            
            output = finder(sample_features)
            assert output['scores'].shape == (16, 20)


class TestCrossModalFeatureDiscovery:
    """Tests for CrossModalFeatureDiscovery class."""
    
    def test_initialization(self):
        """Test cross-modal discovery initialization."""
        cross_modal = CrossModalFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256
        )
        
        assert cross_modal.input_dim == 128
        assert cross_modal.num_modalities == 4
        assert cross_modal.hidden_dim == 256
        assert len(cross_modal.modality_encoders) == 4
        assert len(cross_modal.cross_modal_attention) == 4
    
    def test_forward(self, sample_features, modality_masks):
        """Test cross-modal forward pass."""
        cross_modal = CrossModalFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256
        )
        
        output = cross_modal(sample_features, modality_masks)
        
        assert 'fused_features' in output
        assert 'modal_features' in output
        assert 'attended_features' in output
        assert 'attention_weights' in output
        assert 'modality_importance' in output
        
        assert output['fused_features'].shape == (16, 256)
        assert len(output['modal_features']) == 4
        assert len(output['attended_features']) == 4
    
    def test_modality_importance(self, sample_features, modality_masks):
        """Test modality importance computation."""
        cross_modal = CrossModalFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256
        )
        
        output = cross_modal(sample_features, modality_masks)
        importance = output['modality_importance']
        
        assert importance.shape == (16, 4)
        # Importance should sum to 1 (softmax output)
        assert torch.allclose(importance.sum(dim=1), torch.ones(16), atol=1e-5)
    
    def test_modal_features_shape(self, sample_features, modality_masks):
        """Test that modal features have correct shapes."""
        cross_modal = CrossModalFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256
        )
        
        output = cross_modal(sample_features, modality_masks)
        
        for modal_feat in output['modal_features']:
            assert modal_feat.shape == (16, 256)
    
    def test_cross_attention_weights(self, sample_features, modality_masks):
        """Test cross-attention weight computation."""
        cross_modal = CrossModalFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256
        )
        
        output = cross_modal(sample_features, modality_masks)
        attention_weights = output['attention_weights']
        
        assert len(attention_weights) == 4
        
        # Each attention weight should sum to 1
        for attn in attention_weights:
            weight_sums = attn.sum(dim=-1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_pipeline(self, sample_features, modality_masks):
        """Test complete pipeline from features to predictions."""
        # Create discovery model
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5,
            use_attention=True,
            use_interaction=True,
            use_cross_modal=True
        )
        
        # Forward pass
        output = discovery(
            sample_features,
            modality_masks=modality_masks,
            return_importance=True,
            return_patterns=True
        )
        
        # Check all expected outputs
        assert 'logits' in output
        assert 'probabilities' in output
        assert 'features' in output
        assert 'importance_scores' in output
        assert 'combinations' in output
        
        # Check predictions
        predictions = output['probabilities'].argmax(dim=1)
        assert predictions.shape == (16,)
        assert torch.all(predictions >= 0) and torch.all(predictions < 5)
    
    def test_training_loop(self, sample_features):
        """Test training loop functionality."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5
        )
        
        optimizer = torch.optim.Adam(discovery.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create fake labels
        labels = torch.randint(0, 5, (16,))
        
        # Training step
        discovery.train()
        optimizer.zero_grad()
        output = discovery(sample_features)
        loss = criterion(output['logits'], labels)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        
        # Check that parameters were updated
        for param in discovery.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_evaluation_mode(self, sample_features):
        """Test evaluation mode behavior."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5
        )
        
        discovery.eval()
        
        with torch.no_grad():
            output1 = discovery(sample_features)
            output2 = discovery(sample_features)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1['logits'], output2['logits'], atol=1e-5)
    
    def test_device_transfer(self):
        """Test transferring model to different devices."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5
        )
        
        # CPU test
        features_cpu = torch.randn(8, 128)
        output_cpu = discovery(features_cpu)
        assert output_cpu['logits'].device == features_cpu.device
        
        # GPU test (if available)
        if torch.cuda.is_available():
            discovery = discovery.cuda()
            features_gpu = torch.randn(8, 128).cuda()
            output_gpu = discovery(features_gpu)
            assert output_gpu['logits'].device == features_gpu.device


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_batch_size(self):
        """Test with batch size of 1."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5
        )
        
        features = torch.randn(1, 128)
        output = discovery(features)
        
        assert output['logits'].shape == (1, 5)
    
    def test_large_input_dim(self):
        """Test with large input dimension."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=1024,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=10
        )
        
        features = torch.randn(8, 1024)
        output = discovery(features)
        
        assert output['logits'].shape == (8, 10)
    
    def test_single_modality(self):
        """Test with single modality."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=1,
            hidden_dim=256,
            num_diseases=5,
            use_cross_modal=False
        )
        
        features = torch.randn(8, 128)
        output = discovery(features)
        
        assert output['logits'].shape == (8, 5)
    
    def test_zero_dropout(self):
        """Test with zero dropout."""
        discovery = AutomatedFeatureDiscovery(
            input_dim=128,
            num_modalities=4,
            hidden_dim=256,
            num_diseases=5,
            dropout=0.0
        )
        
        features = torch.randn(8, 128)
        output = discovery(features)
        
        assert output['logits'].shape == (8, 5)

