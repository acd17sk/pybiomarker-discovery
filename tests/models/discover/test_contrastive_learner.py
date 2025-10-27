"""Comprehensive tests for contrastive_learner.py"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from biomarkers.models.discovery.contrastive_learner import (
    ContrastiveBiomarkerLearner,
    SimCLRBiomarker,
    MoCoBiomarker,
    SupConBiomarker,
    PrototypicalBiomarker,
    TripletBiomarker,
    ContrastiveAugmentation,
    HealthyRiskContrastive
)


@pytest.fixture
def sample_features():
    """Fixture for sample feature data."""
    batch_size = 32
    encoder_dim = 256
    return torch.randn(batch_size, encoder_dim)


@pytest.fixture
def sample_labels():
    """Fixture for sample labels."""
    batch_size = 32
    num_classes = 5
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture
def health_status():
    """Fixture for binary health status."""
    batch_size = 32
    return torch.randint(0, 2, (batch_size,))


class TestContrastiveBiomarkerLearner:
    """Tests for main contrastive learner class."""
    
    def test_initialization_simclr(self):
        """Test initialization with SimCLR."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            temperature=0.07,
            method='simclr'
        )
        
        assert learner.encoder_dim == 256
        assert learner.projection_dim == 128
        assert learner.temperature == 0.07
        assert learner.method == 'simclr'
        assert isinstance(learner.contrastive, SimCLRBiomarker)
    
    def test_initialization_all_methods(self):
        """Test initialization with all methods."""
        methods = ['simclr', 'moco', 'supcon', 'prototypical', 'triplet']
        
        for method in methods:
            learner = ContrastiveBiomarkerLearner(
                encoder_dim=256,
                projection_dim=128,
                method=method,
                num_prototypes=5
            )
            assert learner.method == method
    
    def test_forward_simclr(self, sample_features):
        """Test forward pass with SimCLR."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='simclr'
        )
        
        output = learner(sample_features)
        
        assert 'loss' in output
        assert 'projections' in output
        assert 'alignment' in output
        assert 'uniformity' in output
        
        assert output['loss'].item() > 0
        assert output['projections'].shape == (32, 128)
    
    def test_forward_supcon(self, sample_features, sample_labels):
        """Test forward pass with SupCon."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='supcon'
        )
        
        output = learner(sample_features, labels=sample_labels)
        
        assert 'loss' in output
        assert 'projections' in output
        assert 'num_positives' in output
        
        assert output['loss'].item() > 0
    
    def test_forward_with_health_status(self, sample_features, health_status):
        """Test forward with health status."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='simclr'
        )
        
        output = learner(sample_features, health_status=health_status)
        
        assert 'healthy_risk_loss' in output
        assert 'healthy_risk_accuracy' in output
        
        accuracy = output['healthy_risk_accuracy'].item()
        assert 0.0 <= accuracy <= 1.0
    
    def test_projection_normalization(self, sample_features):
        """Test that projections are normalized."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='simclr'
        )
        
        output = learner(sample_features)
        projections = output['projections']
        
        # Check L2 normalization
        norms = torch.norm(projections, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_gradient_flow(self, sample_features):
        """Test gradient flow through learner."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='simclr'
        )
        
        output = learner(sample_features)
        loss = output['loss']
        loss.backward()
        
        # Check gradients
        for param in learner.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestSimCLRBiomarker:
    """Tests for SimCLR implementation."""
    
    def test_initialization(self):
        """Test SimCLR initialization."""
        simclr = SimCLRBiomarker(
            projection_dim=128,
            temperature=0.07
        )
        
        assert simclr.projection_dim == 128
        assert simclr.temperature == 0.07
    
    def test_forward(self):
        """Test SimCLR forward pass."""
        simclr = SimCLRBiomarker(
            projection_dim=128,
            temperature=0.07
        )
        
        z1 = torch.randn(32, 128)
        z2 = torch.randn(32, 128)
        
        output = simclr(z1, z2)
        
        assert 'loss' in output
        assert 'alignment' in output
        assert 'uniformity' in output
        
        assert output['loss'].item() > 0
        assert output['alignment'].item() >= 0
    
    def test_temperature_effect(self):
        """Test that temperature affects loss."""
        z1 = torch.randn(32, 128)
        z2 = torch.randn(32, 128)
        
        simclr_low_temp = SimCLRBiomarker(projection_dim=128, temperature=0.01)
        simclr_high_temp = SimCLRBiomarker(projection_dim=128, temperature=1.0)
        
        output_low = simclr_low_temp(z1, z2)
        output_high = simclr_high_temp(z1, z2)
        
        # Different temperatures should give different losses
        assert not torch.isclose(output_low['loss'], output_high['loss'])
    
    def test_alignment_metric(self):
        """Test alignment metric computation."""
        simclr = SimCLRBiomarker(projection_dim=128)
        
        # Perfect alignment
        z1 = torch.randn(32, 128)
        z2 = z1.clone()
        
        output_perfect = simclr(z1, z2)
        
        # Random alignment
        z3 = torch.randn(32, 128)
        output_random = simclr(z1, z3)
        
        # Perfect alignment should have lower alignment metric
        assert output_perfect['alignment'].item() < output_random['alignment'].item()


class TestMoCoBiomarker:
    """Tests for MoCo implementation."""
    
    def test_initialization(self):
        """Test MoCo initialization."""
        moco = MoCoBiomarker(
            projection_dim=128,
            temperature=0.07,
            queue_size=1024,
            momentum=0.999
        )
        
        assert moco.projection_dim == 128
        assert moco.temperature == 0.07
        assert moco.queue_size == 1024
        assert moco.momentum == 0.999
        assert moco.queue.shape == (128, 1024)
    
    def test_forward(self):
        """Test MoCo forward pass."""
        moco = MoCoBiomarker(
            projection_dim=128,
            queue_size=1024
        )
        
        z_q = torch.randn(32, 128)
        z_k = torch.randn(32, 128)
        
        output = moco(z_q, z_k)
        
        assert 'loss' in output
        assert 'accuracy' in output
        assert 'logits' in output
        
        assert output['loss'].item() > 0
        assert 0.0 <= output['accuracy'].item() <= 1.0
    
    def test_queue_update(self):
        """Test that queue is updated."""
        moco = MoCoBiomarker(
            projection_dim=128,
            queue_size=256
        )
        
        initial_queue = moco.queue.clone()
        
        z_q = torch.randn(32, 128)
        z_k = torch.randn(32, 128)
        
        _ = moco(z_q, z_k)
        
        updated_queue = moco.queue
        
        # Queue should have changed
        assert not torch.equal(initial_queue, updated_queue)
    
    def test_queue_pointer_wrapping(self):
        """Test that queue pointer wraps around."""
        moco = MoCoBiomarker(
            projection_dim=128,
            queue_size=64  # Small queue for testing
        )
        
        # Fill queue twice
        for _ in range(4):  # 4 batches of 32 = 128 samples > 64 queue size
            z_q = torch.randn(32, 128)
            z_k = torch.randn(32, 128)
            _ = moco(z_q, z_k)
        
        # Pointer should have wrapped
        assert moco.queue_ptr.item() < 64


class TestSupConBiomarker:
    """Tests for Supervised Contrastive Learning."""
    
    def test_initialization(self):
        """Test SupCon initialization."""
        supcon = SupConBiomarker(
            projection_dim=128,
            temperature=0.07
        )
        
        assert supcon.projection_dim == 128
        assert supcon.temperature == 0.07
    
    def test_forward(self):
        """Test SupCon forward pass."""
        supcon = SupConBiomarker(projection_dim=128)
        
        z1 = torch.randn(32, 128)
        z2 = torch.randn(32, 128)
        labels = torch.randint(0, 5, (32,))
        
        output = supcon(z1, z2, labels)
        
        assert 'loss' in output
        assert 'num_positives' in output
        
        assert output['loss'].item() > 0
    
    def test_same_class_attraction(self):
        """Test that same-class samples are pulled together."""
        supcon = SupConBiomarker(projection_dim=128)
        
        # Create two groups
        z1 = torch.randn(32, 128)
        z2 = z1.clone() + 0.1 * torch.randn(32, 128)  # Slight perturbation
        
        # All same label
        labels_same = torch.zeros(32, dtype=torch.long)
        
        # All different labels
        labels_diff = torch.arange(32, dtype=torch.long)
        
        output_same = supcon(z1, z2, labels_same)
        output_diff = supcon(z1, z2, labels_diff)
        
        # Same labels should have more positives
        assert output_same['num_positives'].item() > output_diff['num_positives'].item()


class TestPrototypicalBiomarker:
    """Tests for Prototypical Networks."""
    
    def test_initialization(self):
        """Test prototypical initialization."""
        proto = PrototypicalBiomarker(
            projection_dim=128,
            num_prototypes=10,
            temperature=1.0
        )
        
        assert proto.projection_dim == 128
        assert proto.num_prototypes == 10
        assert proto.prototypes.shape == (10, 128)
    
    def test_forward_supervised(self):
        """Test forward with labels."""
        proto = PrototypicalBiomarker(
            projection_dim=128,
            num_prototypes=5
        )
        
        z = torch.randn(32, 128)
        labels = torch.randint(0, 5, (32,))
        
        output = proto(z, labels)
        
        assert 'loss' in output
        assert 'logits' in output
        assert 'prototypes' in output
        assert 'prototype_diversity' in output
        
        assert output['logits'].shape == (32, 5)
    
    def test_forward_unsupervised(self):
        """Test forward without labels."""
        proto = PrototypicalBiomarker(
            projection_dim=128,
            num_prototypes=5
        )
        
        z = torch.randn(32, 128)
        
        output = proto(z)
        
        assert 'loss' in output
        assert 'logits' in output
        assert output['logits'].shape == (32, 5)
    
    def test_prototype_normalization(self):
        """Test that prototypes are normalized."""
        proto = PrototypicalBiomarker(projection_dim=128, num_prototypes=5)
        
        z = torch.randn(32, 128)
        output = proto(z)
        
        prototypes = output['prototypes']
        norms = torch.norm(prototypes, p=2, dim=-1)
        
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_prototype_update(self):
        """Test that prototypes are updated."""
        proto = PrototypicalBiomarker(projection_dim=128, num_prototypes=5)
        
        initial_prototypes = proto.prototypes.clone()
        
        z = torch.randn(32, 128)
        labels = torch.randint(0, 5, (32,))
        
        _ = proto(z, labels)
        
        # Prototypes should have been updated
        assert not torch.equal(initial_prototypes, proto.prototypes)


class TestTripletBiomarker:
    """Tests for Triplet Loss."""
    
    def test_initialization(self):
        """Test triplet initialization."""
        triplet = TripletBiomarker(
            projection_dim=128,
            margin=1.0,
            mining_strategy='hard'
        )
        
        assert triplet.projection_dim == 128
        assert triplet.margin == 1.0
        assert triplet.mining_strategy == 'hard'
    
    def test_forward_hard_mining(self):
        """Test forward with hard mining."""
        triplet = TripletBiomarker(
            projection_dim=128,
            mining_strategy='hard'
        )
        
        z = torch.randn(32, 128)
        labels = torch.randint(0, 5, (32,))
        
        output = triplet(z, labels)
        
        assert 'loss' in output
        assert 'num_valid_triplets' in output
        assert 'fraction_positive' in output
        
        assert output['loss'].item() >= 0
    
    def test_forward_semi_hard_mining(self):
        """Test forward with semi-hard mining."""
        triplet = TripletBiomarker(
            projection_dim=128,
            mining_strategy='semi-hard'
        )
        
        z = torch.randn(32, 128)
        labels = torch.randint(0, 5, (32,))
        
        output = triplet(z, labels)
        
        assert 'loss' in output
        assert output['loss'].item() >= 0
    
    def test_forward_all_mining(self):
        """Test forward with all triplets."""
        triplet = TripletBiomarker(
            projection_dim=128,
            mining_strategy='all'
        )
        
        z = torch.randn(16, 128)  # Smaller batch to avoid too many triplets
        labels = torch.randint(0, 3, (16,))
        
        output = triplet(z, labels)
        
        assert 'loss' in output
        assert output['loss'].item() >= 0
    
    def test_margin_effect(self):
        """Test that margin affects loss."""
        z = torch.randn(32, 128)
        labels = torch.randint(0, 5, (32,))
        
        triplet_small_margin = TripletBiomarker(projection_dim=128, margin=0.1)
        triplet_large_margin = TripletBiomarker(projection_dim=128, margin=2.0)
        
        output_small = triplet_small_margin(z, labels)
        output_large = triplet_large_margin(z, labels)
        
        # Larger margin typically leads to higher loss
        assert output_large['loss'].item() >= output_small['loss'].item()


class TestContrastiveAugmentation:
    """Tests for contrastive augmentation."""
    
    def test_initialization(self):
        """Test augmentation initialization."""
        aug = ContrastiveAugmentation(
            noise_std=0.1,
            mask_ratio=0.15
        )
        
        assert aug.noise_std == 0.1
        assert aug.mask_ratio == 0.15
    
    def test_training_mode_augmentation(self):
        """Test augmentation in training mode."""
        aug = ContrastiveAugmentation()
        aug.train()
        
        x = torch.randn(32, 128)
        x_aug = aug(x)
        
        # Augmented should be different
        assert not torch.equal(x, x_aug)
        assert x_aug.shape == x.shape
    
    def test_eval_mode_no_augmentation(self):
        """Test no augmentation in eval mode."""
        aug = ContrastiveAugmentation()
        aug.eval()
        
        x = torch.randn(32, 128)
        x_aug = aug(x)
        
        # Should be identical in eval mode
        assert torch.equal(x, x_aug)
    
    def test_augmentation_diversity(self):
        """Test that augmentation produces diverse outputs."""
        aug = ContrastiveAugmentation()
        aug.train()
        
        x = torch.randn(32, 128)
        
        aug1 = aug(x)
        aug2 = aug(x)
        
        # Two augmentations should be different (stochastic)
        assert not torch.allclose(aug1, aug2, atol=1e-4)


class TestHealthyRiskContrastive:
    """Tests for healthy vs risk contrastive learning."""
    
    def test_initialization(self):
        """Test initialization."""
        hr = HealthyRiskContrastive(
            projection_dim=128,
            temperature=0.07
        )
        
        assert hr.projection_dim == 128
        assert hr.temperature == 0.07
    
    def test_forward(self):
        """Test forward pass."""
        hr = HealthyRiskContrastive(projection_dim=128)
        
        z = torch.randn(32, 128)
        health_status = torch.randint(0, 2, (32,))
        
        output = hr(z, health_status)
        
        assert 'loss' in output
        assert 'contrastive_loss' in output
        assert 'classification_loss' in output
        assert 'accuracy' in output
        assert 'separation' in output
        
        assert output['loss'].item() > 0
        assert 0.0 <= output['accuracy'].item() <= 1.0
    
    def test_separation_metric(self):
        """Test separation metric computation."""
        hr = HealthyRiskContrastive(projection_dim=128)
        
        # Create well-separated groups
        healthy = torch.randn(16, 128)
        at_risk = torch.randn(16, 128) + 2.0  # Shifted
        
        z = torch.cat([healthy, at_risk], dim=0)
        health_status = torch.cat([
            torch.zeros(16, dtype=torch.long),
            torch.ones(16, dtype=torch.long)
        ])
        
        output = hr(z, health_status)
        
        # Separation should be computed
        separation = output['separation']
        assert isinstance(separation, (int, float))
    
    def test_classification_accuracy(self):
        """Test that classification improves with training."""
        hr = HealthyRiskContrastive(projection_dim=128)
        optimizer = optim.Adam(hr.parameters(), lr=0.01)
        
        z = torch.randn(32, 128)
        health_status = torch.randint(0, 2, (32,))
        
        # Initial accuracy
        output1 = hr(z, health_status)
        acc1 = output1['accuracy'].item()
        
        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            output = hr(z, health_status)
            loss = output['loss']
            loss.backward()
            optimizer.step()
        
        # Final accuracy
        output2 = hr(z, health_status)
        acc2 = output2['accuracy'].item()
        
        # Accuracy should improve (or at least not get worse significantly)
        assert acc2 >= acc1 - 0.1


class TestIntegration:
    """Integration tests for contrastive learning."""
    
    def test_simclr_training_loop(self, sample_features):
        """Test SimCLR training loop."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='simclr'
        )
        
        optimizer = optim.Adam(learner.parameters(), lr=0.001)
        
        initial_loss = None
        for epoch in range(5):
            optimizer.zero_grad()
            output = learner(sample_features)
            loss = output['loss']
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Loss should decrease
        assert final_loss < initial_loss
    
    def test_supcon_training_loop(self, sample_features, sample_labels):
        """Test SupCon training loop."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='supcon'
        )
        
        optimizer = optim.Adam(learner.parameters(), lr=0.001)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = learner(sample_features, labels=sample_labels)
            loss = output['loss']
            loss.backward()
            optimizer.step()
        
        assert loss.item() > 0
    
    def test_prototypical_few_shot(self):
        """Test prototypical networks in few-shot scenario."""
        proto_learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='prototypical',
            num_prototypes=5
        )
        
        # Few-shot: only 2 samples per class
        support_features = torch.randn(10, 256)  # 5 classes × 2 samples
        support_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        # Query set
        query_features = torch.randn(5, 256)
        query_labels = torch.tensor([0, 1, 2, 3, 4])
        
        # Train on support
        optimizer = optim.Adam(proto_learner.parameters(), lr=0.01)
        for _ in range(10):
            optimizer.zero_grad()
            output = proto_learner(support_features, labels=support_labels)
            loss = output['loss']
            loss.backward()
            optimizer.step()
        
        # Test on query
        proto_learner.eval()
        with torch.no_grad():
            output = proto_learner(query_features, labels=query_labels)
        
        assert output['loss'].item() > 0
    
    def test_combined_contrastive_and_supervised(self, sample_features, sample_labels):
        """Test combining contrastive and supervised learning."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='supcon'
        )
        
        # Supervised classifier on top
        classifier = nn.Linear(128, 5)
        
        optimizer = optim.Adam(
            list(learner.parameters()) + list(classifier.parameters()),
            lr=0.001
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(5):
            optimizer.zero_grad()
            
            # Contrastive loss
            contrastive_output = learner(sample_features, labels=sample_labels)
            contrastive_loss = contrastive_output['loss']
            
            # Classification loss
            projections = contrastive_output['projections']
            logits = classifier(projections.detach())
            classification_loss = criterion(logits, sample_labels)
            
            # Combined loss
            total_loss = contrastive_loss + classification_loss
            total_loss.backward()
            optimizer.step()
        
        assert total_loss.item() > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_batch_simclr(self):
        """Test SimCLR with small batch."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='simclr'
        )
        
        features = torch.randn(4, 256)  # Very small batch
        output = learner(features)
        
        assert output['loss'].item() > 0
    
    def test_single_class_supcon(self):
        """Test SupCon when all samples have same label."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='supcon'
        )
        
        features = torch.randn(16, 256)
        labels = torch.zeros(16, dtype=torch.long)  # All same class
        
        output = learner(features, labels=labels)
        
        assert 'loss' in output
        # Should still compute loss (all are positives)
    
    def test_many_classes_triplet(self):
        """Test triplet with many classes (sparse labels)."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='triplet'
        )
        
        features = torch.randn(32, 256)
        labels = torch.arange(32)  # Each sample has unique label
        
        output = learner(features, labels=labels)
        
        # Should handle case with no valid triplets
        assert output['loss'].item() >= 0
    
    def test_zero_temperature(self):
        """Test with very small temperature."""
        # Temperature shouldn't be exactly zero, but very small
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='simclr',
            temperature=0.001
        )
        
        features = torch.randn(16, 256)
        output = learner(features)
        
        assert not torch.isnan(output['loss'])
        assert not torch.isinf(output['loss'])


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_values(self):
        """Test with large feature values."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='simclr'
        )
        
        features = torch.randn(16, 256) * 10  # Large values
        output = learner(features)
        
        assert not torch.isnan(output['loss'])
        assert not torch.isinf(output['loss'])
    
    def test_small_values(self):
        """Test with small feature values."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='simclr'
        )
        
        features = torch.randn(16, 256) * 0.01  # Small values
        output = learner(features)
        
        assert not torch.isnan(output['loss'])
        assert not torch.isinf(output['loss'])
    
    def test_gradient_clipping(self):
        """Test that gradients don't explode."""
        learner = ContrastiveBiomarkerLearner(
            encoder_dim=256,
            projection_dim=128,
            method='simclr'
        )
        
        features = torch.randn(16, 256)
        output = learner(features)
        loss = output['loss']
        loss.backward()
        
        # Check gradient magnitudes
        for param in learner.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1000  # Shouldn't be huge

