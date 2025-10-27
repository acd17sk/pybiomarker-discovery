"""
Shared fixtures and configuration for discovery module tests
"""

import pytest
import torch
import numpy as np



@pytest.fixture(scope="session")
def device():
    """Get device for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def small_batch():
    """Small batch for quick tests."""
    return 8


@pytest.fixture
def medium_batch():
    """Medium batch for standard tests."""
    return 32


@pytest.fixture
def large_batch():
    """Large batch for stress tests."""
    return 128


@pytest.fixture
def small_dim():
    """Small dimension for quick tests."""
    return 64


@pytest.fixture
def medium_dim():
    """Medium dimension for standard tests."""
    return 256


@pytest.fixture
def large_dim():
    """Large dimension for stress tests."""
    return 512


@pytest.fixture
def num_classes_small():
    """Small number of classes."""
    return 3


@pytest.fixture
def num_classes_medium():
    """Medium number of classes."""
    return 10


@pytest.fixture
def num_classes_large():
    """Large number of classes."""
    return 100


@pytest.fixture
def sample_biomarker_features(medium_batch, medium_dim):
    """Generate sample biomarker features."""
    return torch.randn(medium_batch, medium_dim)


@pytest.fixture
def sample_labels(medium_batch, num_classes_medium):
    """Generate sample class labels."""
    return torch.randint(0, num_classes_medium, (medium_batch,))


@pytest.fixture
def sample_health_status(medium_batch):
    """Generate sample binary health status."""
    return torch.randint(0, 2, (medium_batch,))


@pytest.fixture
def sample_modality_masks(medium_batch, medium_dim):
    """Generate sample modality masks for 4 modalities."""
    num_modalities = 4
    masks = torch.zeros(medium_batch, num_modalities, medium_dim)
    
    features_per_modality = medium_dim // num_modalities
    for i in range(num_modalities):
        start_idx = i * features_per_modality
        end_idx = (i + 1) * features_per_modality
        masks[:, i, start_idx:end_idx] = 1.0
    
    return masks


@pytest.fixture
def multimodal_biomarker_data(medium_batch, medium_dim):
    """Generate multimodal biomarker data."""
    num_modalities = 4
    
    data = {
        'features': torch.randn(medium_batch, medium_dim),
        'modality_masks': torch.zeros(medium_batch, num_modalities, medium_dim),
        'labels': torch.randint(0, 5, (medium_batch,)),
        'health_status': torch.randint(0, 2, (medium_batch,))
    }
    
    # Create modality masks
    features_per_modality = medium_dim // num_modalities
    for i in range(num_modalities):
        start_idx = i * features_per_modality
        end_idx = (i + 1) * features_per_modality
        data['modality_masks'][:, i, start_idx:end_idx] = 1.0
    
    return data


@pytest.fixture
def discovery_config(medium_dim, num_classes_medium):
    """Standard configuration for feature discovery."""
    return {
        'input_dim': medium_dim,
        'num_modalities': 4,
        'hidden_dim': 128,
        'num_diseases': num_classes_medium,
        'dropout': 0.3,
        'use_attention': True,
        'use_interaction': True,
        'use_cross_modal': True
    }


@pytest.fixture
def nas_config(medium_dim, small_dim):
    """Standard configuration for NAS."""
    return {
        'input_dim': medium_dim,
        'output_dim': small_dim,
        'num_cells': 4,
        'num_nodes': 3,
        'hidden_dim': 64
    }


@pytest.fixture
def contrastive_config():
    """Standard configuration for contrastive learning."""
    return {
        'encoder_dim': 256,
        'projection_dim': 128,
        'temperature': 0.07,
        'num_prototypes': 10
    }


@pytest.fixture
def clinical_biomarker_data():
    """Generate realistic clinical biomarker data."""
    # Simulate Parkinson's disease biomarkers
    num_patients = 50
    
    # Healthy group (n=25)
    healthy_features = {
        'gait_speed': np.random.normal(1.2, 0.1, 25),  # m/s
        'stride_length': np.random.normal(1.3, 0.1, 25),  # m
        'tremor_amplitude': np.random.normal(0.1, 0.05, 25),  # arbitrary units
        'voice_jitter': np.random.normal(0.005, 0.001, 25),  # %
        'finger_tapping_rate': np.random.normal(5.0, 0.3, 25),  # taps/sec
    }
    
    # Parkinson's group (n=25)
    pd_features = {
        'gait_speed': np.random.normal(0.8, 0.15, 25),  # Slower
        'stride_length': np.random.normal(0.9, 0.15, 25),  # Shorter
        'tremor_amplitude': np.random.normal(0.5, 0.2, 25),  # Higher
        'voice_jitter': np.random.normal(0.015, 0.005, 25),  # Higher
        'finger_tapping_rate': np.random.normal(3.0, 0.5, 25),  # Slower
    }
    
    # Combine
    features = np.vstack([
        np.column_stack([healthy_features[k] for k in healthy_features.keys()]),
        np.column_stack([pd_features[k] for k in pd_features.keys()])
    ])
    
    labels = np.concatenate([np.zeros(25), np.ones(25)])
    
    return {
        'features': torch.from_numpy(features).float(),
        'labels': torch.from_numpy(labels).long(),
        'feature_names': list(healthy_features.keys())
    }


@pytest.fixture
def temporal_biomarker_sequence():
    """Generate temporal sequence of biomarker measurements."""
    batch_size = 16
    sequence_length = 10
    feature_dim = 64
    
    # Simulate temporal progression of disease
    sequences = []
    for i in range(batch_size):
        # Create a sequence that changes over time
        trend = np.linspace(0, 1, sequence_length)
        noise = np.random.randn(sequence_length, feature_dim) * 0.1
        
        sequence = trend[:, np.newaxis] + noise
        sequences.append(sequence)
    
    sequences = np.array(sequences)
    
    return {
        'sequences': torch.from_numpy(sequences).float(),
        'labels': torch.randint(0, 3, (batch_size,)),
        'time_points': torch.arange(sequence_length).float()
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "cuda" in item.nodeid.lower() or "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if any(keyword in item.nodeid.lower() for keyword in ["large", "stress", "performance"]):
            item.add_marker(pytest.mark.slow)