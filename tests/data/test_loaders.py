import pytest
import torch
from biomarkers.data.loaders import MultiModalBiomarkerDataset, create_dataloaders, BiomarkerDataLoader

def test_multimodal_dataset_init(dummy_data_dir):
    """Test initialization and sample preparation."""
    dataset = MultiModalBiomarkerDataset(
        data_dir=str(dummy_data_dir / "train"),
        modalities=["voice", "movement"],
        quality_threshold=0.7 # p004 should be excluded
    )
    assert len(dataset) == 3 # Check that low quality sample is filtered
    assert "p004" not in [s['id'] for s in dataset.samples]

def test_multimodal_dataset_getitem(dummy_data_dir):
    """Test the __getitem__ method for correct data shapes and structure."""
    dataset = MultiModalBiomarkerDataset(
        data_dir=str(dummy_data_dir / "train"),
        modalities=["voice", "movement", "vision", "text"],
        target_diseases=["A", "B", "C"],
        quality_threshold=0.0 # Include all samples
    )
    sample = dataset[0] # Get first sample (p001)

    assert "input" in sample
    assert "target" in sample
    assert "metadata" in sample
    
    # Check modalities in input
    assert "voice" in sample["input"]
    assert "movement" in sample["input"]
    assert "vision" in sample["input"]
    assert "text" in sample["input"]

    # Check tensor types and shapes
    assert isinstance(sample["input"]["voice"], torch.Tensor)
    assert isinstance(sample["input"]["movement"], torch.Tensor)
    assert sample["input"]["voice"].shape[0] == 1 # 1 channel (for spectrogram)
    assert sample["input"]["movement"].shape[0] == 6 # 6 sensor channels
    assert sample["input"]["vision"].shape == (10, 3, 224, 224) # 10 frames, 3 channels, H, W
    assert sample["input"]["text"].shape == (512,)

    # Check target and metadata
    assert sample["target"].item() == 0 # p001 disease is 'A', which is index 0
    assert sample["metadata"]["patient_id"] == "p001"

def test_balanced_sampler(dummy_data_dir):
    """Test the weighted random sampler generation."""
    dataset = MultiModalBiomarkerDataset(
        data_dir=str(dummy_data_dir / "train"),
        modalities=["voice"],
        quality_threshold=0.7
    )
    # class counts: A=2, B=1. Weights should be higher for class B.
    sampler = dataset.get_sampler(balanced=True)
    assert sampler is not None
    # Weight for patient p002 (disease B) should be highest
    assert torch.argmax(sampler.weights).item() == 1 # p002 is the 2nd valid sample

def test_create_dataloaders(dummy_data_dir):
    """Test the dataloader factory function."""
    config = {
        'data_dir': str(dummy_data_dir),
        'modalities': ['voice', 'movement'],
        'target_diseases': ['A', 'B', 'C'],
        'quality_threshold': 0.7,
        'batch_size': 2,
        'num_workers': 0, # Use 0 for testing to avoid multiprocessing issues
        'persistent_workers': False
    }
    train_loader, val_loader, test_loader = create_dataloaders(config)

    assert isinstance(train_loader, BiomarkerDataLoader)
    assert isinstance(val_loader, BiomarkerDataLoader)
    assert isinstance(test_loader, BiomarkerDataLoader)

    # Check batch size and number of samples
    assert len(train_loader.dataset) == 3
    assert len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 2
    
    # Check a batch from the train loader
    batch = next(iter(train_loader))
    assert batch['input']['voice'].shape[0] <= 2
    assert batch['target'].shape[0] <= 2