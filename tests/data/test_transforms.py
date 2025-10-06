import pytest
import torch
from biomarkers.data.transforms import (
    ComposeTransform,
    Normalize,
    VoiceTransform,
    MovementTransform,
    VisionTransform
)

def test_compose_transform():
    """Test composing multiple transforms."""
    t1 = Normalize(mean=0.5, std=0.5)
    t2 = MovementTransform(transform_type='jitter', p=1.0)
    composed = ComposeTransform([t1, t2])
    
    tensor = torch.ones(6, 100)
    transformed_tensor = composed(tensor)
    
    # After norm, should be 1.0. After jitter, should be slightly different.
    assert not torch.allclose(transformed_tensor, torch.ones(6, 100))
    assert transformed_tensor.mean() > 0.9 and transformed_tensor.mean() < 1.1

def test_voice_transform():
    """Test a voice-specific transform (SpecAugment)."""
    spectrogram = torch.ones(80, 200) # [freq_bins, time_steps]
    transform = VoiceTransform(transform_type='time_mask', p=1.0)

    transformed_spec = transform(spectrogram)

    # Ensure masking has an effect
    assert torch.sum(transformed_spec) <= torch.sum(spectrogram)
    # Check if at least one column is masked
    assert (transformed_spec.sum(dim=0) == 0).any()


def test_movement_transform():
    """Test a movement-specific transform."""
    sensor_data = torch.randn(6, 1000)
    transform = MovementTransform(transform_type='scale', p=1.0)
    
    transformed_data = transform(sensor_data)
    # The norm should be scaled up or down
    assert not torch.allclose(torch.norm(sensor_data), torch.norm(transformed_data))

def test_vision_transform():
    """Test a vision-specific transform."""
    video_frames = torch.rand(10, 3, 224, 224) # T, C, H, W
    transform = VisionTransform(transform_type='horizontal_flip', p=1.0)
    
    transformed_frames = transform(video_frames)
    
    # Compare one pixel from the first frame before and after flip
    original_pixel = video_frames[0, 0, 0, 0]
    flipped_pixel = transformed_frames[0, 0, 0, -1]
    assert torch.allclose(original_pixel, flipped_pixel)