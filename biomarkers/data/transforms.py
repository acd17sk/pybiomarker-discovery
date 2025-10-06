"""Data transformation and augmentation for biomarker data"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Any, Tuple, Union
import random
import librosa
import logging

logger = logging.getLogger(__name__)


class BiomarkerTransform(nn.Module):
    """Base class for biomarker transforms"""
    
    def __init__(self, p: float = 1.0):
        super().__init__()
        self.p = p
    
    def forward(self, x: Any) -> Any:
        if random.random() < self.p:
            return self.apply_transform(x)
        return x
    
    def apply_transform(self, x: Any) -> Any:
        raise NotImplementedError


class ComposeTransform(BiomarkerTransform):
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List[BiomarkerTransform]):
        super().__init__(p=1.0)
        self.transforms = transforms
    
    def apply_transform(self, x: Any) -> Any:
        for transform in self.transforms:
            x = transform(x)
        return x


class Normalize(BiomarkerTransform):
    """Normalize data with mean and std"""
    
    def __init__(self, mean: Union[float, List[float]], 
                 std: Union[float, List[float]], p: float = 1.0):
        super().__init__(p)
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class VoiceTransform(BiomarkerTransform):
    """Transforms specific to voice data"""
    
    def __init__(self, transform_type: str = 'augment', p: float = 0.5):
        super().__init__(p)
        self.transform_type = transform_type
    
    def apply_transform(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        if self.transform_type == 'time_stretch':
            rate = np.random.uniform(0.8, 1.2)
            x_np = librosa.effects.time_stretch(x_np, rate=rate)

        elif self.transform_type == 'pitch_shift':
            n_steps = np.random.randint(-4, 4)
            x_np = librosa.effects.pitch_shift(x_np, sr=16000, n_steps=n_steps)

        elif self.transform_type == 'add_noise':
            noise = np.random.normal(0, 0.005, x_np.shape)
            x_np = x_np + noise

        elif self.transform_type == 'time_mask':
            # SpecAugment-style time masking
            if len(x_np.shape) == 2:  # Spectrogram
                max_len = min(20, x_np.shape[1])
                if max_len > 1:
                    time_mask_len = np.random.randint(1, max_len + 1)  # ensure at least 1 frame masked
                    if x_np.shape[1] - time_mask_len > 0:
                        time_start = np.random.randint(0, x_np.shape[1] - time_mask_len)
                    else:
                        time_start = 0
                    x_np[:, time_start:time_start + time_mask_len] = 0

        elif self.transform_type == 'freq_mask':
            # SpecAugment-style frequency masking
            if len(x_np.shape) == 2:  # Spectrogram
                max_len = min(8, x_np.shape[0])
                if max_len > 1:
                    freq_mask_len = np.random.randint(1, max_len)
                    freq_start = np.random.randint(0, x_np.shape[0] - freq_mask_len)
                    x_np[freq_start:freq_start + freq_mask_len, :] = 0

        if is_tensor:
            return torch.from_numpy(x_np).float()
        return x_np

class MovementTransform(BiomarkerTransform):
    """Transforms for movement/sensor data"""
    
    def __init__(self, transform_type: str = 'augment', p: float = 0.5):
        super().__init__(p)
        self.transform_type = transform_type
    
    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_type == 'jitter':
            # Add random jitter
            jitter = torch.randn_like(x) * 0.01
            return x + jitter
        
        elif self.transform_type == 'scale':
            # Random scaling
            scale = torch.FloatTensor(1).uniform_(0.8, 1.2)
            return x * scale
        
        elif self.transform_type == 'rotation':
            # Random rotation (for 3D accelerometer data)
            if x.shape[0] >= 3:  # At least 3 channels (x, y, z)
                theta = np.random.uniform(-np.pi/6, np.pi/6)
                rotation_matrix = torch.tensor([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ]).float()
                
                # Apply rotation to first 3 channels
                x[:3] = torch.matmul(rotation_matrix, x[:3])
        
        elif self.transform_type == 'temporal_warp':
            # Time warping
            if len(x.shape) == 2:
                warped = torch.nn.functional.interpolate(
                    x.unsqueeze(0),
                    scale_factor=np.random.uniform(0.9, 1.1),
                    mode='linear',
                    align_corners=False
                ).squeeze(0)
                
                # Crop or pad to original size
                if warped.shape[1] > x.shape[1]:
                    return warped[:, :x.shape[1]]
                else:
                    pad_len = x.shape[1] - warped.shape[1]
                    return torch.nn.functional.pad(warped, (0, pad_len))
        
        return x


class VisionTransform(BiomarkerTransform):
    """Transforms for vision/video data"""
    
    def __init__(self, transform_type: str = 'augment', p: float = 0.5):
        super().__init__(p)
        self.transform_type = transform_type
    
    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_type == 'color_jitter':
            # Random color jittering
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            x = x * contrast + (brightness - 1)
            return torch.clamp(x, 0, 1)
        
        elif self.transform_type == 'random_crop':
            # Random crop and resize
            if len(x.shape) == 4:  # [T, C, H, W]
                t, c, h, w = x.shape
                crop_h = int(h * 0.9)
                crop_w = int(w * 0.9)
                
                top = np.random.randint(0, h - crop_h)
                left = np.random.randint(0, w - crop_w)
                
                x_cropped = x[:, :, top:top+crop_h, left:left+crop_w]
                
                # Resize back to original size
                x_resized = torch.nn.functional.interpolate(
                    x_cropped.reshape(t*c, 1, crop_h, crop_w),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).reshape(t, c, h, w)
                
                return x_resized
        
        elif self.transform_type == 'horizontal_flip':
            # Random horizontal flip
            if len(x.shape) >= 3:
                return torch.flip(x, dims=[-1])
        
        elif self.transform_type == 'gaussian_blur':
            # Apply Gaussian blur
            if len(x.shape) >= 3:
                # Simplified blur using average pooling
                blurred = torch.nn.functional.avg_pool2d(
                    x.view(-1, 1, x.shape[-2], x.shape[-1]),
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
                return blurred.view(x.shape)
        
        return x


class TextTransform(BiomarkerTransform):
    """Transforms for text data"""
    
    def __init__(self, transform_type: str = 'augment', 
                 vocab_size: int = 10000, p: float = 0.5):
        super().__init__(p)
        self.transform_type = transform_type
        self.vocab_size = vocab_size
    
    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_type == 'token_replacement':
            # Random token replacement
            mask = torch.rand_like(x.float()) < 0.1
            random_tokens = torch.randint_like(x, 0, self.vocab_size)
            x = torch.where(mask, random_tokens, x)
        
        elif self.transform_type == 'token_deletion':
            # Random token deletion
            mask = torch.rand_like(x.float()) < 0.9
            x = x * mask.long()
        
        elif self.transform_type == 'token_shuffle':
            # Shuffle tokens within windows
            if len(x.shape) == 1:
                window_size = 5
                for i in range(0, len(x) - window_size, window_size):
                    window = x[i:i+window_size].clone()
                    shuffled_idx = torch.randperm(window_size)
                    x[i:i+window_size] = window[shuffled_idx]
        
        return x


class TemporalAugmentation(BiomarkerTransform):
    """Temporal augmentation for time-series data"""
    
    def __init__(self, augmentation_type: str = 'cutmix', p: float = 0.5):
        super().__init__(p)
        self.augmentation_type = augmentation_type
    
    def apply_transform(self, x: torch.Tensor, 
                       y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if self.augmentation_type == 'cutmix' and y is not None:
            # CutMix augmentation
            batch_size = x.shape[0]
            indices = torch.randperm(batch_size)
            
            lam = np.random.beta(1.0, 1.0)
            cut_len = int(x.shape[-1] * (1 - lam))
            cut_start = np.random.randint(0, x.shape[-1] - cut_len)
            
            x[:, :, cut_start:cut_start + cut_len] = x[indices, :, cut_start:cut_start + cut_len]
            
            # Mix labels
            y_mixed = lam * y + (1 - lam) * y[indices]
            return x, y_mixed
        
        elif self.augmentation_type == 'mixup' and y is not None:
            # MixUp augmentation
            batch_size = x.shape[0]
            indices = torch.randperm(batch_size)
            
            lam = np.random.beta(1.0, 1.0)
            x_mixed = lam * x + (1 - lam) * x[indices]
            y_mixed = lam * y + (1 - lam) * y[indices]
            
            return x_mixed, y_mixed
        
        return x, y


class SpectralAugmentation(BiomarkerTransform):
    """Spectral augmentation for frequency-domain data"""
    
    def __init__(self, augmentation_type: str = 'spec_augment', p: float = 0.5):
        super().__init__(p)
        self.augmentation_type = augmentation_type
    
    def apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.augmentation_type == 'spec_augment':
            # SpecAugment
            if len(x.shape) >= 2:
                # Time masking
                time_mask = int(x.shape[-1] * 0.1)
                t0 = np.random.randint(0, x.shape[-1] - time_mask)
                x[..., t0:t0 + time_mask] = 0
                
                # Frequency masking
                freq_mask = int(x.shape[-2] * 0.1)
                f0 = np.random.randint(0, x.shape[-2] - freq_mask)
                x[..., f0:f0 + freq_mask, :] = 0
        
        elif self.augmentation_type == 'freq_shift':
            # Frequency shifting
            if len(x.shape) >= 2:
                shift = np.random.randint(-5, 5)
                x = torch.roll(x, shift, dims=-2)
        
        return x


class Augmentation(BiomarkerTransform):
    """General augmentation wrapper"""
    
    def __init__(self, 
                 modality: str,
                 augmentations: List[str],
                 p: float = 0.5):
        super().__init__(p)
        self.modality = modality
        self.augmentations = self._create_augmentations(modality, augmentations)
    
    def _create_augmentations(self, modality: str, 
                             aug_names: List[str]) -> List[BiomarkerTransform]:
        augmentations = []
        
        for aug_name in aug_names:
            if modality == 'voice':
                augmentations.append(VoiceTransform(aug_name))
            elif modality == 'movement':
                augmentations.append(MovementTransform(aug_name))
            elif modality == 'vision':
                augmentations.append(VisionTransform(aug_name))
            elif modality == 'text':
                augmentations.append(TextTransform(aug_name))
        
        return augmentations
    
    def apply_transform(self, x: Any) -> Any:
        for aug in self.augmentations:
            x = aug(x)
        return x