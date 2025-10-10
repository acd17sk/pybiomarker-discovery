"""
Acoustic encoding modules for voice biomarkers - IMPROVED VERSION

This module provides state-of-the-art acoustic encoders for speech processing:
- Spectral encoders (mel-spectrograms, MFCCs)
- Waveform encoders (SincNet-based)
- Conformer encoders (convolution + self-attention)

All improvements applied:
- Proper buffer registration for device consistency
- Complete type hints
- Named constants for magic numbers
- Consistent return types
- Optimized computations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math
import numpy as np


class AcousticEncoder(nn.Module):
    """Base acoustic encoder for voice signals."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass - must be implemented by subclasses.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (features, intermediates_dict)
        """
        raise NotImplementedError


class SpectralEncoder(AcousticEncoder):
    """
    Encoder for spectral representations (mel-spectrograms, etc.).
    
    Uses CNN for local feature extraction, LSTM for temporal modeling,
    and optional multi-head attention for sequence modeling.
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 input_dim: int = 80,  # mel bins
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 dropout: float = 0.3,
                 use_attention: bool = True):
        super().__init__(input_dim, hidden_dim, output_dim, dropout)
        
        self.input_channels = input_channels
        self.use_attention = use_attention
        
        # Convolutional layers for spectral pattern extraction
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Preserve temporal dimension
        )
        
        # Temporal modeling
        self.temporal_model = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch, channels, freq, time]
            
        Returns:
            Tuple of (features [batch, output_dim], intermediates dict)
        """
        # Convolutional processing
        conv_out = self.conv_layers(x)  # [batch, 256, 1, time]
        conv_out = conv_out.squeeze(2)  # [batch, 256, time]
        conv_out = conv_out.transpose(1, 2)  # [batch, time, 256]
        
        # Temporal modeling
        temporal_out, (h_n, c_n) = self.temporal_model(conv_out)
        
        # Apply attention if enabled
        attention_weights = None
        if self.use_attention:
            attended, attention_weights = self.attention(
                temporal_out,
                temporal_out,
                temporal_out
            )
            temporal_out = attended
        
        # Global pooling (mean + max)
        mean_pool = torch.mean(temporal_out, dim=1)
        max_pool = torch.max(temporal_out, dim=1)[0]
        pooled = (mean_pool + max_pool) / 2
        
        # Output projection
        features = self.output_proj(pooled)
        
        intermediates = {
            'conv_features': conv_out,
            'temporal_features': temporal_out,
            'attention_weights': attention_weights,
            'pooled_features': pooled
        }
        
        return features, intermediates


class MelSpectrogramEncoder(SpectralEncoder):
    """
    Specialized encoder for mel-spectrograms with delta features.
    
    Optionally computes delta and delta-delta features for better
    temporal dynamics modeling.
    """
    
    def __init__(self,
                 n_mels: int = 80,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 dropout: float = 0.3,
                 use_delta: bool = True):
        super().__init__(
            input_channels=3 if use_delta else 1,
            input_dim=n_mels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        self.n_mels = n_mels
        self.use_delta = use_delta
        
    def compute_deltas(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute delta and delta-delta features.
        
        Args:
            x: Input tensor [batch, 1, freq, time]
            
        Returns:
            Tensor with deltas [batch, 3, freq, time]
        """
        # First derivative (delta)
        delta = x[:, :, :, 1:] - x[:, :, :, :-1]
        delta = F.pad(delta, (1, 0), mode='replicate')
        
        # Second derivative (delta-delta)
        delta_delta = delta[:, :, :, 1:] - delta[:, :, :, :-1]
        delta_delta = F.pad(delta_delta, (1, 0), mode='replicate')
        
        return torch.cat([x, delta, delta_delta], dim=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Mel-spectrogram [batch, 1, n_mels, time]
            
        Returns:
            Tuple of (features, intermediates)
        """
        if self.use_delta and x.shape[1] == 1:
            x = self.compute_deltas(x)
        
        return super().forward(x)


class MFCCEncoder(AcousticEncoder):
    """
    Encoder for MFCC features.
    
    MFCCs are classical speech features that capture spectral envelope.
    This encoder uses 1D convolutions followed by GRU for temporal modeling.
    """
    
    def __init__(self,
                 n_mfcc: int = 13,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 dropout: float = 0.3,
                 use_delta: bool = True):
        super().__init__(n_mfcc, hidden_dim, output_dim, dropout)
        
        self.n_mfcc = n_mfcc
        self.use_delta = use_delta
        
        input_size = n_mfcc * 3 if use_delta else n_mfcc
        
        # 1D CNN for MFCC processing
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Temporal modeling with GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    
    def compute_deltas(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute delta and delta-delta MFCC features.
        
        Args:
            x: MFCC features [batch, n_mfcc, time]
            
        Returns:
            Features with deltas [batch, n_mfcc*3, time]
        """
        # Delta
        delta = x[:, :, 1:] - x[:, :, :-1]
        delta = F.pad(delta, (1, 0), mode='replicate')
        
        # Delta-delta
        delta_delta = delta[:, :, 1:] - delta[:, :, :-1]
        delta_delta = F.pad(delta_delta, (1, 0), mode='replicate')
        
        return torch.cat([x, delta, delta_delta], dim=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: MFCC features [batch, n_mfcc, time]
            
        Returns:
            Tuple of (features, intermediates)
        """
        if self.use_delta and x.shape[1] == self.n_mfcc:
            x = self.compute_deltas(x)
        
        # CNN processing
        conv_out = self.conv1d(x)  # [batch, hidden_dim, time]
        conv_out = conv_out.transpose(1, 2)  # [batch, time, hidden_dim]
        
        # GRU processing
        gru_out, h_n = self.gru(conv_out)
        
        # Pool over time
        pooled = torch.mean(gru_out, dim=1)
        
        # Output
        features = self.output_layer(pooled)
        
        intermediates = {
            'conv_features': conv_out,
            'temporal_features': gru_out,
            'final_hidden': h_n
        }
        
        return features, intermediates


class WaveformEncoder(AcousticEncoder):
    """
    Direct waveform encoder using SincNet-inspired learnable filters.
    
    SincNet learns bandpass filters directly from raw waveforms,
    which is more interpretable and efficient than standard convolutions.
    
    Reference:
        Ravanelli & Bengio (2018) "Speaker Recognition from Raw Waveform with SincNet"
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 dropout: float = 0.3,
                 use_sincnet: bool = True):
        super().__init__(1, hidden_dim, output_dim, dropout)
        
        self.sample_rate = sample_rate
        self.use_sincnet = use_sincnet
        
        if use_sincnet:
            self.sincnet_layer = SincConv1d(
                in_channels=1,
                out_channels=80,
                kernel_size=251,
                sample_rate=sample_rate
            )
        else:
            self.conv1 = nn.Conv1d(1, 80, kernel_size=251, stride=1, padding=125)
        
        # Rest of the encoder
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout(dropout * 0.5),
            
            nn.Conv1d(80, 160, kernel_size=5, padding=2),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Dropout(dropout * 0.5),
            
            nn.Conv1d(160, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )
        
        # Temporal aggregation
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Raw waveform [batch, 1, samples] or [batch, samples]
            
        Returns:
            Tuple of (features, intermediates)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Initial convolution
        if self.use_sincnet:
            x = self.sincnet_layer(x)
        else:
            x = self.conv1(x)
        
        # Encode
        encoded = self.encoder(x)  # [batch, hidden_dim, time]
        encoded = encoded.transpose(1, 2)  # [batch, time, hidden_dim]
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(encoded)
        
        # Global pooling
        pooled = torch.mean(lstm_out, dim=1)
        
        # Output
        features = self.output_proj(pooled)
        
        intermediates = {
            'encoded_features': encoded,
            'temporal_features': lstm_out,
            'final_hidden': h_n
        }
        
        return features, intermediates


class SincConv1d(nn.Module):
    """
    SincNet-inspired convolutional layer with learnable sinc filters.
    
    This layer learns bandpass filters parametrized by center frequency
    and bandwidth, which is more interpretable and parameter-efficient
    than standard convolutions.
    
    IMPROVEMENTS APPLIED:
    - Proper buffer registration for device consistency
    - Named constants for magic numbers
    - Complete type hints
    - Optimized computations
    """
    
    # Named constants for Hamming window
    HAMMING_ALPHA: float = 0.54
    HAMMING_BETA: float = 0.46
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 sample_rate: int = 16000,
                 min_low_hz: float = 50.0,
                 min_band_hz: float = 50.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # Initialize filterbank parameters
        low_hz = 30.0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        
        mel = np.linspace(
            self._hz_to_mel(low_hz),
            self._hz_to_mel(high_hz),
            self.out_channels + 1
        )
        hz = self._mel_to_hz(mel)
        
        # Filter lower and upper frequencies (learnable parameters)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        
        # Hamming window (registered as buffer for device consistency)
        n_lin = torch.linspace(0, kernel_size - 1, steps=kernel_size)
        window = self.HAMMING_ALPHA - self.HAMMING_BETA * torch.cos(
            2 * math.pi * n_lin / kernel_size
        )
        self.register_buffer('window_', window)
        
        # Sinc filter time indices (registered as buffer)
        n = (self.kernel_size - 1) / 2.0
        n_range = torch.arange(-n, n + 1).view(1, -1)
        n_normalized = 2 * math.pi * n_range / self.sample_rate
        self.register_buffer('n_', n_normalized)
        
    def _hz_to_mel(self, hz: float) -> float:
        """Convert Hz to Mel scale."""
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel: np.ndarray) -> np.ndarray:
        """Convert Mel scale to Hz."""
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SincNet convolution.
        
        Args:
            x: Input waveform [batch, in_channels, samples]
            
        Returns:
            Filtered output [batch, out_channels, samples]
        """
        # Compute actual filter frequencies
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2
        )
        
        # Compute sinc filters
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        
        # Bandpass filter = high_pass - low_pass
        band_pass = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        
        # Normalize
        band = (high - low)[:, 0]
        band_pass = band_pass / (2 * band[:, None])
        
        # Reshape for convolution
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(
            x, 
            filters, 
            stride=1, 
            padding=(self.kernel_size - 1) // 2
        )


class ConformerEncoder(AcousticEncoder):
    """
    Conformer-based encoder combining convolution and self-attention.
    
    Conformer achieves SOTA results on speech tasks by combining
    the strengths of CNNs (local features) and Transformers (long-range dependencies).
    
    Reference:
        Gulati et al. (2020) "Conformer: Convolution-augmented Transformer for Speech Recognition"
    """
    
    def __init__(self,
                 input_dim: int = 80,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, output_dim, dropout)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input features [batch, time, input_dim]
            
        Returns:
            Tuple of (features, intermediates)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Store layer outputs for analysis
        layer_outputs = []
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x, block_intermediates = block(x)
            layer_outputs.append(x)
        
        # Global pooling
        pooled = torch.mean(x, dim=1)
        
        # Output projection
        features = self.output_proj(pooled)
        
        intermediates = {
            'conformer_output': x,
            'layer_outputs': layer_outputs
        }
        
        return features, intermediates


class ConformerBlock(nn.Module):
    """
    Single Conformer block with feed-forward, attention, and convolution modules.
    
    IMPROVEMENTS APPLIED:
    - Consistent return type with intermediates
    - Better documentation
    """
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # First feed-forward module (half-step)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(dim)
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=31, padding=15, groups=dim),
            nn.BatchNorm1d(dim * 2),
            nn.SiLU(),
            nn.Conv1d(dim * 2, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.conv_norm = nn.LayerNorm(dim)
        
        # Second feed-forward module (half-step)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch, time, dim]
            
        Returns:
            Tuple of (output tensor, intermediates dict)
        """
        intermediates = {}
        
        # First feed-forward (half-step residual)
        x = x + 0.5 * self.ff1(x)
        intermediates['after_ff1'] = x
        
        # Multi-head self-attention
        attn_out = self.attn_norm(x)
        attn_out, attn_weights = self.attn(attn_out, attn_out, attn_out)
        x = x + attn_out
        intermediates['after_attention'] = x
        intermediates['attention_weights'] = attn_weights
        
        # Convolution module
        conv_out = self.conv_norm(x)
        conv_out = conv_out.transpose(1, 2)  # [batch, dim, time]
        conv_out = self.conv(conv_out)
        conv_out = conv_out.transpose(1, 2)  # [batch, time, dim]
        x = x + conv_out
        intermediates['after_conv'] = x
        
        # Second feed-forward (half-step residual)
        x = x + 0.5 * self.ff2(x)
        intermediates['after_ff2'] = x
        
        # Final normalization
        output = self.final_norm(x)
        
        return output, intermediates