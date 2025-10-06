import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from biomarkers.core.base import BiomarkerModel

class MovementBiomarkerModel(BiomarkerModel):
    """Extract biomarkers from movement/accelerometer data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.input_channels = config.get('input_channels', 6)  # 3-axis acc + 3-axis gyro
        self.sequence_length = config.get('sequence_length', 1000)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_diseases = config.get('num_diseases', 5)
        super().__init__(config)
        
    def _build_model(self):
        # Temporal Convolutional Network for movement patterns
        self.tcn = nn.Sequential(
            nn.Conv1d(self.input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Frequency domain analyzer (for tremor detection)
        self.freq_analyzer = nn.Sequential(
            nn.Linear(self.sequence_length // 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Gait analyzer
        self.gait_lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Tremor detector
        self.tremor_detector = nn.Sequential(
            nn.Linear(64 * self.input_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # Tremor characteristics: amplitude, frequency, regularity, etc.
        )
        
        # Balance analyzer
        self.balance_analyzer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Balance metrics: sway, stability, etc.
        )
        
        # Bradykinesia detector
        self.bradykinesia_detector = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Speed, amplitude, rhythm
        )
        
        # Disease classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 12, 512),  # Features + biomarkers
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.num_diseases)
        )
        
        # Uncertainty quantification
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_diseases)
        )
        
    def _compute_fft_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute FFT features for tremor analysis"""
        # Apply FFT along time dimension
        fft = torch.fft.rfft(x, dim=-1)
        magnitude = torch.abs(fft)
        return magnitude
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract movement features"""
        batch_size = x.shape[0]
        
        # Temporal features
        temporal_features = self.tcn(x)
        
        # Frequency features
        fft_features = self._compute_fft_features(x)
        freq_features = self.freq_analyzer(fft_features).reshape(batch_size, -1)
        
        # Gait features
        temporal_features_seq = temporal_features.transpose(1, 2)
        gait_features, _ = self.gait_lstm(temporal_features_seq)
        gait_features = gait_features[:, -1, :]  # Take last timestep
        
        return gait_features, freq_features
        
    def forward(self, x: torch.Tensor, return_biomarkers: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass with biomarker extraction"""
        gait_features, freq_features = self.extract_features(x)
        
        # Extract biomarkers
        tremor_metrics = self.tremor_detector(freq_features)
        balance_metrics = self.balance_analyzer(gait_features)
        bradykinesia_metrics = self.bradykinesia_detector(gait_features)
        
        # Combine all features
        all_biomarkers = torch.cat([
            tremor_metrics,
            balance_metrics,
            bradykinesia_metrics
        ], dim=1)
        
        combined_features = torch.cat([gait_features, all_biomarkers], dim=1)
        
        # Disease prediction
        disease_logits = self.classifier(combined_features)
        
        # Uncertainty estimation
        log_variance = self.uncertainty_head(gait_features)
        uncertainty = torch.exp(log_variance)
        
        output = {
            'logits': disease_logits,
            'probabilities': F.softmax(disease_logits, dim=1),
            'features': gait_features,
            'uncertainty': uncertainty,
            'confidence': 1.0 / (1.0 + uncertainty)
        }
        
        if return_biomarkers:
            output['biomarkers'] = {
                'tremor': tremor_metrics,
                'balance': balance_metrics,
                'bradykinesia': bradykinesia_metrics
            }
        
        return output