import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from biomarkers.core.base import BiomarkerModel

class VoiceBiomarkerModel(BiomarkerModel):
    """Extract biomarkers from voice/speech data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.input_dim = config.get('input_dim', 80)  # Mel spectrogram bins
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_diseases = config.get('num_diseases', 5)
        self.dropout = config.get('dropout', 0.3)
        super().__init__(config)
        
    def _build_model(self):
        # Acoustic encoder (CNN for spectral features)
        self.acoustic_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Temporal encoder (LSTM for temporal patterns)
        self.temporal_encoder = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=8,
            dropout=self.dropout
        )
        
        # Biomarker extractors
        self.tremor_detector = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.prosody_analyzer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10 prosodic features
        )
        
        self.cognitive_load_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Disease prediction head
        self.disease_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 12, 256),  # +12 for biomarkers
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_diseases)
        )
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract voice features from mel spectrogram"""
        batch_size, channels, time, freq = x.shape
        
        # Acoustic features
        acoustic_feat = self.acoustic_encoder(x)
        acoustic_feat = acoustic_feat.view(batch_size, 128, -1).transpose(1, 2)
        
        # Temporal features
        temporal_feat, _ = self.temporal_encoder(acoustic_feat)
        
        # Self-attention
        attended_feat, _ = self.attention(
            temporal_feat.transpose(0, 1),
            temporal_feat.transpose(0, 1),
            temporal_feat.transpose(0, 1)
        )
        attended_feat = attended_feat.transpose(0, 1)
        
        # Global pooling
        features = torch.mean(attended_feat, dim=1)
        
        return features
    
    def forward(self, x: torch.Tensor, return_biomarkers: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass with biomarker extraction"""
        features = self.extract_features(x)
        
        # Extract biomarkers
        tremor_score = self.tremor_detector(features)
        prosody_features = self.prosody_analyzer(features)
        cognitive_load = self.cognitive_load_estimator(features)
        
        # Concatenate all features for disease prediction
        all_features = torch.cat([features, tremor_score, prosody_features, cognitive_load], dim=1)
        disease_logits = self.disease_classifier(all_features)
        
        output = {
            'logits': disease_logits,
            'probabilities': F.softmax(disease_logits, dim=1),
            'features': features
        }
        
        if return_biomarkers:
            output['biomarkers'] = {
                'tremor_score': tremor_score,
                'prosody': prosody_features,
                'cognitive_load': cognitive_load
            }
        
        return output