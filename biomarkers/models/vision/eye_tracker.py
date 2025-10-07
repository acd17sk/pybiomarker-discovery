"""Eye tracking and oculomotor analysis modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EyeTracker(nn.Module):
    """Complete eye tracking and oculomotor analysis"""
    
    def __init__(self,
                 input_channels: int = 3,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # Sub-components
        self.gaze_estimator = GazeEstimator(input_channels, hidden_dim // 2)
        self.pupil_analyzer = PupilAnalyzer(input_channels, hidden_dim // 2)
        self.saccade_detector = SaccadeDetector(hidden_dim // 2)
        self.smooth_pursuit = SmoothPursuitAnalyzer(hidden_dim // 2)
        self.fixation_analyzer = FixationAnalyzer(hidden_dim // 2)
        
        # Feature fusion
        fusion_dim = hidden_dim * 2 + hidden_dim // 2  # 5 components
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Oculomotor disorder classifier
        self.disorder_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Multiple oculomotor disorders
        )
    
    def forward(self, 
                frames: torch.Tensor,
                gaze_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: Video frames [batch, time, channels, height, width]
            gaze_positions: Gaze coordinates [batch, time, 2]
        """
        # Estimate gaze if not provided
        gaze_output = self.gaze_estimator(frames)
        if gaze_positions is None:
            gaze_positions = gaze_output['gaze_coordinates']
        
        # Analyze pupils
        pupil_output = self.pupil_analyzer(frames)
        
        # Detect saccades
        saccade_output = self.saccade_detector(gaze_positions)
        
        # Analyze smooth pursuit
        pursuit_output = self.smooth_pursuit(gaze_positions)
        
        # Analyze fixations
        fixation_output = self.fixation_analyzer(gaze_positions)
        
        # Concatenate features
        combined = torch.cat([
            gaze_output['features'],
            pupil_output['features'],
            saccade_output['features'],
            pursuit_output['features'],
            fixation_output['features']
        ], dim=-1)
        
        # Fuse features
        fused = self.feature_fusion(combined)
        
        # Classify disorders
        disorder_logits = self.disorder_classifier(fused)
        disorder_probs = F.softmax(disorder_logits, dim=-1)
        
        return {
            'eye_features': fused,
            'gaze_coordinates': gaze_positions,
            'gaze_accuracy': gaze_output['accuracy'],
            'pupil_diameter': pupil_output['diameter'],
            'pupil_reactivity': pupil_output['reactivity'],
            'saccade_velocity': saccade_output['velocity'],
            'saccade_accuracy': saccade_output['accuracy'],
            'saccade_latency': saccade_output['latency'],
            'pursuit_gain': pursuit_output['gain'],
            'pursuit_smoothness': pursuit_output['smoothness'],
            'fixation_stability': fixation_output['stability'],
            'fixation_duration': fixation_output['duration'],
            'disorder_logits': disorder_logits,
            'disorder_probs': disorder_probs
        }


class GazeEstimator(nn.Module):
    """Estimate gaze direction and point of regard"""
    
    def __init__(self, input_channels: int = 3, output_dim: int = 128):
        super().__init__()
        
        # Eye region CNN
        self.eye_cnn = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(1, 7, 7), padding=(0, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 4, 4))
        )
        
        # Gaze coordinate regressor
        self.gaze_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # (x, y) coordinates
        )
        
        # Gaze accuracy estimator
        self.accuracy_estimator = nn.Sequential(
            nn.Linear(128 * 8 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for gaze
        self.gaze_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.2
        )
        
        self.output_proj = nn.Linear(128 * 8 * 16, output_dim)
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate gaze direction"""
        # Rearrange for 3D CNN
        x = frames.transpose(1, 2)
        
        # Extract eye features
        eye_features = self.eye_cnn(x)
        eye_flat = eye_features.flatten(1)
        
        # Estimate gaze coordinates
        gaze_coords = self.gaze_regressor(eye_flat)
        gaze_coords = torch.tanh(gaze_coords)  # Normalize to [-1, 1]
        
        # Estimate accuracy
        accuracy = self.accuracy_estimator(eye_flat).squeeze(-1)
        
        # Project features
        features = self.output_proj(eye_flat)
        
        return {
            'features': features,
            'gaze_coordinates': gaze_coords,
            'accuracy': accuracy
        }


class PupilAnalyzer(nn.Module):
    """Analyze pupil size, reactivity, and dynamics"""
    
    def __init__(self, input_channels: int = 3, output_dim: int = 128):
        super().__init__()
        
        # Pupil segmentation network
        self.pupil_segmentor = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((16, 4, 4))
        )
        
        # Pupil diameter estimator
        self.diameter_estimator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Pupil reactivity analyzer (light reflex)
        self.reactivity_analyzer = nn.Sequential(
            nn.Linear(128 * 16 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # latency, amplitude, velocity, duration, recovery
        )
        
        # Hippus detector (rhythmic pupil oscillation)
        self.hippus_detector = nn.Sequential(
            nn.Linear(128 * 16 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # presence, frequency, amplitude
        )
        
        # Anisocoria detector (unequal pupils)
        self.anisocoria_detector = nn.Sequential(
            nn.Linear(128 * 16 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(128 * 16 * 4, output_dim)
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze pupil characteristics"""
        x = frames.transpose(1, 2)
        
        # Segment and extract pupil features
        pupil_features = self.pupil_segmentor(x)
        pupil_flat = pupil_features.flatten(1)
        
        # Estimate diameter
        diameter = self.diameter_estimator(pupil_flat)
        diameter = torch.abs(diameter) * 10  # Scale to mm
        
        # Analyze reactivity
        reactivity = self.reactivity_analyzer(pupil_flat)
        reactivity = torch.sigmoid(reactivity)
        
        # Detect hippus
        hippus = self.hippus_detector(pupil_flat)
        hippus_presence = torch.sigmoid(hippus[:, 0])
        
        # Detect anisocoria
        anisocoria = self.anisocoria_detector(pupil_flat).squeeze(-1)
        
        features = self.output_proj(pupil_flat)
        
        return {
            'features': features,
            'diameter': diameter.squeeze(-1),
            'reactivity': reactivity.mean(dim=-1),
            'reactivity_components': reactivity,
            'hippus_presence': hippus_presence,
            'anisocoria': anisocoria
        }


class SaccadeDetector(nn.Module):
    """Detect and analyze saccadic eye movements"""
    
    def __init__(self, input_dim: int = 128):
        super().__init__()
        
        # Temporal encoder for gaze trajectory
        self.trajectory_encoder = nn.LSTM(
            input_size=2,  # (x, y) coordinates
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Saccade detector
        self.saccade_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Saccade characteristics analyzer
        self.saccade_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # amplitude, velocity, acceleration, latency, accuracy, duration, overshoot, undershoot
        )
        
        # Saccade type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # prosaccade, antisaccade, memory-guided, predictive, spontaneous
        )
        
        # Dysmetria detector (overshoot/undershoot)
        self.dysmetria_detector = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(128, input_dim)
    
    def forward(self, gaze_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            gaze_positions: [batch, time, 2]
        """
        # Encode gaze trajectory
        trajectory_features, _ = self.trajectory_encoder(gaze_positions)
        
        # Pool over time
        pooled = torch.mean(trajectory_features, dim=1)
        
        # Detect saccades
        saccade_prob = self.saccade_detector(pooled).squeeze(-1)
        
        # Analyze saccade characteristics
        characteristics = self.saccade_analyzer(pooled)
        
        # Extract key metrics
        velocity = torch.abs(characteristics[:, 1]) * 500  # deg/s
        latency = torch.abs(characteristics[:, 3]) * 300  # ms
        accuracy = torch.sigmoid(characteristics[:, 4])
        
        # Classify saccade type
        type_logits = self.type_classifier(pooled)
        type_probs = F.softmax(type_logits, dim=-1)
        
        # Detect dysmetria
        dysmetria = self.dysmetria_detector(pooled).squeeze(-1)
        
        features = self.output_proj(pooled)
        
        return {
            'features': features,
            'saccade_probability': saccade_prob,
            'velocity': velocity,
            'latency': latency,
            'accuracy': accuracy,
            'characteristics': characteristics,
            'type_probabilities': type_probs,
            'dysmetria': dysmetria
        }


class SmoothPursuitAnalyzer(nn.Module):
    """Analyze smooth pursuit eye movements"""
    
    def __init__(self, input_dim: int = 128):
        super().__init__()
        
        # Pursuit trajectory encoder
        self.pursuit_encoder = nn.GRU(
            input_size=2,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Pursuit gain calculator (eye velocity / target velocity)
        self.gain_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Pursuit quality analyzer
        self.quality_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # smoothness, accuracy, latency, catch-up_saccades, anticipatory_saccades, pursuit_gain
        )
        
        # Direction-specific pursuit analyzer
        self.direction_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # horizontal_left, horizontal_right, vertical_up, vertical_down
        )
        
        self.output_proj = nn.Linear(128, input_dim)
    
    def forward(self, gaze_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze smooth pursuit"""
        # Encode pursuit trajectory
        pursuit_features, _ = self.pursuit_encoder(gaze_positions)
        
        # Pool over time
        pooled = torch.mean(pursuit_features, dim=1)
        
        # Estimate pursuit gain
        gain = self.gain_estimator(pooled)
        gain = torch.sigmoid(gain) * 1.2  # Gain typically 0.8-1.0
        
        # Analyze quality
        quality = self.quality_analyzer(pooled)
        smoothness = torch.sigmoid(quality[:, 0])
        
        # Direction-specific analysis
        direction_scores = self.direction_analyzer(pooled)
        direction_scores = torch.sigmoid(direction_scores)
        
        features = self.output_proj(pooled)
        
        return {
            'features': features,
            'gain': gain.squeeze(-1),
            'smoothness': smoothness,
            'quality_metrics': quality,
            'direction_scores': direction_scores
        }


class FixationAnalyzer(nn.Module):
    """Analyze fixation patterns and stability"""
    
    def __init__(self, input_dim: int = 128):
        super().__init__()
        
        # Fixation encoder
        self.fixation_encoder = nn.LSTM(
            input_size=2,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Fixation stability calculator
        self.stability_calculator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Fixation characteristics analyzer
        self.fixation_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # duration, frequency, dispersion, drift, microsaccades, square_wave_jerks, nystagmus
        )
        
        # Fixation pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # normal, unstable, scanning, searching, pathological
        )
        
        self.output_proj = nn.Linear(128, input_dim)
    
    def forward(self, gaze_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze fixations"""
        # Encode fixation patterns
        fixation_features, _ = self.fixation_encoder(gaze_positions)
        
        # Pool over time
        pooled = torch.mean(fixation_features, dim=1)
        
        # Calculate stability
        stability = self.stability_calculator(pooled).squeeze(-1)
        
        # Analyze characteristics
        characteristics = self.fixation_analyzer(pooled)
        duration = torch.abs(characteristics[:, 0]) * 1000  # ms
        
        # Classify pattern
        pattern_logits = self.pattern_classifier(pooled)
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        
        features = self.output_proj(pooled)
        
        return {
            'features': features,
            'stability': stability,
            'duration': duration,
            'characteristics': characteristics,
            'pattern_probabilities': pattern_probs
        }


class VergenceAnalyzer(nn.Module):
    """Analyze vergence eye movements (convergence/divergence)"""
    
    def __init__(self, input_dim: int = 128):
        super().__init__()
        
        # Vergence encoder (binocular coordination)
        self.vergence_encoder = nn.Sequential(
            nn.Linear(4, 64),  # Left and right eye positions
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Vergence metrics calculator
        self.vergence_calculator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # convergence_amplitude, divergence_amplitude, velocity, accuracy, latency, fusion_range
        )
        
        # Vergence dysfunction detector
        self.dysfunction_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # convergence_insufficiency, divergence_excess, vergence_adaptation, phoria
        )
        
        self.output_proj = nn.Linear(128, input_dim)
    
    def forward(self, 
                left_eye_positions: torch.Tensor,
                right_eye_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze vergence movements"""
        # Combine binocular positions
        binocular = torch.cat([left_eye_positions, right_eye_positions], dim=-1)
        
        # Encode vergence
        vergence_features = self.vergence_encoder(binocular)
        
        # Calculate metrics
        metrics = self.vergence_calculator(vergence_features)
        metrics = torch.sigmoid(metrics)
        
        # Detect dysfunction
        dysfunction = self.dysfunction_detector(vergence_features)
        dysfunction = torch.sigmoid(dysfunction)
        
        features = self.output_proj(vergence_features)
        
        return {
            'features': features,
            'convergence_amplitude': metrics[:, 0],
            'divergence_amplitude': metrics[:, 1],
            'vergence_metrics': metrics,
            'dysfunction_scores': dysfunction
        }