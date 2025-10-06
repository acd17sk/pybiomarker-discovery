"""Gait analysis modules for movement biomarkers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class GaitAnalyzer(nn.Module):
    """Complete gait analysis module"""
    
    def __init__(self,
                 input_channels: int = 6,  # 3 acc + 3 gyro
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # Sub-components
        self.step_detector = StepDetector(input_channels, hidden_dim // 2)
        self.gait_cycle = GaitCycleAnalyzer(input_channels, hidden_dim // 2)
        self.balance = BalanceAnalyzer(input_channels, hidden_dim // 2)
        self.fog_detector = FreezingOfGaitDetector(input_channels, hidden_dim // 2)
        
        # Feature fusion
        fusion_dim = hidden_dim * 2  # 4 components * hidden_dim/2
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Gait quality assessment
        self.gait_quality = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 gait quality metrics
        )
        
        # Clinical gait patterns
        self.clinical_patterns = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # Normal, Parkinsonian, Ataxic, Hemiplegic, Neuropathic, Antalgic
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Sensor data [batch, channels, time]
        """
        # Extract gait components
        step_features = self.step_detector(x)
        cycle_features = self.gait_cycle(x)
        balance_features = self.balance(x)
        fog_features = self.fog_detector(x)
        
        # Concatenate features
        combined = torch.cat([
            step_features['features'],
            cycle_features['features'],
            balance_features['features'],
            fog_features['features']
        ], dim=-1)
        
        # Fuse features
        fused = self.feature_fusion(combined)
        
        # Assess gait quality
        quality_metrics = self.gait_quality(fused)
        
        # Classify clinical patterns
        pattern_logits = self.clinical_patterns(fused)
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        
        return {
            'gait_features': fused,
            'quality_metrics': quality_metrics,
            'pattern_logits': pattern_logits,
            'pattern_probs': pattern_probs,
            'step_count': step_features['step_count'],
            'cadence': step_features['cadence'],
            'stride_length': cycle_features['stride_length'],
            'stride_variability': cycle_features['variability'],
            'balance_score': balance_features['balance_score'],
            'sway': balance_features['sway'],
            'fog_episodes': fog_features['episodes'],
            'fog_duration': fog_features['duration']
        }


class StepDetector(nn.Module):
    """Detect and analyze individual steps"""
    
    def __init__(self, input_channels: int = 6, output_dim: int = 128):
        super().__init__()
        
        # CNN for step detection
        self.step_cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=output_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Step characteristics predictor
        self.step_characteristics = nn.Sequential(
            nn.Linear(output_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # step_time, swing_time, stance_time, double_support, step_height, step_width
        )
        
        # Cadence estimator
        self.cadence_estimator = nn.Sequential(
            nn.Linear(output_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # steps per minute
        )
        
        # Step regularity
        self.regularity_scorer = nn.Sequential(
            nn.Linear(output_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # regularity score 0-1
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect and analyze steps"""
        # CNN processing
        conv_features = self.step_cnn(x)
        conv_features = conv_features.transpose(1, 2)  # [batch, time, features]
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_features)
        
        # Global pooling
        pooled = torch.mean(lstm_out, dim=1)
        
        # Extract step characteristics
        characteristics = self.step_characteristics(pooled)
        cadence = self.cadence_estimator(pooled).squeeze(-1)
        regularity = self.regularity_scorer(pooled).squeeze(-1)
        
        # Estimate step count (simplified - in practice use peak detection)
        step_count = torch.abs(cadence * x.shape[-1] / (60 * 100))  # Assuming 100Hz sampling
        
        return {
            'features': pooled[:, :x.shape[1]*21],  # Return portion of features
            'step_count': step_count,
            'cadence': cadence,
            'characteristics': characteristics,
            'regularity': regularity
        }


class GaitCycleAnalyzer(nn.Module):
    """Analyze complete gait cycles"""
    
    def __init__(self, input_channels: int = 6, output_dim: int = 128):
        super().__init__()
        
        # Cycle detection network
        self.cycle_detector = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Temporal modeling
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=output_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Stride analyzer
        self.stride_analyzer = nn.Sequential(
            nn.Linear(output_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # stride_length, stride_time, stride_velocity, asymmetry
        )
        
        # Variability analyzer
        self.variability_analyzer = nn.Sequential(
            nn.Linear(output_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # cv_stride_time, cv_stride_length, cv_stride_width
        )
        
        # Phase analyzer
        self.phase_analyzer = nn.Sequential(
            nn.Linear(output_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # stance_phase%, swing_phase%, loading_response%, terminal_stance%
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze gait cycles"""
        # Detect cycles
        cycle_features = self.cycle_detector(x)
        cycle_features = cycle_features.transpose(1, 2)
        
        # GRU processing
        gru_out, _ = self.gru(cycle_features)
        pooled = torch.mean(gru_out, dim=1)
        
        # Analyze stride
        stride_metrics = self.stride_analyzer(pooled)
        variability = self.variability_analyzer(pooled)
        phases = self.phase_analyzer(pooled)
        phases = torch.sigmoid(phases) * 100  # Convert to percentages
        
        return {
            'features': pooled[:, :x.shape[1]*21],
            'stride_length': stride_metrics[:, 0],
            'stride_time': stride_metrics[:, 1],
            'stride_velocity': stride_metrics[:, 2],
            'asymmetry': torch.sigmoid(stride_metrics[:, 3]),
            'variability': variability,
            'phases': phases
        }


class BalanceAnalyzer(nn.Module):
    """Analyze balance and postural control"""
    
    def __init__(self, input_channels: int = 6, output_dim: int = 128):
        super().__init__()
        
        # Center of mass estimation
        self.com_estimator = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Sway analyzer
        self.sway_analyzer = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(100)
        )
        
        # Balance metrics
        self.balance_metrics = nn.Sequential(
            nn.Linear(64 * 100, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # ML_sway, AP_sway, sway_area, sway_velocity, COP_range, stability_index, fall_risk, balance_confidence
        )
        
        # Postural transitions
        self.transition_detector = nn.Sequential(
            nn.Linear(64 * 100, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # sit_to_stand, stand_to_sit, turning, reaching, recovery
        )
        
        self.output_proj = nn.Linear(8 + 5, output_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze balance"""
        # Estimate center of mass
        com_features = self.com_estimator(x)
        
        # Analyze sway
        sway_features = self.sway_analyzer(com_features)
        sway_flat = sway_features.flatten(1)
        
        # Calculate balance metrics
        metrics = self.balance_metrics(sway_flat)
        transitions = self.transition_detector(sway_flat)
        transitions = torch.sigmoid(transitions)
        
        # Combine features
        combined = torch.cat([metrics, transitions], dim=-1)
        features = self.output_proj(combined)
        
        return {
            'features': features[:, :x.shape[1]*21],
            'balance_score': torch.sigmoid(metrics[:, 5]),  # stability_index
            'sway': metrics[:, :3],  # ML, AP, area
            'fall_risk': torch.sigmoid(metrics[:, 6]),
            'transitions': transitions,
            'balance_confidence': torch.sigmoid(metrics[:, 7])
        }


class FreezingOfGaitDetector(nn.Module):
    """Detect freezing of gait episodes"""
    
    def __init__(self, input_channels: int = 6, output_dim: int = 128):
        super().__init__()
        
        # Frequency analyzer for FOG detection
        self.freq_analyzer = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=32, stride=16),  # Analyze frequency components
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=16, stride=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=4),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Temporal context
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # FOG detector
        self.fog_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # no_fog, pre_fog, fog
        )
        
        # FOG characteristics
        self.fog_characteristics = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # duration, frequency, amplitude, trembling, shuffling, complete_block
        )
        
        # Trigger detector
        self.trigger_detector = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # turning, doorway, dual_task, stress, crowded
        )
        
        self.output_proj = nn.Linear(256, output_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect FOG episodes"""
        # Frequency analysis
        freq_features = self.freq_analyzer(x)
        freq_features = freq_features.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(freq_features)
        
        # Pool features
        pooled = torch.mean(lstm_out, dim=1)
        
        # FOG classification
        fog_logits = self.fog_classifier(pooled)
        fog_probs = F.softmax(fog_logits, dim=-1)
        
        # FOG characteristics
        characteristics = self.fog_characteristics(pooled)
        triggers = self.trigger_detector(pooled)
        triggers = torch.sigmoid(triggers)
        
        # Calculate FOG metrics
        fog_presence = fog_probs[:, 2]  # Probability of FOG
        pre_fog = fog_probs[:, 1]  # Pre-FOG state
        
        features = self.output_proj(pooled)
        
        return {
            'features': features[:, :x.shape[1]*21],
            'fog_probability': fog_presence,
            'pre_fog_probability': pre_fog,
            'episodes': characteristics[:, 0],  # Number of episodes
            'duration': characteristics[:, 1],  # Average duration
            'characteristics': characteristics,
            'triggers': triggers
        }


class DualTaskGaitAnalyzer(nn.Module):
    """Analyze gait during dual-task conditions"""
    
    def __init__(self, input_channels: int = 6, output_dim: int = 128):
        super().__init__()
        
        # Single task gait encoder
        self.single_task_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Dual task effect analyzer
        self.dual_task_analyzer = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(50)
        )
        
        # Cost analyzer
        self.cost_analyzer = nn.Sequential(
            nn.Linear(256 * 50, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # speed_cost, variability_cost, stability_cost, cognitive_cost, motor_cost, attention_cost, prioritization, automaticity
        )
        
        self.output_proj = nn.Linear(8, output_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze dual-task gait"""
        # Encode gait
        gait_features = self.single_task_encoder(x)
        
        # Analyze dual-task effects
        dt_features = self.dual_task_analyzer(gait_features)
        dt_flat = dt_features.flatten(1)
        
        # Calculate dual-task costs
        costs = self.cost_analyzer(dt_flat)
        costs_normalized = torch.sigmoid(costs)
        
        features = self.output_proj(costs)
        
        return {
            'features': features,
            'dual_task_costs': costs_normalized,
            'overall_cost': costs_normalized.mean(dim=-1)
        }