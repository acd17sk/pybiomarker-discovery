"""Complete visual biomarker model integrating all vision components"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from biomarkers.core.base import BiomarkerModel
from .face_analyzer import FaceAnalyzer
from .eye_tracker import EyeTracker
from .skin_analyzer import SkinColorAnalyzer



class VisualBiomarkerModel(BiomarkerModel):
    """Complete visual biomarker extraction model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract configuration
        self.input_channels = config.get('input_channels', 3)
        self.num_frames = config.get('num_frames', 16)
        self.image_size = config.get('image_size', (224, 224))
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_diseases = config.get('num_diseases', 5)
        self.dropout = config.get('dropout', 0.3)
        self.use_face = config.get('use_face', True)
        self.use_eyes = config.get('use_eyes', True)
        self.use_skin = config.get('use_skin', True)
        
        self._build_model()
    
    def _build_model(self):
        """Build complete visual biomarker model"""
        
        # Vision encoder (spatiotemporal feature extraction)
        self.vision_encoder = self._build_encoder()
        
        # Face analyzer
        if self.use_face:
            self.face_analyzer = FaceAnalyzer(
                input_channels=self.input_channels,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
        
        # Eye tracker
        if self.use_eyes:
            self.eye_tracker = EyeTracker(
                input_channels=self.input_channels,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
        
        # Skin color analyzer (CRITICAL - was missing!)
        if self.use_skin:
            self.skin_analyzer = SkinColorAnalyzer(
                input_channels=self.input_channels,
                hidden_dim=self.hidden_dim // 2
            )
        
        # Disease classifier
        self._build_disease_classifier()
        
        # Clinical scale predictors
        self._build_clinical_predictors()
        
        # Uncertainty quantification
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_diseases)
        )
    
    def _build_encoder(self):
        """Build spatiotemporal vision encoder"""
        return nn.Sequential(
            # 3D convolutions for spatiotemporal features
            nn.Conv3d(self.input_channels, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 4, 4))
        )
    
    def _build_disease_classifier(self):
        """Build disease classification head"""
        # Calculate feature dimension
        feature_dim = 256 * 16  # Base encoder features
        
        if self.use_face:
            feature_dim += self.hidden_dim // 2
        if self.use_eyes:
            feature_dim += self.hidden_dim // 2
        if self.use_skin:
            feature_dim += self.hidden_dim // 2
        
        self.classifier_proj = nn.Linear(feature_dim, self.hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_diseases)
        )
    
    def _build_clinical_predictors(self):
        """Build clinical scale predictors"""
        self.clinical_predictors = nn.ModuleDict()
        
        # UPDRS facial expression (Parkinson's)
        self.clinical_predictors['UPDRS_face'] = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Facial Action Coding System (FACS)
        self.clinical_predictors['FACS'] = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 46)  # 46 action units
        )
    
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract visual features from video frames"""
        # frames: [batch, time, channels, height, width]
        # Rearrange to [batch, channels, time, height, width]
        x = frames.transpose(1, 2)
        
        encoded = self.vision_encoder(x)
        return encoded.flatten(1)
    
    def extract_biomarkers(self, 
                          frames: torch.Tensor,
                          landmarks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Extract all visual biomarkers"""
        biomarkers = {}
        
        # Base visual features
        encoded_flat = self.extract_features(frames)
        
        # Facial biomarkers
        if self.use_face:
            face_output = self.face_analyzer(frames, landmarks)
            biomarkers.update({
                'expression_diversity': face_output['expression_diversity'],
                'facial_asymmetry': face_output['asymmetry_score'],
                'micro_expressions': face_output['micro_expressions'],
                'blink_rate': face_output['blink_rate'],
                'blink_completeness': face_output['blink_completeness'],
                'facial_mask_score': face_output['facial_mask_score'],
                'hypomimia_score': face_output['hypomimia_score'],
                'facial_pattern': face_output['pattern_probs']
            })
        
        # Eye tracking biomarkers
        if self.use_eyes:
            eye_output = self.eye_tracker(frames)
            biomarkers.update({
                'gaze_accuracy': eye_output['gaze_accuracy'],
                'pupil_diameter': eye_output['pupil_diameter'],
                'pupil_reactivity': eye_output['pupil_reactivity'],
                'saccade_velocity': eye_output['saccade_velocity'],
                'saccade_accuracy': eye_output['saccade_accuracy'],
                'saccade_latency': eye_output['saccade_latency'],
                'pursuit_gain': eye_output['pursuit_gain'],
                'pursuit_smoothness': eye_output['pursuit_smoothness'],
                'fixation_stability': eye_output['fixation_stability'],
                'oculomotor_pattern': eye_output['disorder_probs']
            })
        
        # Skin color biomarkers (CRITICAL - was missing!)
        if self.use_skin:
            skin_output = self.skin_analyzer(frames)
            biomarkers.update({
                'skin_perfusion': skin_output['perfusion_index'],
                'pallor_score': skin_output['pallor_score'],
                'cyanosis_score': skin_output['cyanosis_score'],
                'jaundice_score': skin_output['jaundice_score'],
                'redness_score': skin_output['redness_score'],
                'heart_rate': skin_output['heart_rate'],
                'respiratory_rate': skin_output['respiratory_rate'],
                'skin_temperature': skin_output['temperature_estimate']
            })
        
        return biomarkers
    
    def forward(self,
                frames: torch.Tensor,
                landmarks: Optional[torch.Tensor] = None,
                return_biomarkers: bool = True,
                return_uncertainty: bool = True,
                return_clinical: bool = True) -> Dict[str, torch.Tensor]:
        """Complete forward pass"""
        
        # Extract all biomarkers
        biomarkers = self.extract_biomarkers(frames, landmarks)
        
        # Prepare features for classification
        encoded_flat = self.extract_features(frames)
        classification_features = [encoded_flat]
        
        # Add component features
        if self.use_face:
            face_output = self.face_analyzer(frames, landmarks)
            classification_features.append(face_output['facial_features'])
        
        if self.use_eyes:
            eye_output = self.eye_tracker(frames)
            classification_features.append(eye_output['eye_features'])
        
        if self.use_skin:
            skin_output = self.skin_analyzer(frames)
            classification_features.append(skin_output['skin_features'])
        
        # Concatenate all features
        all_features = torch.cat(classification_features, dim=1)
        
        # Project to classifier dimension
        classifier_input = self.classifier_proj(all_features)
        
        # Disease classification
        disease_logits = self.classifier(classifier_input)
        disease_probs = F.softmax(disease_logits, dim=-1)
        
        # Prepare output
        output = {
            'logits': disease_logits,
            'probabilities': disease_probs,
            'features': encoded_flat,
            'predictions': disease_logits.argmax(dim=1)
        }
        
        # Add biomarkers if requested
        if return_biomarkers:
            output['biomarkers'] = biomarkers
        
        # Add clinical scores if requested
        if return_clinical and self.clinical_predictors:
            clinical_scores = {}
            for scale_name, predictor in self.clinical_predictors.items():
                scores = predictor(classifier_input)
                if scale_name == 'UPDRS_face':
                    scores = torch.sigmoid(scores) * 4  # Scale to 0-4
                else:
                    scores = torch.sigmoid(scores)
                clinical_scores[scale_name] = scores
            output['clinical_scores'] = clinical_scores
        
        # Add uncertainty if requested
        if return_uncertainty:
            log_variance = self.uncertainty_estimator(encoded_flat)
            uncertainty = torch.exp(log_variance)
            output['uncertainty'] = uncertainty
            output['confidence'] = 1.0 / (1.0 + uncertainty)
        
        return output
    
    def get_clinical_interpretation(self, biomarkers: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate clinical interpretation of visual biomarkers"""
        interpretation = {}
        
        # Facial mask interpretation (Parkinson's)
        if 'facial_mask_score' in biomarkers and biomarkers['facial_mask_score'].item() > 0.6:
            interpretation['parkinsonian_mask'] = {
                'detected': True,
                'severity': 'moderate' if biomarkers['facial_mask_score'].item() > 0.7 else 'mild',
                'hypomimia_score': f"{biomarkers['hypomimia_score'].item():.2f}",
                'clinical_note': 'Facial mask (hypomimia) detected, characteristic of Parkinson\'s disease.'
            }
        
        # Oculomotor dysfunction
        if 'saccade_accuracy' in biomarkers and biomarkers['saccade_accuracy'].item() < 0.5:
            interpretation['oculomotor_dysfunction'] = {
                'detected': True,
                'saccade_accuracy': f"{biomarkers['saccade_accuracy'].item():.2f}",
                'pursuit_gain': f"{biomarkers.get('pursuit_gain', torch.tensor(0)).item():.2f}",
                'clinical_note': 'Oculomotor dysfunction detected. Consider neurological evaluation.'
            }
        
        # Skin perfusion (cardiovascular/respiratory)
        if 'skin_perfusion' in biomarkers and biomarkers['skin_perfusion'].item() < 0.5:
            interpretation['perfusion_abnormality'] = {
                'detected': True,
                'perfusion_index': f"{biomarkers['skin_perfusion'].item():.2f}",
                'pallor_score': f"{biomarkers.get('pallor_score', torch.tensor(0)).item():.2f}",
                'clinical_note': 'Reduced skin perfusion detected. Assess cardiovascular status.'
            }
        
        # Cyanosis detection
        if 'cyanosis_score' in biomarkers and biomarkers['cyanosis_score'].item() > 0.6:
            interpretation['cyanosis'] = {
                'detected': True,
                'severity': f"{biomarkers['cyanosis_score'].item():.2f}",
                'clinical_note': 'Cyanosis detected. Check oxygen saturation urgently.'
            }
        
        # Facial asymmetry (stroke risk)
        if 'facial_asymmetry' in biomarkers and biomarkers['facial_asymmetry'].item() > 0.7:
            interpretation['facial_asymmetry'] = {
                'detected': True,
                'asymmetry_score': f"{biomarkers['facial_asymmetry'].item():.2f}",
                'clinical_note': 'Significant facial asymmetry. Consider stroke evaluation (FAST protocol).'
            }
        
        return interpretation