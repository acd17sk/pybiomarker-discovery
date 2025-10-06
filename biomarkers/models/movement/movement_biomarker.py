"""Complete movement biomarker model integrating all components"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from biomarkers.core.base import BiomarkerModel
from .gait_analyzer import (
    GaitAnalyzer,
    DualTaskGaitAnalyzer
)
from .tremor_detector import (
    TremorDetector,
    EssentialTremorAnalyzer,
    ParkinsonianTremorAnalyzer
)

class MovementBiomarkerModel(BiomarkerModel):
    """Complete movement biomarker extraction model"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Extract configuration
        self.input_channels = config.get('input_channels', 6)  # 3 acc + 3 gyro
        self.window_size = config.get('window_size', 1000)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_diseases = config.get('num_diseases', 5)
        self.dropout = config.get('dropout', 0.3)
        self.use_gait = config.get('use_gait', True)
        self.use_tremor = config.get('use_tremor', True)
        self.use_dual_task = config.get('use_dual_task', False)
        self.clinical_scales = config.get('clinical_scales', ['UPDRS', 'MDS-UPDRS'])
        self._build_model()

    def _build_model(self):
        """Build complete movement biomarker model"""
        # Movement encoder
        self.movement_encoder = self._build_encoder()
        # Gait analyzer
        if self.use_gait:
            self.gait_analyzer = GaitAnalyzer(
                input_channels=self.input_channels,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
            if self.use_dual_task:
                self.dual_task_analyzer = DualTaskGaitAnalyzer(
                    input_channels=self.input_channels,
                    output_dim=self.hidden_dim // 2
                )
        # Tremor detector
        if self.use_tremor:
            self.tremor_detector = TremorDetector(
                input_channels=self.input_channels,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
            # Disease-specific tremor analyzers
            self.et_analyzer = EssentialTremorAnalyzer(self.hidden_dim)
            self.pd_tremor_analyzer = ParkinsonianTremorAnalyzer(self.hidden_dim)
        # Cardinal motor feature detectors
        self.bradykinesia_detector = self._build_bradykinesia_detector()
        self.rigidity_analyzer = self._build_rigidity_analyzer()
        self.postural_analyzer = self._build_postural_analyzer()
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
        """Build movement encoder"""
        return nn.Sequential(
            # Multi-scale temporal convolutions
            nn.Conv1d(self.input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=21, padding=10),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def _build_bradykinesia_detector(self):
        """Build bradykinesia detection module"""
        return nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # slowness, amplitude_decrement, arrests, fatiguing, hesitation, sequence_effect
        )

    def _build_rigidity_analyzer(self):
        """Build rigidity analysis module"""
        return nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # tone, cogwheel, lead_pipe, velocity_dependent, distribution
        )

    def _build_postural_analyzer(self):
        """Build postural instability analyzer"""
        return nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # pull_test, stance, turn_stability, rise_from_chair, retropulsion, propulsion, lateral_instability
        )

    def _build_disease_classifier(self):
        """Build disease classification head"""
        # Calculate total feature dimension
        feature_dim = 256  # Base encoder features
        if self.use_gait:
            feature_dim += self.hidden_dim // 2  # Gait features
        if self.use_tremor:
            feature_dim += self.hidden_dim // 2  # Tremor features
        feature_dim += 6 + 5 + 7  # Bradykinesia + Rigidity + Postural
        # Multi-layer classifier
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
        if 'UPDRS' in self.clinical_scales:
            self.clinical_predictors['UPDRS'] = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 33)  # 33 motor items
            )
        if 'MDS-UPDRS' in self.clinical_scales:
            self.clinical_predictors['MDS-UPDRS'] = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 33)  # Part III motor examination
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract movement features"""
        encoded = self.movement_encoder(x)
        return encoded.flatten(1)

    def extract_biomarkers(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract all movement biomarkers"""
        biomarkers = {}
        # Encode movement
        encoded = self.movement_encoder(x)
        encoded_flat = encoded.flatten(1)
        # Gait biomarkers
        if self.use_gait:
            gait_output = self.gait_analyzer(x)
            biomarkers.update({
                'step_count': gait_output['step_count'],
                'cadence': gait_output['cadence'],
                'stride_length': gait_output['stride_length'],
                'stride_variability': gait_output['stride_variability'],
                'balance_score': gait_output['balance_score'],
                'sway': gait_output['sway'],
                'fog_episodes': gait_output['fog_episodes'],
                'fog_duration': gait_output['fog_duration'],
                'gait_pattern': gait_output['pattern_probs']
            })
            if self.use_dual_task:
                dt_output = self.dual_task_analyzer(x)
                biomarkers['dual_task_cost'] = dt_output['overall_cost']
                biomarkers['dual_task_costs'] = dt_output['dual_task_costs']
        # Tremor biomarkers
        if self.use_tremor:
            tremor_output = self.tremor_detector(x)
            biomarkers.update({
                'tremor_present': tremor_output['tremor_present'],
                'tremor_type': tremor_output['tremor_type'],
                'tremor_severity': tremor_output['tremor_severity'],
                'tremor_frequency': tremor_output['frequency'],
                'tremor_amplitude': tremor_output['amplitude']
            })
            # Essential tremor analysis
            if tremor_output['tremor_type'][:, 2].mean() > 0.3:  # Kinetic tremor
                et_output = self.et_analyzer(tremor_output['tremor_features'])
                biomarkers['et_features'] = et_output['et_features']
            # Parkinsonian tremor analysis
            if tremor_output['tremor_type'][:, 0].mean() > 0.3:  # Rest tremor
                pd_output = self.pd_tremor_analyzer(tremor_output['tremor_features'])
                biomarkers['pd_tremor_features'] = pd_output['pd_features']
                biomarkers['updrs_tremor'] = pd_output['updrs_scores']
        # Cardinal motor features
        brady_scores = self.bradykinesia_detector(encoded_flat)
        brady_scores = torch.sigmoid(brady_scores)
        biomarkers['bradykinesia'] = brady_scores.mean(dim=-1)
        biomarkers['bradykinesia_components'] = brady_scores
        rigidity_scores = self.rigidity_analyzer(encoded_flat)
        rigidity_scores = torch.sigmoid(rigidity_scores)
        biomarkers['rigidity'] = rigidity_scores[:, 0]  # Overall tone
        biomarkers['cogwheel'] = rigidity_scores[:, 1]
        postural_scores = self.postural_analyzer(encoded_flat)
        postural_scores = torch.sigmoid(postural_scores)
        biomarkers['postural_instability'] = postural_scores.mean(dim=-1)
        biomarkers['pull_test'] = postural_scores[:, 0] * 4  # Scale to 0-4
        return biomarkers

    def forward(self, x: torch.Tensor,
                return_biomarkers: bool = True,
                return_uncertainty: bool = True,
                return_clinical: bool = True) -> Dict[str, torch.Tensor]:
        """Complete forward pass"""
        # Extract all biomarkers
        biomarkers = self.extract_biomarkers(x)
        # Prepare features for classification
        encoded_flat = self.extract_features(x)
        classification_features = [encoded_flat]
        # Add gait, tremor, and cardinal features
        if self.use_gait and 'gait_pattern' in biomarkers:
            gait_output = self.gait_analyzer(x)
            classification_features.append(gait_output['gait_features'])
        if self.use_tremor and 'tremor_present' in biomarkers:
            tremor_output = self.tremor_detector(x)
            classification_features.append(tremor_output['tremor_features'])
        brady_scores = biomarkers.get('bradykinesia_components', torch.zeros(x.shape[0], 6).to(x.device))
        rigidity_scores = torch.stack([
            biomarkers.get('rigidity', torch.zeros(x.shape[0]).to(x.device)),
            biomarkers.get('cogwheel', torch.zeros(x.shape[0]).to(x.device))
        ], dim=1)
        postural_scores = self.postural_analyzer(encoded_flat) # Recalculate for consistency
        postural_scores = torch.sigmoid(postural_scores)
        classification_features.extend([brady_scores, rigidity_scores, postural_scores])
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
                scores = torch.sigmoid(scores) * 4  # Scale to 0-4
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
        """Generate clinical interpretation of movement biomarkers"""
        interpretation = {}
        # Tremor interpretation
        if 'tremor_present' in biomarkers and biomarkers['tremor_present'].item() > 0.5:
            tremor_types = ['Rest', 'Action', 'Postural', 'Kinetic', 'Intention', 'Mixed']
            dominant_type = tremor_types[biomarkers['tremor_type'].argmax().item()]
            interpretation['tremor'] = {
                'detected': True,
                'type': dominant_type,
                'severity': f"{biomarkers['tremor_severity'].item():.1f}/4",
                'frequency': f"{biomarkers['tremor_frequency'].item():.1f} Hz",
                'clinical_note': f"{dominant_type} tremor detected with moderate severity."
            }
        # Gait interpretation
        if 'fog_episodes' in biomarkers and biomarkers['fog_episodes'].item() > 0:
            interpretation['freezing_of_gait'] = {
                'detected': True,
                'episodes': int(biomarkers['fog_episodes'].item()),
                'duration': f"{biomarkers['fog_duration'].item():.1f}s",
                'clinical_note': 'Freezing of gait detected, suggesting advanced parkinsonism.'
            }
        # Bradykinesia interpretation
        if 'bradykinesia' in biomarkers and biomarkers['bradykinesia'].item() > 0.6:
            interpretation['bradykinesia'] = {
                'detected': True,
                'severity': 'moderate' if biomarkers['bradykinesia'].item() > 0.7 else 'mild',
                'clinical_note': 'Bradykinesia present, a core feature of parkinsonism.'
            }
        # Balance interpretation
        if 'balance_score' in biomarkers and biomarkers['balance_score'].item() < 0.5:
            interpretation['balance'] = {
                'impaired': True,
                'score': f"{biomarkers['balance_score'].item():.2f}",
                'fall_risk': 'high' if biomarkers['balance_score'].item() < 0.3 else 'moderate',
                'clinical_note': 'Balance impairment detected, fall risk assessment recommended.'
            }
        # Overall movement disorder assessment
        cardinal_features = 0
        if 'tremor_present' in biomarkers and biomarkers['tremor_present'].item() > 0.5:
            cardinal_features += 1
        if 'bradykinesia' in biomarkers and biomarkers['bradykinesia'].item() > 0.6:
            cardinal_features += 1
        if 'rigidity' in biomarkers and biomarkers['rigidity'].item() > 0.5:
            cardinal_features += 1
        if 'postural_instability' in biomarkers and biomarkers['postural_instability'].item() > 0.5:
            cardinal_features += 1
        if cardinal_features >= 2:
            interpretation['parkinsonian_syndrome'] = {
                'suspected': True,
                'cardinal_features': cardinal_features,
                'clinical_note': 'Multiple cardinal features of parkinsonism present.'
            }
        return interpretation