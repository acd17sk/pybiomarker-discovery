"""
Voice Biomarker Model - Main BiomarkerModel wrapper

This module provides the main VoiceBiomarkerModel class that integrates
all voice analysis components into a unified SOTA architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from biomarkers.core.base import BiomarkerModel
from .acoustic_encoder import (
    MelSpectrogramEncoder,
    WaveformEncoder,
    ConformerEncoder
)
from .prosody_analyzer import ProsodyAnalyzer
from .voice_disease_analyzers import (
    ParkinsonSpeechAnalyzer,
    DepressionSpeechAnalyzer,
    CognitiveDeclineSpeechAnalyzer,
    DysarthriaAnalyzer
)


class VoiceBiomarkerModel(BiomarkerModel):
    """
    State-of-the-art Voice Biomarker Extraction Model.
    
    This model extracts comprehensive voice biomarkers for disease detection
    and clinical assessment. It supports multiple input modalities and provides
    disease-specific analysis, clinical scale predictions, and uncertainty
    quantification.
    
    Architecture:
        1. Acoustic Encoders (Mel-spectrogram/Waveform with SincNet)
        2. Conformer-based temporal modeling
        3. Prosody analysis (F0, rhythm, intensity, etc.)
            - With raw audio: Clinical-grade metrics (jitter, shimmer, HNR)
            - Without raw audio: Learned prosodic features
        4. Disease-specific analyzers (PD, Depression, Cognitive, Dysarthria)
        5. Cross-modal attention fusion
        6. Multi-task prediction heads
    
    Args:
        config: Configuration dictionary with the following keys:
            - input_type: 'mel', 'waveform', or 'both'
            - sample_rate: Audio sample rate (default: 16000)
            - n_mels: Number of mel bands (default: 80)
            - hidden_dim: Hidden dimension size (default: 256)
            - num_diseases: Number of disease classes (default: 5)
            - dropout: Dropout rate (default: 0.3)
            - use_prosody: Enable prosody analysis (default: True)
            - use_conformer: Enable Conformer encoder (default: True)
            - clinical_scales: List of clinical scales to predict
            - prosody_use_raw_audio: Enable raw audio prosody (default: True)
    
    Example:
        >>> config = {
        ...     'input_type': 'mel',
        ...     'n_mels': 80,
        ...     'hidden_dim': 256,
        ...     'num_diseases': 5,
        ...     'use_prosody': True,
        ...     'clinical_scales': ['UPDRS_Speech', 'Dysarthria_Scale']
        ... }
        >>> model = VoiceBiomarkerModel(config)
        >>> mel_input = torch.randn(4, 1, 80, 200)
        >>> raw_audio = torch.randn(4, 16000)  # Optional for enhanced prosody
        >>> output = model(mel_input, raw_waveform=raw_audio, return_biomarkers=True)
        >>> print(output['biomarkers'].keys())
    
    References:
        - Gulati et al. (2020) "Conformer: Convolution-augmented Transformer..."
        - Ravanelli & Bengio (2018) "Speaker Recognition from Raw Waveform..."
        - Rusz et al. (2011) "Quantitative acoustic measurements..."
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract and validate configuration
        self.input_type = config.get('input_type', 'mel')
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mels = config.get('n_mels', 80)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_diseases = config.get('num_diseases', 5)
        self.dropout = config.get('dropout', 0.3)
        self.use_prosody = config.get('use_prosody', True)
        self.use_conformer = config.get('use_conformer', True)
        self.clinical_scales = config.get('clinical_scales', 
                                         ['UPDRS_Speech', 'Dysarthria_Scale'])
        # OPTIONAL CHANGE 1: Add explicit config flag for raw audio usage
        self.prosody_use_raw_audio = config.get('prosody_use_raw_audio', True)
        
        # Validate configuration
        assert self.input_type in ['mel', 'waveform', 'both'], \
            f"input_type must be 'mel', 'waveform', or 'both', got {self.input_type}"
        
        self._build_model()
    
    def _build_model(self):
        """Build complete SOTA voice biomarker model architecture."""
        
        # === 1. ACOUSTIC ENCODERS ===
        if self.input_type in ['mel', 'both']:
            self.mel_encoder = MelSpectrogramEncoder(
                n_mels=self.n_mels,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                dropout=self.dropout,
                use_delta=True
            )
        
        if self.input_type in ['waveform', 'both']:
            self.waveform_encoder = WaveformEncoder(
                sample_rate=self.sample_rate,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                dropout=self.dropout,
                use_sincnet=True
            )
        
        # === 2. CONFORMER ENCODER (SOTA Temporal Modeling) ===
        if self.use_conformer:
            conformer_input_dim = (self.hidden_dim * 2 
                                  if self.input_type == 'both' 
                                  else self.hidden_dim)
            self.conformer_encoder = ConformerEncoder(
                input_dim=conformer_input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_layers=6,
                num_heads=8,
                dropout=self.dropout
            )
        
        # === 3. PROSODY ANALYZER (with optional raw audio processing) ===
        if self.use_prosody:
            self.prosody_analyzer = ProsodyAnalyzer(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                num_prosodic_features=32,
                dropout=self.dropout,
                # OPTIONAL CHANGE 2: Pass raw audio flag to ProsodyAnalyzer
                use_raw_audio_when_available=self.prosody_use_raw_audio
            )
        
        # === 4. DISEASE-SPECIFIC ANALYZERS ===
        self.pd_speech_analyzer = ParkinsonSpeechAnalyzer(
            input_dim=self.hidden_dim,
            dropout=self.dropout
        )
        self.depression_speech_analyzer = DepressionSpeechAnalyzer(
            input_dim=self.hidden_dim,
            dropout=self.dropout
        )
        self.cognitive_speech_analyzer = CognitiveDeclineSpeechAnalyzer(
            input_dim=self.hidden_dim,
            dropout=self.dropout
        )
        self.dysarthria_analyzer = DysarthriaAnalyzer(
            input_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        # === 5. CROSS-MODAL ATTENTION ===
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # === 6. FEATURE FUSION ===
        feature_dim = self.hidden_dim  # Base acoustic features
        if self.use_prosody:
            feature_dim += 32  # Prosodic features
        feature_dim += 128 * 4  # 4 disease-specific analyzers
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # === 7. DISEASE CLASSIFIER ===
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_diseases)
        )
        
        # === 8. CLINICAL SCALE PREDICTORS ===
        self._build_clinical_predictors()
        
        # === 9. UNCERTAINTY QUANTIFICATION ===
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_diseases)
        )
    
    def _build_clinical_predictors(self):
        """Build clinical scale prediction heads."""
        self.clinical_predictors = nn.ModuleDict()
        
        if 'UPDRS_Speech' in self.clinical_scales:
            self.clinical_predictors['UPDRS_Speech'] = nn.Sequential(
                nn.Linear(self.hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout * 0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        if 'Dysarthria_Scale' in self.clinical_scales:
            self.clinical_predictors['Dysarthria_Scale'] = nn.Sequential(
                nn.Linear(self.hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout * 0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
        
        if 'Voice_Handicap_Index' in self.clinical_scales:
            self.clinical_predictors['Voice_Handicap_Index'] = nn.Sequential(
                nn.Linear(self.hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout * 0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
    
    def extract_features(self, x: torch.Tensor, 
                         input_type: Optional[str] = None) -> torch.Tensor:
        """
        Extract acoustic features from audio input.
        
        Args:
            x: Input audio
                - If mel: [batch, 1, n_mels, time]
                - If waveform: [batch, 1, samples]
            input_type: Override default input type
            
        Returns:
            Acoustic features [batch, hidden_dim]
        """
        input_type = input_type or self.input_type
        acoustic_features = []
        
        # Extract from mel-spectrogram
        if input_type in ['mel', 'both']:
            # The unsqueeze logic should handle 3D input (e.g., [B, n_mels, time]) 
            # or 4D input [B, 1, n_mels, time]
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            mel_features, _ = self.mel_encoder(x)
            acoustic_features.append(mel_features)
        
        # Extract from waveform
        if input_type in ['waveform', 'both']:
            # Handle waveform input of shape [B, samples] or [B, 1, samples]
            # Assumes SincNet is set up to handle 1D raw waveform input [B, samples]
            waveform = x.squeeze(1).squeeze(1) if len(x.shape) > 2 else x
            wave_features, _ = self.waveform_encoder(waveform)
            acoustic_features.append(wave_features)
        
        # Concatenate multi-modal features
        features = (torch.cat(acoustic_features, dim=-1) 
                    if len(acoustic_features) > 1 
                    else acoustic_features[0])
        
        return features
    
    def extract_biomarkers(self, x: torch.Tensor,
                           raw_waveform: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive voice biomarkers.
        
        Args:
            x: Input audio (mel-spectrogram or waveform)
            raw_waveform: Optional raw waveform for enhanced prosody analysis.
            
        Returns:
            Dictionary containing 40+ specific biomarkers.
        """
        biomarkers = {}
        
        # 1. Extract base acoustic features
        acoustic_features = self.extract_features(x)
        
        # 2. Apply Conformer for temporal modeling
        if self.use_conformer:
            conformer_input = acoustic_features.unsqueeze(1)
            conformer_features, _ = self.conformer_encoder(conformer_input)
            acoustic_features = conformer_features
        
        # 3. Extract prosodic features (with optional raw audio enhancement)
        if self.use_prosody:
            # Passes acoustic features and raw waveform (if provided)
            prosody_output = self.prosody_analyzer(acoustic_features, raw_waveform)
            biomarkers.update({
                'prosody_features': prosody_output['prosodic_features'],
                'f0_statistics': prosody_output['f0_features'],
                'rhythm_features': prosody_output['rhythm_features'],
                'intensity_features': prosody_output['intensity_features'],
                'spectral_tilt': prosody_output['spectral_tilt'],
                'articulation_features': prosody_output['articulation_features']
            })
            
            # REQUIRED CHANGE 1: Extract raw audio metrics if available
            if 'raw_audio_metrics' in prosody_output:
                raw_metrics = prosody_output['raw_audio_metrics']
                biomarkers.update({
                    'f0_mean_hz': raw_metrics['f0_mean'],
                    'f0_std_hz': raw_metrics['f0_std'],
                    'jitter': raw_metrics['jitter'],
                    'shimmer': raw_metrics['shimmer'],
                    'hnr': raw_metrics['hnr'],
                    'voicing_rate': raw_metrics['voicing_rate'],
                    'energy_mean': raw_metrics['energy_mean'],
                    'energy_std': raw_metrics['energy_std'],
                    'zcr_mean': raw_metrics['zcr_mean'],
                    'zcr_std': raw_metrics['zcr_std'],
                    'used_raw_audio': prosody_output['used_raw_audio']
                })
            
            # Apply cross-modal attention
            acoustic_attended, _ = self.cross_modal_attention(
                acoustic_features.unsqueeze(1),
                prosody_output['prosodic_features'].unsqueeze(1),
                prosody_output['prosodic_features'].unsqueeze(1)
            )
            acoustic_features = acoustic_attended.squeeze(1)
        
        # 4. Parkinson's disease speech features
        pd_output = self.pd_speech_analyzer(acoustic_features)
        biomarkers.update({
            'pd_hypophonia': pd_output['hypophonia'],
            'pd_monopitch': pd_output['monopitch'],
            'pd_monoloudness': pd_output['monoloudness'],
            'pd_speech_rate_reduction': pd_output['speech_rate_reduction'],
            'pd_imprecise_articulation': pd_output['imprecise_articulation'],
            'pd_voice_quality': pd_output['voice_quality_score']
        })
        
        # 5. Depression speech features
        depression_output = self.depression_speech_analyzer(acoustic_features)
        biomarkers.update({
            'depression_flat_affect': depression_output['flat_affect'],
            'depression_reduced_pitch_variability': depression_output['reduced_pitch_variability'],
            'depression_slow_speech': depression_output['slow_speech'],
            'depression_long_pauses': depression_output['long_pauses'],
            'depression_low_energy': depression_output['low_energy']
        })
        
        # 6. Cognitive decline speech features
        cognitive_output = self.cognitive_speech_analyzer(acoustic_features)
        biomarkers.update({
            'cognitive_hesitations': cognitive_output['hesitations'],
            'cognitive_word_finding_difficulty': cognitive_output['word_finding'],
            'cognitive_semantic_pauses': cognitive_output['semantic_pauses'],
            'cognitive_reduced_complexity': cognitive_output['reduced_complexity'],
            'cognitive_repetitions': cognitive_output['repetitions']
        })
        
        # 7. Dysarthria analysis
        dysarthria_output = self.dysarthria_analyzer(acoustic_features)
        biomarkers.update({
            'dysarthria_type': dysarthria_output['dysarthria_type'],
            'dysarthria_severity': dysarthria_output['severity'],
            'intelligibility': dysarthria_output['intelligibility'],
            'articulation_precision': dysarthria_output['articulation_precision']
        })
        
        return biomarkers
    
    def forward(self, x: torch.Tensor,
                raw_waveform: Optional[torch.Tensor] = None,
                return_biomarkers: bool = True,
                return_uncertainty: bool = True,
                return_clinical: bool = True) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with disease prediction and biomarker extraction.
        
        Args:
            x: Input audio (mel-spectrogram or waveform based on config)
            raw_waveform: Optional raw waveform for enhanced prosody analysis
            return_biomarkers: Whether to return extracted biomarkers
            return_uncertainty: Whether to return uncertainty estimates
            return_clinical: Whether to return clinical scale predictions
            
        Returns:
            Dictionary containing:
                - logits: Disease classification logits [batch, num_diseases]
                - probabilities: Disease probabilities [batch, num_diseases]
                - predictions: Predicted disease class [batch]
                - features: Extracted acoustic features [batch, hidden_dim]
                - biomarkers: Dict of 40+ biomarkers (if return_biomarkers=True)
                - clinical_scores: Clinical scale scores (if return_clinical=True)
                - uncertainty: Uncertainty estimates (if return_uncertainty=True)
                - confidence: Confidence scores (if return_uncertainty=True)
        """
        # Extract all biomarkers
        biomarkers = self.extract_biomarkers(x, raw_waveform)
        
        # The following block in the original code is redundant as feature 
        # extraction is already done in extract_biomarkers.
        # We need to re-extract the base acoustic features for the classifier 
        # input, ensuring it uses the *unattended* features before cross-modal.
        # However, for consistency with the *original* forward implementation:
        
        # 1. Re-extract base acoustic features (for fusion input)
        acoustic_features = self.extract_features(x)
        
        # 2. Apply Conformer (if used)
        if self.use_conformer:
            conformer_input = acoustic_features.unsqueeze(1)
            conformer_features, _ = self.conformer_encoder(conformer_input)
            acoustic_features = conformer_features
            
            # NOTE: The feature used for uncertainty is currently the 
            # acoustic_features (post-conformer, pre-attention). This is preserved.
        
        # Gather all features for fusion
        classification_features = [acoustic_features]
        
        # Add prosodic features (this should be the *learned* feature, not the raw metrics)
        if self.use_prosody and 'prosody_features' in biomarkers:
            # The 'prosody_features' here are the *learned* features from ProsodyAnalyzer
            classification_features.append(biomarkers['prosody_features'])
        
        # Add disease-specific features
        # Note: These analyzers re-run on acoustic_features (post-conformer/pre-attention), 
        # which is consistent with the extract_biomarkers structure.
        pd_features = self.pd_speech_analyzer(acoustic_features)['features']
        depression_features = self.depression_speech_analyzer(acoustic_features)['features']
        cognitive_features = self.cognitive_speech_analyzer(acoustic_features)['features']
        dysarthria_features = self.dysarthria_analyzer(acoustic_features)['features']
        
        classification_features.extend([
            pd_features,
            depression_features,
            cognitive_features,
            dysarthria_features
        ])
        
        # Concatenate and fuse all features
        all_features = torch.cat(classification_features, dim=1)
        fused_features = self.feature_fusion(all_features)
        
        # Disease classification
        disease_logits = self.classifier(fused_features)
        disease_probs = F.softmax(disease_logits, dim=-1)
        
        # Prepare output
        output = {
            'logits': disease_logits,
            'probabilities': disease_probs,
            'features': acoustic_features,
            'predictions': disease_logits.argmax(dim=1)
        }
        
        # Add biomarkers if requested
        if return_biomarkers:
            output['biomarkers'] = biomarkers
        
        # Add clinical scores if requested
        if return_clinical and self.clinical_predictors:
            clinical_scores = {}
            for scale_name, predictor in self.clinical_predictors.items():
                # Predictors run on the final FUSED features
                scores = predictor(fused_features)
                if scale_name == 'UPDRS_Speech':
                    scores = torch.sigmoid(scores) * 4  # Scale to 0-4
                elif scale_name == 'Dysarthria_Scale':
                    scores = torch.sigmoid(scores) * 5  # Scale to 0-5
                elif scale_name == 'Voice_Handicap_Index':
                    scores = torch.sigmoid(scores) * 120  # VHI range 0-120
                clinical_scores[scale_name] = scores
            output['clinical_scores'] = clinical_scores
        
        # Add uncertainty if requested
        if return_uncertainty:
            # Uncertainty predictor runs on the *acoustic_features* (pre-fusion)
            log_variance = self.uncertainty_estimator(acoustic_features)
            uncertainty = torch.exp(log_variance)
            output['uncertainty'] = uncertainty
            output['confidence'] = 1.0 / (1.0 + uncertainty)
        
        return output
    
    def get_clinical_interpretation(self, 
                                    biomarkers: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Generate clinical interpretation of voice biomarkers.
        
        This method translates raw biomarker values into human-readable
        clinical findings with severity assessments and recommendations.
        
        Args:
            biomarkers: Dictionary of extracted biomarkers
            
        Returns:
            Dictionary with clinical interpretations.
        """
        interpretation = {}
        
        # Helper function to extract values
        def get_value(x):
            return x.item() if torch.is_tensor(x) else float(x)
        
        # --- Parkinson's disease indicators ---
        pd_indicators = 0
        
        if biomarkers.get('pd_hypophonia', torch.tensor(0.0)).item() > 0.6:
            pd_indicators += 1
            severity_value = get_value(biomarkers['pd_hypophonia'])
            interpretation['hypophonia'] = {
                'detected': True,
                'severity': 'severe' if severity_value > 0.85 else 'moderate' if severity_value > 0.75 else 'mild',
                'score': f"{severity_value:.3f}",
                'clinical_note': 'Reduced voice volume (hypophonia), characteristic of Parkinson\'s disease'
            }
        
        if biomarkers.get('pd_monopitch', torch.tensor(0.0)).item() > 0.6:
            pd_indicators += 1
            interpretation['monopitch'] = {
                'detected': True,
                'score': f"{get_value(biomarkers['pd_monopitch']):.3f}",
                'clinical_note': 'Reduced pitch variability (monopitch), suggests parkinsonian speech pattern'
            }
        
        if biomarkers.get('pd_monoloudness', torch.tensor(0.0)).item() > 0.6:
            pd_indicators += 1
            interpretation['monoloudness'] = {
                'detected': True,
                'clinical_note': 'Reduced loudness variability, consistent with hypokinetic dysarthria'
            }
        
        if pd_indicators >= 2:
            interpretation['parkinsonian_speech'] = {
                'suspected': True,
                'indicators': pd_indicators,
                'clinical_note': 'Multiple parkinsonian speech features present. Consider neurological evaluation.',
                'recommendation': 'Refer to movement disorder specialist for UPDRS assessment'
            }
        
        # --- Depression indicators ---
        depression_indicators = 0
        if biomarkers.get('depression_flat_affect', torch.tensor(0.0)).item() > 0.6:
            depression_indicators += 1
        if biomarkers.get('depression_low_energy', torch.tensor(0.0)).item() > 0.6:
            depression_indicators += 1
        
        if depression_indicators >= 2:
            interpretation['depression_speech'] = {
                'detected': True,
                'indicators': depression_indicators,
                'clinical_note': 'Flat affect and low vocal energy consistent with depressive symptomatology',
                'recommendation': 'Consider psychiatric evaluation and depression screening'
            }
        
        # --- Cognitive decline indicators ---
        cognitive_indicators = 0
        if biomarkers.get('cognitive_hesitations', torch.tensor(0.0)).item() > 0.6:
            cognitive_indicators += 1
        if biomarkers.get('cognitive_word_finding_difficulty', torch.tensor(0.0)).item() > 0.6:
            cognitive_indicators += 1
        
        if cognitive_indicators >= 1:
            interpretation['cognitive_impairment'] = {
                'suspected': True,
                'features': ['hesitations', 'word-finding difficulty'],
                'clinical_note': 'Speech patterns suggest possible cognitive impairment',
                'recommendation': 'Consider cognitive screening (MMSE, MoCA) and neuropsychological evaluation'
            }
        
        # --- Dysarthria assessment ---
        if 'dysarthria_severity' in biomarkers:
            severity = get_value(biomarkers['dysarthria_severity'])
            if severity > 0.5:
                dysarthria_types = ['spastic', 'flaccid', 'ataxic', 'hypokinetic', 
                                    'hyperkinetic', 'mixed', 'unilateral_upper_motor']
                dominant_type_idx = biomarkers['dysarthria_type'].argmax().item()
                dominant_type = dysarthria_types[dominant_type_idx]
                
                intelligibility = get_value(biomarkers.get('intelligibility', torch.tensor(0.0)))
                
                interpretation['dysarthria'] = {
                    'detected': True,
                    'type': dominant_type.replace('_', ' ').title(),
                    'severity': f"{severity:.2f}",
                    'intelligibility': f"{intelligibility:.2%}",
                    'clinical_note': f'{dominant_type.replace("_", " ").title()} dysarthria detected',
                    'recommendation': 'Speech-language pathology evaluation recommended'
                }
        
        # REQUIRED CHANGE 2: Voice quality assessment from raw audio metrics
        if 'jitter' in biomarkers:
            jitter_val = get_value(biomarkers['jitter'])
            shimmer_val = get_value(biomarkers['shimmer'])
            hnr_val = get_value(biomarkers['hnr'])
            
            voice_quality_issues = []
            
            # Clinical thresholds from literature
            JITTER_THRESHOLD = 0.0104   # >1.04% abnormal
            SHIMMER_THRESHOLD = 0.0381  # >3.81% abnormal
            HNR_THRESHOLD = 0.13        # <0.13 abnormal (normalized)
            
            if jitter_val > JITTER_THRESHOLD:
                voice_quality_issues.append('elevated jitter')
            
            if shimmer_val > SHIMMER_THRESHOLD:
                voice_quality_issues.append('elevated shimmer')
            
            if hnr_val < HNR_THRESHOLD:
                voice_quality_issues.append('reduced harmonic-to-noise ratio')
            
            if voice_quality_issues:
                interpretation['voice_quality'] = {
                    'detected': True,
                    'issues': voice_quality_issues,
                    'jitter': f"{jitter_val:.4f} ({'abnormal' if jitter_val > JITTER_THRESHOLD else 'normal'})",
                    'shimmer': f"{shimmer_val:.4f} ({'abnormal' if shimmer_val > SHIMMER_THRESHOLD else 'normal'})",
                    'hnr': f"{hnr_val:.3f} ({'abnormal' if hnr_val < HNR_THRESHOLD else 'normal'})",
                    'clinical_note': (
                        f'Voice quality concerns detected: {", ".join(voice_quality_issues)}. '
                        f'These metrics suggest possible vocal pathology or neurological voice disorder.'
                    ),
                    'recommendation': 'Refer to otolaryngologist for laryngoscopic examination and voice quality assessment'
                }
            else:
                interpretation['voice_quality'] = {
                    'detected': False,
                    'jitter': f"{jitter_val:.4f} (normal)",
                    'shimmer': f"{shimmer_val:.4f} (normal)",
                    'hnr': f"{hnr_val:.3f} (normal)",
                    'clinical_note': 'Voice quality parameters within normal limits'
                }
            
            # Add F0 information if available
            if 'f0_mean_hz' in biomarkers:
                f0_mean = get_value(biomarkers['f0_mean_hz'])
                interpretation['voice_quality']['f0_mean'] = f"{f0_mean:.1f} Hz"
        
        # --- Overall assessment ---
        
        # Update overall assessment if voice quality issues found
        # (This block was added at the end of the previous section for organization)
        if 'voice_quality' in interpretation and interpretation['voice_quality']['detected']:
            if 'overall' not in interpretation:
                interpretation['overall'] = {
                    'status': 'abnormal_findings',
                    'findings_count': 1,
                    'clinical_note': 'Voice quality abnormalities detected',
                    'recommendation': 'Comprehensive clinical evaluation recommended'
                }
            else:
                interpretation['overall']['findings_count'] += 1
        
        if 'overall' not in interpretation:
            # Recalculate if we need to check non-voice_quality findings 
            # (only if voice_quality wasn't detected, or if voice_quality was missed)
            # The structure above adds voice_quality findings into 'overall' if detected. 
            # We now combine all findings to finalize the overall note.
            
            # Count findings (excluding overall and voice_quality if normal)
            findings_count = len([k for k, v in interpretation.items()  
                                  if k != 'voice_quality' and k != 'overall'])
            
            if 'voice_quality' in interpretation and interpretation['voice_quality']['detected']:
                findings_count += 1
            
            if findings_count == 0:
                interpretation['overall'] = {
                    'status': 'normal',
                    'clinical_note': 'No significant speech abnormalities detected',
                    'recommendation': 'Continue routine monitoring'
                }
            else:
                # This catches the case where overall was set by voice_quality, 
                # or if other findings exist but voice_quality was normal/missing
                interpretation['overall'] = {
                    'status': 'abnormal_findings',
                    'findings_count': findings_count,
                    'clinical_note': f'{findings_count} area(s) of concern identified',
                    'recommendation': 'Comprehensive clinical evaluation recommended'
                }

        
        return interpretation