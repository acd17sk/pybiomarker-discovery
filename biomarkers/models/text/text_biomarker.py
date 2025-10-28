"""Complete text biomarker model integrating all linguistic components"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import numpy as np  
from biomarkers.core.base import BiomarkerModel, BiomarkerConfig


class TextBiomarkerModel(BiomarkerModel):
    """
    Complete text biomarker extraction model.
    
    This model integrates linguistic analysis with deep learning components
    to predict disease states, cognitive scores, and clinical metrics.
    """
    
    # Normative data (placeholders, can be loaded from config or subclassed)
    NORMATIVE_MEANS = {
        'lexical_diversity': 0.65,
        'syntactic_complexity': 0.60,
        'semantic_coherence': 0.75,
        'idea_density': 0.55,
        'information_content': 0.70
    }
    
    NORMATIVE_STDS = {
        'lexical_diversity': 0.12,
        'syntactic_complexity': 0.15,
        'semantic_coherence': 0.10,
        'idea_density': 0.12,
        'information_content': 0.13
    }
    
    def __init__(self, config: Union[Dict[str, Any], BiomarkerConfig]):
        """
        Initializes the TextBiomarkerModel.
        
        Args:
            config: A dictionary or BiomarkerConfig object. Must include:
                - modality: 'text'
                - model_type: 'TextBiomarkerModel'
                - hidden_dim: (int)
                - num_diseases: (int)
                And in metadata (or as top-level keys in dict):
                - embedding_dim: (int)
                - biomarker_feature_dim: (int)
        """
        super().__init__(config)
    
    def _build_model(self):
        """Build complete text biomarker model"""
        
        # Import here to avoid circular dependency
        from biomarkers.models.text.linguistic_analyzer import LinguisticAnalyzer

        # Get params from self.config
        embedding_dim = self.config.metadata.get('embedding_dim', 768)
        hidden_dim = self.config.hidden_dim
        dropout = self.config.dropout
        use_pretrained = self.config.metadata.get('use_pretrained', False)
        num_diseases = self.config.num_diseases
        
        # Text encoder (can be replaced with transformer embeddings)
        if use_pretrained:
            # Placeholder for actual pretrained model loading
            self.text_encoder = self._build_simple_encoder()
        else:
            self.text_encoder = self._build_simple_encoder()
        
        # Linguistic analyzer
        self.linguistic_analyzer = LinguisticAnalyzer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Cognitive assessment predictors
        self._build_cognitive_predictors()
        
        # Disease classifier
        self._build_disease_classifier()
        
        # Clinical scale predictors
        self._build_clinical_predictors()
        
        # Longitudinal change detector
        self.change_detector = nn.Sequential(
            nn.Linear(embedding_dim, 128), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # Uncertainty quantification
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_diseases)
        )
    
    def _build_simple_encoder(self):
        """Build simple text encoder (fallback if not using pretrained)"""
        # --- CORRECTED: Access embedding_dim from config metadata ---
        embedding_dim = self.config.metadata.get('embedding_dim', 768)
        dropout = self.config.dropout
        # ------------------------------------------------------------

        return nn.Sequential(
            nn.Linear(embedding_dim, 512), 
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim)
        )
    
    def _build_cognitive_predictors(self):
        """Build cognitive assessment predictors"""
        hidden_dim = self.config.hidden_dim
        biomarker_feature_dim = self.config.metadata.get('biomarker_feature_dim', 50)
        
        self.cognitive_predictors = nn.ModuleDict()
        
        predictor_input_dim = hidden_dim * 2 # Based on self.classifier_proj output
        
        # MMSE predictor (0-30 scale)
        self.cognitive_predictors['MMSE'] = nn.Sequential(
            nn.Linear(predictor_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # MoCA predictor (0-30 scale)
        self.cognitive_predictors['MoCA'] = nn.Sequential(
            nn.Linear(predictor_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # CDR predictor (Clinical Dementia Rating: 0, 0.5, 1, 2, 3)
        self.cognitive_predictors['CDR'] = nn.Sequential(
            nn.Linear(predictor_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
    
    def _build_disease_classifier(self):
        """Build disease classification head"""
        hidden_dim = self.config.hidden_dim
        biomarker_feature_dim = self.config.metadata.get('biomarker_feature_dim', 50)
        dropout = self.config.dropout
        num_diseases = self.config.num_diseases
        
        # feature_dim from linguistic_features (hidden_dim // 2) + biomarker_features
        feature_dim = hidden_dim // 2 + biomarker_feature_dim
        
        self.classifier_proj = nn.Linear(feature_dim, hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_diseases)
        )
    
    def _build_clinical_predictors(self):
        """Build clinical scale predictors"""
        hidden_dim = self.config.hidden_dim
        
        self.clinical_predictors = nn.ModuleDict()
        
        predictor_input_dim = hidden_dim * 2 # Based on self.classifier_proj output
        
        self.clinical_predictors['language_severity'] = nn.Sequential(
            nn.Linear(predictor_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.clinical_predictors['decline_rate'] = nn.Sequential(
            nn.Linear(predictor_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.clinical_predictors['communication_effectiveness'] = nn.Sequential(
            nn.Linear(predictor_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract base text features from embeddings (as required by BiomarkerModel).
        
        Args:
            x: Embeddings tensor [batch, seq_len, embedding_dim]
            
        Returns:
            Mean-pooled features [batch, embedding_dim]
        """
        features = torch.mean(x.to(self.device), dim=1)
        return features
    
    def extract_biomarkers(self,
                          x: torch.Tensor,
                          text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract all text biomarkers.
        
        Args:
            x: Embeddings tensor [batch, seq_len, embedding_dim]
            text_metadata: Optional dictionary with parsed text features.
        """
        embeddings = x.to(self.device)
        # --- CORRECTED: Use self.device consistently ---
        device = self.device 
        # -----------------------------------------------
        batch_size = embeddings.shape[0]
        
        biomarkers = {}
        
        # Ensure embeddings are on the correct device before passing to sub-module
        linguistic_output = self.linguistic_analyzer(embeddings.to(device), text_metadata)
        
        # Create default tensors on the correct device
        default_scalar = torch.zeros(batch_size, device=device)
        default_vec_8 = torch.zeros(batch_size, 8, device=device)
        default_vec_7 = torch.zeros(batch_size, 7, device=device)
        default_vec_6 = torch.zeros(batch_size, 6, device=device)

        # Lexical biomarkers
        lexical = linguistic_output['lexical_metrics']
        biomarkers.update({
            'lexical_diversity': lexical.get('ttr', default_scalar.clone()),
            'vocabulary_richness': lexical.get('richness_metrics', default_vec_8.clone())[:, :3].mean(dim=-1),
            'semantic_diversity': lexical.get('semantic_diversity', default_scalar.clone()),
            'lexical_density': lexical.get('lexical_density', default_scalar.clone())
        })
        
        # Syntactic biomarkers
        syntactic = linguistic_output['syntactic_metrics']
        syntactic_complexity = torch.sigmoid(syntactic.get('parse_metrics', default_vec_7.clone())[:, 6])
        dependency_distance = torch.sigmoid(syntactic.get('dependency_metrics', default_vec_6.clone())[:, 0])
        
        biomarkers.update({
            'syntactic_complexity': syntactic_complexity,
            'subordination_index': syntactic.get('subordination_index', default_scalar.clone()),
            'dependency_distance': dependency_distance,
            'grammar_accuracy': syntactic.get('grammar_accuracy', default_scalar.clone())
        })
        
        # Semantic biomarkers
        semantic = linguistic_output['semantic_metrics']
        biomarkers.update({
            'semantic_coherence': semantic.get('topic_consistency', default_scalar.clone()),
            'topic_consistency': semantic.get('topic_consistency', default_scalar.clone()),
            'inter_sentence_similarity': semantic.get('inter_sentence_similarity', default_scalar.clone()),
            'global_coherence': semantic.get('global_coherence', default_scalar.clone())
        })
        
        # Discourse biomarkers
        discourse = linguistic_output['discourse_metrics']
        biomarkers.update({
            'reference_quality': discourse.get('reference_quality', default_scalar.clone()),
            'cohesion_density': discourse.get('cohesion_density', default_scalar.clone()),
            'narrative_completeness': discourse.get('narrative_completeness', default_scalar.clone()),
            'information_flow': discourse.get('information_flow', default_scalar.clone())
        })
        
        # Cognitive load biomarkers
        cognitive = linguistic_output['cognitive_load_metrics']
        biomarkers.update({
            'cognitive_effort': cognitive.get('cognitive_effort', default_scalar.clone()),
            'word_finding_difficulty': cognitive.get('word_finding_difficulty', default_scalar.clone()),
            'repetition_score': cognitive.get('repetition_score', default_scalar.clone()),
            'pause_ratio': cognitive.get('pause_ratio', default_scalar.clone())
        })
        
        # Linguistic decline biomarkers
        decline = linguistic_output['decline_markers']
        biomarkers.update({
            'grammar_simplification': decline.get('grammar_simplification', default_scalar.clone()),
            'information_content': decline.get('information_content', default_scalar.clone()),
            'idea_density': decline.get('idea_density', default_scalar.clone()),
            'pronoun_overuse': decline.get('pronoun_overuse', default_scalar.clone()),
            'semantic_impoverishment': decline.get('semantic_impoverishment', default_scalar.clone()),
            'fragmentation': decline.get('fragmentation', default_scalar.clone())
        })
        
        # Temporal biomarkers
        temporal = linguistic_output['temporal_metrics']
        biomarkers.update({
            'writing_fluency': temporal.get('writing_fluency', default_scalar.clone()),
            'revision_rate': temporal.get('revision_rate', default_scalar.clone()),
            'pause_frequency': temporal.get('pause_frequency', default_scalar.clone())
        })
        
        return biomarkers
    
    def aggregate_biomarkers(self, 
                             biomarkers: Dict[str, torch.Tensor],
                             batch_size: int) -> torch.Tensor:
        """Aggregate biomarkers into feature vector"""
        
        biomarker_feature_dim = self.config.metadata.get('biomarker_feature_dim', 50)

        key_biomarkers = [
            'lexical_diversity', 'vocabulary_richness', 'semantic_diversity',
            'syntactic_complexity', 'subordination_index', 'grammar_accuracy',
            'semantic_coherence', 'topic_consistency', 'global_coherence',
            'reference_quality', 'cohesion_density', 'narrative_completeness',
            'cognitive_effort', 'word_finding_difficulty', 'repetition_score',
            'grammar_simplification', 'information_content', 'idea_density',
            'pronoun_overuse', 'semantic_impoverishment'
        ]
        
        biomarker_list = []
        # --- CORRECTED: Ensure all tensors added are on self.device ---
        target_device = self.device
        for key in key_biomarkers:
            if key in biomarkers:
                value = biomarkers[key].to(target_device) # Move tensor to target device
                if len(value.shape) == 1:
                    biomarker_list.append(value.unsqueeze(-1))
                elif len(value.shape) > 1:
                    biomarker_list.append(value[:, :1])
        # -------------------------------------------------------------
        
        if biomarker_list:
            aggregated = torch.cat(biomarker_list, dim=-1)
        else:
            aggregated = torch.zeros(batch_size, 20).to(target_device)
        
        # Pad or truncate to self.biomarker_feature_dim
        if aggregated.shape[-1] < biomarker_feature_dim:
            # --- CORRECTED: Ensure padding is on the same device as aggregated ---
            padding = torch.zeros(aggregated.shape[0], biomarker_feature_dim - aggregated.shape[-1], device=aggregated.device)
            # -------------------------------------------------------------------
            aggregated = torch.cat([aggregated, padding], dim=-1)
        elif aggregated.shape[-1] > biomarker_feature_dim:
            aggregated = aggregated[:, :biomarker_feature_dim]
        
        return aggregated
    
    def forward(self,
                x: torch.Tensor,
                text_metadata: Optional[Dict[str, Any]] = None,
                return_biomarkers: bool = True,
                return_uncertainty: bool = True,
                return_clinical: bool = True,
                return_cognitive: bool = True,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass (as required by BiomarkerModel).
        
        Args:
            x: Embeddings tensor [batch, seq_len, embedding_dim]
            text_metadata: Optional dictionary with parsed text features.
            return_biomarkers: (bool)
            return_uncertainty: (bool)
            return_clinical: (bool)
            return_cognitive: (bool)
            **kwargs: Catches extra args from base class calls.
            
        Returns:
            Dictionary of output tensors.
        """
        embeddings = x.to(self.device)
        # -----------------------------------------------------------
        batch_size = embeddings.shape[0]
        
        biomarkers = self.extract_biomarkers(embeddings, text_metadata)
        base_features = self.extract_features(embeddings) 
        
        # linguistic_analyzer now takes input on self.device from extract_biomarkers call
        linguistic_output = self.linguistic_analyzer(embeddings, text_metadata)
        linguistic_features = linguistic_output['linguistic_features'] # Should be on self.device
        
        biomarker_features = self.aggregate_biomarkers(biomarkers, batch_size) # Should be on self.device
        
        classification_features = torch.cat([
            linguistic_features,
            biomarker_features
        ], dim=-1)
        
        classifier_input = self.classifier_proj(classification_features)
        disease_logits = self.classifier(classifier_input)
        disease_probs = F.softmax(disease_logits, dim=-1)
        
        output = {
            'logits': disease_logits,
            'probabilities': disease_probs,
            'features': base_features, 
            'predictions': disease_logits.argmax(dim=1),
            'linguistic_pattern': linguistic_output['pattern_probs']
        }
        
        if return_biomarkers:
            output['biomarkers'] = biomarkers
        
        if return_cognitive and self.cognitive_predictors:
            cognitive_scores = {}
            for scale_name, predictor in self.cognitive_predictors.items():
                scores = predictor(classifier_input)
                if scale_name in ['MMSE', 'MoCA']:
                    scores = torch.sigmoid(scores) * 30
                elif scale_name == 'CDR':
                    scores = F.softmax(scores, dim=-1)
                cognitive_scores[scale_name] = scores.squeeze(-1) if scores.shape[-1] == 1 else scores
            output['cognitive_scores'] = cognitive_scores
        
        if return_clinical and self.clinical_predictors:
            clinical_scores = {}
            for scale_name, predictor in self.clinical_predictors.items():
                scores = predictor(classifier_input)
                if scale_name == 'language_severity':
                    scores = torch.sigmoid(scores) * 5
                elif scale_name == 'communication_effectiveness':
                    scores = torch.sigmoid(scores) * 10
                else: # decline_rate
                    scores = torch.sigmoid(scores)
                clinical_scores[scale_name] = scores.squeeze(-1) if scores.shape[-1] == 1 else scores
            output['clinical_scores'] = clinical_scores
        
        change_logits = self.change_detector(base_features)
        output['change_prediction'] = F.softmax(change_logits, dim=-1)
        
        if return_uncertainty and self.config.use_uncertainty:
            log_variance = self.uncertainty_estimator(base_features)
            uncertainty = torch.exp(log_variance)
            output['uncertainty'] = uncertainty
            output['confidence'] = 1.0 / (1.0 + uncertainty)
        
        return output
    
    def get_clinical_interpretation(self, biomarkers: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate clinical interpretation of text biomarkers"""
        interpretation = {}
        
        # Helper function to get scalar value from tensor (handles batches by averaging)
        def get_value(tensor):
            if tensor.numel() == 0:
                return 0.0
            # Move tensor to CPU before calling .item()
            if tensor.numel() == 1:
                return tensor.cpu().item() 
            else:
                return tensor.mean().cpu().item()
        
        lex_div = get_value(biomarkers.get('lexical_diversity', torch.tensor(1.0)))
        if lex_div < 0.4:
            interpretation['lexical_impairment'] = {
                'detected': True,
                'severity': 'moderate' if lex_div < 0.3 else 'mild',
                'ttr': f"{lex_div:.2f}",
                'clinical_note': 'Reduced lexical diversity suggests word-finding difficulties or restricted vocabulary.'
            }
        
        syn_comp = get_value(biomarkers.get('syntactic_complexity', torch.tensor(1.0)))
        if syn_comp < 0.4:
            interpretation['syntactic_simplification'] = {
                'detected': True,
                'complexity_score': f"{syn_comp:.2f}",
                'clinical_note': 'Simplified syntactic structures may indicate cognitive decline or language impairment.'
            }
        
        sem_coh = get_value(biomarkers.get('semantic_coherence', torch.tensor(1.0)))
        if sem_coh < 0.5:
            interpretation['semantic_incoherence'] = {
                'detected': True,
                'coherence_score': f"{sem_coh:.2f}",
                'clinical_note': 'Reduced semantic coherence suggests difficulty maintaining topic or thought organization.'
            }
        
        idea_dens = get_value(biomarkers.get('idea_density', torch.tensor(1.0)))
        if idea_dens < 0.4:
            interpretation['low_idea_density'] = {
                'detected': True,
                'severity': 'high_risk' if idea_dens < 0.3 else 'moderate_risk',
                'idea_density': f"{idea_dens:.2f}",
                'clinical_note': 'Low idea density is a validated predictor of Alzheimer\'s disease risk.'
            }
        
        pronoun_over = get_value(biomarkers.get('pronoun_overuse', torch.tensor(0.0)))
        if pronoun_over > 0.6:
            interpretation['pronoun_overuse'] = {
                'detected': True,
                'pronoun_score': f"{pronoun_over:.2f}",
                'clinical_note': 'Excessive pronoun use may indicate word-retrieval difficulties or early cognitive decline.'
            }
        
        cog_effort = get_value(biomarkers.get('cognitive_effort', torch.tensor(0.0)))
        if cog_effort > 0.7:
            interpretation['high_cognitive_load'] = {
                'detected': True,
                'effort_score': f"{cog_effort:.2f}",
                'clinical_note': 'High cognitive effort in language production suggests compensatory mechanisms or decline.'
            }
        
        frag = get_value(biomarkers.get('fragmentation', torch.tensor(0.0)))
        if frag > 0.6:
            interpretation['discourse_fragmentation'] = {
                'detected': True,
                'fragmentation_score': f"{frag:.2f}",
                'clinical_note': 'Fragmented discourse may indicate executive dysfunction or attention deficits.'
            }
        
        wfd = get_value(biomarkers.get('word_finding_difficulty', torch.tensor(0.0)))
        if wfd > 0.6:
            interpretation['word_finding_difficulty'] = {
                'detected': True,
                'severity': 'moderate' if wfd > 0.7 else 'mild',
                'clinical_note': 'Word-finding difficulties present, consider anomia assessment.'
            }
        
        decline_markers = [
            get_value(biomarkers.get('grammar_simplification', torch.tensor(0.0))),
            get_value(biomarkers.get('semantic_impoverishment', torch.tensor(0.0))),
            1.0 - get_value(biomarkers.get('information_content', torch.tensor(1.0))),
            1.0 - get_value(biomarkers.get('idea_density', torch.tensor(1.0)))
        ]
        
        composite_decline = sum(decline_markers) / len(decline_markers)
        
        if composite_decline > 0.5:
            interpretation['linguistic_decline_syndrome'] = {
                'detected': True,
                'severity': 'severe' if composite_decline > 0.7 else 'moderate',
                'composite_score': f"{composite_decline:.2f}",
                'clinical_note': 'Multiple linguistic decline markers detected. Comprehensive cognitive evaluation recommended.'
            }
        
        return interpretation
    
    def predict_cognitive_trajectory(self,
                                    text_samples: List[torch.Tensor],
                                    time_points: List[float]) -> Dict[str, Any]:
        """Predict cognitive trajectory from longitudinal text samples"""
        if len(text_samples) < 2:
            return {'error': 'At least 2 time points required for trajectory analysis'}
        
        trajectories = []
        
        # Ensure model is in eval mode for consistent predictions
        self.eval()
        with torch.no_grad():
            for sample in text_samples:
                # Move sample to model's device - handled in forward now
                output = self.forward(sample, return_biomarkers=True, return_cognitive=True)
                trajectories.append({
                    # Move biomarkers to CPU for consistent processing below
                    'biomarkers': {k: v.cpu() for k, v in output['biomarkers'].items() if isinstance(v, torch.Tensor)},
                    'cognitive_scores': {k: v.cpu() for k, v in output.get('cognitive_scores', {}).items() if isinstance(v, torch.Tensor)},
                    'disease_probs': output['probabilities'].cpu()
                })
        
        key_metrics = ['lexical_diversity', 'idea_density', 'semantic_coherence', 'syntactic_complexity']
        trends = {}
        
        # Helper to get scalar value (already on CPU from above)
        def get_value(tensor):
            if tensor.numel() == 0:
                return 0.0
            if tensor.numel() == 1:
                return tensor.item()
            else:
                return tensor.mean().item()
        
        for metric in key_metrics:
            values = [get_value(t['biomarkers'].get(metric, torch.tensor(0.0))) for t in trajectories]
            
            if len(values) >= 2:
                # Simple linear regression (slope)
                x = np.array(time_points)
                y = np.array(values)
                # Add a small epsilon to denominator to avoid division by zero
                x_mean, y_mean = np.mean(x), np.mean(y)
                slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean)**2) + 1e-8)
                
                trends[metric] = {
                    'values': values,
                    'slope': slope,
                    'trend': 'declining' if slope < -0.01 else 'stable' if slope < 0.01 else 'improving'
                }
        
        decline_count = sum(1 for t in trends.values() if t['trend'] == 'declining')
        
        if decline_count >= 3:
            trajectory_class = 'rapid_decline'
        elif decline_count >= 2:
            trajectory_class = 'moderate_decline'
        elif decline_count >= 1:
            trajectory_class = 'mild_decline'
        else:
            trajectory_class = 'stable'
        
        return {
            'trajectory_class': trajectory_class,
            'metric_trends': trends,
            'decline_markers_count': decline_count,
            'recommendation': self._generate_trajectory_recommendation(trajectory_class, trends)
        }
    
    def _generate_trajectory_recommendation(self, trajectory_class: str, trends: Dict) -> str:
        """Generate clinical recommendation based on trajectory"""
        if trajectory_class == 'rapid_decline':
            return ('Rapid cognitive decline detected across multiple linguistic domains. '
                   'Urgent neurological evaluation and comprehensive cognitive assessment recommended. '
                   'Consider imaging studies and biomarker testing.')
        elif trajectory_class == 'moderate_decline':
            return ('Moderate decline in linguistic abilities observed. '
                   'Follow-up cognitive assessment within 3-6 months recommended. '
                   'Consider referral to memory clinic.')
        elif trajectory_class == 'mild_decline':
            return ('Mild decline in some linguistic measures. '
                   'Continue monitoring with follow-up assessment in 6-12 months. '
                   'Lifestyle interventions and cognitive engagement recommended.')
        else:
            return ('Linguistic abilities remain stable. '
                   'Continue routine monitoring as appropriate for age and risk factors.')
    
    def compare_to_normative_data(self,
                                 biomarkers: Dict[str, torch.Tensor],
                                 age: Optional[int] = None,
                                 education: Optional[int] = None) -> Dict[str, Any]:
        """Compare biomarkers to normative data (age and education adjusted)"""
        
        normative_means = self.NORMATIVE_MEANS
        normative_stds = self.NORMATIVE_STDS
        
        age_adjustment = 0.0
        if age and age > 60:
            age_adjustment = -(age - 60) * 0.01
        
        edu_adjustment = 0.0
        if education and education > 12:
            edu_adjustment = (education - 12) * 0.02
        
        comparisons = {}
        
        for metric, norm_mean in normative_means.items():
            if metric in biomarkers:
                value_tensor = biomarkers[metric]
                # Handle both scalar and batch tensors (by averaging batch)
                # --- CORRECTED: Move to CPU before item() ---
                if value_tensor.numel() == 0:
                    value = 0.0
                elif value_tensor.numel() == 1:
                    value = value_tensor.cpu().item()
                else:
                    value = value_tensor.mean().cpu().item()
                # ---------------------------------------------
                
                adjusted_mean = norm_mean + age_adjustment + edu_adjustment
                std = normative_stds[metric]
                
                # Add epsilon to std to avoid division by zero
                z_score = (value - adjusted_mean) / (std + 1e-8)
                
                if z_score < -2.0:
                    interpretation = 'severely impaired'
                elif z_score < -1.5:
                    interpretation = 'moderately impaired'
                elif z_score < -1.0:
                    interpretation = 'mildly impaired'
                elif z_score < 1.0:
                    interpretation = 'average'
                else:
                    interpretation = 'above average'
                
                comparisons[metric] = {
                    'value': value,
                    'normative_mean': adjusted_mean,
                    'z_score': z_score,
                    'interpretation': interpretation
                }
        
        return comparisons