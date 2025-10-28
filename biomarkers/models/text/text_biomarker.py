"""Complete text biomarker model integrating all linguistic components"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np  # FIXED: Added missing import


class TextBiomarkerModel(nn.Module):
    """
    Complete text biomarker extraction model.
    
    This is a standalone version that doesn't inherit from BiomarkerModel
    to avoid config compatibility issues.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Store configuration directly
        self.config = config
        
        # Extract configuration
        self.embedding_dim = config.get('embedding_dim', 768)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_diseases = config.get('num_diseases', 8)
        self.dropout = config.get('dropout', 0.3)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.use_pretrained = config.get('use_pretrained', False)
        self.pretrained_model = config.get('pretrained_model', 'bert-base-uncased')
        
        # Add modality for compatibility
        self.modality = config.get('modality', 'text')
        
        self._build_model()
    
    def _build_model(self):
        """Build complete text biomarker model"""
        
        # Import here to avoid circular dependency
        from biomarkers.models.text.linguistic_analyzer import LinguisticAnalyzer
        
        # Text encoder (can be replaced with transformer embeddings)
        if self.use_pretrained:
            # Placeholder for actual pretrained model loading
            self.text_encoder = self._build_simple_encoder()
        else:
            self.text_encoder = self._build_simple_encoder()
        
        # Linguistic analyzer
        self.linguistic_analyzer = LinguisticAnalyzer(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        # Cognitive assessment predictors
        self._build_cognitive_predictors()
        
        # Disease classifier
        self._build_disease_classifier()
        
        # Clinical scale predictors
        self._build_clinical_predictors()
        
        # Longitudinal change detector
        self.change_detector = nn.Sequential(
            nn.Linear(self.embedding_dim, 128), # Changed from self.hidden_dim
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # Uncertainty quantification
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.embedding_dim, 128), # Changed from self.hidden_dim
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_diseases)
        )
    
    def _build_simple_encoder(self):
        """Build simple text encoder (fallback if not using pretrained)"""
        # This is more of a projection layer, not an encoder.
        # A simple mean pooling is often used if embeddings are pre-computed.
        # If this model is meant to *process* embeddings, this is fine.
        return nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.embedding_dim)
        )
    
    def _build_cognitive_predictors(self):
        """Build cognitive assessment predictors"""
        self.cognitive_predictors = nn.ModuleDict()
        
        # Input dim for predictors is classifier_input dim
        classifier_feature_dim = self.hidden_dim // 2 + 50
        predictor_input_dim = self.hidden_dim * 2 # Based on self.classifier_proj output
        
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
        # feature_dim from linguistic_features (hidden_dim // 2) + biomarker_features (50)
        feature_dim = self.hidden_dim // 2 + 50
        
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
        
        predictor_input_dim = self.hidden_dim * 2 # Based on self.classifier_proj output
        
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
    
    @property
    def num_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extract_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Extract text features from embeddings"""
        # This is the "base_feature" used for change_detector and uncertainty
        # It should probably just be mean-pooled raw embeddings.
        features = torch.mean(embeddings, dim=1)
        return features
    
    def extract_biomarkers(self,
                          embeddings: torch.Tensor,
                          text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Extract all text biomarkers"""
        
        # FIXED: Get device and batch size from embeddings
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        biomarkers = {}
        
        linguistic_output = self.linguistic_analyzer(embeddings, text_metadata)
        
        # FIXED: Create default tensors on the correct device and with correct batch size
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
        # FIXED: Apply sigmoid to normalize raw linear outputs to [0, 1]
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
        for key in key_biomarkers:
            if key in biomarkers:
                value = biomarkers[key]
                # FIXED: Ensure all values have a batch dimension
                if len(value.shape) == 1: # (batch_size,)
                    biomarker_list.append(value.unsqueeze(-1)) # (batch_size, 1)
                elif len(value.shape) > 1: # (batch_size, K)
                    biomarker_list.append(value[:, :1]) # (batch_size, 1)
                # else: scalar tensor, which shouldn't happen if extract_biomarkers is correct

        
        if biomarker_list:
            aggregated = torch.cat(biomarker_list, dim=-1)
        else:
            # FIXED: Use batch_size for default tensor
            aggregated = torch.zeros(batch_size, 20).to(next(self.parameters()).device)
        
        # Pad or truncate to 50 features
        if aggregated.shape[-1] < 50:
            padding = torch.zeros(aggregated.shape[0], 50 - aggregated.shape[-1]).to(aggregated.device)
            aggregated = torch.cat([aggregated, padding], dim=-1)
        elif aggregated.shape[-1] > 50:
            aggregated = aggregated[:, :50]
        
        return aggregated
    
    def forward(self,
                embeddings: torch.Tensor,
                text_metadata: Optional[Dict[str, Any]] = None,
                return_biomarkers: bool = True,
                return_uncertainty: bool = True,
                return_clinical: bool = True,
                return_cognitive: bool = True) -> Dict[str, torch.Tensor]:
        """Complete forward pass"""
        
        # FIXED: Get batch_size
        batch_size = embeddings.shape[0]
        
        biomarkers = self.extract_biomarkers(embeddings, text_metadata)
        # This is mean-pooled raw embeddings
        base_features = self.extract_features(embeddings) 
        
        linguistic_output = self.linguistic_analyzer(embeddings, text_metadata)
        linguistic_features = linguistic_output['linguistic_features']
        
        # FIXED: Pass batch_size
        biomarker_features = self.aggregate_biomarkers(biomarkers, batch_size)
        
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
            'features': base_features, # This is the mean-pooled raw embedding
            'predictions': disease_logits.argmax(dim=1),
            'linguistic_pattern': linguistic_output['pattern_probs']
        }
        
        if return_biomarkers:
            output['biomarkers'] = biomarkers
        
        if return_cognitive and self.cognitive_predictors:
            cognitive_scores = {}
            for scale_name, predictor in self.cognitive_predictors.items():
                scores = predictor(classifier_input) # Use projected features
                if scale_name in ['MMSE', 'MoCA']:
                    # Scale to [0, 30]
                    scores = torch.sigmoid(scores) * 30
                elif scale_name == 'CDR':
                    scores = F.softmax(scores, dim=-1)
                cognitive_scores[scale_name] = scores.squeeze(-1) if scores.shape[-1] == 1 else scores
            output['cognitive_scores'] = cognitive_scores
        
        if return_clinical and self.clinical_predictors:
            clinical_scores = {}
            for scale_name, predictor in self.clinical_predictors.items():
                scores = predictor(classifier_input) # Use projected features
                if scale_name == 'language_severity':
                    scores = torch.sigmoid(scores) * 5
                elif scale_name == 'communication_effectiveness':
                    scores = torch.sigmoid(scores) * 10
                else: # decline_rate
                    scores = torch.sigmoid(scores)
                clinical_scores[scale_name] = scores.squeeze(-1) if scores.shape[-1] == 1 else scores
            output['clinical_scores'] = clinical_scores
        
        # Use base_features (mean-pooled raw embeddings) for these
        change_logits = self.change_detector(base_features)
        output['change_prediction'] = F.softmax(change_logits, dim=-1)
        
        if return_uncertainty:
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
            if tensor.numel() == 1:
                return tensor.item()
            else:
                return tensor.mean().item()
        
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
                # Move sample to model's device
                device = next(self.parameters()).device
                sample = sample.to(device)
                output = self.forward(sample, return_biomarkers=True, return_cognitive=True)
                trajectories.append({
                    'biomarkers': output['biomarkers'],
                    'cognitive_scores': output.get('cognitive_scores', {}),
                    'disease_probs': output['probabilities']
                })
        
        key_metrics = ['lexical_diversity', 'idea_density', 'semantic_coherence', 'syntactic_complexity']
        trends = {}
        
        # Helper to get scalar value
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
        # These are placeholders; in production, they'd be loaded from a config
        normative_means = {
            'lexical_diversity': 0.65,
            'syntactic_complexity': 0.60,
            'semantic_coherence': 0.75,
            'idea_density': 0.55,
            'information_content': 0.70
        }
        
        normative_stds = {
            'lexical_diversity': 0.12,
            'syntactic_complexity': 0.15,
            'semantic_coherence': 0.10,
            'idea_density': 0.12,
            'information_content': 0.13
        }
        
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
                if value_tensor.numel() == 0:
                    value = 0.0
                elif value_tensor.numel() == 1:
                    value = value_tensor.item()
                else:
                    value = value_tensor.mean().item()
                
                adjusted_mean = norm_mean + age_adjustment + edu_adjustment
                std = normative_stds[metric]
                
                # Add epsilon to std to avoid division by zero
                z_score = (value - adjusted_mean) / (std + 1e-8)
                
                # FIXED: Removed scipy dependency
                # percentile = scipy_stats.norm.cdf(z_score) * 100
                
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
                    # 'percentile': percentile, # Removed
                    'interpretation': interpretation
                }
        
        return comparisons