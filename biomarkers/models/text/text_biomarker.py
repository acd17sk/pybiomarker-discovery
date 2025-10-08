"""Complete text biomarker model integrating all linguistic components"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from biomarkers.core.base import BiomarkerModel
from .linguistic_analyzer import LinguisticAnalyzer


class TextBiomarkerModel(BiomarkerModel):
    """Complete text biomarker extraction model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract configuration
        self.embedding_dim = config.get('embedding_dim', 768)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_diseases = config.get('num_diseases', 8)  # Normal, MCI, AD, PD, Depression, Aphasia, Dysexecutive, TBI
        self.dropout = config.get('dropout', 0.3)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.use_pretrained = config.get('use_pretrained', True)
        self.pretrained_model = config.get('pretrained_model', 'bert-base-uncased')
        
        self._build_model()
    
    def _build_model(self):
        """Build complete text biomarker model"""
        
        # Text encoder (can be replaced with transformer embeddings)
        if self.use_pretrained:
            # Placeholder for pretrained embeddings (e.g., BERT, RoBERTa)
            # In practice, would use: from transformers import AutoModel
            # self.text_encoder = AutoModel.from_pretrained(self.pretrained_model)
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
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # stable, mild_decline, moderate_decline, rapid_decline, improvement
        )
        
        # Uncertainty quantification
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_diseases)
        )
    
    def _build_simple_encoder(self):
        """Build simple text encoder (fallback if not using pretrained)"""
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
        
        # MMSE predictor (0-30 scale)
        self.cognitive_predictors['MMSE'] = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # MoCA predictor (0-30 scale)
        self.cognitive_predictors['MoCA'] = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # CDR predictor (Clinical Dementia Rating: 0, 0.5, 1, 2, 3)
        self.cognitive_predictors['CDR'] = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 severity levels
        )
    
    def _build_disease_classifier(self):
        """Build disease classification head"""
        # Calculate feature dimension
        feature_dim = self.hidden_dim // 2  # From linguistic analyzer
        
        # Add biomarker dimensions
        feature_dim += 50  # Aggregate biomarker features
        
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
        
        # Language impairment severity (0-5 scale)
        self.clinical_predictors['language_severity'] = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Cognitive decline progression rate
        self.clinical_predictors['decline_rate'] = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Communication effectiveness (0-10 scale)
        self.clinical_predictors['communication_effectiveness'] = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def extract_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Extract text features from embeddings"""
        # Encode text
        encoded = self.text_encoder(embeddings)
        
        # Global pooling
        features = torch.mean(encoded, dim=1)
        
        return features
    
    def extract_biomarkers(self,
                          embeddings: torch.Tensor,
                          text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Extract all text biomarkers"""
        biomarkers = {}
        
        # Run linguistic analyzer
        linguistic_output = self.linguistic_analyzer(embeddings, text_metadata)
        
        # Lexical biomarkers
        lexical = linguistic_output['lexical_metrics']
        biomarkers.update({
            'lexical_diversity': lexical.get('ttr', torch.tensor(0.0)),
            'vocabulary_richness': lexical.get('richness_metrics', torch.zeros(1, 8))[:, :3].mean(dim=-1),
            'semantic_diversity': lexical.get('semantic_diversity', torch.tensor(0.0)),
            'lexical_density': lexical.get('lexical_density', torch.tensor(0.0))
        })
        
        # Syntactic biomarkers
        syntactic = linguistic_output['syntactic_metrics']
        biomarkers.update({
            'syntactic_complexity': syntactic.get('parse_metrics', torch.zeros(1, 7))[:, 6],
            'subordination_index': syntactic.get('subordination_index', torch.tensor(0.0)),
            'dependency_distance': syntactic.get('dependency_metrics', torch.zeros(1, 6))[:, 0],
            'grammar_accuracy': syntactic.get('grammar_accuracy', torch.tensor(0.0))
        })
        
        # Semantic biomarkers
        semantic = linguistic_output['semantic_metrics']
        biomarkers.update({
            'semantic_coherence': semantic.get('topic_consistency', torch.tensor(0.0)),
            'topic_consistency': semantic.get('topic_consistency', torch.tensor(0.0)),
            'inter_sentence_similarity': semantic.get('inter_sentence_similarity', torch.tensor(0.0)),
            'global_coherence': semantic.get('global_coherence', torch.tensor(0.0))
        })
        
        # Discourse biomarkers
        discourse = linguistic_output['discourse_metrics']
        biomarkers.update({
            'reference_quality': discourse.get('reference_quality', torch.tensor(0.0)),
            'cohesion_density': discourse.get('cohesion_density', torch.tensor(0.0)),
            'narrative_completeness': discourse.get('narrative_completeness', torch.tensor(0.0)),
            'information_flow': discourse.get('information_flow', torch.tensor(0.0))
        })
        
        # Cognitive load biomarkers
        cognitive = linguistic_output['cognitive_load_metrics']
        biomarkers.update({
            'cognitive_effort': cognitive.get('cognitive_effort', torch.tensor(0.0)),
            'word_finding_difficulty': cognitive.get('word_finding_difficulty', torch.tensor(0.0)),
            'repetition_score': cognitive.get('repetition_score', torch.tensor(0.0)),
            'pause_ratio': cognitive.get('pause_ratio', torch.tensor(0.0))
        })
        
        # Linguistic decline biomarkers
        decline = linguistic_output['decline_markers']
        biomarkers.update({
            'grammar_simplification': decline.get('grammar_simplification', torch.tensor(0.0)),
            'information_content': decline.get('information_content', torch.tensor(0.0)),
            'idea_density': decline.get('idea_density', torch.tensor(0.0)),
            'pronoun_overuse': decline.get('pronoun_overuse', torch.tensor(0.0)),
            'semantic_impoverishment': decline.get('semantic_impoverishment', torch.tensor(0.0)),
            'fragmentation': decline.get('fragmentation', torch.tensor(0.0))
        })
        
        # Temporal biomarkers
        temporal = linguistic_output['temporal_metrics']
        biomarkers.update({
            'writing_fluency': temporal.get('writing_fluency', torch.tensor(0.0)),
            'revision_rate': temporal.get('revision_rate', torch.tensor(0.0)),
            'pause_frequency': temporal.get('pause_frequency', torch.tensor(0.0))
        })
        
        return biomarkers
    
    def aggregate_biomarkers(self, biomarkers: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate biomarkers into feature vector"""
        # Select key biomarkers for classification
        key_biomarkers = [
            'lexical_diversity',
            'vocabulary_richness',
            'semantic_diversity',
            'syntactic_complexity',
            'subordination_index',
            'grammar_accuracy',
            'semantic_coherence',
            'topic_consistency',
            'global_coherence',
            'reference_quality',
            'cohesion_density',
            'narrative_completeness',
            'cognitive_effort',
            'word_finding_difficulty',
            'repetition_score',
            'grammar_simplification',
            'information_content',
            'idea_density',
            'pronoun_overuse',
            'semantic_impoverishment'
        ]
        
        # Stack biomarkers
        biomarker_list = []
        for key in key_biomarkers:
            if key in biomarkers:
                value = biomarkers[key]
                if len(value.shape) == 0:  # Scalar
                    value = value.unsqueeze(0)
                if len(value.shape) == 1:  # 1D
                    biomarker_list.append(value.unsqueeze(-1))
                else:
                    biomarker_list.append(value[:, :1])  # Take first dim if multi-dim
        
        # Pad to 50 features if needed
        aggregated = torch.cat(biomarker_list, dim=-1) if biomarker_list else torch.zeros(1, 20).to(next(self.parameters()).device)
        
        # Pad or truncate to exactly 50 features
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
        
        # Extract all biomarkers
        biomarkers = self.extract_biomarkers(embeddings, text_metadata)
        
        # Get base features
        base_features = self.extract_features(embeddings)
        
        # Get linguistic features
        linguistic_output = self.linguistic_analyzer(embeddings, text_metadata)
        linguistic_features = linguistic_output['linguistic_features']
        
        # Aggregate biomarkers
        biomarker_features = self.aggregate_biomarkers(biomarkers)
        
        # Concatenate features
        classification_features = torch.cat([
            linguistic_features,
            biomarker_features
        ], dim=-1)
        
        # Project to classifier dimension
        classifier_input = self.classifier_proj(classification_features)
        
        # Disease classification
        disease_logits = self.classifier(classifier_input)
        disease_probs = F.softmax(disease_logits, dim=-1)
        
        # Prepare output
        output = {
            'logits': disease_logits,
            'probabilities': disease_probs,
            'features': base_features,
            'predictions': disease_logits.argmax(dim=1),
            'linguistic_pattern': linguistic_output['pattern_probs']
        }
        
        # Add biomarkers if requested
        if return_biomarkers:
            output['biomarkers'] = biomarkers
        
        # Add cognitive scores if requested
        if return_cognitive and self.cognitive_predictors:
            cognitive_scores = {}
            for scale_name, predictor in self.cognitive_predictors.items():
                scores = predictor(classifier_input)
                
                if scale_name in ['MMSE', 'MoCA']:
                    # Scale to 0-30
                    scores = torch.sigmoid(scores) * 30
                elif scale_name == 'CDR':
                    # CDR levels: 0, 0.5, 1, 2, 3
                    scores = F.softmax(scores, dim=-1)
                
                cognitive_scores[scale_name] = scores
            
            output['cognitive_scores'] = cognitive_scores
        
        # Add clinical scores if requested
        if return_clinical and self.clinical_predictors:
            clinical_scores = {}
            for scale_name, predictor in self.clinical_predictors.items():
                scores = predictor(classifier_input)
                
                if scale_name == 'language_severity':
                    scores = torch.sigmoid(scores) * 5  # Scale to 0-5
                elif scale_name == 'communication_effectiveness':
                    scores = torch.sigmoid(scores) * 10  # Scale to 0-10
                else:
                    scores = torch.sigmoid(scores)
                
                clinical_scores[scale_name] = scores
            
            output['clinical_scores'] = clinical_scores
        
        # Add longitudinal change detection
        change_logits = self.change_detector(base_features)
        output['change_prediction'] = F.softmax(change_logits, dim=-1)
        
        # Add uncertainty if requested
        if return_uncertainty:
            log_variance = self.uncertainty_estimator(base_features)
            uncertainty = torch.exp(log_variance)
            output['uncertainty'] = uncertainty
            output['confidence'] = 1.0 / (1.0 + uncertainty)
        
        return output
    
    def get_clinical_interpretation(self, biomarkers: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate clinical interpretation of text biomarkers"""
        interpretation = {}
        
        # Lexical impairment
        if 'lexical_diversity' in biomarkers and biomarkers['lexical_diversity'].item() < 0.4:
            interpretation['lexical_impairment'] = {
                'detected': True,
                'severity': 'moderate' if biomarkers['lexical_diversity'].item() < 0.3 else 'mild',
                'ttr': f"{biomarkers['lexical_diversity'].item():.2f}",
                'clinical_note': 'Reduced lexical diversity suggests word-finding difficulties or restricted vocabulary.'
            }
        
        # Syntactic simplification
        if 'syntactic_complexity' in biomarkers and biomarkers['syntactic_complexity'].item() < 0.4:
            interpretation['syntactic_simplification'] = {
                'detected': True,
                'complexity_score': f"{biomarkers['syntactic_complexity'].item():.2f}",
                'clinical_note': 'Simplified syntactic structures may indicate cognitive decline or language impairment.'
            }
        
        # Semantic coherence issues
        if 'semantic_coherence' in biomarkers and biomarkers['semantic_coherence'].item() < 0.5:
            interpretation['semantic_incoherence'] = {
                'detected': True,
                'coherence_score': f"{biomarkers['semantic_coherence'].item():.2f}",
                'clinical_note': 'Reduced semantic coherence suggests difficulty maintaining topic or thought organization.'
            }
        
        # Idea density (strong Alzheimer's predictor)
        if 'idea_density' in biomarkers and biomarkers['idea_density'].item() < 0.4:
            interpretation['low_idea_density'] = {
                'detected': True,
                'severity': 'high_risk' if biomarkers['idea_density'].item() < 0.3 else 'moderate_risk',
                'idea_density': f"{biomarkers['idea_density'].item():.2f}",
                'clinical_note': 'Low idea density is a validated predictor of Alzheimer\'s disease risk.'
            }
        
        # Pronoun overuse (MCI marker)
        if 'pronoun_overuse' in biomarkers and biomarkers['pronoun_overuse'].item() > 0.6:
            interpretation['pronoun_overuse'] = {
                'detected': True,
                'pronoun_score': f"{biomarkers['pronoun_overuse'].item():.2f}",
                'clinical_note': 'Excessive pronoun use may indicate word-retrieval difficulties or early cognitive decline.'
            }
        
        # Cognitive load markers
        if 'cognitive_effort' in biomarkers and biomarkers['cognitive_effort'].item() > 0.7:
            interpretation['high_cognitive_load'] = {
                'detected': True,
                'effort_score': f"{biomarkers['cognitive_effort'].item():.2f}",
                'clinical_note': 'High cognitive effort in language production suggests compensatory mechanisms or decline.'
            }
        
        # Discourse fragmentation
        if 'fragmentation' in biomarkers and biomarkers['fragmentation'].item() > 0.6:
            interpretation['discourse_fragmentation'] = {
                'detected': True,
                'fragmentation_score': f"{biomarkers['fragmentation'].item():.2f}",
                'clinical_note': 'Fragmented discourse may indicate executive dysfunction or attention deficits.'
            }
        
        # Word-finding difficulty
        if 'word_finding_difficulty' in biomarkers and biomarkers['word_finding_difficulty'].item() > 0.6:
            interpretation['word_finding_difficulty'] = {
                'detected': True,
                'severity': 'moderate' if biomarkers['word_finding_difficulty'].item() > 0.7 else 'mild',
                'clinical_note': 'Word-finding difficulties present, consider anomia assessment.'
            }
        
        # Overall linguistic decline composite
        decline_markers = [
            biomarkers.get('grammar_simplification', torch.tensor(0.0)).item(),
            biomarkers.get('semantic_impoverishment', torch.tensor(0.0)).item(),
            biomarkers.get('information_content', torch.tensor(0.0)).item(),
            1.0 - biomarkers.get('idea_density', torch.tensor(1.0)).item()
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
        
        with torch.no_grad():
            for sample in text_samples:
                output = self.forward(sample, return_biomarkers=True, return_cognitive=True)
                trajectories.append({
                    'biomarkers': output['biomarkers'],
                    'cognitive_scores': output.get('cognitive_scores', {}),
                    'disease_probs': output['probabilities']
                })
        
        # Analyze trends
        key_metrics = ['lexical_diversity', 'idea_density', 'semantic_coherence', 'syntactic_complexity']
        trends = {}
        
        for metric in key_metrics:
            values = [t['biomarkers'].get(metric, torch.tensor(0.0)).item() for t in trajectories]
            
            # Calculate slope (rate of change)
            if len(values) >= 2:
                time_diffs = [time_points[i+1] - time_points[i] for i in range(len(time_points)-1)]
                value_diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
                slopes = [vd / td if td > 0 else 0 for vd, td in zip(value_diffs, time_diffs)]
                avg_slope = sum(slopes) / len(slopes)
                
                trends[metric] = {
                    'values': values,
                    'slope': avg_slope,
                    'trend': 'declining' if avg_slope < -0.05 else 'stable' if avg_slope < 0.05 else 'improving'
                }
        
        # Overall trajectory classification
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
        # Normative values (simplified - in practice, use validated norms)
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
        
        # Age adjustment (decline ~0.01 per year after 60)
        age_adjustment = 0.0
        if age and age > 60:
            age_adjustment = -(age - 60) * 0.01
        
        # Education adjustment (increase ~0.02 per year of education above 12)
        edu_adjustment = 0.0
        if education and education > 12:
            edu_adjustment = (education - 12) * 0.02
        
        comparisons = {}
        
        for metric, norm_mean in normative_means.items():
            if metric in biomarkers:
                value = biomarkers[metric].item()
                adjusted_mean = norm_mean + age_adjustment + edu_adjustment
                std = normative_stds[metric]
                
                # Calculate z-score
                z_score = (value - adjusted_mean) / std
                
                # Percentile (approximate)
                from scipy import stats as scipy_stats
                percentile = scipy_stats.norm.cdf(z_score) * 100
                
                # Clinical interpretation
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
                    'percentile': percentile,
                    'interpretation': interpretation
                }
        
        return comparisons