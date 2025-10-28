"""Comprehensive tests for text_biomarker.py"""

import pytest
import torch
import numpy as np
from typing import Dict, List
from biomarkers.models.text.text_biomarker import TextBiomarkerModel
from biomarkers.core.base import BiomarkerConfig


@pytest.fixture
def basic_config():
    """Create basic configuration for TextBiomarkerModel."""
    # --- MODIFIED: Added required base config keys ---
    return {
        'modality': 'text',                   # REQUIRED by BiomarkerConfig
        'model_type': 'TextBiomarkerModel',   # REQUIRED by BiomarkerConfig
        'embedding_dim': 768,
        'hidden_dim': 256,
        'num_diseases': 8,
        'dropout': 0.3,
        'max_seq_length': 512,
        'use_pretrained': False,
        'pretrained_model': 'bert-base-uncased',
        'biomarker_feature_dim': 50
    }

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    batch_size = 2
    seq_len = 50
    embedding_dim = 768
    return torch.randn(batch_size, seq_len, embedding_dim)


@pytest.fixture
def sample_text_metadata():
    """Create comprehensive text metadata for testing."""
    return {
        'tokens': [
            ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
            ['i', 'think', 'um', 'you', 'know', 'it', 'is', 'important']
        ],
        'content_words': [
            ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog'],
            ['think', 'important']
        ],
        'pos_tags': [
            ['DT', 'JJ', 'JJ', 'NN', 'VBZ', 'IN', 'DT', 'JJ', 'NN'],
            ['PRP', 'VBP', 'UH', 'PRP', 'VBP', 'PRP', 'VBZ', 'JJ']
        ],
        'parse_trees': [
            '(S (NP (DT the) (JJ quick) (JJ brown) (NN fox)) (VP (VBZ jumps)))',
            '(S (NP (PRP i)) (VP (VBP think)))'
        ],
        'timestamps': [
            np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
            np.array([0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0])
        ]
    }


class TestTextBiomarkerModelInitialization:
    """Test suite for TextBiomarkerModel initialization."""
    
    def test_basic_initialization(self, basic_config):
        """Test basic model initialization."""
        model = TextBiomarkerModel(basic_config)
        assert model is not None
        assert isinstance(model, TextBiomarkerModel)
    
    def test_config_parameters(self, basic_config):
        """Test that config parameters are properly set."""
        model = TextBiomarkerModel(basic_config)
        
        # --- MODIFIED: Access params from model.config ---
        assert model.config.metadata.get('embedding_dim') == 768
        assert model.config.hidden_dim == 256
        assert model.config.num_diseases == 8
        assert model.config.dropout == 0.3
        assert model.config.metadata.get('max_seq_length') == 512
        assert model.config.modality == 'text'

    def test_component_initialization(self, basic_config):
        """Test that all components are initialized."""
        model = TextBiomarkerModel(basic_config)
        
        # Check main components
        assert hasattr(model, 'text_encoder')
        assert hasattr(model, 'linguistic_analyzer')
        assert hasattr(model, 'cognitive_predictors')
        assert hasattr(model, 'classifier')
        assert hasattr(model, 'clinical_predictors')
        assert hasattr(model, 'change_detector')
        assert hasattr(model, 'uncertainty_estimator')
    
    def test_cognitive_predictors_initialization(self, basic_config):
        """Test cognitive predictors initialization."""
        model = TextBiomarkerModel(basic_config)
        
        assert 'MMSE' in model.cognitive_predictors
        assert 'MoCA' in model.cognitive_predictors
        assert 'CDR' in model.cognitive_predictors
    
    def test_clinical_predictors_initialization(self, basic_config):
        """Test clinical predictors initialization."""
        model = TextBiomarkerModel(basic_config)
        
        assert 'language_severity' in model.clinical_predictors
        assert 'decline_rate' in model.clinical_predictors
        assert 'communication_effectiveness' in model.clinical_predictors
    
    def test_different_configurations(self):
        """Test initialization with different configurations."""
        configs = [
            # --- MODIFIED: Added required base config keys ---
            {'modality': 'text', 'model_type': 'test', 'embedding_dim': 512, 'hidden_dim': 128, 'num_diseases': 4},
            {'modality': 'text', 'model_type': 'test', 'embedding_dim': 1024, 'hidden_dim': 512, 'num_diseases': 10},
            {'modality': 'text', 'model_type': 'test', 'embedding_dim': 768, 'hidden_dim': 256, 'num_diseases': 8, 'dropout': 0.5}
            # -----------------------------------------------
        ]
        
        for config in configs:
            model = TextBiomarkerModel(config)
            # --- MODIFIED: Access params from model.config ---
            assert model.config.metadata.get('embedding_dim') == config['embedding_dim']
            assert model.config.hidden_dim == config['hidden_dim']
            assert model.config.num_diseases == config['num_diseases']
            # -----------------------------------------------


class TestFeatureExtraction:
    """Test suite for feature extraction methods."""
    
    def test_extract_features(self, basic_config, sample_embeddings):
        """Test basic feature extraction."""
        model = TextBiomarkerModel(basic_config)
        features = model.extract_features(sample_embeddings)
        
        assert isinstance(features, torch.Tensor)
        # Should be mean-pooled raw embeddings
        # --- MODIFIED: Access params from model.config ---
        assert features.shape == (sample_embeddings.shape[0], model.config.metadata.get('embedding_dim'))
        # -----------------------------------------------
    
    def test_extract_biomarkers(self, basic_config, sample_embeddings):
        """Test biomarker extraction."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings)
        
        assert isinstance(biomarkers, dict)
        
        # Check lexical biomarkers
        assert 'lexical_diversity' in biomarkers
        assert 'vocabulary_richness' in biomarkers
        assert 'semantic_diversity' in biomarkers
        
        # Check syntactic biomarkers
        assert 'syntactic_complexity' in biomarkers
        assert 'subordination_index' in biomarkers
        assert 'grammar_accuracy' in biomarkers
        
        # Check semantic biomarkers
        assert 'semantic_coherence' in biomarkers
        assert 'topic_consistency' in biomarkers
        
        # Check cognitive load biomarkers
        assert 'cognitive_effort' in biomarkers
        assert 'word_finding_difficulty' in biomarkers
        
        # Check decline biomarkers
        assert 'grammar_simplification' in biomarkers
        assert 'idea_density' in biomarkers
        assert 'pronoun_overuse' in biomarkers
    
    def test_extract_biomarkers_with_metadata(self, basic_config, sample_embeddings, sample_text_metadata):
        """Test biomarker extraction with metadata."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings, sample_text_metadata)
        
        assert isinstance(biomarkers, dict)
        assert len(biomarkers) > 0
        assert 'lexical_density' in biomarkers
        assert biomarkers['lexical_density'].shape[0] == sample_embeddings.shape[0]

    
    def test_biomarker_values(self, basic_config, sample_embeddings):
        """Test that biomarker values are in valid ranges."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings)
        
        batch_size = sample_embeddings.shape[0]

        # Most biomarkers should be between 0 and 1
        for key, value in biomarkers.items():
            assert isinstance(value, torch.Tensor), f"{key} is not a tensor"
            assert not torch.any(torch.isnan(value)), f"NaN found in {key}"
            assert value.shape[0] == batch_size, f"{key} has wrong batch size"

            # Most metrics should be bounded
            if 'component' not in key and 'metric' not in key:
                # Check range for primary biomarkers (often sigmoided)
                assert torch.all(value >= 0.0), f"Biomarker {key} has values < 0"
                if key not in ['yules_k', 'words_per_minute', 'pause_count', 'speed_variability']:
                     assert torch.all(value <= 1.0), f"Biomarker {key} out of range [0, 1]"
    
    def test_aggregate_biomarkers(self, basic_config, sample_embeddings):
        """Test biomarker aggregation."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings)
        
        # FIXED: Pass batch_size
        batch_size = sample_embeddings.shape[0]
        aggregated = model.aggregate_biomarkers(biomarkers, batch_size)
        
        assert isinstance(aggregated, torch.Tensor)
        # --- MODIFIED: Access params from model.config ---
        assert aggregated.shape == (batch_size, model.config.metadata.get('biomarker_feature_dim'))
        # -----------------------------------------------
        assert not torch.any(torch.isnan(aggregated))
    
    def test_aggregate_biomarkers_consistency(self, basic_config, sample_embeddings):
        """Test that aggregation is consistent."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings)
        
        # FIXED: Pass batch_size
        batch_size = sample_embeddings.shape[0]
        aggregated1 = model.aggregate_biomarkers(biomarkers, batch_size)
        aggregated2 = model.aggregate_biomarkers(biomarkers, batch_size)
        
        assert torch.allclose(aggregated1, aggregated2)


class TestForwardPass:
    """Test suite for forward pass."""
    
    def test_basic_forward_pass(self, basic_config, sample_embeddings):
        """Test basic forward pass."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings)
        
        # Check essential outputs
        assert 'logits' in output
        assert 'probabilities' in output
        assert 'features' in output
        assert 'predictions' in output
        assert 'linguistic_pattern' in output
        
        batch_size = sample_embeddings.shape[0]
        # --- MODIFIED: Access params from model.config ---
        assert output['logits'].shape == (batch_size, model.config.num_diseases)
        assert output['probabilities'].shape == (batch_size, model.config.num_diseases)
        # -----------------------------------------------
        assert output['predictions'].shape == (batch_size,)
    
    def test_forward_with_all_options(self, basic_config, sample_embeddings):
        """Test forward pass with all return options enabled."""
        model = TextBiomarkerModel(basic_config)
        output = model(
            sample_embeddings,
            return_biomarkers=True,
            return_uncertainty=True,
            return_clinical=True,
            return_cognitive=True
        )
        
        assert 'biomarkers' in output
        assert 'cognitive_scores' in output
        assert 'clinical_scores' in output
        assert 'uncertainty' in output
        assert 'confidence' in output
        assert 'change_prediction' in output
    
    def test_forward_with_metadata(self, basic_config, sample_embeddings, sample_text_metadata):
        """Test forward pass with metadata."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings, text_metadata=sample_text_metadata)
        
        assert 'logits' in output
        assert output['logits'].shape[0] == sample_embeddings.shape[0]
    
    def test_probability_distribution(self, basic_config, sample_embeddings):
        """Test that output probabilities form valid distribution."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings)
        
        probabilities = output['probabilities']
        
        # Check that probabilities sum to 1
        prob_sums = probabilities.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
        
        # Check that all probabilities are between 0 and 1
        assert torch.all(probabilities >= 0.0) and torch.all(probabilities <= 1.0)
    
    def test_predictions_consistency(self, basic_config, sample_embeddings):
        """Test that predictions are consistent with probabilities."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings)
        
        predictions = output['predictions']
        probabilities = output['probabilities']
        
        # Predictions should be argmax of probabilities
        expected_predictions = probabilities.argmax(dim=1)
        assert torch.all(predictions == expected_predictions)
    
    def test_biomarkers_output(self, basic_config, sample_embeddings):
        """Test biomarkers output structure."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings, return_biomarkers=True)
        
        biomarkers = output['biomarkers']
        assert isinstance(biomarkers, dict)
        assert len(biomarkers) > 20  # Should have many biomarkers


class TestCognitiveScores:
    """Test suite for cognitive score prediction."""
    
    def test_mmse_prediction(self, basic_config, sample_embeddings):
        """Test MMSE score prediction."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings, return_cognitive=True)
        
        assert 'cognitive_scores' in output
        assert 'MMSE' in output['cognitive_scores']
        
        mmse_scores = output['cognitive_scores']['MMSE']
        assert mmse_scores.shape == (sample_embeddings.shape[0],)
        
        # MMSE should be between 0 and 30
        assert torch.all(mmse_scores >= 0.0) and torch.all(mmse_scores <= 30.0)
    
    def test_moca_prediction(self, basic_config, sample_embeddings):
        """Test MoCA score prediction."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings, return_cognitive=True)
        
        assert 'MoCA' in output['cognitive_scores']
        
        moca_scores = output['cognitive_scores']['MoCA']
        assert moca_scores.shape == (sample_embeddings.shape[0],)

        # MoCA should be between 0 and 30
        assert torch.all(moca_scores >= 0.0) and torch.all(moca_scores <= 30.0)
    
    def test_cdr_prediction(self, basic_config, sample_embeddings):
        """Test CDR score prediction."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings, return_cognitive=True)
        
        assert 'CDR' in output['cognitive_scores']
        
        cdr_scores = output['cognitive_scores']['CDR']
        
        # CDR should be probability distribution over 5 levels
        assert cdr_scores.shape == (sample_embeddings.shape[0], 5)
        
        # Should sum to 1
        cdr_sums = cdr_scores.sum(dim=1)
        assert torch.allclose(cdr_sums, torch.ones_like(cdr_sums), atol=1e-5)
    
    def test_cognitive_scores_batch(self, basic_config):
        """Test cognitive scores with different batch sizes."""
        model = TextBiomarkerModel(basic_config)
        
        for batch_size in [1, 2, 4, 8]:
            embeddings = torch.randn(batch_size, 50, 768)
            output = model(embeddings, return_cognitive=True)
            
            mmse = output['cognitive_scores']['MMSE']
            assert mmse.shape[0] == batch_size


class TestClinicalScores:
    """Test suite for clinical score prediction."""
    
    def test_language_severity_prediction(self, basic_config, sample_embeddings):
        """Test language severity prediction."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings, return_clinical=True)
        
        assert 'clinical_scores' in output
        assert 'language_severity' in output['clinical_scores']
        
        severity = output['clinical_scores']['language_severity']
        assert severity.shape == (sample_embeddings.shape[0],)

        # Should be between 0 and 5
        assert torch.all(severity >= 0.0) and torch.all(severity <= 5.0)
    
    def test_decline_rate_prediction(self, basic_config, sample_embeddings):
        """Test decline rate prediction."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings, return_clinical=True)
        
        assert 'decline_rate' in output['clinical_scores']
        
        decline_rate = output['clinical_scores']['decline_rate']
        assert decline_rate.shape == (sample_embeddings.shape[0],)

        # Should be between 0 and 1
        assert torch.all(decline_rate >= 0.0) and torch.all(decline_rate <= 1.0)
    
    def test_communication_effectiveness_prediction(self, basic_config, sample_embeddings):
        """Test communication effectiveness prediction."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings, return_clinical=True)
        
        assert 'communication_effectiveness' in output['clinical_scores']
        
        effectiveness = output['clinical_scores']['communication_effectiveness']
        assert effectiveness.shape == (sample_embeddings.shape[0],)

        # Should be between 0 and 10
        assert torch.all(effectiveness >= 0.0) and torch.all(effectiveness <= 10.0)


class TestUncertaintyQuantification:
    """Test suite for uncertainty quantification."""
    
    def test_uncertainty_estimation(self, basic_config, sample_embeddings):
        """Test uncertainty estimation."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings, return_uncertainty=True)
        
        assert 'uncertainty' in output
        assert 'confidence' in output
        
        uncertainty = output['uncertainty']
        confidence = output['confidence']
        
        # Check shapes
        batch_size = sample_embeddings.shape[0]
        # --- MODIFIED: Access params from model.config ---
        assert uncertainty.shape == (batch_size, model.config.num_diseases)
        assert confidence.shape == (batch_size, model.config.num_diseases)
        # -----------------------------------------------
        
        # Uncertainty should be positive
        assert torch.all(uncertainty >= 0.0)
        
        # Confidence should be between 0 and 1
        assert torch.all(confidence >= 0.0) and torch.all(confidence <= 1.0)
    
    def test_uncertainty_confidence_relationship(self, basic_config, sample_embeddings):
        """Test relationship between uncertainty and confidence."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings, return_uncertainty=True)
        
        uncertainty = output['uncertainty']
        confidence = output['confidence']
        
        # High uncertainty should mean low confidence
        # confidence = 1 / (1 + uncertainty)
        expected_confidence = 1.0 / (1.0 + uncertainty)
        assert torch.allclose(confidence, expected_confidence, atol=1e-5)


class TestChangeDetection:
    """Test suite for longitudinal change detection."""
    
    def test_change_prediction(self, basic_config, sample_embeddings):
        """Test change prediction output."""
        model = TextBiomarkerModel(basic_config)
        output = model(sample_embeddings)
        
        assert 'change_prediction' in output
        
        change_pred = output['change_prediction']
        
        # Should be probability distribution over 5 change categories
        assert change_pred.shape == (sample_embeddings.shape[0], 5)
        
        # Should sum to 1
        change_sums = change_pred.sum(dim=1)
        assert torch.allclose(change_sums, torch.ones_like(change_sums), atol=1e-5)


class TestClinicalInterpretation:
    """Test suite for clinical interpretation."""
    
    def test_get_clinical_interpretation(self, basic_config, sample_embeddings):
        """Test clinical interpretation generation."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings)
        
        interpretation = model.get_clinical_interpretation(biomarkers)
        
        assert isinstance(interpretation, dict)
    
    def test_interpretation_with_impairments(self, basic_config):
        """Test interpretation detects various impairments."""
        model = TextBiomarkerModel(basic_config)
        
        # Create biomarkers with known impairments
        impaired_biomarkers = {
            'lexical_diversity': torch.tensor([0.3]),  # Low (< 0.4)
            'syntactic_complexity': torch.tensor([0.3]),  # Low
            'semantic_coherence': torch.tensor([0.4]),  # Low
            'idea_density': torch.tensor([0.25]),  # Very low (< 0.3)
            'pronoun_overuse': torch.tensor([0.7]),  # High (> 0.6)
            'cognitive_effort': torch.tensor([0.8]),  # High (> 0.7)
            'fragmentation': torch.tensor([0.7]),  # High
            'word_finding_difficulty': torch.tensor([0.65]),  # Moderate
            'grammar_simplification': torch.tensor([0.6]),
            'semantic_impoverishment': torch.tensor([0.6]),
            'information_content': torch.tensor([0.3])
        }
        
        interpretation = model.get_clinical_interpretation(impaired_biomarkers)
        
        # Should detect multiple issues
        assert 'lexical_impairment' in interpretation
        assert 'syntactic_simplification' in interpretation
        assert 'low_idea_density' in interpretation
        assert 'pronoun_overuse' in interpretation
        assert 'high_cognitive_load' in interpretation
    
    def test_interpretation_severity_levels(self, basic_config):
        """Test that interpretation includes severity levels."""
        model = TextBiomarkerModel(basic_config)
        
        biomarkers = {
            'lexical_diversity': torch.tensor([0.25]),  # Moderate severity
            'idea_density': torch.tensor([0.25])  # High risk
        }
        
        interpretation = model.get_clinical_interpretation(biomarkers)
        
        assert 'lexical_impairment' in interpretation
        assert 'severity' in interpretation['lexical_impairment']
        assert interpretation['lexical_impairment']['severity'] == 'moderate'
        
        assert 'low_idea_density' in interpretation
        assert 'severity' in interpretation['low_idea_density']
        assert interpretation['low_idea_density']['severity'] == 'high_risk'

    
    def test_interpretation_clinical_notes(self, basic_config):
        """Test that interpretation includes clinical notes."""
        model = TextBiomarkerModel(basic_config)
        
        biomarkers = {
            'lexical_diversity': torch.tensor([0.3]),
            'idea_density': torch.tensor([0.3])
        }
        
        interpretation = model.get_clinical_interpretation(biomarkers)
        
        # Each detected issue should have a clinical note
        for issue_key, issue_data in interpretation.items():
            assert 'clinical_note' in issue_data


class TestTrajectoryPrediction:
    """Test suite for cognitive trajectory prediction."""
    
    def test_predict_cognitive_trajectory(self, basic_config):
        """Test cognitive trajectory prediction."""
        model = TextBiomarkerModel(basic_config)
        
        # Create multiple time points
        text_samples = [
            torch.randn(1, 50, 768),
            torch.randn(1, 50, 768),
            torch.randn(1, 50, 768)
        ]
        time_points = [0.0, 6.0, 12.0]  # 0, 6, 12 months
        
        trajectory = model.predict_cognitive_trajectory(text_samples, time_points)
        
        assert 'trajectory_class' in trajectory
        assert 'metric_trends' in trajectory
        assert 'decline_markers_count' in trajectory
        assert 'recommendation' in trajectory
    
    def test_trajectory_insufficient_data(self, basic_config):
        """Test trajectory prediction with insufficient data."""
        model = TextBiomarkerModel(basic_config)
        
        text_samples = [torch.randn(1, 50, 768)]
        time_points = [0.0]
        
        trajectory = model.predict_cognitive_trajectory(text_samples, time_points)
        
        assert 'error' in trajectory
    
    def test_trajectory_trends(self, basic_config):
        """Test that trajectory computes trends correctly."""
        model = TextBiomarkerModel(basic_config)
        
        text_samples = [
            torch.randn(1, 50, 768),
            torch.randn(1, 50, 768),
            torch.randn(1, 50, 768)
        ]
        time_points = [0.0, 6.0, 12.0]
        
        trajectory = model.predict_cognitive_trajectory(text_samples, time_points)
        
        trends = trajectory['metric_trends']
        
        # Check that trends are computed for key metrics
        for metric in ['lexical_diversity', 'idea_density', 'semantic_coherence', 'syntactic_complexity']:
            assert metric in trends
            assert 'values' in trends[metric]
            assert 'slope' in trends[metric]
            assert 'trend' in trends[metric]
            assert trends[metric]['trend'] in ['declining', 'stable', 'improving']
    
    def test_trajectory_classification(self, basic_config):
        """Test trajectory classification."""
        model = TextBiomarkerModel(basic_config)
        
        text_samples = [
            torch.randn(1, 50, 768),
            torch.randn(1, 50, 768),
            torch.randn(1, 50, 768)
        ]
        time_points = [0.0, 6.0, 12.0]
        
        trajectory = model.predict_cognitive_trajectory(text_samples, time_points)
        
        trajectory_class = trajectory['trajectory_class']
        assert trajectory_class in ['rapid_decline', 'moderate_decline', 'mild_decline', 'stable']
    
    def test_trajectory_recommendation(self, basic_config):
        """Test that trajectory includes clinical recommendation."""
        model = TextBiomarkerModel(basic_config)
        
        text_samples = [
            torch.randn(1, 50, 768),
            torch.randn(1, 50, 768)
        ]
        time_points = [0.0, 6.0]
        
        trajectory = model.predict_cognitive_trajectory(text_samples, time_points)
        
        recommendation = trajectory['recommendation']
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0


class TestNormativeComparison:
    """Test suite for normative data comparison."""
    
    def test_compare_to_normative_data_basic(self, basic_config, sample_embeddings):
        """Test basic normative comparison."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings)
        
        comparison = model.compare_to_normative_data(biomarkers)
        
        assert isinstance(comparison, dict)
    
    def test_compare_with_age_adjustment(self, basic_config, sample_embeddings):
        """Test normative comparison with age adjustment."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings)
        
        comparison = model.compare_to_normative_data(biomarkers, age=70)
        
        # Should include comparison for available metrics
        for metric in ['lexical_diversity', 'syntactic_complexity', 'semantic_coherence', 
                      'idea_density', 'information_content']:
            assert metric in comparison
            assert 'normative_mean' in comparison[metric]
            assert 'z_score' in comparison[metric]
    
    def test_compare_with_education_adjustment(self, basic_config, sample_embeddings):
        """Test normative comparison with education adjustment."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings)
        
        comparison = model.compare_to_normative_data(biomarkers, education=16)
        
        assert isinstance(comparison, dict)
    
    def test_compare_with_both_adjustments(self, basic_config, sample_embeddings):
        """Test normative comparison with both age and education."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings)
        
        comparison = model.compare_to_normative_data(biomarkers, age=70, education=16)
        
        assert isinstance(comparison, dict)
    
    def test_comparison_z_scores(self, basic_config, sample_embeddings):
        """Test z-score calculation in comparison."""
        model = TextBiomarkerModel(basic_config)
        # Use single sample for predictable z-score
        
        # Use device from model for embedding
        device = next(model.parameters()).device
        embeddings_single = torch.randn(1, 50, 768).to(device)
        biomarkers = model.extract_biomarkers(embeddings_single)
        
        comparison = model.compare_to_normative_data(biomarkers, age=60, education=12)
        
        # Check z-scores are reasonable
        for metric, metric_comp in comparison.items():
            z_score = metric_comp['z_score']
            value = metric_comp['value']
            mean = metric_comp['normative_mean']
            
            # --- MODIFIED: Correctly access the class attribute ---
            assert metric in model.NORMATIVE_STDS, f"Metric {metric} not in model's NORMATIVE_STDS"
            std = model.NORMATIVE_STDS[metric] # No longer a hack
            # ------------------------------------------------------
            
            # Re-calculate z-score to check logic
            expected_z = (value - mean) / (std + 1e-8)
            assert np.isclose(z_score, expected_z)

    def test_comparison_interpretations(self, basic_config, sample_embeddings):
        """Test that comparison includes interpretations."""
        model = TextBiomarkerModel(basic_config)
        biomarkers = model.extract_biomarkers(sample_embeddings)
        
        comparison = model.compare_to_normative_data(biomarkers)
        
        valid_interpretations = ['severely impaired', 'moderately impaired', 
                                'mildly impaired', 'average', 'above average']
        
        for metric, metric_comp in comparison.items():
            assert 'interpretation' in metric_comp
            assert metric_comp['interpretation'] in valid_interpretations


class TestGradientFlow:
    """Test suite for gradient flow and backpropagation."""
    
    def test_gradient_flow_through_model(self, basic_config, sample_embeddings):
        """Test that gradients flow through the entire model."""
        model = TextBiomarkerModel(basic_config)
        model.train() # Ensure model is in train mode for gradients
        embeddings = sample_embeddings.requires_grad_(True)
        
        output = model(embeddings)
        loss = output['logits'].sum()
        
        model.zero_grad()
        loss.backward()
        
        assert embeddings.grad is not None
        assert torch.any(embeddings.grad != 0)
    
    def test_gradient_flow_with_all_outputs(self, basic_config, sample_embeddings):
        """Test gradient flow with all outputs enabled."""
        model = TextBiomarkerModel(basic_config)
        model.train()
        embeddings = sample_embeddings.requires_grad_(True)
        
        output = model(
            embeddings,
            return_biomarkers=True,
            return_uncertainty=True,
            return_clinical=True,
            return_cognitive=True
        )
        
        # Combine all scalar/vector losses
        loss = output['logits'].sum() + \
               output['cognitive_scores']['MMSE'].sum() + \
               output['cognitive_scores']['CDR'].sum() + \
               output['clinical_scores']['language_severity'].sum() + \
               output['uncertainty'].sum() + \
               output['change_prediction'].sum()

        model.zero_grad()
        loss.backward()
        
        assert embeddings.grad is not None
        assert torch.any(embeddings.grad != 0)


class TestModelTraining:
    """Test suite for model training behavior."""
    
    def test_training_mode_dropout(self, basic_config, sample_embeddings):
        """Test dropout behavior in training mode."""
        model = TextBiomarkerModel(basic_config)
        model.train()
        
        output1 = model(sample_embeddings)
        output2 = model(sample_embeddings)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1['logits'], output2['logits'])
    
    def test_eval_mode_consistency(self, basic_config, sample_embeddings):
        """Test output consistency in eval mode."""
        model = TextBiomarkerModel(basic_config)
        model.eval()
        
        with torch.no_grad():
            output1 = model(sample_embeddings)
            output2 = model(sample_embeddings)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1['logits'], output2['logits'])
    
    def test_parameter_updates(self, basic_config, sample_embeddings):
        """Test that parameters are updated during training."""
        model = TextBiomarkerModel(basic_config)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Get initial parameters
        initial_params = [p.clone().detach() for p in model.parameters()]
        
        # Training step
        optimizer.zero_grad()
        output = model(sample_embeddings)
        loss = output['logits'].sum()
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        params_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break
        
        assert params_changed, "Model parameters did not update after training step"


class TestEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_single_sample(self, basic_config):
        """Test with single sample."""
        model = TextBiomarkerModel(basic_config)
        model.eval()
        embeddings = torch.randn(1, 50, 768)
        
        output = model(embeddings)
        
        assert output['logits'].shape[0] == 1
        assert output['probabilities'].shape[0] == 1
    
    def test_large_batch(self, basic_config):
        """Test with large batch size."""
        model = TextBiomarkerModel(basic_config)
        model.eval()
        embeddings = torch.randn(32, 50, 768)
        
        output = model(embeddings)
        
        assert output['logits'].shape[0] == 32
    
    def test_variable_sequence_lengths(self, basic_config):
        """Test with different sequence lengths."""
        model = TextBiomarkerModel(basic_config)
        model.eval()
        
        for seq_len in [10, 50, 100, 200]:
            embeddings = torch.randn(2, seq_len, 768)
            output = model(embeddings)
            
            # --- MODIFIED: Access params from model.config ---
            assert output['logits'].shape == (2, model.config.num_diseases)
            # -----------------------------------------------
    
    def test_zero_embeddings(self, basic_config):
        """Test with zero embeddings."""
        model = TextBiomarkerModel(basic_config)
        model.eval()
        embeddings = torch.zeros(2, 50, 768)
        
        output = model(embeddings)
        
        assert not torch.any(torch.isnan(output['logits']))
        assert not torch.any(torch.isnan(output['probabilities']))
    
    def test_no_nan_outputs(self, basic_config, sample_embeddings):
        """Test that outputs don't contain NaN values."""
        model = TextBiomarkerModel(basic_config)
        model.eval()
        output = model(
            sample_embeddings,
            return_biomarkers=True,
            return_uncertainty=True,
            return_clinical=True,
            return_cognitive=True
        )
        
        # Check all outputs for NaN
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                assert not torch.any(torch.isnan(value)), f"NaN found in {key}"
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        assert not torch.any(torch.isnan(subvalue)), \
                            f"NaN found in {key}.{subkey}"


class TestDeviceCompatibility:
    """Test suite for device compatibility."""
    
    def test_cpu_computation(self, basic_config, sample_embeddings):
        """Test computation on CPU."""
        # Force config to use CPU
        cpu_config = basic_config.copy()
        cpu_config['device'] = 'cpu'
        
        model = TextBiomarkerModel(cpu_config) 
        
        embeddings_cpu = sample_embeddings.cpu()
        output = model(embeddings_cpu) # model.forward handles moving input
        
        assert output['logits'].device.type == 'cpu'
        # Also check a sub-module parameter device
        assert next(model.linguistic_analyzer.parameters()).device.type == 'cpu'
    
    @pytest.mark.gpu
    # --- MODIFIED: Added sample_embeddings to the signature ---
    def test_gpu_computation(self, basic_config, cuda_device, sample_embeddings):
    # -----------------------------------------------------------
        """Test computation on GPU."""
        # Force config to use GPU
        gpu_config = basic_config.copy()
        gpu_config['device'] = str(cuda_device) # Use string representation
        
        model = TextBiomarkerModel(gpu_config)

        # sample_embeddings fixture already provides GPU tensor if cuda_device is active
        # The 'embeddings' variable now correctly holds the tensor from the fixture
        embeddings = sample_embeddings 
        
        output = model(embeddings) # model.forward handles moving input
        
        assert output['logits'].device.type == 'cuda'
         # Also check a sub-module parameter device
        assert next(model.linguistic_analyzer.parameters()).device.type == 'cuda'

    @pytest.mark.gpu
    def test_device_transfer(self, basic_config, sample_embeddings, cuda_device):
        """Test transferring model between devices."""
        # CPU Model
        cpu_config = basic_config.copy()
        cpu_config['device'] = 'cpu'
        # --- MODIFIED: Removed redundant .cpu() ---
        model_cpu = TextBiomarkerModel(cpu_config)
        # ------------------------------------------
        model_cpu.eval()
        embeddings_cpu = sample_embeddings.cpu()

        # CPU computation
        with torch.no_grad():
            output_cpu = model_cpu(embeddings_cpu)
        
        # GPU Model
        gpu_config = basic_config.copy()
        gpu_config['device'] = str(cuda_device)
        # --- MODIFIED: Removed redundant .to(cuda_device) ---
        model_gpu = TextBiomarkerModel(gpu_config)
        # ----------------------------------------------------
        model_gpu.load_state_dict(model_cpu.state_dict()) # Ensure parameters are identical
        model_gpu.eval()
        embeddings_gpu = sample_embeddings.to(cuda_device) # Test embeddings might be CPU initially
        
        with torch.no_grad():
            output_gpu = model_gpu(embeddings_gpu)
        
        # Results should be similar (allowing for numerical differences)
        assert output_gpu['logits'].device.type == 'cuda'
        assert torch.allclose(
            output_cpu['logits'],
            output_gpu['logits'].cpu(),
            atol=1e-4
        )

class TestModelSerialization:
    """Test suite for model saving and loading."""
    
    def test_model_state_dict(self, basic_config):
        """Test getting model state dict."""
        model = TextBiomarkerModel(basic_config)
        state_dict = model.state_dict()
        
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
    
    def test_save_and_load_state(self, basic_config, tmp_path):
        """Test saving and loading model state."""
        model = TextBiomarkerModel(basic_config)
        model.eval()
        
        # Save state
        save_path = tmp_path / "model_state.pth"
        torch.save(model.state_dict(), save_path)
        
        # Load state
        new_model = TextBiomarkerModel(basic_config)
        new_model.load_state_dict(torch.load(save_path))
        new_model.eval()
        
        # Compare parameters
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

        # Test consistency
        embeddings = torch.randn(2, 50, 768)
        with torch.no_grad():
            out1 = model(embeddings)
            out2 = new_model(embeddings)
        
        assert torch.allclose(out1['logits'], out2['logits'])