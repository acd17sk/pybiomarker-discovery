"""Comprehensive tests for linguistic_analyzer.py"""

import pytest
import torch
import numpy as np
from biomarkers.models.text.linguistic_analyzer import (
    LinguisticAnalyzer,
    LexicalDiversityAnalyzer,
    SyntacticComplexityAnalyzer,
    SemanticCoherenceAnalyzer,
    DiscourseStructureAnalyzer,
    CognitiveLoadAnalyzer,
    LinguisticDeclineAnalyzer,
    TemporalAnalyzer
)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    batch_size = 2
    seq_len = 50
    embedding_dim = 768
    return torch.randn(batch_size, seq_len, embedding_dim)


@pytest.fixture
def sample_tokens():
    """Create sample tokens for testing."""
    return [
        ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', 'and', 
         'then', 'the', 'fox', 'runs', 'away', 'quickly', 'through', 'the', 'forest'],
        ['i', 'think', 'that', 'um', 'you', 'know', 'it', 'is', 'like', 'really', 
         'important', 'to', 'um', 'understand', 'the', 'the', 'concept', 'here']
    ]


@pytest.fixture
def sample_text_metadata(sample_tokens):
    """Create comprehensive text metadata for testing."""
    return {
        'tokens': sample_tokens,
        'content_words': [
            ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog', 'runs', 'quickly', 'forest'],
            ['think', 'important', 'understand', 'concept']
        ],
        'pos_tags': [
            ['DT', 'JJ', 'JJ', 'NN', 'VBZ', 'IN', 'DT', 'JJ', 'NN', 'CC',
             'RB', 'DT', 'NN', 'VBZ', 'RB', 'RB', 'IN', 'DT', 'NN'],
            ['PRP', 'VBP', 'IN', 'UH', 'PRP', 'VBP', 'PRP', 'VBZ', 'IN', 'RB',
             'JJ', 'TO', 'UH', 'VB', 'DT', 'DT', 'NN', 'RB']
        ],
        'parse_trees': [
            '(S (NP (DT the) (JJ quick) (JJ brown) (NN fox)) (VP (VBZ jumps) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))))',
            '(S (NP (PRP i)) (VP (VBP think) (SBAR (IN that) (S (NP (PRP you)) (VP (VBP know) (S (NP (PRP it)) (VP (VBZ is) (ADJP (RB like) (RB really) (JJ important)))))))))'
        ],
        'timestamps': [
            np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                     5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]),
            np.array([0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                     5.5, 6.0, 6.5, 7.5, 8.0, 8.5, 9.0, 9.5])
        ]
    }


class TestLexicalDiversityAnalyzer:
    """Test suite for LexicalDiversityAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization with different parameters."""
        analyzer = LexicalDiversityAnalyzer(embedding_dim=768, output_dim=128)
        assert analyzer is not None
        assert isinstance(analyzer, torch.nn.Module)
        
        # Test with custom dimensions
        analyzer_custom = LexicalDiversityAnalyzer(embedding_dim=512, output_dim=256)
        assert analyzer_custom is not None
    
    def test_forward_pass_basic(self, sample_embeddings):
        """Test basic forward pass without metadata."""
        analyzer = LexicalDiversityAnalyzer()
        output = analyzer(sample_embeddings)
        
        # Check output structure
        assert isinstance(output, dict)
        assert 'features' in output
        assert 'ttr' in output
        assert 'richness_metrics' in output
        assert 'semantic_diversity' in output
        assert 'frequency_metrics' in output
        
        # Check tensor shapes
        batch_size = sample_embeddings.shape[0]
        assert output['features'].shape == (batch_size, 128)
        assert output['ttr'].shape == (batch_size,)
        assert output['richness_metrics'].shape == (batch_size, 8)
        assert output['semantic_diversity'].shape == (batch_size,)
        
        # Check value ranges (should be between 0 and 1 due to sigmoid)
        assert torch.all(output['ttr'] >= 0.0) and torch.all(output['ttr'] <= 1.0)
        assert torch.all(output['semantic_diversity'] >= 0.0) and torch.all(output['semantic_diversity'] <= 1.0)
    
    def test_forward_pass_with_metadata(self, sample_embeddings, sample_text_metadata):
        """Test forward pass with text metadata."""
        analyzer = LexicalDiversityAnalyzer()
        output = analyzer(sample_embeddings, sample_text_metadata)
        
        # Check that statistical features are computed
        assert 'statistical_ttr' in output
        assert 'mattr' in output
        assert 'hapax_ratio' in output
        assert 'yules_k' in output
        assert 'lexical_density' in output
        
        # Check shapes
        batch_size = sample_embeddings.shape[0]
        assert output['statistical_ttr'].shape == (batch_size,)
        assert output['mattr'].shape == (batch_size,)
        
        # Check value ranges
        assert torch.all(output['statistical_ttr'] >= 0.0) and torch.all(output['statistical_ttr'] <= 1.0)
        assert torch.all(output['lexical_density'] >= 0.0) and torch.all(output['lexical_density'] <= 1.0)
    
    def test_compute_statistical_features(self, sample_text_metadata):
        """Test statistical feature computation."""
        analyzer = LexicalDiversityAnalyzer()
        features = analyzer.compute_statistical_features(sample_text_metadata)
        
        assert features is not None
        assert features.shape == (2, 5)  # 2 samples, 5 features
        
        # Check TTR calculation for first sample
        tokens = sample_text_metadata['tokens'][0]
        types = len(set(tokens))
        token_count = len(tokens)
        expected_ttr = types / token_count
        assert torch.isclose(features[0, 0], torch.tensor(expected_ttr), atol=0.01)
    
    def test_compute_statistical_features_no_metadata(self):
        """Test statistical features with no metadata."""
        analyzer = LexicalDiversityAnalyzer()
        features = analyzer.compute_statistical_features(None)
        assert features is None
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        analyzer = LexicalDiversityAnalyzer()
        
        for batch_size in [1, 2, 4, 8]:
            embeddings = torch.randn(batch_size, 50, 768)
            output = analyzer(embeddings)
            
            assert output['features'].shape[0] == batch_size
            assert output['ttr'].shape[0] == batch_size
    
    def test_gradient_flow(self, sample_embeddings):
        """Test that gradients flow properly."""
        analyzer = LexicalDiversityAnalyzer()
        embeddings = sample_embeddings.requires_grad_(True)
        
        output = analyzer(embeddings)
        loss = output['features'].sum()
        loss.backward()
        
        assert embeddings.grad is not None
        assert torch.any(embeddings.grad != 0)


class TestSyntacticComplexityAnalyzer:
    """Test suite for SyntacticComplexityAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = SyntacticComplexityAnalyzer(embedding_dim=768, output_dim=128)
        assert analyzer is not None
    
    def test_forward_pass_basic(self, sample_embeddings):
        """Test basic forward pass."""
        analyzer = SyntacticComplexityAnalyzer()
        output = analyzer(sample_embeddings)
        
        assert isinstance(output, dict)
        assert 'features' in output
        assert 'parse_metrics' in output
        assert 'dependency_metrics' in output
        assert 'subordination_index' in output
        assert 'yngve_depth' in output
        assert 'frazier_score' in output
        assert 'grammar_accuracy' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['features'].shape == (batch_size, 128)
        assert output['parse_metrics'].shape == (batch_size, 7)
        assert output['dependency_metrics'].shape == (batch_size, 6)
    
    def test_forward_pass_with_metadata(self, sample_embeddings, sample_text_metadata):
        """Test forward pass with parse tree metadata."""
        analyzer = SyntacticComplexityAnalyzer()
        output = analyzer(sample_embeddings, sample_text_metadata)
        
        assert 'tree_depth' in output
        assert 'avg_tree_depth' in output
        assert 'clause_count' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['tree_depth'].shape == (batch_size,)
    
    def test_compute_parse_features(self, sample_text_metadata):
        """Test parse tree feature computation."""
        analyzer = SyntacticComplexityAnalyzer()
        features = analyzer.compute_parse_features(sample_text_metadata)
        
        assert features is not None
        assert features.shape == (2, 3)
        
        # Check that depth is computed (should be > 0 for valid parse trees)
        assert torch.all(features[:, 0] > 0)
    
    def test_tree_depth_calculation(self):
        """Test tree depth calculation."""
        analyzer = SyntacticComplexityAnalyzer()
        
        # Simple tree
        simple_tree = "(S (NP (DT the) (NN cat)) (VP (VBZ sits)))"
        depth = analyzer._compute_tree_depth(simple_tree)
        assert depth > 0
        
        # Complex tree with more nesting
        complex_tree = "(S (NP (DT the) (JJ big) (NN cat)) (VP (VBZ sits) (PP (IN on) (NP (DT the) (NN mat)))))"
        complex_depth = analyzer._compute_tree_depth(complex_tree)
        assert complex_depth >= depth
    
    def test_clause_counting(self):
        """Test clause counting in parse trees."""
        analyzer = SyntacticComplexityAnalyzer()
        
        tree_with_sbar = "(S (NP (PRP I)) (VP (VBP think) (SBAR (IN that) (S (NP (PRP you)) (VP (VBP know))))))"
        clause_count = analyzer._count_clauses(tree_with_sbar)
        assert clause_count >= 2  # Main clause + embedded clause
    
    def test_subordination_components(self, sample_embeddings):
        """Test subordination analysis components."""
        analyzer = SyntacticComplexityAnalyzer()
        output = analyzer(sample_embeddings)
        
        subordination = output['subordination_components']
        assert subordination.shape[1] == 8  # 8 subordination metrics
        
        # Values should be between 0 and 1 (sigmoid applied)
        assert torch.all(subordination >= 0.0) and torch.all(subordination <= 1.0)
    
    def test_grammar_accuracy_components(self, sample_embeddings):
        """Test grammar accuracy components."""
        analyzer = SyntacticComplexityAnalyzer()
        output = analyzer(sample_embeddings)
        
        grammar = output['grammar_components']
        assert grammar.shape[1] == 5  # 5 grammar metrics
        
        # Check mean accuracy
        mean_accuracy = output['grammar_accuracy']
        assert torch.all(mean_accuracy >= 0.0) and torch.all(mean_accuracy <= 1.0)


class TestSemanticCoherenceAnalyzer:
    """Test suite for SemanticCoherenceAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = SemanticCoherenceAnalyzer(embedding_dim=768, output_dim=128)
        assert analyzer is not None
        assert hasattr(analyzer, 'semantic_encoder')
        assert hasattr(analyzer, 'coherence_attention')
    
    def test_forward_pass(self, sample_embeddings):
        """Test forward pass."""
        analyzer = SemanticCoherenceAnalyzer()
        output = analyzer(sample_embeddings)
        
        assert 'features' in output
        assert 'topic_consistency' in output
        assert 'semantic_similarity' in output
        assert 'inter_sentence_similarity' in output
        assert 'global_coherence' in output
        assert 'semantic_flow' in output
        assert 'anaphora_quality' in output
        assert 'attention_weights' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['features'].shape == (batch_size, 128)
    
    def test_compute_semantic_similarity(self, sample_embeddings):
        """Test semantic similarity computation."""
        analyzer = SemanticCoherenceAnalyzer()
        sim_metrics = analyzer.compute_semantic_similarity(sample_embeddings)
        
        assert sim_metrics.shape == (sample_embeddings.shape[0], 3)
        
        # Check that similarities are valid (0 to 1)
        # FIXED: Similarity matrix calculation can result in values slightly outside [0, 1] due to precision
        # The function now clamps to [0, 1]
        assert torch.all(sim_metrics >= 0.0) and torch.all(sim_metrics <= 1.0)
    
    def test_attention_weights(self, sample_embeddings):
        """Test attention weight generation."""
        analyzer = SemanticCoherenceAnalyzer()
        output = analyzer(sample_embeddings)
        
        attention = output['attention_weights']
        batch_size = sample_embeddings.shape[0]
        seq_len = sample_embeddings.shape[1]
        num_heads = analyzer.coherence_attention.num_heads
        
        # FIXED: Attention shape with average_attn_weights=False is [batch_size, num_heads, seq_len, seq_len]
        assert attention.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Attention weights (per head) should sum to 1
        attention_sums = attention.sum(dim=-1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=0.01)
    
    def test_topic_metrics(self, sample_embeddings):
        """Test topic consistency metrics."""
        analyzer = SemanticCoherenceAnalyzer()
        output = analyzer(sample_embeddings)
        
        topic_metrics = output['topic_metrics']
        assert topic_metrics.shape[1] == 6
        
        # All metrics should be in [0, 1] range
        assert torch.all(topic_metrics >= 0.0) and torch.all(topic_metrics <= 1.0)
    
    def test_anaphora_components(self, sample_embeddings):
        """Test anaphora resolution quality metrics."""
        analyzer = SemanticCoherenceAnalyzer()
        output = analyzer(sample_embeddings)
        
        anaphora = output['anaphora_components']
        assert anaphora.shape[1] == 4
        
        mean_quality = output['anaphora_quality']
        assert torch.all(mean_quality >= 0.0) and torch.all(mean_quality <= 1.0)


class TestDiscourseStructureAnalyzer:
    """Test suite for DiscourseStructureAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = DiscourseStructureAnalyzer(embedding_dim=768, output_dim=128)
        assert analyzer is not None
    
    def test_forward_pass(self, sample_embeddings):
        """Test forward pass."""
        analyzer = DiscourseStructureAnalyzer()
        output = analyzer(sample_embeddings)
        
        assert 'features' in output
        assert 'reference_chains' in output
        assert 'reference_quality' in output
        assert 'cohesion_markers' in output
        assert 'cohesion_density' in output
        assert 'narrative_flow' in output
        assert 'narrative_completeness' in output
        assert 'story_structure' in output
        assert 'information_structure' in output
        assert 'information_flow' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['features'].shape == (batch_size, 128)
    
    def test_build_discourse_graph(self, sample_embeddings):
        """Test discourse graph construction."""
        analyzer = DiscourseStructureAnalyzer()
        
        # FIXED: Pass the required 'encoder' argument
        graph_features = analyzer.build_discourse_graph(sample_embeddings, analyzer.discourse_encoder)
        
        batch_size, seq_len, _ = sample_embeddings.shape
        # FIXED: The output dimension of the encoder/GAT is 256
        assert graph_features.shape == (batch_size, seq_len, 256)
    
    def test_reference_chain_metrics(self, sample_embeddings):
        """Test reference chain analysis."""
        analyzer = DiscourseStructureAnalyzer()
        output = analyzer(sample_embeddings)
        
        reference = output['reference_chains']
        assert reference.shape[1] == 7
        
        # All should be in [0, 1] range
        assert torch.all(reference >= 0.0) and torch.all(reference <= 1.0)
    
    def test_cohesion_marker_analysis(self, sample_embeddings):
        """Test cohesion marker analysis."""
        analyzer = DiscourseStructureAnalyzer()
        output = analyzer(sample_embeddings)
        
        cohesion = output['cohesion_markers']
        assert cohesion.shape[1] == 8
        
        density = output['cohesion_density']
        assert torch.all(density >= 0.0) and torch.all(density <= 1.0)
    
    def test_narrative_flow_metrics(self, sample_embeddings):
        """Test narrative flow analysis."""
        analyzer = DiscourseStructureAnalyzer()
        output = analyzer(sample_embeddings)
        
        narrative = output['narrative_flow']
        assert narrative.shape[1] == 9
        
        completeness = output['narrative_completeness']
        assert torch.all(completeness >= 0.0) and torch.all(completeness <= 1.0)
    
    def test_information_structure(self, sample_embeddings):
        """Test information structure analysis."""
        analyzer = DiscourseStructureAnalyzer()
        output = analyzer(sample_embeddings)
        
        info_structure = output['information_structure']
        assert info_structure.shape[1] == 6
        
        info_flow = output['information_flow']
        assert torch.all(info_flow >= 0.0) and torch.all(info_flow <= 1.0)


class TestCognitiveLoadAnalyzer:
    """Test suite for CognitiveLoadAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = CognitiveLoadAnalyzer(embedding_dim=768, output_dim=128)
        assert analyzer is not None
    
    def test_forward_pass(self, sample_embeddings):
        """Test forward pass."""
        analyzer = CognitiveLoadAnalyzer()
        output = analyzer(sample_embeddings)
        
        assert 'features' in output
        assert 'filled_pauses' in output
        assert 'word_finding_difficulty' in output
        assert 'repetition_score' in output
        assert 'cognitive_effort' in output
        assert 'self_correction' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['features'].shape == (batch_size, 128)
    
    def test_forward_pass_with_metadata(self, sample_embeddings, sample_text_metadata):
        """Test forward pass with token metadata."""
        analyzer = CognitiveLoadAnalyzer()
        output = analyzer(sample_embeddings, sample_text_metadata)
        
        assert 'pause_count' in output
        assert 'pause_ratio' in output
        assert 'immediate_repetitions' in output
        assert 'phrase_repetitions' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['pause_count'].shape == (batch_size,)
    
    def test_detect_filled_pauses(self, sample_text_metadata):
        """Test filled pause detection."""
        analyzer = CognitiveLoadAnalyzer()
        pause_features = analyzer.detect_filled_pauses(sample_text_metadata)
        
        assert pause_features is not None
        assert pause_features.shape == (2, 4)
        
        # Second sample has "um" markers, should have higher pause count
        assert pause_features[1, 0] > pause_features[0, 0]
    
    def test_detect_repetitions(self, sample_text_metadata):
        """Test repetition detection."""
        analyzer = CognitiveLoadAnalyzer()
        rep_features = analyzer.detect_repetitions(sample_text_metadata)
        
        assert rep_features is not None
        assert rep_features.shape == (2, 4)
        
        # Second sample has "the the" repetition
        assert rep_features[1, 0] > 0
    
    def test_word_finding_components(self, sample_embeddings):
        """Test word-finding difficulty components."""
        analyzer = CognitiveLoadAnalyzer()
        output = analyzer(sample_embeddings)
        
        word_finding = output['word_finding_components']
        assert word_finding.shape[1] == 7
        assert torch.all(word_finding >= 0.0) and torch.all(word_finding <= 1.0)
    
    def test_repetition_components(self, sample_embeddings):
        """Test repetition analysis components."""
        analyzer = CognitiveLoadAnalyzer()
        output = analyzer(sample_embeddings)
        
        repetition = output['repetition_components']
        assert repetition.shape[1] == 6
        assert torch.all(repetition >= 0.0) and torch.all(repetition <= 1.0)
    
    def test_cognitive_effort_components(self, sample_embeddings):
        """Test cognitive effort components."""
        analyzer = CognitiveLoadAnalyzer()
        output = analyzer(sample_embeddings)
        
        effort = output['effort_components']
        assert effort.shape[1] == 5
        assert torch.all(effort >= 0.0) and torch.all(effort <= 1.0)
    
    def test_self_correction_analysis(self, sample_embeddings):
        """Test self-correction analysis."""
        analyzer = CognitiveLoadAnalyzer()
        output = analyzer(sample_embeddings)
        
        correction = output['self_correction']
        assert correction.shape[1] == 4
        
        correction_rate = output['correction_rate']
        assert torch.all(correction_rate >= 0.0) and torch.all(correction_rate <= 1.0)


class TestLinguisticDeclineAnalyzer:
    """Test suite for LinguisticDeclineAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = LinguisticDeclineAnalyzer(embedding_dim=768, output_dim=128)
        assert analyzer is not None
    
    def test_forward_pass(self, sample_embeddings):
        """Test forward pass."""
        analyzer = LinguisticDeclineAnalyzer()
        output = analyzer(sample_embeddings)
        
        assert 'features' in output
        assert 'grammar_simplification' in output
        assert 'information_content' in output
        assert 'idea_density' in output
        assert 'pronoun_overuse' in output
        assert 'semantic_impoverishment' in output
        assert 'fragmentation' in output
        assert 'disease_patterns' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['features'].shape == (batch_size, 128)
    
    def test_forward_pass_with_metadata(self, sample_embeddings, sample_text_metadata):
        """Test forward pass with POS tag metadata."""
        analyzer = LinguisticDeclineAnalyzer()
        output = analyzer(sample_embeddings, sample_text_metadata)
        
        assert 'content_word_ratio' in output
        assert 'function_word_ratio' in output
        assert 'pronoun_ratio' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['content_word_ratio'].shape == (batch_size,)
    
    def test_compute_content_ratio(self, sample_text_metadata):
        """Test content word ratio computation."""
        analyzer = LinguisticDeclineAnalyzer()
        content_features = analyzer.compute_content_ratio(sample_text_metadata)
        
        assert content_features is not None
        assert content_features.shape == (2, 3)
        
        # Content ratio should be between 0 and 1
        assert torch.all(content_features[:, 0] >= 0.0) and torch.all(content_features[:, 0] <= 1.0)
        
        # Content + function should be < 1.0 (there are other POS tags)
        total_ratio = content_features[:, 0] + content_features[:, 1]
        assert torch.all(total_ratio <= 1.0)
    
    def test_grammar_simplification_components(self, sample_embeddings):
        """Test grammar simplification metrics."""
        analyzer = LinguisticDeclineAnalyzer()
        output = analyzer(sample_embeddings)
        
        grammar = output['grammar_components']
        assert grammar.shape[1] == 8
        assert torch.all(grammar >= 0.0) and torch.all(grammar <= 1.0)
    
    def test_information_content_components(self, sample_embeddings):
        """Test information content metrics."""
        analyzer = LinguisticDeclineAnalyzer()
        output = analyzer(sample_embeddings)
        
        info_content = output['info_content_components']
        assert info_content.shape[1] == 7
        assert torch.all(info_content >= 0.0) and torch.all(info_content <= 1.0)
        
        # Idea density is a specific component
        idea_density = output['idea_density']
        assert torch.all(idea_density >= 0.0) and torch.all(idea_density <= 1.0)
    
    def test_pronoun_analysis_components(self, sample_embeddings):
        """Test pronoun overuse metrics."""
        analyzer = LinguisticDeclineAnalyzer()
        output = analyzer(sample_embeddings)
        
        pronoun = output['pronoun_components']
        assert pronoun.shape[1] == 6
        assert torch.all(pronoun >= 0.0) and torch.all(pronoun <= 1.0)
    
    def test_semantic_impoverishment_components(self, sample_embeddings):
        """Test semantic impoverishment metrics."""
        analyzer = LinguisticDeclineAnalyzer()
        output = analyzer(sample_embeddings)
        
        semantic = output['semantic_components']
        assert semantic.shape[1] == 7
        assert torch.all(semantic >= 0.0) and torch.all(semantic <= 1.0)
    
    def test_fragmentation_components(self, sample_embeddings):
        """Test fragmentation metrics."""
        analyzer = LinguisticDeclineAnalyzer()
        output = analyzer(sample_embeddings)
        
        fragmentation = output['fragmentation_components']
        assert fragmentation.shape[1] == 5
        assert torch.all(fragmentation >= 0.0) and torch.all(fragmentation <= 1.0)
    
    def test_disease_patterns(self, sample_embeddings):
        """Test disease pattern prediction."""
        analyzer = LinguisticDeclineAnalyzer()
        output = analyzer(sample_embeddings)
        
        disease_patterns = output['disease_patterns']
        assert disease_patterns.shape[1] == 6
        
        # Should be softmax probabilities (sum to 1)
        pattern_sums = disease_patterns.sum(dim=1)
        assert torch.allclose(pattern_sums, torch.ones_like(pattern_sums), atol=0.01)


class TestTemporalAnalyzer:
    """Test suite for TemporalAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = TemporalAnalyzer(embedding_dim=768, output_dim=128)
        assert analyzer is not None
    
    def test_forward_pass(self, sample_embeddings):
        """Test forward pass."""
        analyzer = TemporalAnalyzer()
        output = analyzer(sample_embeddings)
        
        assert 'features' in output
        assert 'speed_metrics' in output
        assert 'revision_patterns' in output
        assert 'revision_rate' in output
        assert 'pause_patterns' in output
        assert 'pause_frequency' in output
        assert 'temporal_dynamics' in output
        assert 'writing_fluency' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['features'].shape == (batch_size, 128)
    
    def test_forward_pass_with_metadata(self, sample_embeddings, sample_text_metadata):
        """Test forward pass with timestamp metadata."""
        analyzer = TemporalAnalyzer()
        output = analyzer(sample_embeddings, sample_text_metadata)
        
        assert 'words_per_minute' in output
        assert 'pause_ratio' in output
        assert 'pause_count' in output
        assert 'speed_variability' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['words_per_minute'].shape == (batch_size,)
    
    def test_compute_timing_features(self, sample_text_metadata):
        """Test timing feature computation."""
        analyzer = TemporalAnalyzer()
        timing_features = analyzer.compute_timing_features(sample_text_metadata)
        
        assert timing_features is not None
        assert timing_features.shape == (2, 4)
        
        # WPM should be positive
        assert torch.all(timing_features[:, 0] >= 0.0)
        
        # Pause ratio should be between 0 and 1
        assert torch.all(timing_features[:, 1] >= 0.0) and torch.all(timing_features[:, 1] <= 1.0)
    
    def test_speed_metrics(self, sample_embeddings):
        """Test writing speed metrics."""
        analyzer = TemporalAnalyzer()
        output = analyzer(sample_embeddings)
        
        speed = output['speed_metrics']
        assert speed.shape[1] == 6
    
    def test_revision_patterns(self, sample_embeddings):
        """Test revision pattern analysis."""
        analyzer = TemporalAnalyzer()
        output = analyzer(sample_embeddings)
        
        revision = output['revision_patterns']
        assert revision.shape[1] == 8
        assert torch.all(revision >= 0.0) and torch.all(revision <= 1.0)
        
        revision_rate = output['revision_rate']
        assert torch.all(revision_rate >= 0.0) and torch.all(revision_rate <= 1.0)
    
    def test_pause_patterns(self, sample_embeddings):
        """Test pause pattern analysis."""
        analyzer = TemporalAnalyzer()
        output = analyzer(sample_embeddings)
        
        pauses = output['pause_patterns']
        assert pauses.shape[1] == 7
        assert torch.all(pauses >= 0.0) and torch.all(pauses <= 1.0)
    
    def test_temporal_dynamics(self, sample_embeddings):
        """Test temporal dynamics metrics."""
        analyzer = TemporalAnalyzer()
        output = analyzer(sample_embeddings)
        
        dynamics = output['temporal_dynamics']
        assert dynamics.shape[1] == 5
        assert torch.all(dynamics >= 0.0) and torch.all(dynamics <= 1.0)
        
        fluency = output['writing_fluency']
        assert torch.all(fluency >= 0.0) and torch.all(fluency <= 1.0)


class TestLinguisticAnalyzer:
    """Test suite for the main LinguisticAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = LinguisticAnalyzer(embedding_dim=768, hidden_dim=256, dropout=0.3)
        assert analyzer is not None
        
        # Check that all sub-analyzers are initialized
        assert hasattr(analyzer, 'lexical_analyzer')
        assert hasattr(analyzer, 'syntactic_analyzer')
        assert hasattr(analyzer, 'semantic_analyzer')
        assert hasattr(analyzer, 'discourse_analyzer')
        assert hasattr(analyzer, 'cognitive_load_analyzer')
        assert hasattr(analyzer, 'decline_analyzer')
        assert hasattr(analyzer, 'temporal_analyzer')
        assert hasattr(analyzer, 'feature_fusion')
        assert hasattr(analyzer, 'pattern_classifier')
    
    def test_forward_pass(self, sample_embeddings):
        """Test complete forward pass."""
        analyzer = LinguisticAnalyzer()
        output = analyzer(sample_embeddings)
        
        # Check main outputs
        assert 'linguistic_features' in output
        assert 'lexical_metrics' in output
        assert 'syntactic_metrics' in output
        assert 'semantic_metrics' in output
        assert 'discourse_metrics' in output
        assert 'cognitive_load_metrics' in output
        assert 'decline_markers' in output
        assert 'temporal_metrics' in output
        assert 'pattern_logits' in output
        assert 'pattern_probs' in output
        
        batch_size = sample_embeddings.shape[0]
        assert output['linguistic_features'].shape == (batch_size, 128)
        assert output['pattern_logits'].shape == (batch_size, 8)
        assert output['pattern_probs'].shape == (batch_size, 8)
    
    def test_forward_pass_with_metadata(self, sample_embeddings, sample_text_metadata):
        """Test forward pass with complete metadata."""
        analyzer = LinguisticAnalyzer()
        output = analyzer(sample_embeddings, sample_text_metadata)
        
        # Check that all sub-analyzers processed metadata
        assert 'lexical_metrics' in output
        assert 'syntactic_metrics' in output
        
        # Statistical features should be present
        lexical = output['lexical_metrics']
        assert 'statistical_ttr' in lexical
        assert 'mattr' in lexical
    
    def test_pattern_classification(self, sample_embeddings):
        """Test linguistic pattern classification."""
        analyzer = LinguisticAnalyzer()
        output = analyzer(sample_embeddings)
        
        pattern_probs = output['pattern_probs']
        
        # Should be softmax probabilities (sum to 1)
        prob_sums = pattern_probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=0.01)
        
        # All probabilities should be between 0 and 1
        assert torch.all(pattern_probs >= 0.0) and torch.all(pattern_probs <= 1.0)
    
    def test_feature_fusion(self, sample_embeddings):
        """Test feature fusion from all analyzers."""
        analyzer = LinguisticAnalyzer()
        output = analyzer(sample_embeddings)
        
        # Features should be fused to hidden_dim // 2
        linguistic_features = output['linguistic_features']
        assert linguistic_features.shape[1] == 128  # 256 // 2
    
    def test_different_embedding_dims(self):
        """Test with different embedding dimensions."""
        for embedding_dim in [512, 768, 1024]:
            analyzer = LinguisticAnalyzer(embedding_dim=embedding_dim, hidden_dim=256)
            embeddings = torch.randn(2, 50, embedding_dim)
            output = analyzer(embeddings)
            
            assert output['linguistic_features'].shape == (2, 128)
    
    def test_different_hidden_dims(self):
        """Test with different hidden dimensions."""
        for hidden_dim in [128, 256, 512]:
            analyzer = LinguisticAnalyzer(embedding_dim=768, hidden_dim=hidden_dim)
            embeddings = torch.randn(2, 50, 768)
            output = analyzer(embeddings)
            
            assert output['linguistic_features'].shape == (2, hidden_dim // 2)
    
    def test_gradient_flow(self, sample_embeddings):
        """Test gradient flow through entire analyzer."""
        analyzer = LinguisticAnalyzer()
        embeddings = sample_embeddings.requires_grad_(True)
        
        output = analyzer(embeddings)
        loss = output['linguistic_features'].sum()
        loss.backward()
        
        assert embeddings.grad is not None
        assert torch.any(embeddings.grad != 0)
    
    def test_all_metrics_present(self, sample_embeddings):
        """Test that all expected metrics are present in output."""
        analyzer = LinguisticAnalyzer()
        output = analyzer(sample_embeddings)
        
        # Lexical metrics
        lexical = output['lexical_metrics']
        assert 'features' in lexical
        assert 'ttr' in lexical
        assert 'richness_metrics' in lexical
        
        # Syntactic metrics
        syntactic = output['syntactic_metrics']
        assert 'features' in syntactic
        assert 'parse_metrics' in syntactic
        
        # Semantic metrics
        semantic = output['semantic_metrics']
        assert 'features' in semantic
        assert 'topic_consistency' in semantic
        
        # Discourse metrics
        discourse = output['discourse_metrics']
        assert 'features' in discourse
        assert 'reference_chains' in discourse
        
        # Cognitive load metrics
        cognitive = output['cognitive_load_metrics']
        assert 'features' in cognitive
        assert 'cognitive_effort' in cognitive
        
        # Decline markers
        decline = output['decline_markers']
        assert 'features' in decline
        assert 'disease_patterns' in decline
        
        # Temporal metrics
        temporal = output['temporal_metrics']
        assert 'features' in temporal
        assert 'writing_fluency' in temporal
    
    def test_batch_processing(self):
        """Test batch processing with varying batch sizes."""
        analyzer = LinguisticAnalyzer()
        
        for batch_size in [1, 2, 4, 8, 16]:
            embeddings = torch.randn(batch_size, 50, 768)
            output = analyzer(embeddings)
            
            assert output['linguistic_features'].shape[0] == batch_size
            assert output['pattern_probs'].shape[0] == batch_size
    
    def test_variable_sequence_lengths(self):
        """Test with different sequence lengths."""
        analyzer = LinguisticAnalyzer()
        
        for seq_len in [10, 50, 100, 200]:
            embeddings = torch.randn(2, seq_len, 768)
            output = analyzer(embeddings)
            
            # Output should be consistent regardless of sequence length
            assert output['linguistic_features'].shape == (2, 128)
    
    def test_dropout_behavior(self, sample_embeddings):
        """Test dropout behavior in training vs eval mode."""
        analyzer = LinguisticAnalyzer(dropout=0.5)
        
        # Training mode
        analyzer.train()
        output_train1 = analyzer(sample_embeddings)
        output_train2 = analyzer(sample_embeddings)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train1['linguistic_features'], 
                                  output_train2['linguistic_features'])
        
        # Eval mode
        analyzer.eval()
        output_eval1 = analyzer(sample_embeddings)
        output_eval2 = analyzer(sample_embeddings)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval1['linguistic_features'], 
                             output_eval2['linguistic_features'])
    
    def test_no_nan_outputs(self, sample_embeddings):
        """Test that outputs don't contain NaN values."""
        analyzer = LinguisticAnalyzer()
        output = analyzer(sample_embeddings)
        
        # Check main features
        assert not torch.any(torch.isnan(output['linguistic_features']))
        assert not torch.any(torch.isnan(output['pattern_probs']))
        
        # Check all sub-analyzer outputs
        for key in ['lexical_metrics', 'syntactic_metrics', 'semantic_metrics',
                   'discourse_metrics', 'cognitive_load_metrics', 'decline_markers',
                   'temporal_metrics']:
            metrics = output[key]
            for metric_key, metric_value in metrics.items():
                if isinstance(metric_value, torch.Tensor):
                    assert not torch.any(torch.isnan(metric_value)), \
                        f"NaN found in {key}.{metric_key}"
    
    def test_device_compatibility(self, sample_embeddings):
        """Test model works on different devices."""
        analyzer = LinguisticAnalyzer()
        
        # CPU test
        output_cpu = analyzer(sample_embeddings)
        assert output_cpu['linguistic_features'].device.type == 'cpu'
        
        # GPU test (if available)
        if torch.cuda.is_available():
            analyzer_gpu = analyzer.cuda()
            embeddings_gpu = sample_embeddings.cuda()
            output_gpu = analyzer_gpu(embeddings_gpu)
            assert output_gpu['linguistic_features'].device.type == 'cuda'


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_metadata(self, sample_embeddings):
        """Test with empty metadata."""
        analyzer = LinguisticAnalyzer()
        output = analyzer(sample_embeddings, text_metadata={})
        
        assert 'linguistic_features' in output
        assert output['linguistic_features'].shape[0] == sample_embeddings.shape[0]
    
    def test_partial_metadata(self, sample_embeddings):
        """Test with partial metadata (only tokens)."""
        analyzer = LinguisticAnalyzer()
        metadata = {'tokens': [['word1', 'word2', 'word3']]}
        output = analyzer(sample_embeddings[:1], metadata)
        
        assert 'linguistic_features' in output
    
    def test_single_token_sequence(self):
        """Test with very short sequences."""
        analyzer = LinguisticAnalyzer()
        embeddings = torch.randn(1, 1, 768)  # Single token
        output = analyzer(embeddings)
        
        assert output['linguistic_features'].shape == (1, 128)
    
    def test_very_long_sequence(self):
        """Test with very long sequences."""
        analyzer = LinguisticAnalyzer()
        embeddings = torch.randn(1, 1000, 768)  # Long sequence
        output = analyzer(embeddings)
        
        assert output['linguistic_features'].shape == (1, 128)
    
    def test_zero_embeddings(self):
        """Test with zero embeddings."""
        analyzer = LinguisticAnalyzer()
        embeddings = torch.zeros(2, 50, 768)
        output = analyzer(embeddings)
        
        assert not torch.any(torch.isnan(output['linguistic_features']))
    
    def test_identical_embeddings(self):
        """Test with identical embeddings for all tokens."""
        analyzer = LinguisticAnalyzer()
        embeddings = torch.ones(2, 50, 768)
        output = analyzer(embeddings)
        
        assert output['linguistic_features'].shape == (2, 128)