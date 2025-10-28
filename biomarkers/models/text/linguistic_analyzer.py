"""Linguistic analysis modules for text biomarkers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import numpy as np
import re
from collections import Counter


class LinguisticAnalyzer(nn.Module):
    """Complete linguistic analysis module"""
    
    def __init__(self,
                 embedding_dim: int = 768,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Sub-analyzers
        self.lexical_analyzer = LexicalDiversityAnalyzer(embedding_dim, hidden_dim // 2)
        self.syntactic_analyzer = SyntacticComplexityAnalyzer(embedding_dim, hidden_dim // 2)
        self.semantic_analyzer = SemanticCoherenceAnalyzer(embedding_dim, hidden_dim // 2)
        self.discourse_analyzer = DiscourseStructureAnalyzer(embedding_dim, hidden_dim // 2)
        self.cognitive_load_analyzer = CognitiveLoadAnalyzer(embedding_dim, hidden_dim // 2)
        self.decline_analyzer = LinguisticDeclineAnalyzer(embedding_dim, hidden_dim // 2)
        self.temporal_analyzer = TemporalAnalyzer(embedding_dim, hidden_dim // 2)
        
        # Feature fusion
        # 7 analyzers, each outputting hidden_dim // 2
        fusion_dim = 7 * (hidden_dim // 2)
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Linguistic pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
    
    def forward(self, 
                embeddings: torch.Tensor,
                text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: Text embeddings [batch, seq_len, embedding_dim]
            text_metadata: Dictionary containing parsed text features
        """
        # Extract linguistic components
        lexical_output = self.lexical_analyzer(embeddings, text_metadata)
        syntactic_output = self.syntactic_analyzer(embeddings, text_metadata)
        semantic_output = self.semantic_analyzer(embeddings, text_metadata)
        discourse_output = self.discourse_analyzer(embeddings, text_metadata)
        cognitive_output = self.cognitive_load_analyzer(embeddings, text_metadata)
        decline_output = self.decline_analyzer(embeddings, text_metadata)
        temporal_output = self.temporal_analyzer(embeddings, text_metadata)
        
        # Concatenate features
        combined = torch.cat([
            lexical_output['features'],
            syntactic_output['features'],
            semantic_output['features'],
            discourse_output['features'],
            cognitive_output['features'],
            decline_output['features'],
            temporal_output['features']
        ], dim=-1)
        
        # Fuse features
        fused = self.feature_fusion(combined)
        
        # Classify linguistic patterns
        pattern_logits = self.pattern_classifier(fused)
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        
        return {
            'linguistic_features': fused,
            'lexical_metrics': lexical_output,
            'syntactic_metrics': syntactic_output,
            'semantic_metrics': semantic_output,
            'discourse_metrics': discourse_output,
            'cognitive_load_metrics': cognitive_output,
            'decline_markers': decline_output,
            'temporal_metrics': temporal_output,
            'pattern_logits': pattern_logits,
            'pattern_probs': pattern_probs
        }


class LexicalDiversityAnalyzer(nn.Module):
    """Analyze lexical diversity and vocabulary richness"""
    
    def __init__(self, embedding_dim: int = 768, output_dim: int = 128):
        super().__init__()
        
        # Vocabulary encoder
        self.vocab_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Type-token ratio estimator
        self.ttr_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Vocabulary richness metrics
        self.richness_metrics = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
        
        # Semantic diversity estimator
        self.semantic_diversity = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # Word frequency analyzer
        self.frequency_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        self.output_proj = nn.Linear(256, output_dim)
    
    def compute_statistical_features(self, text_metadata: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Compute statistical lexical features from text"""
        if text_metadata is None or 'tokens' not in text_metadata:
            return None
        
        tokens_list = text_metadata.get('tokens', [])
        if not isinstance(tokens_list, list) or len(tokens_list) == 0:
             return None

        batch_size = len(tokens_list)
        features = []
        
        for batch_idx in range(batch_size):
            batch_tokens = tokens_list[batch_idx]
            if not batch_tokens: # Handle empty token list
                features.append([0.0] * 5)
                continue
            
            # Type-Token Ratio (TTR)
            types = len(set(batch_tokens))
            token_count = len(batch_tokens)
            ttr = types / token_count if token_count > 0 else 0.0
            
            # Moving-Average Type-Token Ratio (MATTR)
            window_size = 50
            if token_count >= window_size:
                mattr_values = []
                for i in range(token_count - window_size + 1):
                    window = batch_tokens[i:i+window_size]
                    window_types = len(set(window))
                    mattr_values.append(window_types / window_size)
                mattr = np.mean(mattr_values) if mattr_values else 0.0
            else:
                mattr = ttr
            
            # Hapax Legomena (words appearing once)
            word_freq = Counter(batch_tokens)
            hapax = sum(1 for count in word_freq.values() if count == 1)
            hapax_ratio = hapax / types if types > 0 else 0.0
            
            # Yule's K (lexical diversity measure)
            if token_count > 0:
                freq_spectrum = Counter(word_freq.values())
                M1 = sum(i * freq_spectrum[i] for i in freq_spectrum)
                M2 = sum(i**2 * freq_spectrum[i] for i in freq_spectrum)
                yules_k = 10000 * (M2 - M1) / (M1 ** 2) if M1 > 0 and M1**2 > 0 else 0.0
            else:
                yules_k = 0.0
            
            # Lexical density (content words / total words)
            content_words_list = text_metadata.get('content_words')
            lexical_density = 0.0
            if content_words_list and len(content_words_list) > batch_idx:
                content_words = content_words_list[batch_idx]
                lexical_density = len(content_words) / token_count if token_count > 0 else 0.0
            
            features.append([ttr, mattr, hapax_ratio, yules_k, lexical_density])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, 
                embeddings: torch.Tensor,
                text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Analyze lexical diversity"""
        # Pool embeddings
        pooled = torch.mean(embeddings, dim=1)
        
        # Encode vocabulary
        vocab_features = self.vocab_encoder(pooled)
        
        # Estimate TTR
        ttr = self.ttr_estimator(vocab_features).squeeze(-1)
        
        # Compute richness metrics
        richness = self.richness_metrics(vocab_features)
        richness = torch.sigmoid(richness)
        
        # Semantic diversity
        semantic_div = self.semantic_diversity(vocab_features)
        semantic_div = torch.sigmoid(semantic_div)
        
        # Frequency analysis
        frequency = self.frequency_analyzer(vocab_features)
        
        # Compute statistical features if metadata available
        statistical_features = self.compute_statistical_features(text_metadata)
        
        features = self.output_proj(vocab_features)
        
        output = {
            'features': features,
            'ttr': ttr,
            'richness_metrics': richness,
            'semantic_diversity': semantic_div.mean(dim=-1),
            'semantic_diversity_components': semantic_div,
            'frequency_metrics': frequency
        }
        
        if statistical_features is not None:
            # FIXED: Move computed features to the correct device
            statistical_features = statistical_features.to(embeddings.device)
            output['statistical_ttr'] = statistical_features[:, 0]
            output['mattr'] = statistical_features[:, 1]
            output['hapax_ratio'] = statistical_features[:, 2]
            output['yules_k'] = statistical_features[:, 3]
            output['lexical_density'] = statistical_features[:, 4]
        
        return output


class SyntacticComplexityAnalyzer(nn.Module):
    """Analyze syntactic complexity and grammatical structure"""
    
    def __init__(self, embedding_dim: int = 768, output_dim: int = 128):
        super().__init__()
        
        # Syntactic encoder
        self.syntactic_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Parse tree analyzer
        self.parse_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
        # Dependency distance analyzer
        self.dependency_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        # Subordination analyzer
        self.subordination_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
        
        # Yngve depth calculator
        self.yngve_calculator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        # Frazier score
        self.frazier_calculator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        # Grammatical accuracy
        self.grammar_accuracy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        self.output_proj = nn.Linear(256, output_dim)
    
    def compute_parse_features(self, text_metadata: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Compute syntactic features from parse trees"""
        if text_metadata is None or 'parse_trees' not in text_metadata:
            return None

        parse_trees = text_metadata.get('parse_trees', [])
        if not isinstance(parse_trees, list) or len(parse_trees) == 0:
            return None

        batch_size = len(parse_trees)
        features = []
        
        for batch_idx in range(batch_size):
            tree = parse_trees[batch_idx]
            
            if tree and isinstance(tree, str):
                depth = self._compute_tree_depth(tree)
                avg_depth = self._compute_avg_depth(tree)
                num_clauses = self._count_clauses(tree)
            else:
                depth = 0
                avg_depth = 0
                num_clauses = 0
            
            features.append([depth, avg_depth, num_clauses])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _compute_tree_depth(self, tree: str) -> int:
        """Compute maximum depth of parse tree"""
        return tree.count('(')
    
    def _compute_avg_depth(self, tree: str) -> float:
        """Compute average depth across all nodes (approximate)"""
        nodes = tree.count('(') + tree.count(')')
        return (self._compute_tree_depth(tree) / (nodes / 2.0)) if nodes > 0 else 0.0
    
    def _count_clauses(self, tree: str) -> int:
        """Count clauses in parse tree"""
        clause_markers = ['(S ', '(SBAR ', '(SBARQ ']
        return sum(tree.count(marker) for marker in clause_markers)
    
    def forward(self,
                embeddings: torch.Tensor,
                text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Analyze syntactic complexity"""
        pooled = torch.mean(embeddings, dim=1)
        syntactic_features = self.syntactic_encoder(pooled)
        
        parse_metrics = self.parse_analyzer(syntactic_features)
        dependency = self.dependency_analyzer(syntactic_features)
        subordination = self.subordination_analyzer(syntactic_features)
        subordination = torch.sigmoid(subordination)
        
        yngve = self.yngve_calculator(syntactic_features)
        frazier = self.frazier_calculator(syntactic_features)
        grammar = self.grammar_accuracy(syntactic_features)
        grammar = torch.sigmoid(grammar)
        
        parse_features = self.compute_parse_features(text_metadata)
        features = self.output_proj(syntactic_features)
        
        output = {
            'features': features,
            'parse_metrics': parse_metrics,
            'dependency_metrics': dependency,
            'subordination_index': subordination[:, 0],
            'subordination_components': subordination,
            'yngve_depth': yngve[:, 0],
            'frazier_score': frazier[:, 0],
            'grammar_accuracy': grammar.mean(dim=-1),
            'grammar_components': grammar
        }
        
        if parse_features is not None:
            # FIXED: Move computed features to the correct device
            parse_features = parse_features.to(embeddings.device)
            output['tree_depth'] = parse_features[:, 0]
            output['avg_tree_depth'] = parse_features[:, 1]
            output['clause_count'] = parse_features[:, 2]
        
        return output


class SemanticCoherenceAnalyzer(nn.Module):
    """Analyze semantic coherence and topic consistency"""
    
    def __init__(self, embedding_dim: int = 768, output_dim: int = 128):
        super().__init__()
        
        # Semantic encoder with attention
        self.semantic_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Self-attention for coherence
        self.coherence_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # Topic consistency analyzer
        self.topic_analyzer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        # Semantic similarity calculator
        self.similarity_calculator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
        
        # Anaphora resolution quality
        self.anaphora_analyzer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        
        self.output_proj = nn.Linear(512, output_dim)
    
    def compute_semantic_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise semantic similarity between sentences"""
        normalized = F.normalize(embeddings, p=2, dim=-1)
        similarity_matrix = torch.matmul(normalized, normalized.transpose(1, 2))
        
        # Clamp to [0, 1] range as similarities should be non-negative
        similarity_matrix = torch.clamp(similarity_matrix, 0.0, 1.0)
        
        batch_size = embeddings.shape[0]
        device = embeddings.device
        metrics = []
        
        for b in range(batch_size):
            sim = similarity_matrix[b]
            seq_len = sim.shape[0]
            
            if seq_len > 1:
                adjacent_sim = torch.diagonal(sim, offset=1).mean()
                # Create mask on the correct device
                mask = torch.triu(torch.ones_like(sim), diagonal=1)
                global_coherence = (sim * mask).sum() / (mask.sum() + 1e-8)
                semantic_flow = 1.0 - torch.diagonal(sim, offset=1).std()
            else:
                # FIXED: Create tensors on the correct device
                adjacent_sim = torch.tensor(1.0, device=device)
                global_coherence = torch.tensor(1.0, device=device)
                semantic_flow = torch.tensor(1.0, device=device)
            
            # Clamp all metrics to [0, 1]
            adjacent_sim = torch.clamp(adjacent_sim, 0.0, 1.0)
            global_coherence = torch.clamp(global_coherence, 0.0, 1.0)
            semantic_flow = torch.clamp(semantic_flow, 0.0, 1.0)
            
            metrics.append(torch.stack([adjacent_sim, global_coherence, semantic_flow]))
        
        return torch.stack(metrics)
    
    def forward(self,
                embeddings: torch.Tensor,
                text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Analyze semantic coherence"""
        # LSTM encoding for temporal semantics
        semantic_features, _ = self.semantic_encoder(embeddings)
        
        # Self-attention for coherence
        attended, attention_weights = self.coherence_attention(
            semantic_features,
            semantic_features,
            semantic_features,
            need_weights=True,
            average_attn_weights=False
        )
        
        # Pool features
        pooled = torch.mean(attended, dim=1)
        
        # Topic analysis
        topic_metrics = self.topic_analyzer(pooled)
        topic_metrics = torch.sigmoid(topic_metrics)
        
        # Similarity analysis
        similarity = self.similarity_calculator(pooled)
        similarity = torch.sigmoid(similarity)
        
        # Anaphora analysis
        anaphora = self.anaphora_analyzer(pooled)
        anaphora = torch.sigmoid(anaphora)
        
        # Compute semantic similarity metrics
        sim_metrics = self.compute_semantic_similarity(embeddings)
        
        features = self.output_proj(pooled)
        
        return {
            'features': features,
            'topic_consistency': topic_metrics[:, 0],
            'topic_metrics': topic_metrics,
            'semantic_similarity': similarity.mean(dim=-1),
            'similarity_components': similarity,
            'inter_sentence_similarity': sim_metrics[:, 0],
            'global_coherence': sim_metrics[:, 1],
            'semantic_flow': sim_metrics[:, 2],
            'anaphora_quality': anaphora.mean(dim=-1),
            'anaphora_components': anaphora,
            'attention_weights': attention_weights
        }


class DiscourseStructureAnalyzer(nn.Module):
    """Analyze discourse structure and narrative flow"""
    
    def __init__(self, embedding_dim: int = 768, output_dim: int = 128):
        super().__init__()
        
        # Discourse encoder using GNN for discourse relations
        self.discourse_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Graph attention for discourse relations
        self.discourse_gat = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        # Reference chain analyzer
        self.reference_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
        # Cohesion marker analyzer
        self.cohesion_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
        
        # Narrative flow analyzer
        self.narrative_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        
        # Information structure
        self.information_structure = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        self.output_proj = nn.Linear(256, output_dim)
    
    def build_discourse_graph(self, 
                                raw_embeddings: torch.Tensor,
                                encoder: nn.Module) -> torch.Tensor:
        """
        Build discourse relation graph by encoding and applying attention.
        
        Args:
            raw_embeddings: (batch_size, seq_len, embedding_dim)
            encoder: The discourse_encoder module
        """
        batch_size, seq_len, embed_dim = raw_embeddings.shape
        
        # Project each position in sequence to 256 dimensions
        sequence_projected = encoder(raw_embeddings.view(-1, embed_dim))
        sequence_projected = sequence_projected.view(batch_size, seq_len, 256)
        
        # Apply graph attention
        graph_features, _ = self.discourse_gat(
            sequence_projected,
            sequence_projected,
            sequence_projected
        )
        
        return graph_features
    
    def forward(self,
                embeddings: torch.Tensor,
                text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Analyze discourse structure"""
        
        # FIXED: Pass raw embeddings and encoder to graph builder
        graph_features = self.build_discourse_graph(embeddings, self.discourse_encoder) # [batch, seq_len, 256]
        graph_pooled = torch.mean(graph_features, dim=1)  # [batch, 256]
        
        reference = self.reference_analyzer(graph_pooled)
        reference = torch.sigmoid(reference)
        
        cohesion = self.cohesion_analyzer(graph_pooled)
        cohesion = torch.sigmoid(cohesion)
        
        narrative = self.narrative_analyzer(graph_pooled)
        narrative = torch.sigmoid(narrative)
        
        info_structure = self.information_structure(graph_pooled)
        info_structure = torch.sigmoid(info_structure)
        
        features = self.output_proj(graph_pooled)
        
        return {
            'features': features,
            'reference_chains': reference,
            'reference_quality': reference[:, 1],
            'cohesion_markers': cohesion,
            'cohesion_density': cohesion[:, 6],
            'narrative_flow': narrative,
            'narrative_completeness': narrative[:, 4],
            'story_structure': narrative[:, 0],
            'information_structure': info_structure,
            'information_flow': info_structure[:, 2]
        }


class CognitiveLoadAnalyzer(nn.Module):
    """Analyze cognitive load markers in text"""
    
    def __init__(self, embedding_dim: int = 768, output_dim: int = 128):
        super().__init__()
        
        # Cognitive load encoder
        self.cognitive_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Filled pause detector
        self.filled_pause_detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        # Word-finding difficulty detector
        self.word_finding_detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
        # Repetition analyzer
        self.repetition_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        # Cognitive effort estimator
        self.effort_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        # Self-correction analyzer
        self.correction_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        self.output_proj = nn.Linear(256, output_dim)
    
    def detect_filled_pauses(self, text_metadata: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Detect filled pauses and hesitation markers"""
        if text_metadata is None or 'tokens' not in text_metadata:
            return None
        
        tokens_list = text_metadata.get('tokens', [])
        if not isinstance(tokens_list, list) or len(tokens_list) == 0:
             return None

        batch_size = len(tokens_list)
        filled_pause_markers = {'um', 'uh', 'uhm', 'er', 'ah'}
        
        features = []
        
        for batch_idx in range(batch_size):
            batch_tokens = tokens_list[batch_idx]
            if not batch_tokens:
                features.append([0.0] * 4)
                continue

            text = ' '.join(batch_tokens).lower()
            
            pause_count = sum(1 for token in batch_tokens if token.lower() in filled_pause_markers)
            pause_ratio = pause_count / len(batch_tokens) if len(batch_tokens) > 0 else 0.0
            
            ellipsis_count = text.count('...')
            # More complex hesitations
            hesitation_pattern = len(re.findall(r'\b(like|you know|i mean|sort of|kind of|well|so)\b', text))
            
            features.append([pause_count, pause_ratio, ellipsis_count, hesitation_pattern])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def detect_repetitions(self, text_metadata: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Detect word and phrase repetitions"""
        if text_metadata is None or 'tokens' not in text_metadata:
            return None
        
        tokens_list = text_metadata.get('tokens', [])
        if not isinstance(tokens_list, list) or len(tokens_list) == 0:
             return None
        
        batch_size = len(tokens_list)
        features = []
        
        for batch_idx in range(batch_size):
            batch_tokens = [t.lower() for t in tokens_list[batch_idx]]
            if not batch_tokens:
                features.append([0.0] * 4)
                continue
            
            immediate_reps = sum(1 for i in range(len(batch_tokens)-1) 
                               if batch_tokens[i] == batch_tokens[i+1])
            
            bigrams = [' '.join(batch_tokens[i:i+2]) for i in range(len(batch_tokens)-1)]
            bigram_reps = 0
            if bigrams:
                bigram_counts = Counter(bigrams)
                bigram_reps = sum(count - 1 for count in bigram_counts.values())
            
            total_reps = immediate_reps + bigram_reps
            rep_ratio = total_reps / len(batch_tokens) if len(batch_tokens) > 0 else 0.0
            
            features.append([immediate_reps, bigram_reps, total_reps, rep_ratio])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self,
                embeddings: torch.Tensor,
                text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Analyze cognitive load markers"""
        pooled = torch.mean(embeddings, dim=1)
        cognitive_features = self.cognitive_encoder(pooled)
        
        filled_pauses = self.filled_pause_detector(cognitive_features)
        filled_pauses = torch.sigmoid(filled_pauses)
        
        word_finding = self.word_finding_detector(cognitive_features)
        word_finding = torch.sigmoid(word_finding)
        
        repetition = self.repetition_analyzer(cognitive_features)
        repetition = torch.sigmoid(repetition)
        
        effort = self.effort_estimator(cognitive_features)
        effort = torch.sigmoid(effort)
        
        correction = self.correction_analyzer(cognitive_features)
        correction = torch.sigmoid(correction)
        
        pause_features = self.detect_filled_pauses(text_metadata)
        rep_features = self.detect_repetitions(text_metadata)
        
        features = self.output_proj(cognitive_features)
        
        output = {
            'features': features,
            'filled_pauses': filled_pauses,
            'word_finding_difficulty': word_finding.mean(dim=-1),
            'word_finding_components': word_finding,
            'repetition_score': repetition.mean(dim=-1),
            'repetition_components': repetition,
            'cognitive_effort': effort.mean(dim=-1),
            'effort_components': effort,
            'self_correction': correction,
            'correction_rate': correction[:, 0]
        }
        
        if pause_features is not None:
            # FIXED: Move computed features to the correct device
            pause_features = pause_features.to(embeddings.device)
            output['pause_count'] = pause_features[:, 0]
            output['pause_ratio'] = pause_features[:, 1]
        
        if rep_features is not None:
            # FIXED: Move computed features to the correct device
            rep_features = rep_features.to(embeddings.device)
            output['immediate_repetitions'] = rep_features[:, 0]
            output['phrase_repetitions'] = rep_features[:, 1]
            output['total_repetitions'] = rep_features[:, 2]
        
        return output


class LinguisticDeclineAnalyzer(nn.Module):
    """Analyze linguistic markers of cognitive decline"""
    
    def __init__(self, embedding_dim: int = 768, output_dim: int = 128):
        super().__init__()
        
        # Decline marker encoder
        self.decline_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Simplified grammar detector
        self.grammar_simplification = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
        
        # Information content analyzer
        self.info_content_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
        # Pronoun overuse detector
        self.pronoun_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        # Semantic impoverishment detector
        self.semantic_impoverishment = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
        # Fragmentation analyzer
        self.fragmentation_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        # Disease-specific patterns
        self.disease_patterns = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        self.output_proj = nn.Linear(256, output_dim)
    
    def compute_content_ratio(self, text_metadata: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Compute ratio of content words to function words"""
        if text_metadata is None or 'pos_tags' not in text_metadata:
            return None
        
        pos_tags_list = text_metadata.get('pos_tags', [])
        if not isinstance(pos_tags_list, list) or len(pos_tags_list) == 0:
            return None
        
        batch_size = len(pos_tags_list)
        
        content_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
                       'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        function_tags = {'PRP', 'PRP$', 'DT', 'IN', 'CC', 'WDT', 'WP', 'WP$', 'WRB', 'TO', 'MD'}
        pronoun_tags = {'PRP', 'PRP$', 'WP', 'WP$'}

        features = []
        
        for batch_idx in range(batch_size):
            batch_tags = pos_tags_list[batch_idx]
            if not batch_tags:
                features.append([0.0] * 3)
                continue
            
            content_count = sum(1 for tag in batch_tags if tag in content_tags)
            function_count = sum(1 for tag in batch_tags if tag in function_tags)
            total_count = len(batch_tags)
            
            content_ratio = content_count / total_count if total_count > 0 else 0.0
            function_ratio = function_count / total_count if total_count > 0 else 0.0
            
            pronoun_count = sum(1 for tag in batch_tags if tag in pronoun_tags)
            pronoun_ratio = pronoun_count / total_count if total_count > 0 else 0.0
            
            features.append([content_ratio, function_ratio, pronoun_ratio])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self,
                embeddings: torch.Tensor,
                text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Analyze linguistic decline markers"""
        pooled = torch.mean(embeddings, dim=1)
        decline_features = self.decline_encoder(pooled)
        
        grammar_simp = self.grammar_simplification(decline_features)
        grammar_simp = torch.sigmoid(grammar_simp)
        
        info_content = self.info_content_analyzer(decline_features)
        info_content = torch.sigmoid(info_content)
        
        pronoun = self.pronoun_analyzer(decline_features)
        pronoun = torch.sigmoid(pronoun)
        
        semantic_imp = self.semantic_impoverishment(decline_features)
        semantic_imp = torch.sigmoid(semantic_imp)
        
        fragmentation = self.fragmentation_analyzer(decline_features)
        fragmentation = torch.sigmoid(fragmentation)
        
        disease_patterns = self.disease_patterns(decline_features)
        disease_patterns = F.softmax(disease_patterns, dim=-1)
        
        content_features = self.compute_content_ratio(text_metadata)
        
        features = self.output_proj(decline_features)
        
        output = {
            'features': features,
            'grammar_simplification': grammar_simp.mean(dim=-1),
            'grammar_components': grammar_simp,
            'information_content': info_content[:, 0],
            'info_content_components': info_content,
            'idea_density': info_content[:, 4],
            'pronoun_overuse': pronoun[:, 0],
            'pronoun_components': pronoun,
            'semantic_impoverishment': semantic_imp.mean(dim=-1),
            'semantic_components': semantic_imp,
            'fragmentation': fragmentation.mean(dim=-1),
            'fragmentation_components': fragmentation,
            'disease_patterns': disease_patterns
        }
        
        if content_features is not None:
            # FIXED: Move computed features to the correct device
            content_features = content_features.to(embeddings.device)
            output['content_word_ratio'] = content_features[:, 0]
            output['function_word_ratio'] = content_features[:, 1]
            output['pronoun_ratio'] = content_features[:, 2]
        
        return output


class TemporalAnalyzer(nn.Module):
    """Analyze temporal features of text production"""
    
    def __init__(self, embedding_dim: int = 768, output_dim: int = 128):
        super().__init__()
        
        # Temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Writing speed analyzer
        self.speed_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        # Revision pattern analyzer
        self.revision_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
        
        # Pause analysis
        self.pause_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
        # Temporal dynamics
        self.dynamics_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        self.output_proj = nn.Linear(256, output_dim)
    
    def compute_timing_features(self, text_metadata: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Compute timing-based features"""
        if text_metadata is None or 'timestamps' not in text_metadata:
            return None
        
        timestamps_list = text_metadata.get('timestamps', [])
        tokens_list = text_metadata.get('tokens', [])
        
        if not isinstance(timestamps_list, list) or len(timestamps_list) == 0 or \
           not isinstance(tokens_list, list) or len(tokens_list) != len(timestamps_list):
            return None
        
        batch_size = len(timestamps_list)
        features = []
        
        for batch_idx in range(batch_size):
            batch_timestamps = timestamps_list[batch_idx]
            batch_tokens = tokens_list[batch_idx]
            
            if not isinstance(batch_timestamps, np.ndarray) or len(batch_timestamps) < 2 or \
               len(batch_timestamps) != len(batch_tokens):
                features.append([0.0, 0.0, 0.0, 0.0])
                continue
            
            intervals = np.diff(batch_timestamps)
            if len(intervals) == 0:
                features.append([0.0, 0.0, 0.0, 0.0])
                continue

            total_time = batch_timestamps[-1] - batch_timestamps[0]
            wpm = (len(batch_tokens) / total_time) * 60 if total_time > 0 else 0.0
            
            pause_threshold = 2.0  # 2 seconds
            pauses = np.sum(intervals > pause_threshold)
            pause_ratio = pauses / len(intervals) if len(intervals) > 0 else 0.0
            
            speed_var = np.std(intervals) if len(intervals) > 0 else 0.0
            
            features.append([wpm, pause_ratio, pauses, speed_var])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self,
                embeddings: torch.Tensor,
                text_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Analyze temporal features"""
        pooled = torch.mean(embeddings, dim=1)
        temporal_features = self.temporal_encoder(pooled)
        
        speed = self.speed_analyzer(temporal_features)
        
        revision = self.revision_analyzer(temporal_features)
        revision = torch.sigmoid(revision)
        
        pauses = self.pause_analyzer(temporal_features)
        pauses = torch.sigmoid(pauses)
        
        dynamics = self.dynamics_analyzer(temporal_features)
        dynamics = torch.sigmoid(dynamics)
        
        timing_features = self.compute_timing_features(text_metadata)
        
        features = self.output_proj(temporal_features)
        
        output = {
            'features': features,
            'speed_metrics': speed,
            'revision_patterns': revision,
            'revision_rate': revision[:, 0],
            'pause_patterns': pauses,
            'pause_frequency': pauses[:, 0],
            'temporal_dynamics': dynamics,
            'writing_fluency': dynamics[:, 3]
        }
        
        if timing_features is not None:
            # FIXED: Move computed features to the correct device
            timing_features = timing_features.to(embeddings.device)
            output['words_per_minute'] = timing_features[:, 0]
            output['pause_ratio'] = timing_features[:, 1]
            output['pause_count'] = timing_features[:, 2]
            output['speed_variability'] = timing_features[:, 3]
        
        return output