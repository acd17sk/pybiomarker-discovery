"""
Pytest configuration for text biomarker model tests.

This conftest provides:
- Automatic CUDA/CPU device selection
- Reproducible random seeding
- Path configuration for importing biomarkers module
- Common fixtures for testing
- Test utilities
"""

import os
import sys
import pytest
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

def setup_paths():
    """
    Setup paths to ensure biomarkers module can be imported.
    
    Project structure:
    |- biomarkers/
    |- tests/
       |- models/
          |- text/
             |- conftest.py (this file)
    """
    # Get the directory containing this conftest.py file
    current_file = Path(__file__).resolve()
    
    # Navigate up to the project root
    # tests/models/text/conftest.py -> tests/models/text -> tests/models -> tests -> project_root
    project_root = current_file.parent.parent.parent.parent
    
    # Add project root to Python path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # Verify biomarkers module can be imported
    try:
        import biomarkers
        print(f"✓ Successfully imported biomarkers module from: {project_root}")
    except ImportError as e:
        print(f"✗ Failed to import biomarkers module from: {project_root}")
        print(f"  Error: {e}")
        print(f"  sys.path: {sys.path}")
        raise
    
    return project_root


# Setup paths at module import time
PROJECT_ROOT = setup_paths()


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def get_device() -> torch.device:
    """
    Get the appropriate device for testing.
    
    Returns:
        torch.device: CUDA device if available and enabled, otherwise CPU
    """
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    # Check environment variable to force CPU testing
    force_cpu = os.environ.get('PYTEST_FORCE_CPU', '0') == '1'
    
    if cuda_available and not force_cpu:
        device = torch.device('cuda')
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        if force_cpu:
            print("✓ Using CPU (forced by PYTEST_FORCE_CPU environment variable)")
        else:
            print("✓ Using CPU (CUDA not available)")
    
    return device


# Global device for tests
DEVICE = get_device()


# ============================================================================
# RANDOM SEEDING
# ============================================================================

def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# PYTEST HOOKS
# ============================================================================

def pytest_configure(config):
    """
    Pytest configuration hook - runs before test collection.
    """
    print("\n" + "="*80)
    print("PyTorch Text Biomarker Model Test Suite")
    print("="*80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print(f"Random seed: 42")
    print("="*80 + "\n")
    
    # Seed everything for reproducibility
    seed_everything(42)
    
    # Configure pytest markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers and skip GPU tests if needed.
    """
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    
    for item in items:
        # Add 'gpu' marker to tests that use GPU
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session", autouse=True)
def setup_session():
    """
    Session-level setup that runs once before all tests.
    """
    # Seed at session start
    seed_everything(42)
    
    # Print configuration
    print("\nSession Setup:")
    print(f"  - Device: {DEVICE}")
    print(f"  - NumPy version: {np.__version__}")
    print(f"  - PyTorch version: {torch.__version__}")
    
    yield
    
    # Session teardown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nSession Teardown: Cache cleared")


@pytest.fixture(autouse=True)
def reset_seed():
    """
    Reset random seed before each test for reproducibility.
    """
    seed_everything(42)
    yield


# ============================================================================
# DEVICE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """
    Provide the device for testing.
    
    Returns:
        torch.device: CUDA or CPU device
    """
    return DEVICE


@pytest.fixture(scope="session")
def is_cuda():
    """
    Check if CUDA is available.
    
    Returns:
        bool: True if CUDA is available
    """
    return DEVICE.type == 'cuda'


@pytest.fixture
def cuda_device():
    """
    Provide CUDA device (skip test if not available).
    
    Returns:
        torch.device: CUDA device
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


# ============================================================================
# PATH FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """
    Provide path to project root directory.
    
    Returns:
        Path: Path to project root
    """
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def biomarkers_path():
    """
    Provide path to biomarkers module directory.
    
    Returns:
        Path: Path to biomarkers directory
    """
    return PROJECT_ROOT / "biomarkers"


@pytest.fixture(scope="session")
def tests_path():
    """
    Provide path to tests directory.
    
    Returns:
        Path: Path to tests directory
    """
    return PROJECT_ROOT / "tests"


@pytest.fixture(scope="session")
def test_data_path(tmp_path_factory):
    """
    Provide path for test data (temporary directory).
    
    Returns:
        Path: Path to temporary test data directory
    """
    return tmp_path_factory.mktemp("test_data")


# ============================================================================
# EMBEDDING FIXTURES
# ============================================================================

@pytest.fixture
def sample_embeddings(device):
    """
    Create sample embeddings for testing.
    
    Args:
        device: Device to create tensors on
    
    Returns:
        torch.Tensor: Random embeddings [batch_size=2, seq_len=50, embedding_dim=768]
    """
    batch_size = 2
    seq_len = 50
    embedding_dim = 768
    return torch.randn(batch_size, seq_len, embedding_dim, device=device)


@pytest.fixture
def sample_embeddings_single(device):
    """
    Create single sample embeddings for testing.
    
    Args:
        device: Device to create tensors on
    
    Returns:
        torch.Tensor: Random embeddings [batch_size=1, seq_len=50, embedding_dim=768]
    """
    return torch.randn(1, 50, 768, device=device)


@pytest.fixture
def sample_embeddings_large_batch(device):
    """
    Create large batch embeddings for testing.
    
    Args:
        device: Device to create tensors on
    
    Returns:
        torch.Tensor: Random embeddings [batch_size=16, seq_len=50, embedding_dim=768]
    """
    return torch.randn(16, 50, 768, device=device)


@pytest.fixture
def sample_embeddings_variable_length(device):
    """
    Create variable length embeddings for testing.
    
    Args:
        device: Device to create tensors on
    
    Returns:
        List[torch.Tensor]: List of embeddings with different sequence lengths
    """
    return [
        torch.randn(1, 10, 768, device=device),
        torch.randn(1, 50, 768, device=device),
        torch.randn(1, 100, 768, device=device),
        torch.randn(1, 200, 768, device=device),
    ]


# ============================================================================
# TEXT METADATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_tokens():
    """
    Create sample tokens for testing.
    
    Returns:
        List[List[str]]: List of token sequences (2 samples)
    """
    return [
        ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', 'and', 
         'then', 'the', 'fox', 'runs', 'away', 'quickly', 'through', 'the', 'forest'],
        ['i', 'think', 'that', 'um', 'you', 'know', 'it', 'is', 'like', 'really', 
         'important', 'to', 'um', 'understand', 'the', 'the', 'concept', 'here']
    ]


@pytest.fixture
def sample_pos_tags():
    """
    Create sample POS tags for testing.
    
    Returns:
        List[List[str]]: List of POS tag sequences
    """
    return [
        ['DT', 'JJ', 'JJ', 'NN', 'VBZ', 'IN', 'DT', 'JJ', 'NN', 'CC',
         'RB', 'DT', 'NN', 'VBZ', 'RB', 'RB', 'IN', 'DT', 'NN'],
        ['PRP', 'VBP', 'IN', 'UH', 'PRP', 'VBP', 'PRP', 'VBZ', 'IN', 'RB',
         'JJ', 'TO', 'UH', 'VB', 'DT', 'DT', 'NN', 'RB']
    ]


@pytest.fixture
def sample_parse_trees():
    """
    Create sample parse trees for testing.
    
    Returns:
        List[str]: List of parse tree strings
    """
    return [
        '(S (NP (DT the) (JJ quick) (JJ brown) (NN fox)) (VP (VBZ jumps) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))))',
        '(S (NP (PRP i)) (VP (VBP think) (SBAR (IN that) (S (NP (PRP you)) (VP (VBP know) (S (NP (PRP it)) (VP (VBZ is) (ADJP (RB like) (RB really) (JJ important)))))))))'
    ]


@pytest.fixture
def sample_timestamps():
    """
    Create sample timestamps for testing.
    
    Returns:
        List[np.ndarray]: List of timestamp arrays
    """
    return [
        np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]),
        np.array([0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                 5.5, 6.0, 6.5, 7.5, 8.0, 8.5, 9.0, 9.5])
    ]


@pytest.fixture
def sample_text_metadata(sample_tokens, sample_pos_tags, sample_parse_trees, sample_timestamps):
    """
    Create comprehensive text metadata for testing.
    
    Returns:
        Dict: Dictionary containing all text metadata
    """
    return {
        'tokens': sample_tokens,
        'content_words': [
            ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog', 'runs', 'quickly', 'forest'],
            ['think', 'important', 'understand', 'concept']
        ],
        'pos_tags': sample_pos_tags,
        'parse_trees': sample_parse_trees,
        'timestamps': sample_timestamps
    }


@pytest.fixture
def sample_text_metadata_minimal(sample_tokens):
    """
    Create minimal text metadata for testing.
    
    Returns:
        Dict: Dictionary containing only tokens
    """
    return {
        'tokens': sample_tokens
    }


# ============================================================================
# MODEL CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def basic_config():
    """
    Create basic configuration for TextBiomarkerModel.
    
    Returns:
        Dict: Basic model configuration
    """
    return {
        'embedding_dim': 768,
        'hidden_dim': 256,
        'num_diseases': 8,
        'dropout': 0.3,
        'max_seq_length': 512,
        'use_pretrained': False,
        'pretrained_model': 'bert-base-uncased'
    }


@pytest.fixture
def small_config():
    """
    Create small configuration for fast testing.
    
    Returns:
        Dict: Small model configuration
    """
    return {
        'embedding_dim': 256,
        'hidden_dim': 64,
        'num_diseases': 4,
        'dropout': 0.2,
        'max_seq_length': 128,
        'use_pretrained': False
    }


@pytest.fixture
def large_config():
    """
    Create large configuration for stress testing.
    
    Returns:
        Dict: Large model configuration
    """
    return {
        'embedding_dim': 1024,
        'hidden_dim': 512,
        'num_diseases': 10,
        'dropout': 0.4,
        'max_seq_length': 1024,
        'use_pretrained': False
    }


# ============================================================================
# MODEL FIXTURES
# ============================================================================

@pytest.fixture
def text_biomarker_model(basic_config, device):
    """
    Create TextBiomarkerModel instance.
    
    Returns:
        TextBiomarkerModel: Model instance on appropriate device
    """
    from biomarkers.models.text.text_biomarker import TextBiomarkerModel
    model = TextBiomarkerModel(basic_config)
    return model.to(device)


@pytest.fixture
def linguistic_analyzer(device):
    """
    Create LinguisticAnalyzer instance.
    
    Returns:
        LinguisticAnalyzer: Analyzer instance on appropriate device
    """
    from biomarkers.models.text.linguistic_analyzer import LinguisticAnalyzer
    analyzer = LinguisticAnalyzer(embedding_dim=768, hidden_dim=256, dropout=0.3)
    return analyzer.to(device)


@pytest.fixture
def lexical_analyzer(device):
    """
    Create LexicalDiversityAnalyzer instance.
    
    Returns:
        LexicalDiversityAnalyzer: Analyzer instance on appropriate device
    """
    from biomarkers.models.text.linguistic_analyzer import LexicalDiversityAnalyzer
    analyzer = LexicalDiversityAnalyzer(embedding_dim=768, output_dim=128)
    return analyzer.to(device)


@pytest.fixture
def syntactic_analyzer(device):
    """
    Create SyntacticComplexityAnalyzer instance.
    
    Returns:
        SyntacticComplexityAnalyzer: Analyzer instance on appropriate device
    """
    from biomarkers.models.text.linguistic_analyzer import SyntacticComplexityAnalyzer
    analyzer = SyntacticComplexityAnalyzer(embedding_dim=768, output_dim=128)
    return analyzer.to(device)


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def assert_tensor_properties():
    """
    Provide utility function to assert tensor properties.
    
    Returns:
        callable: Function to assert tensor properties
    """
    def _assert(tensor, shape=None, dtype=None, device=None, requires_grad=None):
        """
        Assert tensor properties.
        
        Args:
            tensor: Tensor to check
            shape: Expected shape (None to skip)
            dtype: Expected dtype (None to skip)
            device: Expected device (None to skip)
            requires_grad: Expected requires_grad (None to skip)
        """
        assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"
        
        if shape is not None:
            assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
        
        if dtype is not None:
            assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
        
        if device is not None:
            expected_device = torch.device(device) if isinstance(device, str) else device
            assert tensor.device.type == expected_device.type, \
                f"Expected device {expected_device}, got {tensor.device}"
        
        if requires_grad is not None:
            assert tensor.requires_grad == requires_grad, \
                f"Expected requires_grad={requires_grad}, got {tensor.requires_grad}"
    
    return _assert


@pytest.fixture
def assert_no_nan():
    """
    Provide utility function to check for NaN values.
    
    Returns:
        callable: Function to check for NaN values
    """
    def _assert(tensor, name="tensor"):
        """
        Assert tensor doesn't contain NaN values.
        
        Args:
            tensor: Tensor to check
            name: Name for error message
        """
        if isinstance(tensor, torch.Tensor):
            assert not torch.any(torch.isnan(tensor)), f"NaN values found in {name}"
        elif isinstance(tensor, dict):
            for key, value in tensor.items():
                _assert(value, f"{name}.{key}")
    
    return _assert


@pytest.fixture
def assert_probability_distribution():
    """
    Provide utility function to check probability distributions.
    
    Returns:
        callable: Function to check probability distributions
    """
    def _assert(probs, dim=-1, atol=0.01):
        """
        Assert tensor is a valid probability distribution.
        
        Args:
            probs: Probability tensor
            dim: Dimension to sum over
            atol: Absolute tolerance for sum check
        """
        assert isinstance(probs, torch.Tensor), "Expected torch.Tensor"
        assert torch.all(probs >= 0.0), "Probabilities must be non-negative"
        assert torch.all(probs <= 1.0), "Probabilities must be <= 1.0"
        
        sums = probs.sum(dim=dim)
        assert torch.allclose(sums, torch.ones_like(sums), atol=atol), \
            f"Probabilities must sum to 1.0, got {sums}"
    
    return _assert


# ============================================================================
# PERFORMANCE FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_timer():
    """
    Provide timer for benchmarking test performance.
    
    Returns:
        callable: Context manager for timing code blocks
    """
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def timer(name="Operation"):
        start = time.time()
        yield
        end = time.time()
        print(f"\n{name} took {end - start:.4f} seconds")
    
    return timer


# ============================================================================
# CLEANUP
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_cuda(device):
    """
    Clean up CUDA memory after each test.
    """
    yield
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ============================================================================
# PARAMETRIZATION HELPERS
# ============================================================================

# Common parameter sets for parametrized tests
BATCH_SIZES = [1, 2, 4, 8, 16]
SEQUENCE_LENGTHS = [10, 50, 100, 200]
EMBEDDING_DIMS = [256, 512, 768, 1024]
HIDDEN_DIMS = [64, 128, 256, 512]


@pytest.fixture
def batch_sizes():
    """Provide common batch sizes for parametrized tests."""
    return BATCH_SIZES


@pytest.fixture
def sequence_lengths():
    """Provide common sequence lengths for parametrized tests."""
    return SEQUENCE_LENGTHS


@pytest.fixture
def embedding_dims():
    """Provide common embedding dimensions for parametrized tests."""
    return EMBEDDING_DIMS


@pytest.fixture
def hidden_dims():
    """Provide common hidden dimensions for parametrized tests."""
    return HIDDEN_DIMS


# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

def pytest_report_header(config):
    """
    Add custom header to pytest output.
    """
    return [
        f"Project root: {PROJECT_ROOT}",
        f"Device: {DEVICE}",
        f"PyTorch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]